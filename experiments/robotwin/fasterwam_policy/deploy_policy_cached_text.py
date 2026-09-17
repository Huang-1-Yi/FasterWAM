"""Memory-light FasterWAM deployment policy for one fixed instruction.

The first initialization creates a T5 embedding cache next to the action
checkpoint. Later initializations load that tiny cache and instantiate
FasterWAM with ``load_text_encoder=False``.
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.robotwin.fasterwam_policy.deploy_policy import (
    WorldActionRobotWinPolicy,
    _compose_sim_cfg,
    _is_none_like,
    _mixed_precision_to_model_dtype,
    _normalize_action_infer_mode,
    _parse_bool,
    _parse_optional_float,
    _parse_optional_int,
    _parse_optional_int_list,
    _resolve_dataset_stats_path,
    encode_obs,
)
from experiments.robotwin.fasterwam_policy.fixed_text_cache import (
    ensure_fixed_text_condition,
)
from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


logger = logging.getLogger(__name__)


class CachedTextWorldActionRobotWinPolicy(WorldActionRobotWinPolicy):
    """Official RoboTwin adapter with a fixed, precomputed language condition."""

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        action_infer_mode: str,
        fixed_instruction: str,
        text_embedding_path: Optional[str],
        create_text_embedding_if_missing: bool,
        text_encoding_device: str,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = False

        context_len = int(model_cfg_copy.get("tokenizer_max_len", 128))
        context, context_mask, resolved_cache, prompt = ensure_fixed_text_condition(
            checkpoint_path=checkpoint_path,
            instruction=fixed_instruction,
            model_id=str(model_cfg_copy.model_id),
            tokenizer_model_id=str(model_cfg_copy.tokenizer_model_id),
            context_len=context_len,
            redirect_common_files=bool(model_cfg_copy.get("redirect_common_files", True)),
            encoding_device=text_encoding_device,
            cache_path=text_embedding_path,
            create_if_missing=create_text_embedding_if_missing,
        )

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        if self.model.text_encoder is not None or self.model.tokenizer is not None:
            raise RuntimeError("Lightweight deployment unexpectedly loaded the text encoder.")

        self.fixed_instruction = str(fixed_instruction).strip()
        self.fixed_prompt = prompt
        self.text_embedding_path = resolved_cache
        self.fixed_context = context.unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        self.fixed_context_mask = context_mask.unsqueeze(0).to(
            device=self.model.device,
            dtype=torch.bool,
        )
        self._warned_instruction_mismatch = False

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.action_infer_mode = _normalize_action_infer_mode(action_infer_mode)
        if self.action_infer_mode == "joint_rollout":
            self._action_infer_fn = self.model.infer_action
        else:
            self._action_infer_fn = getattr(
                self.model, "infer_action_one_pass_future_cache", None
            )
            if self._action_infer_fn is None:
                raise ValueError(
                    "action_infer_mode='one_pass_future_cache' requires "
                    "`infer_action_one_pass_future_cache(...)`."
                )
        self._action_infer_signature = inspect.signature(self._action_infer_fn)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized cached-text policy | ckpt=%s | text_cache=%s | instruction=%r",
            checkpoint_path,
            self.text_embedding_path,
            self.fixed_instruction,
        )

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        if str(instruction).strip() != self.fixed_instruction and not self._warned_instruction_mismatch:
            logger.warning(
                "Ignoring runtime instruction %r; this policy is fixed to %r.",
                instruction,
                self.fixed_instruction,
            )
            self._warned_instruction_mismatch = True

        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)
        infer_kwargs = {
            "prompt": None,
            "context": self.fixed_context,
            "context_mask": self.fixed_context_mask,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in self._action_infer_signature.parameters:
            infer_kwargs["num_video_frames"] = self._num_video_frames

        infer_t0 = __import__("time").perf_counter() if self.timing_enabled else 0.0
        with torch.inference_mode():
            pred = self._action_infer_fn(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += __import__("time").perf_counter() - infer_t0

        return self._denormalize_action(pred["action"])[0]

    @staticmethod
    def _latest_vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1 and array.shape[0] == expected_dim:
            return array
        if array.ndim >= 2 and array.shape[-1] == expected_dim:
            return array.reshape(-1, expected_dim)[-1]
        raise ValueError(
            f"`{name}` must end in dimension {expected_dim}, got {array.shape}"
        )

    @staticmethod
    def _latest_image(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 3:
            return array
        if array.ndim == 4 and array.shape[-1] == 3:
            return array[-1]
        raise ValueError(f"`{name}` must be [H,W,3] or [T,H,W,3], got {array.shape}")

    def predict_dp_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Predict one native 14-D action chunk from DualFrankaRealEnv.get_obs().

        This method performs inference only. It never calls ``exec_actions`` and
        intentionally leaves robot safety checks and gripper discretization to
        the real-robot runner.
        """

        left_pose = self._latest_vector(
            observation["left_robot_eef_pose"], 6, "left_robot_eef_pose"
        )
        right_pose = self._latest_vector(
            observation["right_robot_eef_pose"], 6, "right_robot_eef_pose"
        )
        left_gripper = float(np.asarray(observation["left_gripper_width"]).reshape(-1)[-1])
        right_gripper = float(np.asarray(observation["right_gripper_width"]).reshape(-1)[-1])
        state = np.concatenate(
            [
                left_pose,
                np.asarray([left_gripper], dtype=np.float32),
                right_pose,
                np.asarray([right_gripper], dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)

        robotwin_observation = {
            "observation": {
                "head_camera": {
                    "rgb": self._latest_image(observation["camera_0"], "camera_0")
                },
                "left_camera": {
                    "rgb": self._latest_image(observation["camera_1"], "camera_1")
                },
                "right_camera": {
                    "rgb": self._latest_image(observation["camera_2"], "camera_2")
                },
            },
            "joint_action": {"vector": state},
        }
        return self._infer_action_chunk(robotwin_observation, self.fixed_instruction)


def get_model(usr_args: Dict[str, Any]):
    cfg = _compose_sim_cfg(
        sim_cfg_path=usr_args.get("sim_cfg_path"),
        sim_cfg_name=usr_args.get("sim_cfg_name"),
        sim_task=usr_args.get("sim_task"),
    )
    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU.")
        device = "cpu"
    model_dtype = _mixed_precision_to_model_dtype(
        str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    )
    dataset_stats_path = _resolve_dataset_stats_path(usr_args.get("dataset_stats_path"))

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        action_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
    if action_horizon is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 28))
    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(
            cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps)
        )
    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    for key in ("video_kv_fusion", "video_kv_fusion_init"):
        if key in usr_args:
            cfg.model[key] = usr_args[key]
    attention_layers = _parse_optional_int_list(
        usr_args.get("mot_action_video_attention_layers"),
        field_name="mot_action_video_attention_layers",
    )
    if attention_layers is not None:
        cfg.model.mot_action_video_attention_layers = attention_layers

    fixed_instruction = str(usr_args.get("fixed_instruction", "fold the T-shirt")).strip()
    text_encoding_device = str(usr_args.get("text_encoding_device") or device)
    create_missing = _parse_bool(
        usr_args.get("create_text_embedding_if_missing", True)
    )
    return CachedTextWorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=_parse_optional_int(usr_args.get("seed")),
        text_cfg_scale=float(
            usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0))
        ),
        negative_prompt=str(
            usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", ""))
        ),
        rand_device=str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu"))),
        tiled=_parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False))),
        timing_enabled=_parse_bool(
            usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
        ),
        num_video_frames=(int(cfg.data.train.num_frames) - 1)
        // int(cfg.data.train.action_video_freq_ratio)
        + 1,
        action_infer_mode=_normalize_action_infer_mode(
            usr_args.get(
                "action_infer_mode",
                cfg.EVALUATION.get("action_infer_mode", "one_pass_future_cache"),
            )
        ),
        fixed_instruction=fixed_instruction,
        text_embedding_path=usr_args.get("text_embedding_path"),
        create_text_embedding_if_missing=create_missing,
        text_encoding_device=text_encoding_device,
    )


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()
