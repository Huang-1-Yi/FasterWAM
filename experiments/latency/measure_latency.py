#!/usr/bin/env python
"""Architecture-only inference latency benchmark for FasterWAM models.

The benchmark deliberately constructs the real architectures from the repository
configs with ordinary random initialization.  It never loads model checkpoints,
tokenizers, text encoders, datasets, or dataset statistics.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluate_results" / "latency"

MODEL_ORDER = ("jointwam", "fastwam", "fasterwam")
TASK_BY_MODEL = {
    "fastwam": "libero_fastwam_2cam224_1e-4",
    "jointwam": "libero_jointwam_2cam224_1e-4",
    "fasterwam": "libero_fasterwam_2cam224_1e-4",
}
EXECUTED_PATH = {
    "fastwam": "cached_first_frame",
    "jointwam": "joint_rollout",
    "fasterwam": "one_pass_future_cache",
}
DISPLAY_NAME = {
    "fastwam": "Fast-WAM",
    "jointwam": "Joint-WAM",
    "fasterwam": "Faster-WAM",
}
PAPER_RESULT_ORDER = ("Joint-WAM", "Fast-WAM", "Faster-WAM")

MODEL_SEED = 42
OUTPUT_ALLCLOSE_ATOL = 1e-4
OUTPUT_ALLCLOSE_RTOL = 1e-4
MAX_DIRECT_PROFILE_MEAN_DELTA_PCT = 5.0

INTERNAL_FIELDS = (
    "profile_model_total_ms",
    "video_latent_init_ms",
    "action_latent_init_ms",
    "vae_encode_ms",
    "context_prepare_ms",
    "visual_branch_ms",
    "action_denoise_loop_ms",
    "action_to_cpu_ms",
    "action_branch_ms",
    "joint_video_action_denoise_ms",
    "profile_other_ms",
)
OPTIONAL_INTERNAL_FIELDS = {
    "visual_branch_ms",
    "action_denoise_loop_ms",
    "action_branch_ms",
    "joint_video_action_denoise_ms",
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    task: str
    executed_path: str
    image_height: int
    image_width: int
    context_len: int
    text_dim: int
    action_dim: int
    proprio_dim: int
    action_horizon: int
    num_video_frames: int
    num_inference_steps: int
    model_seed: int
    synthetic_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure native FasterWAM architecture inference latency with random weights."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", *MODEL_ORDER),
        default=["all"],
        help="Models to run; default: all three in sequence.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be > 0")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0")
    if args.action_horizon is not None and args.action_horizon <= 0:
        parser.error("--action-horizon must be > 0")
    if "all" in args.models and len(args.models) != 1:
        parser.error("Use either --models all or an explicit model list")
    return args


def selected_models(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(MODEL_ORDER)
    requested = set(values)
    return [name for name in MODEL_ORDER if name in requested]


def load_config(model_name: str) -> Any:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=[f"task={TASK_BY_MODEL[model_name]}"])


def plain_config(node: Any) -> dict[str, Any]:
    value = OmegaConf.to_container(node, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping config, got {type(value).__name__}")
    return value


def make_run_spec(model_name: str, cfg: Any, args: argparse.Namespace) -> RunSpec:
    height, width = (int(x) for x in cfg.data.train.video_size)
    context_len = int(cfg.data.train.context_len)
    action_dim = int(cfg.data.train.processor.action_output_dim)
    proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
    action_horizon = int(args.action_horizon or (int(cfg.data.train.num_frames) - 1))
    ratio = int(cfg.data.train.action_video_freq_ratio)
    num_video_frames = action_horizon // ratio + 1
    steps = int(args.num_inference_steps or cfg.eval_num_inference_steps)
    text_dim = int(cfg.model.video_dit_config.text_dim)
    return RunSpec(
        model=model_name,
        task=TASK_BY_MODEL[model_name],
        executed_path=EXECUTED_PATH[model_name],
        image_height=height,
        image_width=width,
        context_len=context_len,
        text_dim=text_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        action_horizon=action_horizon,
        num_video_frames=num_video_frames,
        num_inference_steps=steps,
        model_seed=int(cfg.seed),
        synthetic_seed=int(args.synthetic_seed),
    )


def set_model_seed(seed: int) -> None:
    from fasterwam.utils.pytorch_utils import set_global_seed

    set_global_seed(seed)


def architecture_configs(cfg: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    video_cfg = plain_config(cfg.model.video_dit_config)
    action_cfg = plain_config(cfg.model.action_dit_config)
    # Inference does not use checkpointing, and the task configs already disable it.
    video_cfg["use_gradient_checkpointing"] = False
    action_cfg["use_gradient_checkpointing"] = False
    return video_cfg, action_cfg


def build_model(model_name: str, cfg: Any, device: torch.device) -> torch.nn.Module:
    """Construct the actual architecture on GPU without any pretrained loader."""
    from fasterwam.models.wan22.action_dit import ActionDiT
    from fasterwam.models.wan22.fastwam import FastWAM
    from fasterwam.models.wan22.fasterwam import FasterWAM
    from fasterwam.models.wan22.jointwam import JointWAM
    from fasterwam.models.wan22.mot import MoT
    from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT
    from fasterwam.models.wan22.sparse_mot import SparseMoT
    from fasterwam.models.wan22.wan_video_dit import WanVideoDiT
    from fasterwam.models.wan22.wan_video_vae import WanVideoVAE38

    set_model_seed(int(cfg.seed))
    video_cfg, action_cfg = architecture_configs(cfg)
    model_cfg = plain_config(cfg.model)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # Device context prevents a transient full-size CPU copy of the 5B architecture.
        with torch.device(device):
            video_expert = WanVideoDiT(**video_cfg)
            if model_name == "fasterwam":
                action_expert = SparseActionDiT(**action_cfg)
                condition_layers = tuple(int(x) for x in action_cfg["condition_layers"])
                mot = SparseMoT(
                    mixtures={"video": video_expert, "action": action_expert},
                    condition_layers=condition_layers,
                    mot_checkpoint_mixed_attn=False,
                    video_kv_fusion=model_cfg.get("video_kv_fusion"),
                    video_kv_fusion_init=str(model_cfg.get("video_kv_fusion_init", "current_layer")),
                )
                model_cls = FasterWAM
                attention_layers = condition_layers
            else:
                action_expert = ActionDiT(**action_cfg)
                mot = MoT(
                    mixtures={"video": video_expert, "action": action_expert},
                    mot_checkpoint_mixed_attn=False,
                )
                model_cls = FastWAM if model_name == "fastwam" else JointWAM
                configured_layers = model_cfg.get("mot_action_video_attention_layers")
                attention_layers = None if configured_layers is None else tuple(configured_layers)
            vae = WanVideoVAE38(z_dim=48, dim=160)
            video_scheduler = plain_config(cfg.model.video_scheduler)
            action_scheduler = plain_config(cfg.model.action_scheduler)
            loss_cfg = plain_config(cfg.model.loss)
            model = model_cls(
                video_expert=video_expert,
                action_expert=action_expert,
                mot=mot,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                text_dim=int(video_cfg["text_dim"]),
                proprio_dim=int(cfg.model.proprio_dim),
                device=str(device),
                torch_dtype=torch.bfloat16,
                video_train_shift=float(video_scheduler["train_shift"]),
                video_infer_shift=float(video_scheduler["infer_shift"]),
                video_num_train_timesteps=int(video_scheduler["num_train_timesteps"]),
                action_train_shift=float(action_scheduler["train_shift"]),
                action_infer_shift=float(action_scheduler["infer_shift"]),
                action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
                loss_lambda_action=float(loss_cfg["lambda_action"]),
                mot_action_video_attention_layers=attention_layers,
            )
            if model_name == "fasterwam":
                model.condition_layers = condition_layers
    finally:
        torch.set_default_dtype(old_dtype)
    # Keep parameter requires_grad flags identical to the repository's normal
    # evaluation loader.  Autograd is disabled around inference with no_grad().
    return model.eval()


def prepare_inputs(spec: RunSpec, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(spec.synthetic_seed)
    image = torch.rand(
        (1, 3, spec.image_height, spec.image_width),
        generator=generator,
        dtype=torch.float32,
    ).mul_(2.0).sub_(1.0).to(device=device, dtype=torch.bfloat16)
    context = torch.zeros(
        (1, spec.context_len, spec.text_dim), device=device, dtype=torch.bfloat16
    )
    context_mask = torch.ones((1, spec.context_len), device=device, dtype=torch.bool)
    # Match the old timer boundary: normalized proprio exists on CPU and moves
    # to GPU inside infer_action.
    proprio = torch.tensor(
        [[0.45, 0.0, 0.25, 0.0, 0.0, 0.0, 0.04, 0.04]], dtype=torch.float32
    )[:, : spec.proprio_dim]
    return {
        "input_image": image,
        "context": context,
        "context_mask": context_mask,
        "proprio": proprio,
    }


def cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def common_infer_kwargs(spec: RunSpec, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "prompt": None,
        "input_image": inputs["input_image"],
        "action_horizon": spec.action_horizon,
        "proprio": inputs["proprio"],
        "context": inputs["context"],
        "context_mask": inputs["context_mask"],
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "num_inference_steps": spec.num_inference_steps,
        "sigma_shift": None,
        "seed": spec.model_seed,
        "rand_device": "cpu",
        "tiled": False,
    }


def official_infer_fn(model_name: str, model: torch.nn.Module) -> Callable[..., dict[str, Any]]:
    if model_name == "fasterwam":
        return model.infer_action_one_pass_future_cache
    return model.infer_action


def run_official(
    model_name: str,
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], float]:
    kwargs = common_infer_kwargs(spec, inputs)
    if model_name in {"jointwam", "fasterwam"}:
        kwargs["num_video_frames"] = spec.num_video_frames
    cuda_sync(model.device)
    start = time.perf_counter()
    # Match the reference benchmark exactly.  inference_mode() is measurably
    # faster for the action-heavy paths and would change the reported number.
    with torch.no_grad():
        output = official_infer_fn(model_name, model)(**kwargs)
    cuda_sync(model.device)
    return output, (time.perf_counter() - start) * 1000.0


def validate_inputs(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    proprio: torch.Tensor | None,
    num_video_frames: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if input_image.ndim != 4 or tuple(input_image.shape[:2]) != (1, 3):
        raise ValueError(f"Invalid input image shape: {tuple(input_image.shape)}")
    height, width = (int(x) for x in input_image.shape[-2:])
    if num_video_frames is None:
        if height % 16 or width % 16:
            raise ValueError("Image height and width must be multiples of 16")
    else:
        checked_h, checked_w, checked_t = model._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (height, width, num_video_frames):
            raise ValueError("Invalid image/video dimensions")
    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or tuple(proprio.shape) != (1, model.proprio_dim):
            raise ValueError(f"Invalid proprio shape: {tuple(proprio.shape)}")
    return input_image, proprio, height, width


def prepare_context(
    model: torch.nn.Module,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
    context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)
    if proprio is not None:
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(context, context_mask, proprio)
    return context, context_mask


def new_internal_timings() -> dict[str, float | None]:
    values: dict[str, float | None] = {field: 0.0 for field in INTERNAL_FIELDS}
    for field in OPTIONAL_INTERNAL_FIELDS:
        values[field] = None
    return values


def finish_internal(
    timings: dict[str, float | None], executed_path: str
) -> dict[str, float | None]:
    if executed_path == "joint_rollout":
        primitive = (
            "video_latent_init_ms",
            "action_latent_init_ms",
            "vae_encode_ms",
            "context_prepare_ms",
            "joint_video_action_denoise_ms",
            "action_to_cpu_ms",
        )
    else:
        primitive = (
            "video_latent_init_ms",
            "action_latent_init_ms",
            "vae_encode_ms",
            "context_prepare_ms",
            "visual_branch_ms",
            "action_denoise_loop_ms",
            "action_to_cpu_ms",
        )
        timings["action_branch_ms"] = sum(
            float(timings[field] or 0.0)
            for field in ("action_latent_init_ms", "action_denoise_loop_ms", "action_to_cpu_ms")
        )
    accounted = sum(float(timings[field] or 0.0) for field in primitive)
    timings["profile_other_ms"] = max(float(timings["profile_model_total_ms"] or 0.0) - accounted, 0.0)
    return timings


def timed_stage(device: torch.device, callback: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = callback()
    cuda_sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def profile_cached(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
    use_future_video: bool,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    timings = new_internal_timings()
    model.eval()
    total_start = time.perf_counter()
    input_image, proprio, height, width = validate_inputs(
        model,
        inputs["input_image"],
        inputs["proprio"],
        spec.num_video_frames if use_future_video else None,
    )

    def init_video() -> torch.Tensor | None:
        if not use_future_video:
            return None
        latent_t = (spec.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
        generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        return torch.randn(
            (1, model.vae.model.z_dim, latent_t, height // 16, width // 16),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)

    latents_video, timings["video_latent_init_ms"] = timed_stage(model.device, init_video)

    def init_action() -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        return torch.randn(
            (1, spec.action_horizon, model.action_expert.action_dim),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)

    latents_action, timings["action_latent_init_ms"] = timed_stage(model.device, init_action)

    def encode_image() -> torch.Tensor:
        image = input_image.to(device=model.device, dtype=model.torch_dtype)
        return model._encode_input_image_latents_tensor(input_image=image, tiled=False)

    first_frame_latents, timings["vae_encode_ms"] = timed_stage(model.device, encode_image)
    if latents_video is not None:
        latents_video[:, :, 0:1] = first_frame_latents.clone()
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    (context, context_mask), timings["context_prepare_ms"] = timed_stage(
        model.device,
        lambda: prepare_context(model, inputs["context"], inputs["context_mask"], proprio),
    )

    def prepare_visual() -> tuple[Any, list[torch.Tensor], int]:
        if latents_video is None:
            timestep_video = torch.zeros(
                (first_frame_latents.shape[0],),
                dtype=first_frame_latents.dtype,
                device=model.device,
            )
            video_input = first_frame_latents
        else:
            video_steps, _ = model.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=spec.num_inference_steps,
                device=model.device,
                dtype=latents_video.dtype,
                shift_override=None,
            )
            timestep_video = video_steps[0].unsqueeze(0).to(
                dtype=latents_video.dtype, device=model.device
            )
            video_input = latents_video
        video_pre = model.video_expert.pre_dit(
            x=video_input,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_masks = model._build_mot_attention_masks(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_attention_mask = model._build_video_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
            video_attention_mask=video_attention_mask,
        )
        return cache, attention_masks, video_seq_len

    (video_cache, attention_masks, video_seq_len), timings["visual_branch_ms"] = timed_stage(
        model.device, prepare_visual
    )

    def denoise_action() -> torch.Tensor:
        action_steps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=spec.num_inference_steps,
            device=model.device,
            dtype=latents_action.dtype,
            shift_override=None,
        )
        current = latents_action
        for step_t, step_delta in zip(action_steps, action_deltas):
            timestep = step_t.unsqueeze(0).to(dtype=current.dtype, device=model.device)
            prediction = model._predict_action_noise_with_cache(
                latents_action=current,
                timestep_action=timestep,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_cache,
                attention_mask=attention_masks,
                video_seq_len=video_seq_len,
            )
            current = model.infer_action_scheduler.step(prediction, step_delta, current)
        return current

    latents_action, timings["action_denoise_loop_ms"] = timed_stage(model.device, denoise_action)
    action, timings["action_to_cpu_ms"] = timed_stage(
        model.device, lambda: latents_action[0].detach().to(device="cpu", dtype=torch.float32)
    )
    timings["profile_model_total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return {"action": action}, finish_internal(timings, spec.executed_path)


def profile_joint(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, float | None]]:
    timings = new_internal_timings()
    model.eval()
    total_start = time.perf_counter()
    input_image, proprio, height, width = validate_inputs(
        model, inputs["input_image"], inputs["proprio"], spec.num_video_frames
    )

    def init_video() -> torch.Tensor:
        latent_t = (spec.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
        generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        return torch.randn(
            (1, model.vae.model.z_dim, latent_t, height // 16, width // 16),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)

    latents_video, timings["video_latent_init_ms"] = timed_stage(model.device, init_video)

    def init_action() -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        return torch.randn(
            (1, spec.action_horizon, model.action_expert.action_dim),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)

    latents_action, timings["action_latent_init_ms"] = timed_stage(model.device, init_action)

    def encode_image() -> torch.Tensor:
        image = input_image.to(device=model.device, dtype=model.torch_dtype)
        return model._encode_input_image_latents_tensor(input_image=image, tiled=False)

    first_frame_latents, timings["vae_encode_ms"] = timed_stage(model.device, encode_image)
    latents_video[:, :, 0:1] = first_frame_latents.clone()
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    (context, context_mask), timings["context_prepare_ms"] = timed_stage(
        model.device,
        lambda: prepare_context(model, inputs["context"], inputs["context_mask"], proprio),
    )

    def denoise_joint() -> tuple[torch.Tensor, torch.Tensor]:
        video_steps, video_deltas = model.infer_video_scheduler.build_inference_schedule(
            spec.num_inference_steps, model.device, latents_video.dtype, None
        )
        action_steps, action_deltas = model.infer_action_scheduler.build_inference_schedule(
            spec.num_inference_steps, model.device, latents_action.dtype, None
        )
        current_video, current_action = latents_video, latents_action
        for video_t, video_delta, action_t, action_delta in zip(
            video_steps, video_deltas, action_steps, action_deltas
        ):
            prediction_video, prediction_action = model._predict_joint_noise(
                latents_video=current_video,
                latents_action=current_action,
                timestep_video=video_t.unsqueeze(0).to(dtype=current_video.dtype, device=model.device),
                timestep_action=action_t.unsqueeze(0).to(dtype=current_action.dtype, device=model.device),
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
            )
            current_video = model.infer_video_scheduler.step(
                prediction_video, video_delta, current_video
            )
            current_action = model.infer_action_scheduler.step(
                prediction_action, action_delta, current_action
            )
            current_video[:, :, 0:1] = first_frame_latents.clone()
        return current_video, current_action

    (_, latents_action), timings["joint_video_action_denoise_ms"] = timed_stage(
        model.device, denoise_joint
    )
    action, timings["action_to_cpu_ms"] = timed_stage(
        model.device, lambda: latents_action[0].detach().to(device="cpu", dtype=torch.float32)
    )
    timings["profile_model_total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return {"action": action}, finish_internal(timings, spec.executed_path)


def run_profile(
    model_name: str,
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, float | None]]:
    with torch.no_grad():
        if model_name == "jointwam":
            return profile_joint(model, spec, inputs)
        return profile_cached(model, spec, inputs, use_future_video=model_name == "fasterwam")


def validate_pair(
    direct: dict[str, Any], profile: dict[str, Any], spec: RunSpec
) -> tuple[bool, float]:
    direct_action = direct.get("action")
    profile_action = profile.get("action")
    expected = (spec.action_horizon, spec.action_dim)
    if not isinstance(direct_action, torch.Tensor) or tuple(direct_action.shape) != expected:
        raise RuntimeError(f"Official output shape is not {expected}: {getattr(direct_action, 'shape', None)}")
    if not isinstance(profile_action, torch.Tensor) or tuple(profile_action.shape) != expected:
        raise RuntimeError(f"Profile output shape is not {expected}: {getattr(profile_action, 'shape', None)}")
    if not torch.isfinite(direct_action).all() or not torch.isfinite(profile_action).all():
        raise RuntimeError("Inference produced non-finite action values")
    max_abs_diff = float((direct_action - profile_action).abs().max().item())
    allclose = bool(
        torch.allclose(
            direct_action,
            profile_action,
            atol=OUTPUT_ALLCLOSE_ATOL,
            rtol=OUTPUT_ALLCLOSE_RTOL,
        )
    )
    if not allclose:
        raise RuntimeError(f"Official/profile outputs differ (max_abs_diff={max_abs_diff:.6g})")
    return allclose, max_abs_diff


def run_pair(
    model_name: str,
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
    profile_first: bool,
) -> dict[str, Any]:
    if profile_first:
        profile_output, internal = run_profile(model_name, model, spec, inputs)
        direct_output, direct_ms = run_official(model_name, model, spec, inputs)
        order = "profile_then_direct"
    else:
        direct_output, direct_ms = run_official(model_name, model, spec, inputs)
        profile_output, internal = run_profile(model_name, model, spec, inputs)
        order = "direct_then_profile"
    allclose, max_abs_diff = validate_pair(direct_output, profile_output, spec)
    return {
        "model": model_name,
        "task": spec.task,
        "executed_path": spec.executed_path,
        "measurement_order": order,
        "model_infer_ms": float(direct_ms),
        "output_allclose": allclose,
        "output_max_abs_diff": max_abs_diff,
        "action_shape": list(direct_output["action"].shape),
        **internal,
    }


def metric(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=0)),
    }


def optional_metric(records: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return metric(values) if values else None


def summarize_model(spec: RunSpec, records: list[dict[str, Any]]) -> dict[str, Any]:
    direct = metric([float(record["model_infer_ms"]) for record in records])
    profile = metric([float(record["profile_model_total_ms"]) for record in records])
    delta_pct = abs(float(profile["mean_ms"]) - float(direct["mean_ms"])) / float(direct["mean_ms"]) * 100.0
    summary = {
        "model": DISPLAY_NAME[spec.model],
        "vae_encode_ms": optional_metric(records, "vae_encode_ms"),
        "visual_branch_ms": optional_metric(records, "visual_branch_ms"),
        "action_branch_ms": optional_metric(records, "action_branch_ms"),
        "model_infer_ms": direct,
    }
    if delta_pct > MAX_DIRECT_PROFILE_MEAN_DELTA_PCT:
        raise RuntimeError(
            f"{spec.model}: direct/profile mean differs by {delta_pct:.2f}% "
            f"(limit {MAX_DIRECT_PROFILE_MEAN_DELTA_PCT:.2f}%)"
        )
    return summary


def runtime_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "batch_size": 1,
        "warmup": int(args.warmup),
        "iterations": int(args.iters),
        "seeds": {"model": MODEL_SEED, "input": int(args.synthetic_seed)},
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default) + "\n")


def write_markdown(path: Path, metadata: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    def format_metric(value: dict[str, float] | None) -> str:
        if value is None:
            return "N/A"
        return f"{value['mean_ms']:.3f} ± {value['std_ms']:.3f}"

    lines = [
        "# Inference Latency",
        "",
        f"- GPU: {metadata['gpu']}",
        f"- Setting: BF16, batch 1, {metadata['input_size'][0]} × {metadata['input_size'][1]} input, "
        f"action horizon {metadata['action_horizon']}, {metadata['action_denoising_steps']} denoising steps",
        f"- Protocol: {metadata['warmup']} warm-up iterations followed by "
        f"{metadata['iterations']} GPU-synchronized measurements",
        "",
        "Values are mean ± population standard deviation in milliseconds.",
        "",
        "| Model | VAE Encode | Visual Branch | Action Branch | Model Infer |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {format_metric(summary['vae_encode_ms'])} | "
            f"{format_metric(summary['visual_branch_ms'])} | "
            f"{format_metric(summary['action_branch_ms'])} | "
            f"{format_metric(summary['model_infer_ms'])} |"
        )
    lines.extend(
        [
            "",
            "- **VAE Encode:** encode the current observation into the first-frame latent.",
            "- **Visual Branch:** video pre-DiT, attention masks, visual forward, and reusable KV-cache construction.",
            "- **Action Branch:** action-latent initialization, all denoising steps, and final action transfer to CPU.",
            "- **Model Infer:** GPU-synchronized timing of the native `infer_action*` entry point.",
            "- Joint-WAM denoises video and action jointly, so its Visual and Action branches are reported as N/A.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def order_results(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(PAPER_RESULT_ORDER)}
    return sorted(summaries, key=lambda item: order[item["model"]])


def persist_results(
    output_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_summaries = order_results(summaries)
    write_jsonl(output_dir / "raw_latencies.jsonl", records)
    write_json(
        output_dir / "latency_results.json",
        {
            "settings": metadata,
            "results": ordered_summaries,
        },
    )
    write_markdown(output_dir / "latency_report.md", metadata, ordered_summaries)


def release_model(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> None:
    del model
    inputs.clear()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this latency benchmark")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid --gpu-id {args.gpu_id}; found {torch.cuda.device_count()} CUDA device(s)")
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    models = selected_models(args.models)
    configs = {model_name: load_config(model_name) for model_name in models}
    specs = {
        model_name: make_run_spec(model_name, configs[model_name], args)
        for model_name in models
    }
    reference_spec = specs[models[0]]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (DEFAULT_RESULTS_ROOT / timestamp)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = runtime_metadata(args, device)
    metadata.update(
        input_size=[reference_spec.image_height, reference_spec.image_width],
        action_horizon=reference_spec.action_horizon,
        video_frames=reference_spec.num_video_frames,
        action_denoising_steps=reference_spec.num_inference_steps,
    )
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    persist_results(output_dir, metadata, records, summaries)

    print(f"Output directory: {output_dir}", flush=True)
    for model_name in models:
        cfg = configs[model_name]
        spec = specs[model_name]
        if spec.model_seed != MODEL_SEED:
            raise RuntimeError(f"Expected repository seed {MODEL_SEED}, got {spec.model_seed}")
        print(f"\n[{model_name}] constructing random-weight architecture on {device} ...", flush=True)
        setup_start = time.perf_counter()
        model = build_model(model_name, cfg, device)
        inputs = prepare_inputs(spec, device)
        cuda_sync(device)
        setup_s = time.perf_counter() - setup_start
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"[{model_name}] {parameter_count / 1e9:.3f}B parameters; setup {setup_s:.1f}s", flush=True)

        for index in range(args.warmup):
            run_pair(model_name, model, spec, inputs, profile_first=bool(index % 2))
            print(f"[{model_name}] warmup pair {index + 1}/{args.warmup}", flush=True)

        model_records: list[dict[str, Any]] = []
        try:
            for index in range(args.iters):
                record = run_pair(model_name, model, spec, inputs, profile_first=bool(index % 2))
                record.update(
                    iteration=index,
                    model_seed=spec.model_seed,
                    synthetic_seed=spec.synthetic_seed,
                    num_inference_steps=spec.num_inference_steps,
                    action_horizon=spec.action_horizon,
                    num_video_frames=(None if model_name == "fastwam" else spec.num_video_frames),
                    image_shape=[1, 3, spec.image_height, spec.image_width],
                    dtype="bfloat16",
                )
                model_records.append(record)
                records.append(record)
                persist_results(output_dir, metadata, records, summaries)
                print(
                    f"[{model_name}] measured pair {index + 1}/{args.iters}: "
                    f"direct={record['model_infer_ms']:.3f} ms, "
                    f"profile={record['profile_model_total_ms']:.3f} ms",
                    flush=True,
                )
            summary = summarize_model(spec, model_records)
            summaries.append(summary)
            persist_results(output_dir, metadata, records, summaries)
            print(
                f"[{model_name}] mean native inference: "
                f"{summary['model_infer_ms']['mean_ms']:.3f} ms",
                flush=True,
            )
        finally:
            release_model(model, inputs)

    persist_results(output_dir, metadata, records, summaries)
    print(f"\nCompleted. Report: {output_dir / 'latency_report.md'}", flush=True)


if __name__ == "__main__":
    main()
