from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from fasterwam.models.wan22.fasterwam import FasterWAM


class CachedFasterWAM(FasterWAM):
    """FasterWAM variant that accepts deterministic, precomputed VAE latents."""

    def build_inputs(self, sample, tiled: bool = False):
        if "video_latents" not in sample:
            return super().build_inputs(sample, tiled=tiled)

        input_latents = sample["video_latents"]
        if input_latents.ndim != 5:
            raise ValueError(
                "`sample['video_latents']` must be [B,C,T,H,W], "
                f"got {tuple(input_latents.shape)}"
            )
        if input_latents.shape[1] != int(self.vae.z_dim):
            raise ValueError(
                f"Cached latent channels must be {self.vae.z_dim}, "
                f"got {input_latents.shape[1]}."
            )
        input_latents = input_latents.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        batch_size = int(input_latents.shape[0])

        context = sample["context"]
        context_mask = sample["context_mask"]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if int(context.shape[0]) != batch_size:
            raise ValueError("Cached latent and context batch sizes do not match.")
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        action = sample["action"]
        if action.ndim != 3 or int(action.shape[0]) != batch_size:
            raise ValueError(
                f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}"
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        image_is_pad = sample.get("image_is_pad")
        action_is_pad = sample.get("action_is_pad")
        if image_is_pad is None or image_is_pad.ndim != 2:
            raise ValueError("Cached training requires 2D `image_is_pad`.")
        if int(image_is_pad.shape[0]) != batch_size:
            raise ValueError("Cached latent and image mask batch sizes do not match.")
        num_video_frames = int(image_is_pad.shape[1])
        expected_latent_t = (num_video_frames - 1) // int(self.vae.temporal_downsample_factor) + 1
        if int(input_latents.shape[2]) != expected_latent_t:
            raise ValueError(
                "Cached latent temporal length mismatch: "
                f"got {input_latents.shape[2]}, expected {expected_latent_t} "
                f"for {num_video_frames} video frames."
            )
        if int(action.shape[1]) % (num_video_frames - 1) != 0:
            raise ValueError(
                "Action horizon must be divisible by video transitions: "
                f"{action.shape[1]} vs {num_video_frames - 1}."
            )

        proprio = sample.get("proprio")
        if self.proprio_encoder is not None:
            if proprio is None or proprio.ndim != 3:
                raise ValueError("Cached training requires `proprio` with shape [B,T,D].")
            if int(proprio.shape[2]) != int(self.proprio_dim):
                raise ValueError(
                    f"Proprio dimension must be {self.proprio_dim}, got {proprio.shape[2]}."
                )
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype),
            )

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        image_is_pad = image_is_pad.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        first_frame_latents = input_latents[:, :, 0:1] if fuse_flag else None
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }


def _plain_dict(value: Any, default=None):
    if value is None:
        return {} if default is None else default
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict-like configuration, got {type(value)}")
    return dict(value)


def create_cached_fasterwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    condition_layers=None,
    video_kv_fusion: str | None = None,
    video_kv_fusion_init: str = "current_layer",
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    video_dit_config = _plain_dict(video_dit_config)
    action_dit_config = _plain_dict(action_dit_config)
    video_scheduler = _plain_dict(video_scheduler)
    action_scheduler = _plain_dict(action_scheduler)
    loss = _plain_dict(loss)
    if condition_layers is not None:
        condition_layers = list(condition_layers)

    return CachedFasterWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        condition_layers=condition_layers,
        video_kv_fusion=video_kv_fusion,
        video_kv_fusion_init=str(video_kv_fusion_init),
    )
