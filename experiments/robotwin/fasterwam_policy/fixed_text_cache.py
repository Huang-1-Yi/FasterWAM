"""Create and load a fixed FasterWAM text condition next to a checkpoint."""

from __future__ import annotations

import gc
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch

from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fasterwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer


def build_fixed_prompt(instruction: str) -> str:
    task = str(instruction).strip()
    if not task:
        raise ValueError("`fixed_instruction` must not be empty.")
    return DEFAULT_PROMPT.format(task=task)


def _encoder_id(model_id: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "", str(model_id).split("/")[-1].lower())
    return value or "textenc"


def default_cache_path(
    checkpoint_path: str | Path,
    prompt: str,
    context_len: int,
    model_id: str,
) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve().parent
    return checkpoint_dir / "text_embeddings" / (
        f"{digest}.t5_len{context_len}.{_encoder_id(model_id)}.pt"
    )


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _load_and_validate(
    cache_path: Path,
    prompt: str,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "context" not in payload or "mask" not in payload:
        raise ValueError(f"Invalid text cache payload: {cache_path}")

    cached_prompt = payload.get("prompt")
    if cached_prompt is not None and cached_prompt != prompt:
        raise ValueError(
            f"Text cache prompt mismatch: expected {prompt!r}, got {cached_prompt!r}"
        )

    expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cached_hash = payload.get("prompt_sha256")
    if cached_hash is not None and cached_hash != expected_hash:
        raise ValueError(
            f"Text cache hash mismatch: expected {expected_hash}, got {cached_hash}"
        )

    context = payload["context"].detach().to(device="cpu", dtype=torch.bfloat16)
    mask = payload["mask"].detach().to(device="cpu", dtype=torch.bool)
    if context.ndim != 2:
        raise ValueError(f"Cached context must be [L,D], got {tuple(context.shape)}")
    if mask.ndim != 1:
        raise ValueError(f"Cached mask must be [L], got {tuple(mask.shape)}")
    if context.shape[0] != context_len or mask.shape[0] != context_len:
        raise ValueError(
            f"Text cache length mismatch: expected {context_len}, "
            f"got context={context.shape[0]} mask={mask.shape[0]}"
        )
    return context.contiguous(), mask.contiguous()


@torch.no_grad()
def _encode_and_save(
    *,
    cache_path: Path,
    prompt: str,
    context_len: int,
    model_id: str,
    tokenizer_model_id: str,
    redirect_common_files: bool,
    device: str,
) -> None:
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )
    ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device=device, dtype=torch.bool)
    context = text_encoder(ids, mask)

    payload = {
        "schema_version": 1,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model_id": model_id,
        "tokenizer_model_id": tokenizer_model_id,
        "context_len": int(context_len),
        # Store the same raw form used by scripts/precompute_text_embeds.py.
        "context": context[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "mask": mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous(),
    }
    _atomic_save(payload, cache_path)

    del context, mask, ids, tokenizer, text_encoder
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_fixed_text_condition(
    *,
    checkpoint_path: str | Path,
    instruction: str,
    model_id: str,
    tokenizer_model_id: str,
    context_len: int,
    redirect_common_files: bool,
    encoding_device: str,
    cache_path: str | Path | None = None,
    create_if_missing: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Path, str]:
    """Return inference-ready context and mask, creating a sidecar cache once."""

    prompt = build_fixed_prompt(instruction)
    resolved_cache = (
        Path(cache_path).expanduser().resolve()
        if cache_path is not None and str(cache_path).strip()
        else default_cache_path(checkpoint_path, prompt, context_len, model_id)
    )
    if not resolved_cache.is_file():
        if not create_if_missing:
            raise FileNotFoundError(f"Fixed text embedding cache not found: {resolved_cache}")
        _encode_and_save(
            cache_path=resolved_cache,
            prompt=prompt,
            context_len=context_len,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
            device=encoding_device,
        )

    context, raw_mask = _load_and_validate(resolved_cache, prompt, context_len)
    # Match RobotVideoDataset and FastWAM.encode_prompt exactly: zero padded
    # token vectors, then expose an all-true attention mask.
    context[~raw_mask] = 0.0
    inference_mask = torch.ones_like(raw_mask)
    return context, inference_mask, resolved_cache, prompt
