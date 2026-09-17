"""Offline action evaluation for the local fold100 adaptation.

This entrypoint deliberately evaluates actions only.  It loads the released
FasterWAM checkpoint first, overlays each lightweight action delta, runs the
same deterministic validation windows for every candidate, denormalizes the
predictions, and reports errors in the dataset's native 14-D action space.
It never sends commands to a robot.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fasterwam.utils import misc
from fasterwam.utils.logging_config import setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path(
    "/media/sata4t/hy_fasterwam_checkpoints/fasterwam_release/robotwin/step_029355.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--delta", type=Path, action="append", default=[])
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/sata4t/hy_fasterwam_runs/fold100/offline_eval"),
    )
    parser.add_argument("--task-config", default="fold100_fasterwam_action_adapter")
    args = parser.parse_args()
    if not args.include_base and not args.delta:
        parser.error("pass --include-base and/or at least one --delta")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    return args


def load_config(task_config: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train", overrides=[f"task={task_config}"])


def load_delta_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "fasterwam_action_delta_v1":
        raise ValueError(f"Unsupported action delta format in {path}: {payload.get('format')!r}")
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Action delta has no trainable_state_dict: {path}")
    return payload


def overlay_named_parameters(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    unknown = sorted(set(state) - set(named))
    if unknown:
        raise RuntimeError(f"Delta contains unknown parameter names: {unknown[:20]}")
    with torch.no_grad():
        for name, value in state.items():
            target = named[name]
            if target.shape != value.shape:
                raise RuntimeError(
                    f"Delta shape mismatch for {name}: {tuple(value.shape)} vs {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def choose_validation_indices(dataset, count: int) -> list[int]:
    """Choose deterministic episode-centre windows before filling uniformly."""
    count = min(count, len(dataset))
    base = dataset.lerobot_dataset
    indices: list[int] = []
    episode_index = getattr(base, "episode_data_index", None)
    if episode_index is not None:
        starts = episode_index["from"].tolist()
        ends = episode_index["to"].tolist()
        centres = [int((start + end - 1) // 2) for start, end in zip(starts, ends) if end > start]
        if centres:
            positions = np.linspace(0, len(centres) - 1, num=min(count, len(centres)), dtype=int)
            indices.extend(centres[int(pos)] for pos in positions)
    if len(indices) < count:
        fill = np.linspace(0, len(dataset) - 1, num=count, dtype=int).tolist()
        indices.extend(idx for idx in fill if idx not in indices)
    return indices[:count]


def denormalize_action_and_state(
    processor, action: torch.Tensor, proprio: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if proprio.ndim == 2:
        proprio = proprio.unsqueeze(0)
    batch = {
        "action": action.detach().cpu().float(),
        "state": proprio.detach().cpu().float(),
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged = {
        "action": {
            meta["key"]: batch["action"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["action"]
        },
        "state": {
            meta["key"]: batch["state"][meta["key"]].squeeze(0)
            for meta in processor.shape_meta["state"]
        },
    }
    output = processor.action_state_merger.forward(merged)
    return output["action"].float(), output["state"].float()


def denormalize_action(processor, action: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    return denormalize_action_and_state(processor, action, proprio)[0]


def normalize_native_action(
    processor, action_native: torch.Tensor, state_native: torch.Tensor
) -> torch.Tensor:
    action_fields = {}
    cursor = 0
    for meta in processor.shape_meta["action"]:
        width = int(meta["shape"])
        action_fields[meta["key"]] = action_native[:, cursor : cursor + width]
        cursor += width
    state_fields = {}
    cursor = 0
    for meta in processor.shape_meta["state"]:
        width = int(meta["shape"])
        state_fields[meta["key"]] = state_native[:, cursor : cursor + width]
        cursor += width
    batch = processor.normalizer.forward({"action": action_fields, "state": state_fields})
    return processor.action_state_merger.forward(batch)["action"].float()


def summarize_errors(diff_native: torch.Tensor, diff_normalized: torch.Tensor) -> dict[str, Any]:
    abs_native = diff_native.abs()
    sq_native = diff_native.square()
    abs_normalized = diff_normalized.abs()
    sq_normalized = diff_normalized.square()
    groups = {
        "left_position": slice(0, 3),
        "left_rotation": slice(3, 6),
        "left_gripper": slice(6, 7),
        "right_position": slice(7, 10),
        "right_rotation": slice(10, 13),
        "right_gripper": slice(13, 14),
    }
    return {
        "mae": float(abs_native.mean().item()),
        "rmse": float(sq_native.mean().sqrt().item()),
        "max_abs": float(abs_native.max().item()),
        "normalized_mae": float(abs_normalized.mean().item()),
        "normalized_rmse": float(sq_normalized.mean().sqrt().item()),
        "per_dimension_mae": [float(value) for value in abs_native.mean(dim=0).tolist()],
        "group_mae": {
            name: float(abs_native[:, dims].mean().item()) for name, dims in groups.items()
        },
    }


def evaluate_simple_baselines(dataset, indices: list[int]) -> dict[str, Any]:
    processor = dataset.lerobot_dataset.processor
    errors: dict[str, dict[str, list[torch.Tensor]]] = {
        "dataset_mean_action": {"native": [], "normalized": []},
        "hold_current_state": {"native": [], "normalized": []},
    }
    for index in indices:
        sample = dataset[index]
        gt_normalized = sample["action"].detach().cpu().float()
        proprio = sample["proprio"].detach().cpu().float()
        gt_native, state_native = denormalize_action_and_state(
            processor, gt_normalized, proprio
        )
        valid = ~sample["action_is_pad"].detach().cpu().bool()

        mean_normalized = torch.zeros_like(gt_normalized)
        mean_native = denormalize_action(processor, mean_normalized, proprio)
        errors["dataset_mean_action"]["native"].append(mean_native[valid] - gt_native[valid])
        errors["dataset_mean_action"]["normalized"].append(
            mean_normalized[valid] - gt_normalized[valid]
        )

        hold_native = state_native[0:1].expand_as(gt_native).clone()
        hold_normalized = normalize_native_action(processor, hold_native, state_native)
        errors["hold_current_state"]["native"].append(hold_native[valid] - gt_native[valid])
        errors["hold_current_state"]["normalized"].append(
            hold_normalized[valid] - gt_normalized[valid]
        )

    return {
        name: summarize_errors(
            torch.cat(values["native"], dim=0),
            torch.cat(values["normalized"], dim=0),
        )
        for name, values in errors.items()
    }
@torch.inference_mode()
def evaluate_candidate(
    model,
    dataset,
    indices: list[int],
    *,
    num_inference_steps: int,
    seed: int,
) -> dict[str, Any]:
    processor = dataset.lerobot_dataset.processor
    sample_results: list[dict[str, Any]] = []
    absolute_errors: list[torch.Tensor] = []
    squared_errors: list[torch.Tensor] = []
    normalized_absolute_errors: list[torch.Tensor] = []
    normalized_squared_errors: list[torch.Tensor] = []
    started = time.perf_counter()

    for ordinal, index in enumerate(indices):
        sample = dataset[index]
        action = sample["action"]
        proprio = sample["proprio"]
        # Match the default RoboTwin deployment path.  FasterWAM's one-pass
        # future-cache method still needs the configured video horizon even
        # though this evaluator only consumes its action output.
        pred = model.infer_action_one_pass_future_cache(
            prompt=None,
            input_image=sample["video"][:, 0],
            action_horizon=int(action.shape[0]),
            num_video_frames=int(sample["video"].shape[1]),
            proprio=proprio[0],
            context=sample["context"],
            context_mask=sample["context_mask"],
            num_inference_steps=num_inference_steps,
            seed=seed + ordinal,
            rand_device="cpu",
            tiled=False,
        )["action"]

        pred_native = denormalize_action(processor, pred, proprio)
        gt_native = denormalize_action(processor, action, proprio)
        if pred_native.shape != gt_native.shape:
            raise RuntimeError(f"Prediction/GT shape mismatch: {pred_native.shape} vs {gt_native.shape}")

        valid = ~sample["action_is_pad"].detach().cpu().bool()
        if not bool(valid.any()):
            raise RuntimeError(f"Validation sample {index} contains only padded actions")
        diff = pred_native[valid] - gt_native[valid]
        diff_normalized = pred.detach().cpu().float()[valid] - action.detach().cpu().float()[valid]
        absolute_errors.append(diff.abs())
        squared_errors.append(diff.square())
        normalized_absolute_errors.append(diff_normalized.abs())
        normalized_squared_errors.append(diff_normalized.square())
        sample_results.append(
            {
                "index": int(index),
                "valid_steps": int(valid.sum().item()),
                "mae": float(diff.abs().mean().item()),
                "rmse": float(diff.square().mean().sqrt().item()),
                "max_abs": float(diff.abs().max().item()),
            }
        )
        print(
            f"  sample {ordinal + 1}/{len(indices)} index={index} "
            f"MAE={sample_results[-1]['mae']:.6f} RMSE={sample_results[-1]['rmse']:.6f}",
            flush=True,
        )

    abs_all = torch.cat(absolute_errors, dim=0)
    sq_all = torch.cat(squared_errors, dim=0)
    normalized_abs_all = torch.cat(normalized_absolute_errors, dim=0)
    normalized_sq_all = torch.cat(normalized_squared_errors, dim=0)
    metrics = {
        "num_samples": len(indices),
        "indices": indices,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "samples": sample_results,
    }
    metrics.update(summarize_errors(abs_all, normalized_abs_all))
    # summarize_errors only depends on absolute/squared magnitudes for these
    # aggregate metrics, so passing the concatenated absolute errors is valid.
    metrics["rmse"] = float(sq_all.mean().sqrt().item())
    metrics["normalized_rmse"] = float(normalized_sq_all.mean().sqrt().item())
    return metrics


def main() -> None:
    args = parse_args()
    setup_logging(log_level=logging.INFO, is_main_process=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(args.output_dir))

    if not args.base.is_file():
        raise FileNotFoundError(args.base)
    delta_payloads = [(path, load_delta_payload(path)) for path in args.delta]

    cfg = load_config(args.task_config)
    precision = _normalize_mixed_precision(cfg.mixed_precision)
    dtype = _mixed_precision_to_model_dtype(precision)
    model = instantiate(cfg.model, model_dtype=dtype, device="cuda:0")
    model.load_checkpoint(str(args.base))
    model.eval()
    dataset = instantiate(cfg.data.val)
    indices = choose_validation_indices(dataset, args.num_samples)
    print(f"Validation dataset size={len(dataset)}, fixed indices={indices}", flush=True)

    changed_names = sorted(
        {name for _, payload in delta_payloads for name in payload["trainable_state_dict"]}
    )
    named_parameters = dict(model.named_parameters())
    base_state = {
        name: named_parameters[name].detach().cpu().clone()
        for name in changed_names
    }

    candidates: list[tuple[str, dict[str, torch.Tensor] | None]] = []
    if args.include_base:
        candidates.append(("base_step_029355", None))
    candidates.extend(
        (path.stem, payload["trainable_state_dict"])
        for path, payload in delta_payloads
    )

    results: dict[str, Any] = {
        "base_checkpoint": str(args.base.resolve()),
        "validation_size": len(dataset),
        "indices": indices,
        "candidates": {},
    }
    results["baselines"] = evaluate_simple_baselines(dataset, indices)
    for label, state in candidates:
        overlay_named_parameters(model, base_state)
        if state is not None:
            overlay_named_parameters(model, state)
        print(f"Evaluating {label}", flush=True)
        results["candidates"][label] = evaluate_candidate(
            model,
            dataset,
            indices,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary (lower is better)")
    for label, metrics in results["baselines"].items():
        print(
            f"  {label}: MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
            f"normalized_MAE={metrics['normalized_mae']:.6f}"
        )
    for label, metrics in results["candidates"].items():
        print(
            f"  {label}: MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
            f"normalized_MAE={metrics['normalized_mae']:.6f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
