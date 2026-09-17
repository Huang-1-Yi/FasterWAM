#!/usr/bin/env python3
"""Merge a FasterWAM action delta into a full deployment checkpoint.

The released FasterWAM checkpoints store the model under ``mot`` plus an
optional top-level ``proprio_encoder``.  Action-only training checkpoints use
the lightweight ``fasterwam_action_delta_v1`` format and keep names from
``model.named_parameters()``.  This tool validates and translates those names,
then writes a full checkpoint that ``FastWAM.load_checkpoint`` can consume.

Both inputs are opened with mmap-backed ``torch.load``.  The merge replaces
dictionary references rather than cloning tensors, keeping process memory well
below the combined checkpoint size.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DELTA_FORMAT = "fasterwam_action_delta_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Full FasterWAM base checkpoint")
    parser.add_argument("--delta", required=True, type=Path, help="fasterwam_action_delta_v1 checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="Merged full checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the complete merge plan without writing an output file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output file",
    )
    return parser.parse_args()


def load_mmap(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError as exc:
        raise RuntimeError(
            "This script requires a PyTorch version whose torch.load supports mmap=True"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} payload must be a dict, got {type(payload).__name__}")
    return payload


def require_tensor_mapping(value: object, label: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    bad = [name for name, tensor in value.items() if not isinstance(name, str) or not isinstance(tensor, torch.Tensor)]
    if bad:
        raise TypeError(f"{label} contains non-string keys or non-tensor values: {bad[:10]}")
    return value


def destination_for(delta_name: str) -> tuple[str, str]:
    """Return (top-level state name, key within that state)."""
    translations = (
        ("action_expert.", "mot", "mixtures.action."),
        ("video_expert.", "mot", "mixtures.video."),
        ("mot.", "mot", ""),
        ("dit.", "mot", ""),
        ("proprio_encoder.", "proprio_encoder", ""),
    )
    for prefix, group, output_prefix in translations:
        if delta_name.startswith(prefix):
            suffix = delta_name[len(prefix) :]
            if not suffix:
                break
            return group, output_prefix + suffix
    raise KeyError(
        f"Unsupported delta parameter {delta_name!r}; it cannot be represented "
        "by the full FasterWAM mot/proprio_encoder checkpoint format"
    )


def build_merge_plan(
    base: dict[str, Any], delta: dict[str, Any]
) -> list[tuple[str, str, str, torch.Tensor]]:
    if delta.get("format") != DELTA_FORMAT:
        raise ValueError(
            f"Unsupported delta format {delta.get('format')!r}; expected {DELTA_FORMAT!r}"
        )
    base_mot = require_tensor_mapping(base.get("mot"), "base['mot']")
    delta_state = require_tensor_mapping(
        delta.get("trainable_state_dict"), "delta['trainable_state_dict']"
    )
    base_groups: dict[str, Mapping[str, torch.Tensor]] = {"mot": base_mot}
    if "proprio_encoder" in base:
        base_groups["proprio_encoder"] = require_tensor_mapping(
            base["proprio_encoder"], "base['proprio_encoder']"
        )

    plan: list[tuple[str, str, str, torch.Tensor]] = []
    destinations: dict[tuple[str, str], str] = {}
    for source_name, source_tensor in delta_state.items():
        group, destination_name = destination_for(source_name)
        if group not in base_groups:
            raise KeyError(f"Delta parameter {source_name!r} requires missing base group {group!r}")
        destination = (group, destination_name)
        if destination in destinations:
            raise ValueError(
                f"Delta parameters {destinations[destination]!r} and {source_name!r} "
                f"both map to {group}.{destination_name}"
            )
        destinations[destination] = source_name
        if destination_name not in base_groups[group]:
            raise KeyError(
                f"Delta parameter {source_name!r} maps to unknown base key "
                f"{group}.{destination_name}"
            )
        target_tensor = base_groups[group][destination_name]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {source_name}: delta={tuple(source_tensor.shape)} "
                f"base={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"Dtype mismatch for {source_name}: delta={source_tensor.dtype} "
                f"base={target_tensor.dtype}"
            )
        plan.append((source_name, group, destination_name, source_tensor))

    if len(plan) != len(delta_state):
        raise AssertionError("Internal error: merge plan did not cover every delta tensor")
    return plan


def apply_plan(
    base: dict[str, Any],
    delta: dict[str, Any],
    plan: list[tuple[str, str, str, torch.Tensor]],
    base_path: Path,
    delta_path: Path,
) -> None:
    for _, group, destination_name, source_tensor in plan:
        base[group][destination_name] = source_tensor
    if "step" in delta:
        base["step"] = int(delta["step"])
    base["merged_action_delta"] = {
        "format": DELTA_FORMAT,
        "base_checkpoint": str(base_path.resolve()),
        "delta_checkpoint": str(delta_path.resolve()),
        "training_scope": delta.get("training_scope"),
        "step": delta.get("step"),
        "epoch": delta.get("epoch"),
        "tensor_count": len(plan),
    }


def verify_serialized_output(
    temp_path: Path,
    base: dict[str, Any],
    delta: dict[str, Any],
    plan: list[tuple[str, str, str, torch.Tensor]],
) -> None:
    output = load_mmap(temp_path, "temporary merged")
    output_mot = require_tensor_mapping(output.get("mot"), "output['mot']")
    base_mot = require_tensor_mapping(base.get("mot"), "base['mot']")
    if set(output_mot) != set(base_mot):
        raise RuntimeError("Serialized output changed the set of mot keys")
    if output.get("step") != int(delta.get("step", output.get("step"))):
        raise RuntimeError("Serialized output step does not match the delta step")

    for source_name, group, destination_name, source_tensor in plan:
        output_group = require_tensor_mapping(output.get(group), f"output[{group!r}]")
        output_tensor = output_group[destination_name]
        if output_tensor.shape != source_tensor.shape or output_tensor.dtype != source_tensor.dtype:
            raise RuntimeError(f"Serialized output schema mismatch for {source_name}")
        if not torch.equal(output_tensor, source_tensor):
            raise RuntimeError(f"Serialized output data mismatch for {source_name}")


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    args = parse_args()
    base = load_mmap(args.base, "base")
    delta = load_mmap(args.delta, "delta")
    plan = build_merge_plan(base, delta)

    group_counts: dict[str, int] = {}
    parameter_count = 0
    for _, group, _, tensor in plan:
        group_counts[group] = group_counts.get(group, 0) + 1
        parameter_count += tensor.numel()
    print(
        f"validated delta format={DELTA_FORMAT} step={delta.get('step')} "
        f"tensors={len(plan)} parameters={parameter_count} groups={group_counts}",
        flush=True,
    )
    if args.dry_run:
        print("dry_run=true output_written=false", flush=True)
        return 0

    output = args.output.resolve()
    if output == args.base.resolve() or output == args.delta.resolve():
        raise ValueError("Output must not overwrite either input checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace): {output}")
    temp_path = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temp_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_path}")

    apply_plan(base, delta, plan, args.base, args.delta)
    try:
        torch.save(base, temp_path)
        fsync_file(temp_path)
        verify_serialized_output(temp_path, base, delta, plan)
        os.replace(temp_path, output)
        fsync_directory(output.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(
        f"merge_complete=true output={output} size_bytes={output.stat().st_size} "
        f"verified_delta_tensors={len(plan)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
