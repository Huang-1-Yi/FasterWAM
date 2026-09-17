import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from fasterwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fasterwam.utils import misc
from fasterwam.utils.config_resolvers import register_default_resolvers
from fasterwam.utils.logging_config import setup_logging


class StrictIndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        idx = int(self.indices[offset])
        sample = self.dataset._get(idx)
        return {"cache_index": idx, "video": sample["video"]}


def atomic_json(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def config_fingerprint(node) -> str:
    payload = OmegaConf.to_container(node, resolve=True)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.inference_mode()
def encode_batch(vae, videos, device):
    videos = videos.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    # The public wrapper loops over samples. The underlying VAE supports a true
    # batch dimension and is substantially faster on high-memory GPUs.
    return vae.model.encode(videos, vae.scale).detach().cpu().contiguous()


def prepare_split(dataset, split, split_cfg, cache_root, vae, device, args):
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"
    data_file = "latents.bf16.bin"
    done_file = "completed.u8.bin"
    data_path = split_dir / data_file
    done_path = split_dir / done_file
    length = len(dataset)
    fingerprint = config_fingerprint(split_cfg)

    manifest = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if int(manifest["length"]) != length or manifest["config_sha256"] != fingerprint:
            raise RuntimeError(
                f"Existing cache metadata does not match {split} dataset. "
                f"Use a new cache directory instead of overwriting {split_dir}."
            )
        latent_shape = tuple(int(v) for v in manifest["latent_shape"])
    else:
        first = dataset._get(0)
        first_latent = encode_batch(vae, first["video"].unsqueeze(0), device)[0]
        latent_shape = tuple(int(v) for v in first_latent.shape)
        latent_mm = np.memmap(
            data_path,
            mode="w+",
            dtype=np.uint16,
            shape=(length, *latent_shape),
        )
        done_mm = np.memmap(done_path, mode="w+", dtype=np.uint8, shape=(length,))
        done_mm[:] = 0
        latent_mm[0] = first_latent.view(torch.uint16).numpy()
        latent_mm.flush()
        done_mm[0] = 1
        done_mm.flush()
        del latent_mm, done_mm
        manifest = {
            "version": 1,
            "split": split,
            "length": length,
            "latent_shape": list(latent_shape),
            "dtype": "bfloat16",
            "storage_dtype": "uint16_bits",
            "encoder_mode": "direct_batched_vae_model_encode",
            "encoder_batch_size": int(args.batch_size),
            "data_file": data_file,
            "done_file": done_file,
            "config_sha256": fingerprint,
            "complete": False,
        }
        atomic_json(manifest, manifest_path)

    latent_mm = np.memmap(
        data_path,
        mode="r+",
        dtype=np.uint16,
        shape=(length, *latent_shape),
    )
    done_mm = np.memmap(done_path, mode="r+", dtype=np.uint8, shape=(length,))
    pending = np.flatnonzero(done_mm == 0).tolist()
    if not pending:
        manifest["complete"] = True
        atomic_json(manifest, manifest_path)
        print(f"[{split}] already complete: {length} samples", flush=True)
        return

    loader = DataLoader(
        StrictIndexedDataset(dataset, pending),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    started = time.perf_counter()
    encoded = 0
    pending_done = []
    for batch in loader:
        indices = [int(v) for v in batch["cache_index"].tolist()]
        latents = encode_batch(vae, batch["video"], device)
        if tuple(latents.shape[1:]) != latent_shape:
            raise RuntimeError(
                f"Latent shape changed in {split}: {tuple(latents.shape[1:])} vs {latent_shape}"
            )
        latent_mm[indices] = latents.view(torch.uint16).numpy()
        pending_done.extend(indices)
        encoded += len(indices)

        if len(pending_done) >= args.flush_every:
            latent_mm.flush()
            done_mm[pending_done] = 1
            done_mm.flush()
            pending_done.clear()

        if encoded % args.log_every < len(indices):
            elapsed = time.perf_counter() - started
            rate = encoded / max(elapsed, 1e-6)
            remaining = (len(pending) - encoded) / max(rate, 1e-6)
            print(
                f"[{split}] new={encoded}/{len(pending)} total_done={int(done_mm.sum())}/{length} "
                f"rate={rate:.3f} sample/s eta={remaining / 3600:.2f}h",
                flush=True,
            )

    if pending_done:
        latent_mm.flush()
        done_mm[pending_done] = 1
        done_mm.flush()
    complete_count = int(done_mm.sum())
    del latent_mm, done_mm
    if complete_count != length:
        raise RuntimeError(f"Cache completion mismatch for {split}: {complete_count}/{length}")
    manifest["complete"] = True
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(manifest, manifest_path)
    print(f"[{split}] complete: {length} samples, shape={latent_shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fold100_fasterwam_action_full_long")
    parser.add_argument(
        "--cache-dir",
        default="/media/sata4t/hy_fasterwam_cache/fold100_vae_latents_wan22",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--flush-every", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    args = parser.parse_args()

    register_default_resolvers()
    setup_logging(log_level=logging.INFO)
    repo_root = Path(__file__).resolve().parents[1]
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(cache_root / "work"))
    (cache_root / "work").mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg.model
    _, _, vae_cfg, _ = _resolve_configs(
        model_id=str(model_cfg.model_id),
        tokenizer_model_id=str(model_cfg.tokenizer_model_id),
        redirect_common_files=bool(model_cfg.redirect_common_files),
    )
    vae_cfg.download_if_necessary()
    vae = _load_registered_model(
        vae_cfg.path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()

    for split in args.splits:
        split_cfg = cfg.data[split]
        dataset = instantiate(split_cfg)
        prepare_split(dataset, split, split_cfg, cache_root, vae, device, args)

    print(f"All VAE latent caches complete under {cache_root}", flush=True)


if __name__ == "__main__":
    main()
