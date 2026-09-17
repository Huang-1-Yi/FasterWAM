import json
from pathlib import Path

import numpy as np
import torch

from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset


class CachedRobotVideoDataset(RobotVideoDataset):
    """RobotVideoDataset that replaces each processed video with a cached VAE latent."""

    def __init__(
        self,
        *args,
        vae_latent_cache_dir: str,
        cache_split: str,
        drop_video_after_cache: bool = True,
        allow_incomplete_cache: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Cache generation uses the original dataset. Cached training no longer
        # needs to decode any camera stream from MP4.
        self.lerobot_dataset._set_return_images(False)
        self.vae_latent_cache_dir = Path(vae_latent_cache_dir)
        self.cache_split = str(cache_split)
        self.drop_video_after_cache = bool(drop_video_after_cache)
        self.allow_incomplete_cache = bool(allow_incomplete_cache)
        split_dir = self.vae_latent_cache_dir / self.cache_split
        manifest_path = split_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing VAE latent manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as file:
            self.latent_manifest = json.load(file)
        if not bool(self.latent_manifest.get("complete", False)) and not self.allow_incomplete_cache:
            raise RuntimeError(f"VAE latent cache is incomplete: {manifest_path}")
        if int(self.latent_manifest["length"]) != len(self):
            raise ValueError(
                "VAE cache length mismatch: "
                f"cache={self.latent_manifest['length']} dataset={len(self)}"
            )
        if self.latent_manifest.get("dtype") != "bfloat16":
            raise ValueError(
                f"Expected bfloat16 latent cache, got {self.latent_manifest.get('dtype')}"
            )
        self.latent_shape = tuple(int(v) for v in self.latent_manifest["latent_shape"])
        self.latent_path = split_dir / self.latent_manifest["data_file"]
        self.done_path = split_dir / self.latent_manifest["done_file"]
        if not self.latent_path.is_file() or not self.done_path.is_file():
            raise FileNotFoundError(f"Incomplete VAE cache files under {split_dir}")
        done = np.memmap(self.done_path, mode="r", dtype=np.uint8, shape=(len(self),))
        if int(done.sum()) != len(self) and not self.allow_incomplete_cache:
            raise RuntimeError(f"VAE cache completion bitmap is incomplete: {self.done_path}")
        del done
        self._latent_memmap = None

    def _open_latent_memmap(self):
        if self._latent_memmap is None:
            self._latent_memmap = np.memmap(
                self.latent_path,
                mode="r",
                dtype=np.uint16,
                shape=(len(self), *self.latent_shape),
            )
        return self._latent_memmap

    def _cached_latent(self, idx: int) -> torch.Tensor:
        if self.allow_incomplete_cache:
            done = np.memmap(self.done_path, mode="r", dtype=np.uint8, shape=(len(self),))
            is_done = int(done[int(idx)]) == 1
            del done
            if not is_done:
                raise RuntimeError(f"VAE latent cache index {idx} is not complete.")
        raw_bits = np.array(self._open_latent_memmap()[int(idx)], copy=True)
        return torch.from_numpy(raw_bits).view(torch.bfloat16)

    def _get_action_only_sample(self, idx):
        base = self.lerobot_dataset
        raw = base.multi_dataset[idx]
        raw = base._split_lerobot_sample(raw)
        sample = {
            "idx": idx,
            "task": raw["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in base.state_meta:
            sample["state"][meta["key"]] = base._get_state(meta, raw)
        for meta in base.action_meta:
            sample["action"][meta["key"]] = base._get_action(meta, raw)
        sample["action_is_pad"] = raw[f"{base.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = raw[f"{base.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = raw[f"{base.image_meta[0]['lerobot_key']}_is_pad"]
        sample = base._get_additional_data(sample, raw)
        for key, value in raw.items():
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = value
        return base.processor.preprocess(sample)

    def _get(self, idx):
        sample = self._get_action_only_sample(idx)
        image_is_pad = sample["image_is_pad"][self.video_sample_indices]
        num_video_frames = len(self.video_sample_indices)
        action = sample["action"]
        if int(action.shape[0]) % (num_video_frames - 1) != 0:
            raise ValueError(
                "Action horizon must be divisible by cached video transitions: "
                f"{action.shape[0]} vs {num_video_frames - 1}."
            )

        instruction = DEFAULT_PROMPT.format(task=sample["instruction"])
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return {
            "video_latents": self._cached_latent(idx),
            "action": action,
            "proprio": sample["proprio"][:-1, :],
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_latent_memmap"] = None
        return state
