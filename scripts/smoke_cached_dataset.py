from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from fasterwam.utils import misc
from fasterwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()
repo_root = Path(__file__).resolve().parents[1]
work_dir = Path("/media/sata4t/hy_fasterwam_cache/cached_dataset_smoke")
work_dir.mkdir(parents=True, exist_ok=True)
misc.register_work_dir(str(work_dir))

with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
    original_cfg = compose(
        config_name="train", overrides=["task=fold100_fasterwam_action_full_long"]
    )
    cached_cfg = compose(
        config_name="train",
        overrides=["task=fold100_fasterwam_action_full_long_vae_cached"],
    )

with open_dict(cached_cfg.data.train):
    cached_cfg.data.train.allow_incomplete_cache = True

original = instantiate(original_cfg.data.train)
cached = instantiate(cached_cfg.data.train)
original_sample = original._get(0)
cached_sample = cached._get(0)

assert "video" in original_sample
assert "video" not in cached_sample
assert tuple(cached_sample["video_latents"].shape) == (48, 3, 24, 20)
for key in ("action", "proprio", "context", "context_mask", "image_is_pad", "action_is_pad"):
    left = original_sample[key]
    right = cached_sample[key]
    if left.dtype.is_floating_point:
        delta = (left.float() - right.float()).abs().max().item()
        assert delta == 0.0, (key, delta)
    else:
        assert torch.equal(left, right), key

print("cached_dataset_smoke=OK")
print(f"latent_shape={tuple(cached_sample['video_latents'].shape)}")
print("raw_video_decoding_in_cached_path=false")
