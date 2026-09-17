import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fasterwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fasterwam.utils import misc
from fasterwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()
repo_root = Path(__file__).resolve().parents[1]
misc.register_work_dir("/media/sata4t/hy_fasterwam_cache/vae_batch_benchmark")
Path("/media/sata4t/hy_fasterwam_cache/vae_batch_benchmark").mkdir(parents=True, exist_ok=True)
with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
    cfg = compose(config_name="train", overrides=["task=fold100_fasterwam_action_full_long"])

dataset = instantiate(cfg.data.train)
videos = torch.stack([dataset._get(idx)["video"] for idx in range(8)]).to(
    device="cuda", dtype=torch.bfloat16
)
_, _, vae_cfg, _ = _resolve_configs(
    model_id=str(cfg.model.model_id),
    tokenizer_model_id=str(cfg.model.tokenizer_model_id),
    redirect_common_files=bool(cfg.model.redirect_common_files),
)
vae_cfg.download_if_necessary()
vae = _load_registered_model(
    vae_cfg.path, "wan_video_vae", torch_dtype=torch.bfloat16, device="cuda"
).eval()

torch.cuda.synchronize()
start = time.perf_counter()
individual = vae.encode(videos, device="cuda", tiled=False)
torch.cuda.synchronize()
individual_seconds = time.perf_counter() - start

torch.cuda.synchronize()
start = time.perf_counter()
batched = vae.model.encode(videos, vae.scale)
torch.cuda.synchronize()
batched_seconds = time.perf_counter() - start

delta = (individual.float() - batched.float()).abs()
print(f"shape={tuple(individual.shape)}")
print(f"individual_seconds={individual_seconds:.6f}")
print(f"batched_seconds={batched_seconds:.6f}")
print(f"speedup={individual_seconds / batched_seconds:.3f}")
print(f"max_abs_diff={delta.max().item():.9f}")
print(f"mean_abs_diff={delta.mean().item():.9f}")
print(f"exact_equal={torch.equal(individual, batched)}")
