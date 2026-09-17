# FasterWAM fold100 action adaptation

This checkout adapts the released FasterWAM RoboTwin model to the local
fold100 bimanual Franka dataset while keeping the video expert frozen.

## Fixed inputs

- Repository commit: `83667817df0d4f823f39d90700e61ea2f432ac45`
- Dataset: `/media/disk7t/openpi_franka/data/fold_100`
- Dataset statistics: `/media/disk7t/FastWAM/data/fold100_train90_dataset_stats.json`
- Text cache: `/media/disk7t/FastWAM/data/text_embeds_cache/fold100`
- Released base checkpoint:
  `/media/sata4t/hy_fasterwam_checkpoints/fasterwam_release/robotwin/step_029355.pt`
- Outputs: `/media/sata4t/hy_fasterwam_runs/fold100`

RoboTwin uses 14-D joint-position actions, while fold100 stores 14-D absolute
dual-arm end-effector targets. The released action expert is therefore an
initialization, not an action-space-compatible policy.

## Stage A

Stage A freezes the video expert, VAE, and action transformer blocks. It trains
the action input/output and time/text projections, proprio encoder, and
Interval KV-Fusion logits. Checkpoints are lightweight action deltas and record
the released checkpoint path they depend on.

Before every run, inspect `nvidia-smi` and select only the agreed GPU.

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam
cd /media/disk7t/FasterWAM

export CUDA_VISIBLE_DEVICES=1
export DIFFSYNTH_MODEL_BASE_PATH=/media/disk7t/FastWAM/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_DATASETS_CACHE=/media/sata4t/hy_fasterwam_cache/hf_datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train_action_single.sh
```

The default Stage A run uses batch size 1, eight-step gradient accumulation,
800 optimizer steps, and saves a lightweight delta every 200 steps.

For a detached run, create a tmux session around the fixed runner:

```bash
tmux new-session -d -s fasterwam_fold100_stage_a \
  "bash /media/disk7t/FasterWAM/scripts/run_fold100_stage_a.sh"
cat /media/sata4t/hy_fasterwam_runs/fold100/latest_stage_a_run.txt
```

## Verified on 2026-09-15

- Core environment: Python 3.10.21, torch 2.7.1+cu128, CUDA runtime 12.8.
- Released checkpoint size: 11,117,757,817 bytes.
- Released checkpoint SHA256:
  `934684f2b60f78d493d14f30dba4554c0c064803f0a7f659aaf2b16e60d6c7ef`.
- Checkpoint payload: `mot`, `proprio_encoder`, `step`, `torch_dtype`;
  step 29355 with 1393 MoT tensors.
- Real fold100 sample:
  video `(3, 9, 384, 320)`, action `(32, 14)`, proprio `(32, 14)`,
  context `(128, 4096)`.
- One-step adapter smoke passed with 12,946,334 trainable parameters out of
  6.263B total; lightweight checkpoint size was about 75MB.
- Smoke output:
  `/media/sata4t/hy_fasterwam_runs/fold100_smoke`.
- Stage A background run:
  `/media/sata4t/hy_fasterwam_runs/fold100/stage_a_20260915_122926`.
- tmux session: `fasterwam_fold100_stage_a`.

The Hugging Face Datasets cache must stay on SATA because the default home
filesystem is effectively full. The prepared cache root is
`/media/sata4t/hy_fasterwam_cache/hf_datasets`.

## Clean full-action-expert pilot

The original Stage A froze all 30 action Transformer blocks and regressed in
offline evaluation.  The next controlled experiment starts again from the
released RoboTwin checkpoint (without loading any Stage A delta), freezes the
video/world path, and fine-tunes the complete action path for 500 optimizer
steps.

| Logical group | State | Reason |
|---|---|---|
| Video expert / world DiT | Frozen, eval mode | Preserve released world representation and avoid 5B optimizer state |
| VAE | Frozen, eval mode | Image latent codec is not task-specific action policy capacity |
| Text encoder | Not loaded / frozen | Training uses cached text embeddings |
| Action expert, including all 30 blocks | Trainable, train mode | Adapt RoboTwin action sequence prior to fold100 |
| Action input encoder and output head | Trainable | Adapt the 14 action channels |
| Action text/time projections | Trainable | Keep projections and action blocks jointly consistent |
| Proprio encoder | Trainable | Adapt RoboTwin proprioception to dual-Franka EEF state |
| Video KV-fusion logits | Trainable | Adapt use of video/world context |

The expected trainable count from the architecture is 558,957,611 parameters:
558,896,142 in the action expert, 61,440 in the proprio encoder, and 29 fusion
logits.  The runtime performs hard guards that fail the launch if the video
expert, VAE, or text encoder has any trainable parameter.

Pilot configuration:

- `configs/task/fold100_fasterwam_action_full_pilot.yaml`
- Base checkpoint: released RoboTwin step 29355
- Stage A delta: none
- Learning rate: `1e-5`
- Batch size: 1 with gradient accumulation 8
- Optimizer steps: 500
- Checkpoint interval: 100
- Video/action loss weights: 0/1

Only after checking `nvidia-smi`, launch on an explicitly approved physical
GPU.  Foreground:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_fold100_action_full_pilot.sh
```

Background:

```bash
tmux new-session -d -s fasterwam_fold100_action_full_pilot \
  "CUDA_VISIBLE_DEVICES=1 bash /media/disk7t/FasterWAM/scripts/run_fold100_action_full_pilot.sh"
```

Two-GPU launch uses DeepSpeed ZeRO-2 with no optimizer or parameter CPU
offload.  It data-parallelizes training and shards gradients/optimizer state;
it does not pool both GPUs into one large-memory device, so the frozen video
expert is still replicated on each GPU.

```bash
tmux new-session -d -s fasterwam_fold100_action_full_pilot_2gpu \
  "CUDA_VISIBLE_DEVICES=0,1 bash /media/disk7t/FasterWAM/scripts/run_fold100_action_full_pilot_2gpu.sh"
```

The two-GPU runner overrides gradient accumulation to 4, keeping the effective
batch at 8 while reducing the number of micro-steps per optimizer step.
Do not start the two-GPU run merely because allocated memory fits: inspect
utilization, temperature, and existing compute processes because concurrent
jobs can make it slower than an uncontended single-GPU run.

Every run writes these audit artifacts before or during training:

- `launch_manifest.txt`: time, host, user, physical GPU, Git commit, Python,
  PyTorch/CUDA, and the pre-launch GPU/process snapshot.
- `config.yaml`: fully resolved Hydra training configuration.
- `parameter_manifest.json`: exact total/trainable/frozen counts, logical group
  summaries, base/delta initialization, and every trainable parameter name.
- `train.log`: initialization, freeze guards, loss, gradient norm, learning
  rate, speed, checkpoint paths, and completion/error details.
- `checkpoints/weights/step_*.pt`: action deltas relative to the released base.

## Offline action evaluation

The Stage A files are lightweight action deltas, not standalone FasterWAM
checkpoints.  Do not pass them directly to the RoboTwin evaluator.  Use the
fold100 evaluator below: it loads the released RoboTwin checkpoint first,
overlays each delta, runs action-only inference on deterministic held-out
episode windows, and reports errors after converting actions back to the
native 14-D fold100 action space.  It does not control a robot.

Always inspect GPU use immediately before evaluation.  The following minimal
comparison evaluates the released base and the final Stage A delta on one
validation window with the same random seed:

```bash
ssh pro6000d
nvidia-smi

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam
cd /media/disk7t/FasterWAM

export CUDA_VISIBLE_DEVICES=1
export DIFFSYNTH_MODEL_BASE_PATH=/media/disk7t/FastWAM/checkpoints
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_DATASETS_CACHE=/media/sata4t/hy_fasterwam_cache/hf_datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_DIR=/media/sata4t/hy_fasterwam_runs/fold100/offline_eval_stage_a_quick
python -u scripts/eval_fold100_action.py \
  --include-base \
  --delta /media/sata4t/hy_fasterwam_runs/fold100/stage_a_20260915_122926/checkpoints/weights/step_000800.pt \
  --num-samples 1 \
  --num-inference-steps 4 \
  --output-dir "$EVAL_DIR" \
  2>&1 | tee "$EVAL_DIR.log"
```

After the minimal comparison succeeds, compare all saved deltas on five fixed
validation windows:

```bash
EVAL_DIR=/media/sata4t/hy_fasterwam_runs/fold100/offline_eval_stage_a_all
python -u scripts/eval_fold100_action.py \
  --include-base \
  --delta /media/sata4t/hy_fasterwam_runs/fold100/stage_a_20260915_122926/checkpoints/weights/step_000200.pt \
  --delta /media/sata4t/hy_fasterwam_runs/fold100/stage_a_20260915_122926/checkpoints/weights/step_000400.pt \
  --delta /media/sata4t/hy_fasterwam_runs/fold100/stage_a_20260915_122926/checkpoints/weights/step_000600.pt \
  --delta /media/sata4t/hy_fasterwam_runs/fold100/stage_a_20260915_122926/checkpoints/weights/step_000800.pt \
  --num-samples 5 \
  --num-inference-steps 10 \
  --output-dir "$EVAL_DIR" \
  2>&1 | tee "$EVAL_DIR.log"
```

Lower native-space MAE and RMSE are better.  Select a checkpoint from the
multi-sample result rather than from training loss alone.  The detailed JSON
result is written to `<output-dir>/metrics.json`.

### Stage A evaluation result (2026-09-15)

The action-only inference path was verified on physical GPU 1 using the same
one-pass future-cache method as the default RoboTwin deployment adapter.
Evaluation covered the centre window of every one of the ten held-out fold100
episodes, with 10 denoising steps and deterministic per-window seeds.

| Candidate | Native MAE | Native RMSE | Normalized MAE |
|---|---:|---:|---:|
| Hold current state baseline | 0.080404 | 0.203559 | 0.398504 |
| Released RoboTwin base | 0.145478 | 0.304983 | 0.485795 |
| Stage A step 400 | 0.241916 | 0.546130 | 0.564680 |
| Dataset mean-action baseline | 0.311090 | 0.487994 | 0.997089 |

The earlier five-window comparison found that all Stage A deltas were worse
than the released base; step 400 was the least bad Stage A candidate (native
MAE 0.315164 versus base 0.143603 in that run).  Per-dimension diagnostics
showed that Stage A slightly improved some Cartesian position dimensions but
strongly degraded both arms' rotation dimensions.  Therefore do **not** start
Stage B from this Stage A checkpoint yet.  First correct the action objective
or representation and add a validation gate; unfreezing the full action
expert now would amplify an already regressing action policy.

Detailed results:

- `/media/sata4t/hy_fasterwam_runs/fold100/offline_eval_stage_a_all_20260915_1431/metrics.json`
- `/media/sata4t/hy_fasterwam_runs/fold100/offline_eval_stage_a_diagnostic_20260915_1437/metrics.json`
- `/media/sata4t/hy_fasterwam_runs/fold100/offline_eval_stage_a_10episodes_20260915_1441/metrics.json`

## Optional VAE-latent cache for action training

The fold100 cached pipeline keeps the official Wan2.2 VAE transformation but
computes it once before training.  It stores BF16 latent bits with shape
`[48, 3, 24, 20]` for each sample.  During cached training, camera MP4 decoding,
image resize/crop, and VAE encoding are all bypassed; action, proprioception,
text context, and padding masks remain identical to the original dataset path.

Cache location:

```text
/media/sata4t/hy_fasterwam_cache/fold100_vae_latents_wan22_batched4_v1
```

The training cache contains a manifest, a BF16-bit mmap, and a completion
bitmap. An interrupted run resumes only the unfinished indices, and training
refuses an incomplete cache. Validation intentionally keeps the original video
dataset because the built-in evaluation renders rollout videos and computes
PSNR/SSIM against RGB ground truth every 500 steps.

To generate the training cache and automatically start single-GPU
action-expert training afterward:

```bash
ssh pro6000d
nvidia-smi
tmux new-session -s fasterwam_fold100_cache_then_train_gpu0
CUDA_VISIBLE_DEVICES=0 bash /media/disk7t/FasterWAM/scripts/cache_then_train_fold100_gpu0.sh
```

Detach with `Ctrl-B`, then `D`. Monitor with:

```bash
tail -f /media/sata4t/hy_fasterwam_cache/fold100_vae_latents_wan22_batched4_v1/precompute.log
```

After caching completes, the run directory is recorded in:

```text
/media/sata4t/hy_fasterwam_runs/fold100/latest_action_full_long_vae_cached_1gpu_run.txt
```

Relevant new files:

- `scripts/precompute_vae_latents.py`
- `scripts/run_fold100_vae_cache_gpu0.sh`
- `scripts/run_fold100_action_full_long_vae_cached_1gpu.sh`
- `scripts/cache_then_train_fold100_gpu0.sh`
- `src/fasterwam/models/wan22/cached_fasterwam.py`
- `src/fasterwam/datasets/lerobot/cached_robot_video_dataset.py`
- `src/fasterwam/datasets/lerobot/processors/cached_action_processor.py`
- `configs/model/fasterwam_vae_cached.yaml`
- `configs/data/fold100_vae_cached.yaml`
- `configs/task/fold100_fasterwam_action_full_long_vae_cached.yaml`
