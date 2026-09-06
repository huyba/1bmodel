# 1B Dense Model — Toy Pretraining Pipeline

A from-scratch pretraining pipeline for a ~1.24B parameter dense transformer (GQA + RoPE + SwiGLU + RMSNorm), trained on ~25B tokens sampled from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). It's a toy/learning project: the goal is to exercise the full stack of a real LLM pretraining run — data download, tokenization/packing, a hand-written model and training loop, and multi-GPU DDP — end to end on a small-enough model to actually finish training.

The full architecture, parameter count, and memory/throughput derivation live in [design.md](design.md) (worked through by hand: ~1.24B params, ~12h estimated on 8×H100, batch/sequence-length sizing, etc.) — read that first if you want the reasoning behind the numbers used below.

## Repo layout

| File | Purpose |
|---|---|
| [model.py](model.py) | Model definition: `Transformer1B` (GQA attention, RoPE, SwiGLU FFN, RMSNorm, tied embeddings) |
| [dataset.py](dataset.py) | `PretrainBinaryDataset` — streams fixed-length token chunks out of packed `.bin` shards, sharded per DDP rank |
| [train.py](train.py) | Training loop: LR schedule, gradient accumulation, DDP, checkpointing |
| [download_script.bash](download_script.bash) | Downloads the raw FineWeb-Edu parquet shards |
| [prepare_data_local.py](prepare_data_local.py) | Tokenizes local parquet files and packs them into a single binary token shard |
| [prepare_data_multiprocess.py](prepare_data_multiprocess.py) | Same idea, parallelized across CPU cores for faster tokenization |
| [prepare_data.py](prepare_data.py) | Alternative: streams FineWeb-Edu directly from the Hub (no local parquet download needed) and packs on the fly |
| [design.md](design.md) | Architecture, parameter count, and memory/throughput design notes |

## Requirements

Python 3.x with a virtualenv (this repo uses `.venv/`). Core dependencies:

```bash
pip install torch transformers datasets pyarrow pandas tqdm huggingface_hub
```

You'll also need the `hf` CLI (from `huggingface_hub`) authenticated if the dataset requires it:

```bash
hf auth login
```

## 1. Download the raw data

```bash
bash download_script.bash
```

This pulls a ~86GB slice (`sample/100BT` groups 000-003, ~28.6B raw tokens — comfortable headroom over the 25B-token target) of `HuggingFaceFW/fineweb-edu` into `./fineweb_raw/`, rather than the full dataset. See the comments in [download_script.bash](download_script.bash) for the disk-space math behind that choice. Check available space before running it either way — the packed token shard alone is ~50GB and needs to coexist on disk with the parquet download during packing (see step 2).

## 2. Tokenize and pack into a binary shard

`train.py` reads pretokenized data from flat `.bin` files (uint16 token ids, `EleutherAI/gpt-neox-20b` tokenizer), not from parquet directly. Pack the downloaded data first:

```bash
python prepare_data_local.py
```

This tokenizes every parquet file under `./fineweb_raw/`, appends an EOS token after each document, and writes a fixed-size memory-mapped file (`train_25b_packed.bin`, 25B tokens × 2 bytes ≈ 50GB) to the repo root.

For faster tokenization on a multi-core machine, use the parallelized variant instead:

```bash
python prepare_data_multiprocess.py
```

If you'd rather skip the local parquet download entirely and stream straight from the Hub, use:

```bash
python prepare_data.py
```

Once packing is done, move the `.bin` file(s) into their own directory — `PretrainBinaryDataset` reads every `*.bin` file in whatever `--data_dir` you point it at:

```bash
mkdir -p packed_data
mv train_25b_packed.bin packed_data/
```

## 3. Test locally before committing to a full run

A handful of tiny sanity checks, cheapest first:

```bash
# Model forward pass + param count + initial-loss sanity check
python model.py

# Dataset chunking sanity check (auto-creates ./dummy_data/test.bin if missing)
python dataset.py

# Full training-loop smoke test — a few steps on dummy data, runs on
# CUDA / MPS / CPU automatically depending on what's available
python train.py \
  --data_dir ./dummy_data \
  --out_dir ./checkpoints_local \
  --batch_size 1 --grad_accum_steps 1 --seq_len 128 \
  --max_steps 20 --warmup_steps 5
```

This is the fastest way to catch a broken config, an OOM, or a data-pipeline bug before spending GPU-cluster time on it. Swap `--data_dir` to point at your real `packed_data/` directory (with a small `--max_steps`) to sanity-check against real data too.

## 4. Run the full pretraining job on a GPU cluster

`train.py` auto-detects DDP from the environment (`RANK` / `LOCAL_RANK` / `WORLD_SIZE`), so launch it with `torchrun`. Single node, 8 GPUs:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --data_dir ./packed_data \
  --out_dir ./checkpoints \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --seq_len 2048 \
  --max_steps 97000 \
  --warmup_steps 1000 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --compile
```

`--max_steps 97000` comes from the design-doc target of 25B tokens (see [design.md](design.md#L104-L108)); recompute it for your own settings as:

```
max_steps = TOTAL_TOKENS / (batch_size * seq_len * grad_accum_steps * world_size)
```

For multi-node runs, add the standard `torchrun` multi-node flags (`--nnodes`, `--node_rank`, `--master_addr`, `--master_port`) — `train.py` itself needs no changes.

Full CLI options:

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | `./dummy_data` | Directory of packed `.bin` shards |
| `--out_dir` | `./checkpoints_local` | Where checkpoints are written locally |
| `--batch_size` | `2` | Micro batch size per GPU |
| `--grad_accum_steps` | `8` | Gradient accumulation steps |
| `--seq_len` | `2048` | Training sequence length |
| `--max_steps` | `1000` | Total optimizer steps |
| `--warmup_steps` | `100` | LR warmup steps |
| `--max_lr` / `--min_lr` | `3e-4` / `3e-5` | Cosine LR schedule bounds |
| `--weight_decay` | `0.1` | AdamW weight decay (only applied to ≥2D weights, not RMSNorm scales) |
| `--save_interval` | `500` | Save `model_latest.pt` every N steps, in addition to the final checkpoint |
| `--resume_from` | `None` | Path to a checkpoint to resume model + optimizer + step from |
| `--auto_resume` | off | If `--resume_from` isn't given, look for `out_dir/model_latest.pt` locally, then `gs://<gcs_bucket>/model_latest.pt` if not found locally; starts fresh if neither exists |
| `--gcs_bucket` | `None` | If set (e.g. `gs://bucket/checkpoints`), upload each checkpoint there via `gcloud storage cp` right after it saves locally |
| `--compile` | off | Enable `torch.compile` |
| `--wandb` | off | Log metrics to Weights & Biases |
| `--wandb_project` | `1bmodel-pretrain` | W&B project name |
| `--wandb_run_name` | `run-YYYY-MM-DD-HH-MM-SS` | W&B run name (auto-timestamped if not set) |
| `--wandb_log_interval` | `10` | Log to W&B every N steps |

## Checkpoints

Two checkpoints are written to `--out_dir`, each asynchronously in a background thread so disk I/O never blocks training:
- `model_latest.pt` — overwritten every `--save_interval` steps (rolling, not versioned)
- `model_final.pt` — written once, after `--max_steps` completes

Both include `model_state_dict`, `optimizer_state_dict` (AdamW momentum/variance — needed to resume without losing training stability), `config`, and `step`. A failed save raises loudly (`RuntimeError`) instead of silently continuing, so a corrupted or incomplete write is never mistaken for success.

**Resuming**: pass `--resume_from path/to/checkpoint.pt` to continue from an exact file, or `--auto_resume` to have it figure out the path itself (see table above) — the latter is what makes unattended recovery after a preemption possible (see below).

**Backing up to GCS**: pass `--gcs_bucket gs://your-bucket/checkpoints` and every save also gets pushed there under the same filename. A failed upload only logs a warning — it never fails the run, since the local copy is already safe.

## 5. Running on Google Cloud (GCP)

This section documents the full VM setup this pipeline was actually validated against, including the non-obvious gotchas hit along the way — worth reading before spinning up a real run rather than rediscovering them.

### Create the VM

Use a Deep Learning VM image so CUDA/the NVIDIA driver come pre-installed — check the current image family first (names change):

```bash
gcloud compute images list --project=deeplearning-platform-release --no-standard-images
```

Then create the instance (example: single L4 for smoke-testing; see the H100/DDP note below for the real run):

```bash
gcloud compute instances create my-train-vm \
  --zone=us-west1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

Notes:
- Instance names can't start with a digit (GCP naming rule).
- If a zone reports `ZONE_RESOURCE_POOL_EXHAUSTED` for your chosen GPU, it's transient capacity, not a config error — try another zone.
- For H100s specifically, GCP's `a3` machine family comes in `a3-highgpu-1g`/`2g`/`4g`/`8g` (GPU count baked into the machine type name, no separate `--accelerator` flag needed) — pick the size matching your actual quota (`gcloud compute regions describe <region> --format=json` to check, though newer quota like H100 sometimes only shows up in the Console's Quotas page, not this command).
- For a **preemptible** run, add `--preemptible` (check the Console's Quotas page for "Preemptible NVIDIA \<GPU\> GPUs" — this is a separate quota bucket from on-demand and from "Spot").

### Add swap — do this every time, including after a stop/start

GCE VMs have **0 swap by default**. Checkpoint saving briefly needs ~3x model size in CPU memory (model + optimizer state, cloned for the async write); without swap, a memory spike here gets hard-killed by the OOM killer instead of just slowing down. This bit us twice in testing.

```bash
sudo fallocate -l 20G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**The `/etc/fstab` line matters** — without it, swap silently disappears every time the VM stops and starts again (e.g. after a preemption, or after changing its service account scopes), and you'll hit the exact same OOM again on the next checkpoint save with no obvious cause.

### Grant GCS access — two separate permission systems, both required

If you'll use `--gcs_bucket`/`--auto_resume`, the VM needs *both* of these (missing either one fails uploads, with different, easily-confused error messages):

**1. IAM — who can touch the bucket.** A fresh VM's default compute service account is usually not implicitly granted access:
```bash
gcloud storage buckets add-iam-policy-binding gs://your-bucket \
  --member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

**2. OAuth scopes — what the VM's own credentials are allowed to request.** This is separate from IAM and easy to miss: a VM created without an explicit `--scopes` flag often only gets `devstorage.read_only`, which blocks writes *even with correct IAM* (error: `Provided scope(s) are not authorized`). Fixing this requires stopping the VM:
```bash
gcloud compute instances stop my-train-vm --zone=us-west1-a
gcloud compute instances set-service-account my-train-vm --zone=us-west1-a \
  --service-account=<PROJECT_NUMBER>-compute@developer.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write
gcloud compute instances start my-train-vm --zone=us-west1-a
```

If uploads still fail with a scope error right after this, `gcloud`'s locally cached credentials on the VM may be stale (they persist on the boot disk across a stop/start). Force a fresh token:
```bash
rm -f ~/.config/gcloud/credentials.db ~/.config/gcloud/access_tokens.db
```

### Set up the code and data

```bash
git clone https://github.com/<you>/1bmodel.git   # public HTTPS clone needs no auth on a fresh VM
cd 1bmodel
pip install torch transformers pyarrow pandas tqdm huggingface_hub wandb

# .env (WANDB_API_KEY) is gitignored, so it doesn't come with git clone — copy it separately:
#   gcloud compute scp .env my-train-vm:~/1bmodel/.env --zone=us-west1-a
source .env   # if using --wandb

# Pull the packed dataset from GCS (same-cloud transfer, much faster than the original upload):
mkdir -p packed_data
gcloud storage cp gs://your-bucket/train_25b_packed.bin packed_data/train_25b_packed.bin
```

### Smoke test before the real run

Cheapest first, on whatever single GPU you provisioned — this exercises the same training/checkpoint/resume code paths as the real run at negligible cost:

```bash
# dummy_data isn't in git (gitignored) — generate it:
python3 dataset.py

# Basic run: env, CUDA, training loop, checkpoint save
python3 train.py --data_dir ./dummy_data --out_dir ./ckpt_test \
  --batch_size 1 --grad_accum_steps 1 --seq_len 128 \
  --max_steps 5 --warmup_steps 1 --save_interval 2

# Resume: loads model + optimizer + step from the checkpoint above
python3 train.py --data_dir ./dummy_data --out_dir ./ckpt_test2 \
  --resume_from ./ckpt_test/model_latest.pt \
  --batch_size 1 --grad_accum_steps 1 --seq_len 128 \
  --max_steps 5 --warmup_steps 1 --save_interval 2

# auto_resume + GCS round trip: uploads, then (after deleting the local copy)
# downloads and resumes automatically — the actual preemption-recovery path
python3 train.py --data_dir ./dummy_data --out_dir ./ckpt_auto \
  --batch_size 1 --grad_accum_steps 1 --seq_len 128 \
  --max_steps 4 --warmup_steps 1 --save_interval 2 \
  --auto_resume --gcs_bucket gs://your-bucket/checkpoints
rm -rf ./ckpt_auto
python3 train.py --data_dir ./dummy_data --out_dir ./ckpt_auto \
  --batch_size 1 --grad_accum_steps 1 --seq_len 128 \
  --max_steps 6 --warmup_steps 1 --save_interval 2 \
  --auto_resume --gcs_bucket gs://your-bucket/checkpoints
```

Pass criteria: `DEVICE: CUDA` (not CPU/MPS) printed at startup, `Initial Loss CHECK PASSED`, every checkpoint line reads `✅ Saved`/`☁️ Uploaded` (never `❌ FAILED`), and the second `auto_resume` run prints `📂 Resuming from checkpoint` continuing from the right step — all with exit code 0.

Delete test checkpoints (`rm -rf ckpt_test ckpt_test2 ckpt_auto`) before moving to the real run — at ~13-14GB each (model + optimizer state, fp32), they add up fast against a VM's boot disk.

### Preemptible/Spot: what actually happens on preemption

GCP stops (not deletes) a preempted VM — the boot disk and its local checkpoints survive. But nothing restarts it or re-launches training automatically. Without further setup, recovering means: notice it stopped → `gcloud compute instances start` it yourself → SSH in → re-run `train.py --auto_resume ...` yourself. `--auto_resume` makes that last step safe and mindless (no need to track which checkpoint or where), but doesn't eliminate the manual restart.

For fully hands-off recovery, the standard GCP pattern is a **Managed Instance Group** with target size 1, built from an instance template whose `startup-script` runs the training command (with `--auto_resume --gcs_bucket ...`) on every boot. When preempted, the MIG detects it and creates a *replacement* VM from the template — a fresh disk with no local checkpoint, which is exactly the case `--auto_resume`'s GCS fallback exists for. This isn't set up in this repo yet; it's the next piece needed before an unattended multi-day preemptible run.
