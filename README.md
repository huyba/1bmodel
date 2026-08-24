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

This pulls all `*.parquet` shards of `HuggingFaceFW/fineweb-edu` into `./fineweb_raw/`. It's a large download — check available disk space first (the full packed token shard alone is ~50GB; see step 2).

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
| `--out_dir` | `./checkpoints_local` | Where checkpoints are written |
| `--batch_size` | `2` | Micro batch size per GPU |
| `--grad_accum_steps` | `8` | Gradient accumulation steps |
| `--seq_len` | `2048` | Training sequence length |
| `--max_steps` | `1000` | Total optimizer steps |
| `--warmup_steps` | `100` | LR warmup steps |
| `--max_lr` / `--min_lr` | `3e-4` / `3e-5` | Cosine LR schedule bounds |
| `--compile` | off | Enable `torch.compile` |

## Checkpoints

A single checkpoint (`model_final.pt`: model weights + config + step) is written to `--out_dir` asynchronously in a background thread once training completes, so the final disk write doesn't block the last step. There's currently no periodic mid-run checkpointing and no optimizer state is saved, so a run that crashes or is preempted partway through loses all progress — factor that into how long a single `--max_steps` run you're willing to risk.
