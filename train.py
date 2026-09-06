import os
import math
import time
import argparse
import threading
import subprocess
from dataclasses import asdict

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import Transformer1B, ModelConfig
from dataset import PretrainBinaryDataset


def _async_save_worker(checkpoint_data, save_path, result, gcs_dest=None):
    try:
        torch.save(checkpoint_data, save_path)
        result['ok'] = True
        print(f"\n✅ [Async Checkpoint] Saved to: {save_path}\n")
    except Exception as e:
        result['ok'] = False
        result['error'] = e
        print(f"\n❌ [Async Checkpoint] FAILED to save {save_path}: {e}\n")
        return  # local save failed — nothing valid to upload

    if gcs_dest:
        # Runs in this same background thread so the upload doesn't block training
        # either. A failed upload does NOT fail the checkpoint overall (result['ok']
        # stays True) — the local file is already safe; this is a best-effort backup.
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", save_path, gcs_dest],
                check=True, capture_output=True, text=True,
            )
            print(f"\n☁️  [GCS Upload] Uploaded to: {gcs_dest}\n")
        except Exception as e:
            detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            print(f"\n⚠️  [GCS Upload] FAILED to upload {save_path} to {gcs_dest}: {detail}\n")


def _to_cpu_clone(obj):
    """Recursively move tensors to CPU inside (possibly nested) dicts/lists —
    optimizer state_dicts nest tensors under {'state': {idx: {...}}, 'param_groups': [...]}.
    .cpu() on a non-CPU tensor already allocates an independent copy (crossing a
    device boundary always copies), so only actually-on-CPU tensors need an
    explicit .clone() to decouple them from the live training tensors. Skipping
    the redundant double-copy roughly halves peak memory during checkpointing —
    which matters: with optimizer state included this can be ~3x model size."""
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.device.type == 'cpu' else obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_clone(v) for v in obj]
    return obj


def async_save_checkpoint(checkpoint_data, save_path, gcs_dest=None):
    checkpoint_data['model_state_dict'] = _to_cpu_clone(checkpoint_data['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint_data:
        checkpoint_data['optimizer_state_dict'] = _to_cpu_clone(checkpoint_data['optimizer_state_dict'])
    result = {}
    thread = threading.Thread(target=_async_save_worker, args=(checkpoint_data, save_path, result, gcs_dest))
    thread.result = result
    thread.start()
    return thread


def _check_save_ok(thread, label):
    """Join a checkpoint-save thread and raise loudly if the write failed
    (Thread.join() alone swallows exceptions raised inside the thread)."""
    thread.join()
    if not thread.result.get('ok'):
        raise RuntimeError(f"{label} checkpoint failed to save: {thread.result.get('error')}")


def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dummy_data")
    parser.add_argument("--out_dir", type=str, default="./checkpoints_local")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=500, help="Save a checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None,
                         help="Path to a checkpoint to resume model + optimizer + step from")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                         help="If set (e.g. gs://bucket/checkpoints), upload each checkpoint there "
                              "via `gcloud storage cp` right after it saves locally")
    parser.add_argument("--auto_resume", action="store_true",
                         help="If --resume_from isn't given, look for out_dir/model_latest.pt "
                              "locally first, then gs://<gcs_bucket>/model_latest.pt if not found "
                              "locally, and resume from whichever turns up (starts fresh if neither "
                              "exists — meant for restarting on a fresh VM after preemption)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="1bmodel-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                         help="Defaults to run-YYYY-MM-DD-HH-MM-SS (start time) if not set")
    parser.add_argument("--wandb_log_interval", type=int, default=10, help="Log to W&B every N steps")
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
        autocast_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = 'mps'
        autocast_dtype = torch.float32
    else:
        device = 'cpu'
        autocast_dtype = torch.float32

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (int(os.environ['RANK']) == 0)
        world_size = dist.get_world_size()
    else:
        master_process = True
        world_size = 1

    use_cuda = device.startswith('cuda')

    tokens_per_iter = args.batch_size * args.seq_len * args.grad_accum_steps * world_size

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        print("=" * 70)
        print(f"🖥️  DEVICE: {device.upper()} | WORLD SIZE: {world_size}")
        print(f"📦 GLOBAL BATCH SIZE: {tokens_per_iter:,} tokens/step")
        print("=" * 70)

    config = ModelConfig(
        vocab_size=50280,
        d_model=2048,
        n_layers=24,
        max_seq_len=8192,
    )

    model = Transformer1B(config).to(device)

    if master_process:
        n_params = sum(p.numel() for p in model.parameters())
        expected_loss = math.log(config.vocab_size)
        print(f"  - Model Params    : {n_params / 1e9:.2f}B")
        print(f"  - Micro Batch Size: {args.batch_size}")
        print(f"  - Grad Accum Steps: {args.grad_accum_steps}")
        print(f"  - Expected Loss   : {expected_loss:.4f}")
        print("-" * 70)

    if master_process and args.wandb:
        run_name = args.wandb_run_name or time.strftime("run-%Y-%m-%d-%H-%M-%S")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "learning_rate": args.max_lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "seq_len": args.seq_len,
                "max_steps": args.max_steps,
                "warmup_steps": args.warmup_steps,
                "world_size": world_size,
                "model_params": f"{n_params / 1e9:.2f}B",
            }
        )

    if ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    raw_model = model.module if ddp else model

    if args.compile and hasattr(torch, 'compile'):
        if master_process:
            print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)

    # Only apply weight decay to matmul-participating weights (ndim >= 2).
    # RMSNorm scale params (ndim == 1) are excluded to avoid decaying them toward zero.
    decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True if use_cuda else False
    )

    # 1. Pre-run Initial Loss Verification
    raw_model.eval()
    with torch.no_grad():
        x_dummy = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        y_dummy = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        ctx = torch.autocast(device_type='cuda', dtype=autocast_dtype) if use_cuda else torch.no_grad()
        with ctx:
            _, init_loss = raw_model(x_dummy, y_dummy)

        if master_process:
            print(f"🧪 Pre-run Loss Check: {init_loss.item():.4f} (Target: ~{expected_loss:.4f})")
            assert abs(init_loss.item() - expected_loss) < 1.5, "Initial Loss mismatch!"
            print("  -> Initial Loss CHECK PASSED!\n")

    # Resume from checkpoint (after the sanity check above, so it still validates a
    # fresh init) — restores model + optimizer state, and continues the step/LR
    # schedule from where it left off. Note: the data iterator itself restarts from
    # the beginning of the dataset rather than resuming an exact mid-epoch position.
    resume_path = args.resume_from
    if resume_path is None and args.auto_resume:
        local_latest = os.path.join(args.out_dir, "model_latest.pt")
        if os.path.exists(local_latest):
            resume_path = local_latest
            if master_process:
                print(f"🔎 --auto_resume: found local checkpoint at {resume_path}")
        elif args.gcs_bucket:
            gcs_latest = f"{args.gcs_bucket.rstrip('/')}/model_latest.pt"
            if master_process:
                print(f"🔎 --auto_resume: no local checkpoint, trying {gcs_latest} ...")
                try:
                    subprocess.run(["gcloud", "storage", "cp", gcs_latest, local_latest],
                                    check=True, capture_output=True, text=True)
                    print(f"   -> downloaded from GCS\n")
                except Exception as e:
                    detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
                    print(f"   -> nothing found on GCS either ({detail.strip()}); starting fresh\n")
            # One rank downloads; the rest wait so they don't race on the same file.
            if ddp:
                dist.barrier()
            if os.path.exists(local_latest):
                resume_path = local_latest

    start_step = 1
    if resume_path:
        if master_process:
            print(f"📂 Resuming from checkpoint: {resume_path}")
        # Load to CPU, not `device`: model/optimizer .load_state_dict() already cast
        # and move each tensor to the right device on their own. Loading straight to
        # GPU would leave the now-redundant loaded tensors (~model size) referenced
        # by resume_ckpt and stranded in VRAM for the rest of training, on top of
        # the model's own real footprint — enough to OOM a memory-constrained GPU.
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        raw_model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        start_step = resume_ckpt['step'] + 1
        if master_process:
            print(f"   -> Resumed at step {resume_ckpt['step']}, continuing from step {start_step}\n")
        del resume_ckpt
        if use_cuda:
            torch.cuda.empty_cache()

    # 2. Dataset & DataLoader Setup
    # PretrainBinaryDataset shards itself across ranks/workers inside __iter__,
    # so no sampler is used (and none would work: it's an IterableDataset).
    dataset = PretrainBinaryDataset(data_dir=args.data_dir, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=use_cuda
    )
    data_iter = iter(dataloader)
    current_epoch = 0

    model.train()
    if master_process:
        print("🚀 Starting training pipeline...")

    pending_save_thread = None

    for step in range(start_step, args.max_steps + 1):
        t0 = time.time()
        
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        # Gradient Accumulation Loop
        for micro_step in range(args.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                current_epoch += 1
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_last_micro_step = (micro_step == args.grad_accum_steps - 1)
            
            if ddp and not is_last_micro_step:
                ctx = model.no_sync()
            else:
                ctx = torch.enable_grad()

            with ctx:
                if use_cuda:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps
                loss_accum += loss.detach() * args.grad_accum_steps
                loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Average accumulated loss across steps
        loss_log = loss_accum.item() / args.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = tokens_per_iter / dt

        if master_process and (step % 1 == 0 or step == args.max_steps):
            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"Loss: {loss_log:.4f} | "
                f"LR: {lr:.2e} | "
                f"GradNorm: {grad_norm:.2f} | "
                f"Time: {dt*1000:.1f}ms | "
                f"Throughput: {tokens_per_sec:.0f} tok/s"
            )

        if master_process and args.wandb and step % args.wandb_log_interval == 0:
            wandb.log({
                "train/loss": loss_log,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "perf/tokens_per_sec": tokens_per_sec,
            }, step=step)

        if master_process and step % args.save_interval == 0:
            if pending_save_thread is not None:
                _check_save_ok(pending_save_thread, "Periodic")
            save_path = os.path.join(args.out_dir, "model_latest.pt")
            ckpt = {
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(config),
                'step': step,
            }
            print(f"\n💾 [Step {step}] Saving periodic checkpoint...")
            gcs_dest = f"{args.gcs_bucket.rstrip('/')}/model_latest.pt" if args.gcs_bucket else None
            pending_save_thread = async_save_checkpoint(ckpt, save_path, gcs_dest)

    # 3. Save Final Checkpoint
    if master_process:
        if pending_save_thread is not None:
            _check_save_ok(pending_save_thread, "Periodic")
        print("\n💾 Saving final checkpoint...")
        save_path = os.path.join(args.out_dir, "model_final.pt")
        ckpt = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(config),
            'step': args.max_steps,
        }
        gcs_dest = f"{args.gcs_bucket.rstrip('/')}/model_final.pt" if args.gcs_bucket else None
        save_thread = async_save_checkpoint(ckpt, save_path, gcs_dest)
        print("⏳ Waiting for final checkpoint to finish writing to disk...")
        _check_save_ok(save_thread, "Final")
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")

    if master_process and args.wandb:
        wandb.finish()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
