import os
import math
import time
import argparse
import threading
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import Transformer1B, ModelConfig
from dataset import PretrainBinaryDataset


def _async_save_worker(checkpoint_data, save_path):
    torch.save(checkpoint_data, save_path)
    print(f"\n✅ [Async Checkpoint] Saved to: {save_path}\n")


def async_save_checkpoint(checkpoint_data, save_path):
    cpu_state_dict = {
        k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in checkpoint_data['model_state_dict'].items()
    }
    checkpoint_data['model_state_dict'] = cpu_state_dict
    thread = threading.Thread(target=_async_save_worker, args=(checkpoint_data, save_path))
    thread.start()


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
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
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

    if ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    raw_model = model.module if ddp else model

    if args.compile and hasattr(torch, 'compile'):
        if master_process:
            print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
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

    for step in range(1, args.max_steps + 1):
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

    # 3. Save Checkpoint
    if master_process:
        print("\n💾 Saving Checkpoint...")
        save_path = os.path.join(args.out_dir, "model_final.pt")
        ckpt = {'model_state_dict': raw_model.state_dict(), 'config': asdict(config), 'step': args.max_steps}
        async_save_checkpoint(ckpt, save_path)
        time.sleep(1)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
