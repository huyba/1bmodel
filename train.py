import os
import math
import time
import argparse
import threading
import shutil
import subprocess
from dataclasses import asdict

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from model import Transformer1B, ModelConfig
from dataset import PretrainBinaryDataset


def remote_cp(src, dst):
    """Copy src -> dst where either side may be a gs:// or s3:// URL."""
    if src.startswith("s3://") or dst.startswith("s3://"):
        cmd = ["aws", "s3", "cp", src, dst]
    else:
        cmd = ["gcloud", "storage", "cp", src, dst]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _atomic_torch_save(checkpoint_data, save_path):
    """Write to a temp file, then rename over save_path, so save_path is never a
    half-written file — anything syncing/uploading the checkpoint directory in the
    background (SageMaker does) or a preemption mid-write can't leave a corrupt
    model_latest.pt behind. The temp file sits in the directory *above* save_path's
    so such a sync doesn't pick it up; if that's on a different filesystem the
    rename can't be atomic and falls back to a plain move."""
    out_dir = os.path.dirname(os.path.abspath(save_path))
    tmp_path = os.path.join(os.path.dirname(out_dir), f".{os.path.basename(save_path)}.partial")
    torch.save(checkpoint_data, tmp_path)
    try:
        os.replace(tmp_path, save_path)
    except OSError:
        shutil.move(tmp_path, save_path)


def _async_save_worker(checkpoint_data, save_path, result, remote_dest=None):
    try:
        _atomic_torch_save(checkpoint_data, save_path)
        result['ok'] = True
        print(f"\n✅ [Async Checkpoint] Saved to: {save_path}\n")
    except Exception as e:
        result['ok'] = False
        result['error'] = e
        print(f"\n❌ [Async Checkpoint] FAILED to save {save_path}: {e}\n")
        return  # local save failed — nothing valid to upload

    if remote_dest:
        # Runs in this same background thread so the upload doesn't block training
        # either. A failed upload does NOT fail the checkpoint overall (result['ok']
        # stays True) — the local file is already safe; this is a best-effort backup.
        try:
            remote_cp(save_path, remote_dest)
            print(f"\n☁️  [Remote Upload] Uploaded to: {remote_dest}\n")
        except Exception as e:
            detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            print(f"\n⚠️  [Remote Upload] FAILED to upload {save_path} to {remote_dest}: {detail}\n")


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


def async_save_checkpoint(checkpoint_data, save_path, remote_dest=None, already_copied=False):
    # already_copied: the state dicts were gathered into fresh CPU tensors (FSDP path),
    # so they're already decoupled from the live training tensors — skip the extra copy.
    if not already_copied:
        checkpoint_data['model_state_dict'] = _to_cpu_clone(checkpoint_data['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint_data:
            checkpoint_data['optimizer_state_dict'] = _to_cpu_clone(checkpoint_data['optimizer_state_dict'])
    result = {}
    thread = threading.Thread(target=_async_save_worker, args=(checkpoint_data, save_path, result, remote_dest))
    thread.result = result
    thread.start()
    return thread


_FULL_SD = StateDictOptions(full_state_dict=True, cpu_offload=True)


def gather_checkpoint(model, optimizer, config, step):
    """Collective — EVERY rank must call this. Gathers the sharded FSDP state into
    full (unsharded) CPU state dicts on rank 0 (other ranks get empty dicts), so the
    checkpoint on disk is a single portable file independent of world size. Optimizer
    state is keyed by parameter name."""
    return {
        'model_state_dict': get_model_state_dict(model, options=_FULL_SD),
        'optimizer_state_dict': get_optimizer_state_dict(model, optimizer, options=_FULL_SD),
        'config': asdict(config),
        'step': step,
    }


def _param_names_in_optimizer_order(model):
    """Parameter names in the order the optimizer's param groups were built:
    decay group (ndim >= 2) first, then no-decay group. Activation-checkpoint
    wrapper prefixes are stripped so names match the state_dict."""
    names = [(n.replace("_checkpoint_wrapped_module.", ""), p) for n, p in model.named_parameters() if p.requires_grad]
    return [n for n, p in names if p.dim() >= 2] + [n for n, p in names if p.dim() < 2]


def _optim_sd_for_resume(optim_sd, model, optimizer):
    """Checkpoints written by the old DDP script key optimizer state by integer
    parameter index; the FSDP save keys it by parameter name. Convert the old form
    so a checkpoint from either script can be resumed.

    Also backfills any param_group key the checkpoint doesn't have but the live
    optimizer does (e.g. 'decoupled_weight_decay', added to AdamW's param_groups in
    a later PyTorch release than whatever wrote the checkpoint) — DCP's optimizer
    load matches param_groups by key set, and a mismatch there raises a bare
    KeyError with no clue it's a version issue. Groups line up by index in both
    the checkpoint and the live optimizer (decay group first, then no-decay)."""
    is_old_format = any(isinstance(k, int) for k in optim_sd['state']) or \
        any(isinstance(i, int) for g in optim_sd['param_groups'] for i in g['params'])
    if is_old_format:
        names = _param_names_in_optimizer_order(model)
        state = {names[i]: v for i, v in optim_sd['state'].items()}
        param_groups = [{**g, 'params': [names[i] for i in g['params']]} for g in optim_sd['param_groups']]
    else:
        state = optim_sd['state']
        param_groups = [dict(g) for g in optim_sd['param_groups']]

    live_groups = optimizer.param_groups
    assert len(param_groups) == len(live_groups), \
        f"checkpoint has {len(param_groups)} optimizer param groups, live optimizer has {len(live_groups)}"
    for g, live_g in zip(param_groups, live_groups):
        for k, v in live_g.items():
            g.setdefault(k, v)

    return {'state': state, 'param_groups': param_groups}


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


# Peak bf16 dense (no sparsity) TFLOPS — matches how this model actually runs
# (autocast bf16, no structured sparsity). Extend as new GPUs get used.
_KNOWN_GPU_PEAK_BF16_TFLOPS = {
    "H100": 989.0,
    "A100": 312.0,
}


def detect_gpu_peak_tflops(override=None):
    if override is not None:
        return override
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for key, tflops in _KNOWN_GPU_PEAK_BF16_TFLOPS.items():
        if key in name:
            return tflops
    return None  # unknown GPU: MFU just won't be computed/logged


def compute_mfu(n_params, tokens_per_sec_per_gpu, gpu_peak_tflops):
    """MFU = achieved FLOPs/s ÷ peak FLOPs/s. Achieved FLOPs/s uses the
    standard 6N-per-token approximation (2N fwd + 4N bwd) from the
    Chinchilla/PaLM papers — ignores attention's own FLOPs, fine as an estimate.
    Pass per-GPU throughput: the peak is for a single GPU."""
    if gpu_peak_tflops is None:
        return None
    achieved_flops = 6 * n_params * tokens_per_sec_per_gpu
    return achieved_flops / (gpu_peak_tflops * 1e12)


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
    parser.add_argument("--remote_dir", "--gcs_bucket", dest="remote_dir", type=str, default=None,
                         help="If set (gs://bucket/path or s3://bucket/path), upload each checkpoint "
                              "there (via `gcloud storage cp` / `aws s3 cp`) right after it saves locally")
    parser.add_argument("--activation_checkpointing", action="store_true",
                         help="Recompute each transformer block's activations in backward instead of "
                              "storing them: ~30%% slower, but lets a larger --batch_size fit in 24GB")
    parser.add_argument("--defer_grad_sync", action="store_true",
                         help="FSDP: only reduce-scatter gradients on the last micro-step of each "
                              "accumulation window (much less communication, but holds unsharded "
                              "gradients in memory in between)")
    parser.add_argument("--stop_after_steps", type=int, default=None,
                         help="Preflight/testing: run only this many steps (counting from the resume "
                              "point) with the real --max_steps LR schedule, then exit WITHOUT writing "
                              "the final checkpoint")
    parser.add_argument("--seed", type=int, default=1337,
                         help="Model-init seed; must be identical on every rank under FSDP")
    parser.add_argument("--d_model", type=int, default=2048, help="Override only for smoke tests")
    parser.add_argument("--n_layers", type=int, default=24, help="Override only for smoke tests")
    parser.add_argument("--auto_resume", action="store_true",
                         help="If --resume_from isn't given, look for out_dir/model_latest.pt "
                              "locally first, then <remote_dir>/model_latest.pt if not found "
                              "locally, and resume from whichever turns up (starts fresh if neither "
                              "exists — meant for restarting on a fresh VM after preemption)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="1bmodel-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                         help="Defaults to run-YYYY-MM-DD-HH-MM-SS (start time) if not set")
    parser.add_argument("--wandb_log_interval", type=int, default=10, help="Log to W&B every N steps")
    parser.add_argument("--gpu_peak_tflops", type=float, default=None,
                         help="Peak bf16 dense TFLOPS for MFU calc. Auto-detected for "
                              "H100/A100 from the GPU name if not set; MFU is skipped "
                              "if the GPU isn't recognized and this isn't provided.")
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

    distributed = int(os.environ.get('RANK', -1)) != -1
    if distributed:
        dist.init_process_group(backend='nccl' if device == 'cuda' else 'gloo')
        if device == 'cuda':
            device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
            torch.cuda.set_device(device)
        master_process = (int(os.environ['RANK']) == 0)
        world_size = dist.get_world_size()
    else:
        master_process = True
        world_size = 1

    use_cuda = device.startswith('cuda')
    gpu_peak_tflops = detect_gpu_peak_tflops(args.gpu_peak_tflops)

    tokens_per_iter = args.batch_size * args.seq_len * args.grad_accum_steps * world_size

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        print("=" * 70)
        print(f"🖥️  DEVICE: {device.upper()} | WORLD SIZE: {world_size}")
        print(f"📦 GLOBAL BATCH SIZE: {tokens_per_iter:,} tokens/step")
        if use_cuda:
            gpu_name = torch.cuda.get_device_name()
            tflops_str = f"{gpu_peak_tflops:.0f} TFLOPS bf16" if gpu_peak_tflops else "unknown, MFU disabled"
            print(f"🎮 GPU: {gpu_name} | Peak: {tflops_str}")
        print("=" * 70)

    config = ModelConfig(
        vocab_size=50280,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=8192,
    )

    # FSDP shards each rank's *own* locally-initialised weights (unlike DDP it does not
    # broadcast rank 0's), so every rank must initialise identically.
    torch.manual_seed(args.seed)
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

    if distributed:
        # FSDP2: each transformer block is its own all-gather unit; the root group holds
        # the (tied) embedding/output weight and the final norm. Params are gathered and
        # computed in bf16, gradients reduced in fp32; the fp32 master weights and Adam
        # state stay sharded across ranks.
        mp_policy = MixedPrecisionPolicy(param_dtype=autocast_dtype, reduce_dtype=torch.float32)
        mesh = init_device_mesh('cuda' if use_cuda else 'cpu', (world_size,))
        if args.activation_checkpointing:
            for i, layer in enumerate(model.layers):
                model.layers[i] = checkpoint_wrapper(layer, preserve_rng_state=False)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)
        if use_cuda:
            torch.cuda.empty_cache()  # release the transient full fp32 copy from init
    elif args.activation_checkpointing:
        for i, layer in enumerate(model.layers):
            model.layers[i] = checkpoint_wrapper(layer, preserve_rng_state=False)

    # fully_shard mutates the module in place (no wrapper), so this is the same object.
    raw_model = model

    if args.compile and hasattr(torch, 'compile'):
        if master_process:
            print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)

    # Only apply weight decay to matmul-participating weights (ndim >= 2).
    # RMSNorm scale params (ndim == 1) are excluded to avoid decaying them toward zero.
    # (_param_names_in_optimizer_order relies on this exact grouping/order.)
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

    if distributed:
        # A forward with no backward leaves the root FSDP group unsharded (only the
        # post-backward hook reshards it). Do it now: otherwise state_dict() hands back
        # plain tensors for the embedding/output/norm and loading a checkpoint fails,
        # and the gathered bf16 embedding would stay resident in VRAM.
        raw_model.reshard()

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
        elif args.remote_dir:
            remote_latest = f"{args.remote_dir.rstrip('/')}/model_latest.pt"
            if master_process:
                print(f"🔎 --auto_resume: no local checkpoint, trying {remote_latest} ...")
                try:
                    remote_cp(remote_latest, local_latest)
                    print(f"   -> downloaded from remote\n")
                except Exception as e:
                    detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
                    print(f"   -> nothing found remotely either ({detail.strip()}); starting fresh\n")
            # One rank downloads; the rest wait so they don't race on the same file.
            if distributed:
                dist.barrier()
            if os.path.exists(local_latest):
                resume_path = local_latest

    start_step = 1
    if resume_path:
        if master_process:
            print(f"📂 Resuming from checkpoint: {resume_path}")
        # Load to CPU, not `device`: the state-dict setters already move each tensor
        # (or, under FSDP, just this rank's shard of it) to the right device. Loading
        # straight to GPU would leave the redundant full tensors stranded in VRAM.
        # Every rank reads the full file; mmap keeps that as one shared page-cache copy
        # instead of world_size private ~14GB copies in host RAM.
        resume_ckpt = torch.load(resume_path, map_location='cpu', mmap=True, weights_only=False)
        load_opts = StateDictOptions(full_state_dict=True)
        set_model_state_dict(raw_model, resume_ckpt['model_state_dict'], options=load_opts)
        set_optimizer_state_dict(
            raw_model, optimizer,
            optim_state_dict=_optim_sd_for_resume(resume_ckpt['optimizer_state_dict'], raw_model, optimizer),
            options=load_opts,
        )
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

    last_step = args.max_steps
    if args.stop_after_steps is not None:
        last_step = min(args.max_steps, start_step + args.stop_after_steps - 1)
    stopped_early = last_step < args.max_steps

    for step in range(start_step, last_step + 1):
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
            
            if distributed and args.defer_grad_sync:
                model.set_requires_gradient_sync(is_last_micro_step)

            if use_cuda:
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)

            # Scale loss for gradient accumulation
            loss = loss / args.grad_accum_steps
            loss_accum += loss.detach() * args.grad_accum_steps
            loss.backward()

        if distributed:
            # SUM then divide (not ReduceOp.AVG): gloo, used for CPU smoke tests, lacks AVG.
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= world_size

        # Average accumulated loss across steps
        loss_log = loss_accum.item() / args.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if hasattr(grad_norm, 'full_tensor'):  # FSDP returns a DTensor
            grad_norm = grad_norm.full_tensor()
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = tokens_per_iter / dt
        # n_params is only assigned under `if master_process:` above — guard here too
        mfu = compute_mfu(n_params, tokens_per_sec / world_size, gpu_peak_tflops) if master_process else None

        if master_process and (step % 1 == 0 or step == args.max_steps):
            mfu_str = f" | MFU: {mfu*100:.1f}%" if mfu is not None else ""
            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"Loss: {loss_log:.4f} | "
                f"LR: {lr:.2e} | "
                f"GradNorm: {grad_norm:.2f} | "
                f"Time: {dt*1000:.1f}ms | "
                f"Throughput: {tokens_per_sec:.0f} tok/s"
                f"{mfu_str}"
            )

        if master_process and args.wandb and step % args.wandb_log_interval == 0:
            log_dict = {
                "train/loss": loss_log,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "perf/tokens_per_sec": tokens_per_sec,
            }
            if mfu is not None:
                log_dict["perf/mfu"] = mfu
            wandb.log(log_dict, step=step)

        if step % args.save_interval == 0:
            if master_process and pending_save_thread is not None:
                _check_save_ok(pending_save_thread, "Periodic")
            if master_process:
                print(f"\n💾 [Step {step}] Saving periodic checkpoint...")
            # Gathering the shards is collective, so all ranks take part; only rank 0
            # gets (and writes) the data.
            ckpt = gather_checkpoint(raw_model, optimizer, config, step)
            if master_process:
                save_path = os.path.join(args.out_dir, "model_latest.pt")
                remote_dest = f"{args.remote_dir.rstrip('/')}/model_latest.pt" if args.remote_dir else None
                pending_save_thread = async_save_checkpoint(ckpt, save_path, remote_dest, already_copied=True)

    # 3. Save Final Checkpoint
    if stopped_early:
        if master_process:
            print(f"\n🛑 --stop_after_steps: stopped at step {last_step}/{args.max_steps}; "
                  "no final checkpoint written.")
            if args.wandb:
                wandb.finish()
        if distributed:
            dist.destroy_process_group()
        return
    if master_process:
        if pending_save_thread is not None:
            _check_save_ok(pending_save_thread, "Periodic")
        print("\n💾 Saving final checkpoint...")
    ckpt = gather_checkpoint(raw_model, optimizer, config, args.max_steps)
    if master_process:
        save_path = os.path.join(args.out_dir, "model_final.pt")
        remote_dest = f"{args.remote_dir.rstrip('/')}/model_final.pt" if args.remote_dir else None
        save_thread = async_save_checkpoint(ckpt, save_path, remote_dest, already_copied=True)
        print("⏳ Waiting for final checkpoint to finish writing to disk...")
        _check_save_ok(save_thread, "Final")
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")

    if master_process and args.wandb:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
