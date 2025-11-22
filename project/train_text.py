import os
import time
import math
import torch
import torch.nn as nn
from pathlib import Path
from pprint import pprint
from dataclasses import asdict    
from contextlib import nullcontext
from transformer_engine.pytorch import fp8_autocast
import transformer_engine.pytorch as te

from models import Transformer, TransformerConfig
from data import text_dataset
import binary_layers

import wandb
import logging
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Wandb Login key
os.environ["WANDB_API_KEY"] = "d594f859224e08959ccfb537de51d8761c5c289f"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="full", choices=["full", "binary", "fp8"])
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--local_rank", type=int, default=-1, help="For DDP")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--resume_run", type=str, default=None, help="W&B run id or 'auto' to resume latest run")

args = parser.parse_args()

# If launched with torchrun, override local_rank from env
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

# Distributed setup
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
else:
    rank = 0
    
is_main_process = (rank == 0)

# Select projection layer: either 100% full or 100% binary
if args.mode == "full":
    proj = nn.Linear
elif args.mode == "fp8":
    proj = binary_layers.Linear_fp8
else:  # binary
    proj = binary_layers.Linear

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


learning_rate = 6e-4 # max learning rate
max_iters = 25000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True # whether to decay the learning rate
warmup_iters = 40
lr_decay_iters = 25000 
min_lr = 6e-5 

backend = 'nccl' 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
compile = True


torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


out_dir = 'out'
eval_interval = 1000
log_interval = 1
eval_iters = 200
always_save_checkpoint = True 

gradient_accumulation_steps = 3 * 4
batch_size = 20 
block_size = 1024

model_config = TransformerConfig(
    n_layer     = 12,
    n_head      = 12,
    n_embd      = args.n_embd,
    dropout     = 0.0,
    vocab_size  = 50304,
    bias        = False,
    max_len     = block_size,
    
    mlp_proj    = proj,  
    qkv_proj    = proj,
    c_proj      = nn.Linear,
)

model = Transformer(model_config).to(device)

if compile:
    if is_main_process:
        print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

if args.local_rank != -1:
    ddp_kwargs = {
        "device_ids": [args.local_rank],
    }
    # binary/fp8 modes may have params not used every step
    if args.mode != "full":
        ddp_kwargs["find_unused_parameters"] = True

    model = DDP(model, **ddp_kwargs)

    # Patch: forward missing attributes to underlying module
    class _DDPWrapper(torch.nn.parallel.DistributedDataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    model.__class__ = _DDPWrapper

run_config = {
    "learning_rate"               : learning_rate,
    "max_iters"                   : max_iters,
    "weight_decay"                : weight_decay,
    "beta1"                       : beta1,
    "beta2"                       : beta2,
    "grad_clip"                   : grad_clip,
    "decay_lr"                    : decay_lr,
    "warmup_iters"                : warmup_iters,
    "lr_decay_iters"              : lr_decay_iters,
    "min_lr"                      : min_lr,
    "backend"                     : backend,
    "device"                      : device,
    "dtype"                       : dtype,
    "compile"                     : compile,
    "gradient_accumulation_steps" : gradient_accumulation_steps,
    "batch_size"                  : batch_size,
    "block_size"                  : block_size,
    "out_dir"                     : out_dir,
    "eval_interval"               : eval_interval,
    "log_interval"                : log_interval,
    "eval_iters"                  : eval_iters,
    "always_save_checkpoint"      : always_save_checkpoint,
    # model architecture
    "model_config" : {
        "n_layer"   : model_config.n_layer,
        "n_head"    : model_config.n_head,
        "n_embd"    : model_config.n_embd,
        "dropout"   : model_config.dropout,
        "vocab_size": model_config.vocab_size,
        "bias"      : model_config.bias,
        "max_len"   : model_config.max_len,
        "mlp_proj"  : model_config.mlp_proj,
        "qkv_proj"  : model_config.qkv_proj,
        "c_proj"    : model_config.c_proj,
        "max_len"   : model_config.max_len,
    },
}
if is_main_process:
    print("\n===== Training / Model Configuration =====")
    pprint(run_config, sort_dicts=False)
    print("==========================================\n")
    underlying_model = model.module if hasattr(model, "module") else model
    print(f"{underlying_model.num_params()} parameters")

dataloader, val_dataloader = text_dataset.get_loader(
    batch_size   = batch_size, 
    max_len      = block_size, 
    num_workers  = 8,
    prefetch_factor = 16,
)
    
optimizer = torch.optim.AdamW(lr = learning_rate, weight_decay = weight_decay, betas=(beta1, beta2), params = model.parameters())
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

def loss_fn(logits, targets):
    # logits: [B, T, vocab], targets: [B, T]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
    )


# ────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation loop
# ────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def estimate_loss(model, dataloader, iters=eval_iters):
    model.eval()
    losses = []
    for i, (idx, labels) in enumerate(dataloader):
        if i >= iters:
            break
        idx = idx.to(device)
        labels = labels.to(device)
        with fp8_autocast():
            with ctx:
                logits = model(idx)  # (B, T, vocab)
                loss = loss_fn(logits, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)



# ────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ────────────────────────────────────────────────────────────────────────────


if not os.path.exists(out_dir):
    os.makedirs(out_dir)
model.train()


iter_num = 0
best_val_loss = 1e9

latest_iter = -1
resume_path = None

if os.path.isdir(out_dir):
    for fname in os.listdir(out_dir):
        if fname.startswith("ckpt_") and fname.endswith(".pt"):
            try:
                it = int(fname.split("_")[1].split(".")[0])
                if it > latest_iter:
                    latest_iter = it
                    resume_path = os.path.join(out_dir, fname)
            except ValueError:
                pass

if resume_path is not None:
    if is_main_process:
        print(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["val_loss"]

# WandB (only rank 0)
if is_main_process:
    logging.basicConfig(
        filename="log.txt",
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%m-%d %H:%M"
    )

    if args.resume_run is not None:
        # AUTO-DETECT MOST RECENT RUN
        if args.resume_run == "auto":
            api = wandb.Api()
            runs = api.runs("1bit-llm-c4")
            if len(runs) == 0:
                raise RuntimeError("No prior wandb runs found for auto-resume.")
            latest = sorted(runs, key=lambda r: r.created_at)[-1]
            resume_id = latest.id
            print(f"[W&B] Auto-resuming latest run: {resume_id}")
        else:
            resume_id = args.resume_run
            print(f"[W&B] Resuming provided run: {resume_id}")

        wandb.init(
            project="1bit-llm-c4",
            id=resume_id,
            name=f"{args.mode}_embd{args.n_embd}",
            config=vars(args),
            resume="allow",
        )
    else:
        # Start a NEW run
        wandb.init(
            project="1bit-llm-c4",
            name=f"{args.mode}_embd{args.n_embd}",
            config=vars(args)
        )

    # os.environ["WANDB_MODE"] = "disabled"

micro_batch = 0
iter_start = time.time()

if is_main_process:
    with torch.no_grad():
        print(model.generate("Hello, I am an LLM", dataloader.dataset.tok))

model.train()

prof = None
if args.profile and is_main_process:
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    prof.start()

while iter_num < max_iters:
    # fetch a batch
    for idx, labels in dataloader:
        
        idx = idx.to(device)  # shape [B, T]
        labels = labels.to(device)  # shape [B, T]
        
        # adjust learning-rate
        if decay_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = get_lr(iter_num)

        # ---------------- forward ----------------
        with fp8_autocast():
            with ctx:
                logits = model(idx)  # (B, T, vocab)
                loss = loss_fn(logits, labels) / gradient_accumulation_steps

        # ---------------- backward ----------------
        scaler.scale(loss).backward()
        micro_batch += 1
        
        # gradient accumulation
        if micro_batch % gradient_accumulation_steps == 0:
            # gradient clipping
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            micro_batch = 0

            model.update_cache()
            
            if prof is not None:
                prof.step()
        # ---------------- logging ----------------
            if iter_num % log_interval == 0 and is_main_process:
                lr_current = optimizer.param_groups[0]["lr"]
                train_loss = loss.item() * gradient_accumulation_steps
                print(f"iter {iter_num:>8d} | lr {lr_current:.3e} | "
                      f"loss {train_loss:.4f} | {(time.time() - iter_start)*1000:.2f}ms")
                
                iter_start = time.time()
                # WandB + file logging
                wandb.log({"iter": iter_num, "train_loss": train_loss, "lr": lr_current})
                logging.info(f"{iter_num} loss {train_loss:.4f} lr {lr_current:.2e}")
                    
            # ---------------- evaluation / checkpoint ----------------
            if (iter_num % eval_interval == 0 or iter_num == max_iters) and is_main_process:
                
                t0 = time.time()
                val_loss = estimate_loss(model, val_dataloader)
                print(f"\nstep {iter_num}: val loss {val_loss:.4f}, "
                      f"time {time.time()-t0:.1f}s\n")
                
                wandb.log({"val_loss": val_loss, "iter": iter_num})
                logging.info(f"{iter_num} val_loss {val_loss:.4f}")   
                
                torch.cuda.synchronize()
                gen_model = model.module if hasattr(model, "module") else model
                with torch.no_grad():
                    print(gen_model.generate("Hello, I am an LLM", dataloader.dataset.tok))
                   
                if val_loss < best_val_loss or always_save_checkpoint:
                    best_val_loss = min(best_val_loss, val_loss)

                    ckpt_name = f"ckpt_{iter_num}.pt"
                    ckpt_path = os.path.join(out_dir, ckpt_name)

                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "val_loss": val_loss,
                        "model_config": asdict(model.config),
                    }
                    torch.save(checkpoint, ckpt_path)
                    print("Checkpoint saved:", ckpt_path)
                    if iter_num % 5000 == 0 and iter_num > 0:
                        extra_ckpt_name = f"extra_ckpt_{iter_num}.pt"
                        extra_ckpt_path = os.path.join(out_dir, extra_ckpt_name)
                        torch.save(checkpoint, extra_ckpt_path)
                        print("Saved 5000-iter checkpoint:", extra_ckpt_path)
                    # delete all other checkpoints
                    for fname in os.listdir(out_dir):
                        if (
                            fname.startswith("ckpt_")
                            and fname.endswith(".pt")
                            and fname != ckpt_name
                        ):
                            try:
                                os.remove(os.path.join(out_dir, fname))
                                print("Deleted old ckpt:", fname)
                            except Exception as e:
                                print("Could not delete", fname, "reason:", e)

            if iter_num >= max_iters:
                break

if is_main_process:
    for i in range(10): 
        print(model.generate("Hello, I am an LLM", dataloader.dataset.tok))

if prof is not None:
    prof.stop()

if is_main_process:
    wandb.save("log.txt")

if is_main_process:
    print("Training complete.")