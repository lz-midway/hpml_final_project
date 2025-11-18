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

import os
import wandb
import logging
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# Wandb Login key
os.environ["WANDB_API_KEY"] = "d594f859224e08959ccfb537de51d8761c5c289f"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="full", choices=["full", "binary", "fp8"], help="full = FP32 baseline, binary = 1-bit model")
parser.add_argument("--n_embd", type=int, default=768, help="Model size: 512, 768, 1024, etc.")
parser.add_argument("--local_rank", type=int, default=-1, help="For DDP")
parser.add_argument("--profile", action="store_true", help="Enable profiling")
args = parser.parse_args()

# Distributed setup
if args.local_rank != -1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

# WandB (only rank 0)
if args.local_rank in [-1, 0]:
    wandb.init(project="1bit-llm-c4", name=f"{args.mode}_embd{args.n_embd}", config=vars(args))

# log.txt
logging.basicConfig(filename="log.txt", level=logging.INFO,
                    format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M")

# Select projection layer: either 100% full or 100% binary
if args.mode == "full":
    proj = binary_layers.LinearFull
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

gradient_accumulation_steps = 3 * 8
batch_size = 20 
block_size = 1024

model_config = TransformerConfig(
    n_layer     = 12,
    n_head      = 12,
    n_embd      = args.n_embd,           # ← now configurable
    dropout     = 0.0,
    vocab_size  = 50304,
    bias        = False,
    max_len     = block_size,
    
    mlp_proj    = proj,  
    qkv_proj    = proj,
    c_proj      = nn.Linear,
)

model = Transformer(model_config)
model.to(device)

if args.local_rank != -1:
    model = DDP(model, device_ids=[args.local_rank])

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

print("\n===== Training / Model Configuration =====")
pprint(run_config, sort_dicts=False)
print("==========================================\n")


print(f"{model.num_params()} parameters")




if compile:
    print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

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
micro_batch = 0
iter_start = time.time()

for i in range(10):
    print(model.generate("Hello, I am an LLM", dataloader.dataset.tok))
model.train()

prof = None
if args.profile and args.local_rank in [-1, 0]:
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
            if iter_num % log_interval == 0:
                lr_current = optimizer.param_groups[0]["lr"]
                train_loss = loss.item() * gradient_accumulation_steps
                print(f"iter {iter_num:>8d} | lr {lr_current:.3e} | "
                      f"loss {train_loss:.4f} | {(time.time() - iter_start)*1000:.2f}ms")
                
                iter_start = time.time()
                # WandB + file logging
                if args.local_rank in [-1, 0]:
                    wandb.log({"iter": iter_num, "train_loss": train_loss, "lr": lr_current})
                    logging.info(f"{iter_num} loss {train_loss:.4f} lr {lr_current:.2e}")
                    
            # ---------------- evaluation / checkpoint ----------------
            if iter_num % eval_interval == 0 or iter_num == max_iters:
                
                t0 = time.time()
                val_loss = estimate_loss(model, val_dataloader)
                print(f"\nstep {iter_num}: val loss {val_loss:.4f}, "
                      f"time {time.time()-t0:.1f}s\n")
                
                if args.local_rank in [-1, 0]:
                    wandb.log({"val_loss": val_loss, "iter": iter_num})
                    logging.info(f"{iter_num} val_loss {val_loss:.4f}")   

                for i in range(10):
                    print(model.generate("Hello, I am an LLM", dataloader.dataset.tok))
                    
                if val_loss < best_val_loss or always_save_checkpoint:
                    best_val_loss = min(best_val_loss, val_loss)
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter_num": iter_num,
                        "val_loss": val_loss,
                        "model_config": asdict(model.config),
                    }
                    torch.save(checkpoint, os.path.join(out_dir, f"ckpt_{iter_num}.pt"))
                    print("Checkpoint saved.")
    
            if iter_num >= max_iters:
                break

if prof is not None:
    prof.stop()

print("Training complete.")












    