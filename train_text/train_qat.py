"""
QAT Training Script - Train transformer with fake quantization for 25000 iterations
Uses pure nn.Linear architecture for comparing standard quantization vs binary layers.
Matches train_text.py hyperparameters exactly.
Usage: torchrun --nproc_per_node=N train_qat.py --bits 8
"""
import os, time, math, argparse, logging, sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import LayerNorm, functional as F
from dataclasses import dataclass, asdict
from contextlib import nullcontext

from data import text_dataset
from qat import prepare_qat, convert_to_int

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def fp8_autocast():
    return nullcontext()

import wandb

os.environ["WANDB_API_KEY"] = "d594f859224e08959ccfb537de51d8761c5c289f"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Model (pure nn.Linear for QAT) ----------
@dataclass
class TransformerConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    vocab_size: int = 50304
    dropout: float = 0.0
    max_len: int = 1024
    bias: bool = False

def sdpa_mixed_dtype(q, k, v, **kw):
    q16, k16, v16 = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
    out = F.scaled_dot_product_attention(q16, k16, v16, **kw)
    return out.to(q.dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = sdpa_mixed_dtype(q, k, v, attn_mask=None,
                            dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embd = nn.Embedding(config.max_len, config.n_embd)
        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.head.weight = self.tok_embd.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.01 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.tok_embd(idx) + self.pos_embd(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# ---------- End Model ----------

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--from_pretrained", action="store_true", help="Start from non_binary.pt instead of scratch")
args = parser.parse_args()
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

# DDP setup
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
else:
    rank = 0
is_main = (rank == 0)

# ===== Hyperparameters (exactly matching train_text.py) =====
learning_rate = 6e-4
max_iters = 25000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 40
lr_decay_iters = 25000
min_lr = 6e-5

gradient_accumulation_steps = 3 * 8  # = 24
batch_size = 20
block_size = 1024

eval_interval = 1000
log_interval = 1
eval_iters = 200

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

out_dir = f'out/qat_int{args.bits}'
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Model
model_cfg = TransformerConfig(
    n_layer=12, n_head=12, n_embd=768,
    dropout=0.0, vocab_size=50304, bias=False, max_len=block_size,
)
model = Transformer(model_cfg).to(device)

if is_main:
    print(f"\n{model.num_params():,} parameters")

# Load pretrained weights (only if --from_pretrained)
ckpt_path = '/workspace/non_binary.pt'
if args.from_pretrained and os.path.exists(ckpt_path):
    if is_main:
        print(f"Loading pretrained: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k.replace('module._orig_mod.', '').replace('_orig_mod.', ''): v
          for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
else:
    if is_main:
        print("Training from scratch")

# Apply QAT
model = prepare_qat(model, bits=args.bits, per_channel=True)
if is_main:
    print(f"QAT prepared with INT{args.bits}")

# Compile & DDP
model = torch.compile(model)
if args.local_rank != -1:
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

# Data (exactly matching train_text.py)
dataloader, val_dataloader = text_dataset.get_loader(
    batch_size=batch_size,
    max_len=block_size,
    num_workers=8,
    prefetch_factor=16,
)

# Optimizer (exactly matching train_text.py)
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(beta1, beta2),
)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

def loss_fn(logits, targets):
    return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

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
                logits = model(idx)
                loss = loss_fn(logits, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# WandB (only rank 0)
if is_main:
    wandb.init(project="1bit-llm-c4", name=f"qat_int{args.bits}")
    logging.basicConfig(
        filename=f"{out_dir}/log.txt",
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%m-%d %H:%M"
    )

# Training loop
iter_num = 0
micro_batch = 0
best_val_loss = 1e9
dataloader_iter = iter(dataloader)
iter_start = time.time()

if is_main:
    print(f"\nStarting QAT training for {max_iters} iterations")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}\n")

model.train()

while iter_num < max_iters:
    try:
        idx, labels = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        idx, labels = next(dataloader_iter)

    idx = idx.to(device)
    labels = labels.to(device)

    # LR schedule
    if decay_lr:
        lr_value = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_value

    # Forward
    with fp8_autocast():
        with ctx:
            logits = model(idx)
            loss = loss_fn(logits, labels) / gradient_accumulation_steps

    # Backward
    scaler.scale(loss).backward()
    micro_batch += 1

    if micro_batch % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1
        micro_batch = 0

        # Logging
        if iter_num % log_interval == 0 and is_main:
            lr_current = optimizer.param_groups[0]["lr"]
            train_loss = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num:>8d} | lr {lr_current:.3e} | "
                  f"loss {train_loss:.4f} | {(time.time() - iter_start)*1000:.2f}ms")
            iter_start = time.time()
            wandb.log({"iter": iter_num, "train_loss": train_loss, "lr": lr_current})
            logging.info(f"{iter_num} loss {train_loss:.4f} lr {lr_current:.2e}")

        # Checkpoint / eval
        if (iter_num % eval_interval == 0 or iter_num == max_iters) and is_main:
            t0 = time.time()
            val_loss = estimate_loss(model, val_dataloader)
            print(f"\nstep {iter_num}: val loss {val_loss:.4f}, time {time.time()-t0:.1f}s\n")
            wandb.log({"val_loss": val_loss, "iter": iter_num})
            logging.info(f"{iter_num} val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Save checkpoint
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "val_loss": val_loss,
                "qat_bits": args.bits,
                "model_config": asdict(model_cfg),
            }
            ckpt_name = f"ckpt_{iter_num}.pt"
            torch.save(ckpt, os.path.join(out_dir, ckpt_name))
            print(f"Checkpoint saved: {out_dir}/{ckpt_name}")

            # Delete older checkpoints (keep only latest)
            for fname in os.listdir(out_dir):
                if fname.startswith("ckpt_") and fname.endswith(".pt") and fname != ckpt_name:
                    try:
                        os.remove(os.path.join(out_dir, fname))
                    except:
                        pass

# Final save & convert
if is_main:
    print("\nTraining complete! Converting to quantized weights...")
    torch.save({"model": model.state_dict(), "qat_bits": args.bits}, f"{out_dir}/qat_final.pt")

    # Convert to quantized and save
    raw_model = model.module._orig_mod if hasattr(model, 'module') else (
        model._orig_mod if hasattr(model, '_orig_mod') else model)
    q_model = convert_to_int(raw_model)
    torch.save(q_model.state_dict(), f"{out_dir}/quantized_int{args.bits}.pt")
    print(f"Saved: {out_dir}/quantized_int{args.bits}.pt")
    wandb.finish()

print("Done.")
