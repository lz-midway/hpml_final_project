# prune_and_finetune_food101_full.py
import os
import time
import argparse
import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

import wandb

import data.image_dataset
import models.Models

# pruning utilities
from pruning_utils import (
    collect_structured_prunable_params,
    structured_prune_model,
    make_pruning_permanent,
    model_sparsity_report,
)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    filename="prune_finetune.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%m-%d %H:%M"
)

torch.manual_seed(42)

os.environ["WANDB_API_KEY"] = "0f853b7aa9e6bd44416474b253642758cf20704f"
os.environ["PYTHONWARNINGS"] = "ignore"

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Prune + Finetune Food101 (DDP-capable)")
parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--prune_amount", type=float, default=0.3, help="Structured prune fraction (per-layer)")
parser.add_argument("--finetune_epochs", type=int, default=10, help="Finetune epochs after pruning")
parser.add_argument("--make_prune_permanent", action="store_true", help="Remove pruning reparam (keeps zeros)")
parser.add_argument("--compile", action="store_true", help="torch.compile the model (optional)")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--filter_dimension", type=int, default=5)
parser.add_argument("--num_classes", type=int, default=101)
args = parser.parse_args()

if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

use_ddp = args.local_rank != -1
if use_ddp:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
else:
    rank = 0
    world_size = 1

is_main_process = (rank == 0)
device = torch.device(f"cuda:{args.local_rank}" if use_ddp else "cuda" if torch.cuda.is_available() else "cpu")

# AMP dtype / GradScaler logic
if device.type == "cuda":
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    else:
        amp_dtype = torch.float16
        use_grad_scaler = True
else:
    amp_dtype = None
    use_grad_scaler = False

scaler = GradScaler(enabled=use_grad_scaler)
ctx = nullcontext() if amp_dtype is None else torch.amp.autocast(device_type=device.type, dtype=amp_dtype)

# -------------------------
# Model config (must match checkpoint)
# -------------------------
model_config = {
    "conv1": "binary",
    "conv2": "binary",
    "conv3": "binary",
    "conv4": "binary",
    "conv5": "binary",
    "conv6": "binary",
    "conv7": "binary",
    "conv8": "binary",
    "conv9": "binary",
    "conv10": "binary",
    "fc1": "binary",
    "fc2": "binary",
    "final": "binary",
}

ORDER = [
    "conv1", "conv2", "conv3", "conv4", "conv5",
    "conv6", "conv7", "conv8", "conv9", "conv10",
    "fc1", "fc2", "final",
]
config_string = "".join(model_config[k][0] for k in ORDER)

# -------------------------
# WandB init (main only)
# -------------------------
if not is_main_process:
    os.environ["WANDB_MODE"] = "disabled"

if is_main_process:
    wandb.init(
        project="hpml-final",
        name=f"food101-prune-ft-{config_string}",
        config={
            "model_name": "Modular-CVNN",
            "gpu-type": "RTX 5090",
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "Adam",
            "num_workers": args.num_workers,
            "filter_dimension": args.filter_dimension,
            "finetune_epochs": args.finetune_epochs,
            "prune_amount": args.prune_amount,
            "compile": args.compile,
            "model_config": model_config,
            "device": str(device),
        }
    )
else:
    wandb.init(mode="disabled")

# Broadcast config from rank 0 to others (if DDP)
cfg = None
if use_ddp:
    # dist must be initialized already
    raw_cfg = dict(wandb.config) if is_main_process else None
    obj_list = [raw_cfg]
    dist.broadcast_object_list(obj_list, src=0)
    cfg = obj_list[0]
else:
    cfg = dict(wandb.config)

# Use config values (ensures consistency)
prune_amount = cfg.get("prune_amount", args.prune_amount)
finetune_epochs = cfg.get("finetune_epochs", args.finetune_epochs)
batch_size = cfg.get("batch_size", args.batch_size)
num_workers = cfg.get("num_workers", args.num_workers)
lr = cfg.get("lr", args.lr)
filter_dimension = cfg.get("filter_dimension", args.filter_dimension)
num_classes = cfg.get("num_classes", args.num_classes) if "num_classes" in cfg else args.num_classes
compile_mode = cfg.get("compile", args.compile)

# -------------------------
# Build model and load checkpoint
# -------------------------
device_str = str(device)
if is_main_process:
    logging.info(f"Device: {device_str} | DDP: {use_ddp} | rank: {rank} | world_size: {world_size}")

model = models.Models.MixedCVNN(
    model_config=model_config,
    image_channels=3,
    filter_dimension=filter_dimension,
    num_classes=num_classes
).to(device)

# -------------------------
# compile FIRST (checkpoints save compiled)
# -------------------------
if compile_mode:
    if is_main_process:
        print("Compiling model via torch.compile() ...")
    model = torch.compile(model)

# -------------------------
# Load checkpoint AFTER compile
# -------------------------
map_location = {"cuda:0": f"cuda:{args.local_rank}"} if use_ddp else device
ckpt = torch.load(args.checkpoint, map_location=map_location)
state = ckpt["model"] if "model" in ckpt else ckpt

model.load_state_dict(state, strict=True)

# -------------------------
# Apply structured pruning (fake pruning)
# -------------------------
if is_main_process:
    logging.info(f"Applying structured pruning amount={prune_amount}")

structured_prune_model(
    model,
    amount=prune_amount,
    include_bias=False,
    verbose=is_main_process
)

if args.make_prune_permanent:
    make_pruning_permanent(model)
    if is_main_process:
        print("Pruning made permanent (prune reparam removed).")

# Report sparsity to logs & wandb
sparsity = model_sparsity_report(model)
if is_main_process:
    pprint = None
    try:
        import pprint as _pp
        _pp.pprint(sparsity)
    except Exception:
        print(sparsity)
    logging.info(f"Sparsity report: {sparsity}")
    wandb.log({"global_sparsity": sparsity["_global"]["global_sparsity"]})

# Optionally compile the model (do this before wrapping in DDP)
if compile_mode:
    if is_main_process:
        print("Compiling model via torch.compile() ...")
    model = torch.compile(model)

# Wrap in DDP if needed
if use_ddp:
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False
    )

# -------------------------
# Optimizer, criterion, scheduler
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# simple cosine scheduler over finetune epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, finetune_epochs))

# -------------------------
# Data loaders
# -------------------------
train_loader, test_loader = data.image_dataset.get_food101_dataloaders(
    distributed=use_ddp,
    rank=rank,
    world_size=world_size,
    batch_size=batch_size,
    num_workers=num_workers
)

# -------------------------
# Finetune loop
# -------------------------
for epoch in range(1, finetune_epochs + 1):
    model.train()
    if use_ddp and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    train_correct = 0
    train_total = 0

    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    scheduler.step()

    avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float("nan")
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    train_time = time.perf_counter() - start_time

    # Validation
    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_sum += loss.item()
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    avg_test_loss = test_loss_sum / len(test_loader) if len(test_loader) > 0 else float("nan")
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    valid_time = time.perf_counter() - start_time

    if is_main_process:
        logging.info(
            f"[Prune-FineT] epoch: {epoch} | "
            f"train_loss: {avg_train_loss:.4f} | train_accuracy: {train_acc:.4f} | "
            f"test_loss: {avg_test_loss:.4f} | test_accuracy: {test_acc:.4f} | "
            f"training_time: {train_time:.2f}s | validation_time: {valid_time:.2f}s"
        )
        wandb.log({
            "train/loss": avg_train_loss,
            "train/accuracy": train_acc,
            "test/loss": avg_test_loss,
            "test/accuracy": test_acc,
            "epoch": epoch,
            "training_time": train_time,
            "validation_time": valid_time,
            "global_sparsity": sparsity["_global"]["global_sparsity"],
        }, step=epoch)

# final sparsity report & checkpoint save
final_sparsity = model_sparsity_report(model.module if use_ddp else model)
if is_main_process:
    logging.info(f"Final sparsity: {final_sparsity}")
    wandb.log({"final_global_sparsity": final_sparsity["_global"]["global_sparsity"]})
    # save checkpoint
    ckpt_out = {
        "epoch": 0,
        "model": (model.module.state_dict() if use_ddp else model.state_dict()),
        "optimizer": optimizer.state_dict(),
    }
    out_path = f"pruned_finetuned_{config_string}.pt"
    torch.save(ckpt_out, out_path)
    logging.info(f"Saved pruned+finetuned checkpoint to {out_path}")

# cleanup
if is_main_process:
    wandb.finish()
if use_ddp:
    dist.destroy_process_group()
