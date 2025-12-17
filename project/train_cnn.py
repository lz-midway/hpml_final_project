
# Food 101 training with CNN
import os
import torch
import torch.nn as nn
import torch.optim as optim
import data.image_dataset
import models.Models
import wandb
import time
import logging
from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

# distributed training parts
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import argparse


# log.txt
logging.basicConfig(filename="log.txt", level=logging.INFO,
                    format="%(asctime)s | %(message)s", datefmt="%m-%d %H:%M")

torch.manual_seed(42)

# Wandb Login key
# os.environ["WANDB_API_KEY"] = ""
# os.environ["PYTHONWARNINGS"] = "ignore"

# parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help="For DDP")
args = parser.parse_args()

# If launched with torchrun, override local_rank from env
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

logging.info(f"local rank: {str(args.local_rank)}")

# Distributed setup
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    use_ddp = True
else:
    rank = 0
    world_size = 1
    use_ddp = False
    
is_main_process = (rank == 0) # for DDP, check if it is main process

# setting device
if use_ddp:
    device = torch.device(f"cuda:{args.local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use bf16 and drop grad_scaler if bf16 is supported
if device.type == "cuda":
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    else:
        amp_dtype = torch.float16
        use_grad_scaler = True
else:
    amp_dtype = None


# WandB (only rank 0 has wandb logging)
if not is_main_process:
    os.environ["WANDB_MODE"] = "disabled"

if is_main_process:
    # NOTE: config may be overridden by sweep agent; defaults work for normal runs
    wandb.init(
        project="hpml-final",
        name="food101-modular-nn",
        config={
            "model_name": "Modular-CVNN",
            "gpu-type": "RTX 5090",
            "batch_size": 64,
            "lr": 1e-4,
            "optimizer": "Adam",
            "num_workers": 4,
            # "kernel_size": 5,
            "filter_dimension": 5,
            "epochs": 1000,
            "compile": True,
            "log_interval": 10,
            "save_every": 50,
            "checkpoint_path": "checkpoint.pt",
            "resume": False,
            "model_config": {
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
                "final": "binary"
            },
            "device": str(device)
        }
    )
else:
    wandb.init(mode="disabled")

# broadcast sweep/default config to all ranks
raw_cfg = dict(wandb.config) if is_main_process else None
obj_list = [raw_cfg]
dist.broadcast_object_list(obj_list, src=0)
cfg = obj_list[0]      # all ranks now share identical config

# parameters
epochs = cfg["epochs"]
lr = cfg["lr"]
filter_dimension = cfg["filter_dimension"]
compile_mode = cfg["compile"]
batch_size = cfg["batch_size"]
num_workers = cfg["num_workers"]
# kernel_size = cfg["kernel_size"]
log_interval = cfg["log_interval"]
save_every = cfg["save_every"]
checkpoint_path = cfg["checkpoint_path"]
resume = cfg["resume"]

model_config = cfg["model_config"]

ORDER = [
    "conv1", "conv2", "conv3", "conv4", "conv5",
    "conv6", "conv7", "conv8", "conv9", "conv10",
    "fc1", "fc2", "final",
]

config_string = "".join(model_config[k][0] for k in ORDER)


if is_main_process:
    wandb.run.name = f"food101-modular-nn-{config_string}-test-1"


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    torch.save({
        "epoch": epoch,
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }, path)

ctx = nullcontext() if amp_dtype is None else torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
scaler = GradScaler(enabled=use_grad_scaler)





# Initialize model, loss, optimizer, prepare for DDP
model = models.Models.MixedCVNN(model_config=model_config, image_channels=3, filter_dimension=filter_dimension).to(device)

# # Initialize model, loss, optimizer, prepare for DDP
# model = RealLayers.RealCVNN(image_channels=3, filter_dimension=filter_dimension).to(device)

# torch compile before DDP wrap
if compile_mode:
    if is_main_process:
        print("compiling the model...")
    model = torch.compile(model)

# Wrap in DDP
if use_ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False
    )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Warm-up: 20 steps
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=20)

# Decay: 1100 steps total
decay = CosineAnnealingLR(optimizer, T_max=1100)

scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[20])

# load model checkpoint if resuming
start_epoch = 1
if resume and os.path.exists(checkpoint_path):
    map_location = {"cuda:%d" % 0: f"cuda:{args.local_rank}"} if use_ddp else device
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    # load model state into underlying module when DDP-wrapped
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    if is_main_process:
        logging.info(f"Resumed from checkpoint at epoch {start_epoch}")

# acquire data
if use_ddp:
    train_loader, test_loader = data.image_dataset.get_food101_dataloaders(
        distributed=(args.local_rank != -1),
        rank=rank,
        world_size=dist.get_world_size() if args.local_rank != -1 else 1,
        batch_size=batch_size,
        num_workers=num_workers
    )
else:
    train_loader, test_loader = data.image_dataset.get_food101_dataloaders()


# Training loop
for epoch in range(start_epoch, epochs+1):
    model.train()
    # needed for distributed training for proper shuffling
    if use_ddp and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)
    
    total_loss = 0.0
    train_correct = 0
    train_total = 0

    start_time = time.perf_counter()

    # training pass
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # accumulate train stats
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # compute train averages
    avg_train_loss = total_loss / len(train_loader)
    train_acc = train_correct / train_total if train_total > 0 else 0.0

    train_time = time.perf_counter() - start_time

    scheduler.step()

    start_time = time.perf_counter()

    # validation/testing pass
    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss_sum += loss.item()
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    avg_test_loss = test_loss_sum / len(test_loader)
    test_acc = test_correct / test_total if test_total > 0 else 0.0

    valid_time = time.perf_counter() - start_time

    # log to wandb
    if is_main_process:
        if epoch % log_interval == 0:
            logging.info(
                f"epoch: {epoch} | "
                f"train_loss: {avg_train_loss:.4f} | train_accuracy: {train_acc:.4f} | "
                f"test_loss: {avg_test_loss:.4f} | test_accuracy: {test_acc:.4f} | "
                f"training_time: {train_time:.2f}s | validation_time: {valid_time:.2f}s | "
                f"device: {device}"
            )

        wandb.log({
        "train/loss": avg_train_loss,
        "train/accuracy": train_acc,
        "test/loss": avg_test_loss,
        "test/accuracy": test_acc,
        "training_time": train_time,
        "validation_time": valid_time,
        "epoch": epoch,
        "device": str(device)
        }, step=epoch)

    if is_main_process and epoch % save_every == 0:
        save_checkpoint(epoch, model, optimizer, scheduler, scaler if use_grad_scaler else None, checkpoint_path)



wandb.finish()