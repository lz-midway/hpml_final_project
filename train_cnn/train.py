import os, time, logging, argparse, sys
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# -----------------------------------------------------------------------------
#  Project imports
# -----------------------------------------------------------------------------
from models import CNN, CNNConfig, Conv2dConfig, ConnectedConfig
from data import image_dataset       
import binary_layers
# -----------------------------------------------------------------------------

# ------------------------------- DDP imports ---------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from contextlib import nullcontext
# -----------------------------------------------------------------------------

# ------------------------------ misc set-up ----------------------------------
torch.manual_seed(42)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format="%(asctime)s | %(message)s",
                    datefmt="%m-%d %H:%M")

# WandB key / warnings
os.environ["WANDB_API_KEY"] = "0f853b7aa9e6bd44416474b253642758cf20704f" 
os.environ["PYTHONWARNINGS"] = "ignore"

# ------------------------------ CLI args -------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help="For DDP")
args = parser.parse_args()

if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

# -------------------------- Distributed set-up -------------------------------
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank        = dist.get_rank()
    world_size  = dist.get_world_size()
    use_ddp     = True
else:
    rank, world_size, use_ddp = 0, 1, False

is_main_process = (rank == 0)

# ------------------------------ Device & AMP ---------------------------------
device = torch.device(f"cuda:{args.local_rank}" if use_ddp
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

if device.type == "cuda" and torch.cuda.is_bf16_supported():
    amp_dtype       = torch.bfloat16
    use_grad_scaler = False
else:
    amp_dtype       = torch.float16 if device.type == "cuda" else None
    use_grad_scaler = (amp_dtype is not None)

ctx    = nullcontext() if amp_dtype is None else torch.amp.autocast(device_type=device.type,
                                                                    dtype=amp_dtype)
scaler = GradScaler(enabled=use_grad_scaler)

# ------------------------------ WandB config ------------------------------
import os

use_wandb = True 
if not is_main_process:
    os.environ["WANDB_MODE"] = "disabled"

# canonical default model_config (kept as a dict so nesting is preserved)
DEFAULT_MODEL_CONFIG = {
    "conv1": "binary", "conv2": "binary",
    "conv3": "real", "conv4": "real",
    "conv5": "real", "conv6": "real",
    "conv7": "real", "conv8": "real",
    "conv9": "real", "conv10": "real",
    "fc1": "binary", "fc2": "binary", "final": "binary"
}

# top-level defaults for wandb.init (these are the defaults — sweep can override top-level keys)
default_config = {
    "model_name": "Modular-CVNN",
    "gpu-type": "RTX 5090",
    "batch_size": 64,
    "lr": 1e-4,
    "optimizer": "Adam",
    "num_workers": 12,
    "filter_dimension": 5,
    "epochs": 4,
    "compile": True,
    "log_interval": 10,
    "save_every": 4,
    "checkpoint_path": "checkpoint.pt",
    "resume": False,
    "prefetch_factor": 4,
    # put the default nested model_config here (sweeps may replace it entirely or partially)
    "model_config": DEFAULT_MODEL_CONFIG,
    "device": str(device)
}

# Initialize wandb on main; other ranks disabled
if is_main_process:
    wandb.init(project="hpml-final",
               name="food101-cnn-refactor_verify",
               config=default_config)
else:
    wandb.init(mode="disabled")

# Grab the runtime config from wandb (this contains sweep overrides if any)
# then deep-merge the nested model_config with DEFAULT_MODEL_CONFIG so partial sweep overrides work.
if is_main_process:
    runtime_cfg = dict(wandb.config)  # snapshot
else:
    runtime_cfg = None

# DDP-safe broadcast of the runtime config snapshot
obj_list = [runtime_cfg]
dist.broadcast_object_list(obj_list, src=0)
cfg = obj_list[0]   # now every rank has the same flat dictionary

# Ensure model_config is a merged dict: default values filled in if sweep provided partial dict
raw_model_config = cfg.get("model_config", {}) or {}
# If wandb provided a non-dict for model_config (unlikely), coerce to dict safely
if not isinstance(raw_model_config, dict):
    raw_model_config = dict(raw_model_config)

# deep-merge: keys from raw_model_config override defaults, missing keys preserved
merged_model_config = dict(DEFAULT_MODEL_CONFIG)
merged_model_config.update(raw_model_config)

# write the merged model_config back into the active cfg so later code reads merged values
cfg["model_config"] = merged_model_config

# build a short config string for run naming
ORDER = [
    "conv1", "conv2", "conv3", "conv4", "conv5",
    "conv6", "conv7", "conv8", "conv9", "conv10",
    "fc1", "fc2", "final",
]
config_string = "".join(cfg["model_config"][k][0] for k in ORDER)

# Detect whether this run is part of a sweep for naming purposes.
# We check common signals: wandb.run.sweep_id (if available) or env var.
is_sweep_run = False
if is_main_process:
    try:
        sweep_id = getattr(wandb.run, "sweep_id", None)
    except Exception:
        sweep_id = None
    # common env var set when running a sweep agent
    sweep_env = os.environ.get("WANDB_SWEEP_ID") or os.environ.get("WANDB_AGENT_ID")
    is_sweep_run = bool(sweep_id or sweep_env)

    if is_sweep_run:
        # include sweep id if available to make names unique & traceable
        suffix = f"sweep-{sweep_id}" if sweep_id else "sweep"
        wandb.run.name = f"food101-modular-nn-{config_string}-{suffix}"
    else:
        wandb.run.name = f"food101-modular-nn-{config_string}-manual"



# ------------------------- helper: pretty print cfg --------------------------
def pretty_print_cnn_config(cnn_cfg: CNNConfig):
    if not is_main_process: return
    header = "=========== CNN CONFIGURATION ==========="
    lines  = [header]

    lines.append("--- Convolutional Blocks ‑--")
    for i, c in enumerate(cnn_cfg.ConvLayers, 1):
        lines.append(f" Block {i}: {c.proj_1.__name__} -> {c.proj_2.__name__} | "
                     f"ch={c.channels}->{c.out_channels}")

    lines.append("--- Connected Layers ‑--")
    for i, c in enumerate(cnn_cfg.ConnectedLayers, 1):
        lines.append(f" FC {i}: {c.proj.__name__} | {c.in_dim}->{c.out_dim}")

    full = "\n".join(lines)
    print(full)
    logging.info("\n" + full)

# --------------------------- Build the model ---------------------------------
# Helper to map string config to actual class
def get_layer_cls(name, layer_type="conv"):
    if name == "binary":
        return binary_layers.Conv2d if layer_type == "conv" else binary_layers.Linear
    elif name == "real":
        return nn.Conv2d if layer_type == "conv" else nn.Linear
    raise ValueError(f"Unknown layer type: {name}")

mc = cfg["model_config"]
model_cfg = CNNConfig()
ORDER = [
    "conv1", "conv2", "conv3", "conv4", "conv5",
    "conv6", "conv7", "conv8", "conv9", "conv10",
    "fc1", "fc2", "final",
]
config_string = "".join(mc[k][0] for k in ORDER)

if is_main_process:
    wandb.run.name = f"food101-modular-nn-{config_string}-verify-1"

# Map the 10 conv layers from config to the 5 Blocks (2 layers per block)
# Structure matches original BCVNN: 
# Block 1 (32->64), Block 2 (64->64), Block 3 (64->64), Block 4 (64->128), Block 5 (128->256)
model_cfg.ConvLayers = [
    Conv2dConfig(channels=32,  out_channels=64,  kernel_size=cfg["filter_dimension"], pool=True, 
                 proj_1=get_layer_cls(mc['conv1']), proj_2=get_layer_cls(mc['conv2'])),
    
    Conv2dConfig(channels=64,  out_channels=64,  kernel_size=cfg["filter_dimension"], pool=True, 
                 proj_1=get_layer_cls(mc['conv3']), proj_2=get_layer_cls(mc['conv4'])),
    
    Conv2dConfig(channels=64,  out_channels=64,  kernel_size=cfg["filter_dimension"], pool=True, 
                 proj_1=get_layer_cls(mc['conv5']), proj_2=get_layer_cls(mc['conv6'])),
    
    Conv2dConfig(channels=64,  out_channels=128, kernel_size=cfg["filter_dimension"], pool=True, 
                 proj_1=get_layer_cls(mc['conv7']), proj_2=get_layer_cls(mc['conv8'])),
                 
    Conv2dConfig(channels=128, out_channels=256, kernel_size=cfg["filter_dimension"], pool=False, 
                 proj_1=get_layer_cls(mc['conv9']), proj_2=get_layer_cls(mc['conv10'])),
]

model_cfg.ConnectedLayers = [
    ConnectedConfig(in_dim=256, out_dim=256, proj=get_layer_cls(mc['fc1'], "linear")),
    ConnectedConfig(in_dim=256, out_dim=256, proj=get_layer_cls(mc['fc2'], "linear")),
    ConnectedConfig(in_dim=256, out_dim=101, proj=get_layer_cls(mc['final'], "linear"))
]

pretty_print_cnn_config(model_cfg)

model = CNN(config=model_cfg, img_channels=3, num_classes=101).to(device)

if cfg["compile"]:
    if is_main_process:
        print("Compiling model ...")
    model = torch.compile(model)

if use_ddp:
    model = DDP(model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False)

# ----------------------- Optimizer / scheduler -------------------------------
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=cfg["lr"])

warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=20)
decay  = CosineAnnealingLR(optimizer, T_max=1100)
scheduler = SequentialLR(optimizer, [warmup, decay], milestones=[20])

# -------------------------- Checkpoint resume --------------------------------
start_epoch = 1
if cfg["resume"] and os.path.exists(cfg["checkpoint_path"]):
    map_loc = {"cuda:0": f"cuda:{args.local_rank}"} if use_ddp else device
    ckpt = torch.load(cfg["checkpoint_path"], map_location=map_loc)

    mdl_state = ckpt["model"]
    (model.module if isinstance(model, DDP) else model).load_state_dict(mdl_state)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    logging.info(f"Resumed from checkpoint at epoch {start_epoch}")

# ------------------------------- Data ----------------------------------------
train_loader, test_loader = image_dataset.get_food101_dataloaders(
    distributed = use_ddp,
    rank        = rank,
    world_size  = world_size,
    batch_size  = cfg["batch_size"],
    num_workers = cfg["num_workers"],
    prefetch_factor = cfg["prefetch_factor"]
)

# ---------------------------- Training loop ----------------------------------
for epoch in range(start_epoch, cfg["epochs"] + 1):
    if use_ddp and hasattr(train_loader, "sampler"):
        train_loader.sampler.set_epoch(epoch)

    # ---------------------- Train ------------------------------------------
    model.train()
    t0 = time.perf_counter()
    loss_sum, correct, seen = 0.0, 0, 0

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with ctx:
            logits = model(x)
            loss   = criterion(logits, y)

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        loss_sum += loss.item()
        preds     = logits.argmax(1)
        correct  += (preds == y).sum().item()
        seen     += y.size(0)

        if isinstance(model, DDP):
            model.module.update_cache()
        else:
            model.update_cache()

    train_loss = loss_sum / len(train_loader)
    train_acc  = correct   / max(1, seen)
    train_time = time.perf_counter() - t0

    # ---------------------- Validation -------------------------------------
    model.eval()
    t0 = time.perf_counter()
    val_loss_sum, val_correct, val_seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ctx:
                logits = model(x)
                loss   = criterion(logits, y)

            val_loss_sum += loss.item()
            val_correct  += (logits.argmax(1) == y).sum().item()
            val_seen     += y.size(0)

    val_loss = val_loss_sum / len(test_loader)
    val_acc  = val_correct  / max(1, val_seen)
    val_time = time.perf_counter() - t0

    # ---------------------- Logging ----------------------------------------
    if is_main_process:
        if epoch % cfg["log_interval"] == 0:
            msg = (f"Epoch {epoch:3d} | "
                   f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                   f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
                   f"train_time {train_time:.1f}s | val_time {val_time:.1f}s")
            print(msg)
            logging.info(msg)

        wandb.log(dict(
            epoch           = epoch,
            train_loss      = train_loss,
            train_accuracy  = train_acc,
            val_loss        = val_loss,
            val_accuracy    = val_acc,
            train_time      = train_time,
            val_time        = val_time
        ), step=epoch)

        if epoch % cfg["save_every"] == 0:
            state_path = cfg["checkpoint_path"]
            if isinstance(model, DDP):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save({
                "epoch":     epoch,
                "model":     state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict() if use_grad_scaler else None
            }, state_path)
            logging.info(f"Saved checkpoint to {state_path}")

wandb.finish()
