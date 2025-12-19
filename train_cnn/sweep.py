import os
import wandb
import subprocess
import yaml


# Wandb Login key
# os.environ["WANDB_API_KEY"] = "" 
# os.environ["PYTHONWARNINGS"] = "ignore"

with open("train_cnn/sweep.yaml") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep_config, project="hpml-final")
print(f"Created sweep: {sweep_id}")

def train_ddp():
    # Launch distributed training for 2 GPUs
    subprocess.run([
        "torchrun",
        "--nproc_per_node=2",
        "train_cnn/train.py"
    ])

wandb.agent(sweep_id, function=train_ddp)
