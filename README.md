# hpml_final_project
Repository for the HPML final project


## Steps to run the project:
(Create and install required environment)


### Steps to run the CNN section

1. **Navigate to the project directory**

   Move into the root folder of the project:

   ```bash
   cd hpml_final_project
   ```

2. **Set required environment variables**

   Before running any training, configure your Weights & Biases (wandb) API key and suppress Python warnings. Replace the API key with your own.

   ```bash
   export WANDB_API_KEY="YOUR_WANDB_API_KEY"
   export PYTHONWARNINGS="ignore"
   ```

3. **Run a single CNN training job (multi-GPU)**

   To launch a standard training run using 2 GPUs:

   ```bash
   torchrun --nproc_per_node=2 train_cnn/train.py
   ```

   To run the same job in the background and redirect logs to a file:

   ```bash
   nohup torchrun --nproc_per_node=2 train_cnn/train.py > out.log 2>&1 &
   ```

4. **Run hyperparameter sweeps**

   To start a wandb sweep agent:

   ```bash
   python3 train_cnn/sweep.py
   ```

   To run the sweep agent in the background:

   ```bash
   nohup python3 train_cnn/sweep.py </dev/null > agent.log 2>&1 &
   ```

5. **Resuming from a previous checkpoint**

   To resume training from a saved checkpoint, edit the wandb configuration in `train_cnn/train.py` and set the `resume` flag to `True`. Ensure that `checkpoint_path` points to the correct checkpoint file:

   ```python
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
        "epochs": 8,
        "compile": True,
        "log_interval": 10,
        "save_every": 4,
        "checkpoint_path": "checkpoint.pt",
        "resume": True,
        "prefetch_factor": 4,
        # put the default nested model_config here (sweeps may replace it entirely or partially)
        "model_config": DEFAULT_MODEL_CONFIG,
        "device": str(device)
    }

   ```

   **Note:**
   The epoch stored in a checkpoint reflects the number of epochs already completed. If a checkpoint was saved at epoch 400 or 600 and the target is 1000 total epochs, the `epochs` parameter should remain set to `1000`; training will resume automatically from the appropriate epoch.

### CNN configuration

This section describes how to configure and control the CNN experiments, both for single runs and for hyperparameter sweeps. Configuration is handled through a `wandb.init()` call in `train_cnn.py` for standard runs, and through `sweep.yaml` for automated sweeps.

---

#### Single-run configuration (`train_cnn/train.py`)

For a standard (non-sweep) run, experiment parameters are defined in the `config` dictionary passed to `wandb.init()`. Make sure to adjust project name (not name, which is set later) as needed:

```python
wandb.init(
    project="hpml-final",
    name="food101-modular-nn",
    config={ ... }
)
```

The key configuration fields are:

* **`model_name`**
  A descriptive name for the model architecture. This value is used for logging and experiment tracking only and does not affect model behavior.

* **`gpu-type`**
  Records the GPU model used for training. This parameter is informational and intended for experiment bookkeeping.

* **`batch_size`**
  Number of training samples processed per iteration on each process.

* **`lr`**
  Learning rate used by the optimizer.

* **`optimizer`**
  Optimization algorithm. Currently set to Adam.

* **`num_workers`**
  Number of worker processes used by the PyTorch `DataLoader` for dataset loading and preprocessing. Increasing this can improve input pipeline throughput.

* **`filter_dimension`**
  Controls the spatial size of convolutional filters. This parameter replaces `kernel_size`, which is unused and therefore commented out in the code.

* **`epochs`**
  Total number of training epochs. When resuming from a checkpoint, this value still refers to the *final target epoch*, not the remaining number of epochs.

* **`compile`**
  Boolean flag indicating whether to enable `torch.compile()` for model compilation and potential performance improvements.

* **`log_interval`**
  Interval (in epochs) at which training and evaluation statistics are printed via the logging system.

* **`save_every`**
  Interval (in epochs) at which model checkpoints are saved locally.

* **`checkpoint_path`**
  File path where model checkpoints are written and from which they are loaded when resuming.

* **`resume`**
  Boolean flag indicating whether training should resume from an existing checkpoint. When set to `True`, `checkpoint_path` must point to a valid checkpoint file.

* **`device`**
  The compute device used for training (e.g., CUDA device). This value is set programmatically.

---

#### Layer-wise model configuration (`model_config`)

The `model_config` field defines the type of each layer in the CNN:

```python
"model_config": {
    "conv1": "binary",
    ...
    "final": "binary"
}
```

Each entry corresponds to a specific layer in the network:

* Convolutional layers: `conv1` through `conv10`
* Fully connected layers: `fc1`, `fc2`
* Output layer: `final`

Each layer can be set to:

* **`"binary"`**: uses the proposed binary-normalized layer with binarized parameters.
* **`"real"`**: uses a standard 32-bit floating-point layer.

By modifying these values, the network can be partially or fully binarized without changing the overall architecture.

---

#### Sweep configuration (`sweep.yaml`)

Hyperparameter sweeps are defined in `sweep.yaml`. The sweep configuration largely mirrors the single-run configuration, with some important differences:

* Parameters must be specified using either:

  * **`value`** for fixed parameters, or
  * **`values`** for parameters being swept over.

Example:

```yaml
batch_size:
  value: 64

conv1:
  values: ["real", "binary"]
```

In the sweep setup:

* Most training parameters (`batch_size`, `lr`, `epochs`, etc.) are fixed using `value`.
* The `model_config` section uses `values` to sweep over `"real"` and `"binary"` options for selected layers.
* Fully connected and final layers are fixed to `"binary"`.

The sweep is launched programmatically in `sweep.py`:

```python
sweep_id = wandb.sweep(sweep_config, project="hpml-final")
```

The `project` name should be updated as needed to match your wandb workspace.

---

#### Run naming and identification

Within `train_cnn/train.py`, each run’s name is programmatically adjusted:

```python
wandb.run.name = f"food101-modular-nn-{config_string}-test-1"
```

Here:

* `config_string` is automatically constructed from `model_config` by encoding each layer in order:

  * `r` for real-valued layers
  * `b` for binary layers
* The order of letters reflects the layer order in the network.

Note that earlier runs may not strictly follow a consistent ordering, as explicit enforcement of layer order was added only after most experiments had already been conducted.

---

#### Notes on overrides

When running sweeps, parameters defined by the sweep agent override the defaults specified in `train_cnn.py`. The default configuration is therefore suitable for single runs, while `sweep.yaml` should be treated as the authoritative source during sweep execution.

