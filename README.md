# HPML Project: 1-Bit Model Parameter Reproduction

## Team Information
- **Team Name**: 1-Bit Reproducers
- **Members**:
  - Nghi Dao (nmd2167)
  - Shangchen Cai (sc5386)
  - Lance Chou (lz2837)
  - Qinghui Zeng (qz2548)

---

## 1. Problem Statement
Deep neural networks have achieved remarkable progress but come with high computational and memory costs, limiting deployment on resource-constrained devices. This project reproduces and improves upon the paper **"1 Bit Is All We Need: Binary Normalized Neural Networks"** (Cabral et al.).

Our goal is to implement and evaluate **Binary Normalized Neural Networks (BNNNs)**, where weights and biases are quantized to 1-bit. We aim to verify if these models can maintain competitive performance compared to full-precision (32-bit) counterparts while significantly reducing memory footprint. We also introduce a **learnable scaling parameter** ($\alpha$) to the normalization layer to improve stability and performance in deeper networks.

---

## 2. Model Description
We implemented binary variants of two distinct architectures using **PyTorch**.

### Core Mechanism: Binary Normalized Layers

We implemented custom layers (`BinaryLinear` and `BinaryConv2d`) located in `binary_layers/`.
- **Forward Pass:**  
  Weights `W` and biases `b` are binarized based on their mean:  
  `Quant(v) = I(v > mean(v))`
- **Normalization:**  
  To address distribution shifts (where the output distribution tends toward `N(0, n)` for `n` layers), we added a learnable scale parameter `α`:  
  `z = (z - mean(z)) / (std(z) + ε) * α`
- **Backward Pass:**  
  Gradients are computed on the full-precision weights.

### Architectures
1.  **CNN (Food-101 Classification):**
    - **Structure:** 5 Convolutional blocks (2 layers each) followed by 2 Fully Connected layers.
    - **Channel Sizes:** 32, 64, 64, 128, 256.
    - **Customization:** Standard `Conv2d` and `Linear` layers are replaced with our custom binary layers.

2.  **Transformer (Language Modeling):**
    - **Structure:** GPT-2 architecture (124M parameters).
    - **Specs:** Embedding dim 768, 12 heads, 12 layers.
    - **Customization:** We explored hybrid quantization, replacing MLP and QKV projections with binary layers while keeping embeddings and LayerNorm in full precision.

---

## 3. Final Results Summary

| Metric | CNN (Mixed Binary) | LLM (6 Binary Layers) |
|----------------------|-------------|-----------------------|
| **Test Accuracy / Val Loss** | **~73.6%** (Acc) | **3.348** (Val Loss) |
| **Baseline Metric** | 74.0% (Full Precision) | 3.300 (Full Precision) |
| **Model Size Reduction** | ~70% | ~50%  |
| **Training Device** | Multi-GPU (DDP) | Multi-GPU (DDP) |
| **Dataset** | Food-101 | C4 |

*Note: The CNN maintains stability up to ~70% binary parameter ratio. The LLM showed that a hybrid approach (6 binary layers) offers the best trade-off, while a fully binary 12-layer transformer failed to converge.*

---

## 4. Reproducibility Instructions

### A. Requirements
Ensure you have Python 3.8+ and PyTorch, and datasets installed.

### B. Wandb Dashboard
View our training runs, sweeps, and evaluation metrics here:

https://wandb.ai/chadcai2023-columbia-university/1bit-llm-c4/table?nw=nwuserchadcai2023

https://wandb.ai/lz2837-columbia-university/hpml-final?nw=nwuserlz2837

### C. Wandb Dashboard
To train the models, execute the scripts from the root directory to ensure module imports resolve correctly. 

## Training the CNN:
```bash
python train_cnn/train.py [options]

#For distributed training:
torchrun train_cnn/train.py --nproc_per_node=2 [options]
```

| Argument | Type | Default | Choices | Description |
|--------|------|---------|---------|-------------|
| `model_name` | `str` | — | — | Descriptive name for the model architecture. Used only for logging and experiment tracking; does not affect model behavior. |
| `gpu_type` | `str` | — | — | Records the GPU model used for training. Informational only, for experiment bookkeeping. |
| `batch_size` | `int` | — | — | Number of training samples processed per iteration on each process. |
| `lr` | `float` | — | — | Learning rate used by the optimizer. |
| `optimizer` | `str` | `"Adam"` | — | Optimization algorithm used for training. |
| `num_workers` | `int` | — | — | Number of worker processes used by the PyTorch DataLoader for data loading and preprocessing. |
| `filter_dimension` | `int` | — | — | Controls the spatial size of convolutional filters. Replaces `kernel_size`, which is unused. |
| `epochs` | `int` | — | — | Total number of training epochs. When resuming, this refers to the final target epoch, not remaining epochs. |
| `compile` | flag | `False` | — | Enable `torch.compile()` for model compilation and potential performance improvements. |
| `log_interval` | `int` | — | — | Interval (in epochs) at which training and evaluation statistics are logged. |
| `save_every` | `int` | — | — | Interval (in epochs) at which model checkpoints are saved locally. |
| `checkpoint_path` | `str` | — | — | File path used to save checkpoints and to load from when resuming training. |
| `resume` | flag | `False` | — | Resume training from an existing checkpoint. Requires `checkpoint_path` to point to a valid file. |
| `device` | `str` | — | — | Compute device used for training (e.g., CUDA device). Set programmatically. |

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

## Training the LLM:
```bash
python train_text/train.py [options]

#For distributed training:
torchrun train_text/train.py --nproc_per_node=2 [options]
```
| Argument | Type | Default | Choices | Description |
|--------|------|---------|---------|-------------|
| `--qkv_proj` | `str` | `"fp"` | `"fp"`, `"bin"` | Projection type for QKV layers (`"fp"` = floating point, `"bin"` = binary). |
| `--mlp_proj` | `str` | `"fp"` | `"fp"`, `"bin"` | Projection type for MLP layers. |
| `--c_proj` | `str` | `"fp"` | `"fp"`, `"bin"` | Projection type for output (`c_proj`) layers. |
| `--n_embd` | `int` | `768` | — | Embedding dimension size. |
| `--n_binary` | `int` | `0` | — | Number of binary layers (0 disables binary layers). |
| `--local_rank` | `int` | `-1` | — | Local rank for Distributed Data Parallel (DDP) training. |
| `--profile` | flag | `False` | — | Enable profiling mode when set. |
| `--resume_run` | `str` | `None` | — | Weights & Biases run ID to resume, or `"auto"` to resume the latest run. |


*Note: For the LLM pruning experiments, refer to train_text/pruning.ipynb.*

## Quantizing the LLM:
After training a full precision model, save it as .pt and run:
```bash
python ptq.py --ckpt path/to/full_model.pt
```

### D. Evaluation
All evaluation results are saved in WandB during training.


### E. Quickstart: Minimum Reproducible Result
```bash
torchrun train_cnn/train.py --local_rank 0
python train_text/train.py --n_binary=6
```
