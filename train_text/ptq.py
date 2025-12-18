"""
Native PyTorch Post-Training Quantization (PTQ)
Fully native implementation using torch.ao.quantization API.

Two approaches:
1. Dynamic Quantization - Simplest, weights quantized, activations computed dynamically
2. Static Quantization - Both weights and activations quantized after calibration

This is SOTA native PyTorch quantization for fair comparison.
"""
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig,
    quantize_dynamic,
    prepare,
    convert,
    default_dynamic_qconfig,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from tqdm import tqdm
import copy


# ============================================================================
# Dynamic Quantization (Recommended for Transformers/LLMs)
# ============================================================================

def dynamic_quantize_int8(model):
    """
    Apply INT8 dynamic quantization using native PyTorch.

    This is the recommended approach for transformer models:
    - Weights are quantized to INT8
    - Activations are quantized dynamically at runtime
    - No calibration needed
    - Works out of the box with any model

    Returns:
        Quantized model with INT8 weights
    """
    model_cpu = copy.deepcopy(model).cpu().eval()

    return quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantize all Linear layers
        dtype=torch.qint8
    )


def dynamic_quantize_float16(model):
    """
    Apply float16 dynamic quantization.
    Useful for GPU inference with reduced memory.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()

    return quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.float16
    )


# ============================================================================
# Static Quantization (Eager Mode)
# ============================================================================

def static_quantize_int8(model, calibration_loader, device='cpu', num_batches=32, backend='fbgemm'):
    """
    Apply INT8 static quantization using native PyTorch eager mode.

    Workflow:
    1. Prepare model (insert observers)
    2. Calibrate with representative data
    3. Convert to quantized model

    Args:
        model: Model to quantize
        calibration_loader: DataLoader for calibration
        device: Device for calibration (should be 'cpu' for quantization)
        num_batches: Number of calibration batches
        backend: 'fbgemm' for x86, 'qnnpack' for ARM

    Returns:
        Quantized model
    """
    model = copy.deepcopy(model).cpu().eval()

    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Get default qconfig for the backend
    model.qconfig = get_default_qconfig(backend)

    # Prepare model - inserts observers
    prepared_model = prepare(model, inplace=False)

    # Calibration - run representative data
    prepared_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_loader, total=num_batches, desc="Calibrating")):
            if i >= num_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.cpu()  # Static quant requires CPU
            try:
                prepared_model(x)
            except Exception as e:
                print(f"Calibration warning: {e}")
                continue

    # Convert to quantized model
    quantized_model = convert(prepared_model, inplace=False)

    return quantized_model


# ============================================================================
# FX Graph Mode Quantization (More Flexible)
# ============================================================================

def fx_quantize_int8(model, calibration_loader, num_batches=32, backend='fbgemm'):
    """
    Apply INT8 quantization using FX Graph Mode.

    FX mode is more flexible than eager mode:
    - Automatically handles module fusion
    - Better pattern matching
    - Works with more model architectures

    Args:
        model: Model to quantize
        calibration_loader: DataLoader for calibration
        num_batches: Number of calibration batches
        backend: 'fbgemm' for x86, 'qnnpack' for ARM

    Returns:
        Quantized model
    """
    model = copy.deepcopy(model).cpu().eval()

    # Create qconfig mapping
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig(backend))

    # Get example input for tracing
    example_input = next(iter(calibration_loader))
    if isinstance(example_input, (list, tuple)):
        example_input = example_input[0]
    example_input = example_input.cpu()

    try:
        # Prepare with FX
        prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=(example_input,))

        # Calibration
        prepared_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_loader, total=num_batches, desc="FX Calibrating")):
                if i >= num_batches:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                prepared_model(x.cpu())

        # Convert
        quantized_model = convert_fx(prepared_model)
        return quantized_model

    except Exception as e:
        print(f"FX quantization failed: {e}")
        print("Falling back to dynamic quantization...")
        return dynamic_quantize_int8(model)


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def count_quantized_layers(model):
    """Count quantized vs regular linear layers."""
    from torch.ao.nn.quantized import Linear as QuantizedLinear
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

    quantized = 0
    dynamic_quantized = 0
    regular = 0

    for m in model.modules():
        if isinstance(m, DynamicQuantizedLinear):
            dynamic_quantized += 1
        elif isinstance(m, QuantizedLinear):
            quantized += 1
        elif isinstance(m, nn.Linear):
            regular += 1

    return {
        'static_quantized': quantized,
        'dynamic_quantized': dynamic_quantized,
        'regular': regular
    }


def compare_model_outputs(fp_model, q_model, test_input, device='cpu'):
    """Compare outputs between FP and quantized models."""
    fp_model = fp_model.to(device).eval()
    q_model = q_model.to('cpu').eval()  # Quantized models run on CPU

    test_input_cpu = test_input.cpu()
    test_input_device = test_input.to(device)

    with torch.no_grad():
        fp_out = fp_model(test_input_device).cpu()
        q_out = q_model(test_input_cpu)

    mse = ((fp_out - q_out) ** 2).mean().item()
    max_diff = (fp_out - q_out).abs().max().item()

    return {
        'mse': mse,
        'max_diff': max_diff,
        'fp_mean': fp_out.mean().item(),
        'q_mean': q_out.mean().item(),
    }


# ============================================================================
# High-Level API
# ============================================================================

def quantize_for_inference(model, method='dynamic', calibration_loader=None, **kwargs):
    """
    High-level API for quantizing a model for inference.

    Args:
        model: Model to quantize
        method: 'dynamic' (recommended), 'static', or 'fx'
        calibration_loader: Required for 'static' and 'fx' methods
        **kwargs: Additional arguments for the quantization function

    Returns:
        Quantized model
    """
    if method == 'dynamic':
        return dynamic_quantize_int8(model)
    elif method == 'static':
        if calibration_loader is None:
            raise ValueError("calibration_loader required for static quantization")
        return static_quantize_int8(model, calibration_loader, **kwargs)
    elif method == 'fx':
        if calibration_loader is None:
            raise ValueError("calibration_loader required for FX quantization")
        return fx_quantize_int8(model, calibration_loader, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dynamic', 'static', or 'fx'")


# ============================================================================
# Evaluation helpers
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device='cpu', max_batches=100):
    """Evaluate perplexity on a dataset."""
    model = model.to(device).eval()

    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Evaluating")):
        if i >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            idx, labels = batch[0], batch[1]
        else:
            idx = batch
            labels = batch

        idx = idx.to(device)
        labels = labels.to(device)

        logits = model(idx)

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {'loss': avg_loss, 'perplexity': perplexity}
