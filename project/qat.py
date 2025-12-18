"""
Native PyTorch Quantization-Aware Training (QAT)
Following official PyTorch QAT workflow:
1. Fuse modules (Linear patterns for transformers)
2. Set qconfig
3. Prepare model (insert FakeQuantize)
4. Train
5. Convert to quantized model
"""
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qat_qconfig,
    get_default_qconfig,
    prepare_qat,
    convert,
    fuse_modules,
)
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization import QConfig
import copy


def get_qconfig(bits=8, per_channel=True):
    """
    Get QAT qconfig for INT8 or INT4.
    Uses native PyTorch observers and fake quantize modules.
    """
    if bits == 8:
        qmin, qmax = -128, 127
    elif bits == 4:
        qmin, qmax = -8, 7
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    # Weight fake quantization (per-channel for better accuracy)
    if per_channel:
        weight_fake_quant = FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=qmin,
            quant_max=qmax,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )
    else:
        weight_fake_quant = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=qmin,
            quant_max=qmax,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )

    # Activation fake quantization (per-tensor)
    act_fake_quant = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0 if bits == 8 else 0,
        quant_max=255 if bits == 8 else 15,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )

    return QConfig(activation=act_fake_quant, weight=weight_fake_quant)


def fuse_transformer_modules(model):
    """
    Step 2: Fuse modules for QAT.
    For transformers, we can fuse Linear+GELU patterns in MLP blocks.
    Note: PyTorch doesn't have built-in Linear+GELU fusion, so we handle manually.
    """
    # For transformers, main fusion opportunity is in attention and MLP
    # PyTorch's fuse_modules works on Conv+BN+ReLU patterns
    # For Linear layers, we skip fusion as it's not directly supported
    # The quantization will still work, just without fusion optimization
    return model


def prepare_qat_model(model, bits=8, per_channel=True, skip_patterns=('head', 'embd')):
    """
    Step 3: Prepare model for QAT by inserting FakeQuantize modules.

    This replaces nn.Linear with quantization-aware versions that simulate
    quantization during training using native PyTorch ops.
    """
    model = copy.deepcopy(model)
    model.train()

    # Step 2: Fuse modules (limited for transformers)
    model = fuse_transformer_modules(model)

    # Get qconfig
    qconfig = get_qconfig(bits, per_channel)

    # Apply qconfig to all Linear layers except skipped ones
    def apply_qconfig(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear):
                if not any(p in full_name for p in skip_patterns):
                    child.qconfig = qconfig
                else:
                    child.qconfig = None  # Skip quantization
            else:
                apply_qconfig(child, full_name)

    apply_qconfig(model)

    # Step 3: Prepare for QAT - inserts FakeQuantize modules
    model = prepare_qat(model, inplace=True)

    return model


def convert_qat_model(model):
    """
    Step 4: Convert QAT model to quantized model for inference.
    Replaces FakeQuantize with actual quantized operations.
    """
    model.eval()
    model = convert(model, inplace=False)
    return model


# ============================================================================
# Alternative: Manual FakeQuantLinear wrapper for better DDP compatibility
# ============================================================================

class FakeQuantLinear(nn.Module):
    """
    Native PyTorch FakeQuantize wrapper for nn.Linear.
    Uses torch.ao.quantization.FakeQuantize for optimized CUDA ops.

    This is more DDP-friendly than the module-level prepare_qat approach.
    """
    def __init__(self, in_features, out_features, bias=False, bits=8, per_channel=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # The actual linear weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Quantization parameters
        if bits == 8:
            qmin, qmax = -128, 127
        elif bits == 4:
            qmin, qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        # Native FakeQuantize for weights
        if per_channel:
            self.weight_fake_quant = FakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=qmin,
                quant_max=qmax,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            )()
        else:
            self.weight_fake_quant = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=qmin,
                quant_max=qmax,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
            )()

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits=8, per_channel=True):
        """Create FakeQuantLinear from existing nn.Linear."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bits,
            per_channel
        )
        layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x):
        # Apply native fake quantization to weights (optimized CUDA kernel)
        w_q = self.weight_fake_quant(self.weight)
        return nn.functional.linear(x, w_q, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, bits={self.bits}'


def prepare_qat(model, bits=8, per_channel=True, skip_patterns=('head', 'embd')):
    """
    Prepare model for QAT by replacing nn.Linear with FakeQuantLinear.

    This is the recommended approach for DDP training as it:
    - Uses native PyTorch FakeQuantize (optimized CUDA kernels)
    - Works well with torch.compile
    - Compatible with DDP without find_unused_parameters issues

    Args:
        model: The model to prepare
        bits: Quantization bits (8 or 4)
        per_channel: Use per-channel quantization for weights
        skip_patterns: Module names to skip (embeddings, head)

    Returns:
        Model with FakeQuantLinear layers
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not any(p in name for p in skip_patterns):
            setattr(model, name, FakeQuantLinear.from_linear(child, bits, per_channel))
        else:
            prepare_qat(child, bits, per_channel, skip_patterns)
    return model


def convert_to_int(model):
    """
    Convert QAT model to inference mode with quantized weights.
    Replaces FakeQuantLinear with regular nn.Linear containing quantized weights.
    """
    for name, m in list(model.named_modules()):
        if isinstance(m, FakeQuantLinear):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            # Get quantized weights through fake quant
            with torch.no_grad():
                w_q = m.weight_fake_quant(m.weight)

            # Create regular linear with quantized weights
            new_linear = nn.Linear(m.in_features, m.out_features, m.bias is not None)
            new_linear.weight.data.copy_(w_q)
            if m.bias is not None:
                new_linear.bias.data.copy_(m.bias)

            setattr(parent, parts[-1], new_linear)

    return model


def count_quantized_layers(model):
    """Count quantized vs regular linear layers."""
    quantized = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    regular = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return {'quantized': quantized, 'regular': regular}
