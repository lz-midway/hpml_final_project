import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple

# -------------------------
# Utilities for pruning (compiled & non-compiled)
# -------------------------

def _unwrap_module(m: nn.Module) -> nn.Module:
    """
    Return the underlying module if compiled (has _orig_mod), else return itself.
    Works recursively for multiple layers of wrapping.
    """
    while hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

def collect_structured_prunable_params(model: nn.Module, include_bias: bool = False) -> List[Tuple[nn.Module, str]]:
    params = []
    for m in model.modules():
        m_real = _unwrap_module(m)
        # Conv2d
        if isinstance(m_real, nn.Conv2d):
            params.append((m_real, "weight"))
            if include_bias and m_real.bias is not None:
                params.append((m_real, "bias"))
            continue
        # Linear
        if isinstance(m_real, nn.Linear):
            params.append((m_real, "weight"))
            if include_bias and m_real.bias is not None:
                params.append((m_real, "bias"))
            continue
        # wrapper conv (RealCVL)
        if hasattr(m_real, "conv") and isinstance(getattr(m_real, "conv"), nn.Conv2d):
            params.append((m_real.conv, "weight"))
            if include_bias and getattr(m_real.conv, "bias", None) is not None:
                params.append((m_real.conv, "bias"))
            continue
        # wrapper linear (RealLinear)
        if hasattr(m_real, "fc") and isinstance(getattr(m_real, "fc"), nn.Linear):
            params.append((m_real.fc, "weight"))
            if include_bias and getattr(m_real.fc, "bias", None) is not None:
                params.append((m_real.fc, "bias"))
            continue
        # custom modules with weight/bias directly exposed
        for name in ("weight", "bias"):
            if hasattr(m_real, name) and isinstance(getattr(m_real, name), nn.Parameter):
                w = getattr(m_real, name)
                if w is not None and w.dim() in (2, 4):
                    params.append((m_real, name))

    # deduplicate
    seen = set()
    uniq = []
    for mod, name in params:
        key = (id(mod), name)
        if key not in seen:
            uniq.append((mod, name))
            seen.add(key)
    return uniq

def structured_prune_model(model: nn.Module, amount: float = 0.3, include_bias: bool = False, verbose: bool = True):
    prunable = collect_structured_prunable_params(model, include_bias=include_bias)
    applied = []
    for mod, name in prunable:
        tensor = getattr(mod, name)
        if tensor is None:
            continue
        # conv-style (out, in, k, k) or linear-style (out, in)
        if tensor.dim() in (2, 4):
            prune.ln_structured(mod, name=name, amount=amount, n=2, dim=0)
            applied.append((mod, name))
    if verbose:
        print(f"[prune] applied structured ln_structured to {len(applied)} params (amount={amount})")
    return applied

def make_pruning_permanent(model: nn.Module):
    for m in model.modules():
        m_real = _unwrap_module(m)
        for name in ("weight", "bias"):
            if hasattr(m_real, name):
                try:
                    prune.remove(m_real, name)
                except Exception:
                    pass

def model_sparsity_report(model: nn.Module):
    report = {}
    total = 0
    zeros = 0
    for name, p in model.named_parameters():
        if p is None:
            continue
        z = int((p == 0).sum().item())
        n = p.numel()
        report[name] = {"shape": tuple(p.shape), "sparsity": z / n}
        total += n
        zeros += z
    report["_global"] = {"total_params": total, "zero_params": zeros, "global_sparsity": zeros / total}
    return report
