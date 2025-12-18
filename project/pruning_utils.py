
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from typing import Optional, List, Tuple
import pprint

# -----------------------
# PRUNING UTILITIES (structured fake pruning)
# -----------------------

def collect_structured_prunable_params(model: nn.Module, include_bias: bool = False) -> List[Tuple[nn.Module, str]]:
    params = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            params.append((m, "weight"))
            if include_bias and m.bias is not None:
                params.append((m, "bias"))
            continue
        if isinstance(m, nn.Linear):
            params.append((m, "weight"))
            if include_bias and m.bias is not None:
                params.append((m, "bias"))
            continue
        # wrapper conv (RealCVL)
        if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
            params.append((m.conv, "weight"))
            if include_bias and getattr(m.conv, "bias", None) is not None:
                params.append((m.conv, "bias"))
            continue
        # wrapper linear (RealLinear)
        if hasattr(m, "fc") and isinstance(getattr(m, "fc"), nn.Linear):
            params.append((m.fc, "weight"))
            if include_bias and getattr(m.fc, "bias", None) is not None:
                params.append((m.fc, "bias"))
            continue
        # custom modules that directly expose Parameter named weight/bias
        if hasattr(m, "weight") and isinstance(getattr(m, "weight"), nn.Parameter):
            w = getattr(m, "weight")
            if w.dim() in (2, 4):
                params.append((m, "weight"))
        if include_bias and hasattr(m, "bias") and isinstance(getattr(m, "bias"), nn.Parameter):
            params.append((m, "bias"))
    # deduplicate
    seen = set()
    uniq = []
    for mod, name in params:
        key = (id(mod), name)
        if key not in seen:
            uniq.append((mod, name)); seen.add(key)
    return uniq

def structured_prune_model(model: nn.Module, amount: float = 0.3, include_bias: bool = False, verbose: bool = True):
    """
    Applies structured pruning masks (ln_structured) across:
      - Conv / BNCVL: output filters (dim=0)
      - Linear / BinaryLinear: output neurons (dim=0)
    This is FAKE (masking) pruning; shapes remain unchanged.
    """
    prunable = collect_structured_prunable_params(model, include_bias=include_bias)
    applied = []
    for mod, name in prunable:
        tensor = getattr(mod, name)
        if tensor is None:
            continue
        # conv-style (out, in, k, k)
        if tensor.dim() == 4:
            prune.ln_structured(mod, name=name, amount=amount, n=2, dim=0)
            applied.append((mod, name))
        # linear-style (out, in)
        elif tensor.dim() == 2:
            prune.ln_structured(mod, name=name, amount=amount, n=2, dim=0)
            applied.append((mod, name))
        else:
            # skip unexpected shapes
            pass
    if verbose:
        print(f"[prune] applied structured ln_structured to {len(applied)} params (amount={amount})")
    return applied

def make_pruning_permanent(model: nn.Module):
    # removes pruning reparam and keeps weights (which are zeroed)
    for m in model.modules():
        for name in ("weight", "bias"):
            if hasattr(m, name):
                try:
                    prune.remove(m, name)
                except Exception:
                    # not pruned or unsupported; ignore
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
