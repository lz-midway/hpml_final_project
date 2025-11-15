import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F
import math

class Linear(nn.Module):
    """
    Binary-Normalised Fully-Connected Layer (BNFCL).

    All parameters are stored in fp32 but binarised (0/1) on the forward path.
    After the affine transform, features are normalised to zero-mean/unit-std
    per example, then an optional activation is applied.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 activation: str | None = None,
                 eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.eps = eps

    @staticmethod
    def _binarise(w: torch.Tensor):
        """Binary quantisation: 1 if w > mean(w) else 0 (per tensor)."""
        thresh = w.mean()
        return (w > thresh).float()

    # ---------- forward ---------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, *, in_features)
        returns: shape (batch, *, out_features)
        """
        w_q = self._binarise(self.weight)
        
        if self.training:
            w_eff = self.weight + (w_q - self.weight).detach()
            b_eff = None
            if self.bias is not None:
                b_q = self._binarise(self.bias)
                b_eff = self.bias + (b_q - self.bias).detach()
        else:
            w_eff = w_q
            b_eff = self._binarise(self.bias) if self.bias is not None else None

        # Affine transform
        z = F.linear(x, w_eff, b_eff)        # ... shape (batch, *, out_features)

        # Per-example normalisation  (last dimension)
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True, unbiased=False)
        z = (z - mean) / (std + self.eps)

        return z