import time
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Custom Linear that binarises its weights and applies per-example norm
# ────────────────────────────────────────────────────────────────────────────

class Linear(torch.nn.Module):
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
                 eps: float = 1e-5,
                 scale_init: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()
        self.eps = eps

        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        
        self.register_buffer('_w_cache', None, persistent=False)
        self.register_buffer('_w_bin', None, persistent=False)
        if bias is not None:
            self.register_buffer('_b_cache', None, persistent=False)
            self.register_buffer('_b_bin', None, persistent=False)

        self.update_cache()
        

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _binarise(w: torch.Tensor):
        """Binary quantisation: 1 if w > mean(w) else 0 (per tensor)."""
        thresh = w.mean()
        return (w > thresh).float()

        
    def update_cache(self):
        device = next(self.parameters()).device
        self._w_bin = self._binarise(self.weight).detach().to(device)
        self._w_cache = (self._w_bin - self.weight).detach().to(device)
        
        if self.bias is not None:
            self._b_bin = self._binarise(self.bias).detach().to(device)  
            self._b_cache = (self._b_bin - self.bias).detach().to(device)
            
    # ---------- forward ---------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, *, in_features)
        returns: shape (batch, *, out_features)
        """
        # Straight-Through Estimator for gradients
        if self.training:
            w_eff = self.weight + self._w_cache
            b_eff = None
            if self.bias is not None:
                b_eff = self.bias + self._b_cache
        else: 
            w_eff = self._w_bin
            b_eff = self._b_bin

        z = F.linear(x, w_eff, b_eff)   

        mean = z.mean(dim=-1).unsqueeze(-1)
        std = z.std(dim=-1, unbiased=False).unsqueeze(-1)
        z = (z - mean) / (std + self.eps) * self.scale

        return z


        


class Linear_fp8(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__()
        
        self.core = te.Linear(in_features, out_features, bias=bias)
        tmp = te.Linear(in_features, out_features, bias=bias).eval()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.eps = eps
        
        object.__setattr__(self, "_binary_layer", tmp)

        # Ensure the binary layer is not saved
        self._binary_layer.weight.requires_grad = False
        if bias:
            self._binary_layer.bias.requires_grad = False

        # Caches for binarized parameters
        self.register_buffer("_w_bin", None, persistent=False)
        self.register_buffer("_b_bin", None, persistent=False)
        self.update_cache()

    @staticmethod
    def _binarise(t: torch.Tensor) -> torch.Tensor:
        return (t > t.mean()).float()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._binary_layer.to(*args, **kwargs)
        return self

    def update_cache(self):
        self._w_bin = self._binarise(self.core.weight.detach())
        self._b_bin = (
            None if self.core.bias is None
            else self._binarise(self.core.bias.detach())
        )

        self._binary_layer.weight.data = self._w_bin
        if self._binary_layer.bias is not None and self._b_bin is not None:
            self._binary_layer.bias.data = self._b_bin

    def forward(self, x):
        
        if self.training:
            normal_output = self.core(x)
            binary_output = self._binary_layer(x)
            z = normal_output + (binary_output - normal_output).detach()
        else:
            z = self._binary_layer(x)

        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True, unbiased=False)
        z = (z - mean) / (std + self.eps) 

        return z * self.scale