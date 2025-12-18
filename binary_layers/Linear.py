import time
import torch
import torch.nn as nn
import torch.nn.functional as F



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
                 eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()
        self.eps = eps

        self.scale = 1/4
        
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


        