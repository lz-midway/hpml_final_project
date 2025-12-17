# these are layers for CNN


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

class BinaryLinear(nn.Module):
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
        self.activation = self._get_act(activation)
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ---------- helpers --------------------------------------------------- #
    @staticmethod
    def _get_act(name):
        if name is None or name == "linear":
            return lambda x: x
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        if name == "softmax":
            return lambda x: F.softmax(x, dim=-1)
        raise ValueError(f"Unknown activation {name}")

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
        # Straight-Through Estimator for gradients
        if self.training:
            w_eff = self.weight + (w_q - self.weight).detach()
            b_eff = None
            if self.bias is not None:
                b_q = self._binarise(self.bias)
                b_eff = self.bias + (b_q - self.bias).detach()
        else:  # inference: use discrete params
            w_eff = w_q
            b_eff = self._binarise(self.bias) if self.bias is not None else None

        # Affine transform
        z = F.linear(x, w_eff, b_eff)        # ... shape (batch, *, out_features)

        # Per-example normalisation  (last dimension)
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True, unbiased=False)
        z = (z - mean) / (std + self.eps) * self.scale

        # Activation
        return self.activation(z)


class BEMB(nn.Module):
    """
    Binary Normalised Embedding Layer (Algorithm 4).
    Produces token + positional embeddings with binary parameters.
    """

    def __init__(self,
                 max_len: int,
                 vocab_size: int,
                 emb_dim: int):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        # Binary-normalised fully-connected layers for token & position lookup
        self.token_fc = BinaryLinear(vocab_size, emb_dim, bias=True, activation="linear")
        self.pos_fc   = BinaryLinear(max_len,    emb_dim, bias=True, activation="linear")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : LongTensor of shape (batch, seq_len) with token indices
        returns: Tensor of shape (batch, seq_len, emb_dim)
        """
        B, L = seq.shape
        device = seq.device

        # ---- Position embeddings ---- #
        pos_idx = torch.arange(L, device=device)                   # (L,)
        pos_onehot = F.one_hot(pos_idx, num_classes=self.max_len).float()  # (L, max_len)
        pos_emb = self.pos_fc(pos_onehot)                          # (L, emb_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)           # (B, L, emb_dim)

        # ---- Token embeddings ---- #
        tk_onehot = F.one_hot(seq, num_classes=self.vocab_size).float()    # (B, L, vocab_size)
        tk_emb = self.token_fc(tk_onehot)                          # (B, L, emb_dim)

        # ---- Combine ---- #
        tk_pos_emb = tk_emb + pos_emb                              # (B, L, emb_dim)
        return tk_pos_emb



class BNCVL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation: str = None, padding: str | None = "same",  scale_init: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 32-bit float convolution weights and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # # adding scaling factor for the one before training computation
        # self.scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1) * scale_init)

        # make scale broadcastable: shape (1, out_channels, 1, 1)
        self.scale = nn.Parameter(torch.full((1, out_channels, 1, 1), float(scale_init)))
        self.eps = 1e-8

        self.activation = self._get_act(activation)
        # if padding == "same":
        #     self.padding = kernel_size // 2
        # else:
        self.padding = 0 if padding is None else padding

        # # use BatchNorm2d for normalization instead of the original per sample one
        # self.bn = nn.BatchNorm2d(out_channels)

    def no_grad(self, x):
        """Stops gradient flow (Straight-Through Estimator)."""
        return x.detach()
    
    def quantize(self, x):
        """
        Binary quantization based on the mean of x.
        1 if x > mean(x), else 0.
        """
        threshold = x.mean()
        return (x > threshold).float()
    
    @staticmethod
    def _get_act(name):
        if name is None or name == "linear":
            return lambda x: x
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        raise ValueError(f"Unknown activation {name}")
    
    def normalize(self, z):
        """Normalize each feature map in a sample."""
       # per-sample, per-channel spatial normalization:
        mean = z.mean(dim=(2, 3), keepdim=True)   # shape (B, C, 1, 1)
        std  = z.std(dim=(2, 3), keepdim=True) # + self.eps  # shape (B, C, 1, 1)
        # normalize then apply channel scale (broadcasts over batch/spatial)
        return (z - mean) / std * self.scale  # self.scale shape (1,C,1,1) broadcasts correctly
    
    def forward(self, x):
        if self.training:
            # Quantization during training (STE approximation)
            w_q = self.weight + self.no_grad(self.quantize(self.weight) - self.weight)
            b_q = self.bias + self.no_grad(self.quantize(self.bias) - self.bias)
        else:
            # Quantization during inference
            w_q = self.quantize(self.weight)
            b_q = self.quantize(self.bias)

        # Perform convolution with quantized weights
        z = F.conv2d(x, w_q, b_q, stride=1, padding=self.padding)

        # Normalize and activate
        z = self.normalize(z)
        return self.activation(z)

    # def forward(self, x):
    #     w_bin = self.quantize(self.weight)

    #     w_scaled = w_bin * self.scale

    #     if self.training:
    #         # Quantization during training (STE approximation)
    #         w_q = self.weight + self.no_grad(w_scaled - self.weight)
    #         b_q = self.bias + self.no_grad(self.quantize(self.bias) - self.bias)
    #     else:
    #         # Quantization during inference
    #         w_q = w_scaled
    #         b_q = self.quantize(self.bias)

    #     # Perform convolution with quantized weights
    #     z = F.conv2d(x, w_q, b_q, stride=1, padding=self.padding)

    #     # Normalize and activate
    #     # z = self.normalize(z)
    #     z = self.bn(z)
    #     return self.activation(z)


    


class BCVNBlocks(nn.Module):
    def __init__(self, layers_config):
        """
        a block of multiple BNCVL layers + one maxpool2d layer

        layers_config: list of dicts with parameters for each sub-block
        Example:
            [
                {"in_channels": 32, "out_channels": 64, "kernel_size": 2, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": 2, "activation": "relu", "padding": "same"}
            ]
        """
        super().__init__()
        blocks = []
        for cfg in layers_config:
            block = BNCVL(**cfg)
            blocks.append(block)
        blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.sequence = nn.Sequential(*blocks)

    def forward(self, x):
        return self.sequence(x)


class ResidualBCVNBlock(nn.Module):
    def __init__(self, layers_config):
        """
        layers_config: list of exactly TWO dicts for BNCVL layers.
        Includes residual connections
        """
        super().__init__()
        assert len(layers_config) == 2, "Residual block requires exactly 2 layers"

        cfg1, cfg2 = layers_config
        self.conv1 = BNCVL(**cfg1)
        self.conv2 = BNCVL(**cfg2)

        in_c = cfg1["in_channels"]
        out_c = cfg2["out_channels"]

        # 1x1 conv to match channels if needed
        if in_c != out_c:
            self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_c)
        else:
            self.proj = nn.Identity()
            self.proj_bn = None

        # pooling comes after the residual merge
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x
        if not isinstance(self.proj, nn.Identity):
            identity = self.proj(identity)
            identity = self.proj_bn(identity)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.pool(out)
        return out

