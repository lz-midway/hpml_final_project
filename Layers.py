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
                 eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()
        self.activation = self._get_act(activation)
        self.eps = eps

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
        z = (z - mean) / (std + self.eps)

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
    def __init__(self, in_channels, out_channels, kernel_size, activation: str = None, padding: str | None = "same" ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 32-bit float convolution weights and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.activation = self._get_act(activation)
        self.padding = 0 if padding is None else padding

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
        mean = z.mean(dim=(1, 2, 3), keepdim=True)
        std = z.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        return (z - mean) / std

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


class CVNBlocks(nn.Module):
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


class BCVNN(nn.Module):
    def __init__(self, image_channels=3, filter_dimension=3, num_classes=101):
        """
        image_channels: number of input channels (3 for RGB)
        filter_dimension: kernel size for all convolution layers
        """
        super().__init__()

        # block 1
        self.block1 = CVNBlocks(
            [
                {"in_channels": image_channels, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 32, "out_channels": 32, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 2
        self.block2 = CVNBlocks(
            [
                {"in_channels": 32, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 3
        self.block3 = CVNBlocks(
            [
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 64, "out_channels": 64, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 4
        self.block4 = CVNBlocks(
            [
                {"in_channels": 64, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
                {"in_channels": 128, "out_channels": 128, "kernel_size": filter_dimension, "activation": "relu", "padding": "same"},
            ]
        )

        # block 5
        self.bncvl9 = BNCVL(in_channels=128, out_channels=256, kernel_size=filter_dimension, activation="relu", padding="same")
        self.bncvl10 = BNCVL(in_channels=256, out_channels=256, kernel_size=filter_dimension, activation="relu", padding="same")
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # block 6
        self.bnfcl1 = BinaryLinear(in_features=256, out_features=256, activation="relu")
        self.bnfcl2 = BinaryLinear(in_features=256, out_features=256, activation="relu")
        self.finallayer = BinaryLinear(in_features=256, out_features=num_classes, activation=None)
    
    def forward(self, x):
        # --- feature extraction ---
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # --- final conv + normalization + activation ---
        x = self.bncvl9(x)
        x = self.bncvl10(x)

        # --- global average pooling ---
        x = self.GAP(x)          # shape: [batch, channels, 1, 1]
        x = torch.flatten(x, 1)  # shape: [batch, channels]

        # --- fully connected binary layers ---
        x = self.bnfcl1(x)
        x = self.bnfcl2(x)
        x = self.finallayer(x)

        return x