import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RealLinear(nn.Module):
    """Standard fully-connected layer with optional LayerNorm and activation."""
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, activation: Optional[str] = None,
                 use_layernorm: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.use_ln = use_layernorm
        self.ln = nn.LayerNorm(out_features) if use_layernorm else None
        self.activation = self._get_act(activation)

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

    def forward(self, x):
        z = self.fc(x)
        if self.use_ln:
            z = self.ln(z)
        return self.activation(z)


class RealCVL(nn.Module):
    """
    Standard conv block:
      Conv2d -> BatchNorm2d -> Activation
    Accepts padding="same" (uses kernel//2).
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation: Optional[str] = "relu", padding: Optional[str] = "same"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else int(kernel_size)

        # compute padding for "same"
        if padding is None:
            pad = 0
        elif padding == "same":
            pad = padding
        else:
            pad = int(padding)

        # conv without bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                              stride=1, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self._get_act(activation)

    @staticmethod
    def _get_act(name):
        if name is None or name == "linear":
            return lambda x: x
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        raise ValueError(f"Unknown activation {name}")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class RealResidualCVNBlock(nn.Module):
    """
    Residual block that mirrors ResidualBCVNBlock:
      x -> conv1 -> conv2 -> add(proj(x)) -> pool
    The conv block here already contains activation (usually relu)
    """
    def __init__(self, layers_config):
        """
        layers_config: list of exactly TWO dicts for RealConvBlock layers.
        use_pool: whether to apply MaxPool2d at the end (True for blocks 1-4, False for block5 alternative)
        """
        super().__init__()
        assert len(layers_config) == 2, "Residual block requires exactly 2 layer configs"
        cfg1, cfg2 = layers_config
        self.conv1 = RealCVL(**cfg1)
        self.conv2 = RealCVL(**cfg2)

        in_c = cfg1["in_channels"]
        out_c = cfg2["out_channels"]

        # 1x1 projection to match channels if needed
        if in_c != out_c:
            self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_c)
        else:
            self.proj = nn.Identity()
            self.proj_bn = None

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
