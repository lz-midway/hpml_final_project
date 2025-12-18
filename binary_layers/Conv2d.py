import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding: str | None = "same",  scale_init: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.full((1, out_channels, 1, 1), float(scale_init)))
        self.eps = 1e-8

        self.padding = 0 if padding is None else padding

    
    def quantize(self, x):
        """
        Binary quantization based on the mean of x.
        1 if x > mean(x), else 0.
        """
        threshold = x.mean()
        return (x > threshold).float()
    
    def normalize(self, z):
        """Normalize each feature map in a sample."""
       # per-sample, per-channel spatial normalization:
        mean = z.mean(dim=(2, 3), keepdim=True)   # shape (B, C, 1, 1)
        std  = z.std(dim=(2, 3), keepdim=True) # + self.eps  # shape (B, C, 1, 1)
        # normalize then apply channel scale (broadcasts over batch/spatial)
        return (z - mean) / std * self.scale  # self.scale shape (1,C,1,1) broadcasts correctly
    
    def forward(self, x):
        if self.training:
            w_q = self.weight + (self.quantize(self.weight) - self.weight).detach()
            b_q = self.bias + (self.quantize(self.bias) - self.bias).detach()
        else:
            # Quantization during inference
            w_q = self.quantize(self.weight)
            b_q = self.quantize(self.bias)

        z = F.conv2d(x, w_q, b_q, stride=1, padding=self.padding)

        z = self.normalize(z)
        return z

        