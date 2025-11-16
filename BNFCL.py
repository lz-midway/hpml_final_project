import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_binary(x):
    """
    Quantization function:
        pb = 1 if p > mean(p)
             0 otherwise
    """
    mean_val = x.mean()
    return (x > mean_val).float()

def no_grad_quantize(x):
    """
    Implements: W_q = W + NoGradient(Quant(W) - W)
    """
    return x + (quantize_binary(x) - x).detach()

def normalize_features(z, eps=1e-8):
    """
    z = (z - mean) / std
    """
    mean = z.mean(dim=-1, keepdim=True)
    std = z.std(dim=-1, keepdim=True) + eps
    return (z - mean) / std

# --- Binary Normalized Fully Connected Layer ---
class BinaryNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=None, trainable=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(out_features))
        self.trainable = trainable
        self.activation = activation

    def forward(self, x):
        if self.trainable:
            #training phase
            W_q = no_grad_quantize(self.W)
            b_q = no_grad_quantize(self.b)
        else:
            #inference phase
            W_q = quantize_binary(self.W)
            b_q = quantize_binary(self.b)

        z = F.linear(x, W_q, b_q)
        z = normalize_features(z)
        if self.activation is not None:
            z = self.activation(z)

        return z
