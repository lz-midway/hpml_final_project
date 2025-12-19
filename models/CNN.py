import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Type
import binary_layers

@dataclass
class Conv2dConfig:
    """
    Configuration for a single 2D Convolutional Residual Block.

    Attributes:
        proj_1 (type): Class type for the first convolution (e.g., nn.Conv2d or binary_layers.Conv2d).
        proj_2 (type): Class type for the second convolution.
        proj_out (type): Class type for the skip connection projection (if channels change).
        channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        pool (bool): Whether to apply MaxPooling at the end of the block.
    """
    proj_1:  type = nn.Conv2d
    proj_2:  type = nn.Conv2d
    proj_out: type = nn.Conv2d

    channels:     int  = 8
    out_channels: int  = 8
    kernel_size:  int  = 3
    pool:         bool = False      

@dataclass
class ConnectedConfig:
    """
    Configuration for a fully connected (linear) layer.

    Attributes:
        proj (type): Class type for the linear layer (e.g., nn.Linear or binary_layers.Linear).
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
    """
    proj:     type = nn.Linear
    in_dim:   int  = 256
    out_dim:  int  = 256


@dataclass
class CNNConfig:
    """
    Master configuration for the entire CNN architecture.
    
    Attributes:
        ConvLayers (list): A list of Conv2dConfig objects defining the feature extraction backbone.
        ConnectedLayers (list): A list of ConnectedConfig objects defining the classification head.
    """
    ConvLayers:     list = field(default_factory=lambda: [
        Conv2dConfig(channels=32,  out_channels=64,  kernel_size=3, pool=True),
        Conv2dConfig(channels=64,  out_channels=64,  kernel_size=3, pool=False),
        Conv2dConfig(channels=64,  out_channels=128, kernel_size=3, pool=True),
        Conv2dConfig(channels=128, out_channels=256, kernel_size=3, pool=True),
    ])
    ConnectedLayers: list = field(default_factory=lambda: [
        ConnectedConfig(in_dim=256, out_dim=256),
        ConnectedConfig(in_dim=256, out_dim=256)
    ])



class ResidualBlock(nn.Module):
    """
    A generic Residual Block containing two convolution operations, normalization, 
    activations, and a skip connection.
    
    Structure:
        Input -> Conv1 -> BN -> ReLU -> Conv2 -> BN -> ReLU -> (+ Identity) -> Pool -> Output
    """
    def __init__(self, cfg: Conv2dConfig):
        super().__init__()
        
 
        self.conv1 = cfg.proj_1(cfg.channels, cfg.out_channels, kernel_size=cfg.kernel_size, padding='same')
        self.norm1 = nn.BatchNorm2d(cfg.out_channels) if isinstance(self.conv1, nn.Conv2d) else nn.Identity()

        # --- Conv 2 ---
        self.conv2 = cfg.proj_2(cfg.out_channels, cfg.out_channels, kernel_size=cfg.kernel_size, padding='same')
        self.norm2 = nn.BatchNorm2d(cfg.out_channels) if isinstance(self.conv2, nn.Conv2d) else nn.Identity()

        
        if cfg.channels != cfg.out_channels:
            self.proj = cfg.proj_out(cfg.channels, cfg.out_channels, kernel_size=1, padding=0, bias=False)
            self.proj_bn = nn.BatchNorm2d(cfg.out_channels) if isinstance(self.proj, nn.Conv2d) else nn.Identity()
        else:
            self.proj = nn.Identity()
            self.proj_bn = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2) if cfg.pool else nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out) 

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        
        if not isinstance(self.proj, nn.Identity):
            identity = self.proj(identity)
            identity = self.proj_bn(identity)

        out = out + identity
        out = self.pool(out)
        return out


class CNN(nn.Module):
    """
    The main Convolutional Neural Network class composed of stacked ResidualBlocks 
    and fully connected layers.
    """
    def __init__(self,
                 config: CNNConfig = CNNConfig(),  
                 img_channels: int = 3,
                 num_classes: int = 101):
        super().__init__()

        
        first_block_in = config.ConvLayers[0].channels
        if img_channels != first_block_in:
            self.adapter = nn.Conv2d(img_channels, first_block_in, kernel_size=3, padding='same')
        else:
            self.adapter = nn.Identity()

        
        self.conv_blocks = nn.ModuleList(
            [ResidualBlock(cfg) for cfg in config.ConvLayers]
        )

        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_layers = nn.ModuleList(
            [cfg.proj(cfg.in_dim, cfg.out_dim) for cfg in config.ConnectedLayers]
        )

        
        last_dim = config.ConnectedLayers[-1].out_dim
        final_proj_type = config.ConnectedLayers[-1].proj
        self.head = final_proj_type(last_dim, num_classes)

    def forward(self, x):
        x = self.adapter(x)

        # Feature Extraction
        for block in self.conv_blocks:
            x = block(x)

        # Global Average Pooling & Flatten
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        for layer in self.linear_layers:
            x = layer(x)
            x = F.relu(x) # Manual activation

        x = self.head(x)
        return x

    def update_cache(self):
        self.apply(self._update_cache)
    def _update_cache(self, module):
        if isinstance(module, binary_layers.Linear):
            module.update_cache()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
