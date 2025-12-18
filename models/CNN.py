import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Type
import binary_layers

@dataclass
class Conv2dConfig:
    # constructors
    proj_1:  type = nn.Conv2d
    proj_2:  type = nn.Conv2d
    proj_out: type = nn.Conv2d

    channels:     int  = 8
    out_channels: int  = 8
    kernel_size:  int  = 3
    pool:         bool = False      

@dataclass
class ConnectedConfig:
    proj:     type = nn.Linear
    in_dim:   int  = 256
    out_dim:  int  = 256


@dataclass
class CNNConfig:
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
