from dataclasses import dataclass
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import SE, BaseCfg, Residual


@dataclass
class ConvMixerCfg(BaseCfg):
    num_layers: int = 8
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    squeeze_factor: int = 4

    layer_norm_zero_init: bool = True
    skip_connection: bool = True
    eca_kernel_size: int = 3


class block(nn.Module):
    def __init__(self, hidden_dim, kernel_size, drop_rate=0.):
        super(block, self).__init__()
        # self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim,  padding='same')
        # self.act = nn.GELU() 
        # self.norm = nn.BatchNorm2d(hidden_dim)
        self.se = SE(hidden_dim)

        
    def forward(self, x):
        # x = self.conv(x)
        # x = self.act(x)
        # x = self.norm(x)
        x = self.se(x)
        return x
    
class mlp(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, drop_rate=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, mlp_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_dim, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int,hidden_dim: int,  patch_size: int):
        super().__init__()
        # self.image_size = image_size
        self.patch_size = patch_size

        self.embed = self.embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.image_size[0] and W == self.image_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.embed(x)
        return x
    
class formerblock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, drop_rate=0.):
        super().__init__()
        mlp_dim = hidden_dim*4
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout(drop_rate)
        self.block = block(hidden_dim, kernel_size, drop_rate)
        self.mlp = mlp(hidden_dim, mlp_dim, drop_rate)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        # x = self.norm1(x + self.drop(self.block(x)))    
        # x = self.norm2(x + self.drop(self.mlp(x)))
        x = x + self.block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.drop(x)

        return x
    
class convformer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = [64, 128, 256, 512]
        self.patch_size = [4, 2, 2, 2]
        self.depth = [2, 2, 6, 2]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim[0],
                                      patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(
                PatchEmbed(in_channels=self.hidden_dim[i], 
                           hidden_dim=self.hidden_dim[i+1], 
                           patch_size=self.patch_size[i+1]))
        self.formerblock = nn.ModuleList()
        for i in range(4):
            formerblock_list = nn.Sequential(
                *[formerblock(self.hidden_dim[i], cfg.kernel_size, cfg.drop_rate) for _ in range(self.depth[i])]
            )
            self.formerblock.append(formerblock_list)
        
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim[-1]),
        )
        self.fc = nn.Linear(self.hidden_dim[-1], cfg.num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.formerblock[i](x)
        x = self.neck(x)
        x = self.fc(x)
        return x