from dataclasses import dataclass
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn * u
    

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm2d(d_model, eps=7e-5)
        self.spatial_gating_unit = LKA(d_model)
        self.activation2 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(d_model, eps=7e-5)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.spatial_gating_unit(x)
        x = self.activation2(x)
        x = self.norm2(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    

class attention(nn.Sequential):
    def __init__(self, hidden_dim: int):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            LKA(hidden_dim),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            nn.Conv2d(hidden_dim, hidden_dim, 1),  
        )   