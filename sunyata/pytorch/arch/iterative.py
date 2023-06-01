from dataclasses import dataclass
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from sunyata.pytorch.arch.attentionpool import Attention
from sunyata.pytorch.arch.base import SE, BaseCfg, ConvMixerLayer

@dataclass
class ConvMixerCfg(BaseCfg):
    num_layers: int = 8
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    squeeze_factor: int = 4

    skip_connection: bool = True

    eca_kernel_size: int = 3


class ConvMixerattn(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        # self.digup = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(cfg.hidden_dim, cfg.num_classes)
        # )
        self.attn = Attention(cfg.hidden_dim)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        self.cfg = cfg
        logits = torch.zeros(1, cfg.hidden_dim)
        self.register_buffer('logits', logits)

    def forward(self, x):
        # data = rearrange(x, 'b ... d -> b (...) d')
        data = x.flatten(2).transpose(1, 2)
        x = self.embed(x)
        # data = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        logits = self.logits
        for layer in self.layers:
            x = x + layer(x)
            logits = self.attn(x, data) + logits
            logits = self.layer_norm(logits)
        logits = self.fc(logits)
        # x = self.digup(x)
        return logits
    

class ConvMixerattn2(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        # self.digup = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(cfg.hidden_dim, cfg.num_classes)
        # )
        self.attn = Attention(cfg.hidden_dim, context_dim=3)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        self.cfg = cfg
        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        # data = rearrange(x, 'b ... d -> b (...) d')
        data = x.flatten(2).transpose(1, 2)
        x = self.embed(x)
        # data = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        logits = self.attn(x, data)
        # logits = self.logits
        for layer in self.layers:
            x = x + layer(x)
            logits = self.attn(x, data) + logits
            logits = self.layer_norm(logits)
        logits = self.fc(logits)
        # x = self.digup(x)
        return logits