from dataclasses import dataclass
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch.layer.attention import Attention
from sunyata.pytorch_lightning.base import ClassifierModule

from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration

pair = lambda x: x if isinstance(x, tuple) else (x, x)

@dataclass
class MlpCfg(BaseCfg):
    image_size: int = 224  # 224
    patch_size: int = 16  # 16
    hidden_dim: int = 384
    expansion_factor: int = 4
    expansion_factor_token: float = 0.5

    num_layers: int = 12
    num_classes: int = 100
    channels: int = 3
    scale: float = 1.
    type: str = 'standard'


class PLResMlp(ClassifierModule):
    def __init__(self, cfg: MlpCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.cfg = cfg

        image_height, image_width = pair(cfg.image_size)
        assert (image_height % cfg.patch_size) == 0 and (image_width % cfg.patch_size) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // cfg.patch_size) * (image_width // cfg.patch_size)
        wrapper = lambda i, fn: PreAffinePostLayerScale(cfg.hidden_dim, i + 1, fn)

        self.layers = nn.ModuleList([
            nn.Sequential(
                wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
                wrapper(i, nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.expansion_factor),
                    nn.GELU(),
                    nn.Linear(cfg.hidden_dim * cfg.expansion_factor, cfg.hidden_dim)
                ))
            ) for i in range(cfg.num_layers)
        ])
        self.layers = nn.ModuleList([nn.Sequential(*self.layers)])

        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=cfg.patch_size, p2=cfg.patch_size),
            nn.Linear((cfg.patch_size ** 2) * cfg.channels, cfg.hidden_dim)
        )

        self.digup = nn.Sequential(
            Affine(cfg.hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.digup(x)
        return x
    

class PLAttnResMlp(ClassifierModule):
    def __init__(self, cfg: MlpCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.cfg = cfg

        image_height, image_width = pair(cfg.image_size)
        assert (image_height % cfg.patch_size) == 0 and (image_width % cfg.patch_size) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // cfg.patch_size) * (image_width // cfg.patch_size)
        wrapper = lambda i, fn: PreAffinePostLayerScale(cfg.hidden_dim, i + 1, fn)

        self.layers = nn.ModuleList([
            nn.Sequential(
                wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
                wrapper(i, nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.expansion_factor),
                    nn.GELU(),
                    nn.Linear(cfg.hidden_dim * cfg.expansion_factor, cfg.hidden_dim)
                ))
            ) for i in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=cfg.patch_size, p2=cfg.patch_size),
            nn.Linear((cfg.patch_size ** 2) * cfg.channels, cfg.hidden_dim)
        )

        self.digup = nn.Sequential(
            Affine(cfg.hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )
        
        self.attn = Attention(query_dim=cfg.hidden_dim,
                              context_dim=cfg.hidden_dim,
                              heads=1,
                              dim_head=cfg.hidden_dim,
                              scale=cfg.scale)
        
        self.latent = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = B)

        x = self.embed(x)
        latent = latent + self.attn(latent, x)
        latent = self.norm(latent)

        for layer in self.layers:
            x = layer(x)
            latent = latent + self.attn(latent, x)
            latent = self.norm(latent)

        logits = self.digup(latent)
        return logits




class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

