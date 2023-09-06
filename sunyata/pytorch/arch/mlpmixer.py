from dataclasses import dataclass
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch.layer.attention import Attention
from sunyata.pytorch_lightning.base import ClassifierModule
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

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
    dropout: float = 0. 

    scale: float = 1.
    type: str = 'standard'



class PlMlpMixer(ClassifierModule):
    def __init__(self, cfg:MlpCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.cfg = cfg

        image_h, image_w = pair(cfg.image_size)
        assert (image_h % cfg.patch_size) == 0 and (image_w % cfg.patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // cfg.patch_size) * (image_w // cfg.patch_size)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.layers = nn.ModuleList([
            nn.Sequential(
                PreNormResidual(cfg.hidden_dim, FeedForward(num_patches, cfg.expansion_factor, cfg.dropout, chan_first)),
                PreNormResidual(cfg.hidden_dim, FeedForward(cfg.hidden_dim, cfg.expansion_factor_token, cfg.dropout, chan_last))
            ) for _ in range(cfg.num_layers)
        ])
        self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to one layer

        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=cfg.patch_size, p2=cfg.patch_size),
            nn.Linear((cfg.patch_size ** 2) * cfg.channels, cfg.hidden_dim)
        )

        self.digup = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.digup(x)
        return x

        
class PlAttnMlpMixer(ClassifierModule):
    def __init__(self, cfg:MlpCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")
        self.cfg = cfg

        image_h, image_w = pair(cfg.image_size)
        assert (image_h % cfg.patch_size) == 0 and (image_w % cfg.patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // cfg.patch_size) * (image_w // cfg.patch_size)

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.layers = nn.ModuleList([
            nn.Sequential(
                PreNormResidual(cfg.hidden_dim, FeedForward(num_patches, cfg.expansion_factor, cfg.dropout, chan_first)),
                PreNormResidual(cfg.hidden_dim, FeedForward(cfg.hidden_dim, cfg.expansion_factor_token, cfg.dropout, chan_last))
            ) for _ in range(cfg.num_layers)
        ])
        # self.layers = nn.ModuleList([nn.Sequential(*self.layers)])  # to one layer

        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=cfg.patch_size, p2=cfg.patch_size),
            nn.Linear((cfg.patch_size ** 2) * cfg.channels, cfg.hidden_dim)
        )

        self.digup = nn.Sequential(
            # nn.LayerNorm(cfg.hidden_dim),
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


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )