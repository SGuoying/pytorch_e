from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch.layer.attention import Attention
from sunyata.pytorch_lightning.base import ClassifierModule


from sunyata.pytorch.arch.bayes.core import log_bayesian_iteration
from sunyata.pytorch.layer.transformer import (
    TransformerCfg,
    TransformerLayer, 
    TransformerLayerPreNorm,
    TransformerLayerPostNorm,
    )

@dataclass
class ViTCfg(BaseCfg):
    transformer: TransformerCfg = TransformerCfg(
                                    hidden_dim=192,
                                    num_heads=3,
                                    expansion=4,
                                    )

    num_layers: int = 12
    hidden_dim: int = 192
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 100

    posemb: str = 'sincos2d'  # or 'learn'
    pool: str = 'mean' # or 'cls'

    emb_dropout: float = 0. 

    scale: float = 1.

    type: str = 'standard'


class ViT(ClassifierModule):
    def __init__(self, cfg: ViTCfg):
        super().__init__(cfg)

        self.save_hyperparameters("cfg")

        self.layers = nn.Sequential(*[
            TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)
        ])

        image_height, image_width = pair(cfg.image_size)
        patch_height, patch_width = pair(cfg.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 3 * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )

        assert cfg.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = cfg.pool
        assert cfg.posemb in {'learn', 'sincos2d'}, 'posemb type must be either learn or sincos2d'
        self.posemb = cfg.posemb

        if self.posemb == 'learn':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, cfg.hidden_dim))
        elif self.posemb == 'sincos2d':
            pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = cfg.hidden_dim,
            )
            self.register_buffer('pos_embedding', pos_embedding)

        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))

        self.emb_dropout = nn.Dropout(cfg.emb_dropout)
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )        

        self.cfg = cfg

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        x += self.pos_embedding

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.emb_dropout(x)

        x = self.layers(x)

        x = self.final_ln(x)
        x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        logits = self.mlp_head(x_chosen)

        return logits


class ViTPreNorm(ViT):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            TransformerLayerPreNorm(cfg.transformer) for _ in range(cfg.num_layers)
        ])


class ViTPostNorm(ViT):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            TransformerLayerPostNorm(cfg.transformer) for _ in range(cfg.num_layers)
        ])


class IterViTPreNorm(ViTPreNorm):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding

        x = self.emb_dropout(x)

        logits = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        for layer in self.layers:
            x = layer(x)
            x_chosen = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            logits = logits + x_chosen
            logits = self.final_ln(logits)

        logits = self.mlp_head(logits)

        return logits

    

class IterAttnViTPreNorm(ViTPreNorm):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        
        self.latent = nn.Parameter(torch.zeros(1, cfg.hidden_dim))

        self.digup = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim,
                      scale=cfg.scale,
                      )
        
    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding

        x = self.emb_dropout(x)

        latent = latent + self.digup(latent, x)
        latent = self.final_ln(latent)

        for layer in self.layers:
            x = layer(x)

            latent = latent + self.digup(latent, x)
            latent = self.final_ln(latent)

        latent = nn.Flatten()(latent)
        logits = self.mlp_head(latent)

        return logits


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)







