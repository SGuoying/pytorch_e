from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from sunyata.pytorch.arch.base import BaseCfg, Residual
from sunyata.pytorch_lightning.base import ClassifierModule
from torchvision.ops import StochasticDepth


@dataclass
class TransformerCfg:
    hidden_dim: int = 128

     # attention
    num_heads: int = 8
    attn_scale: Optional[float] = None
    attn_dropout: float = 0.

    # feed forward
    expansion: int = 4
    ff_dropout: float = 0.
    ff_act_nn: nn.Module = nn.GELU()
    
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
    kernel_size: int = 5

    posemb: str = 'sincos2d'  # or 'learn'
    pool: str = 'mean' # or 'cls'

    emb_dropout: float = 0. 
    conv_drop_rate: float = 0.

    scale: float = 1.

    type: str = 'standard'
    

class ConvMixerLayer(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
            Residual(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
            )),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim), 
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )


class Attention(nn.Module):
    """
    Cross Attention modified from Perceiver.

    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, scale=None, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (inner_dim == query_dim and heads == 1)

        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5 if scale is None else scale # / 10.
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()

    def forward(self, x, context = None):

        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q,k,v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expanded_dim, act_nn:nn.Module=nn.GELU(), dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            act_nn,
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class TransformerLayer(nn.Module):
    def __init__(self, cfg:TransformerCfg):
        super().__init__()
        
        dim_head = cfg.hidden_dim // cfg.num_heads
        self.attn = Attention(query_dim=cfg.hidden_dim, 
                              context_dim=cfg.hidden_dim, 
                              heads=cfg.num_heads,
                              dim_head=dim_head, 
                              scale=cfg.attn_scale, 
                              dropout=cfg.attn_dropout
                              )
        
        expanded_dim = cfg.hidden_dim * cfg.expansion
        self.ff = FeedForward(hidden_dim=cfg.hidden_dim,
                              expanded_dim=expanded_dim,
                              act_nn=cfg.ff_act_nn,
                              dropout=cfg.ff_dropout
                              )
                              
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, latent, context = None):
        latent = latent + self.attn(self.norm1(latent), context)
        latent = latent + self.ff(self.norm2(latent))
        return latent


class ConvVit(ClassifierModule):
    def __init__(self, cfg:ViTCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")

        # patch embedding  block
        self.conv_patch = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim)
        )

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

        # pool and posemb
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

        #  conv block and trans block
        # self.layers = nn.Sequential(
        #     *[TransformerLayer(cfg.transformer) for _ in range(cfg.num_layers)],
        # )

        # self.conv_layers = nn.Sequential(*[
        #     ConvMixerLayer(hidden_dim=cfg.hidden_dim, kernel_size=cfg.kernel_size, drop_rate=cfg.conv_drop_rate) for _ in range(cfg.num_layers)
        # ])
        self.num_layers = cfg.num_layers
        self.conv = nn.ModuleList()
        self.trans = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = ConvMixerLayer(hidden_dim=cfg.hidden_dim, kernel_size=cfg.kernel_size, drop_rate=cfg.conv_drop_rate)
            self.conv.append(conv)
            trans = TransformerLayer(cfg.transformer) 
            self.trans.append(trans)

        # classification head
        self.emb_dropout = nn.Dropout(cfg.emb_dropout)
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )        

        self.cfg = cfg
        
        # self.latent = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))

    def forward(self, x):
        B, _, _, _ = x.shape
        x_trans = x.clone()
        # latent = repeat(self.latent, '1 1 d -> b 1 d', b = B)

        x = self.conv_patch(x)
        latent = self.to_patch_embedding(x_trans)
        latent += self.pos_embedding

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
            x = torch.cat((cls_tokens, latent), dim=1)
        
        for i in range(self.num_layers):
            x = self.conv[i](x)
            context = rearrange(x, 'b c h w -> b (h w) c')
            latent = self.trans[i](latent, context)

        latent = self.final_ln(latent)
        x_chosen = latent.mean(dim = 1) if self.pool == 'mean' else latent[:, 0]
        logits = self.mlp_head(x_chosen)

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

