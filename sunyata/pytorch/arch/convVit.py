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
    kernel_size: int = 3

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


class Attention(nn.Module):
    """
    Cross Attention modified from Perceiver.

    """
    def __init__(self, query_dim, 
                 kernel_size, 
                 context_dim=None, 
                 heads=8, 
                 dim_head=64, 
                 scale=None, 
                 use_conv=True, 
                 with_cls_token=True, 
                 dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (inner_dim == query_dim and heads == 1)

        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5 if scale is None else scale 
        self.heads = heads
        self.with_cls_token = with_cls_token

        self.conv_vk = self.conv_block(hidden_dim=context_dim,out_dim=context_dim*2, kernel_size=kernel_size, use_conv=use_conv)
        # self.conv_v = self.conv_block(hidden_dim=context_dim, kernel_size=kernel_size, use_conv=use_conv)
        self.reduce_b = self.reduce_block(hidden_dim=context_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()

    def conv_block(self, hidden_dim, out_dim, kernel_size, use_conv=True):
        if use_conv:
            conv = nn.Sequential(
                nn.Conv2d(hidden_dim, out_dim, kernel_size, groups=hidden_dim, padding='same'),
                nn.BatchNorm2d(out_dim),
                # Rearrange('b c h w -> b (h w) c'),
            )
        else:
            conv = None

        return conv
    
    def reduce_block(self, hidden_dim):
        reduce = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim)
        )
        return reduce
    
    def forward_conv(self, x):
        res = x
        if self.conv_vk is not None:
            x = self.conv_vk(x)
            k, v = x.chunk(2, dim=1)
            k = rearrange(k, 'b c h w -> b (h w) c')
            v = rearrange(x, 'b c h w -> b (h w) c')
            x = self.reduce_b(x) + res
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')
            v = rearrange(x, 'b c h w -> b (h w) c')
        # if self.conv_v is not None:
        #     v = self.conv_v(x)
        #     v = rearrange(x, 'b c h w -> b (h w) c')
            
        # else:

        return x, k, v

    def forward(self, latent, context = None):
        if self.conv_vk is not None:
            x, k, v = self.forward_conv(context)

        h = self.heads

        q = self.to_q(latent)
        context = context if context is not None else latent
        # k, v = self.to_kv(context).chunk(2, dim=-1)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q,k,v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out, x
    

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
                              kernel_size=cfg.kernel_size, 
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
        res = latent
        latent = self.norm1(latent)
        latent, x = self.attn(latent, context)
        latent = res + latent

        # latent = latent + self.attn(self.norm1(latent), context)
        latent = latent + self.ff(self.norm2(latent))
        return latent, x
    

class patch_embed(nn.Module):
    def __init__(self, in_dim=3, dim=256, patch_size=7, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.patch_size = patch_size

        self.norm_layer = norm_layer

        self.conv = nn.Conv2d(in_dim, dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = norm_layer(dim) if norm_layer == nn.BatchNorm2d else nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape

        if self.norm_layer != nn.BatchNorm2d:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            x = self.norm(x)

        return x
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ConvTransformer(ClassifierModule):
    def __init__(self, cfg: ViTCfg):
        super().__init__(cfg)
        self.save_hyperparameters("cfg")

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

        self.patch_embed = patch_embed(in_dim=3,
                                       dim=cfg.hidden_dim,
                                       patch_size=cfg.patch_size,
                                       )
        with_cls_token = False
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        else:
            self.cls_token = None
        
        layers = []
        for _ in range(cfg.num_layers):
             layers.append(
             TransformerLayer(cfg.transformer)
             )

        self.layers = nn.ModuleList(layers)
        
        # classification head
        self.emb_dropout = nn.Dropout(cfg.emb_dropout)
        self.final_ln = nn.LayerNorm(cfg.hidden_dim)
        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )        

        self.cfg = cfg
        
        self.latent = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))

    def forward(self, x):
        B, _, _, _ = x.shape
        latent = repeat(self.latent, '1 1 d -> b 1 d', b = B)

        x = self.patch_embed(x)

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            latent, context = layer(latent, x)
            x = context + x

        latent = self.final_ln(latent)
        x_chosen = latent.mean(dim = 1)
        logits = self.mlp_head(x_chosen)

        return logits




