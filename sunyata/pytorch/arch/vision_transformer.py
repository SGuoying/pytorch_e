import torch
from torch import nn
from dataclasses import dataclass

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from sunyata.pytorch.arch.base import BaseCfg


@dataclass
class ViTCfg(BaseCfg):
    hidden_dim: int = 128
    image_size: int = 224  # 224
    patch_size: int = 8  # 16
    num_classes: int = 200
    num_layers: int = 8
    num_heads: int = 8

    expanded_dim: int = None
    head_dim: int = None   

    pool: str = 'cls' # or 'mean'
    channels: int = 3

    emb_dropout: float = 0. 

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attention = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.feedforward = FeedForward(dim, mlp_dim, dropout = dropout)

        self.attn_layernorm = nn.LayerNorm(dim)
        self.ff_layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn_layernorm(x + self.attention(x))
        x = self.ff_layernorm(x + self.feedforward(x))
        return x

class ViT(nn.Module):
    def __init__(self, cfg: ViTCfg):
        super().__init__(cfg)
        image_height, image_width = pair(cfg.image_size)
        patch_height, patch_width = pair(cfg.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = cfg.channels * patch_height * patch_width
        assert cfg.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
        self.dropout = nn.Dropout(cfg.emb_dropout)

        # self.transformer = Transformer(cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.head_dim, cfg.expanded_dim, cfg.emb_dropout)

        self.transformer = nn.Sequential(*[
            transformer(cfg.hidden_dim, cfg.num_heads, cfg.head_dim, cfg.expanded_dim, cfg.emb_dropout)
              for _ in range(cfg.num_layers)])
        self.pool = cfg.pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )
        self.cfg = cfg

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    

class bayes_ViT(ViT):
    def __init__(self, cfg: ViTCfg):
        super().__init__(cfg)
        
        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.mlp_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        batch_size, num_patches, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_patches + 1)]
        x = self.dropout(x)
        logits = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.transformer(x)
        for transformer in self.transformer:
            x = transformer(x)
            logits = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] + logits
            logits = self.logits_layer_norm(logits)

        return self.mlp_head(logits)

