from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from sunyata.pytorch.layer.attention import SelfAttention, Attention


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


class TransformerLayer(nn.Module):
    def __init__(self, cfg:TransformerCfg):
        super().__init__()
        assert cfg.hidden_dim % cfg.num_heads == 0
        dim_head = cfg.hidden_dim // cfg.num_heads
        self.attention = Attention(query_dim=cfg.hidden_dim, context_dim=cfg.hidden_dim, heads=cfg.num_heads,
                                    dim_head=dim_head, dropout=cfg.attn_dropout, scale = cfg.attn_scale)

        expanded_dim = cfg.expansion * cfg.hidden_dim
        self.feed_forward = FeedForward(cfg.hidden_dim, expanded_dim, cfg.ff_act_nn, 
                                        cfg.ff_dropout)
        

        self.attn_layernorm = nn.LayerNorm(cfg.hidden_dim)
        self.ff_layernorm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x):
        x = x + self.attn_layernorm(self.attention(x))
        x = x + self.ff_layernorm(self.feed_forward(x))
        return x
        

class RevTransformerLayer(TransformerLayer):
    def __init__(self, cfg:TransformerCfg):
        super().__init__(cfg)

    def forward1(self, x1:torch.Tensor, x2:torch.Tensor):
        y2 = x2 + self.attention(self.attn_layernorm(x1))
        y1 = x1 + self.feed_forward(self.ff_layernorm(y2))
        return y1, y2

    def forward2(self, y1:torch.Tensor, y2:torch.Tensor):
        x1 = y1 - self.feed_forward(self.ff_layernorm(y2))
        x2 = y2 - self.attention(self.attn_layernorm(x1))
        return x1, x2


class TransformerLayerPreNorm(TransformerLayer):
    def __init__(self, cfg:TransformerCfg):
        super().__init__(cfg)

    def forward(self, x):
        x = x + self.attention(self.attn_layernorm(x))
        x = x + self.feed_forward(self.ff_layernorm(x))
        return x


class TransformerLayerPostNorm(TransformerLayer):
    def __init__(self, cfg:TransformerCfg):
        super().__init__(cfg)

    def forward(self, x):
        x = self.attn_layernorm(x + self.attention(x))
        x = self.ff_layernorm(x + self.feed_forward(x))
        return x


class TransformerLayerNoShortcut(TransformerLayer):
    def __init__(self, cfg:TransformerCfg):
        super().__init__(cfg)

    def forward(self, x):
        x1 = self.attn_layernorm(self.attention(x))
        x2 = self.ff_layernorm(self.feed_forward(x1))
        return x1, x2


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