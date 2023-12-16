# %%
from typing import Optional
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange

# %%
class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) layer.
    (B, C, H, W) -- nn.AdaptiveAvgPool2d(1) --> (B, C, 1, 1) -- squeeze(-1) -->
    (B, C, 1) -- transpose(-1,-2) --> (B, 1, C) -- nn.Conv1d(1, 1) -->
    (B, 1, C) -- transpose(-1,-2) --> (B, C, 1) -- squeeze(-1) --> (B, C)

    References
    ----------
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. https://github.com/BangguWu/ECANet
    """
    def __init__(self, kernel_size: int = 3):
        super(EfficientChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).squeeze(-1)
        return y

class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


# %%
class SelfAttention(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            num_heads: int, 
            scale: Optional[float]=None, 
            dropout=0.,
            is_mask=False, is_softmax=False, fore_mask=False):
        super().__init__()
        head_dim = hidden_dim // num_heads
        assert head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads

        self.scale = head_dim ** -0.5 if scale is None else scale

        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        
        self.attend = nn.Softmax(dim=-1) if is_softmax else nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
         )

        self.is_mask, self.is_softmax, self.fore_mask = is_mask, is_softmax, fore_mask

    def forward(self, x):
        _, seq_len, _ = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # q = F.normalize(q, dim=-1, p=2)
        # k = F.normalize(k, dim=-1, p=2)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if self.is_mask and self.is_softmax:
            attn_mask = torch.full((seq_len, seq_len), -float('Inf'), device=x.device, dtype=x.dtype)
            if self.fore_mask:
                attn_mask = torch.triu(attn_mask, diagonal=1)
            else:
                attn_mask = torch.tril(attn_mask, diagonal=-1)
            dots = dots + attn_mask
        elif self.is_mask and not self.is_softmax:
            attn_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=x.dtype)
            attn_mask = torch.tril(attn_mask)
            dots = dots * attn_mask
            # dots = F.normalize(dots, dim=-1, p=2)

        attn = self.attend(dots)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

# %%
class Attention(nn.Module):
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

# %%
class AttentionWithoutParams(nn.Module):
    """
    Cross Attention modified from Perceiver.

    Three dimensions:
    - query_dim (or latent_dim): dim of the latent space
    - context_dim: dim of the input space
    - inner_dim: dim of the inner space of attention, equal to dim_head * heads

    Two inputs:
    - x: input from the latent space, with shape (latent, query_dim)
    - context: input from the input space, with shape (input, context_dim)

    Data flow:
    x(latent, query_dim) -- Linear(query_dim, inner_dim) --> q(latent, inner_dim)
    context(input, context_dim) -- nn.Linear(context_dim, inner_dim * 2) --> k,v(input, inner_dim)
    q(latent, inner_dim) * k(input, inner_dim) --> sim(latent, input) -- softmax(dim=-1) -->
    attn(latent, input) * v(input, inner_dim) --> 
    out(latent, inner_dim) -- Linear(inner_dim, query_dim) --> out(latent, query_dim)

    Refs:
    https://github.com/lucidrains/perceiver-pytorch/
    """
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        assert query_dim == inner_dim
        project_out = not (inner_dim == query_dim and heads == 1)


        self.scale = dim_head ** -0.5
        self.heads = heads

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()

    def forward(self, x, context = None):
        h = self.heads

        context = context if context is not None else x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (x,context,context))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out


