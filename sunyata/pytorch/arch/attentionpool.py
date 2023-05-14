from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    "Attention for Learned Aggregation"
    def __init__(self,
        dim:int,
        bias:bool=True,
        norm:Callable[[int], nn.Module]=nn.LayerNorm
    ):
        super().__init__()
        self.norm = norm(dim)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.vk = nn.Linear(dim, dim*2, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cls_q):
        x = self.norm(x.flatten(2).transpose(1,2))
        B, N, C = x.shape

        q = self.q(cls_q.expand(B, -1, -1))
        k, v = self.vk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).chunk(2, 0)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, C)
        return self.proj(x)
    
class AvgAttnPooling2d(nn.Module):
    def __init__(self,
        dim:int,
        attn_bias:bool=True,
        # ffn_expand:int|float=3,
        ffn_expand:int=3,
        norm:Callable[[int], nn.Module]=nn.LayerNorm,
        act_cls:Callable[[None], nn.Module]=nn.GELU,
    ):
        super().__init__()
        self.cls_q = nn.Parameter(torch.zeros([1,dim]))
        self.attn = AttentionPool2d(dim, attn_bias, norm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = norm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim*ffn_expand)),
            act_cls(),
            norm(int(dim*ffn_expand)),
            nn.Linear(int(dim*ffn_expand), dim)
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        # self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(self.pool(x).flatten(1) + self.attn(x, self.cls_q))
        x =  x + self.ffn(x)
        x_reshaped = torch.unsqueeze(x, 2)
        x_reshaped = torch.unsqueeze(x_reshaped, 3)
        return x_reshaped

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class AvgAttnPooling2dS(nn.Module):
    def __init__(self,
        ni:int,
        attn_bias:bool=True,
        # ffn_expand:int=3,
        norm:Callable[[int], nn.Module]=nn.LayerNorm,
        act_cls:Callable[[None], nn.Module]=nn.GELU,
    ):
        super().__init__()
        self.cls_q = nn.Parameter(torch.zeros([1,ni]))
        self.attn = AttentionPool2d(ni, attn_bias, norm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = norm(ni)
        self.act = act_cls()
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(self.pool(x).flatten(1) + self.attn(x, self.cls_q))
        x = self.act(x)
        x_reshaped = torch.unsqueeze(x, 2)
        x_reshaped = torch.unsqueeze(x_reshaped, 3)
        return x_reshaped

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



## clip
# class AttentionPool2d(nn.Module):
class Attention(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)