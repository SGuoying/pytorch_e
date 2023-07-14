# %%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, ConvMixerLayer, ConvMixerLayer2, Residual, ecablock
from sunyata.pytorch.layer.attention import Attention, AttentionWithoutParams, EfficientChannelAttention


# %%
class eca_layer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super(eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)

    def forward(self, x: torch.Tensor):  # x: (batch_size, channels)
        assert x.ndim == 2
        #  (batch_size, channels, 1, 1)
        # y = self.avg_pool(x)
        y = self.conv(x.unsqueeze(-1).transpose(-1,-2))
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = y.transpose(-1,-2).squeeze(-1)
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        #  squeeze：(batch_size, channels, 1)变为(batch_size, channels)
        return y

from torch import einsum
from einops import rearrange, reduce, repeat
# class FCUUp(nn.Module):
#     """ Transformer patch embeddings -> CNN feature maps
#     """

#     def __init__(self, hidden_dim, out_dim, up_stride=1):
#         super(FCUUp, self).__init__()

#         self.up_stride = up_stride
#         self.conv_project = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_dim)
#         self.act = nn.GELU()

#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
#         x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
#         x_r = self.act(self.bn(self.conv_project(x_r)))

#         return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
    

class Attnlayer(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (inner_dim == query_dim and heads == 1)

        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()
        self.fcup = FCUUp(query_dim, query_dim,)

    def forward(self, latent, context = None):
        h = self.heads
        # context (b c h w) --> (b h*w c)
        assert context.ndim == 4
        B, C, H, W = context.shape
        input = context.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent = latent[:, :-1, :]
        # context = input

        q = self.to_q(latent)
        # context = context if context is not None else latent
        # k, v = self.to_kv(context).chunk(2, dim=-1)
        # context = context if context is not None else latent
        k, v = self.to_kv(input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q,k,v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        # latent = out
        context = self.fcup(out, H, W)
        return context
@dataclass
class ConvMixerCfg(BaseCfg):
    num_layers: int = 8
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    squeeze_factor: int = 4

    layer_norm_zero_init: bool = True
    skip_connection: bool = True

    eca_kernel_size: int = 3

# %%


class ConvMixer(nn.Module):
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

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.digup(x)
        return x

class ConvMixer2(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()

        self.layers = nn.Sequential(*[
            ConvMixerLayer2(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
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

        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.digup = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim,
                      )
        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.cfg = cfg

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = latent + self.digup(latent, input)
        latent = self.logits_layer_norm(latent)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = latent + self.digup(latent, input)
            latent = self.logits_layer_norm(latent)

        # x = self.digup(x)
        latent = nn.Flatten()(latent)
        logits = self.fc(latent)
        return logits

  
class ConvMixer3(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            ConvMixerLayer2(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.convs = nn.Sequential(*[
            ConvMixerLayer2(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.digup = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim,
                      )
        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.cfg = cfg

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.digup(latent, input)
        latent = self.logits_layer_norm(latent)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.digup(latent, input)
            latent = self.logits_layer_norm(latent)

        # x = self.digup(x)
        # latent = nn.Flatten()(latent)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        logits = self.fc(latent)
        return logits

class BayesConvMixer(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        # self.digup = eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

class CombineConvMixer(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        # self.digup = eca_layer(kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        logits = self.logits_layer_norm(logits)
        for i, layer in enumerate(self.layers):
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            
            if i == 0:
                logit = self.digup(x) + logits
            else:
                logit = logits + self.digup(x)
            
            logits = self.logits_layer_norm(logit) + logits
            # logits = self.logits_layer_norm(self.digup(
            #     x) if i == 0 else logits + self.digup(x)) + logits
            # logits = logits + self.digup(x)
            # logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

from einops.layers.torch import Rearrange, Reduce


class BayesConvMixer3(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(self.logits_layer_norm.weight.data.shape)
        
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.digup = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim,
                      )
        
        # self.digup = eca_layer(kernel_size=cfg.eca_kernel_size)
        self.fcn_up = FCUUp(cfg.hidden_dim, up_stride=1)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )
        # self.skip_connection = cfg.skip_connection

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.embed(x)
        _, _, H, W = x.shape
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        # new add
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        # latent = torch.cat([latent, input], dim=1)

        latent = latent + self.digup(latent, input)
        latent = self.logits_layer_norm(latent)

        for layer in self.layers:
            x = x + layer(x)
            
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            # new add
            # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.digup(latent, input)
            latent = self.logits_layer_norm(latent)

        x1 = self.fcn_up(latent, H, W)
        x = self.fc(x1 + x)

        # latent = nn.Flatten()(latent)
        # latent = reduce(latent, 'b n d -> b d', 'mean')
        # logits = self.fc(latent)
        return x
    
# %%
class BayesConvMixer4(BayesConvMixer3):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.digup = AttentionWithoutParams(query_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim
                      )

# %%
class BayesConvMixer5(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(self.logits_layer_norm.weight.data.shape)
        
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.depth_attn = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim
                      )
        # self.digup = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Flatten(),
        # )
        self.digup = EfficientChannelAttention(kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.embed(x)
        logits = [self.digup(x)]
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + [self.digup(x)]
        logits = rearrange(logits, "depth b d -> b depth d")
        latent = self.depth_attn(latent, logits)
        latent = self.logits_layer_norm(latent)
        latent = nn.Flatten()(latent)
        logits = self.fc(latent)
        return logits



class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, hidden_dim, up_stride=1):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
    

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None,  drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class token_mixer(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__()
        self.layer1 = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
            )),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim), 
            # StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )
        self.attn_layers = Attention(query_dim=hidden_dim, 
                                context_dim=hidden_dim, 
                                heads=1, 
                                dim_head=hidden_dim, 
                                )
        self.drop = nn.Dropout(drop_rate)
        self.latent = nn.Parameter(torch.randn(1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.layer1(x)
        x = self.drop(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.attn_layers(latent, input)
        latent = self.norm(latent)
        return latent
    
class formerblock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(hidden_dim)

        self.token_mixer = token_mixer(hidden_dim, kernel_size, drop_rate)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.mlp = Mlp(hidden_dim, hidden_features=hidden_dim*4, drop=drop_rate)

        self.fcup = FCUUp(hidden_dim, hidden_dim)

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        x = x + self.drop(self.fcup(self.token_mixer(self.norm1(x)), H, W))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x
    
class Former(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        layers = []
        for _ in range(cfg.num_layers):
            layers.append(
                formerblock(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
                )
        self.layers = nn.Sequential(*layers)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.layers(x)
        x = self.digup(x)
        return x


class BayesFormer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        layers = []
        for _ in range(cfg.num_layers):
            layers.append(
                formerblock(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
                )
        self.layers = nn.Sequential(*layers)

        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.digup = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim,
                      )
        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        # self.digup = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(cfg.hidden_dim, cfg.num_classes)
        # )

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.digup(latent, input)
        latent = self.logits_layer_norm(latent)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.digup(latent, input)
            latent = self.logits_layer_norm(latent)

        latent = reduce(latent, 'b n d -> b d', 'mean')
        # x = self.layers(x)
        # x = self.digup(x)
        return self.fc(latent)