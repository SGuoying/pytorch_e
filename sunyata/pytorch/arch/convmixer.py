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
    patch_size: int = 7
    num_classes: int = 10

    drop_rate: float = 0.

    layer_norm_zero_init: bool = True
    skip_connection: bool = True

    eca_kernel_size: int = 3

# %%

class ConvMixer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super(ConvMixer, self).__init__()

        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),  # eps>6.1e-5 to avoid nan in half precision
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
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

# %%
class ConvMixer2(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            ConvMixerLayer2(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x= self.layers(x)
        x= self.digup(x)
        return x


# %%
class IterConvMixer(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(self.logits_layer_norm.weight.data.shape)
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
#         self.digup = EfficientChannelAttention(kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

    def forward(self, x):
        x = self.embed(x)
        logits = self.logits_layer_norm(self.digup(x))
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + self.logits_layer_norm(self.digup(x))
#             logits = self.logits_layer_norm(logits)
        logits = self.fc(self.logits_layer_norm(logits))
        return logits

# %%
class BayesConvMixer2(ConvMixer2):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)


    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

# %%
class IterAttnConvMixer(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(self.logits_layer_norm.weight.data.shape)
        
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

        self.digup = Attention(query_dim=cfg.hidden_dim,
                      context_dim=cfg.hidden_dim,
                      heads=1, 
                      dim_head=cfg.hidden_dim
                      )
        
        # self.digup = eca_layer(kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = latent + self.digup(latent, input)
        latent = self.logits_layer_norm(latent)

        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = latent + self.digup(latent, input)
            latent = self.logits_layer_norm(latent)

        latent = nn.Flatten()(latent)
        logits = self.fc(latent)
        return logits


# %%
class BayesConvMixer4(IterAttnConvMixer):
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
    


class block(nn.Module):
    def __init__(self, hidden_dim, kernel_size, drop_rate=0.):
        super(block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim,  padding='same'),
            nn.GELU() 
            )
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout(drop_rate)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x
    
class mlp(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, drop_rate=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, mlp_dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_dim, kernel_size=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int,hidden_dim: int,  patch_size: int):
        super().__init__()
        # self.image_size = image_size
        self.patch_size = patch_size

        self.embed = self.embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.image_size[0] and W == self.image_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.embed(x)
        return x
    
class formerblock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, drop_rate=0.):
        super().__init__()
        mlp_dim = hidden_dim*4
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout(drop_rate)
        self.block = block(hidden_dim, kernel_size, drop_rate)
        self.mlp = mlp(hidden_dim, mlp_dim, drop_rate)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        x = self.drop(self.block(self.norm1(x))) + x
        x = self.drop(self.mlp(self.norm2(x))) + x

        return x
    
class convformer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = [64, 128, 256, 512]
        self.patch_size = [4, 2, 2, 2]
        self.depth = [2, 2, 6, 2]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim[0],
                                      patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(
                PatchEmbed(in_channels=self.hidden_dim[i], 
                           hidden_dim=self.hidden_dim[i+1], 
                           patch_size=self.patch_size[i+1]))
        self.formerblock = nn.ModuleList()
        for i in range(4):
            formerblock_list = nn.Sequential(
                *[formerblock(self.hidden_dim[i], cfg.kernel_size, cfg.drop_rate) for _ in range(self.depth[i])]
            )
            self.formerblock.append(formerblock_list)
        
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim[-1]),
        )
        self.fc = nn.Linear(self.hidden_dim[-1], cfg.num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.formerblock[i](x)
        x = self.neck(x)
        x = self.fc(x)
        return x




