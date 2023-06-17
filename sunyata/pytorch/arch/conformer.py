from dataclasses import dataclass
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, Residual


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



class ConvLayer(nn.Sequential):
    def __init__(self, hidden_dim, kernel_size, bias=False):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout = 0.):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class ConvLayer2(nn.Sequential):
    def __init__(self, hidden_dim, kernel_size, bias=False):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),  
        ) 

class ConvLayer3(nn.Sequential):
    def __init__(self, hidden_dim, kernel_size, bias=False):
        super().__init__(
            Residual(nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            )),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )    

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs['context']
            normed_context = self.norm_context(context)
            # kwargs.update(context = normed_context)
            kwargs['context'] = normed_context

        return self.fn(x, **kwargs)

class AttnLayer(nn.Module):
    def __init__(self, 
                 query_dim, context_dim=None,
                 heads=8, dim_head=64, dropout=0.):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = dim_head * heads
        project_out = not (inner_dim == query_dim and heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()
    
    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q,k,v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, 'b (h n) d -> b n (h d)', h=h)
        return self.to_out(out)


### layer 0 #######################################
class Convolution(nn.Module):
    def __init__(self,
                 cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            ConvLayer(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.digup(x)
        return x


##  layer 1 ###############################################

class Conformer(nn.Module):
    def __init__(self,
                 cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            ConvLayer(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])

        self.attn_layers = AttnLayer(query_dim=cfg.hidden_dim,
                                     context_dim=cfg.hidden_dim,
                                     heads=1,
                                     dim_head=cfg.hidden_dim,
                                     dropout=cfg.drop_rate)
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.to_logits = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.attn_layers(latent, input)
        # latent = rearrange(latent[:, 1:], 'b (h w) d -> b d h w', h=x.shape[2])
        latent = self.norm(latent)

        for layer in self.layers:
            x = x + layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.attn_layers(latent, input)
            latent = self.norm(latent)

        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.to_logits(latent)


### layer 2######################################################

class Conformer2(Conformer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList([
            ConvLayer(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1),
        #     nn.BatchNorm2d(cfg.hidden_dim),
        #     nn.GELU(),
        # )

        self.attn_mlp = nn.ModuleList([])
        self.attn_mlp.append(nn.ModuleList([PreNorm(cfg.hidden_dim, AttnLayer(query_dim=cfg.hidden_dim,
                                                               context_dim=cfg.hidden_dim,
                                                               heads=1,
                                                               dim_head=cfg.hidden_dim),
                                                               context_dim=cfg.hidden_dim),
                                            PreNorm(cfg.hidden_dim, Mlp(cfg.hidden_dim, cfg.hidden_dim * 4))
                                                               ]))
        # self.attn_mlp.append(PreNorm(cfg.hidden_dim, Mlp(cfg.hidden_dim, cfg.hidden_dim * 4)))

        # self.mlp = Mlp(cfg.hidden_dim, cfg.hidden_dim * 4)

        # self.attn_layers = AttnLayer(query_dim=cfg.hidden_dim,
        #                             context_dim=cfg.hidden_dim,
        #                             heads=1,
        #                             dim_head=cfg.hidden_dim,
        #                             )
        # self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.to_logits = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=batch_size)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        for attn, mlp in self.attn_mlp:
            latent = latent + attn(latent, input)
            mlp = mlp(latent)
        # latent = latent + self.attn_layers(latent, input)
        # mlp = self.mlp(latent) 

        for layer in self.layers:
            x = x + layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            # latent = latent + self.attn_layers(latent, input)
            # latent = self.norm(latent)
            # mlp = self.mlp(latent) + mlp
            # mlp = self.norm(mlp)
            for attn, mlp in self.attn_mlp:
                latent = latent + attn(latent, input)
                mlp = mlp + mlp(latent)

        latent = reduce(mlp, 'b n d -> b d', 'mean')
        return self.to_logits(latent)
    

# layer 3 #########################################################
class Conformer3(nn.Module):
    def __init__(self,
                 cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        # self.layers = nn.ModuleList([
        #     ConvLayer(cfg.hidden_dim, cfg.kernel_size)
        #     for _ in range(cfg.num_layers)
        # ])
        self.layers = nn.Sequential(*[
            ConvLayer3(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.attn_layers = AttnLayer(query_dim=cfg.hidden_dim,
                                     context_dim=cfg.hidden_dim,
                                     heads=1,
                                     dim_head=cfg.hidden_dim,
                                     dropout=cfg.drop_rate)
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.to_logits = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.embed(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.attn_layers(latent, input)
        # latent = rearrange(latent[:, 1:], 'b (h w) d -> b d h w', h=x.shape[2])
        latent = self.norm(latent)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.attn_layers(latent, input)
            latent = self.norm(latent)

        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.to_logits(latent)





        
        