from dataclasses import dataclass
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, ConvMixerLayer, ConvMixerLayer2, Residual, ecablock


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
            nn.Conv2d(hidden_dim, hidden_dim//4, 1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.GELU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//4, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.BatchNorm2d(hidden_dim//4),
            nn.GELU(),
            nn.Conv2d(hidden_dim//4, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
# class ConvLayer(nn.Sequential):
#     def __init__(self, hidden_dim, kernel_size, bias=False):
#         super().__init__(
#             nn.Conv2d(hidden_dim, hidden_dim, 1),
#             nn.GELU(),
#             nn.BatchNorm2d(hidden_dim),
            
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, bias=bias),
#             nn.GELU(),
#             nn.BatchNorm2d(hidden_dim),
            
#             nn.Conv2d(hidden_dim, hidden_dim, 1),
#             nn.GELU(),
#             nn.BatchNorm2d(hidden_dim),
            
#         )


class ConvLayer3(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate=0.):
        super().__init__(
           
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            Residual(nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, bias=False),
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),)),
        )    

# class ConvLayer3(nn.Sequential):
#     def __init__(self, hidden_dim: int, kernel_size: int, drop_rate=0.):
#         super().__init__(
           
#             nn.Conv2d(hidden_dim, hidden_dim, 1),
#             nn.GELU(),
#             nn.BatchNorm2d(hidden_dim),
            
#             Residual(nn.Sequential(
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, bias=False),
#             nn.GELU(),
#             # nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
#             nn.BatchNorm2d(hidden_dim),
            
            
#             nn.Conv2d(hidden_dim, hidden_dim, 1),
#             nn.GELU(),
#             nn.BatchNorm2d(hidden_dim),
#             )),
#         ) 

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
            kwargs.update(context = normed_context)
            # kwargs['context'] = normed_context

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

class AttnLayer2(nn.Module):
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

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = torch.einsum('b i n d, b j n d -> b i j n ', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b i j n, b j n d -> b i n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
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
            # nn.GELU(),
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


class Conformer_1(nn.Module):
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
        # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.attn_layers(latent, input)
        # latent = rearrange(latent[:, 1:], 'b (h w) d -> b d h w', h=x.shape[2])
        latent = self.norm(latent)

        for layer in self.layers:
            x = x + layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.attn_layers(latent, input)
            latent = self.norm(latent)

        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.to_logits(latent)


class Conformer_2(nn.Module):
    def __init__(self,
                 cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            ConvLayer(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])

        # self.attn_layers = AttnLayer(query_dim=cfg.hidden_dim,
        #                              context_dim=cfg.hidden_dim,
        #                              heads=1,
        #                              dim_head=cfg.hidden_dim,
        #                              dropout=cfg.drop_rate)
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.to_logits = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.embed(x)
        # input = x.permute(0, 2, 3, 1)
        # input = rearrange(input, 'b ... d -> b (...) d')
        # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        # latent = latent + self.attn_layers(latent, input)
        # # latent = rearrange(latent[:, 1:], 'b (h w) d -> b d h w', h=x.shape[2])
        # latent = self.norm(latent)

        for layer in self.layers:
            x = x + layer(x)
            # input = x.permute(0, 2, 3, 1)
            # input = rearrange(input, 'b ... d -> b (...) d')
            # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            # latent = latent + self.attn_layers(latent, input)
            # latent = self.norm(latent)
        x = self.fc(x)
        return x

        # latent = reduce(latent, 'b n d -> b d', 'mean')
        # return self.to_logits(latent)
### layer 2######################################################

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

class Conformer2(nn.Module):
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

        self.fcn_up = FCUUp(cfg.hidden_dim, up_stride=1)

        self.conv = ConvLayer(cfg.hidden_dim, cfg.kernel_size)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )
        
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.to_logits = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.embed(x)
        _, _, H, W = x.shape
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

        x1 = self.fcn_up(latent, H, W)
        x = self.conv(x + x1)
        x = self.digup(x)

        # latent = reduce(latent, 'b n d -> b d', 'mean')
        # return self.to_logits(latent)
        return x
    

# layer 3 #########################################################
class Conformer3(nn.Module):
    def __init__(self,
                 cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.Sequential(*[
            ConvLayer3(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])

        self.attn_layers = AttnLayer(query_dim=cfg.hidden_dim,
                                     context_dim=cfg.hidden_dim,
                                     heads=1,
                                     dim_head=cfg.hidden_dim,
                                     dropout=cfg.drop_rate)
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            # nn.GELU(),
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
        # latent = torch.cat([latent, input], dim=1)
        latent = latent + self.attn_layers(latent, input)
        latent = self.norm(latent)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.attn_layers(latent, input)
            latent = self.norm(latent)

        latent = reduce(latent, 'b n d -> b d', 'mean')
        latent = self.norm(latent)
        return self.to_logits(latent)

class Conformer3_1(nn.Module):
    def __init__(self,
                 cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.Sequential(*[
            ConvLayer3(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])

        self.conv = nn.Sequential(*[
            ConvLayer3(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers//2)
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
        
        x = self.conv(x)

        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        # latent = latent[:, :-1, :]
        # latent = torch.cat([latent, input], dim=1)
        latent = latent + self.attn_layers(latent, input)
        latent = self.norm(latent)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            # latent = latent[:, :-1, :]
            latent = latent + self.attn_layers(latent, input)
            latent = self.norm(latent)

        latent = reduce(latent, 'b n d -> b d', 'mean')
        latent = self.norm(latent)
        return self.to_logits(latent)
    
class Conformer3_2(nn.Module):
    def __init__(self,
                 cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.Sequential(*[
            ConvLayer3(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])

        # self.attn_layers = AttnLayer(query_dim=cfg.hidden_dim,
        #                              context_dim=cfg.hidden_dim,
        #                              heads=1,
        #                              dim_head=cfg.hidden_dim,
        #                              dropout=cfg.drop_rate)
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

        # self.norm = nn.LayerNorm(cfg.hidden_dim)
        # self.to_logits = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        # self.latent = nn.Parameter(torch.randn(1, cfg.hidden_dim))

    def forward(self, x):
        # b, _, _, _ = x.shape
        # latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.embed(x)
        # input = x.permute(0, 2, 3, 1)
        # input = rearrange(input, 'b ... d -> b (...) d')
        # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        # # latent = torch.cat([latent, input], dim=1)
        # latent = latent + self.attn_layers(latent, input)
        # latent = self.norm(latent)

        for layer in self.layers:
            x = layer(x)
            # input = x.permute(0, 2, 3, 1)
            # input = rearrange(input, 'b ... d -> b (...) d')
            # latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            # latent = latent + self.attn_layers(latent, input)
            # latent = self.norm(latent)

        x = self.fc(x)

        # latent = reduce(latent, 'b n d -> b d', 'mean')
        # return self.to_logits(latent)
        return x
    


# layer 4 #########################################################
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
    

class transformer(nn.Module):
    def __init__(self, hidden_dim, context_dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.attn = AttnLayer(query_dim=hidden_dim,
                                context_dim=context_dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout)
        self.mlp = Mlp(hidden_dim, hidden_dim * 4)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, latent, input):
        latent = latent + self.attn(latent, input)
        latent = self.norm1(latent)
        latent = latent + self.mlp(latent)
        latent = self.norm2(latent)
        return latent

class ConvTransBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size, heads, dim_head, drop_rate=0.):
        super().__init__()
        # self.cnn_block = ConvLayer(hidden_dim, kernel_size, drop_rate)
        self.cnn_block = ConvLayer3(hidden_dim, kernel_size, drop_rate)
        self.transformer_block = transformer(hidden_dim, heads, dim_head, drop_rate)

    def forward(self, x, latent):
        x = x + self.cnn_block(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = self.transformer_block(latent, input)
        return x, latent

class Conformer4(Conformer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, cfg.patch_size, stride=cfg.patch_size),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.Sequential(*[
            ConvLayer3(cfg.hidden_dim, cfg.kernel_size)
            for _ in range(cfg.num_layers)
        ])

        self.transformer = transformer(hidden_dim=cfg.hidden_dim,
                                       context_dim=cfg.hidden_dim,
                                       heads=1,
                                       dim_head=cfg.hidden_dim,)

        # self.layers = nn.ModuleList()
        # for _ in range(cfg.num_layers):
        #     self.layers.append(
        #         ConvTransBlock(cfg.hidden_dim, 
        #                        cfg.kernel_size, 
        #                        1, 
        #                        cfg.hidden_dim))


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
        latent = self.transformer(latent, input)

        for layer in self.layers:
            x = layer(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = self.transformer(latent, input)
           
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.to_logits(latent)
    




        
        