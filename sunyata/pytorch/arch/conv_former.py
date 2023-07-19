
from dataclasses import dataclass
from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg


@dataclass
class ConvMixerCfg(BaseCfg):
    num_layers: int = 8
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    mlp_rate: int = 4

    layer_norm_zero_init: bool = True
    skip_connection: bool = True
    eca_kernel_size: int = 3


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    

class Convblock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding=kernel_size//2)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.drop = nn.Dropout(drop_rate)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        clone = x.clone()
        x = self.conv(x)
        x = self.gelu(x)
        x = clone + self.bn(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.bn2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, hidden_dim: int, mlp_rate: int, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        hidden_features = hidden_dim * mlp_rate     
        self.fc1 = nn.Linear(hidden_dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
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

  
class transformer(nn.Module):
    def __init__(self, hidden_dim: int, mlp_rate: int, kernel_size: int, drop_rate: float=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = Attention(
            query_dim=hidden_dim,
            context_dim=hidden_dim,
            heads=1,
            dim_head=hidden_dim,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim=hidden_dim, mlp_rate=mlp_rate, drop=drop_rate)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, context=None):
        context = context if context is not None else x
        x = self.drop(self.attn(self.norm1(x), context=context)) + x
        x = self.drop(self.mlp(self.norm2(x))) + x
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
    

class Convformer2(nn.Module):
    def __init__(self, 
                 cfg:ConvMixerCfg):
        super().__init__()

        self.cfg = cfg
        self.patch_size = [4, 4, 2]
        self.hidden_dim = [64, 128, 256]
        self.depth = [2, 2, 2]
        
        #  stage 1
        self.patch_embed1 = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim[0], patch_size=self.patch_size[0])
        conv1 = []
        for _ in range(self.depth[0]):
            conv1.append(Convblock(hidden_dim=self.hidden_dim[0], 
                                   kernel_size=cfg.kernel_size, 
                                   drop_rate=cfg.drop_rate))
        # conv1.append()
        self.conv1 = nn.Sequential(*conv1)
        self.stage1 = nn.ModuleList([
            self.conv1,
            transformer(hidden_dim=self.hidden_dim[0],
                                 mlp_rate=4,
                                 kernel_size=cfg.kernel_size,
                                 drop_rate=cfg.drop_rate)
        ])

        #  stage 2 **********************************************
        self.patch_embed2 = PatchEmbed(in_channels=self.hidden_dim[0], hidden_dim=self.hidden_dim[1], patch_size=self.patch_size[1])
        self.up1 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        conv2 = []
        for _ in range(self.depth[1]):
            conv2.append(Convblock(hidden_dim=self.hidden_dim[1], 
                                   kernel_size=cfg.kernel_size, 
                                   drop_rate=cfg.drop_rate))
        # conv2.append(transformer(hidden_dim=self.hidden_dim[1],
        #                             mlp_rate=cfg.mlp_rate,
        #                             kernel_size=cfg.kernel_size,
        #                             drop_rate=cfg.drop_rate))
        self.conv2 = nn.Sequential(*conv2)
        self.stage2 = nn.ModuleList([
            self.conv2,
            transformer(hidden_dim=self.hidden_dim[1],
                                 mlp_rate=4,
                                 kernel_size=cfg.kernel_size,
                                 drop_rate=cfg.drop_rate)
        ])

        #  stage 3 ********************************************
        self.patch_embed3 = PatchEmbed(in_channels=self.hidden_dim[1], hidden_dim=self.hidden_dim[2], patch_size=self.patch_size[2])
        self.up2 = nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
        conv3 = []
        for _ in range(self.depth[2]):
            conv3.append(Convblock(hidden_dim=self.hidden_dim[2], 
                                   kernel_size=cfg.kernel_size, 
                                   drop_rate=cfg.drop_rate))
        # conv3.append(transformer(hidden_dim=self.hidden_dim[2],
        #                             mlp_rate=cfg.mlp_rate,
        #                             kernel_size=cfg.kernel_size,
        #                             drop_rate=cfg.drop_rate))
        self.conv3 = nn.Sequential(*conv3)
        self.stage3 = nn.ModuleList([
            self.conv3,
            transformer(hidden_dim=self.hidden_dim[2],
                                 mlp_rate=4,
                                 kernel_size=cfg.kernel_size,
                                 drop_rate=cfg.drop_rate)
        ])

        #  classifier ********************************************
        self.norm = nn.LayerNorm(self.hidden_dim[2])
        self.fc = nn.Linear(self.hidden_dim[2], cfg.num_classes)

        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim[0]))

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.patch_embed1(x)
        _, _, H, W = x.shape
        for conv, transformer in self.stage1:
            # x = conv(x)
            for layer in conv:
                x = layer(x)
            context = x.permute(0, 2, 3, 1)
            context = rearrange(context, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], context], dim=1)
            latent = transformer(latent, context=context) + latent
            B, _, C = latent.shape
            # latent = self.norm(latent)
        x = latent[:, 1:].transpose(1, 2).reshape(B, C, H, W)

        x = self.patch_embed2(x)
        _, _, H, W = x.shape
        latent = self.up1(latent)
        for conv, transformer in self.stage2:
            # x = conv(x)
            for layer in conv:
                x = layer(x)
            context = x.permute(0, 2, 3, 1)
            context = rearrange(context, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], context], dim=1)
            latent = transformer(latent, context=context) + latent
            B, _, C = latent.shape
            # latent = self.norm(latent)
        x = latent[:, 1:].transpose(1, 2).reshape(B, C, H, W)

        x = self.patch_embed3(x)
        _, _, H, W = x.shape
        latent = self.up2(latent)
        for conv, transformer in self.stage3:
            # x = conv(x)
            for layer in conv:
                x = layer(x)
            context = x.permute(0, 2, 3, 1)
            context = rearrange(context, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], context], dim=1)
            latent = transformer(latent, context=context) + latent
            latent = self.norm(latent)
        # x = latent[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent)


        
        
class Convformer(nn.Module):
    def __init__(self,
                 cfg: ConvMixerCfg):
        super().__init__()

        self.cfg = cfg
        self.patch_size = [4, 4, 2]
        self.hidden_dim = [64, 128, 256]
        self.depth = [2, 2, 2]

        #  stage 1
        self.patch_embed1 = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim[0], patch_size=self.patch_size[0])
        conv1 = []
        for _ in range(self.depth[0]):
            conv1.append(Convblock(hidden_dim=self.hidden_dim[0],
                                   kernel_size=cfg.kernel_size,
                                   drop_rate=cfg.drop_rate))
        # conv1.append()
        self.conv1 = nn.Sequential(*conv1)
        # self.stage1 = nn.ModuleList([
        #     self.conv1,
        #     transformer(hidden_dim=self.hidden_dim[0],
        #                 mlp_rate=4,
        #                 kernel_size=cfg.kernel_size,
        #                 drop_rate=cfg.drop_rate)
        # ])
        self.stage1 = transformer(hidden_dim=self.hidden_dim[0],
                                  mlp_rate=4,
                                  kernel_size=cfg.kernel_size,
                                  drop_rate=cfg.drop_rate)

        #  stage 2 **********************************************
        self.patch_embed2 = PatchEmbed(in_channels=self.hidden_dim[0], hidden_dim=self.hidden_dim[1],
                                       patch_size=self.patch_size[1])
        self.up1 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        conv2 = []
        for _ in range(self.depth[1]):
            conv2.append(Convblock(hidden_dim=self.hidden_dim[1],
                                   kernel_size=cfg.kernel_size,
                                   drop_rate=cfg.drop_rate))
        # conv2.append(transformer(hidden_dim=self.hidden_dim[1],
        #                             mlp_rate=cfg.mlp_rate,
        #                             kernel_size=cfg.kernel_size,
        #                             drop_rate=cfg.drop_rate))
        self.conv2 = nn.Sequential(*conv2)
        self.stage2 =  transformer(hidden_dim=self.hidden_dim[1],
                        mlp_rate=4,
                        kernel_size=cfg.kernel_size,
                        drop_rate=cfg.drop_rate)

        #  stage 3 ********************************************
        self.patch_embed3 = PatchEmbed(in_channels=self.hidden_dim[1], hidden_dim=self.hidden_dim[2],
                                       patch_size=self.patch_size[2])
        self.up2 = nn.Linear(self.hidden_dim[1], self.hidden_dim[2])
        conv3 = []
        for _ in range(self.depth[2]):
            conv3.append(Convblock(hidden_dim=self.hidden_dim[2],
                                   kernel_size=cfg.kernel_size,
                                   drop_rate=cfg.drop_rate))
        # conv3.append(transformer(hidden_dim=self.hidden_dim[2],
        #                             mlp_rate=cfg.mlp_rate,
        #                             kernel_size=cfg.kernel_size,
        #                             drop_rate=cfg.drop_rate))
        self.conv3 = nn.Sequential(*conv3)

        self.stage3 = transformer(hidden_dim=self.hidden_dim[2],
                                  mlp_rate=4,
                                  kernel_size=cfg.kernel_size,
                                  drop_rate=cfg.drop_rate)


        #  classifier ********************************************
        self.norm = nn.LayerNorm(self.hidden_dim[2])
        self.fc = nn.Linear(self.hidden_dim[2], cfg.num_classes)

        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim[0]))

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)

        x = self.patch_embed1(x)
        _, _, H, W = x.shape
        # for conv, transformer in self.stage1:
        x = self.conv1(x)
            # for layer in conv:
            #     x = layer(x)
        context = x.permute(0, 2, 3, 1)
        context = rearrange(context, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], context], dim=1)
        latent = self.stage1(latent, context) + latent
        B, _, C = latent.shape
            # latent = self.norm(latent)
        x = latent[:, 1:].transpose(1, 2).reshape(B, C, H, W)

        x = self.patch_embed2(x)
        _, _, H, W = x.shape
        latent = self.up1(latent)

        x = self.conv2(x)
            # for layer in conv:
            #     x = layer(x)
        context = x.permute(0, 2, 3, 1)
        context = rearrange(context, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], context], dim=1)
        latent = self.stage2(latent, context) + latent
        B, _, C = latent.shape
            # latent = self.norm(latent)
        x = latent[:, 1:].transpose(1, 2).reshape(B, C, H, W)

        x = self.patch_embed3(x)
        _, _, H, W = x.shape
        latent = self.up2(latent)

        x = self.conv3(x)
            # for layer in conv:
            #     x = layer(x)
        context = x.permute(0, 2, 3, 1)
        context = rearrange(context, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], context], dim=1)
        latent = self.stage3(latent, context) + latent
        
        # x = latent[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        latent = self.norm(latent)
        return self.fc(latent)








