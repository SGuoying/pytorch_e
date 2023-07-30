
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
        # self.attn = self_Attention(
        #     dim=hidden_dim, 
        #     heads=1, 
        #     dim_head=hidden_dim
        #     )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim=hidden_dim, mlp_rate=mlp_rate, drop=drop_rate)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, context=None):
        context = context if context is not None else x
        x = self.drop(self.attn(self.norm1(x), context=context)) + x
        # x = self.drop(self.attn(self.norm1(x))) + x
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
    

class block(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.):
        super().__init__()
        self.block = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 7, groups=hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
            )),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim), 
            nn.Dropout(drop_rate),
        )
    def forward(self, x):
        x = self.block(x)
        return x


class block2(nn.Module):
    def __init__(self, hidden_dim, drop_rate=0.):
        super().__init__()
        self.block = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim), 
            
                nn.Conv2d(hidden_dim, hidden_dim, 5, groups=hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
            )),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim), 
            nn.Dropout(drop_rate),
        )
    def forward(self, x):
        x = self.block(x)
        return x

class ConvMixerV0(nn.Module):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = [4, 2, 2, 2]
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 1]
        # self.depth = [3, 3, 9, 3]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
            )
            self.conv.append(stage)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.conv[i](x)

        x = self.digup(x)
        return self.fc(x)
    
class ConvMixerV1(nn.Module):
    '''
    ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)
ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)
ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)
ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)

    '''
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
                PatchEmbed(in_channels=self.hidden_dim[i], hidden_dim=self.hidden_dim[i+1], patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        for i in range(4):
            if i != 2:
                stage = nn.Sequential(
                    *[block(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
                )
                self.conv.append(stage)
            else:
                stage = nn.ModuleList()
                for j in range(self.depth[i] // self.depth[0]):
                    stage_j = nn.Sequential(
                        *[block(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate)
                          for _ in range(self.depth[i]//3)],
                    )
                    stage.append(stage_j)
                self.conv.append(stage)

        self.count = self.depth[2] // self.depth[0]

        self.upsample = nn.ModuleList()
        for i in range(3):
            upsample = nn.Sequential(
                nn.Conv2d(self.hidden_dim[i], self.hidden_dim[3], 1),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_dim[3])
            )
            self.upsample.append(upsample)
        self.upsample.append(nn.Identity())

        self.attn = Attention(query_dim=self.hidden_dim[3],
                              context_dim=self.hidden_dim[3],
                              heads=1,
                              dim_head=self.hidden_dim[3], )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim[3]),
        )
        self.fc = nn.Linear(self.hidden_dim[3], cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim[3])
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim[3]))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            if i != 2:
                x = self.downsample[i](x)
                x = self.conv[i](x)
                context = self.upsample[i](x)
                context = context.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn(latent, context) + latent
                latent = self.norm(latent)
            else:
                x = self.downsample[i](x)
                for conv in self.conv[i]:
                    x = conv(x)
                    context = self.upsample[i](x)
                    context = context.permute(0, 2, 3, 1)
                    context = rearrange(context, 'b ... d -> b (...) d')
                    latent = self.attn(latent, context) + latent
                    latent = self.norm(latent)
                # for j in range(self.count):
                #     x = self.conv[i][j](x)
                #     context = self.upsample[i](x)
                #     print(j)
                #     context = context.permute(0, 2, 3, 1)
                #     context = rearrange(context, 'b ... d -> b (...) d')
                #     latent = self.attn(latent, context) + latent
                #     latent = self.norm(latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)
      
class ConvMixerV2(nn.Module):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = [4, 2, 2, 2]
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 1]
        # self.depth = [3, 3, 9, 3]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        for i in range(4):
            if i != 2:
                stage = nn.Sequential(
                    *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
                )
                self.conv.append(stage)
            else:
                stage = nn.ModuleList()
                for _ in range(self.depth[i] // self.depth[0]):
                    conv = nn.Sequential(
                        *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i] //3)]
                    )
                    stage.append(conv)
                self.conv.append(stage)

        count = self.depth[i] // self.depth[0]
        self.count = count

        self.attn = Attention(query_dim=self.hidden_dim,
                              context_dim=self.hidden_dim,
                              heads=1,
                              dim_head=self.hidden_dim,)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            if i != 2:
                x = self.downsample[i](x)
                x = self.conv[i](x)
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn(latent, context) + latent
                latent = self.norm(latent)
            else:
                x = self.downsample[i](x)
                for conv in self.conv[i]:
                    x = conv(x)
                    context = x.permute(0, 2, 3, 1)
                    context = rearrange(context, 'b ... d -> b (...) d')
                    latent = self.attn(latent, context) + latent
                    latent = self.norm(latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)
    
class ConvMixerV3(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = [64, 128, 256, 512]
        # self.patch_size = [4, 2, 2, 2]
        self.depth = [2, 2, 6, 2]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim[0],
                                      patch_size=4)
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(
                PatchEmbed(in_channels=self.hidden_dim[i], hidden_dim=self.hidden_dim[i+1], patch_size=2))

        self.conv = nn.ModuleList()
        self.attn = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        for i in range(4):

            attn = Attention(query_dim=self.hidden_dim[i],
                             context_dim=self.hidden_dim[i],
                             heads=1,
                             dim_head=self.hidden_dim[i],)
            self.attn.append(attn)

            norm = nn.LayerNorm(self.hidden_dim[i])
            self.norm.append(norm)

            if i != 2:
                stage = nn.Sequential(
                    *[block2(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
                )
                self.conv.append(stage)
            else:
                stage = nn.ModuleList()
                for j in range(self.depth[i] // self.depth[0]):
                    stage_j = nn.Sequential(
                        *[block2(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate)
                          for _ in range(self.depth[i]//3)],
                    )
                    stage.append(stage_j)
                self.conv.append(stage)

        self.count = self.depth[2] // self.depth[0]

        self.upsample = nn.ModuleList()
        
        for i in range(3):
            upsample = nn.Sequential(
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]),
                nn.LayerNorm(self.hidden_dim[i+1])
            )
            self.upsample.append(upsample)
        self.upsample.append(nn.Identity())

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim[3]),
        )
        self.fc = nn.Linear(self.hidden_dim[3], cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim[0]))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            if i != 2:
                x = self.downsample[i](x)
                x = self.conv[i](x)
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn[i](latent, context) + latent
                latent = self.norm[i](latent)
                latent = self.upsample[i](latent)
            else:
                x = self.downsample[i](x)
                for conv in self.conv[i]:
                    x = conv(x)
                    context = x.permute(0, 2, 3, 1)
                    context = rearrange(context, 'b ... d -> b (...) d')
                    latent = self.attn[i](latent, context) + latent
                    latent = self.norm[i](latent)
                latent = self.upsample[i](latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)
    
class ConvMixerV4(nn.Module):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 1]
        # self.depth = [3, 3, 9, 3]

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=7)
        self.conv = nn.ModuleList()
        # self.attn = nn.ModuleList([])
        for i in range(4):

            attn = Attention(query_dim=self.hidden_dim,
                             context_dim=self.hidden_dim,
                             heads=1,
                             dim_head=self.hidden_dim,)
            # self.attn.append(attn)
            if i != 2:
                stage = nn.Sequential(
                    *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
                )
                self.conv.append(stage)
            else:
                stage = nn.ModuleList()
                for _ in range(self.depth[i] // self.depth[0]):
                    conv = nn.Sequential(
                        *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i] //3)]
                    )
                    stage.append(conv)
                self.conv.append(stage)

        count = self.depth[i] // self.depth[0]
        self.count = count

        self.attn = Attention(query_dim=self.hidden_dim,
                              context_dim=self.hidden_dim,
                              heads=1,
                              dim_head=self.hidden_dim,)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        x = self.patch_embed(x)

        for i in range(4):
            if i != 2:
                x = self.conv[i](x)
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                # latent = self.attn[i](latent, context) + latent
                latent = self.attn(latent, context) + latent
                latent = self.norm(latent)
            else:
                for conv in self.conv[i]:
                    x = conv(x)
                    context = x.permute(0, 2, 3, 1)
                    context = rearrange(context, 'b ... d -> b (...) d')
                    # latent = self.attn[i](latent, context) + latent
                    latent = self.attn(latent, context) + latent
                    latent = self.norm(latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)
    

class ConvMixerV2_1(nn.Module):
    # def __init__(self, cfg:ConvMixerCfg):
    #     super().__init__()
    #     self.cfg = cfg
    #     self.hidden_dim = cfg.hidden_dim
    #     # self.patch_size = [4, 2, 2, 2]
    #     # self.depth = [2, 2, 6, 2]
    #     self.depth = [1, 2, 3, 1]

    #     self.downsample = nn.ModuleList()

    #     self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
    #                                    patch_size=7)
    #     # self.downsample.append(self.patch_embed)
    #     # for i in range(3):
    #     #     self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=2))

    #     self.conv = nn.ModuleList()
    #     self.attn = nn.ModuleList([])
    #     for i in range(4):

    #         attn = Attention(query_dim=self.hidden_dim,
    #                          context_dim=self.hidden_dim,
    #                          heads=1,
    #                          dim_head=self.hidden_dim,)
    #         self.attn.append(attn)
    #         # if i != 2:
    #         #     stage = nn.Sequential(
    #         #         *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
    #         #     )
    #         #     self.conv.append(stage)
    #         # else:
    #         #     stage = nn.ModuleList()
    #         #     for j in range(self.depth[i] // self.depth[0]):
    #         #         conv = nn.Sequential(
    #         #             *[block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i] //3)]
    #         #         )
    #         #         stage.append(conv)
    #         #     self.conv.append(stage)
    #         conv1 = nn.ModuleList([])
    #         for _ in range(self.depth[i]):
    #             conv2 = block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate)
    #             conv1.append(conv2)
    #         self.conv.append(conv1)

    #     count = self.depth[i] // self.depth[0]
    #     self.count = count

    #     self.digup = nn.Sequential(
    #         nn.AdaptiveAvgPool2d((1, 1)),
    #         nn.Flatten(),
    #         nn.LayerNorm(self.hidden_dim),
    #     )
    #     self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
    #     self.norm = nn.LayerNorm(self.hidden_dim)
    #     self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    # def forward(self, x):
    #     B, _, H, W = x.shape
    #     latent = repeat(self.latent, 'n d -> b n d', b=B)

    #     x = self.patch_embed(x)

    #     for i in range(4):
    #         # if i != 2:
    #         #     x = self.downsample[i](x)
    #         #     x = self.conv[i](x)
    #         #     context = x.permute(0, 2, 3, 1)
    #         #     context = rearrange(context, 'b ... d -> b (...) d')
    #         #     latent = self.attn[i](latent, context) + latent
    #         #     latent = self.norm(latent)
    #         # else:
    #         #     x = self.downsample[i](x)
    #         #     for conv in self.conv[i]:
    #         #         x = conv(x)
    #         #         context = x.permute(0, 2, 3, 1)
    #         #         context = rearrange(context, 'b ... d -> b (...) d')
    #         #         latent = self.attn[i](latent, context) + latent
    #         #         latent = self.norm(latent)
    #         for conv in self.conv[i]:
    #             x = conv(x)
    #             context = x.permute(0, 2, 3, 1)
    #             context = rearrange(context, 'b ... d -> b (...) d')
    #             latent = self.attn[i](latent, context) + latent
    #             latent = self.norm(latent)

    #     x = self.digup(x)
    #     latent = reduce(latent, 'b n d -> b d', 'mean')
    #     return self.fc(latent + x)
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        # self.patch_size = [4, 2, 2, 2]
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 1]
        # self.depth = [3, 3, 9, 3]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=4)
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=2))

        self.conv = nn.ModuleList([])
        # self.attn = nn.ModuleList([])
        for i in range(4):

            # attn = Attention(query_dim=self.hidden_dim,
            #                  context_dim=self.hidden_dim,
            #                  heads=1,
            #                  dim_head=self.hidden_dim,)
            # self.attn.append(attn)

            conv1 = nn.ModuleList([])
            for _ in range(self.depth[i]):
                conv2 = block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate)
                conv1.append(conv2)
            self.conv.append(conv1)

        count = self.depth[i] // self.depth[0]
        self.count = count

        self.attn = Attention(query_dim=self.hidden_dim,
                              context_dim=self.hidden_dim,
                              heads=1,
                              dim_head=self.hidden_dim,)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            x = self.downsample[i](x)
            for conv in self.conv[i]:
                x = conv(x)
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn(latent, context) + latent
                latent = self.norm(latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)
    
class ConvMixerV2_2(nn.Module):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        # self.patch_size = [4, 2, 2, 2]
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 1]
        # self.depth = [3, 3, 9, 3]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=4)
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=2))

        self.conv = nn.ModuleList([])
        self.attn = nn.ModuleList([])
        for i in range(4):

            attn = Attention(query_dim=self.hidden_dim,
                             context_dim=self.hidden_dim,
                             heads=1,
                             dim_head=self.hidden_dim,)
            self.attn.append(attn)

            conv1 = nn.ModuleList([])
            for _ in range(self.depth[i]):
                conv2 = block2(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate)
                conv1.append(conv2)
            self.conv.append(conv1)

        count = self.depth[i] // self.depth[0]
        self.count = count

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            x = self.downsample[i](x)
            for conv in self.conv[i]:
                x = conv(x)
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn[i](latent, context) + latent
                latent = self.norm(latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        # return self.fc(latent + x)
        return self.fc(latent)
        

class ConvMixerV3_1(nn.Module):
    '''
        pico:   c = [64, 128, 256, 512], B = [2, 2, 6, 2]
        ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)
        ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)
        ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
        ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)
        ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)

    '''
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = [96, 192, 384, 768]
        self.patch_size = [4, 2, 2, 2]
        self.depth = [3, 3, 9, 3]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim[0],
                                      patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(
                PatchEmbed(in_channels=self.hidden_dim[i], hidden_dim=self.hidden_dim[i+1], patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        self.attn = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        for i in range(4):

            attn = Attention(query_dim=self.hidden_dim[-1],
                             context_dim=self.hidden_dim[-1],
                             heads=1,
                             dim_head=self.hidden_dim[-1],)
            self.attn.append(attn)

            norm = nn.LayerNorm(self.hidden_dim[-1])
            self.norm.append(norm)

            if i != 2:
                stage = nn.Sequential(
                    *[block(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
                )
                self.conv.append(stage)
            else:
                stage = nn.ModuleList()
                for j in range(self.depth[i] // self.depth[0]):
                    stage_j = nn.Sequential(
                        *[block(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate)
                          for _ in range(self.depth[i]//3)],
                    )
                    stage.append(stage_j)
                self.conv.append(stage)

        self.count = self.depth[2] // self.depth[0]

        self.upsample = nn.ModuleList()
        # for i in range(3):
        #     upsample = nn.Sequential(
        #         nn.Linear(self.hidden_dim[i], self.hidden_dim[-1]),
        #         nn.LayerNorm(self.hidden_dim[-1])
        #     )
        #     self.upsample.append(upsample)
        # self.upsample.append(nn.Identity())
        for i in range(3):
            upsample = nn.Sequential(
                nn.Conv2d(self.hidden_dim[i], self.hidden_dim[-1], 1),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_dim[-1])
            )
            self.upsample.append(upsample)
        self.upsample.append(nn.Identity())

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim[3]),
        )
        self.fc = nn.Linear(self.hidden_dim[3], cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim[-1]))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            if i != 2:
                x = self.downsample[i](x)
                x = self.conv[i](x)
                context = self.upsample[i](x)
                context = context.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                # context = self.upsample[i](context)
                latent = self.attn[i](latent, context) + latent
                latent = self.norm[i](latent)
            else:
                x = self.downsample[i](x)
                for conv in self.conv[i]:
                    x = conv(x)
                    context = self.upsample[i](x)
                    context = context.permute(0, 2, 3, 1)
                    context = rearrange(context, 'b ... d -> b (...) d')
                    # context = self.upsample[i](context)
                    latent = self.attn[i](latent, context) + latent
                    latent = self.norm[i](latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)


class ConvMixerV3_2(nn.Module):
    '''
        pico:   c = [64, 128, 256, 512], B = [2, 2, 6, 2]
        ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)
        ConvNeXt-S: C = (96, 192, 384, 768), B = (3, 3, 27, 3)
        ConvNeXt-B: C = (128, 256, 512, 1024), B = (3, 3, 27, 3)
        ConvNeXt-L: C = (192, 384, 768, 1536), B = (3, 3, 27, 3)
        ConvNeXt-XL: C = (256, 512, 1024, 2048), B = (3, 3, 27, 3)

    '''
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
                PatchEmbed(in_channels=self.hidden_dim[i], hidden_dim=self.hidden_dim[i+1], patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        self.attn = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        for i in range(4):

            attn = Attention(query_dim=self.hidden_dim[-1],
                             context_dim=self.hidden_dim[-1],
                             heads=1,
                             dim_head=self.hidden_dim[-1],)
            self.attn.append(attn)

            norm = nn.LayerNorm(self.hidden_dim[-1])
            self.norm.append(norm)

            if i != 2:
                stage = nn.Sequential(
                    *[block2(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
                )
                self.conv.append(stage)
            else:
                stage = nn.ModuleList()
                for j in range(self.depth[i] // self.depth[0]):
                    stage_j = nn.Sequential(
                        *[block2(hidden_dim=self.hidden_dim[i], drop_rate=cfg.drop_rate)
                          for _ in range(self.depth[i]//3)],
                    )
                    stage.append(stage_j)
                self.conv.append(stage)

        self.count = self.depth[2] // self.depth[0]

        self.upsample = nn.ModuleList()
        # for i in range(3):
        #     upsample = nn.Sequential(
        #         nn.Linear(self.hidden_dim[i], self.hidden_dim[-1]),
        #         nn.LayerNorm(self.hidden_dim[-1])
        #     )
        #     self.upsample.append(upsample)
        # self.upsample.append(nn.Identity())
        for i in range(3):
            upsample = nn.Sequential(
                nn.Conv2d(self.hidden_dim[i], self.hidden_dim[-1], 1),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_dim[-1])
            )
            self.upsample.append(upsample)
        self.upsample.append(nn.Identity())

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim[-1]),
        )
        self.fc = nn.Linear(self.hidden_dim[-1], cfg.num_classes)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim[-1]))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            if i != 2:
                x = self.downsample[i](x)
                x = self.conv[i](x)

                context = self.upsample[i](x)

                context = context.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                # context = self.upsample[i](context)
                latent = self.attn[i](latent, context) + latent
                latent = self.norm[i](latent)
            else:
                x = self.downsample[i](x)
                for conv in self.conv[i]:
                    x = conv(x)
                    
                    context = self.upsample[i](x)

                    context = context.permute(0, 2, 3, 1)
                    context = rearrange(context, 'b ... d -> b (...) d')
                    # context = self.upsample[i](context)
                    latent = self.attn[i](latent, context) + latent
                    latent = self.norm[i](latent)

        x = self.digup(x)
        latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent + x)







