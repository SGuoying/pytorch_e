from dataclasses import dataclass
from einops import repeat, rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from sunyata.pytorch.arch.base import BaseCfg, Residual
from sunyata.pytorch.layer.drop import DropPath
from sunyata.pytorch.layer.attention import Attention
from sunyata.pytorch_lightning.base import BaseModule, ClassifierModule

# copy from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0) # type: ignore

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.reduction = nn.Conv2d(4 * hidden_dim, out_dim, kernel_size=1)
        self.norm = LayerNorm(hidden_dim * 4, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]    # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]    # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]    # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]    # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.norm(x)

        x = self.reduction(x)

        return x
    
class ConvNeXtMerg(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                PatchMerging(hidden_dim=dims[i],
                            out_dim=dims[i+1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0) # type: ignore

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@dataclass
class ConvNeXtCfg(BaseCfg):
    num_classes: int = 100
    arch_type: str = 'pico'  # femto pico nano tiny small base large xlarge huge

    drop_path_rate: float = 0.  # drop path rate
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.

    scale: int = 1
    heads: int = 1

    type: str = 'standard'  # standard iter iter_attn
    

def convnext(cfg:ConvNeXtCfg):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    depths = arch_settings[cfg.arch_type]['depths']
    dims = arch_settings[cfg.arch_type]['channels']
    model = ConvNeXt(num_classes=cfg.num_classes, depths=depths, dims=dims,
                     drop_path_rate=cfg.drop_path_rate, 
                     layer_scale_init_value=cfg.layer_scale_init_value,
                     head_init_scale=cfg.head_init_scale,
                     )
    model.depths = depths
    model.dims = dims
    return model


def convnextmerg(cfg:ConvNeXtCfg):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    depths = arch_settings[cfg.arch_type]['depths']
    dims = arch_settings[cfg.arch_type]['channels']
    model = ConvNeXtMerg(num_classes=cfg.num_classes, depths=depths, dims=dims,
                     drop_path_rate=cfg.drop_path_rate, 
                     layer_scale_init_value=cfg.layer_scale_init_value,
                     head_init_scale=cfg.head_init_scale,
                     )
    model.depths = depths
    model.dims = dims
    return model

class IterAttnConvNeXt(nn.Module):
    def __init__(self, cfg:ConvNeXtCfg):
        super().__init__()
        self.convnext = convnext(cfg)
        self.dims = self.convnext.dims
        del self.convnext.norm

        self.digups = nn.ModuleList()
        for dim in self.dims:
            multiple = dim // self.dims[0]
            scale = dim ** -0.5 / (8 * multiple)
            digup = Attention(
                query_dim=self.dims[-1],
                context_dim=dim,
                heads=1,
                dim_head=self.dims[-1],
                scale= scale,
            )
            self.digups.append(digup)

        self.features = nn.Parameter(torch.zeros(1, self.dims[-1]))
        self.iter_layer_norm = nn.LayerNorm(self.dims[-1])


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        features = repeat(self.features, 'n d -> b n d', b = batch_size)

        for i, stage in enumerate(self.convnext.stages):
            x = self.convnext.downsample_layers[i](x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            features = features + self.digups[i](features, input)
            features = self.iter_layer_norm(features)

            for layer in stage:
                x = layer(x)
                input = x.permute(0, 2, 3, 1)
                input = rearrange(input, 'b ... d -> b (...) d')
                features = features + self.digups[i](features, input)
                features = self.iter_layer_norm(features)

        features = nn.Flatten()(features)
        logits = self.convnext.head(features)
        return logits


##########
class IterAttnConvNeXtMerg(nn.Module):
    def __init__(self, cfg:ConvNeXtCfg):
        super().__init__()
        self.convnext = convnextmerg(cfg)
        self.dims = self.convnext.dims
        del self.convnext.norm

        self.digups = nn.ModuleList()
        for dim in self.dims:
            multiple = dim // self.dims[0]
            scale = dim ** -0.5 / (8 * multiple)
            digup = Attention(
                query_dim=self.dims[-1],
                context_dim=dim,
                heads=1,
                dim_head=self.dims[-1],
                scale= scale,
            )
            self.digups.append(digup)

        self.features = nn.Parameter(torch.zeros(1, self.dims[-1]))
        self.iter_layer_norm = nn.LayerNorm(self.dims[-1])


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        features = repeat(self.features, 'n d -> b n d', b = batch_size)

        for i, stage in enumerate(self.convnext.stages):
            x = self.convnext.downsample_layers[i](x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            features = features + self.digups[i](features, input)
            features = self.iter_layer_norm(features)

            for layer in stage:
                x = layer(x)
                input = x.permute(0, 2, 3, 1)
                input = rearrange(input, 'b ... d -> b (...) d')
                features = features + self.digups[i](features, input)
                features = self.iter_layer_norm(features)

        features = nn.Flatten()(features)
        logits = self.convnext.head(features)
        return logits

class PlConvNeXt(ClassifierModule):
    def __init__(self, cfg:ConvNeXtCfg):
        super(PlConvNeXt, self).__init__(cfg)
        self.convnext = convnext(cfg)
    
    def forward(self, x):
        return self.convnext(x)


class PlIterAttnConvNeXt(ClassifierModule):
    def __init__(self, cfg:ConvNeXtCfg):
        super(PlIterAttnConvNeXt, self).__init__(cfg)
        self.convnext = IterAttnConvNeXt(cfg)
    
    def forward(self, x):
        return self.convnext(x)
    

######
class PlConvNeXtMerg(ClassifierModule):
    def __init__(self, cfg:ConvNeXtCfg):
        super(PlConvNeXtMerg, self).__init__(cfg)
        self.convnext = convnextmerg(cfg)
    
    def forward(self, x):
        return self.convnext(x)


class PlIterAttnConvNeXtMerg(ClassifierModule):
    def __init__(self, cfg:ConvNeXtCfg):
        super(PlIterAttnConvNeXtMerg, self).__init__(cfg)
        self.convnext = IterAttnConvNeXtMerg(cfg)
    
    def forward(self, x):
        return self.convnext(x)


def convnext_atto(**kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnext_femto(**kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnext_pico(**kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnext_nano(**kwargs):
    model = ConvNeXt(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnext_small(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

