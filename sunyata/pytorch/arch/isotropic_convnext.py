from dataclasses import dataclass
from einops import repeat, rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


from sunyata.pytorch.arch.base import BaseCfg, Residual
from sunyata.pytorch.layer.attention import Attention
from sunyata.pytorch_lightning.base import ClassifierModule

from sunyata.pytorch.arch.convnext2 import Block, LayerNorm, ConvNeXtCfg



class ConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depth=18, dim=384, drop_path_rate=0., 
                 layer_scale_init_value=0., head_init_scale=1.,
                 ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i], 
                                    layer_scale_init_value=layer_scale_init_value)
                                    for i in range(depth)])

        self.norm = LayerNorm(dim, eps=1e-6) # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def isotropic_convnext(cfg:ConvNeXtCfg):
    arch_settings = {
        'small': {
            'depth': 18,
            'dim': 384
        },
        'base': {
            'depth': 18,
            'dim': 768
        },
        'large': {
            'depth': 36,
            'dim': 1024
        },
    }
    depth = arch_settings[cfg.arch_type]['depth']
    dim = arch_settings[cfg.arch_type]['dim']
    model = ConvNeXtIsotropic(num_classes=cfg.num_classes, depth=depth, dim=dim,
                     drop_path_rate=cfg.drop_path_rate, 
                     layer_scale_init_value=cfg.layer_scale_init_value,
                     head_init_scale=cfg.head_init_scale,
                     )
    model.depth = depth
    model.dim = dim
    return model


class IterAttnConvNeXtIsotropic(nn.Module):
    def __init__(self, cfg:ConvNeXtCfg):
        super().__init__()
        self.convnext = isotropic_convnext(cfg)
        self.dim = self.convnext.dim
        del self.convnext.norm

        self.digup = Attention(
            query_dim=self.dim,
            context_dim=self.dim,
            heads=1,
            dim_head=self.dim,
            scale= cfg.scale,
        )

        self.latent = nn.Parameter(torch.zeros(1, self.dim))
        self.logits_layer_norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.convnext.stem(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = latent + self.digup(latent, input)
        latent = self.logits_layer_norm(latent)

        for layer in self.convnext.blocks:
            x = layer(x)

            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = latent + self.digup(latent, input)
            latent = self.logits_layer_norm(latent)

        latent = nn.Flatten()(latent)
        logits = self.convnext.head(latent)
        return logits


class PlConvNeXtIsotropic(ClassifierModule):
    def __init__(self, cfg:ConvNeXtCfg):
        super(PlConvNeXtIsotropic, self).__init__(cfg)
        self.isotropic_convnext = isotropic_convnext(cfg)
    
    def forward(self, x):
        return self.isotropic_convnext(x)


class PlIterAttnConvNeXtIsotropic(ClassifierModule):
    def __init__(self, cfg:ConvNeXtCfg):
        super(PlIterAttnConvNeXtIsotropic, self).__init__(cfg)
        self.isotropic_convnext = IterAttnConvNeXtIsotropic(cfg)
    
    def forward(self, x):
        return self.isotropic_convnext(x)