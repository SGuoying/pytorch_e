# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
from einops import rearrange, reduce, repeat

import torch
import torch.nn as nn
from mmcv.cnn.bricks import (Conv2dAdaptivePadding, build_activation_layer,
                             build_norm_layer)
from mmengine.utils import digit_version

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model import BaseModule

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvBlock(BaseModule):

    def __init__(self,
                 hidden_dim,
                 kernel_size=5,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ConvBlock, self).__init__(init_cfg=init_cfg)

        self.conv1 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, hidden_dim)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            # stride=stride,
            # groups=groups,
            padding=kernel_size // 2,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, hidden_dim)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, hidden_dim)[1]
        self.act3 = build_activation_layer(act_cfg)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    # def zero_init_last_bn(self):
    #     nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x) 
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.act3(x)

        return x


class AttnBlock(BaseModule):
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
    

@MODELS.register_module()
class Conformer(BaseBackbone):
    """ConvMixer.                              .

    A PyTorch implementation of : `Patches Are All You Need?
    <https://arxiv.org/pdf/2201.09792.pdf>`_

    Modified from the `official repo
    <https://github.com/locuslab/convmixer/blob/main/convmixer.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convmixer.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvMixer.arch_settings``. And if dict, it
            should include the following two keys:

            - embed_dims (int): The dimensions of patch embedding.
            - depth (int): Number of repetitions of ConvMixer Layer.
            - patch_size (int): The patch size.
            - kernel_size (int): The kernel size of depthwise conv layers.

            Defaults to '768/32'.
        in_channels (int): Number of input image channels. Defaults to 3.
        patch_size (int): The size of one patch in the patch embed layer.
            Defaults to 7.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): Initialization config dict.
    """
    arch_settings = {
        '256/8': {
            'hidden_dim': 256,
            'depth': 8,
            'patch_size': 7,
            'kernel_size': 5
        },
        '256/12': {
            'hidden_dim': 256,
            'depth': 12,
            'patch_size': 14,
            'kernel_size': 7
        },
        '512/12': {
            'hidden_dim': 512,
            'depth': 12,
            'patch_size': 7,
            'kernel_size': 7
        },
    }

    def __init__(self,
                 arch='768/32',
                 in_channels=3,
                 norm_cfg=dict(type='BN'),
                 last_norm=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 out_indices=-1,
                 frozen_stages=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            essential_keys = {
                'embed_dims', 'depth', 'patch_size', 'kernel_size'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'

        self.embed_dims = arch['hidden_dim']
        self.depth = arch['depth']
        self.patch_size = arch['patch_size']
        self.kernel_size = arch['kernel_size']
        self.act = build_activation_layer(act_cfg)

        # check out indices and frozen stages
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.depth + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Set stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.embed_dims,
                kernel_size=self.patch_size,
                stride=self.patch_size), self.act,
            build_norm_layer(norm_cfg, self.embed_dims)[1])

        # Repetitions of ConvMixer Layer
        self.stages = nn.ModuleList([
            ConvBlock(hidden_dim=self.embed_dims,kernel_size=self.kernel_size)
            for _ in range(self.depth)
        ])

        self.attns = AttnBlock(query_dim=self.embed_dims,
                               context_dim=self.embed_dims,
                               heads=1,
                               dim_head=self.embed_dims,
                               )
        self.latent = nn.Parameter(torch.randn(1, self.embed_dims))
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        self._freeze_stages()

    def forward(self, x):
        b, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=b)
        
        x = self.stem(x)
        input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
        latent = latent + self.attns(latent, input)
        # latent = rearrange(latent[:, 1:], 'b (h w) d -> b d h w', h=x.shape[2])
        latent = self.norm(latent)

        outs = []
        for i, stage in enumerate(self.stages):
            x = x + stage(x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            latent = torch.cat([latent[:, 0][:, None, :], input], dim=1)
            latent = latent + self.attns(latent, input)
            latent = self.norm(latent)
            if i in self.out_indices:
                outs.append(
                    reduce(latent, 'b n d -> b d', 'mean')
                    )

        # x = self.pooling(x).flatten(1)
        return tuple(outs)

    def train(self, mode=True):
        super(Conformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False