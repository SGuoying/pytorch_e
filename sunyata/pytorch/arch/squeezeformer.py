from dataclasses import dataclass
from einops import repeat, rearrange
import os
import copy
import torch
import torch.nn as nn
from sunyata.pytorch.arch.base import BaseCfg
from sunyata.pytorch.arch.poolformer import PoolformerCfg
from sunyata.pytorch_lightning.base import BaseModule, ClassifierModule

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


# class Squeezeing(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         assert x.ndim == 4
#         #  (batch_size, channels, 1, 1)
#         y = self.squeeze(x)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)
class Squeezeing(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        #  (batch_size, channels, 1, 1)
        y = self.avg_pool(x)
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        # unsqueeze：(batch_size, channels, 1)变为(batch_size, channels, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class SqueezeFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, mlp_ratio=4., 
                 kernel_size=3,
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Squeezeing(kernel_size=kernel_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def squeeze_basic_blocks(dim, index, layers, 
                 mlp_ratio=4., kernel_size=3,
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(SqueezeFormerBlock(
            dim, mlp_ratio=mlp_ratio, kernel_size=kernel_size,
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class SqueezeFormer(nn.Module):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, kernel_size, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = squeeze_basic_blocks(embed_dims[i], i, layers, 
                                 kernel_size=kernel_size,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)

        
        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()

        self.apply(self.cls_init_weights)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)

        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)

        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out
    

def squeezeformer(cfg:PoolformerCfg):
    arch_settings ={
        'tiny': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 320],
            'mlp_ratios': [4, 4, 4, 4],
            'downsamples': [True, True, True, True]
        },
        'small': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'downsamples': [True, True, True, True]
        },
        'base': {
            'depths': [4, 4, 12, 4],
            'channels': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'downsamples': [True, True, True, True]
        },
    }

    layers = arch_settings[cfg.arch_type]['depths']
    embed_dims = arch_settings[cfg.arch_type]['channels']
    mlp_ratios = arch_settings[cfg.arch_type]['mlp_ratios']
    downsamples = arch_settings[cfg.arch_type]['downsamples']
    model = SqueezeFormer(layers=layers, embed_dims=embed_dims,
                          kernel_size=cfg.kernel_size,
                       mlp_ratios=mlp_ratios, downsamples=downsamples,
                       num_classes=cfg.num_classes,drop_rate=cfg.drop_rate,
                       drop_path_rate=cfg.drop_path_rate,)
    model.layers = layers
    model.embed_dims = embed_dims
    model.mlp_ratios = mlp_ratios
    model.downsamples = downsamples
    return model

class PlSqueezeFormer(ClassifierModule):
    def __init__(self, cfg: PoolformerCfg):
        super(PlSqueezeFormer, self).__init__(cfg)
        self.model = squeezeformer(cfg)

    def forward(self, x):
        return self.model(x)

