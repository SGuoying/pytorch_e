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



@dataclass
class ConvNeXtCfg(BaseCfg):
    num_classes: int = 100
    arch_type: str = 'atto'  # femto pico nano tiny small base large xlarge huge

    drop_path_rate: float = 0.  # drop path rate
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.

    scale: float = 1.
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


class IterAttnConvNeXt(nn.Module):
    def __init__(self, cfg:ConvNeXtCfg):
        super().__init__()
        self.convnext = convnext(cfg)
        self.dims = self.convnext.dims
        del self.convnext.norm

        self.digups = nn.ModuleList()
        for dim in self.dims:
            digup = Attention(
                query_dim=self.dims[-1],
                context_dim=dim,
                heads=1,
                dim_head=self.dims[-1],
                scale= cfg.scale,
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


# @dataclass
# class ConvNextCfg:
#     batch_size: int = 64 # Per GPU batch size
#     epochs: int = 300
#     update_freq: int = 1  # gradient accumulation steps

#     drop_path: float = 0.  # drop path rate
#     input_size: int = 224  # image input size
#     layer_scale_init_value: float = 1e-6

#     # EMA related parameters
#     model_ema: bool = False
#     model_ema_decay: float = 0.9999
#     model_ema_force_cpu: bool = False
#     model_ema_eval: bool = False  # using ema to eval during training

#     # Optimization parameters
#     opt: str = "adamw"  # Optimizer
#     opt_eps: float = 1e-8  # Optimizer Epsilon
#     opt_betas: float = None  # Optimizer Betas (default: None, use opt default)
#     clip_grad: float = None  # Clip gradient norm (default: None, no clipping)
#     momentum: float = 0.9  # SGD momentum (default: 0.9)
#     weight_decay: float = 0.05  #
#     weight_decay_end: float = None # Final value of the weight decay. We use a cosine schedule for WD

#     lr: float = 4e-3  # learning rate (default: 4e-3), with total batch size 4096
#     layer_decay: float = 1.0
#     min_lr: float = 1e-6  # lower lr bound for cyclic schedulers that hit 0 (1e-6)
#     warmup_epochs: int = 20 # epochs to warmup LR, if scheduler supports
#     warmup_steps: int = -1 # num of steps to warmup LR, will overload warmup_epochs if set > 0

#     # Augmentation parameters
#     color_jitter: float = 0.4 # Color jitter factor
#     aa: str = "rand-m9-mstd0.5-incl" # use AutoAugment policy. "v0" or "original"
#     smoothing: float = 0.1 # label smoothing
#     train_interpolation: str = 'bicubic' # training interpolation (random, bilinear, bicubic)

#     # Evaluation parameters
#     crop_pct: float = None

#     # Random Erase params
#     reprob: float = 0.25 # random erase prob
#     remode: str = 'pixel' # random erase mode
#     recount: int = 1 # random erase count
#     resplit: bool = False # do not random erase first (clean) augmentation split

#     # Mixup params
#     mixup: float = 0.8 # mixup alpha, mixup enabled if > 0
#     cutmix: float = 1.0 # cutmix alpha, cutmix enabled if > 0
#     cutmix_minmax: float = None # cutmix min/max ratio, overrides alpha and enables cutmix if set
#     mixup_prob: float = 1.0 # probability of performing mixup or cutmix when either/both is enabled
#     mixup_switch_prob: float = 0.5 # probability of switching to cutmix when both mixup and cutmix enabled
#     mixup_mode: str = 'batch' # how to apply mixup/cutmix params, Per 'batch', 'pair' or 'elem'

#     # Finetuning params
#     finetune: str = '' # finetune from checkpoint
#     head_init_scale: float = 1.0 # classifier head initial scale, typically adjusted in fine-tuning
#     # model_key, model_prefix

#     # Dataset params
#     data_path: str = None # dataset path
#     eval_data_path: str = None # evaluation dataset path
#     nb_classes: int = 1000 # number of the classification types
#     imagenet_default_mean_and_std: bool = True
#     data_set: str = 'IMNET'
#     output_dir: str = '' # where to save, empty for no saving
#     log_dir: str = None # where to tensorboard log
#     device: str = 'cuda' # device to use for training / testing
#     seed: int = 0

#     resume: str = '' # resume from checkpoint
#     auto_resume: bool = True
#     save_ckpt: bool = True
#     save_ckpt_freq: int = 1
#     save_ckpt_num: int = 3

#     start_epoch: int = 0
#     eval: bool = False # perform evaluation only
#     dist_eval: bool = True # enabling distributed evaluation
#     disable_eval: bool = False # disabling eval during training
#     num_workers: int = 10
#     pin_mem: bool = True
    



# class Layer(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int,
#         num_heads: int,
#         query_idx: int = -1,
#         temperature: float = 1.,
#         init_scale: float = 1.,
#         drop_path:float=0., 
#         layer_scale_init_value:float=1e-6,
#     ):
#         super().__init__()
#         self.attn_layer = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)
#         self.block = Block(hidden_dim, drop_path, layer_scale_init_value)

#     def forward(self, xs, all_squeezed):
#         x_new, all_squeezed = self.attn_layer(xs, all_squeezed)
#         # x_new shape (batch_size, hidden_dim, height, width)
#         x_next = self.block(x_new)
#         x_next = x_next.unsqueeze(0)
#         return torch.cat((xs, x_next), dim=0), all_squeezed


# class Stage(nn.Module):
#     def __init__(
#         self,
#         num_layers: int,
#         hidden_dim: int,
#         num_heads: int,
#         query_idx: int = -1,
#         temperature: float = 1.,
#         init_scale: float = 1.,
#         drop_paths:tuple = None, 
#         layer_scale_init_value:float=1e-6,
#     ):
#         super().__init__()
#         assert num_layers > 1
#         self.first_block = Block(hidden_dim, drop_paths[0], layer_scale_init_value)
#         self.first_squeeze = Squeeze(hidden_dim, init_scale)
#         self.layers = nn.ModuleList([
#             Layer(hidden_dim, num_heads, query_idx, temperature, init_scale, drop_paths[i + 1], layer_scale_init_value)
#             for i in range(num_layers - 1)
#         ])
#         self.final_attn = AttnLayer(hidden_dim, num_heads, query_idx, temperature, init_scale)

#     def forward(self, x):
#         xs = torch.stack([x, self.first_block(x)], dim=0)
#         all_squeezed = self.first_squeeze(x).unsqueeze(0)
#         for layer in self.layers:
#             xs, all_squeezed = layer(xs, all_squeezed)
#         x, all_squeezed = self.final_attn(xs, all_squeezed)
#         return x


# class AttnConvNext(ConvNeXt):
#     def __init__(
#         self,
#         in_chans=3,
#         num_classes=1000,
#         depths=[3, 3, 9, 3],
#         dims=[96, 192, 384, 768],
#         drop_path_rate=0.,
#         layer_scale_init_value=1e-6,
#         head_init_scale=1.,
#         num_heads: int = 1,
#         query_idx: int = -1,
#         temperature: float = 1.,
#         init_scale: float = 1.,

#     ):
#         super().__init__(in_chans,num_classes, depths, dims, drop_path_rate, layer_scale_init_value, head_init_scale)

#         self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
#         drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         for i in range(4):
#             stage = Stage(depths[i], dims[i], 
#                 num_heads, query_idx, temperature, init_scale, drop_rates[cur:cur+depths[i]], layer_scale_init_value)
#             self.stages.append(stage)
#             cur += depths[i]



# def attnconvnext_tiny(**kwargs):
#     model = AttnConvNext(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     return model