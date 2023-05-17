import math
from typing import Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision.ops import StochasticDepth

from sunyata.pytorch.arch.attentionpool import AvgAttnPooling2dS


class RevSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
    ):
        # self.params = params
        # self.lr = lr
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure: Callable=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.add_(p.grad.data * group["lr"])

        return loss

@dataclass
class BaseCfg:
    batch_size: int = 16

    num_layers: int = 8
    
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = None
    optimizer_method: str = "Adam"
    learning_rate_scheduler: str = "CosineAnnealing"
    warmup_epochs: int = None
    warmup_start_lr: float = None
    steps_per_epoch: int = None
    last_epoch: int = -1



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def set_requires_grad(model:nn.Module, val: bool):
    for p in model.parameters():
        p.requires_grad = val

            
class LayerScaler(nn.Module):
    def __init__(self, dim: int, init_scale: float):
        super().__init__()
        self.gamma = nn.Parameter(init_scale * torch.ones(dim))

    def forward(self, x):
        return self.gamma[None,...] * x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class SE(nn.Module):
    def __init__(self, hidden_dim: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_c = hidden_dim // squeeze_factor
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
			nn.Conv2d(hidden_dim, squeeze_c, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(squeeze_c , hidden_dim, 1),
			nn.Sigmoid())
        
    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale   
    

class eca_layer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super(eca_layer, self).__init__()
        self.attn_pool = AvgAttnPooling2dS(dim=dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        #  (batch_size, channels, 1, 1)
        y = self.avg_pool(x)
        # y = self.attn_pool(x)
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        #  squeeze：(batch_size, channels, 1)变为(batch_size, channels)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
class eca_layer2(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x
class ConvMixerLayer(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            # nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            # nn.Dropout(drop_rate)
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )

class ConvMixerLayereca(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            # nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim, eps=7e-5),
            # nn.Dropout(drop_rate)
            eca_layer2(hidden_dim, kernel_size=3),
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )
        
class ConvMixerLayer2(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
            Residual(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
            )),
            SE(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim), 
            StochasticDepth(drop_rate, 'row') if drop_rate > 0. else nn.Identity(),
        )

        
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

class LayerNorm(nn.Module):
    def __init__(self, dim, scale = True):
        super().__init__()
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)

