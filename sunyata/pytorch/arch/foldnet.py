from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg, Residual, ConvMixerLayer
from sunyata.pytorch_lightning.base import BaseModule

# from sunyata.pytorch.arch.convnext import BottleNeckBlock
from sunyata.pytorch.arch.patchconvnet import PatchConvBlock


@dataclass
class FoldNetCfg(BaseCfg):
    block: nn.Module = None

    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    fold_num: int = 1
    drop_rate: float = 0.
    expansion: int = 1
    layer_scaler_init_value: float = 1e-6


class ResConvMixer(BaseModule):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)

        self.layers = nn.Sequential(*[
            Residual(nn.Sequential(
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, groups=cfg.hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim),
                nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(cfg.hidden_dim)
            )
            ) for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        x= self.layers(x)
        x= self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss


class Block2(nn.Sequential):
    def __init__(self, hidden_dim: int, kernel_size: int, drop_rate: float=0.):
        super().__init__(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding="same"),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.Dropout(drop_rate)
        )


class FoldBlock(nn.Module):
    "Basic block of folded ResNet"
    def __init__(self, fold_num:int, Unit:nn.Module, *args, **kwargs):  # , hidden_dim: int, kernel_size: int, drop_rate:float=0.
        super(FoldBlock, self).__init__()
        self.fold_num = fold_num
        units = []
        for i in range(max(1, fold_num - 1)):
            units += [Unit(*args, **kwargs)]
        self.units = nn.ModuleList(units)
        
    def forward(self, *xs):
        xs = list(xs)
        if self.fold_num == 1:
            xs[0] = xs[0] + self.units[0](xs[0])
            return xs
        for i in range(self.fold_num - 1):
            xs[i+1] = xs[i+1] + self.units[i](xs[i])
        xs.reverse()
        return xs


class FoldNet(BaseModule):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)
        
        if cfg.block == ConvMixerLayer or cfg.block == Block2:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
                for _ in range(cfg.num_layers)
            ])
        elif cfg.block == BottleNeckBlock:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, in_features = cfg.hidden_dim, out_features = cfg.hidden_dim,
                                kernel_size = cfg.kernel_size, expansion = cfg.expansion, drop_p = cfg.drop_rate, layer_scaler_init_value = cfg.layer_scaler_init_value
                )
                for _ in range(cfg.num_layers)
            ])
        elif cfg.block == PatchConvBlock:
            self.layers = nn.ModuleList([
                FoldBlock(cfg.fold_num, cfg.block, cfg.hidden_dim, cfg.drop_rate, cfg.layer_scaler_init_value)
                for _ in range(cfg.num_layers)
            ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        xs = [x for _ in range(self.cfg.fold_num)]
        for layer in self.layers:
            xs= layer(*xs)
        x = xs[-1]
        x = self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss


class FoldNetRepeat(FoldNet):
    def __init__(self, cfg:FoldNetCfg):
        super().__init__(cfg)
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim * cfg.fold_num, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        xs = x.repeat(1, self.cfg.fold_num, 1, 1)
        xs = torch.chunk(xs, self.cfg.fold_num, dim = 1)
        for layer in self.layers:
            xs = layer(*xs)
        xs = torch.cat(xs, dim = 1)
        x = self.digup(xs)
        return x


class FoldNetRepeat2(FoldNet):
    def forward(self, x):
        x = self.embed(x)
        xs = x.repeat(1, self.cfg.fold_num, 1, 1)
        xs = torch.chunk(xs, self.cfg.fold_num, dim = 1)
        for layer in self.layers:
            xs = layer(*xs)
        x = xs[-1]
        x = self.digup(x)
        return x