from functools import reduce
import torch.nn as nn
from torch import Tensor
from typing import List


class ConvNormAct(nn.Sequential):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm = nn.BatchNorm2d,
        act = nn.ReLU,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act(),
        )


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        reduction: int = 4,
        stride: int = 1,
    ):
        super().__init__()
        reduced_features = out_features // reduction
        self.block = nn.Sequential(
            # wide -> narrow
            ConvNormAct(
                in_features, reduced_features, kernel_size=1, stride=stride, bias=False
            ),
            # narrow -> narrow
            ConvNormAct(reduced_features, reduced_features, kernel_size=3, bias=False),
            # narrow -> wide
            ConvNormAct(reduced_features, out_features, kernel_size=1, act=nn.Identity, bias=False),
        )
        self.shortcut = (
            nn.Sequential(
                ConvNormAct(
                    in_features, out_features, kernel_size=1, stride=stride, bias=False
                )
            )
            if in_features != out_features
            else nn.Identity()
        )

        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        res = self.shortcut(res)
        x = x + res
        x = self.act(x)
        return x


class ConvNextStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, stride: int = 2, **kwargs
    ):
        super().__init__(
            # downsample is done here
            BottleNeckBlock(in_features, out_features, stride=stride, **kwargs),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth - 1)
            ],
        )


class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            ConvNormAct(
                in_features, out_features, kernel_size=7, stride=2
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ),


class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))

        self.stages = nn.ModuleList(
            [
                ConvNextStage(stem_features, widths[0], depths[0], stride=1),
                *[
                    ConvNextStage(in_features, out_features, depth)
                    for (in_features, out_features), depth in zip(
                        in_out_widths, depths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
