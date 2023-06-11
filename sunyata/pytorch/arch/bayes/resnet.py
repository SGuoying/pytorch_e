# %%
from typing import Any, Callable, List, Optional, Tuple, Type, Union
from einops import rearrange, repeat
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from sunyata.pytorch.layer.attention import Attention


# %%
class BayesResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        avgpool1 = nn.AdaptiveAvgPool2d((2, 4))
        avgpool2 = nn.AdaptiveAvgPool2d((2, 2))
        avgpool3 = nn.AdaptiveAvgPool2d((2, 1))
        self.avgpools = nn.ModuleList([
            avgpool1,
            avgpool2, 
            avgpool3,
            self.avgpool,
        ])
        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        self.logits_bias = Parameter(torch.zeros(1, num_classes))

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.avgpools[i](x)
                logits = torch.flatten(logits, start_dim=1)
                logits = self.fc(logits)
                log_prior = log_prior + logits
                log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                log_prior = F.log_softmax(log_prior, dim=-1)
        return log_prior

# %%
class BayesResNet2(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        expansion = block.expansion
        self.digups = nn.ModuleList([
            *[nn.Sequential(
                nn.Conv2d(64 * i * expansion, 2048, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                self.avgpool,
                nn.Flatten(),
                # self.fc,
            )
        ])

        log_prior = torch.zeros(1, 2048)
        self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(2048)
        # self.logits_bias = Parameter(torch.zeros(1, num_classes), requires_grad=True)

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)
        # log_priors = torch.empty(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.digups[i](x)
                log_prior = log_prior + logits
                log_prior = self.logits_layer_norm(log_prior)
                # log_priors = torch.cat([log_priors, log_prior])
                # log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                # log_prior = F.log_softmax(log_prior, dim=-1)
        return self.fc(log_prior)


class eca_layer(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).squeeze(-1)
        return y
    

# %%
class ResNet2(ResNet):
   def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        expansion = block.expansion
        self.digups = nn.ModuleList([
            *[nn.Sequential(
                nn.Conv2d(64 * i * expansion, 2048, kernel_size=1),
                # eca_layer(3),
                nn.AdaptiveAvgPool2d((1, 1)),
                # nn.Flatten(),
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                # eca_layer(3),
                self.avgpool,
                # nn.Flatten(),
            )
        ])
        self.attn = Attention(2048, 2048, 1, 2048)
        self.fc = nn.Sequential(

        )

        # log_prior = torch.zeros(1, 2048)
        # self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(2048)
        # self.logits_bias = Parameter(torch.zeros(1, num_classes), requires_grad=True)

        self.latent = nn.Parameter(torch.randn(1, 2048))
   def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        # log_prior = self.log_prior.repeat(batch_size, 1)
        latent = repeat(self.latent, 'n d -> b n d', b = batch_size)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        input = input = x.permute(0, 2, 3, 1)
        input = rearrange(input, 'b ... d -> b (...) d')
        latent = self.attn(latent, input) + latent
        latent = self.logits_layer_norm(latent)


        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.digups[i](x)
                input = input = x.permute(0, 2, 3, 1)
                input = rearrange(input, 'b ... d -> b (...) d')
                logits = self.attn(logits, input) + logits
                logits = self.logits_layer_norm(logits)
        latent = nn.Flatten()(latent)        
        return self.fc(logits)



def Resnet50(num_classes=100, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def bayesResnet50(num_classes=100, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet2(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def bayesResnet(num_classes=100, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return BayesResNet2(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)