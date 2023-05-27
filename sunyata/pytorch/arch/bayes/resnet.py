# %%
from typing import Any, Callable, List, Optional, Tuple, Type, Union
from einops import rearrange
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


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
    
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=3):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()
        
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        # self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        
        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        # x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = x
        
        return outputs

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
                spatial_attention(3),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                # eca_layer(3),
                spatial_attention(3),
                self.avgpool,
                nn.Flatten(),
            )
        ])

        log_prior = torch.zeros(1, 2048)
        self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(2048)
        # self.logits_bias = Parameter(torch.zeros(1, num_classes), requires_grad=True)

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
                logits = self.digups[i](x)
                log_prior = log_prior + logits
                log_prior = self.logits_layer_norm(log_prior)
        return self.fc(log_prior)



def Resnet50(num_classes=100, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def bayesResnet50(num_classes=100, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet2(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def bayesResnet(num_classes=100, **kwargs):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return BayesResNet2(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)