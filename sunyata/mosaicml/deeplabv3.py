# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""DeepLabV3 model extending :class:`.ComposerClassifier`."""

import functools
import textwrap
import warnings
from typing import Callable, Dict, Optional

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import torchvision
from composer.loss import DiceLoss, soft_cross_entropy
from composer.metrics import CrossEntropy, MIoU
from composer.models.tasks import ComposerClassifier
from composer.utils import dist
from packaging import version
from torchmetrics import MetricCollection
from torchvision.models import _utils, resnet

from sunyata.pytorch.arch.deeplabv3 import DeepLabHead, IntermediateLayerGetter

__all__ = ['deeplabv3', 'build_composer_deeplabv3']


class SimpleSegmentationModel(torch.nn.Module):

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(tuple(features.values()))
        logits = F.interpolate(logits,
                               size=input_shape,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=False)
        return logits


def deeplabv3(num_classes: int,
              backbone_arch: str = 'resnet50',
              backbone_weights: Optional[str] = None,
              ):
   

    if version.parse(torchvision.__version__) < version.parse('0.13.0'):
        pretrained = False
        if backbone_weights:
            pretrained = True
            if backbone_weights == 'IMAGENET1K_V1':
                resnet.model_urls[
                    backbone_arch] = 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
            elif backbone_weights == 'IMAGENET1K_V2':
                resnet.model_urls[
                    backbone_arch] = 'https://download.pytorch.org/models/resnet101-cd907fc2.pth'
            else:
                ValueError(
                    textwrap.dedent(f"""\
                        `backbone_weights` must be either "IMAGENET1K_V1" or "IMAGENET1K_V2"
                        if torchvision.__version__ < 0.13.0. `backbone_weights` was {backbone_weights}."""
                                   ))
        backbone = getattr(resnet, backbone_arch)(
            pretrained=pretrained,
            replace_stride_with_dilation=[False, True, True])
    else:
        backbone = getattr(resnet, backbone_arch)(
            weights=backbone_weights,
            replace_stride_with_dilation=[False, True, True])

    # specify which layers to extract activations from
    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    backbone =IntermediateLayerGetter(backbone, return_layers=return_layers)

    head = DeepLabHead(in_channels=out_inplanes,
                    num_classes=num_classes,)

    model = SimpleSegmentationModel(backbone, head)

    # Only apply initialization to classifier head if pre-trained weights are used

    return model


def build_composer_deeplabv3(num_classes: int,
                             backbone_arch: str = 'resnet50',
                             backbone_weights: Optional[str] = None,
                             ignore_index: int = -1,
                             cross_entropy_weight: float = 1.0,
                             dice_weight: float = 0.0,
                            ):
    """Create a :class:`.ComposerClassifier` for a DeepLabv3(+) model.

    Logs Mean Intersection over Union (MIoU) and Cross Entropy during training and validation.
    From `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`_
        (Chen et al, 2017).

    Args:
        num_classes (int): Number of classes in the segmentation task.
        backbone_arch (str, optional): The architecture to use for the backbone. Must be either
            ``'resnet50'`` or ``'resnet101'``. Default: ``'resnet101'``.
        backbone_weights (str, optional): If specified, the PyTorch pre-trained weights to load for the backbone.
            Currently, only ['IMAGENET1K_V1', 'IMAGENET1K_V2'] are supported. Default: ``None``.
        sync_bn (bool, optional): If ``True``, replace all BatchNorm layers with SyncBatchNorm layers.
            Default: ``True``.
        use_plus (bool, optional): If ``True``, use DeepLabv3+ head instead of DeepLabv3. Default: ``True``.
        ignore_index (int): Class label to ignore when calculating the loss and other metrics. Default: ``-1``.
        cross_entropy_weight (float): Weight to scale the cross entropy loss. Default: ``1.0``.
        dice_weight (float): Weight to scale the dice loss. Default: ``0.0``.
        init_fn (Callable, optional): initialization function for the model. ``None`` for no initialization.
            Default: ``None``.

    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a DeepLabv3(+) model.

    Example:
    .. code-block:: python
        from composer.models import composer_deeplabv3
        model = composer_deeplabv3(num_classes=150, backbone_arch='resnet101', backbone_weights=None)
    """
    model = deeplabv3(backbone_arch=backbone_arch,
                      backbone_weights=backbone_weights,
                      num_classes=num_classes,
                    )

    train_metrics = MetricCollection([
        CrossEntropy(ignore_index=ignore_index),
        MIoU(num_classes, ignore_index=ignore_index)
    ])
    val_metrics = MetricCollection([
        CrossEntropy(ignore_index=ignore_index),
        MIoU(num_classes, ignore_index=ignore_index)
    ])

    ce_loss_fn = functools.partial(soft_cross_entropy,
                                   ignore_index=ignore_index)
    dice_loss_fn = DiceLoss(softmax=True,
                            batch=True,
                            ignore_absent_classes=True)

    def _combo_loss(output, target) -> Dict[str, torch.Tensor]:
        loss = {
            'total': torch.zeros(1, device=output.device, dtype=output.dtype)
        }
        if cross_entropy_weight:
            loss['cross_entropy'] = ce_loss_fn(output, target)
            loss['total'] += loss['cross_entropy'] * cross_entropy_weight
        if dice_weight:
            loss['dice'] = dice_loss_fn(output, target)
            loss['total'] += loss['dice'] * dice_weight
        return loss

    composer_model = ComposerClassifier(module=model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=_combo_loss)
    return composer_model