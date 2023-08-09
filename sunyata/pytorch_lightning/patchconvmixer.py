from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import pytorch_lightning as pl
from sunyata.pytorch.arch.base import BaseCfg, Residual
from sunyata.pytorch_lightning.base import BaseModule, ClassifierModule

from sunyata.pytorch.arch.convmixer import ConvMixerCfg
from sunyata.pytorch.arch.conv_former import PatchConvMixerV0, PatchConvMixerV1


class PlPatchConvMixerV0(ClassifierModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlPatchConvMixerV0, self).__init__(cfg)
        self.convmixer = PatchConvMixerV0(cfg)
    
    def forward(self, x):
        return self.convmixer(x)
    

class PlPatchConvMixerV1(ClassifierModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlPatchConvMixerV1, self).__init__(cfg)
        self.convmixer = PatchConvMixerV1(cfg)

    def forward(self, x):     
        return self.convmixer(x)