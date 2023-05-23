# %%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from sunyata.pytorch.arch.attentionpool import AvgAttnPooling2d, AvgAttnPooling2dS

from sunyata.pytorch.arch.base import BaseCfg, ConvMixerLayer, ConvMixerLayer2, ConvMixerLayereca, ecablock


# %%
class eca_layer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        #  (batch_size, channels, 1, 1)
        y = self.avg_pool(x)
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        #  squeeze：(batch_size, channels, 1)变为(batch_size, channels)
        y = y.transpose(-1, -2).squeeze(-1)
        return y

class eca_layer2(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super(eca_layer2, self).__init__()
        self.attn_pool = AvgAttnPooling2dS(dim=dim)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)
        
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4
        #  (batch_size, channels, 1, 1)
        y = self.attn_pool(x)
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        #  squeeze：(batch_size, channels, 1)变为(batch_size, channels)
        y = y.transpose(-1, -2).squeeze(-1)
        return y
@dataclass
class ConvMixerCfg(BaseCfg):
    num_layers: int = 8
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    squeeze_factor: int = 4

    layer_norm_zero_init: bool = True
    skip_connection: bool = True

    eca_kernel_size: int = 3

# %%


class ConvMixer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.digup(x)
        return x


class Isotropic2(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        self.layers = nn.Sequential(*[
            ConvMixerLayer2(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x= self.layers(x)
        x= self.digup(x)
        return x

class ConvMixereca(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])
        self.eca = ecablock(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        # x = self.layers(x)
        for layer in self.layers:
            x1 = layer(x)
            x = self.eca(x1) + x
            # x = x + layer(x)
        x = self.digup(x)
        return x


class ConvMixerattn(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)
        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size,
                      stride=cfg.patch_size),
            nn.GELU(),
            # eps>6.1e-5 to avoid nan in half precision
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),
        )

        self.digup = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            AvgAttnPooling2dS(dim=cfg.hidden_dim),
            nn.Flatten(),
            # eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        # x = self.layers(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.digup(x)
        return x


class BayesConvMixer(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        # self.digup = eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        # logits = self.logits
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

class BayesConvMixer2(Isotropic2):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        # self.digup = eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        # logits = self.logits
        for layer in self.layers:
            x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

class CombineConvMixer(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        # self.digup = eca_layer(kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        logits = self.logits_layer_norm(logits)
        for i, layer in enumerate(self.layers):
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            
            if i == 0:
                logit = self.digup(x) + logits
            else:
                logit = logits + self.digup(x)
            
            logits = self.logits_layer_norm(logit) + logits
            # logits = self.logits_layer_norm(self.digup(
            #     x) if i == 0 else logits + self.digup(x)) + logits
            # logits = logits + self.digup(x)
            # logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits


class BayesConvMixereca(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        # logits = self.logits
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits


class BayesConvMixerattn(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        # self.digup = eca_layer(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.digup = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            AvgAttnPooling2dS(dim=cfg.hidden_dim),
            nn.Flatten(),
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        # logits = self.logits
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits
   
    
class BayesConvMixerecaAttn(ConvMixer):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(
                self.logits_layer_norm.weight.data.shape)

        self.digup = eca_layer2(dim=cfg.hidden_dim, kernel_size=cfg.eca_kernel_size)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.skip_connection = cfg.skip_connection

        # logits = torch.zeros(1, cfg.hidden_dim)
        # self.register_buffer('logits', logits)

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        # logits = self.logits
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
                x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits
    
