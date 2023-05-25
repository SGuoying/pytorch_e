# %%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import SE, BaseCfg, ConvMixerLayer, ConvMixerLayer2
from sunyata.pytorch_lightning.base import BaseModule


@dataclass
class IsotropicCfg(BaseCfg):
    hidden_dim: int = 128
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    drop_rate: float = 0.
    layer_norm_zero_init: bool = False    

# %%
class Isotropic(BaseModule):
    def __init__(self, cfg: IsotropicCfg):
        super().__init__(cfg)

        self.layers = nn.ModuleList([
            ConvMixerLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])

        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim, eps=7e-5),  # eps>6.1e-5 to avoid nan in half precision
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
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

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss    

# %%
class Isotropic2(Isotropic):
    def __init__(self, cfg: IsotropicCfg):
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


# %%
class BayesIsotropic(Isotropic):
    def __init__(self, cfg: IsotropicCfg):
        super().__init__(cfg)

        self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        # self.logits_layer_norm.weight.data = torch.zeros(self.logits_layer_norm.weight.data.shape)
        if cfg.layer_norm_zero_init:
            self.logits_layer_norm.weight.data = torch.zeros(self.logits_layer_norm.weight.data.shape)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)


    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            x = x + layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

# %%
class BayesIsotropic2(Isotropic2):
    def __init__(self, cfg: IsotropicCfg):
        super().__init__(cfg)

        # self.logits_layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.logits_layer_norm = SE(hidden_dim=cfg.hidden_dim)
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            # nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )


    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            x = layer(x)
            logits = logits + self.digup(x)
            logits = self.logits_layer_norm(logits)
        logits = self.fc(logits)
        return logits

# %%
input = torch.randn(2, 3, 256, 256)
cfg = IsotropicCfg(
    patch_size = 8,
)
model = BayesIsotropic(cfg)

# model = Isotropic(cfg)
output = model(input)
output.shape

# %%
