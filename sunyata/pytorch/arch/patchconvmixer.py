# %%
from dataclasses import dataclass
from typing import Optional
from einops import rearrange, repeat, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, Residual
from sunyata.pytorch_lightning.base import ClassifierModule, BaseModule
import pytorch_lightning as pl

from sunyata.pytorch.layer.attention import Attention, EfficientChannelAttention, AttentionWithoutParams

# %%
@dataclass
class ConvMixerCfg(BaseCfg):
    num_layers: int = 8
    hidden_dim: int = 256
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10

    scale: Optional[float] = None

    type: str = 'standard'  # 'iter',  'iter_attn'

    drop_rate: float = 0.    

    layer_norm_zero_init: bool = False
    skip_connection: bool = True

    eca_kernel_size: int = 3

    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int,hidden_dim: int,  patch_size: int):
        super().__init__()
        # self.image_size = image_size
        self.patch_size = patch_size

        self.embed = self.embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.image_size[0] and W == self.image_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.embed(x)
        return x
    

class block(nn.Module):
    def __init__(self, hidden_dim, kernel_size, drop_rate=0.):
        super().__init__()
        self.block = nn.Sequential(
            Residual(nn.Sequential(
                # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                # nn.GELU(),
                # nn.BatchNorm2d(hidden_dim), 
            
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
            )),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_dim), 
            nn.Dropout(drop_rate),
        )
    def forward(self, x):
        x = self.block(x)
        return x
    
    
class ConvMixer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.layers = nn.Sequential(*[
            block(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
            for _ in range(cfg.num_layers)
        ])
        
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),  # eps>6.1e-5 to avoid nan in half precision
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
    
    
class ConvMixer1(nn.Module):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = [4, 2, 2, 2]
        self.depth = [1, 2, 3, 2]
        # self.depth = [2, 2, 6, 2]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[block(hidden_dim=self.hidden_dim, kernel_size=cfg.kernel_size ,drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
            )
            self.conv.append(stage)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            x = self.conv[i](x)

        x = self.digup(x)
        return self.fc(x)
    
class ConvMixer2(nn.Module):
    def __init__(self, cfg:ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.patch_size = [4, 2, 2, 2]
        self.depth = [1, 2, 3, 2]
        # self.depth = [2, 2, 6, 2]
        # self.depth = [3, 3, 9, 3]

        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=self.patch_size[0])
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(PatchEmbed(in_channels=self.hidden_dim, hidden_dim=self.hidden_dim, patch_size=self.patch_size[i+1]))

        self.conv = nn.ModuleList()
        for i in range(4):
            conv = nn.ModuleList([])
            for _ in range(self.depth[i]):
                conv.append(
                    block(hidden_dim=self.hidden_dim, kernel_size=cfg.kernel_size, drop_rate=cfg.drop_rate)
                )
            self.conv.append(conv)

        self.attn = Attention(query_dim=self.hidden_dim,
                              context_dim=self.hidden_dim,
                              heads=1,
                              dim_head=self.hidden_dim,
                              scale=cfg.scale)
        
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            x = self.downsample[i](x)
            for conv in self.conv[i]:
                x = conv(x)
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn(latent, context) + latent
                latent = self.norm(latent)
        latent = nn.Flatten()(latent)
        # latent = reduce(latent, 'b n d -> b d', 'mean')
        return self.fc(latent)
    
class PatchMerging(nn.Module):
    def __init__(self, hidden_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.reduction = nn.Conv2d(4 * hidden_dim, hidden_dim, kernel_size=1)
        self.norm = norm_layer(4 * hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]    # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]    # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]    # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]    # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.norm(x)

        x = self.reduction(x)

        return x
    

class PatchConvMixer(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 2]
        # self.depth = [3, 3, 9, 3]
        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=4)
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(
                PatchMerging(hidden_dim=self.hidden_dim))
            
        self.conv = nn.ModuleList()
        for i in range(4):
            conv = nn.ModuleList([])
            for _ in range(self.depth[i]):
                conv.append(
                    block(hidden_dim=self.hidden_dim,kernel_size=cfg.kernel_size, drop_rate=cfg.drop_rate)
                )
            self.conv.append(conv)
            # conv = nn.Sequential(
            #     *[block(hidden_dim=self.hidden_dim, drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
            # )
            # self.conv.append(conv)

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample[i](x)
            for conv in self.conv[i]:
                x = conv(x)        
        x = self.digup(x)
        logits = self.fc(x)
        return logits
    

class PatchConvMixer1(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        # self.depth = [2, 2, 6, 2]
        self.depth = [1, 2, 3, 2]
        # self.depth = [3, 3, 9, 3]
        self.downsample = nn.ModuleList()

        self.patch_embed = PatchEmbed(in_channels=3, hidden_dim=self.hidden_dim,
                                       patch_size=4)
        self.downsample.append(self.patch_embed)
        for i in range(3):
            self.downsample.append(
                PatchMerging(hidden_dim=self.hidden_dim))
            
        self.conv = nn.ModuleList()
        for i in range(4):
            conv = nn.ModuleList([])
            for _ in range(self.depth[i]):
                conv.append(
                    block(hidden_dim=self.hidden_dim, kernel_size=cfg.kernel_size, drop_rate=cfg.drop_rate)
                )
            self.conv.append(conv)
            # conv = nn.Sequential(
            #     *[block(hidden_dim=self.hidden_dim, kernel_size=cfg.kernel_size, drop_rate=cfg.drop_rate) for _ in range(self.depth[i])]
            # )
            # self.conv.append(conv)

        self.attn = Attention(query_dim=self.hidden_dim,
                              context_dim=self.hidden_dim,
                              heads=1,
                              dim_head=self.hidden_dim,
                              scale=cfg.scale)
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc = nn.Linear(self.hidden_dim, cfg.num_classes)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.latent = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, x):
        B, _, _, _ = x.shape
        latent = repeat(self.latent, 'n d -> b n d', b=B)

        for i in range(4):
            x = self.downsample[i](x)
            for conv in self.conv[i]:
                x = conv(x)
                
                context = x.permute(0, 2, 3, 1)
                context = rearrange(context, 'b ... d -> b (...) d')
                latent = self.attn(latent, context) + latent
                latent = self.norm(latent)

        # latent = reduce(latent, 'b n d -> b d', 'mean')
        latent = nn.Flatten()(latent)
        logits = self.fc(latent)
        return logits

    
    
class PlConvMixer(pl.LightningModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlConvMixer, self).__init__()
        self.save_hyperparameters("cfg")
        self.cfg = cfg
        
        self.model = ConvMixer(cfg)
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                # [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW and Lamb optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from sunyata.pytorch.lr_scheduler import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]    
    
class PlConvMixer1(pl.LightningModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlConvMixer1, self).__init__()
        self.model = ConvMixer1(cfg)
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                # [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW and Lamb optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from sunyata.pytorch.lr_scheduler import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]
    
    
class PlConvMixer2(pl.LightningModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlConvMixer2, self).__init__()
        self.model = ConvMixer2(cfg)
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                # [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW and Lamb optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from sunyata.pytorch.lr_scheduler import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]
    
    
class PlPatchConvMixer(pl.LightningModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlPatchConvMixer, self).__init__()
        self.model = PatchConvMixer(cfg)
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                # [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW and Lamb optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from sunyata.pytorch.lr_scheduler import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]
    
class PlPatchConvMixer1(pl.LightningModule):
    def __init__(self, cfg:ConvMixerCfg):
        super(PlPatchConvMixer1, self).__init__()
        self.model = PatchConvMixer1(cfg)
    
    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode="val")

    def configure_optimizers(self):
        if self.cfg.optimizer_method == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_method == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                # [{'params': self.parameters(), 'initial_lr': self.cfg.learning_rate}], 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay)
        else:
            raise Exception("Only supportSGD, Adam and AdamW and Lamb optimizer till now.")

        if self.cfg.learning_rate_scheduler == "CosineAnnealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "OneCycleLR":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.learning_rate,
                steps_per_epoch=self.cfg.steps_per_epoch, epochs=self.cfg.num_epochs)
        elif self.cfg.learning_rate_scheduler == "LinearWarmupCosineAnnealingLR":
            from sunyata.pytorch.lr_scheduler import LinearWarmupCosineAnnealingLR
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=self.cfg.warmup_epochs, max_epochs=self.cfg.num_epochs,
                warmup_start_lr=self.cfg.warmup_start_lr)
        else:
            lr_scheduler = None

        if lr_scheduler is None:
            return optimizer
        else:
            return [optimizer], [lr_scheduler]