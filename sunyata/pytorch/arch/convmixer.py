# %%
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from sunyata.pytorch.arch.attentionpool import AvgAttnPooling2d, AvgAttnPooling2dS

from sunyata.pytorch.arch.base import SE, BaseCfg, ConvMixerLayer, ConvMixerLayer2, ConvMixerLayer3, ecablock
from sunyata.pytorch.arch.van import attention


# %%
class eca_layer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super(eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2, bias=False)

    def forward(self, x: torch.Tensor):  # x: (batch_size, channels)
        assert x.ndim == 2
        #  (batch_size, channels, 1, 1)
        # y = self.avg_pool(x)
        y = self.conv(x.unsqueeze(-1).transpose(-1,-2))
        # squeeze： (batch_size, channels, 1, 1)变为(batch_size, channels, 1)，
        # transpose：从(batch_size, channels, 1)变为(batch_size, 1, channels)
        y = y.transpose(-1,-2).squeeze(-1)
        # transpose： (batch_size, 1, channels)变为(batch_size, channels, 1)，
        #  squeeze：(batch_size, channels, 1)变为(batch_size, channels)
        return y

from torch import einsum
from einops import rearrange
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None):
        # x: [B, C]
        x = x.unsqueeze(-1).transpose(1, 2)  # [B, HW, C]
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)
        # B, HW, C = out.size()
        # h = int(HW ** 0.5)
        # out = out.transpose(1, 2).view(B, C, h, h)
        return out


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

class ConvMixer2(nn.Module):
    def __init__(self, cfg: ConvMixerCfg):
        super().__init__()

        self.layers = nn.Sequential(*[
            ConvMixerLayer2(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate)
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
        x = self.layers(x)
        x = self.digup(x)
        return x
  
class ConvMixereca(ConvMixer):
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
        self.ecalayer = nn.Sequential(
            eca_layer(cfg.hidden_dim, kernel_size=cfg.eca_kernel_size),
            nn.LayerNorm(cfg.hidden_dim),
            )

        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        logits = self.digup(x)
        for layer in self.layers:
            x = x + layer(x)
            logits = self.digup(x) + logits
            logits = self.ecalayer(logits)
        logits = self.fc(logits)
        return logits


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
        for layer in self.layers:
            if self.skip_connection:
                x = x + layer(x)
            else:
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


from einops.layers.torch import Rearrange, Reduce

class ConvMixerCat(nn.Module):
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
            # Rearrange('b c h w -> b c (h w)'),
            nn.Flatten(),
            # nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )
        self.attn = Attention(cfg.hidden_dim*cfg.num_layers),
            
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.fc = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

        self.cfg = cfg

    def forward(self, x):
        logits_list = []

        x = self.embed(x)
        data = x.flatten(2).transpose(1, 2)  # data:init   [B, HW, C]  
        logits = self.digup(x)
        for layer in self.layers:
            x = layer(x) + x
            logits = self.digup(x) + logits
            logits = self.layer_norm(logits)
            logits_list.append(logits)
        logits = torch.cat(logits_list, dim=1)
        logits = self.attn(logits, data)
        logits = self.fc(logits)
        # x = self.digup(x)
        return logits