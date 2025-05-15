import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob)
        return x * mask / keep_prob


@MODELS.register_module()
class LCMBlock(nn.Module):
    """Lightweight Convolution Module"""

    def __init__(self,
                 in_channels,
                 expand_ratio=4,
                 groups=4,
                 drop_path=0.):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.dw_conv = nn.Conv2d(
            hidden_dim, hidden_dim, 3,
            padding=1, groups=groups, bias=False)
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_dim, 1)
        self.pw_conv2 = nn.Conv2d(hidden_dim, in_channels, 1)
        self.norm = nn.GroupNorm(groups, hidden_dim)
        self.drop_path = DropPath(drop_path)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw_conv.weight, mode='fan_out')
        nn.init.zeros_(self.pw_conv2.weight)

    def forward(self, x):
        identity = x
        x = self.pw_conv1(x)
        x = self.dw_conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        return identity + self.drop_path(x)


@MODELS.register_module()
class LCM(nn.Module):
    """Stacked LCM Blocks"""

    def __init__(self,
                 num_blocks,
                 in_channels=256,
                 **block_kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[
            LCMBlock(in_channels, **block_kwargs)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)