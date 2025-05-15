import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Dict, Optional
from torch import Tensor
import contextlib


@MODELS.register_module()
class ADEMHead(BaseModule):
    """Enhanced Auxiliary Density Estimation Module (ADEM)

    This module generates density estimation maps from FPN features to help understand object distribution.

    Args:
        in_channels (int): Number of input channels, default 256
        num_scales (int): Number of feature scales to process, default 3
        min_sigma (float): Minimum uncertainty for density prediction, default 0.1
        max_sigma (float): Maximum uncertainty for density prediction, default 2.0
        loss_weight (float): Weight for density estimation loss, default 0.3
        use_modern_norm (bool): Whether to use modern normalization (GN), default True
        use_modern_act (str): Modern activation function to use ('silu'/'mish'), default 'silu'
        use_dw_conv (bool): Whether to use depthwise separable convolutions, default True
        channel_reduction (int): Channel reduction factor, default 2
    """

    def __init__(self,
                 in_channels: int = 256,
                 num_scales: int = 3,
                 min_sigma: float = 0.1,
                 max_sigma: float = 2.0,
                 loss_weight: float = 0.3,
                 use_modern_norm: bool = True,
                 use_modern_act: str = 'silu',
                 use_dw_conv: bool = True,
                 channel_reduction: int = 2,
                 init_cfg: Optional[dict] = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.loss_weight = loss_weight
        reduced_channels = in_channels // channel_reduction

        # Modern activation
        if use_modern_act == 'silu':
            act_layer = nn.SiLU
        elif use_modern_act == 'mish':
            act_layer = nn.Mish
        else:
            act_layer = nn.ReLU

        # Feature processing modules with memory optimization
        self.scale_processors = nn.ModuleList()
        for _ in range(num_scales):
            layers = []
            if use_dw_conv:
                layers.extend([
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
                ])
            else:
                layers.append(nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False))

            layers.extend([
                nn.GroupNorm(16, reduced_channels) if use_modern_norm else nn.BatchNorm2d(reduced_channels),
                act_layer(inplace=True)
            ])
            self.scale_processors.append(nn.Sequential(*layers))

        fused_channels = reduced_channels * num_scales

        # Optimized prediction branches
        def create_branch(in_ch: int, use_dw: bool) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, groups=in_ch // 2, bias=False) if use_dw
                else nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
                nn.GroupNorm(16, in_ch // 2) if use_modern_norm else nn.BatchNorm2d(in_ch // 2),
                act_layer(inplace=True),
                nn.Conv2d(in_ch // 2, 1, 1, bias=True)
            )

        self.density_branch = create_branch(fused_channels, use_dw_conv)
        self.sigma_branch = create_branch(fused_channels, use_dw_conv)

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with memory-efficient approach."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.cuda.amp.autocast()
    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        """Memory-optimized forward pass"""
        base_size = feats[0].shape[2:]
        processed_feats = []

        for i, feat in enumerate(feats[:len(self.scale_processors)]):
            # Optimize memory usage during interpolation
            if i > 0:
                with torch.no_grad():
                    feat = F.interpolate(feat, size=base_size, mode='bilinear', align_corners=False)

            # Process features and immediately release memory
            processed = self.scale_processors[i](feat)
            processed_feats.append(processed)
            del feat

        # Concatenate and free memory
        fused = torch.cat(processed_feats, dim=1)
        del processed_feats

        # Generate predictions with optimized memory usage
        density = torch.sigmoid(self.density_branch(fused))

        # Detach fused features for sigma branch to save memory
        sigma_input = fused.detach() if self.training else fused
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * torch.sigmoid(self.sigma_branch(sigma_input))
        del fused, sigma_input

        return {'density': density, 'sigma': sigma}

    def loss(self,
             pred_density: Tensor,
             gt_density: Tensor,
             pred_sigma: Optional[Tensor] = None,
             eps: float = 1e-6) -> Tensor:
        """Memory-efficient loss calculation"""
        if gt_density.dim() == 3:
            gt_density = gt_density.unsqueeze(1)

        # Calculate difference with memory optimization
        diff = (pred_density - gt_density).pow(2)
        del gt_density

        if pred_sigma is not None:
            # Memory-efficient uncertainty-aware loss calculation
            weight = (2 * pred_sigma.pow(2) + eps).reciprocal()
            loss = (diff * weight + torch.log(pred_sigma + eps)).mean()
            del weight, diff, pred_sigma
        else:
            loss = diff.mean()
            del diff

        return loss * self.loss_weight
