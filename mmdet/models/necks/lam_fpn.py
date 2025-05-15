import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint
from mmdet.registry import MODELS
from mmdet.models.necks.fpn import FPN


@MODELS.register_module()
class LAMFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 apply_lam_levels=[1, 2],
                 use_dw_conv=True,
                 channel_reduction=4,
                 skip_attention_thresh=0.05,
                 use_modern_norm=True,
                 use_modern_act='silu',
                 use_checkpoint=True,
                 balance_all_scales=True,  # 新增全尺度平衡选项
                 **kwargs):
        super(LAMFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg,
            **kwargs)

        self.use_checkpoint = use_checkpoint
        self.modern_norm_cfg = dict(type='GN', num_groups=16, requires_grad=True) if use_modern_norm else norm_cfg
        self.modern_act_cfg = dict(type='SiLU') if use_modern_act == 'silu' else dict(
            type='Mish') if use_modern_act == 'mish' else act_cfg

        # LAM模块初始化
        self.apply_lam_levels = apply_lam_levels
        self.lam_modules = nn.ModuleList([
            LAMModule(
                out_channels,
                out_channels,
                conv_cfg=conv_cfg,
                norm_cfg=self.modern_norm_cfg,
                act_cfg=self.modern_act_cfg,
                use_dw_conv=use_dw_conv,
                channel_reduction=channel_reduction
            ) if i in self.apply_lam_levels else None
            for i in range(num_outs - 1)
        ])

        # 全尺度注意力机制 - 为各层级配备专门的注意力模块
        self.feature_attention = nn.ModuleList([
            # P2层 - 小目标增强
            SmallObjectEnhancer(out_channels) if i == 0 else
            # P3层 - 中小目标平衡
            MidObjectProcessor(out_channels) if i == 1 else
            # P4层 - 中大目标平衡
            LargeObjectProcessor(out_channels) if i == 2 else
            # P5及以上层 - 纯大目标增强
            LargeObjectEnhancer(out_channels) if i >= 3 and balance_all_scales else None
            for i in range(num_outs)
        ])

        # 添加跨尺度信息交换模块 - 增强各尺度目标的上下文联系
        self.scale_interaction = nn.ModuleList([
            CrossScaleExchange(out_channels) if i < num_outs - 1 and balance_all_scales else None
            for i in range(num_outs)
        ])

        self.skip_attention_thresh = skip_attention_thresh
        self.balance_all_scales = balance_all_scales

    def forward(self, inputs):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, inputs, use_reentrant=False)
        return self._forward_impl(inputs)

    def _forward_impl(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 构建laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if i - 1 in self.apply_lam_levels and self.lam_modules[i - 1] is not None:
                if self.use_checkpoint and self.training:
                    laterals[i - 1] = checkpoint(
                        self.lam_modules[i - 1],
                        laterals[i - 1],
                        F.interpolate(
                            laterals[i],
                            size=laterals[i - 1].shape[2:],
                            **self.upsample_cfg),
                        use_reentrant=False
                    )
                else:
                    laterals[i - 1] = self.lam_modules[i - 1](
                        laterals[i - 1],
                        F.interpolate(
                            laterals[i],
                            size=laterals[i - 1].shape[2:],
                            **self.upsample_cfg)
                    )
            else:
                # 标准FPN融合
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=laterals[i - 1].shape[2:],
                    **self.upsample_cfg)

        # 应用各层级特定的注意力模块
        enhanced_laterals = []
        for i, lateral in enumerate(laterals):
            if self.feature_attention[i] is not None:
                # 应用各层级专门的注意力/增强器
                enhanced_laterals.append(self.feature_attention[i](lateral))
            else:
                enhanced_laterals.append(lateral)

        # 跨尺度信息交换 - 平衡各尺度目标检测能力
        if self.balance_all_scales:
            # 自底向上收集信息
            contexts = []
            for i in range(used_backbone_levels):
                if i > 0 and self.scale_interaction[i - 1] is not None:
                    # 与下一层级交换信息
                    enhanced_laterals[i] = self.scale_interaction[i - 1](
                        enhanced_laterals[i], enhanced_laterals[i - 1])

        # 构建输出
        outs = [
            self.fpn_convs[i](enhanced_laterals[i])
            for i in range(used_backbone_levels)
        ]

        # 处理额外层级
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                extra_source = inputs[self.backbone_end_level - 1] \
                    if self.add_extra_convs == 'on_input' else laterals[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)


class LAMModule(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 use_dw_conv=True,
                 channel_reduction=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LAMModule, self).__init__(init_cfg)

        # 确保通道数兼容GroupNorm
        reduced_channels = in_channels // channel_reduction
        if norm_cfg is not None and norm_cfg.get('type') == 'GN':
            num_groups = norm_cfg.get('num_groups', 16)
            reduced_channels = ((reduced_channels + num_groups - 1) // num_groups) * num_groups

        conv_module = DepthwiseSeparableConvModule if use_dw_conv else ConvModule

        self.attention_conv = conv_module(
            in_channels * 2,
            reduced_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.attention_weights = nn.Conv2d(
            reduced_channels, 2, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.fusion_conv = conv_module(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, target_feat, source_feat):
        concat_feat = torch.cat([target_feat, source_feat], dim=1)
        attention_feat = self.attention_conv(concat_feat)
        attention_weights = self.attention_weights(attention_feat)
        attention_weights = self.softmax(attention_weights)

        fused_feat = (
                attention_weights[:, 0:1] * target_feat +
                attention_weights[:, 1:2] * source_feat
        )
        return self.fusion_conv(fused_feat)


# 小目标增强模块 (P2层) - 聚焦局部特征
class SmallObjectEnhancer(BaseModule):
    """专门针对小目标设计的特征增强模块"""

    def __init__(self,
                 channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SmallObjectEnhancer, self).__init__(init_cfg)

        # 多尺度局部特征增强
        self.local_branch = nn.Sequential(
            # 使用3x3+1x1分解，轻量高效
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

        # 高频信息捕获 - 小目标通常是高频信息
        self.highpass_filter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

        # 特征增强门控 - 专注增强而非抑制
        self.enhance_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 16), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 8, 16), channels, 1, bias=False),
            # 使用Sigmoid+偏置确保门控值>=0.5
            nn.Sigmoid()
        )

        # 输出融合
        self.fusion = nn.Conv2d(channels * 2, channels, 1, bias=True)

    def forward(self, x):
        # 局部特征增强
        local_feat = self.local_branch(x)

        # 高通滤波捕获高频信息
        highpass_feat = self.highpass_filter(x) - F.avg_pool2d(x, 3, stride=1, padding=1)

        # 融合两种特征
        enhanced_feat = torch.cat([local_feat, highpass_feat], dim=1)
        enhanced_feat = self.fusion(enhanced_feat)

        # 使用特征增强门控 - 增强系数在0.5~1.5之间
        enhance_coef = self.enhance_gate(x)
        enhance_coef = enhance_coef * 1.5 + 0.5  # 确保取值范围在0.5~1.5

        # 自适应融合 - 保持原始信息的同时增强特征
        return x + enhanced_feat * enhance_coef


# 中小目标处理模块 (P3层) - 平衡局部与全局特征
class MidObjectProcessor(BaseModule):
    """中小目标处理模块，兼顾局部细节和一定上下文"""

    def __init__(self,
                 channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(MidObjectProcessor, self).__init__(init_cfg)

        # 多尺度特征提取
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 3, padding=dilation, dilation=dilation, bias=False,
                          groups=channels // 4),
                nn.Conv2d(channels // 4, channels // 4, 1, bias=False),
                nn.GroupNorm(4, channels // 4),
                nn.ReLU(inplace=True)
            ) for dilation in [1, 2, 3]
        ])

        # 有限范围上下文
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(4, channels // 4),
            nn.ReLU(inplace=True)
        )

        # 轻量级通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 多尺度特征
        multi_scale_feats = [branch(x) for branch in self.multi_scale]
        multi_scale_feat = torch.cat(multi_scale_feats + [self.context(x)], dim=1)

        # 通道注意力
        channel_weights = self.channel_att(x)

        # 特征融合 - 平衡的方式
        enhanced = self.fusion(multi_scale_feat)

        # 平衡的处理方式 - 保持原特征并叠加增强特征
        return x + enhanced * channel_weights


# 中大目标处理模块 (P4层) - 增强上下文感知
class LargeObjectProcessor(BaseModule):
    """中大目标处理模块，注重上下文和全局结构"""

    def __init__(self,
                 channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LargeObjectProcessor, self).__init__(init_cfg)

        # 弱化的局部特征处理
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 2, 1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True)
        )

        # 增强的上下文感知
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True)
        )

        # 全局信息处理 - 增强对大目标的感知
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(channels + channels // 4, channels, 1, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 局部特征与上下文特征
        local_feat = self.local_conv(x)
        context_feat = self.context_branch(x)
        local_context = torch.cat([local_feat, context_feat], dim=1)

        # 全局信息
        global_info = self.global_pool(x)
        global_info = global_info.expand(-1, -1, x.size(2), x.size(3))

        # 组合所有特征
        combined = torch.cat([local_context, global_info], dim=1)
        enhanced = self.fusion(combined)

        # 残差连接 - 保持原信息
        return x + enhanced


# 大目标增强模块 (P5层) - 专注全局和整体结构
class LargeObjectEnhancer(BaseModule):
    """大目标增强模块，专注全局特征与语义结构"""

    def __init__(self,
                 channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(LargeObjectEnhancer, self).__init__(init_cfg)

        # 扩展感受野 - 大目标需要广阔视野
        self.large_context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

        # 全局信息处理 - 对大目标尤为重要
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力 - 聚焦于目标的关键部分
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 大范围上下文
        context_feat = self.large_context(x)

        # 全局通道特征
        global_weights = self.global_context(x)

        # 空间注意力
        spatial_weights = self.spatial_att(x)

        # 自适应融合 - 强调整体性质同时保留空间结构
        enhanced = context_feat * global_weights * spatial_weights

        # 保持原始信息并增强
        return x + enhanced


# 跨尺度信息交换模块 - 平衡各尺度目标
class CrossScaleExchange(BaseModule):
    """跨尺度信息交换模块，平衡不同尺度目标的表示能力"""

    def __init__(self,
                 channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CrossScaleExchange, self).__init__(init_cfg)

        # 自层级特征提取
        self.self_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True)
        )

        # 相邻层级特征采样
        self.cross_sample = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.GroupNorm(8, channels // 2),
            nn.ReLU(inplace=True)
        )

        # 自适应融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, curr_feat, other_feat):
        # 处理当前层级特征
        curr_processed = self.self_attn(curr_feat)

        # 处理相邻层级特征 - 采样到当前尺度
        other_processed = self.cross_sample(
            F.interpolate(other_feat, size=curr_feat.shape[2:], mode='bilinear', align_corners=False)
        )

        # 融合特征
        fused = torch.cat([curr_processed, other_processed], dim=1)
        enhanced = self.fusion(fused)

        # 残差连接
        return curr_feat + enhanced * 0.5  # 温和的影响