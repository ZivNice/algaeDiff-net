# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/detector.py   # noqa
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/head.py   # noqa

# This work is licensed under the CC-BY-NC 4.0 License.
# Users should be careful about adopting these features in any commercial matters.    # noqa
# For more details, please refer to https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE    # noqa

import copy
import math
import random
import warnings
from typing import Tuple, List, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox2roi, bbox_cxcywh_to_xyxy,
                                   bbox_xyxy_to_cxcywh, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList
# 在文件开头添加导入语句
from .lcm_module import LCMMapper

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@MODELS.register_module()
class EnhancedDiffusionDetHead(nn.Module):
    """增强版DiffusionDet头部，集成LCM功能"""

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 num_proposals=500,
                 num_heads=6,
                 prior_prob=0.01,
                 snr_scale=2.0,
                 timesteps=1000,
                 sampling_timesteps=1,
                 self_condition=False,
                 box_renewal=True,
                 use_ensemble=True,
                 deep_supervision=True,
                 ddim_sampling_eta=1.0,
                 # 添加动态DDIM参数 ↓↓↓
                 dynamic_steps=10,
                 base_steps=3,
                 complexity_thresh=0.5,
                 ddim_eta=0.6,
                 # 添加知识蒸馏参数 ↓↓↓
                 enable_distill=False,
                 teacher_config=None,
                 distill_weight=0.5,
                 # 添加小目标蒸馏参数 ↓↓↓
                 small_object_distill=False,
                 small_object_distiller=None,
                 # LCM相关参数
                 use_lcm=False,
                 lcm_guidance_scale=1.0,
                 use_adem=False,
                 adem_head=None,
                 adem_loss_weight=0.3,
                 criterion=dict(
                     type='DiffusionDetCriterion',
                     num_classes=80,
                     assigner=dict(
                         type='DiffusionDetMatcher',
                         match_costs=[
                             dict(
                                 type='FocalLossCost',
                                 alpha=2.0,
                                 gamma=0.25,
                                 weight=2.0),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xyxy'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ],
                         center_radius=2.5,
                         candidate_topk=5),
                 ),
                 single_head=dict(
                     type='EnhancedSingleDiffusionDetHead',
                     num_cls_convs=1,
                     num_reg_convs=3,
                     dim_feedforward=2048,
                     num_heads=8,
                     dropout=0.0,
                     act_cfg=dict(type='ReLU'),
                     dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
                 roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 test_cfg=None,
                 **kwargs) -> None:
        super().__init__()
        self.roi_extractor = MODELS.build(roi_extractor)

        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        # Build Diffusion
        assert isinstance(timesteps, int), 'The type of `timesteps` should ' \
                                           f'be int but got {type(timesteps)}'
        assert sampling_timesteps <= timesteps
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.snr_scale = snr_scale

        self.ddim_sampling = self.sampling_timesteps < self.timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.self_condition = self_condition
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble

        self._build_diffusion()

        # Build assigner
        assert criterion.get('assigner', None) is not None
        assigner = TASK_UTILS.build(criterion.get('assigner'))
        # Init parameters.
        self.use_focal_loss = assigner.use_focal_loss
        self.use_fed_loss = assigner.use_fed_loss

        # build criterion
        criterion.update(deep_supervision=deep_supervision)
        self.criterion = TASK_UTILS.build(criterion)

        # Build Dynamic Head.
        single_head_ = single_head.copy()
        single_head_num_classes = single_head_.get('num_classes', None)
        if single_head_num_classes is None:
            single_head_.update(num_classes=num_classes)
        else:
            if single_head_num_classes != num_classes:
                warnings.warn(
                    'The `num_classes` of `EnhancedDiffusionDetHead` and '
                    '`EnhancedSingleDiffusionDetHead` should be same, changing '
                    f'`single_head.num_classes` to {num_classes}')
                single_head_.update(num_classes=num_classes)

        single_head_feat_channels = single_head_.get('feat_channels', None)
        if single_head_feat_channels is None:
            single_head_.update(feat_channels=feat_channels)
        else:
            if single_head_feat_channels != feat_channels:
                warnings.warn(
                    'The `feat_channels` of `EnhancedDiffusionDetHead` and '
                    '`EnhancedSingleDiffusionDetHead` should be same, changing '
                    f'`single_head.feat_channels` to {feat_channels}')
                single_head_.update(feat_channels=feat_channels)

        default_pooler_resolution = roi_extractor['roi_layer'].get(
            'output_size')
        assert default_pooler_resolution is not None
        single_head_pooler_resolution = single_head_.get('pooler_resolution')
        if single_head_pooler_resolution is None:
            single_head_.update(pooler_resolution=default_pooler_resolution)
        else:
            if single_head_pooler_resolution != default_pooler_resolution:
                warnings.warn(
                    'The `pooler_resolution` of `EnhancedDiffusionDetHead` '
                    'and `EnhancedSingleDiffusionDetHead` should be same, changing '
                    f'`single_head.pooler_resolution` to {default_pooler_resolution}')
                single_head_.update(
                    pooler_resolution=default_pooler_resolution)

        single_head_.update(
            use_focal_loss=self.use_focal_loss, use_fed_loss=self.use_fed_loss)
        single_head_module = MODELS.build(single_head_)

        self.num_heads = num_heads
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head_module) for _ in range(num_heads)])

        self.deep_supervision = deep_supervision

        # Gaussian random feature embedding layer for time
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim))

        # LCM相关设置
        self.use_lcm = use_lcm
        self.lcm_guidance_scale = lcm_guidance_scale

        if self.use_lcm:
            # 获取ROI池化分辨率
            pooler_resolution = roi_extractor['roi_layer'].get('output_size')
            if isinstance(pooler_resolution, int):
                roi_feat_dim = feat_channels * pooler_resolution * pooler_resolution
            else:  # 如果是元组 (h, w)
                roi_feat_dim = feat_channels * pooler_resolution[0] * pooler_resolution[1]

            # 特征降维网络 - 将ROI特征降维到合适尺寸
            self.feat_reducer = nn.Sequential(
                nn.Linear(roi_feat_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feat_channels)
            )

            # LCM映射器
            self.lcm_mapper = LCMMapper(feat_channels)

        # ADEM相关配置
        self.use_adem = use_adem
        if self.use_adem and adem_head is not None:
            self.adem_head = MODELS.build(adem_head)
            self.adem_loss_weight = adem_loss_weight
        else:
            self.adem_head = None

        self.prior_prob = prior_prob
        self.test_cfg = test_cfg
        self.use_nms = self.test_cfg.get('use_nms', True)
        self._init_weights()

        # 动态DDIM参数初始化
        self.dynamic_steps = dynamic_steps
        self.base_steps = base_steps
        self.complexity_thresh = complexity_thresh
        self.ddim_eta = ddim_eta
        # 注册性能统计缓存
        self.register_buffer('step_history', torch.zeros(100))
        self.register_buffer('performance_metrics', torch.zeros(4))

        # 知识蒸馏相关初始化
        self.enable_distill = enable_distill
        self.distill_weight = distill_weight
        self.teacher = None
        if enable_distill and teacher_config is not None:
            self.teacher = self._build_teacher(teacher_config)

        # 添加小目标蒸馏相关属性
        self.small_object_distill = small_object_distill
        self.distill_weight = distill_weight

        # 如果启用小目标蒸馏，构建蒸馏器
        if self.small_object_distill and small_object_distiller is not None:
            self.small_object_distiller = MODELS.build(small_object_distiller)
        else:
            self.small_object_distiller = None

    def _init_weights(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal_loss or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or \
                        p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, bias_value)

    def _build_diffusion(self):
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) /
                             (1. - alphas_cumprod))

    # 添加提取ROI特征并降维的方法
    def _get_roi_features(self, features, bboxes, batch_time=None):
        """提取并降维ROI特征"""
        batch_size = len(features[0])
        rois = bbox2roi([bboxes[i] for i in range(batch_size)])

        # 原始ROI特征 [N*num_boxes, 256, 7, 7]
        roi_features = self.roi_extractor(features, rois)

        # 展平特征
        flattened = roi_features.flatten(1)

        # 降维到256
        reduced_features = self.feat_reducer(flattened)
        return reduced_features

    def forward(self, features, init_bboxes, init_t, init_features=None):
        time = self.time_mlp(init_t)

        inter_class_logits = []
        inter_pred_bboxes = []
        adem_outputs = None  # 用于存储ADEM输出

        bs = len(features[0])
        bboxes = init_bboxes

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = single_head(
                features, bboxes, proposal_features, self.roi_extractor, time)
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        # 如果启用了ADEM，则计算密度估计
        if self.deep_supervision:
            if self.use_adem and self.adem_head is not None:
                # 计算ADEM输出
                adem_outputs = self.adem_head(features)
                return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes), adem_outputs
            else:
                return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)
        else:
            if self.use_adem and self.adem_head is not None:
                # 计算ADEM输出
                adem_outputs = self.adem_head(features)
                return class_logits[None, ...], pred_bboxes[None, ...], adem_outputs
            else:
                return class_logits[None, ...], pred_bboxes[None, ...]

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """执行推理并预测检测结果"""
        device = x[-1].device
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        (time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw,
         batch_image_size) = self.prepare_testing_targets(
            batch_img_metas, device)

        # 根据模块启用状态选择推理路径
        if self.use_adem and self.use_lcm:
            # 同时使用ADEM和LCM的组合推理路径
            predictions = self._adem_lcm_inference(
                x,
                batch_noise_bboxes,
                batch_image_size,
                device,
                batch_img_metas,
                self.test_cfg)
        elif self.use_adem:
            # 只使用ADEM的推理路径
            predictions = self._adem_inference(
                x,
                batch_noise_bboxes,
                batch_image_size,
                device,
                batch_img_metas,
                self.test_cfg)
        elif self.use_lcm:
            # 只使用LCM的推理路径
            predictions = self._lcm_inference(
                x,
                batch_noise_bboxes,
                batch_image_size,
                device,
                batch_img_metas,
                self.test_cfg)
        else:
            # 原始推理路径
            predictions = self.predict_by_feat(
                x,
                time_pairs=time_pairs,
                batch_noise_bboxes=batch_noise_bboxes,
                batch_noise_bboxes_raw=batch_noise_bboxes_raw,
                batch_image_size=batch_image_size,
                device=device,
                batch_img_metas=batch_img_metas)

        if rescale:
            predictions = self.do_results_post_process(
                predictions, self.test_cfg, batch_img_metas=batch_img_metas)
        return predictions

    def _lcm_inference(self, x, init_bboxes, image_size, device, img_metas, cfg, rescale=True):
        """使用LCM进行推理的特殊方法"""
        batch_size = len(img_metas)
        num_boxes = init_bboxes.shape[1]  # 通常是500

        # 预测原始边界框
        pred_logits, pred_bboxes = self.forward(
            x,
            init_bboxes,
            torch.zeros(batch_size, device=device, dtype=torch.long)
        )

        # 提取ROI特征
        roi_features = self._get_roi_features(x, pred_bboxes[-1])

        # 通过LCM预测调整量
        consistency_score = self.lcm_mapper(roi_features)

        # 重塑成[batch_size, num_boxes, 4]以匹配pred_bboxes[-1]的形状
        consistency_score = consistency_score.view(batch_size, num_boxes, 4)

        # 调整预测边界框
        adjusted_bboxes = pred_bboxes[-1] + self.lcm_guidance_scale * consistency_score

        # 进行最终预测
        inference_result = self.inference(pred_logits[-1], adjusted_bboxes, cfg, device)

        # 处理inference可能返回的不同类型的结果
        if self.use_ensemble and self.sampling_timesteps > 1:
            # 在这种情况下inference返回的是元组(box_pred_list, scores_list, labels_list)
            box_pred_list, scores_list, labels_list = inference_result
            results_list = []
            for i in range(batch_size):
                # 将元组转换成结果列表
                if cfg.get('use_nms', True):
                    # 如果需要NMS，先进行NMS处理
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_list[i], scores_list[i], labels_list[i], cfg.nms)
                    box_pred = box_pred_list[i][keep_idxs]
                    labels = labels_list[i][keep_idxs]
                    scores = det_bboxes[:, -1]
                else:
                    # 不需要NMS，直接使用
                    box_pred = box_pred_list[i]
                    scores = scores_list[i]
                    labels = labels_list[i]

                result = InstanceData()
                result.bboxes = box_pred
                result.scores = scores
                result.labels = labels
                results_list.append(result)
        else:
            # 已经是结果列表
            results_list = inference_result

        return results_list

    # 添加ADEM推理方法
    def _adem_inference(self, x, init_bboxes, image_size, device, img_metas, cfg):
        """使用ADEM进行推理的方法"""
        batch_size = len(img_metas)

        # 使用零时间步（完全去噪）进行前向传播
        # 修改这里，接收三个返回值而不是两个
        pred_logits, pred_bboxes, adem_outputs = self.forward(
            x,
            init_bboxes,
            torch.zeros(batch_size, device=device, dtype=torch.long)
        )

        # 获取ADEM密度图 - 现在从adem_outputs直接获取而不是再次计算
        density_maps = adem_outputs['density']

        # 使用密度图作为置信度增强
        enhanced_logits = []
        for i in range(batch_size):
            # 获取当前批次的边界框和logits
            cur_boxes = pred_bboxes[-1][i]  # [num_boxes, 4]
            cur_logits = pred_logits[-1][i]  # [num_boxes, num_classes]

            # 从密度图中提取每个框的密度分数
            density_scores = self._extract_box_density(
                density_maps[i:i + 1],
                cur_boxes,
                img_metas[i]
            )  # [num_boxes]

            # 将密度分数融入到类别分数中
            # 这里我们简单地将密度分数作为一个加权因子
            density_weight = density_scores.unsqueeze(1).expand_as(cur_logits)
            enhanced_cur_logits = cur_logits + self.adem_loss_weight * torch.log1p(density_weight)
            enhanced_logits.append(enhanced_cur_logits)

        # 使用增强后的logits进行最终预测
        enhanced_pred_logits = torch.stack(enhanced_logits)

        # 使用增强后的logits和原始边界框进行推理
        inference_result = self.inference(enhanced_pred_logits, pred_bboxes[-1], cfg, device)

        # 处理inference可能返回的不同类型的结果
        if self.use_ensemble and self.sampling_timesteps > 1:
            # 在这种情况下inference返回的是元组
            box_pred_list, scores_list, labels_list = inference_result
            results_list = []
            for i in range(batch_size):
                if cfg.get('use_nms', True):
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_list[i], scores_list[i], labels_list[i], cfg.nms)
                    box_pred = box_pred_list[i][keep_idxs]
                    labels = labels_list[i][keep_idxs]
                    scores = det_bboxes[:, -1]
                else:
                    box_pred = box_pred_list[i]
                    scores = scores_list[i]
                    labels = labels_list[i]

                result = InstanceData()
                result.bboxes = box_pred
                result.scores = scores
                result.labels = labels
                results_list.append(result)
        else:
            # 已经是结果列表
            results_list = inference_result

        return results_list

    def _adem_lcm_inference(self, x, init_bboxes, image_size, device, img_metas, cfg):
        """同时使用ADEM和LCM进行推理的方法"""
        batch_size = len(img_metas)
        num_boxes = init_bboxes.shape[1]  # 通常是500

        # 1. 首先执行初始预测
        # 修改这里，接收三个返回值
        pred_logits, pred_bboxes, adem_outputs = self.forward(
            x,
            init_bboxes,
            torch.zeros(batch_size, device=device, dtype=torch.long)
        )

        # 2. 获取ADEM密度图 - 直接从adem_outputs获取
        density_maps = adem_outputs['density']

        # 3. 提取ROI特征并通过LCM进行调整
        roi_features = self._get_roi_features(x, pred_bboxes[-1])

        # 4. 通过LCM模块预测调整量
        consistency_score = self.lcm_mapper(roi_features)

        # 5. 将调整量重塑为[batch_size, num_boxes, 4]
        consistency_score = consistency_score.view(batch_size, num_boxes, 4)

        # 6. 使用LCM调整边界框
        adjusted_bboxes = pred_bboxes[-1] + self.lcm_guidance_scale * consistency_score

        # 7. 基于密度图增强置信度得分
        enhanced_logits = []
        for i in range(batch_size):
            # 获取当前批次的logits
            cur_logits = pred_logits[-1][i]  # [num_boxes, num_classes]

            # 从密度图中提取每个框的密度分数
            density_scores = self._extract_box_density(
                density_maps[i:i + 1],
                adjusted_bboxes[i],
                img_metas[i]
            )  # [num_boxes]

            # 融合密度得分到分类得分
            density_weight = density_scores.unsqueeze(1).expand_as(cur_logits)
            enhanced_cur_logits = cur_logits + self.adem_loss_weight * torch.log1p(density_weight)
            enhanced_logits.append(enhanced_cur_logits)

        # 8. 组合增强后的logits
        enhanced_pred_logits = torch.stack(enhanced_logits)

        # 9. 最终推理
        inference_result = self.inference(enhanced_pred_logits, adjusted_bboxes, cfg, device)

        # 10. 处理inference可能返回的不同类型的结果
        if self.use_ensemble and self.sampling_timesteps > 1:
            # 在这种情况下inference返回的是元组
            box_pred_list, scores_list, labels_list = inference_result
            results_list = []
            for i in range(batch_size):
                if cfg.get('use_nms', True):
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_list[i], scores_list[i], labels_list[i], cfg.nms)
                    box_pred = box_pred_list[i][keep_idxs]
                    labels = labels_list[i][keep_idxs]
                    scores = det_bboxes[:, -1]
                else:
                    box_pred = box_pred_list[i]
                    scores = scores_list[i]
                    labels = labels_list[i]

                    result = InstanceData()
                    result.bboxes = box_pred
                    result.scores = scores
                    result.labels = labels
                    results_list.append(result)
        else:
        # 已经是结果列表
            results_list = inference_result

        return results_list

    def loss(self, x, batch_data_samples):
        """
        损失计算函数

        Args:
            x: 特征图列表
            batch_data_samples: 包含标注数据的批次样本

        Returns:
            dict: 损失字典
        """
        # 准备训练目标
        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore, batch_img_metas) = prepare_outputs

        # 准备输入
        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs
            for pred_instances in batch_pred_instances
        ])
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])

        # 前向传播
        outputs = self(x, batch_diff_bboxes, batch_time)

        # 处理输出
        if isinstance(outputs, tuple) and len(outputs) == 3:
            pred_logits, pred_bboxes, adem_outputs = outputs
            use_adem = True
        else:
            pred_logits, pred_bboxes = outputs
            use_adem = False

        # 构建输出字典
        output_dict = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }

        # 添加深度监督输出
        if self.deep_supervision:
            output_dict['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]

        # 计算标准检测损失
        losses = self.criterion(output_dict, batch_gt_instances, batch_img_metas)

        # 如果启用小目标蒸馏，计算蒸馏损失
        if self.small_object_distill:
            # 收集边界框信息
            gt_bboxes_list = [gt_instances.bboxes for gt_instances in batch_gt_instances]

            # 收集学生和教师的特征图
            student_features = [x[-1]]  # 使用最后一层特征图
            teacher_features = [x[-1]]  # 使用最后一层特征图

            # 计算小目标感知蒸馏损失
            distill_loss = self.small_object_distiller(
                student_features,
                teacher_features,
                gt_bboxes_list,
                batch_img_metas
            )

            # 添加到损失字典
            losses['loss_small_obj_distill'] = distill_loss * self.distill_weight

            # 更新总损失 - 只在必要时创建或更新
            if 'loss' in losses:
                # 如果已有总损失，直接添加
                losses['loss'] = losses['loss'] + losses['loss_small_obj_distill']
            else:
                # 如果没有总损失，检查是否有其他损失项
                other_losses = [v for k, v in losses.items()
                                if isinstance(v, torch.Tensor) and k != 'loss_small_obj_distill']

                if other_losses:
                    # 如果有其他损失项，创建总损失
                    losses['loss'] = sum(other_losses) + losses['loss_small_obj_distill']
                else:
                    # 如果没有其他损失项，将小目标蒸馏损失作为总损失
                    losses['loss'] = losses['loss_small_obj_distill']

        # 如果启用ADEM，计算密度估计损失
        if use_adem:
            # 收集批次中所有数据样本的密度图
            gt_densities = []
            for data_sample in batch_data_samples:
                if hasattr(data_sample, 'gt_density'):
                    # 确保密度图在正确的设备上
                    gt_density = data_sample.gt_density.to(adem_outputs['density'].device)
                    gt_densities.append(gt_density)

            # 如果有密度图，计算损失
            if gt_densities:
                gt_densities = torch.stack(gt_densities)
                # 可能需要调整尺寸以匹配预测
                if gt_densities.shape != adem_outputs['density'].shape:
                    h, w = adem_outputs['density'].shape[2:]
                    gt_densities = F.interpolate(gt_densities.unsqueeze(1), size=(h, w),
                                                 mode='bilinear', align_corners=False).squeeze(1)

                # 计算密度估计损失 (MSE或其他适合的损失函数)
                adem_loss = F.mse_loss(adem_outputs['density'], gt_densities)
                losses['loss_adem'] = adem_loss * self.adem_head.loss_weight
                losses['loss'] = losses['loss'] + losses['loss_adem']

        return losses

    # 以下为原方法的实现，保持不变
    def prepare_training_targets(self, batch_data_samples):
        batch_gt_instances = []
        batch_pred_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            img_meta = data_sample.metainfo
            gt_instances = data_sample.gt_instances

            gt_bboxes = gt_instances.bboxes
            h, w = img_meta['img_shape']
            image_size = gt_bboxes.new_tensor([w, h, w, h])

            norm_gt_bboxes = gt_bboxes / image_size
            norm_gt_bboxes_cxcywh = bbox_xyxy_to_cxcywh(norm_gt_bboxes)
            pred_instances = self.prepare_diffusion(norm_gt_bboxes_cxcywh,
                                                    image_size)

            gt_instances.set_metainfo(dict(image_size=image_size))
            gt_instances.norm_bboxes_cxcywh = norm_gt_bboxes_cxcywh

            batch_gt_instances.append(gt_instances)
            batch_pred_instances.append(pred_instances)
            batch_img_metas.append(data_sample.metainfo)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        return (batch_gt_instances, batch_pred_instances,
                batch_gt_instances_ignore, batch_img_metas)

    def prepare_diffusion(self, gt_boxes, image_size):
        device = gt_boxes.device
        time = torch.randint(
            0, self.timesteps, (1,), dtype=torch.long, device=device)
        noise = torch.randn(self.num_proposals, 4, device=device)

        num_gt = gt_boxes.shape[0]
        if num_gt < self.num_proposals:
            # 3 * sigma = 1/2 --> sigma: 1/6
            box_placeholder = torch.randn(
                self.num_proposals - num_gt, 4, device=device) / 6. + 0.5
            box_placeholder[:, 2:] = torch.clip(
                box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        else:
            select_mask = [True] * self.num_proposals + \
                          [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]

        x_start = (x_start * 2. - 1.) * self.snr_scale

        # noise sample
        x = self.q_sample(x_start=x_start, time=time, noise=noise)

        x = torch.clamp(x, min=-1 * self.snr_scale, max=self.snr_scale)
        x = ((x / self.snr_scale) + 1) / 2.

        diff_bboxes = bbox_cxcywh_to_xyxy(x)
        # convert to abs bboxes
        diff_bboxes_abs = diff_bboxes * image_size

        metainfo = dict(time=time.squeeze(-1))
        pred_instances = InstanceData(metainfo=metainfo)
        pred_instances.diff_bboxes = diff_bboxes
        pred_instances.diff_bboxes_abs = diff_bboxes_abs
        pred_instances.noise = noise
        return pred_instances

    # forward diffusion
    def q_sample(self, x_start, time, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_start_shape = x_start.shape

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, time,
                                        x_start_shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, time, x_start_shape)

        return sqrt_alphas_cumprod_t * x_start + \
            sqrt_one_minus_alphas_cumprod_t * noise

    def predict_by_feat(self,
                        x,
                        time_pairs,
                        batch_noise_bboxes,
                        batch_noise_bboxes_raw,
                        batch_image_size,
                        device,
                        batch_img_metas=None,
                        cfg=None,
                        rescale=True):
        """使用动态DDIM进行预测"""
        batch_size = len(batch_img_metas)
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        # 计算图像复杂度
        complexity = self._calc_image_complexity(x)
        steps = self.dynamic_step_scheduler(complexity)
        max_step = steps.max().item()

        # 记录使用的步数以便调试
        with torch.no_grad():
            self.step_history = torch.roll(self.step_history, 1)
            self.step_history[0] = max_step

        # 生成动态时间步序列
        if max_step < self.timesteps:
            times = torch.linspace(-1, self.timesteps - 1, steps=max_step + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))

        # 保持与原代码相同的结构
        ensemble_score, ensemble_label, ensemble_coord = [], [], []

        # 使用动态步长进行采样
        for time, time_next in time_pairs:
            batch_time = torch.full((batch_size,),
                                    time,
                                    device=device,
                                    dtype=torch.long)

            # 混合精度计算 - 可选
            with torch.cuda.amp.autocast(enabled=False):
                pred_logits, pred_bboxes = self(x, batch_noise_bboxes, batch_time)

                x_start = pred_bboxes[-1]
                x_start = x_start / batch_image_size[:, None, :]
                x_start = bbox_xyxy_to_cxcywh(x_start)
                x_start = (x_start * 2 - 1.) * self.snr_scale
                x_start = torch.clamp(
                    x_start, min=-1 * self.snr_scale, max=self.snr_scale)

                pred_noise = self.predict_noise_from_start(
                    batch_noise_bboxes_raw, batch_time, x_start)

            # 以下是原有的box renewal和噪声更新逻辑
            pred_noise_list, x_start_list = [], []
            noise_bboxes_list, num_remain_list = [], []

            if self.box_renewal:  # filter
                score_thr = cfg.get('score_thr', 0)
                for img_id in range(batch_size):
                    score_per_image = pred_logits[-1][img_id]
                    score_per_image = torch.sigmoid(score_per_image)
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                    keep_idx = value > score_thr

                    num_remain_list.append(torch.sum(keep_idx))
                    pred_noise_list.append(pred_noise[img_id, keep_idx, :])
                    x_start_list.append(x_start[img_id, keep_idx, :])
                    noise_bboxes_list.append(batch_noise_bboxes[img_id, keep_idx, :])

            if time_next < 0:
                if self.use_ensemble and max_step > 1:
                    box_pred_per_image, scores_per_image, labels_per_image = \
                        self.inference(
                            box_cls=pred_logits[-1],
                            box_pred=pred_bboxes[-1],
                            cfg=cfg,
                            device=device)
                    ensemble_score.append(scores_per_image)
                    ensemble_label.append(labels_per_image)
                    ensemble_coord.append(box_pred_per_image)
                continue

            # 动态DDIM更新步骤 - 使用自定义eta
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # 使用动态eta参数
            sigma = self.ddim_eta * ((1 - alpha / alpha_next) *
                                     (1 - alpha_next) /
                                     (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            batch_noise_bboxes_list = []
            batch_noise_bboxes_raw_list = []
            for idx in range(batch_size):
                pred_noise = pred_noise_list[idx]
                x_start = x_start_list[idx]
                noise_bboxes = noise_bboxes_list[idx]
                num_remain = num_remain_list[idx]
                noise = torch.randn_like(noise_bboxes)

                noise_bboxes = x_start * alpha_next.sqrt() + \
                               c * pred_noise + sigma * noise

                if self.box_renewal:  # filter
                    # replenish with randn boxes
                    if num_remain < self.num_proposals:
                        noise_bboxes = torch.cat(
                            (noise_bboxes,
                             torch.randn(
                                 self.num_proposals - num_remain,
                                 4,
                                 device=device)),
                            dim=0)
                    else:
                        select_mask = [True] * self.num_proposals + \
                                      [False] * (num_remain - self.num_proposals)
                        random.shuffle(select_mask)
                        noise_bboxes = noise_bboxes[select_mask]

                    # raw noise boxes
                    batch_noise_bboxes_raw_list.append(noise_bboxes)
                    # resize to xyxy
                    noise_bboxes = torch.clamp(
                        noise_bboxes,
                        min=-1 * self.snr_scale,
                        max=self.snr_scale)
                    noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2
                    noise_bboxes = bbox_cxcywh_to_xyxy(noise_bboxes)
                    noise_bboxes = noise_bboxes * batch_image_size[idx]

                batch_noise_bboxes_list.append(noise_bboxes)

            batch_noise_bboxes = torch.stack(batch_noise_bboxes_list)
            batch_noise_bboxes_raw = torch.stack(batch_noise_bboxes_raw_list)

            if self.use_ensemble and max_step > 1:
                box_pred_per_image, scores_per_image, labels_per_image = \
                    self.inference(
                        box_cls=pred_logits[-1],
                        box_pred=pred_bboxes[-1],
                        cfg=cfg,
                        device=device)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        # 处理结果
        if self.use_ensemble and max_step > 1:
            steps = len(ensemble_score)
            results_list = []
            for idx in range(batch_size):
                ensemble_score_per_img = [
                    ensemble_score[i][idx] for i in range(steps)
                ]
                ensemble_label_per_img = [
                    ensemble_label[i][idx] for i in range(steps)
                ]
                ensemble_coord_per_img = [
                    ensemble_coord[i][idx] for i in range(steps)
                ]

                scores_per_image = torch.cat(ensemble_score_per_img, dim=0)
                labels_per_image = torch.cat(ensemble_label_per_img, dim=0)
                box_pred_per_image = torch.cat(ensemble_coord_per_img, dim=0)

                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    scores_per_image = det_bboxes[:, -1]
                results = InstanceData()
                results.bboxes = box_pred_per_image
                results.scores = scores_per_image
                results.labels = labels_per_image
                results_list.append(results)
        else:
            box_cls = pred_logits[-1]
            box_pred = pred_bboxes[-1]
            results_list = self.inference(box_cls, box_pred, cfg, device)

        return results_list
    @staticmethod
    def do_results_post_process(results_list, cfg, batch_img_metas=None):
        processed_results = []
        for results, img_meta in zip(results_list, batch_img_metas):
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)
            # clip w, h
            h, w = img_meta['ori_shape']
            results.bboxes[:, 0::2] = results.bboxes[:, 0::2].clamp(
                min=0, max=w)
            results.bboxes[:, 1::2] = results.bboxes[:, 1::2].clamp(
                min=0, max=h)

            # filter small size bboxes
            if cfg.get('min_bbox_size', 0) >= 0:
                w, h = get_box_wh(results.bboxes)
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    results = results[valid_mask]
            processed_results.append(results)

        return processed_results

    def prepare_testing_targets(self, batch_img_metas, device):
        """修改测试目标准备，支持动态时间步"""
        # 计算动态时间步
        if hasattr(self, 'dynamic_steps') and self.dynamic_steps > 0:
            # 使用预设的动态步数
            steps = self.dynamic_steps
            times = torch.linspace(-1, self.timesteps - 1, steps=steps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
        else:
            # 使用原始的时间步生成逻辑
            times = torch.linspace(
                -1, self.timesteps - 1, steps=self.sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))

        # 原有的噪声框准备逻辑
        noise_bboxes_list = []
        noise_bboxes_raw_list = []
        image_size_list = []
        for img_meta in batch_img_metas:
            h, w = img_meta['img_shape']
            image_size = torch.tensor([w, h, w, h],
                                      dtype=torch.float32,
                                      device=device)
            noise_bboxes_raw = torch.randn((self.num_proposals, 4),
                                           device=device)
            noise_bboxes = torch.clamp(
                noise_bboxes_raw, min=-1 * self.snr_scale, max=self.snr_scale)
            noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2
            noise_bboxes = bbox_cxcywh_to_xyxy(noise_bboxes)
            noise_bboxes = noise_bboxes * image_size

            noise_bboxes_raw_list.append(noise_bboxes_raw)
            noise_bboxes_list.append(noise_bboxes)
            image_size_list.append(image_size[None])
        batch_noise_bboxes = torch.stack(noise_bboxes_list)
        batch_image_size = torch.cat(image_size_list)
        batch_noise_bboxes_raw = torch.stack(noise_bboxes_raw_list)
        return (time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw,
                batch_image_size)
    def predict_noise_from_start(self, x_t, t, x0):
        results = (extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                  extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return results

    # 辅助函数：从密度图中提取边界框区域的密度得分
    def _extract_box_density(self, density_map, boxes, img_meta):
        """
        从密度图中提取每个边界框的密度分数

        Args:
            density_map: [1, 1, H, W] 密度图
            boxes: [num_boxes, 4] 框坐标 (x1, y1, x2, y2)
            img_meta: 图像元信息

        Returns:
            Tensor: [num_boxes] 每个框的密度分数
        """
        device = density_map.device
        h, w = density_map.shape[2:]

        # 缩放到密度图尺寸
        img_h, img_w = img_meta['img_shape'][:2]
        scale_x, scale_y = w / img_w, h / img_h

        # 创建结果tensor
        num_boxes = boxes.shape[0]
        density_scores = torch.zeros(num_boxes, device=device)

        for i in range(num_boxes):
            # 框坐标
            x1, y1, x2, y2 = boxes[i]

            # 缩放到密度图尺寸
            x1_d = (x1 * scale_x).int().clamp(0, w - 1)
            y1_d = (y1 * scale_y).int().clamp(0, h - 1)
            x2_d = (x2 * scale_x).int().clamp(0, w - 1)
            y2_d = (y2 * scale_y).int().clamp(0, h - 1)

            # 确保区域有效
            if x1_d == x2_d:
                x2_d = x1_d + 1
            if y1_d == y2_d:
                y2_d = y1_d + 1

            # 提取区域并计算平均密度
            roi = density_map[0, 0, y1_d:y2_d, x1_d:x2_d]
            if roi.numel() > 0:
                density_scores[i] = roi.mean()

        return density_scores

    def inference(self, box_cls, box_pred, cfg, device):
        """
        Args:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for
                each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        if self.use_focal_loss or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(
                self.num_classes,
                device=device).unsqueeze(0).repeat(self.num_proposals,
                                                   1).flatten(0, 1)
            box_pred_list = []
            scores_list = []
            labels_list = []
            for i, (scores_per_image,
                    box_pred_per_image) in enumerate(zip(scores, box_pred)):

                scores_per_image, topk_indices = scores_per_image.flatten(
                    0, 1).topk(
                    self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(
                    1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    box_pred_list.append(box_pred_per_image)
                    scores_list.append(scores_per_image)
                    labels_list.append(labels_per_image)
                    continue

                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    # some nms would reweight the score, such as softnms
                    scores_per_image = det_bboxes[:, -1]
                result = InstanceData()
                result.bboxes = box_pred_per_image
                result.scores = scores_per_image
                result.labels = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second
            # best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image,
                    box_pred_per_image) in enumerate(
                zip(scores, labels, box_pred)):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, \
                        labels_per_image

                if self.use_nms:
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image, scores_per_image, labels_per_image,
                        cfg.nms)
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    # some nms would reweight the score, such as softnms
                    scores_per_image = det_bboxes[:, -1]

                result = InstanceData()
                result.bboxes = box_pred_per_image
                result.scores = scores_per_image
                result.labels = labels_per_image
                results.append(result)
        if self.use_ensemble and self.sampling_timesteps > 1:
            return box_pred_list, scores_list, labels_list
        else:
            return results

    # 在EnhancedDiffusionDetHead类中添加
    def _calc_image_complexity(self, features):
        """优化的图像复杂度计算，减少计算量并提高特征区分能力"""
        # 从特征图计算复杂度指标
        complexity_metrics = []
        for feat in features:
            # 简化计算：直接使用通道间标准差作为空间复杂度指标
            # 这比计算均值再计算方差更高效
            spatial_complexity = feat.std(dim=1).mean(dim=[1, 2])

            # 使用下采样后的特征图计算边缘复杂度，减少计算量
            # 使用平均池化降低分辨率
            if feat.shape[-1] > 64:
                pool_size = 2
                mean_feat = F.avg_pool2d(feat.mean(dim=1, keepdim=True), pool_size)
            else:
                mean_feat = feat.mean(dim=1, keepdim=True)

            # 使用更简单的Sobel算子 (3x3 -> 2x2)
            edge_x = torch.abs(mean_feat[:, :, :, 1:] - mean_feat[:, :, :, :-1]).mean()
            edge_y = torch.abs(mean_feat[:, :, 1:, :] - mean_feat[:, :, :-1, :]).mean()
            edge_magnitude = edge_x + edge_y  # 使用L1范数代替L2范数，避免开方操作

            # 自适应权重：根据特征图大小调整空间复杂度和边缘复杂度的权重
            # 深层特征更关注边缘，浅层特征更关注空间复杂度
            feat_size = feat.shape[-1]
            edge_weight = min(0.8, 0.2 + 0.6 * (1 - feat_size / 128))  # 特征图越小，边缘权重越大

            # 综合复杂度 - 使用自适应权重
            combined_complexity = (1 - edge_weight) * spatial_complexity + edge_weight * edge_magnitude

            # 使用softsign代替sigmoid，计算更简单且数值稳定
            complexity_metrics.append(combined_complexity / (1 + torch.abs(combined_complexity)))

        # 优化权重分配：更注重中层特征
        if len(complexity_metrics) == 4:
            weights = torch.tensor([0.2, 0.4, 0.3, 0.1], device=features[0].device)
        elif len(complexity_metrics) == 3:
            weights = torch.tensor([0.3, 0.5, 0.2], device=features[0].device)
        else:
            # 动态生成权重，中间层权重更高
            n = len(complexity_metrics)
            mid = n // 2
            weights = torch.zeros(n, device=features[0].device)
            for i in range(n):
                weights[i] = 1.0 - 0.8 * abs(i - mid) / n
            weights = weights / weights.sum()

        complexity = torch.stack(complexity_metrics, dim=1) @ weights
        return complexity

    def dynamic_step_scheduler(self, complexity):
        """基于图像复杂度的自适应步数调度"""
        # 基于复杂度的指数增长策略
        base_steps = self.base_steps
        max_steps = self.dynamic_steps

        # 复杂度阈值分段函数
        steps = torch.where(
            complexity > self.complexity_thresh,
            # 高复杂度场景：指数增长
            base_steps * torch.exp((complexity - self.complexity_thresh) * 3),
            # 低复杂度场景：线性增长
            base_steps + (complexity / self.complexity_thresh) * (max_steps - base_steps) * 0.5
        )

        # 确保步数在合理范围内
        steps = steps.clamp(min=base_steps, max=max_steps).long()

        # 记录使用的步数以便调试
        with torch.no_grad():
            self.step_history = torch.roll(self.step_history, 1)
            self.step_history[0] = steps.mean().item()

        return steps

    def _build_teacher(self, config):
        """构建教师模型"""
        # 创建教师模型实例
        if isinstance(config, dict):
            config.update(test_cfg=self.test_cfg)  # 确保使用相同的测试配置
            teacher = type(self)(**config)

            # 冻结教师模型参数
            for param in teacher.parameters():
                param.requires_grad_(False)
            return teacher
        else:
            warnings.warn("无效的教师模型配置")
            return None


    def forward_distill(self, x, init_bboxes, init_t):
        """知识蒸馏前向传播"""
        # 学生模型前向
        student_outputs = self(x, init_bboxes, init_t)

        # 教师模型前向（使用不梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher(x, init_bboxes, init_t)

        return student_outputs, teacher_outputs


    def calc_distill_loss(self, student_features, teacher_features):
        """计算特征蒸馏损失"""
        # 特征对齐 - 确保维度匹配
        if student_features.shape != teacher_features.shape:
            # 简单的平均池化对齐
            if student_features.dim() == 4:  # 2D特征图
                s_h, s_w = student_features.shape[2:]
                t_h, t_w = teacher_features.shape[2:]
                if s_h > t_h or s_w > t_w:
                    # 降采样学生特征到教师尺寸
                    student_features = F.adaptive_avg_pool2d(
                        student_features, (t_h, t_w))
                else:
                    # 降采样教师特征到学生尺寸
                    teacher_features = F.adaptive_avg_pool2d(
                        teacher_features, (s_h, s_w))

        # 计算KL散度损失
        if student_features.dim() > 2:
            # 展平高维特征
            student_features = student_features.flatten(2).mean(dim=2)
            teacher_features = teacher_features.flatten(2).mean(dim=2)

        # 应用温度缩放
        temp = 2.0
        student_features = F.log_softmax(student_features / temp, dim=1)
        teacher_features = F.softmax(teacher_features / temp, dim=1)

        # KL散度损失
        distill_loss = F.kl_div(student_features, teacher_features, reduction='batchmean') * (temp * temp)
        return distill_loss
@MODELS.register_module()
class EnhancedSingleDiffusionDetHead(nn.Module):
    """增强版的SingleDiffusionDetHead，集成了LCM功能"""
    def __init__(
            self,
            num_classes=80,
            feat_channels=256,
            dim_feedforward=2048,
            num_cls_convs=1,
            num_reg_convs=3,
            num_heads=8,
            dropout=0.0,
            pooler_resolution=7,
            scale_clamp=_DEFAULT_SCALE_CLAMP,
            bbox_weights=(2.0, 2.0, 1.0, 1.0),
            use_focal_loss=True,
            use_fed_loss=False,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
            # LCM参数
            use_lcm=False
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.use_lcm = use_lcm

        # Dynamic
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = build_activation_layer(act_cfg)

        # block time mlp
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(feat_channels * 4, feat_channels * 2))

        # cls.
        cls_module = list()
        for _ in range(num_cls_convs):
            cls_module.append(nn.Linear(feat_channels, feat_channels, False))
            cls_module.append(nn.LayerNorm(feat_channels))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        reg_module = list()
        for _ in range(num_reg_convs):
            reg_module.append(nn.Linear(feat_channels, feat_channels, False))
            reg_module.append(nn.LayerNorm(feat_channels))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss
        if self.use_focal_loss or self.use_fed_loss:
            self.class_logits = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta = nn.Linear(feat_channels, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        # LCM特定处理，额外的特征增强模块
        if self.use_lcm:
            self.lcm_feature_enhance = nn.Sequential(
                nn.Linear(feat_channels, feat_channels),
                nn.LayerNorm(feat_channels),
                nn.GELU(),
            )

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, num_boxes, 4)
        :param pro_features: (N, num_boxes, feat_channels)
        """

        N, num_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b])
        rois = bbox2roi(proposal_boxes)

        roi_features = pooler(features, rois)

        # 如果pro_features为None，则初始化它
        if pro_features is None:
            pro_features = roi_features.view(N, num_boxes, self.feat_channels, -1).mean(-1)

        # 以下代码不再放在if条件内，确保总是执行
        roi_features = roi_features.view(N * num_boxes, self.feat_channels, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, num_boxes, self.feat_channels).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(
            num_boxes, N, self.feat_channels).permute(1, 0, 2).reshape(1, N * num_boxes, self.feat_channels)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * num_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        # 如果启用LCM，增强特征表示
        if self.use_lcm:
            fc_feature = self.lcm_feature_enhance(fc_feature) + fc_feature

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (class_logits.view(N, num_boxes, -1),
                pred_bboxes.view(N, num_boxes, -1),
                obj_features)

    def apply_deltas(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

    # 保持DynamicConv类不变
class DynamicConv(nn.Module):
    def __init__(self,
                 feat_channels: int,
                 dynamic_dim: int = 64,
                 dynamic_num: int = 2,
                 pooler_resolution: int = 7) -> None:
        super().__init__()

        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim
        self.dynamic_layer = nn.Linear(self.feat_channels,
                                       self.dynamic_num * self.num_params)

        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.feat_channels * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, pro_features: Tensor, roi_features: Tensor) -> Tensor:
        """Forward function.

        Args:
            pro_features: (1,  N * num_boxes, self.feat_channels)
            roi_features: (49, N * num_boxes, self.feat_channels)

        Returns:
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(
            -1, self.feat_channels, self.dynamic_dim)
        param2 = parameters[:, :,
                 self.num_params:].view(-1, self.dynamic_dim,
                                        self.feat_channels)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


