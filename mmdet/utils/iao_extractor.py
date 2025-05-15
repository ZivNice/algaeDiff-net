# mmdet/models/utils/iao_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class IAOFeatureExtractor(BaseModule):
    """智能增强优化的特征提取器

    从图像和目标框中提取用于策略决策的特征，包括:
    - 图像全局特征
    - 目标框统计特征
    - 图像质量指标

    Args:
        in_channels (int): 输入特征通道数
        feat_dim (int): 输出特征维度
        init_cfg (dict, optional): 初始化配置
    """

    def __init__(self,
                 in_channels=256,
                 feat_dim=128,
                 init_cfg=None):
        super(IAOFeatureExtractor, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.feat_dim = feat_dim

        # 图像特征提取网络
        self.img_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 目标框特征提取网络
        self.bbox_encoder = nn.Sequential(
            nn.Linear(5, 32),  # [x, y, w, h, area]
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )

        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 10, 128),  # 图像特征 + 框特征 + 统计特征
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, images, gt_bboxes):
        """提取特征

        Args:
            images (Tensor): 形状为 (N, C, H, W) 的图像张量
            gt_bboxes (list[Tensor]): 每张图像的目标框列表

        Returns:
            Tensor: 形状为 (N, feat_dim) 的特征张量
        """
        batch_size = images.size(0)
        device = images.device

        # 提取图像特征
        img_feats = self.img_encoder(images).view(batch_size, -1)

        # 计算每张图像的目标框特征
        bbox_feats = []
        stat_feats = []

        for i, bboxes in enumerate(gt_bboxes):
            if len(bboxes) == 0:
                # 如果没有目标框，使用零向量
                bbox_feat = torch.zeros(64, device=device)
                stat_feat = torch.zeros(10, device=device)
            else:
                # 计算目标框特征
                # [x1, y1, x2, y2] -> [x, y, w, h, area]
                bbox_info = torch.zeros((len(bboxes), 5), device=device)
                bbox_info[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x中心
                bbox_info[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y中心
                bbox_info[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # 宽度
                bbox_info[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # 高度
                bbox_info[:, 4] = bbox_info[:, 2] * bbox_info[:, 3]  # 面积

                # 归一化
                h, w = images.shape[2:]
                bbox_info[:, 0] /= w
                bbox_info[:, 1] /= h
                bbox_info[:, 2] /= w
                bbox_info[:, 3] /= h
                bbox_info[:, 4] /= (w * h)

                # 编码每个框并平均
                box_feats = self.bbox_encoder(bbox_info)
                bbox_feat = box_feats.mean(dim=0)

                # 计算统计特征
                stat_feat = torch.zeros(10, device=device)
                stat_feat[0] = len(bboxes)  # 目标数量
                stat_feat[1] = bbox_info[:, 4].mean()  # 平均面积
                stat_feat[2] = bbox_info[:, 4].std() + 1e-6  # 面积标准差
                stat_feat[3] = bbox_info[:, 2].mean()  # 平均宽度
                stat_feat[4] = bbox_info[:, 3].mean()  # 平均高度
                stat_feat[5] = (bbox_info[:, 2] / (bbox_info[:, 3] + 1e-6)).mean()  # 平均宽高比

                # 计算目标密度和分布
                stat_feat[6] = len(bboxes) / (w * h / 1000)  # 目标密度

                # 计算图像质量指标 (简化版)
                img_i = images[i].mean(dim=0)  # 平均通道
                stat_feat[7] = img_i.std()  # 对比度
                stat_feat[8] = img_i.mean()  # 亮度

                # 归一化
                stat_feat = F.normalize(stat_feat, dim=0)

            bbox_feats.append(bbox_feat)
            stat_feats.append(stat_feat)

        # 堆叠特征
        bbox_feats = torch.stack(bbox_feats, dim=0)
        stat_feats = torch.stack(stat_feats, dim=0)

        # 融合特征
        concat_feats = torch.cat([img_feats, bbox_feats, stat_feats], dim=1)
        final_feats = self.fusion(concat_feats)

        return final_feats
