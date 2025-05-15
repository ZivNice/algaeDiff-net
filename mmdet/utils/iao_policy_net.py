# mmdet/models/utils/iao_policy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class IAOPolicyNetwork(BaseModule):
    """智能增强优化的策略决策网络

    根据输入特征动态生成数据增强策略权重，支持:
    - 基于特征的策略选择
    - 训练进度感知的调整
    - 策略使用统计跟踪

    Args:
        feat_dim (int): 输入特征维度
        strategy_num (int): 增强策略数量
        hidden_dim (int): 隐藏层维度
        temperature (float): softmax温度系数
        init_cfg (dict, optional): 初始化配置
    """

    def __init__(self,
                 feat_dim=128,
                 strategy_num=8,
                 hidden_dim=64,
                 temperature=1.0,
                 init_cfg=None):
        super(IAOPolicyNetwork, self).__init__(init_cfg)

        self.feat_dim = feat_dim
        self.strategy_num = strategy_num
        self.temperature = temperature

        # 主网络
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),  # +1 for iteration progress
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, strategy_num)
        )

        # 初始化策略选择统计
        self.register_buffer('strategy_counts', torch.zeros(strategy_num))
        self.register_buffer('total_calls', torch.tensor(0.))

        # 初始化策略效果评估
        self.register_buffer('strategy_losses', torch.zeros(strategy_num))
        self.register_buffer('strategy_calls', torch.zeros(strategy_num))

    def forward(self, features, iteration=None, max_iter=90000):
        """生成策略权重

        Args:
            features (Tensor): 形状为 (N, feat_dim) 的特征张量
            iteration (int, optional): 当前迭代次数
            max_iter (int, optional): 最大迭代次数

        Returns:
            dict: 包含策略权重和选择的策略
        """
        batch_size = features.size(0)
        device = features.device

        # 归一化迭代进度
        if iteration is None:
            progress = torch.zeros(batch_size, 1, device=device)
        else:
            progress = torch.ones(batch_size, 1, device=device) * (iteration / max_iter)

        # 拼接特征和进度
        input_feats = torch.cat([features, progress], dim=1)

        # 生成策略权重
        logits = self.net(input_feats)
        weights = F.softmax(logits / self.temperature, dim=1)

        # 对每个样本选择策略
        if self.training:
            # 训练时使用采样
            strategy_indices = torch.multinomial(weights, 1).squeeze(1)
        else:
            # 测试时使用最大概率
            strategy_indices = weights.argmax(dim=1)

        # 更新策略统计
        for idx in strategy_indices:
            self.strategy_counts[idx] += 1
        self.total_calls += batch_size

        return {
            'strategy_weights': weights,
            'strategy_indices': strategy_indices
        }

    def update_strategy_loss(self, strategy_indices, losses):
        """更新策略效果评估

        Args:
            strategy_indices (Tensor): 选择的策略索引
            losses (Tensor): 对应的损失值
        """
        for idx, loss in zip(strategy_indices, losses):
            self.strategy_losses[idx] += loss.item()
            self.strategy_calls[idx] += 1

    def get_stats(self):
        """获取策略使用统计

        Returns:
            dict: 策略使用统计信息
        """
        stats = {}

        # 计算策略使用频率
        if self.total_calls > 0:
            freqs = self.strategy_counts / self.total_calls
            for i, freq in enumerate(freqs):
                stats[f'strategy_{i}_freq'] = freq.item()

        # 计算策略平均损失
        valid_strategies = self.strategy_calls > 0
        if valid_strategies.any():
            avg_losses = torch.zeros_like(self.strategy_losses)
            avg_losses[valid_strategies] = self.strategy_losses[valid_strategies] / self.strategy_calls[
                valid_strategies]

            for i, avg_loss in enumerate(avg_losses):
                if self.strategy_calls[i] > 0:
                    stats[f'strategy_{i}_avg_loss'] = avg_loss.item()

        return stats
