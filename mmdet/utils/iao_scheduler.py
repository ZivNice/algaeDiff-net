# mmdet/models/utils/iao_scheduler.py

import torch
import numpy as np
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class IAOScheduler(BaseModule):
    """智能增强优化的动态调度算法

    根据训练进度和模型性能动态调整增强策略的应用概率和参数，
    实现自适应的数据增强调度。

    Args:
        strategy_num (int): 增强策略数量
        warmup_iters (int): 预热迭代次数
        cooldown_iters (int): 冷却迭代次数
        max_iters (int): 最大迭代次数
        init_probs (list, optional): 初始策略概率
        min_probs (list, optional): 最小策略概率
        init_cfg (dict, optional): 初始化配置
    """

    def __init__(self,
                 strategy_num=8,
                 warmup_iters=1000,
                 cooldown_iters=5000,
                 max_iters=90000,
                 init_probs=None,
                 min_probs=None,
                 init_cfg=None):
        super(IAOScheduler, self).__init__(init_cfg)

        self.strategy_num = strategy_num
        self.warmup_iters = warmup_iters
        self.cooldown_iters = cooldown_iters
        self.max_iters = max_iters

        # 初始化策略概率
        if init_probs is None:
            init_probs = [1.0 / strategy_num] * strategy_num
        self.register_buffer('init_probs', torch.tensor(init_probs))

        # 最小策略概率
        if min_probs is None:
            min_probs = [0.05] * strategy_num
        self.register_buffer('min_probs', torch.tensor(min_probs))

        # 当前策略概率
        self.register_buffer('current_probs', torch.tensor(init_probs))

        # 策略效果历史
        self.register_buffer('strategy_history', torch.zeros(strategy_num, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('history_full', torch.tensor(False))

        # 性能指标历史
        self.register_buffer('performance_history', torch.zeros(10))

    def update(self, iteration, strategy_stats, performance=None):
        """更新调度策略

        Args:
            iteration (int): 当前迭代次数
            strategy_stats (dict): 策略使用统计
            performance (float, optional): 当前性能指标

        Returns:
            dict: 更新后的策略参数
        """
        device = self.current_probs.device

        # 预热阶段使用初始概率
        if iteration < self.warmup_iters:
            return {'probs': self.init_probs.clone()}

        # 冷却阶段逐渐恢复均匀分布
        if iteration > self.max_iters - self.cooldown_iters:
            progress = (iteration - (self.max_iters - self.cooldown_iters)) / self.cooldown_iters
            uniform_probs = torch.ones_like(self.current_probs) / self.strategy_num
            self.current_probs = (1 - progress) * self.current_probs + progress * uniform_probs
            return {'probs': self.current_probs.clone()}

        # 更新性能历史
        if performance is not None:
            idx = int(self.history_ptr)
            self.performance_history[idx] = performance

        # 更新策略效果历史
        for i in range(self.strategy_num):
            if f'strategy_{i}_avg_loss' in strategy_stats:
                self.strategy_history[i, idx] = strategy_stats[f'strategy_{i}_avg_loss']

        # 更新指针
        self.history_ptr = (self.history_ptr + 1) % 10
        if self.history_ptr == 0:
            self.history_full = torch.tensor(True)

        # 如果历史数据不足，返回当前概率
        if not self.history_full:
            return {'probs': self.current_probs.clone()}

        # 计算每个策略的效果得分
        strategy_scores = torch.zeros(self.strategy_num, device=device)

        for i in range(self.strategy_num):
            # 计算策略使用前后的性能变化
            if torch.all(self.strategy_history[i] > 0):
                avg_loss = self.strategy_history[i].mean()
                # 损失越小，得分越高
                strategy_scores[i] = 1.0 / (avg_loss + 1e-6)

        # 归一化得分
        if torch.sum(strategy_scores) > 0:
            strategy_scores = strategy_scores / torch.sum(strategy_scores)
        else:
            strategy_scores = torch.ones_like(strategy_scores) / self.strategy_num

        # 更新策略概率 (指数移动平均)
        alpha = 0.8  # 平滑因子
        new_probs = alpha * self.current_probs + (1 - alpha) * strategy_scores

        # 确保最小概率
        new_probs = torch.max(new_probs, self.min_probs)
        new_probs = new_probs / torch.sum(new_probs)  # 重新归一化

        self.current_probs = new_probs

        return {'probs': self.current_probs.clone()}

    def get_params(self, strategy_idx):
        """获取特定策略的参数

        根据训练进度和策略效果动态调整策略参数

        Args:
            strategy_idx (int): 策略索引

        Returns:
            dict: 策略参数
        """
        # 示例参数调整，可根据实际需求扩展
        params = {}

        # 根据策略类型设置不同参数
        if strategy_idx == 0:  # mixup
            params['alpha'] = 0.8
        elif strategy_idx == 1:  # gridmask
            params['ratio'] = 0.5
        elif strategy_idx == 2:  # mosaic
            params['center_ratio'] = 0.5
        elif strategy_idx == 3:  # color_jitter
            params['brightness'] = 0.4
            params['contrast'] = 0.4
            params['saturation'] = 0.4

        return params
