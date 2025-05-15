# mmdet/engine/hooks/iao_hook.py

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class IAOHook(Hook):
    """智能增强优化Hook，用于动态调整数据增强策略

    在训练过程中周期性地更新数据增强策略网络，根据检测损失反馈
    优化增强策略选择。

    Args:
        update_interval (int): 更新策略网络的迭代间隔
        lr (float): 策略网络的学习率
        weight_decay (float): 权重衰减系数
        priority (str or int): Hook优先级
    """

    def __init__(self,
                 update_interval=500,
                 lr=1e-4,
                 weight_decay=1e-5,
                 priority='NORMAL'):
        self.update_interval = update_interval
        self.lr = lr
        self.weight_decay = weight_decay
        self.priority = priority

    def before_train(self, runner):
        """初始化策略网络优化器

        Args:
            runner (Runner): mmengine的运行器实例
        """
        # 获取模型实例
        model = runner.model

        # 初始化策略网络优化器
        # 注意：在mmdet 3.x中，模型结构有所变化
        policy_params = model.bbox_head.policy_net.parameters()

        # 创建优化器并保存到模型中
        model.iao_optimizer = torch.optim.Adam(
            policy_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 记录到日志
        runner.logger.info(
            f'IAOHook initialized with update interval {self.update_interval}'
        )

    def after_train_iter(self, runner):
        """每次训练迭代后执行策略更新

        Args:
            runner (Runner): mmengine的运行器实例
        """
        # 检查是否需要更新
        if runner.iter % self.update_interval != 0:
            return

        # 获取模型和数据
        model = runner.model
        data_batch = runner.train_dataloader.current_batch

        # 提取特征和策略权重
        with torch.no_grad():
            # 获取图像和目标框
            images = data_batch['inputs']
            gt_bboxes = data_batch['data_samples'].gt_instances.bboxes

            # 提取特征
            features = model.bbox_head.feat_extractor(images, gt_bboxes)

        # 计算策略网络输出
        policy_out = model.bbox_head.policy_net(features, runner.iter)
        strategy_weights = policy_out['strategy_weights']

        # 获取当前迭代的损失
        loss_dict = runner.train_metrics

        # 计算策略梯度
        policy_loss = self._calculate_policy_loss(loss_dict, strategy_weights)

        # 更新策略网络
        model.iao_optimizer.zero_grad()
        policy_loss.backward()
        model.iao_optimizer.step()

        # 记录策略损失
        runner.log_buffer.update({'policy_loss': policy_loss.item()})

        # 记录当前策略权重
        for i, weight in enumerate(strategy_weights.detach().cpu().numpy()):
            runner.log_buffer.update({f'strategy_{i}_weight': float(weight)})

    def _calculate_policy_loss(self, loss_dict, strategy_weights):
        """计算策略网络的损失

        使用强化学习方法，根据检测损失反馈优化策略选择

        Args:
            loss_dict (dict): 检测模型的损失字典
            strategy_weights (Tensor): 策略网络输出的权重

        Returns:
            Tensor: 策略网络的损失
        """
        # 计算总损失
        total_loss = sum(v for k, v in loss_dict.items()
                         if 'loss' in k and isinstance(v, torch.Tensor))

        # 计算策略网络的损失（强化学习中的策略梯度）
        log_probs = torch.log(strategy_weights + 1e-10)
        policy_loss = (log_probs * total_loss.detach()).mean()

        return policy_loss

    def after_train_epoch(self, runner):
        """每个训练周期后记录策略统计信息

        Args:
            runner (Runner): mmengine的运行器实例
        """
        # 获取模型
        model = runner.model

        # 记录当前策略统计信息
        if hasattr(model.bbox_head, 'policy_net') and \
                hasattr(model.bbox_head.policy_net, 'get_stats'):
            stats = model.bbox_head.policy_net.get_stats()
            for k, v in stats.items():
                runner.logger.info(f'{k}: {v}')
