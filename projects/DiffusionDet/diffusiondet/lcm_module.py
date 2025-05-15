import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入（优化版）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 预计算常量，避免重复计算log和除法
        self.freq_factor = 1.0 / (10000 ** (torch.arange(0, dim // 2) / (dim // 2 - 1)))
        # 注册为buffer而非参数，不参与梯度更新但会被保存
        self.register_buffer('freq_factor', self.freq_factor, persistent=False)

    def forward(self, time):
        """
        Args:
            time: [batch_size] 时间步张量
        Returns:
            [batch_size, dim] 位置编码
        """
        # 直接使用预计算的频率因子，减少计算量
        t = time.unsqueeze(-1)  # [batch_size, 1]
        # 广播乘法，避免循环
        t_freq = t * self.freq_factor  # [batch_size, dim//2]

        # 使用torch.cat而非拼接两个单独计算的张量，减少内存分配
        emb = torch.zeros(time.shape[0], self.dim, device=time.device)
        emb[:, 0::2] = torch.sin(t_freq)  # 偶数位置
        emb[:, 1::2] = torch.cos(t_freq)  # 奇数位置

        return emb


class LCMMapper(nn.Module):
    """LCM一致性映射层（优化版）"""

    def __init__(self, feat_channels):
        super().__init__()
        # 使用Sequential一次性定义所有层，更清晰且可能更优化
        hidden_dim = int(feat_channels * 4 / 3)  # 减小隐藏层大小以提高效率

        # 特征处理器 - 使用更高效的SiLU激活函数替代GELU
        self.mapper = nn.Sequential(
            nn.Linear(feat_channels, hidden_dim),
            nn.SiLU(),  # SiLU/Swish通常比GELU更快且效果相当
            nn.Linear(hidden_dim, feat_channels)
        )

        # 保持输出层
        self.output_layer = nn.Linear(feat_channels, 4)

        # 初始化权重以提高训练稳定性
        self._reset_parameters()

    def _reset_parameters(self):
        # 使用xavier初始化以获得更好的训练开始点
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 保持原始功能，但可能得益于优化的内部结构
        features = self.mapper(x)
        offsets = self.output_layer(features)
        return offsets


class LCMTimeEmbedding(nn.Module):
    """LCM时间嵌入层（优化版）"""

    def __init__(self, feat_channels):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(feat_channels)

        # 使用更高效的SiLU激活函数
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels, feat_channels * 4),
            nn.SiLU(),
            nn.Linear(feat_channels * 4, feat_channels)
        )

        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t):
        # 首先通过正弦位置编码
        t_emb = self.sinusoidal(t)
        # 然后通过MLP
        return self.mlp(t_emb)
