import torch
import torch.nn as nn
import torch.nn.functional as F

class EGA_Attention(nn.Module):
    """
    基于结构图实现的EGA注意力模块
    包含EGA（边缘高斯聚合）、GAP（全局平均池化）、1D卷积、Sigmoid激活、逐元素乘法和加法
    """
    def __init__(self, channels, kernel_size=3):
        """
        :param channels: 输入特征的通道数
        :param kernel_size: 1D卷积的核大小，默认为3
        """
        super(EGA_Attention, self).__init__()
        # EGA模块，这里用深度可分离卷积模拟
        self.ega = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        # 1D卷积，处理通道维度
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # Sigmoid激活
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: 输入特征，形状为 [B, C, H, W]
        :return: 输出特征，形状同输入
        """
        # 1. EGA模块，提取边缘特征
        fega = self.ega(x)  # [B, C, H, W]

        # 2. GAP，全局平均池化，得到每个通道的全局特征
        gap = F.adaptive_avg_pool2d(fega, 1)  # [B, C, 1, 1]
        gap = gap.view(x.size(0), x.size(1))  # [B, C]

        # 3. 1D卷积，增强通道间的关系
        gap = gap.unsqueeze(1)  # [B, 1, C]
        conv_out = self.conv1d(gap)  # [B, 1, C]
        conv_out = conv_out.squeeze(1)  # [B, C]

        # 4. Sigmoid激活，获得通道注意力权重
        attn = self.sigmoid(conv_out).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 5. 逐元素乘法和加法，融合输入特征和注意力特征
        out = x * attn + x  # [B, C, H, W]

        return out