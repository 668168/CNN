import torch
import torch.nn as nn

# 假设输入的特征图大小为 [batch_size, channels, height, width]
input_tensor = torch.randn(1, 64, 10, 10)  # 例如，1个样本，64个通道，10x10的特征图

# 创建一个AdaptiveAvgPool2d层，目标输出大小为 5x7
adaptive_pool = nn.AdaptiveAvgPool2d((5, 7))

# 应用自适应平均池化
output_tensor = adaptive_pool(input_tensor)

print("Output size:", output_tensor.size())
