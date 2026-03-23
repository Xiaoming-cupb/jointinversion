import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, feature_dim=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_dim)
        self.prelu = nn.PReLU(feature_dim)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_dim)
        
        # He normal initialization
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # 残差连接
        return out


class FullyConvModel(nn.Module):
    """全卷积模型实现"""
    def __init__(self, input_channels=2, n_init_features=16):
        super(FullyConvModel, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.initial_prelu = nn.PReLU(16)
        
        # He normal initialization
        nn.init.kaiming_normal_(self.initial_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # 构建主体网络层
        self.layers = nn.ModuleList()
        
        # 4个主要的卷积-批归一化-激活-残差块组合
        current_channels = 16
        for i in range(4):
            # 卷积层
            conv = nn.Conv2d(current_channels, n_init_features, kernel_size=3, padding=1)
            nn.init.orthogonal_(conv.weight)  # Orthogonal initialization
            
            # 批归一化层
            bn = nn.BatchNorm2d(n_init_features, momentum=0.01, eps=0.0001)  # PyTorch默认momentum是0.1，这里转换为0.01
            
            # ReLU激活
            relu = nn.ReLU(inplace=True)
            
            # 残差块
            residual = ResidualBlock(feature_dim=n_init_features)
            
            self.layers.extend([conv, bn, relu, residual])
            current_channels = n_init_features
        
        # 最终输出层
        self.final_conv = nn.Conv2d(n_init_features, 1, kernel_size=1, padding=0)
        nn.init.orthogonal_(self.final_conv.weight)  # Orthogonal initialization
        
    def forward(self, x):
        # 初始卷积
        x = self.initial_conv(x)
        x = self.initial_prelu(x)
        
        # 主体网络
        for i in range(0, len(self.layers), 4):
            conv = self.layers[i]
            bn = self.layers[i + 1]
            relu = self.layers[i + 2]
            residual = self.layers[i + 3]

            x = conv(x)
            x = bn(x)
            x = relu(x)
            x = residual(x)

        # 最终输出
        x1 = self.final_conv(x)

        return x1


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = FullyConvModel(n_init_features=16)
    
    model = model.to('cuda')
    
    # 创建测试输入 (batch_size, channels, height, width)
    test_input = torch.randn(1, 2, 256, 256).to('cuda')
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
        print(f"\nInput shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
