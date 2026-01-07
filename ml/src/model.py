"""
神经网络模型定义 - 文件类型分类器
"""
import torch
import torch.nn as nn
from typing import List, Optional


class SimpleNet(nn.Module):
    """
    简单的全连接网络
    适用于快速验证和小数据集
    """

    def __init__(
        self,
        input_dim: int = 261,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepNet(nn.Module):
    """
    深层全连接网络
    支持批归一化和残差连接
    """

    def __init__(
        self,
        input_dim: int = 261,
        num_classes: int = 10,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.use_batch_norm = use_batch_norm

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim: int, dropout: float = 0.3, use_batch_norm: bool = True):
        super().__init__()

        layers = [nn.Linear(dim, dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim)])
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ResNet(nn.Module):
    """
    带残差连接的网络
    适用于更深的网络结构
    """

    def __init__(
        self,
        input_dim: int = 261,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        )

        # 残差块
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout, use_batch_norm) for _ in range(num_blocks)]
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)


def create_model(
    model_type: str = "deep",
    input_dim: int = 261,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    创建模型的工厂函数

    Args:
        model_type: "simple", "deep", "resnet"
        input_dim: 输入特征维度
        num_classes: 分类数
        **kwargs: 传递给模型的额外参数

    Returns:
        模型实例
    """
    models = {
        "simple": SimpleNet,
        "deep": DeepNet,
        "resnet": ResNet,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](input_dim=input_dim, num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_dim: int = 261) -> str:
    """生成模型摘要"""
    lines = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append(f"Parameters: {count_parameters(model):,}")
    lines.append("-" * 50)

    for name, module in model.named_modules():
        if name:
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                lines.append(f"{name}: {module.__class__.__name__} ({params:,} params)")

    return "\n".join(lines)


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("模型测试")
    print("=" * 60)

    batch_size = 32
    input_dim = 261
    num_classes = 10

    x = torch.randn(batch_size, input_dim)

    for model_type in ["simple", "deep", "resnet"]:
        print(f"\n--- {model_type.upper()} ---")
        model = create_model(model_type, input_dim, num_classes)
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(model_summary(model))
