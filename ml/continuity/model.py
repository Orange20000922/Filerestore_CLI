"""
Block Continuity Classifier Model

用于判断两个相邻数据块是否属于同一文件的二分类模型。
输入: 64维特征向量
输出: 2类 (连续/不连续)
"""

import torch
import torch.nn as nn


class ContinuityClassifier(nn.Module):
    """
    块连续性二分类器

    使用多层全连接网络，带 BatchNorm 和 Dropout 正则化。
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回概率分布"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """返回预测标签"""
        proba = self.predict_proba(x)
        return (proba[:, 1] > threshold).long()


class ContinuityClassifierLight(nn.Module):
    """
    轻量级版本，适用于推理速度要求高的场景
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContinuityClassifierResidual(nn.Module):
    """
    带残差连接的版本，适用于更深的网络
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


def get_model(
    model_type: str = "default",
    input_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    获取模型实例

    Args:
        model_type: 模型类型 ("default", "light", "residual")
        input_dim: 输入维度
        **kwargs: 其他模型参数

    Returns:
        模型实例
    """
    if model_type == "light":
        return ContinuityClassifierLight(input_dim=input_dim, **kwargs)
    elif model_type == "residual":
        return ContinuityClassifierResidual(input_dim=input_dim, **kwargs)
    else:
        return ContinuityClassifier(input_dim=input_dim, **kwargs)


if __name__ == "__main__":
    # 测试模型
    batch_size = 32
    input_dim = 64

    # 测试默认模型
    model = ContinuityClassifier(input_dim=input_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"Default model output shape: {output.shape}")  # [32, 2]

    # 测试轻量级模型
    model_light = ContinuityClassifierLight(input_dim=input_dim)
    output_light = model_light(x)
    print(f"Light model output shape: {output_light.shape}")  # [32, 2]

    # 测试残差模型
    model_res = ContinuityClassifierResidual(input_dim=input_dim)
    output_res = model_res(x)
    print(f"Residual model output shape: {output_res.shape}")  # [32, 2]

    # 统计参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter counts:")
    print(f"  Default: {count_params(model):,}")
    print(f"  Light: {count_params(model_light):,}")
    print(f"  Residual: {count_params(model_res):,}")
