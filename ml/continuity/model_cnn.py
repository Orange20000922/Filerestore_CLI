"""
1D CNN Model for Block Continuity Detection

Uses 1D convolutions on 64-dimensional feature vectors to detect
whether two adjacent data blocks belong to the same file.

Architecture:
- Input: (batch, 64) feature vector
- Reshape to (batch, 1, 64) for Conv1d
- 3 Conv1d layers: 32 -> 64 -> 128 channels
- Global Average Pooling
- FC classifier: 128 -> 64 -> 2

Compared to FC baseline:
- Better pattern detection across feature groups
- Parameter sharing reduces overfitting
- ~40K parameters (vs ~20K for FC)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ContinuityCNN1D(nn.Module):
    """
    1D CNN for block continuity classification

    The 64-dim feature vector is treated as a 1D signal:
    - Features 0-15: Block 1 statistics
    - Features 16-31: Block 2 statistics
    - Features 32-47: Boundary features
    - Features 48-63: ZIP-specific features

    Conv1d can learn patterns across these feature groups.
    """

    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 2,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build conv layers
        conv_layers = []
        in_channels = 1

        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            # Conv1d with same padding
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )

            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))

            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(dropout))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, 64)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Reshape: (batch, 64) -> (batch, 1, 64)
        x = x.unsqueeze(1)

        # Conv layers
        x = self.conv_layers(x)  # (batch, 128, 64)

        # Global average pooling
        x = self.gap(x)  # (batch, 128, 1)

        # Classifier
        x = self.classifier(x)  # (batch, num_classes)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability distribution"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return predicted labels"""
        proba = self.predict_proba(x)
        return (proba[:, 1] > threshold).long()


class ContinuityCNNResidual(nn.Module):
    """
    1D CNN with residual connections

    Better for deeper networks, prevents gradient vanishing.
    """

    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 2,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, dropout)
            for _ in range(num_blocks)
        ])

        # Global pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, 64)
        x = self.input_conv(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.gap(x)
        x = self.classifier(x)
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual block"""

    def __init__(self, channels: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ContinuityCNNLight(nn.Module):
    """
    Lightweight 1D CNN for fast inference

    Fewer parameters, suitable for edge deployment.
    """

    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv = nn.Sequential(
            # Single conv layer
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 64 -> 32

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4)  # -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.classifier(x)
        return x


# =============================================================================
# Model Factory
# =============================================================================

def get_cnn_model(
    model_type: str = "cnn1d",
    input_dim: int = 64,
    **kwargs
) -> nn.Module:
    """
    Get CNN model instance

    Args:
        model_type: "cnn1d", "cnn_residual", "cnn_light"
        input_dim: Input feature dimension
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    models = {
        "cnn1d": ContinuityCNN1D,
        "cnn_residual": ContinuityCNNResidual,
        "cnn_light": ContinuityCNNLight,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](input_dim=input_dim, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    batch_size = 32
    input_dim = 64

    print("=" * 60)
    print("1D CNN Model Tests")
    print("=" * 60)

    # Test each model type
    for model_type in ["cnn1d", "cnn_residual", "cnn_light"]:
        print(f"\n{model_type}:")
        model = get_cnn_model(model_type, input_dim=input_dim)

        # Test forward pass
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters:   {count_parameters(model):,}")

        # Test inference
        proba = model.predict_proba(x)
        print(f"  Proba shape:  {proba.shape}")
        print(f"  Proba sum:    {proba.sum(dim=1).mean():.4f} (should be ~1.0)")

    # Compare with FC baseline
    print("\n" + "=" * 60)
    print("Comparison with FC baseline:")
    print("=" * 60)

    # Simple FC model for comparison
    class FCBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2)
            )

        def forward(self, x):
            return self.net(x)

    fc_model = FCBaseline()
    cnn_model = get_cnn_model("cnn1d")

    print(f"\nFC Baseline:   {count_parameters(fc_model):,} parameters")
    print(f"CNN 1D:        {count_parameters(cnn_model):,} parameters")

    # ONNX export test
    print("\n" + "=" * 60)
    print("ONNX Export Test:")
    print("=" * 60)

    try:
        import onnx
        import tempfile
        import os

        model = get_cnn_model("cnn1d")
        model.eval()

        dummy_input = torch.randn(1, 64)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            temp_path = f.name

        torch.onnx.export(
            model,
            dummy_input,
            temp_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['logits'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )

        # Verify
        onnx_model = onnx.load(temp_path)
        onnx.checker.check_model(onnx_model)

        file_size = os.path.getsize(temp_path)
        print(f"  Export successful!")
        print(f"  File size: {file_size / 1024:.1f} KB")

        os.unlink(temp_path)

    except ImportError:
        print("  ONNX not installed, skipping export test")
    except Exception as e:
        print(f"  Export failed: {e}")

    print("\nAll tests passed!")
