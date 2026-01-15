"""
Continuity Dataset Loader

加载 C++ 端生成的连续性训练数据集。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ContinuityDataset(Dataset):
    """
    连续性数据集

    加载 C++ DatasetGenerator 生成的 CSV 格式数据集。
    CSV 格式: f0,f1,...,f63,is_continuous,file_type,sample_type
    """

    def __init__(
        self,
        csv_path: str,
        feature_dim: int = 64,
        normalize: bool = True,
        norm_params: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Args:
            csv_path: CSV 数据集路径
            feature_dim: 特征维度 (默认 64)
            normalize: 是否标准化特征
            norm_params: 预计算的标准化参数 {"mean": ..., "std": ...}
        """
        self.csv_path = csv_path
        self.feature_dim = feature_dim
        self.normalize = normalize

        # 加载数据
        logger.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)

        # 提取特征列
        feature_cols = [f"f{i}" for i in range(feature_dim)]
        self.features = self.df[feature_cols].values.astype(np.float32)

        # 提取标签
        self.labels = self.df["is_continuous"].values.astype(np.int64)

        # 提取元数据（可选）
        if "file_type" in self.df.columns:
            self.file_types = self.df["file_type"].values
        else:
            self.file_types = None

        if "sample_type" in self.df.columns:
            self.sample_types = self.df["sample_type"].values
        else:
            self.sample_types = None

        # 标准化
        if normalize:
            if norm_params is not None:
                self.mean = norm_params["mean"]
                self.std = norm_params["std"]
            else:
                self.mean = self.features.mean(axis=0)
                self.std = self.features.std(axis=0) + 1e-8

            self.features = (self.features - self.mean) / self.std

        logger.info(f"Loaded {len(self)} samples")
        logger.info(f"  Positive (continuous): {(self.labels == 1).sum()}")
        logger.info(f"  Negative (discontinuous): {(self.labels == 0).sum()}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

    def get_norm_params(self) -> Dict[str, np.ndarray]:
        """获取标准化参数"""
        return {
            "mean": self.mean if hasattr(self, "mean") else None,
            "std": self.std if hasattr(self, "std") else None
        }

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_type_distribution(self) -> Dict[str, int]:
        """获取样本类型分布"""
        if self.sample_types is None:
            return {}
        unique, counts = np.unique(self.sample_types, return_counts=True)
        return dict(zip(unique, counts))


def load_dataset(
    csv_path: str,
    val_split: float = 0.2,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    加载数据集并创建 DataLoader

    Args:
        csv_path: CSV 数据集路径
        val_split: 验证集比例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        random_seed: 随机种子

    Returns:
        (train_loader, val_loader, metadata)
    """
    # 加载完整数据集
    full_dataset = ContinuityDataset(csv_path, normalize=False)

    # 分割训练集和验证集
    np.random.seed(random_seed)
    indices = np.random.permutation(len(full_dataset))
    val_size = int(len(full_dataset) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # 计算训练集的标准化参数
    train_features = full_dataset.features[train_indices]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-8
    norm_params = {"mean": mean, "std": std}

    # 创建标准化后的数据集
    train_dataset = ContinuityDatasetSubset(
        full_dataset, train_indices, norm_params
    )
    val_dataset = ContinuityDatasetSubset(
        full_dataset, val_indices, norm_params
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 元数据
    metadata = {
        "feature_dim": full_dataset.feature_dim,
        "num_classes": 2,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "norm_params": norm_params,
        "class_weights": full_dataset.get_class_weights(),
        "sample_type_distribution": full_dataset.get_sample_type_distribution()
    }

    return train_loader, val_loader, metadata


class ContinuityDatasetSubset(Dataset):
    """数据集子集（带标准化）"""

    def __init__(
        self,
        full_dataset: ContinuityDataset,
        indices: np.ndarray,
        norm_params: Dict[str, np.ndarray]
    ):
        self.features = full_dataset.features[indices]
        self.labels = full_dataset.labels[indices]

        # 应用标准化
        mean = norm_params["mean"]
        std = norm_params["std"]
        self.features = (self.features - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features[idx].astype(np.float32))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


if __name__ == "__main__":
    # 测试数据加载
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

        train_loader, val_loader, metadata = load_dataset(csv_path)

        print(f"\nDataset loaded:")
        print(f"  Feature dim: {metadata['feature_dim']}")
        print(f"  Train size: {metadata['train_size']}")
        print(f"  Val size: {metadata['val_size']}")
        print(f"  Class weights: {metadata['class_weights']}")
        print(f"  Sample type distribution: {metadata['sample_type_distribution']}")

        # 测试一个批次
        for features, labels in train_loader:
            print(f"\nBatch shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label distribution: {labels.bincount()}")
            break
    else:
        print("Usage: python dataset.py <csv_path>")
