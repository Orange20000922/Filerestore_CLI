"""
C++ 生成数据集加载器

加载由 C++ mlscan 命令生成的特征数据集，用于模型训练。
支持 CSV 和二进制格式。

用法:
    python train.py --csv ml_dataset.csv
    python train.py --binary ml_dataset.bin
"""

import numpy as np
import pandas as pd
import struct
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder


class CppDatasetLoader:
    """C++ 生成数据集加载器"""

    FEATURE_DIM = 261  # 256 字节频率 + 5 统计特征
    BINARY_MAGIC = b'MLFD'
    BINARY_VERSION = 1

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def load_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        加载 CSV 格式数据集

        Args:
            csv_path: CSV 文件路径

        Returns:
            features: 特征数组 (N, 261)
            labels: 标签数组 (N,)
            label_map: {索引: 类型名} 映射
        """
        print(f"Loading CSV dataset: {csv_path}")

        df = pd.read_csv(csv_path)

        # 检查列数
        if len(df.columns) < self.FEATURE_DIM + 1:
            raise ValueError(f"CSV must have at least {self.FEATURE_DIM + 1} columns")

        # 提取特征（前 261 列）
        feature_cols = [f'f{i}' for i in range(self.FEATURE_DIM)]
        if all(col in df.columns for col in feature_cols):
            features = df[feature_cols].values.astype(np.float32)
        else:
            # 如果列名不匹配，使用位置索引
            features = df.iloc[:, :self.FEATURE_DIM].values.astype(np.float32)

        # 提取标签
        if 'extension' in df.columns:
            labels_str = df['extension'].values
        else:
            # 假设标签在第 262 列
            labels_str = df.iloc[:, self.FEATURE_DIM].values

        # 编码标签
        labels = self.label_encoder.fit_transform(labels_str)

        # 构建标签映射
        label_map = {i: name for i, name in enumerate(self.label_encoder.classes_)}

        print(f"  Loaded {len(features)} samples")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Classes: {list(label_map.values())}")

        return features, labels, label_map

    def load_binary(self, bin_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        加载二进制格式数据集

        Args:
            bin_path: 二进制文件路径

        Returns:
            features: 特征数组 (N, 261)
            labels: 标签数组 (N,)
            label_map: {索引: 类型名} 映射
        """
        print(f"Loading binary dataset: {bin_path}")

        features_list = []
        labels_list = []

        with open(bin_path, 'rb') as f:
            # 读取头部 (16 bytes)
            magic = f.read(4)
            if magic != self.BINARY_MAGIC:
                raise ValueError(f"Invalid magic: {magic}, expected {self.BINARY_MAGIC}")

            version = struct.unpack('<I', f.read(4))[0]
            if version != self.BINARY_VERSION:
                raise ValueError(f"Unsupported version: {version}")

            sample_count = struct.unpack('<I', f.read(4))[0]
            feature_dim = struct.unpack('<I', f.read(4))[0]

            if feature_dim != self.FEATURE_DIM:
                raise ValueError(f"Feature dimension mismatch: {feature_dim} vs {self.FEATURE_DIM}")

            # 跳过保留字段
            f.read(8)

            print(f"  Header: {sample_count} samples, {feature_dim} features")

            # 读取样本
            for _ in range(sample_count):
                # 读取特征
                feat_data = f.read(self.FEATURE_DIM * 4)
                if len(feat_data) < self.FEATURE_DIM * 4:
                    break
                features = np.frombuffer(feat_data, dtype=np.float32)
                features_list.append(features)

                # 读取扩展名
                ext_len = struct.unpack('<B', f.read(1))[0]
                ext = f.read(ext_len).decode('utf-8')
                labels_list.append(ext)

                # 注意：如果 includeFilePath 为 true，还需要读取路径
                # 这里假设不包含路径（默认配置）

        features = np.array(features_list, dtype=np.float32)
        labels_str = np.array(labels_list)

        # 编码标签
        labels = self.label_encoder.fit_transform(labels_str)
        label_map = {i: name for i, name in enumerate(self.label_encoder.classes_)}

        print(f"  Loaded {len(features)} samples")
        print(f"  Classes: {list(label_map.values())}")

        return features, labels, label_map

    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        自动检测格式并加载数据集

        Args:
            path: 数据集文件路径

        Returns:
            features, labels, label_map
        """
        path = str(path)
        if path.endswith('.csv'):
            return self.load_csv(path)
        elif path.endswith('.bin'):
            return self.load_binary(path)
        else:
            # 尝试读取文件头判断格式
            with open(path, 'rb') as f:
                magic = f.read(4)
            if magic == self.BINARY_MAGIC:
                return self.load_binary(path)
            else:
                return self.load_csv(path)


def load_cpp_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    便捷函数：加载 C++ 生成的数据集

    Args:
        path: 数据集文件路径（CSV 或二进制）

    Returns:
        features: 特征数组 (N, 261)
        labels: 标签数组 (N,)
        label_map: {索引: 类型名} 映射
    """
    loader = CppDatasetLoader()
    return loader.load(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cpp_dataset_loader.py <dataset_path>")
        print("Example: python cpp_dataset_loader.py ml_dataset.csv")
        sys.exit(1)

    path = sys.argv[1]
    features, labels, label_map = load_cpp_dataset(path)

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(labels)}")
    print(f"Feature shape: {features.shape}")
    print(f"Number of classes: {len(label_map)}")
    print("\nClass distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  {label_map[idx]}: {count}")

    print("\nFeature statistics:")
    print(f"  Mean: {features.mean():.6f}")
    print(f"  Std:  {features.std():.6f}")
    print(f"  Min:  {features.min():.6f}")
    print(f"  Max:  {features.max():.6f}")
