#!/usr/bin/env python3
"""
分析训练结果，找出问题所在
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from dataset import ContinuityDataset
from model_cnn import get_cnn_model
from feature_extractor import ContinuityFeatureExtractor


def analyze_dataset(csv_path: str):
    """分析数据集分布"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_path}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)

    # 基本统计
    print(f"\nTotal samples: {len(df)}")
    print(f"Positive (continuous): {(df['is_continuous'] == 1).sum()}")
    print(f"Negative (discontinuous): {(df['is_continuous'] == 0).sum()}")
    print(f"Ratio: {(df['is_continuous'] == 1).sum() / (df['is_continuous'] == 0).sum():.2f}")

    # 按文件类型统计
    if 'file_type' in df.columns:
        print(f"\n--- By File Type ---")
        for ft in df['file_type'].unique():
            subset = df[df['file_type'] == ft]
            pos = (subset['is_continuous'] == 1).sum()
            neg = (subset['is_continuous'] == 0).sum()
            print(f"  {ft}: {len(subset)} samples (pos={pos}, neg={neg})")

    # 按样本类型统计
    if 'sample_type' in df.columns:
        print(f"\n--- By Sample Type ---")
        for st in df['sample_type'].unique():
            subset = df[df['sample_type'] == st]
            pos = (subset['is_continuous'] == 1).sum()
            neg = (subset['is_continuous'] == 0).sum()
            print(f"  {st}: {len(subset)} samples (pos={pos}, neg={neg})")

    # 特征统计
    feature_cols = [f"f{i}" for i in range(64)]
    features = df[feature_cols].values

    print(f"\n--- Feature Statistics ---")
    print(f"Mean range: [{features.mean(axis=0).min():.4f}, {features.mean(axis=0).max():.4f}]")
    print(f"Std range: [{features.std(axis=0).min():.4f}, {features.std(axis=0).max():.4f}]")

    # 检查是否有常量特征
    const_features = []
    for i in range(64):
        if features[:, i].std() < 1e-6:
            const_features.append(i)
    if const_features:
        print(f"WARNING: Constant features (no variance): {const_features}")

    # 检查正负样本的特征分布差异
    pos_features = features[df['is_continuous'] == 1]
    neg_features = features[df['is_continuous'] == 0]

    print(f"\n--- Feature Separability (|mean_pos - mean_neg| / pooled_std) ---")
    separability = []
    for i in range(64):
        mean_diff = abs(pos_features[:, i].mean() - neg_features[:, i].mean())
        pooled_std = np.sqrt((pos_features[:, i].std()**2 + neg_features[:, i].std()**2) / 2)
        if pooled_std > 1e-8:
            sep = mean_diff / pooled_std
        else:
            sep = 0
        separability.append((i, sep))

    separability.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 most separable features:")
    for idx, sep in separability[:10]:
        print(f"  f{idx}: {sep:.4f}")

    print("\nBottom 10 least separable features:")
    for idx, sep in separability[-10:]:
        print(f"  f{idx}: {sep:.4f}")

    return df


def analyze_model_predictions(checkpoint_path: str, test_csv: str):
    """分析模型预测"""
    print(f"\n{'='*60}")
    print("Analyzing Model Predictions")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_cnn_model("cnn1d", input_dim=64, num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载数据
    norm_params = {
        'mean': np.array(checkpoint['norm_params']['mean']),
        'std': np.array(checkpoint['norm_params']['std'])
    }

    df = pd.read_csv(test_csv)
    feature_cols = [f"f{i}" for i in range(64)]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['is_continuous'].values

    # 标准化
    features = (features - norm_params['mean']) / norm_params['std']

    # 预测
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

    # 分析
    print(f"\nPrediction distribution:")
    print(f"  Predicted positive: {preds.sum()}")
    print(f"  Predicted negative: {(1-preds).sum()}")

    print(f"\nProbability distribution:")
    print(f"  Mean prob: {probs.mean():.4f}")
    print(f"  Std prob: {probs.std():.4f}")
    print(f"  Min prob: {probs.min():.4f}")
    print(f"  Max prob: {probs.max():.4f}")

    # 按文件类型分析
    if 'file_type' in df.columns:
        print(f"\n--- Accuracy by File Type ---")
        for ft in df['file_type'].unique():
            mask = df['file_type'] == ft
            ft_labels = labels[mask]
            ft_preds = preds[mask]
            ft_probs = probs[mask]

            acc = (ft_labels == ft_preds).mean()

            # 分别计算正负样本的准确率
            pos_mask = ft_labels == 1
            neg_mask = ft_labels == 0

            pos_acc = (ft_preds[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
            neg_acc = (ft_preds[neg_mask] == 0).mean() if neg_mask.sum() > 0 else 0

            print(f"  {ft}: acc={acc:.2%}, pos_recall={pos_acc:.2%}, neg_recall={neg_acc:.2%}, avg_prob={ft_probs.mean():.3f}")

    # 检查模型是否倾向于预测某一类
    print(f"\n--- Bias Analysis ---")
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    print(f"  Avg prob for TRUE positives: {pos_probs.mean():.4f}")
    print(f"  Avg prob for TRUE negatives: {neg_probs.mean():.4f}")

    # 如果两者接近，说明模型没有学到有效区分
    if abs(pos_probs.mean() - neg_probs.mean()) < 0.1:
        print("  WARNING: Model has poor discrimination!")

    return probs, preds, labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./experiment_results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # 分析数据集
    train_csv = results_dir / "dataset_train.csv"
    test_csv = results_dir / "dataset_test.csv"

    if train_csv.exists():
        analyze_dataset(str(train_csv))

    if test_csv.exists():
        analyze_dataset(str(test_csv))

    # 分析模型
    checkpoint = results_dir / "best_model.pt"
    if checkpoint.exists() and test_csv.exists():
        analyze_model_predictions(str(checkpoint), str(test_csv))

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    print("""
1. 如果正负样本特征可分离性低 → 特征设计问题，需要改进特征
2. 如果某个文件类型准确率特别低 → 该格式特征不适用
3. 如果模型偏向预测某一类 → 可能是类别不平衡或学习失败
4. 如果所有格式都差 → 可能需要重新考虑方法
    """)


if __name__ == "__main__":
    main()
