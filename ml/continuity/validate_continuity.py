#!/usr/bin/env python3
"""
Block Continuity Detection - Validation Experiment

本脚本用于验证 ML 模型是否能准确判断两个数据块是否连续。

实验流程：
1. 从原始文件生成测试数据集（已知ground truth）
2. 训练模型
3. 在测试集上评估分类准确率
4. 进阶：碎片重组实验

使用方法：
    # 完整流程（生成数据 -> 训练 -> 评估）
    python validate_continuity.py run --data-dir D:/temp/ml_training

    # 仅生成数据集
    python validate_continuity.py generate --data-dir D:/temp/ml_training --output dataset.csv

    # 仅训练
    python validate_continuity.py train --csv dataset.csv --output ./models/

    # 仅评估
    python validate_continuity.py evaluate --checkpoint ./models/best_cnn.pt --csv test.csv

    # 碎片重组实验
    python validate_continuity.py reassembly --checkpoint ./models/best_cnn.pt --test-file test.zip
"""

import argparse
import json
import os
import sys
import random
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor import ContinuityFeatureExtractor
from model_cnn import get_cnn_model, count_parameters
from dataset import ContinuityDataset, load_dataset


# =============================================================================
# 配置
# =============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据
    data_dir: str = "D:/temp/ml_training"
    output_dir: str = "./experiment_results"

    # 数据集参数
    block_size: int = 8192  # 8KB
    samples_per_file: int = 20  # 每个文件采样数
    min_file_size: int = 65536  # 最小文件大小 (64KB)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 自适应采样
    use_adaptive_sampling: bool = True
    adaptive_sampling_rate: float = 0.05  # 5% (was 1%)
    min_samples: int = 50   # 增加最小采样 (was 10)
    max_samples: int = 5000  # 大幅增加最大采样 (was 500)

    # 训练参数
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10

    # GPU 优化
    use_amp: bool = True  # 混合精度训练
    num_workers: int = 4  # 数据加载线程数（Windows 建议 0-4）
    pin_memory: bool = True  # 锁页内存加速传输

    # 模型
    model_type: str = "cnn1d"

    # 碎片重组实验
    reassembly_test_files: int = 10  # 测试文件数量
    reassembly_max_blocks: int = 50  # 每个文件最大块数


# =============================================================================
# 数据生成
# =============================================================================

class ValidationDatasetGenerator:
    """验证数据集生成器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.extractor = ContinuityFeatureExtractor(block_size=config.block_size)

    def calculate_samples_per_file(self, file_size: int) -> int:
        """计算单个文件的采样数（自适应）"""
        if not self.config.use_adaptive_sampling:
            return self.config.samples_per_file

        num_blocks = file_size // self.config.block_size
        available_pairs = max(1, num_blocks - 1)

        samples = int(available_pairs * self.config.adaptive_sampling_rate)
        samples = max(self.config.min_samples, samples)
        samples = min(self.config.max_samples, samples)
        samples = min(samples, available_pairs)

        return samples

    def collect_files(self, data_dir: str) -> Dict[str, List[Path]]:
        """收集文件并按类型分组"""
        files_by_type = {}
        data_path = Path(data_dir)

        # 支持的文件类型
        supported_types = {
            'zip': ['.zip', '.docx', '.xlsx', '.pptx'],
           # 'mp3': ['.mp3'],
            #'mp4': ['.mp4', '.mov', '.m4v'],
        }

        for file_type, extensions in supported_types.items():
            files_by_type[file_type] = []
            for ext in extensions:
                files_by_type[file_type].extend(
                    [f for f in data_path.rglob(f"*{ext}")
                     if f.stat().st_size >= self.config.min_file_size]
                )

        return files_by_type

    def generate_samples_from_file(
        self,
        file_path: Path,
        file_type: str
    ) -> List[Tuple[np.ndarray, int, str]]:
        """从单个文件生成样本"""
        samples = []

        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        num_blocks = len(data) // self.config.block_size
        if num_blocks < 4:
            return []

        samples_to_generate = self.calculate_samples_per_file(len(data))

        # 生成正样本（连续块）
        block_indices = list(range(num_blocks - 1))
        random.shuffle(block_indices)

        for idx in block_indices[:samples_to_generate]:
            offset1 = idx * self.config.block_size
            offset2 = (idx + 1) * self.config.block_size

            block1 = data[offset1:offset1 + self.config.block_size]
            block2 = data[offset2:offset2 + self.config.block_size]

            features = self.extractor.extract(block1, block2, file_type)
            samples.append((features, 1, 'same_file'))  # 正样本

        # 生成负样本（非连续块，来自同一文件）
        neg_samples_count = samples_to_generate // 2
        for _ in range(neg_samples_count):
            idx1 = random.randint(0, num_blocks - 1)
            # 确保 idx2 和 idx1 不相邻
            while True:
                idx2 = random.randint(0, num_blocks - 1)
                if abs(idx2 - idx1) > 1:
                    break

            offset1 = idx1 * self.config.block_size
            offset2 = idx2 * self.config.block_size

            block1 = data[offset1:offset1 + self.config.block_size]
            block2 = data[offset2:offset2 + self.config.block_size]

            features = self.extractor.extract(block1, block2, file_type)
            samples.append((features, 0, 'same_file_non_adjacent'))  # 负样本

        return samples

    def generate_cross_file_negatives(
        self,
        file_data_list: List[Tuple[Path, bytes, str]],
        count: int
    ) -> List[Tuple[np.ndarray, int, str]]:
        """生成跨文件负样本"""
        samples = []

        if len(file_data_list) < 2:
            return []

        for _ in range(count):
            # 随机选择两个不同的文件
            idx1, idx2 = random.sample(range(len(file_data_list)), 2)
            path1, data1, type1 = file_data_list[idx1]
            path2, data2, type2 = file_data_list[idx2]

            num_blocks1 = len(data1) // self.config.block_size
            num_blocks2 = len(data2) // self.config.block_size

            if num_blocks1 < 1 or num_blocks2 < 1:
                continue

            # 随机选择块
            b1_idx = random.randint(0, num_blocks1 - 1)
            b2_idx = random.randint(0, num_blocks2 - 1)

            offset1 = b1_idx * self.config.block_size
            offset2 = b2_idx * self.config.block_size

            block1 = data1[offset1:offset1 + self.config.block_size]
            block2 = data2[offset2:offset2 + self.config.block_size]

            features = self.extractor.extract(block1, block2, type1)
            samples.append((features, 0, 'different_files'))

        return samples

    def generate_dataset(self, output_csv: str) -> Dict:
        """生成完整数据集"""
        print(f"\n{'='*60}")
        print("Generating Validation Dataset")
        print(f"{'='*60}")
        print(f"Data directory: {self.config.data_dir}")
        print(f"Block size: {self.config.block_size} bytes")
        print(f"Adaptive sampling: {self.config.use_adaptive_sampling}")

        # 收集文件
        files_by_type = self.collect_files(self.config.data_dir)

        total_files = sum(len(files) for files in files_by_type.values())
        print(f"\nFound files:")
        for file_type, files in files_by_type.items():
            print(f"  {file_type}: {len(files)} files")
        print(f"  Total: {total_files} files")

        if total_files == 0:
            print("Error: No files found!")
            return {}

        # 生成样本
        all_samples = []
        file_data_cache = []

        for file_type, files in files_by_type.items():
            print(f"\nProcessing {file_type} files...")

            for file_path in tqdm(files, desc=f"  {file_type}"):
                samples = self.generate_samples_from_file(file_path, file_type)
                all_samples.extend(samples)

                # 缓存文件数据用于跨文件负样本
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    file_data_cache.append((file_path, data, file_type))
                    if len(file_data_cache) > 200:
                        file_data_cache.pop(0)
                except:
                    pass

        # 生成跨文件负样本
        positive_count = sum(1 for s in all_samples if s[1] == 1)
        current_negative = sum(1 for s in all_samples if s[1] == 0)
        cross_file_needed = max(0, positive_count - current_negative)

        if cross_file_needed > 0:
            print(f"\nGenerating {cross_file_needed} cross-file negative samples...")
            cross_samples = self.generate_cross_file_negatives(
                file_data_cache, cross_file_needed
            )
            all_samples.extend(cross_samples)

        # 打乱并分割数据集
        random.shuffle(all_samples)

        n = len(all_samples)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        train_samples = all_samples[:train_end]
        val_samples = all_samples[train_end:val_end]
        test_samples = all_samples[val_end:]

        # 保存数据集
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存训练集
        train_csv = output_path.with_name(output_path.stem + "_train.csv")
        self._save_csv(train_samples, train_csv)

        # 保存验证集
        val_csv = output_path.with_name(output_path.stem + "_val.csv")
        self._save_csv(val_samples, val_csv)

        # 保存测试集
        test_csv = output_path.with_name(output_path.stem + "_test.csv")
        self._save_csv(test_samples, test_csv)

        # 统计
        stats = {
            'total_samples': len(all_samples),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'positive_samples': sum(1 for s in all_samples if s[1] == 1),
            'negative_samples': sum(1 for s in all_samples if s[1] == 0),
            'files_by_type': {k: len(v) for k, v in files_by_type.items()},
            'train_csv': str(train_csv),
            'val_csv': str(val_csv),
            'test_csv': str(test_csv),
        }

        print(f"\n{'='*60}")
        print("Dataset Generation Complete")
        print(f"{'='*60}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"  Positive (continuous): {stats['positive_samples']}")
        print(f"  Negative (discontinuous): {stats['negative_samples']}")
        print(f"Split:")
        print(f"  Train: {stats['train_samples']} samples -> {train_csv}")
        print(f"  Val:   {stats['val_samples']} samples -> {val_csv}")
        print(f"  Test:  {stats['test_samples']} samples -> {test_csv}")

        # 保存统计信息
        stats_path = output_path.with_name(output_path.stem + "_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def _save_csv(self, samples: List[Tuple[np.ndarray, int, str]], path: Path):
        """保存样本到CSV"""
        import csv

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = [f"f{i}" for i in range(64)] + ['is_continuous', 'sample_type']
            writer.writerow(header)

            # Data
            for features, label, sample_type in samples:
                row = list(features) + [label, sample_type]
                writer.writerow(row)


# =============================================================================
# 评估
# =============================================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict:
    """评估模型性能"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 评估时也可用 AMP 加速（可选，影响较小）
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算 AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total': len(all_labels)
    }


# =============================================================================
# 碎片重组实验
# =============================================================================

class FragmentReassemblyExperiment:
    """碎片重组实验"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: ExperimentConfig
    ):
        self.model = model
        self.device = device
        self.config = config
        self.extractor = ContinuityFeatureExtractor(config.block_size)

    def run_single_file(self, file_path: Path) -> Dict:
        """对单个文件进行碎片重组实验"""
        with open(file_path, 'rb') as f:
            data = f.read()

        file_type = file_path.suffix.lower().lstrip('.')
        block_size = self.config.block_size
        num_blocks = len(data) // block_size

        # 限制块数
        if num_blocks > self.config.reassembly_max_blocks:
            num_blocks = self.config.reassembly_max_blocks

        if num_blocks < 4:
            return {'error': 'File too small'}

        # 提取所有块
        blocks = []
        for i in range(num_blocks):
            offset = i * block_size
            blocks.append(data[offset:offset + block_size])

        # 原始顺序
        original_order = list(range(num_blocks))

        # 打乱顺序
        shuffled_order = original_order.copy()
        random.shuffle(shuffled_order)
        shuffled_blocks = [blocks[i] for i in shuffled_order]

        # 使用模型预测所有块对的连续性分数
        self.model.eval()
        scores = np.zeros((num_blocks, num_blocks))

        with torch.no_grad():
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if i == j:
                        scores[i, j] = -1  # 自己不能接自己
                        continue

                    features = self.extractor.extract(
                        shuffled_blocks[i],
                        shuffled_blocks[j],
                        file_type
                    )

                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    features_tensor = features_tensor.to(self.device)

                    outputs = self.model(features_tensor)
                    prob = torch.softmax(outputs, dim=1)[0, 1].item()
                    scores[i, j] = prob

        # 贪心重组算法
        reconstructed_order = self._greedy_reconstruct(scores, num_blocks)

        # 将打乱顺序映射回原始索引
        reconstructed_original = [shuffled_order[i] for i in reconstructed_order]

        # 评估重组结果
        correct_pairs = 0
        total_pairs = num_blocks - 1

        for i in range(total_pairs):
            if reconstructed_original[i] + 1 == reconstructed_original[i + 1]:
                correct_pairs += 1

        # 检查是否完全正确
        is_perfect = reconstructed_original == original_order

        # 计算 MD5
        original_md5 = hashlib.md5(data[:num_blocks * block_size]).hexdigest()
        reconstructed_data = b''.join(blocks[i] for i in reconstructed_original)
        reconstructed_md5 = hashlib.md5(reconstructed_data).hexdigest()

        return {
            'file': str(file_path.name),
            'num_blocks': num_blocks,
            'correct_pairs': correct_pairs,
            'total_pairs': total_pairs,
            'pair_accuracy': correct_pairs / total_pairs if total_pairs > 0 else 0,
            'is_perfect': is_perfect,
            'md5_match': original_md5 == reconstructed_md5,
            'original_order': original_order,
            'shuffled_order': shuffled_order,
            'reconstructed_order': reconstructed_original,
        }

    def _greedy_reconstruct(self, scores: np.ndarray, n: int) -> List[int]:
        """贪心算法重组块顺序"""
        # 找到最可能的起始块（作为第二块得分最低的块可能是第一块）
        first_block_scores = scores.sum(axis=0)  # 每个块作为"后续块"的总得分
        first_block = np.argmin(first_block_scores)

        used = {first_block}
        order = [first_block]

        current = first_block
        while len(order) < n:
            # 找到当前块最可能的下一个块
            best_next = -1
            best_score = -1

            for j in range(n):
                if j not in used and scores[current, j] > best_score:
                    best_score = scores[current, j]
                    best_next = j

            if best_next == -1:
                # 没有找到，选择任意未使用的块
                for j in range(n):
                    if j not in used:
                        best_next = j
                        break

            order.append(best_next)
            used.add(best_next)
            current = best_next

        return order

    def run_experiment(self, test_files: List[Path]) -> Dict:
        """运行完整的碎片重组实验"""
        results = []

        print(f"\n{'='*60}")
        print("Fragment Reassembly Experiment")
        print(f"{'='*60}")
        print(f"Test files: {len(test_files)}")
        print(f"Max blocks per file: {self.config.reassembly_max_blocks}")

        for file_path in tqdm(test_files, desc="Reassembling"):
            result = self.run_single_file(file_path)
            results.append(result)

        # 汇总统计
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            print("No valid results!")
            return {}

        avg_pair_accuracy = np.mean([r['pair_accuracy'] for r in valid_results])
        perfect_count = sum(1 for r in valid_results if r['is_perfect'])
        md5_match_count = sum(1 for r in valid_results if r['md5_match'])

        summary = {
            'total_files': len(test_files),
            'valid_files': len(valid_results),
            'avg_pair_accuracy': avg_pair_accuracy,
            'perfect_reconstructions': perfect_count,
            'perfect_rate': perfect_count / len(valid_results) if valid_results else 0,
            'md5_matches': md5_match_count,
            'md5_match_rate': md5_match_count / len(valid_results) if valid_results else 0,
            'individual_results': results
        }

        print(f"\n{'='*60}")
        print("Reassembly Results")
        print(f"{'='*60}")
        print(f"Valid files tested: {len(valid_results)}")
        print(f"Average pair accuracy: {avg_pair_accuracy:.2%}")
        print(f"Perfect reconstructions: {perfect_count}/{len(valid_results)} ({summary['perfect_rate']:.2%})")
        print(f"MD5 matches: {md5_match_count}/{len(valid_results)} ({summary['md5_match_rate']:.2%})")

        return summary


# =============================================================================
# 主流程
# =============================================================================

def run_full_experiment(config: ExperimentConfig) -> Dict:
    """运行完整实验流程"""
    start_time = time.time()

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Block Continuity Detection - Validation Experiment")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Data directory: {config.data_dir}")

    # Step 1: 生成数据集
    print("\n" + "-"*60)
    print("Step 1: Generating Dataset")
    print("-"*60)

    generator = ValidationDatasetGenerator(config)
    dataset_csv = output_path / "dataset.csv"
    stats = generator.generate_dataset(str(dataset_csv))

    if not stats:
        print("Dataset generation failed!")
        return {}

    # Step 2: 训练模型
    print("\n" + "-"*60)
    print("Step 2: Training Model")
    print("-"*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # 启用 cudnn 优化
        torch.backends.cudnn.benchmark = True
        print(f"  cuDNN benchmark: enabled")

    # 加载数据
    train_csv = stats['train_csv']
    val_csv = stats['val_csv']
    test_csv = stats['test_csv']

    train_dataset = ContinuityDataset(train_csv)
    val_dataset = ContinuityDataset(val_csv, norm_params=train_dataset.get_norm_params())
    test_dataset = ContinuityDataset(test_csv, norm_params=train_dataset.get_norm_params())

    # DataLoader 优化参数
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers if device.type == "cuda" else 0,
        'pin_memory': config.pin_memory if device.type == "cuda" else False,
        'persistent_workers': config.num_workers > 0 and device.type == "cuda",
    }
    print(f"  DataLoader workers: {loader_kwargs['num_workers']}")
    print(f"  Pin memory: {loader_kwargs['pin_memory']}")

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # 创建模型
    model = get_cnn_model(config.model_type, input_dim=64, num_classes=2)
    model = model.to(device)
    print(f"Model: {config.model_type}, Parameters: {count_parameters(model):,}")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )

    # 混合精度训练 (AMP)
    use_amp = config.use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    print(f"  Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # 训练循环
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training for {config.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            if use_amp:
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += features.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        val_acc = val_metrics['accuracy']
        val_loss = 0  # 简化，不计算验证损失

        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'norm_params': {
                    'mean': train_dataset.get_norm_params()['mean'].tolist(),
                    'std': train_dataset.get_norm_params()['std'].tolist()
                }
            }, output_path / "best_model.pt")
            print(f"  -> Saved best model (acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break

    # Step 3: 测试集评估
    print("\n" + "-"*60)
    print("Step 3: Evaluating on Test Set")
    print("-"*60)

    # 加载最佳模型
    checkpoint = torch.load(output_path / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(model, test_loader, device)

    print(f"\nTest Set Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {test_metrics['tp']:5d}  FP: {test_metrics['fp']:5d}")
    print(f"    FN: {test_metrics['fn']:5d}  TN: {test_metrics['tn']:5d}")

    # Step 4: 碎片重组实验（可选）
    print("\n" + "-"*60)
    print("Step 4: Fragment Reassembly Experiment")
    print("-"*60)

    # 收集测试文件
    data_path = Path(config.data_dir)
    test_files = list(data_path.rglob("*.zip"))[:config.reassembly_test_files]
    test_files += list(data_path.rglob("*.mp3"))[:config.reassembly_test_files]
    test_files += list(data_path.rglob("*.mp4"))[:config.reassembly_test_files]

    # 过滤太小的文件
    test_files = [f for f in test_files if f.stat().st_size >= config.min_file_size]

    reassembly_results = {}
    if test_files:
        reassembly_exp = FragmentReassemblyExperiment(model, device, config)
        reassembly_results = reassembly_exp.run_experiment(test_files[:config.reassembly_test_files])
    else:
        print("No suitable files for reassembly experiment")

    # 汇总结果
    total_time = time.time() - start_time

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data_dir': config.data_dir,
            'block_size': config.block_size,
            'model_type': config.model_type,
            'epochs': config.epochs,
        },
        'dataset_stats': stats,
        'training': {
            'best_val_acc': best_val_acc,
            'total_epochs': len(history['train_acc']),
        },
        'test_metrics': test_metrics,
        'reassembly_results': {
            k: v for k, v in reassembly_results.items()
            if k != 'individual_results'
        } if reassembly_results else {},
        'total_time_seconds': total_time
    }

    # 保存结果
    results_path = output_path / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_path}")

    # 总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Classification Task:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"  Test F1 Score: {test_metrics['f1']:.2%}")

    if reassembly_results:
        print(f"\nReassembly Task:")
        print(f"  Avg Pair Accuracy: {reassembly_results.get('avg_pair_accuracy', 0):.2%}")
        print(f"  Perfect Rate: {reassembly_results.get('perfect_rate', 0):.2%}")

    # 结论
    print("\n" + "-"*60)
    print("CONCLUSION")
    print("-"*60)

    if test_metrics['f1'] >= 0.85:
        print("Result: GOOD - Model shows strong ability to detect block continuity.")
        print("The ML approach is VIABLE for file fragment detection.")
    elif test_metrics['f1'] >= 0.70:
        print("Result: MODERATE - Model shows reasonable performance.")
        print("Consider improvements: more data, better features, or different architecture.")
    else:
        print("Result: POOR - Model struggles with this task.")
        print("May need to reconsider the approach or feature design.")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Block Continuity Detection - Validation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整流程
  python validate_continuity.py run --data-dir D:/temp/ml_training

  # 仅生成数据集
  python validate_continuity.py generate --data-dir D:/temp/ml_training --output dataset.csv

  # 使用现有数据训练
  python validate_continuity.py train --train-csv train.csv --val-csv val.csv --output ./models/

  # 评估模型
  python validate_continuity.py evaluate --checkpoint best_model.pt --test-csv test.csv
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # run 命令
    run_parser = subparsers.add_parser("run", help="Run full experiment")
    run_parser.add_argument("--data-dir", required=True, help="Data directory")
    run_parser.add_argument("--output", "-o", default="./experiment_results", help="Output directory")
    run_parser.add_argument("--epochs", type=int, default=50)
    run_parser.add_argument("--batch-size", type=int, default=256)
    run_parser.add_argument("--model", default="cnn1d", choices=["cnn1d", "cnn_residual", "cnn_light"])
    run_parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    run_parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (0-4 for Windows)")
    run_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # generate 命令
    gen_parser = subparsers.add_parser("generate", help="Generate dataset only")
    gen_parser.add_argument("--data-dir", required=True, help="Data directory")
    gen_parser.add_argument("--output", "-o", required=True, help="Output CSV path")

    # train 命令
    train_parser = subparsers.add_parser("train", help="Train model only")
    train_parser.add_argument("--train-csv", required=True)
    train_parser.add_argument("--val-csv", required=True)
    train_parser.add_argument("--output", "-o", required=True)
    train_parser.add_argument("--epochs", type=int, default=50)

    # evaluate 命令
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--test-csv", required=True)

    # reassembly 命令
    reassembly_parser = subparsers.add_parser("reassembly", help="Run reassembly experiment")
    reassembly_parser.add_argument("--checkpoint", required=True)
    reassembly_parser.add_argument("--data-dir", required=True)
    reassembly_parser.add_argument("--num-files", type=int, default=10)

    args = parser.parse_args()

    if args.command == "run":
        config = ExperimentConfig(
            data_dir=args.data_dir,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_type=args.model,
            use_amp=not args.no_amp,
            num_workers=args.workers,
            learning_rate=args.lr,
        )
        run_full_experiment(config)

    elif args.command == "generate":
        config = ExperimentConfig(data_dir=args.data_dir)
        generator = ValidationDatasetGenerator(config)
        generator.generate_dataset(args.output)

    elif args.command == "train":
        # 简化的训练流程
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = ContinuityDataset(args.train_csv)
        val_dataset = ContinuityDataset(args.val_csv, norm_params=train_dataset.get_norm_params())

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model = get_cnn_model("cnn1d", input_dim=64, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(features), labels)
                loss.backward()
                optimizer.step()

            val_metrics = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch}: Val Acc = {val_metrics['accuracy']:.4f}, F1 = {val_metrics['f1']:.4f}")

            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'norm_params': {
                        'mean': train_dataset.get_norm_params()['mean'].tolist(),
                        'std': train_dataset.get_norm_params()['std'].tolist()
                    }
                }, output_path / "best_model.pt")

    elif args.command == "evaluate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = get_cnn_model("cnn1d", input_dim=64, num_classes=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        norm_params = {
            'mean': np.array(checkpoint['norm_params']['mean']),
            'std': np.array(checkpoint['norm_params']['std'])
        }

        test_dataset = ContinuityDataset(args.test_csv, norm_params=norm_params)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        metrics = evaluate_model(model, test_loader, device)

        print(f"\nTest Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

    elif args.command == "reassembly":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = get_cnn_model("cnn1d", input_dim=64, num_classes=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        config = ExperimentConfig(
            data_dir=args.data_dir,
            reassembly_test_files=args.num_files
        )

        # 收集测试文件
        data_path = Path(args.data_dir)
        test_files = list(data_path.rglob("*.zip"))[:args.num_files]
        test_files += list(data_path.rglob("*.mp3"))[:args.num_files]
        test_files = [f for f in test_files if f.stat().st_size >= config.min_file_size]

        experiment = FragmentReassemblyExperiment(model, device, config)
        experiment.run_experiment(test_files)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
