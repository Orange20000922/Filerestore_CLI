"""
图像头部修复模型训练脚本

支持从 C++ mlscan --repair 生成的 CSV 数据集训练模型

CSV 格式:
  - f0-f30: 31维特征向量
  - image_type: 图像类型 (jpeg/png)
  - damage_type: 损坏类型 (0-5)
  - damage_severity: 损坏严重程度 (0.0-1.0)
  - damage_offset: 损坏偏移
  - damage_size: 损坏大小
  - is_repairable: 是否可修复 (0/1)

训练两个模型:
1. 图像类型分类器 (jpeg vs png)
2. 可修复性预测器 (是否可修复)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort

# 特征维度 (与 C++ ImageFeatureVector::FEATURE_DIM 一致)
FEATURE_DIM = 31


class RepairCSVDataset(Dataset):
    """从 C++ 生成的 CSV 加载修复训练数据集"""

    def __init__(self, csv_path, task='type_classification'):
        """
        Args:
            csv_path: CSV 文件路径
            task: 'type_classification' 或 'repairability'
        """
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.task = task

        # 提取特征列 (f0 - f30)
        feature_cols = [f'f{i}' for i in range(FEATURE_DIM)]
        self.features = self.df[feature_cols].values.astype(np.float32)

        # 标准化特征
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # 提取标签
        if task == 'type_classification':
            # image_type: jpeg=0, png=1
            self.labels = (self.df['image_type'] == 'png').astype(np.int64).values
        else:
            # is_repairable: 0 或 1
            self.labels = self.df['is_repairable'].astype(np.int64).values

        # 额外元数据
        self.damage_types = self.df['damage_type'].values
        self.damage_severity = self.df['damage_severity'].values

        print(f"  Loaded {len(self.features)} samples")
        print(f"  Feature shape: {self.features.shape}")

        if task == 'type_classification':
            jpeg_count = (self.labels == 0).sum()
            png_count = (self.labels == 1).sum()
            print(f"  Classes: JPEG={jpeg_count}, PNG={png_count}")
        else:
            repairable = (self.labels == 1).sum()
            not_repairable = (self.labels == 0).sum()
            print(f"  Classes: Repairable={repairable}, Not Repairable={not_repairable}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return features, label

    def get_scaler_params(self):
        """返回标准化参数，用于推理时"""
        return {
            'mean': self.scaler.mean_.tolist(),
            'std': self.scaler.scale_.tolist()
        }


class ImageTypeClassifier(nn.Module):
    """图像类型分类器 (JPEG vs PNG)"""

    def __init__(self, input_size=FEATURE_DIM, hidden_sizes=[128, 64, 32], num_classes=2, dropout=0.3):
        super(ImageTypeClassifier, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class RepairabilityPredictor(nn.Module):
    """可修复性预测器"""

    def __init__(self, input_size=FEATURE_DIM, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(RepairabilityPredictor, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 2))  # repairable / not repairable

        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)


def train_model(model, train_loader, val_loader, epochs, device, model_name, save_path):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_acc = 0.0
    best_model_state = None

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, save_path)

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {save_path}")

    return model, best_val_acc


def export_to_onnx(model, output_path, input_size=FEATURE_DIM):
    """导出为 ONNX 格式"""
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to {output_path}")

    # 验证 ONNX 模型
    try:
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        print(f"  ONNX verification passed, output shape: {ort_outs[0].shape}")
    except Exception as e:
        print(f"  ONNX verification warning: {e}")


def save_metadata(output_dir, scaler_params, class_names, model_type):
    """保存模型元数据"""
    import json

    metadata = {
        'model_type': model_type,
        'feature_dim': FEATURE_DIM,
        'class_names': class_names,
        'normalization': scaler_params
    }

    metadata_path = os.path.join(output_dir, f'{model_type}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train image repair models from C++ generated CSV')
    parser.add_argument('csv_file', help='CSV file generated by mlscan --repair')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available)')
    parser.add_argument('--output', default=None, help='Output directory (default: ml/models/repair)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')

    args = parser.parse_args()

    # 设置输出目录
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, '..', 'models', 'repair')

    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*60}")
    print(f"Image Repair Model Training")
    print(f"{'='*60}")
    print(f"CSV file: {args.csv_file}")
    print(f"Output directory: {args.output}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Validation split: {args.val_split}")

    # =========================================================================
    # 训练图像类型分类器
    # =========================================================================
    print("\n" + "="*60)
    print("Task 1: Image Type Classification (JPEG vs PNG)")
    print("="*60)

    type_dataset = RepairCSVDataset(args.csv_file, task='type_classification')
    scaler_params = type_dataset.get_scaler_params()

    # 划分数据集
    train_size = int((1 - args.val_split) * len(type_dataset))
    val_size = len(type_dataset) - train_size
    train_dataset, val_dataset = random_split(type_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nDataset split: Train={train_size}, Val={val_size}")

    # 训练
    type_classifier = ImageTypeClassifier().to(args.device)
    type_classifier, type_acc = train_model(
        type_classifier, train_loader, val_loader,
        args.epochs, args.device,
        "Image Type Classifier",
        os.path.join(args.output, 'image_type_classifier.pth')
    )

    # 导出 ONNX
    export_to_onnx(type_classifier, os.path.join(args.output, 'image_type_classifier.onnx'))
    save_metadata(args.output, scaler_params, ['jpeg', 'png'], 'image_type_classifier')

    # =========================================================================
    # 训练可修复性预测器
    # =========================================================================
    print("\n" + "="*60)
    print("Task 2: Repairability Prediction")
    print("="*60)

    repair_dataset = RepairCSVDataset(args.csv_file, task='repairability')

    train_size = int((1 - args.val_split) * len(repair_dataset))
    val_size = len(repair_dataset) - train_size
    train_dataset, val_dataset = random_split(repair_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nDataset split: Train={train_size}, Val={val_size}")

    # 训练
    repair_predictor = RepairabilityPredictor().to(args.device)
    repair_predictor, repair_acc = train_model(
        repair_predictor, train_loader, val_loader,
        args.epochs, args.device,
        "Repairability Predictor",
        os.path.join(args.output, 'repairability_predictor.pth')
    )

    # 导出 ONNX
    export_to_onnx(repair_predictor, os.path.join(args.output, 'repairability_predictor.onnx'))
    save_metadata(args.output, scaler_params, ['not_repairable', 'repairable'], 'repairability_predictor')

    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved to: {args.output}")
    print(f"  - image_type_classifier.onnx (Accuracy: {type_acc:.2f}%)")
    print(f"  - repairability_predictor.onnx (Accuracy: {repair_acc:.2f}%)")
    print(f"\nTo use in C++:")
    print(f"  1. Copy *.onnx files to exe directory or models/repair/")
    print(f"  2. Copy *_metadata.json for normalization parameters")


if __name__ == '__main__':
    main()
