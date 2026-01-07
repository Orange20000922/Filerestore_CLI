"""
训练脚本 - 文件类型分类器训练与可视化
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, MODEL_CONFIG, TRAIN_CONFIG,
    CHECKPOINTS_DIR, EXPORTED_DIR, DATA_DIR
)
from model import create_model, count_parameters, model_summary
from dataset import prepare_dataset, FileFeatureExtractor, create_data_loaders
from cpp_dataset_loader import load_cpp_dataset


class TrainingHistory:
    """训练历史记录"""

    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0

    def update(self, train_loss: float, val_loss: float,
               train_acc: float, val_acc: float, lr: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.val_acc)

    def to_dict(self) -> Dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
        }


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    val_loader: DataLoader,
    label_map: Dict[int, str],
    device: str,
) -> Dict[str, Dict[str, float]]:
    """按类别评估模型"""
    model.eval()

    num_classes = len(label_map)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        _, predicted = outputs.max(1)

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1

    results = {}
    for idx, name in label_map.items():
        if class_total[idx] > 0:
            acc = class_correct[idx] / class_total[idx]
            results[name] = {
                "accuracy": acc,
                "correct": class_correct[idx],
                "total": class_total[idx],
            }

    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    history: TrainingHistory,
    label_map: Dict[int, str],
    norm_params: Dict,
    filepath: Path,
):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history.to_dict(),
        "label_map": label_map,
        "norm_params": {
            "mean": norm_params["mean"].tolist(),
            "std": norm_params["std"].tolist(),
        },
        "model_config": MODEL_CONFIG,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def plot_training_history(history: TrainingHistory, save_path: Optional[Path] = None):
    """绘制训练历史曲线"""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return
    except OSError:
        import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(history.train_loss, label='Train Loss', color='blue', alpha=0.8)
    ax1.plot(history.val_loss, label='Val Loss', color='red', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot([acc * 100 for acc in history.train_acc], label='Train Acc', color='blue', alpha=0.8)
    ax2.plot([acc * 100 for acc in history.val_acc], label='Val Acc', color='red', alpha=0.8)
    ax2.axhline(y=history.best_val_acc * 100, color='green', linestyle='--',
                label=f'Best Val Acc: {history.best_val_acc*100:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 学习率曲线
    ax3 = axes[1, 0]
    ax3.plot(history.learning_rates, color='purple', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 过拟合差距
    ax4 = axes[1, 1]
    gap = [t - v for t, v in zip(history.train_acc, history.val_acc)]
    ax4.plot([g * 100 for g in gap], color='orange', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train - Val Accuracy (%)')
    ax4.set_title('Overfitting Gap (Train - Val)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")

    plt.show()


def plot_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    label_map: Dict[int, str],
    device: str,
    save_path: Optional[Path] = None,
):
    """绘制混淆矩阵"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        print("matplotlib/sklearn/seaborn 未安装，跳过混淆矩阵绘制")
        return

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [label_map[i] for i in range(len(label_map))]

    # 绘制
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")

    plt.show()


def export_to_onnx(
    model: nn.Module,
    input_dim: int,
    output_path: Path,
    norm_params: Dict,
    label_map: Dict[int, str],
):
    """导出模型为 ONNX 格式"""
    model.eval()
    model.to("cpu")

    # 创建示例输入
    dummy_input = torch.randn(1, input_dim)

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX 模型已导出: {output_path}")

    # 保存元数据
    metadata = {
        "input_dim": input_dim,
        "num_classes": len(label_map),
        "label_map": label_map,
        "norm_params": {
            "mean": norm_params["mean"].tolist() if isinstance(norm_params["mean"], np.ndarray) else norm_params["mean"],
            "std": norm_params["std"].tolist() if isinstance(norm_params["std"], np.ndarray) else norm_params["std"],
        },
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"元数据已保存: {metadata_path}")

    return output_path


def train(
    model_type: str = "deep",
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    resume_from: Optional[Path] = None,
    export_onnx: bool = True,
    local_dirs: List[str] = None,
    cpp_dataset_path: Optional[str] = None,
):
    """
    主训练函数

    Args:
        model_type: 模型类型 ("simple", "deep", "resnet")
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        resume_from: 从检查点恢复
        export_onnx: 训练后是否导出 ONNX
        local_dirs: 本地文件目录列表（不使用 Govdocs）
        cpp_dataset_path: C++ mlscan 生成的数据集文件路径
    """
    # 使用默认配置
    epochs = epochs or TRAIN_CONFIG["epochs"]
    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    learning_rate = learning_rate or TRAIN_CONFIG["learning_rate"]

    print("=" * 60)
    print("文件类型分类器训练")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"模型类型: {model_type}")
    print(f"训练轮数: {epochs}")
    print(f"批大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    if cpp_dataset_path:
        print(f"C++ 数据集: {cpp_dataset_path}")
    print("=" * 60)

    # 准备数据集
    print("\n[1/4] 准备数据集...")

    if cpp_dataset_path:
        # 使用 C++ 生成的数据集
        print("使用 C++ mlscan 生成的数据集...")
        features, labels, label_map = load_cpp_dataset(cpp_dataset_path)

        # 打印数据集统计
        print(f"\n数据集统计:")
        print(f"  样本数: {len(labels)}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  类别数: {len(label_map)}")

        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n各类别样本数:")
        for idx, count in zip(unique, counts):
            print(f"  {label_map[idx]}: {count}")

        # 创建数据加载器
        train_loader, val_loader, norm_params = create_data_loaders(
            features, labels, batch_size,
            TRAIN_CONFIG["val_split"], TRAIN_CONFIG["random_state"]
        )
    else:
        # 使用原有数据集准备流程
        use_local = local_dirs is not None and len(local_dirs) > 0
        train_loader, val_loader, norm_params, label_map = prepare_dataset(
            use_local=use_local,
            local_dirs=[Path(d) for d in local_dirs] if local_dirs else None
        )

    num_classes = len(label_map)
    input_dim = MODEL_CONFIG["input_dim"]

    # 创建模型
    print("\n[2/4] 创建模型...")
    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=MODEL_CONFIG["hidden_dims"],
        dropout=MODEL_CONFIG["dropout"],
        use_batch_norm=MODEL_CONFIG["use_batch_norm"],
    )
    model = model.to(DEVICE)

    print(model_summary(model))
    print(f"参数总数: {count_parameters(model):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # 学习率调度器
    if TRAIN_CONFIG["lr_scheduler"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=TRAIN_CONFIG["lr_factor"],
            patience=TRAIN_CONFIG["lr_patience"],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练历史和早停
    history = TrainingHistory()
    early_stopping = EarlyStopping(patience=TRAIN_CONFIG["early_stopping_patience"])

    # 恢复训练
    start_epoch = 0
    if resume_from and resume_from.exists():
        print(f"\n从检查点恢复: {resume_from}")
        checkpoint = load_checkpoint(resume_from, model, optimizer)
        start_epoch = checkpoint["epoch"]
        history_dict = checkpoint["history"]
        history.train_loss = history_dict["train_loss"]
        history.val_loss = history_dict["val_loss"]
        history.train_acc = history_dict["train_acc"]
        history.val_acc = history_dict["val_acc"]
        history.best_val_acc = history_dict["best_val_acc"]
        history.best_epoch = history_dict["best_epoch"]

    # 训练循环
    print("\n[3/4] 开始训练...")
    best_model_path = CHECKPOINTS_DIR / f"best_{model_type}.pt"

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if TRAIN_CONFIG["lr_scheduler"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 记录历史
        history.update(train_loss, val_loss, train_acc, val_acc, current_lr)

        epoch_time = time.time() - epoch_start

        # 打印进度
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.2e}")

        # 保存最佳模型
        if val_acc > history.best_val_acc - 0.001:  # 允许小误差
            save_checkpoint(
                model, optimizer, epoch + 1, history,
                label_map, norm_params, best_model_path
            )
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc*100:.2f}%)")

        # 早停检查
        if early_stopping(val_loss):
            print(f"\n早停触发！最佳 epoch: {history.best_epoch}")
            break

    # 加载最佳模型
    print("\n[4/4] 评估最佳模型...")
    load_checkpoint(best_model_path, model)

    # 最终评估
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    print(f"\n最终验证结果: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")

    # 按类别评估
    class_results = evaluate_per_class(model, val_loader, label_map, DEVICE)
    print("\n各类别准确率:")
    for name, stats in sorted(class_results.items()):
        print(f"  {name}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")

    # 可视化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = CHECKPOINTS_DIR.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_training_history(history, plots_dir / f"training_history_{timestamp}.png")
    plot_confusion_matrix(model, val_loader, label_map, DEVICE, plots_dir / f"confusion_matrix_{timestamp}.png")

    # 导出 ONNX
    if export_onnx:
        onnx_path = EXPORTED_DIR / f"file_classifier_{model_type}.onnx"
        export_to_onnx(model, input_dim, onnx_path, norm_params, label_map)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证准确率: {history.best_val_acc*100:.2f}% (Epoch {history.best_epoch})")
    print(f"模型保存位置: {best_model_path}")
    if export_onnx:
        print(f"ONNX 导出位置: {EXPORTED_DIR / f'file_classifier_{model_type}.onnx'}")
    print("=" * 60)

    return model, history, label_map, norm_params


def main():
    parser = argparse.ArgumentParser(description="文件类型分类器训练")
    parser.add_argument("--model", type=str, default="deep",
                        choices=["simple", "deep", "resnet"],
                        help="模型类型")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--no-onnx", action="store_true",
                        help="不导出 ONNX")
    parser.add_argument("--local", type=str, nargs="+", default=None,
                        help="使用本地文件目录（可指定多个）")
    parser.add_argument("--csv", type=str, default=None,
                        help="使用 C++ mlscan 生成的 CSV 数据集")
    parser.add_argument("--binary", type=str, default=None,
                        help="使用 C++ mlscan 生成的二进制数据集")

    args = parser.parse_args()

    resume_path = Path(args.resume) if args.resume else None

    # 确定 C++ 数据集路径
    cpp_dataset = args.csv or args.binary

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_from=resume_path,
        export_onnx=not args.no_onnx,
        local_dirs=args.local,
        cpp_dataset_path=cpp_dataset,
    )


if __name__ == "__main__":
    main()
