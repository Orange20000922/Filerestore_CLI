"""
Block Continuity Classifier Training Script

训练块连续性分类模型，用于判断相邻数据块是否属于同一文件。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import ContinuityClassifier, get_model
from dataset import load_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_CONFIG = {
    "input_dim": 64,
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "use_batch_norm": True,
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 15,
    "val_split": 0.2,
    "random_seed": 42,
    "use_class_weights": True,
}


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> tuple:
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in train_loader:
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # 详细指标
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 计算 TP, FP, FN (以 1=连续 为正类)
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    # 计算 Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, accuracy, precision, recall, f1


def train(
    csv_path: str,
    output_dir: str,
    config: dict = None
) -> dict:
    """
    训练连续性分类模型

    Args:
        csv_path: 训练数据 CSV 路径
        output_dir: 输出目录
        config: 训练配置

    Returns:
        训练结果字典
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # 加载数据
    logger.info("Loading dataset...")
    train_loader, val_loader, metadata = load_dataset(
        csv_path,
        val_split=config["val_split"],
        batch_size=config["batch_size"],
        random_seed=config["random_seed"]
    )

    logger.info(f"Train samples: {metadata['train_size']}")
    logger.info(f"Val samples: {metadata['val_size']}")
    logger.info(f"Sample type distribution: {metadata['sample_type_distribution']}")

    # 创建模型
    model = ContinuityClassifier(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"]
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    if config["use_class_weights"]:
        class_weights = metadata["class_weights"].to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    logger.info("Starting training...")

    for epoch in range(config["epochs"]):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc, precision, recall, f1 = validate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(f1)

        # 日志
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}"
        )

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": f1,
                "config": config
            }
            torch.save(checkpoint, output_path / "best_continuity.pt")
            logger.info(f"  Saved best model (F1: {f1:.4f})")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config["early_stopping_patience"]:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # 保存最终结果
    results = {
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "epochs_trained": epoch + 1,
        "config": config,
        "norm_params": {
            "mean": metadata["norm_params"]["mean"].tolist(),
            "std": metadata["norm_params"]["std"].tolist()
        },
        "history": history
    }

    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTraining completed!")
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Best Val F1: {best_val_f1:.4f}")

    return results


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    config: dict = None
):
    """
    导出模型为 ONNX 格式

    Args:
        checkpoint_path: PyTorch 检查点路径
        output_path: ONNX 输出路径
        config: 模型配置
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 使用保存的配置
    if "config" in checkpoint:
        config.update(checkpoint["config"])

    # 创建模型
    model = ContinuityClassifier(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, config["input_dim"])

    # 导出 ONNX
    output_file = Path(output_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_file),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    logger.info(f"Exported ONNX model to {output_file}")

    # 保存元数据
    metadata_path = output_file.with_suffix("").with_name(
        output_file.stem + "_metadata.json"
    )

    # 尝试加载训练结果中的标准化参数
    training_results_path = Path(checkpoint_path).parent / "training_results.json"
    norm_params = None
    if training_results_path.exists():
        with open(training_results_path) as f:
            training_results = json.load(f)
            norm_params = training_results.get("norm_params")

    metadata = {
        "input_dim": config["input_dim"],
        "num_classes": 2,
        "label_map": {"0": "discontinuous", "1": "continuous"},
        "norm_params": norm_params
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Block Continuity Classifier"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train 命令
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--csv", required=True, help="Path to training CSV file"
    )
    train_parser.add_argument(
        "--output", "-o", default="models/continuity",
        help="Output directory for checkpoints"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    train_parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate"
    )

    # export 命令
    export_parser = subparsers.add_parser("export", help="Export to ONNX")
    export_parser.add_argument(
        "checkpoint", help="Path to PyTorch checkpoint"
    )
    export_parser.add_argument(
        "--output", "-o", default="continuity_classifier.onnx",
        help="Output ONNX file path"
    )

    args = parser.parse_args()

    if args.command == "train":
        config = DEFAULT_CONFIG.copy()
        config["epochs"] = args.epochs
        config["batch_size"] = args.batch_size
        config["learning_rate"] = args.lr
        config["dropout"] = args.dropout

        train(args.csv, args.output, config)

    elif args.command == "export":
        export_onnx(args.checkpoint, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
