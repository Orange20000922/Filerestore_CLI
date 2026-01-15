#!/usr/bin/env python3
"""
1D CNN Training Script for Block Continuity Detection

Optimized for Google Colab with:
- Automatic GPU detection
- Mixed precision training (AMP)
- Google Drive integration
- TensorBoard logging
- Early stopping
- ONNX export

Usage:
    # Train model
    python train_cnn.py train --csv dataset.csv --output ./models/

    # Export to ONNX
    python train_cnn.py export --checkpoint best_model.pt --output model.onnx

    # Evaluate model
    python train_cnn.py eval --checkpoint best_model.pt --csv test.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging

# Import from same package
from model_cnn import get_cnn_model, count_parameters
from dataset import load_dataset, ContinuityDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    "model_type": "cnn1d",
    "channels": [32, 64, 128],
    "kernel_sizes": [5, 3, 3],
    "dropout": 0.3,
    "use_batch_norm": True,

    # Training
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.2,

    # Scheduler
    "scheduler": "cosine",  # "cosine", "plateau", "step"
    "warmup_epochs": 5,

    # Early stopping
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 1e-4,

    # Mixed precision
    "use_amp": True,

    # Data
    "num_workers": 0,  # 0 for Colab compatibility
    "pin_memory": True,

    # Logging
    "log_interval": 10,
    "save_interval": 5,
}


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
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

        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += features.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[str, float]]:
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += features.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    # Compute additional metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return avg_loss, accuracy, metrics


def train(
    csv_path: str,
    output_dir: str,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Full training loop

    Args:
        csv_path: Path to CSV dataset
        output_dir: Output directory for checkpoints and logs
        config: Training configuration

    Returns:
        Training results dict
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading dataset from {csv_path}...")
    train_loader, val_loader, metadata = load_dataset(
        csv_path,
        val_split=config["val_split"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    print(f"  Train samples: {metadata['train_size']}")
    print(f"  Val samples: {metadata['val_size']}")
    print(f"  Class weights: {metadata['class_weights'].tolist()}")

    # Create model
    print(f"\nCreating model ({config['model_type']})...")
    model = get_cnn_model(
        model_type=config["model_type"],
        input_dim=metadata["feature_dim"],
        num_classes=metadata["num_classes"],
        channels=config["channels"],
        kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"]
    )
    model = model.to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # Loss function with class weights
    class_weights = metadata["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Scheduler
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=config["learning_rate"] * 0.01
        )
    elif config["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Mixed precision
    scaler = GradScaler() if config["use_amp"] and device.type == "cuda" else None

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output_path / "logs")
    except ImportError:
        writer = None
        print("TensorBoard not available")

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("=" * 60)

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, config["use_amp"]
        )

        # Validate
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Logging
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            writer.add_scalar("LR", lr, epoch)

        # Save best model
        if val_acc > best_val_acc + config["early_stopping_min_delta"]:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": config,
                "metadata": {
                    "feature_dim": metadata["feature_dim"],
                    "num_classes": metadata["num_classes"],
                    "norm_params": {
                        "mean": metadata["norm_params"]["mean"].tolist(),
                        "std": metadata["norm_params"]["std"].tolist()
                    }
                }
            }
            torch.save(checkpoint, output_path / "best_cnn.pt")
            print(f"  -> Saved best model (acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break

        # Periodic save
        if epoch % config["save_interval"] == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config
            }
            torch.save(checkpoint, output_path / f"checkpoint_epoch{epoch}.pt")

    total_time = time.time() - start_time

    if writer:
        writer.close()

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "config": config,
        "metadata": {
            "feature_dim": metadata["feature_dim"],
            "num_classes": metadata["num_classes"],
            "norm_params": {
                "mean": metadata["norm_params"]["mean"].tolist(),
                "std": metadata["norm_params"]["std"].tolist()
            }
        }
    }, output_path / "final_cnn.pt")

    # Save history
    with open(output_path / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Results
    results = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "total_time": total_time,
        "output_dir": str(output_path)
    }

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"  Best accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Output: {output_path}")

    return results


# =============================================================================
# ONNX Export
# =============================================================================

def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17
) -> bool:
    """
    Export trained model to ONNX format

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path: Output .onnx path
        opset_version: ONNX opset version

    Returns:
        True if successful
    """
    import onnx

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint.get("config", DEFAULT_CONFIG)
    metadata = checkpoint.get("metadata", {})

    # Rebuild model
    model = get_cnn_model(
        model_type=config.get("model_type", "cnn1d"),
        input_dim=metadata.get("feature_dim", 64),
        num_classes=metadata.get("num_classes", 2),
        channels=config.get("channels", [32, 64, 128]),
        kernel_sizes=config.get("kernel_sizes", [5, 3, 3]),
        dropout=config.get("dropout", 0.3),
        use_batch_norm=config.get("use_batch_norm", True)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Export
    print(f"Exporting to {output_path}...")
    dummy_input = torch.randn(1, metadata.get("feature_dim", 64))

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['logits'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Save metadata
    metadata_path = output_path.replace(".onnx", ".json")
    export_metadata = {
        "input_dim": metadata.get("feature_dim", 64),
        "num_classes": metadata.get("num_classes", 2),
        "model_type": config.get("model_type", "cnn1d"),
        "norm_params": metadata.get("norm_params", {}),
        "label_map": {"0": "discontinuous", "1": "continuous"},
        "opset_version": opset_version,
        "exported_at": datetime.now().isoformat()
    }

    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)

    file_size = os.path.getsize(output_path)
    print(f"\nExport successful!")
    print(f"  Model: {output_path} ({file_size/1024:.1f} KB)")
    print(f"  Metadata: {metadata_path}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(1, metadata.get("feature_dim", 64)).astype(np.float32)
        outputs = session.run(None, {"features": test_input})
        print(f"  ONNX Runtime test: OK (output shape: {outputs[0].shape})")
    except ImportError:
        print("  ONNX Runtime not installed, skipping verification")

    return True


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    checkpoint_path: str,
    csv_path: str
) -> Dict[str, float]:
    """Evaluate model on test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", DEFAULT_CONFIG)
    metadata = checkpoint.get("metadata", {})

    # Build model
    model = get_cnn_model(
        model_type=config.get("model_type", "cnn1d"),
        input_dim=metadata.get("feature_dim", 64),
        num_classes=metadata.get("num_classes", 2),
        channels=config.get("channels", [32, 64, 128]),
        kernel_sizes=config.get("kernel_sizes", [5, 3, 3]),
        dropout=config.get("dropout", 0.3),
        use_batch_norm=config.get("use_batch_norm", True)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    norm_params = metadata.get("norm_params", {})
    if norm_params:
        norm_params = {
            "mean": np.array(norm_params["mean"]),
            "std": np.array(norm_params["std"])
        }

    test_dataset = ContinuityDataset(csv_path, norm_params=norm_params)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    _, accuracy, metrics = validate(model, test_loader, criterion, device)

    print("\nEvaluation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    return {"accuracy": accuracy, **metrics}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="1D CNN Training for Continuity Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train
  python train_cnn.py train --csv dataset.csv --output ./models/

  # Export to ONNX
  python train_cnn.py export --checkpoint ./models/best_cnn.pt --output ./models/continuity_cnn.onnx

  # Evaluate
  python train_cnn.py eval --checkpoint ./models/best_cnn.pt --csv test.csv
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--csv", required=True, help="Training CSV path")
    train_parser.add_argument("--output", "-o", required=True, help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--model", default="cnn1d", choices=["cnn1d", "cnn_residual", "cnn_light"])
    train_parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to ONNX")
    export_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    export_parser.add_argument("--output", "-o", required=True, help="Output ONNX path")
    export_parser.add_argument("--opset", type=int, default=17)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    eval_parser.add_argument("--csv", required=True, help="Test CSV path")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.command == "train":
        config = DEFAULT_CONFIG.copy()
        config["epochs"] = args.epochs
        config["batch_size"] = args.batch_size
        config["learning_rate"] = args.lr
        config["model_type"] = args.model
        config["use_amp"] = not args.no_amp

        train(args.csv, args.output, config)

    elif args.command == "export":
        export_onnx(args.checkpoint, args.output, args.opset)

    elif args.command == "eval":
        evaluate(args.checkpoint, args.csv)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
