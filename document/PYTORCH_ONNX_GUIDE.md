# PyTorch + ONNX 文件分类器开发指南

> 创建时间: 2026-01-06
> 目标: 使用 PyTorch 训练神经网络，导出 ONNX 格式，集成到 C++ 项目
> 适用硬件: RTX 5060 + R9 8945HX + 16GB RAM

---

## 一、方案概述

### 1.1 技术栈

```
训练阶段                          部署阶段
┌─────────────┐                 ┌─────────────┐
│   Python    │                 │    C++      │
│  PyTorch    │  ──导出.onnx──→ │ ONNX Runtime│
│   (GPU)     │                 │   (推理)    │
└─────────────┘                 └─────────────┘
```

### 1.2 为什么选择这个方案？

| 优势 | 说明 |
|-----|------|
| **训练友好** | PyTorch 动态图，调试方便，生态丰富 |
| **部署轻量** | ONNX Runtime 比完整 PyTorch 小得多 |
| **跨平台** | ONNX 是开放标准，支持 Windows/Linux/Mac |
| **高性能** | ONNX Runtime 有专门的推理优化 |
| **学习价值** | 掌握完整的训练→部署流程 |

### 1.3 开发路线图

```
Week 1: 环境搭建 + 数据准备
        ↓
Week 2: 模型设计 + 训练调优
        ↓
Week 3: ONNX 导出 + C++ 集成
        ↓
Week 4: 测试优化 + 集成到项目
```

---

## 二、环境配置

### 2.1 Python 环境

```bash
# 创建虚拟环境
python -m venv pytorch_env
pytorch_env\Scripts\activate  # Windows

# 安装 PyTorch (CUDA 12.4，适配 RTX 5060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install numpy pandas scikit-learn matplotlib tqdm onnx onnxruntime-gpu

# 验证 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2.2 依赖版本建议

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scikit-learn>=1.3.0
onnx>=1.15.0
onnxruntime-gpu>=1.16.0
tqdm>=4.66.0
matplotlib>=3.8.0
```

### 2.3 项目结构

```
ml_file_classifier/
├── data/
│   ├── raw/                 # 原始文件样本
│   │   ├── jpg/
│   │   ├── png/
│   │   ├── pdf/
│   │   └── ...
│   └── processed/           # 处理后的特征
│       ├── train.npz
│       └── test.npz
├── models/
│   ├── checkpoints/         # 训练检查点
│   └── exported/            # 导出的 ONNX 模型
├── src/
│   ├── dataset.py           # 数据加载
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   └── export_onnx.py       # ONNX 导出
├── notebooks/
│   └── exploration.ipynb    # 实验笔记
├── config.yaml              # 配置文件
└── requirements.txt
```

---

## 三、数据准备

### 3.1 特征提取模块

```python
# src/feature_extractor.py
import numpy as np
from pathlib import Path
from typing import Tuple, List
import os

class FileFeatureExtractor:
    """文件特征提取器"""

    def __init__(self, fragment_size: int = 4096):
        self.fragment_size = fragment_size

    def extract(self, data: bytes) -> np.ndarray:
        """
        提取特征向量

        返回: 261维特征
        - 256维: 字节频率
        - 1维: Shannon熵
        - 1维: 均值
        - 1维: 标准差
        - 1维: 唯一字节比例
        - 1维: ASCII可打印字符比例
        """
        if len(data) == 0:
            return np.zeros(261, dtype=np.float32)

        # 字节频率 (256维)
        byte_array = np.frombuffer(data, dtype=np.uint8)
        byte_freq = np.bincount(byte_array, minlength=256).astype(np.float32)
        byte_freq /= len(data)

        # Shannon 熵
        non_zero = byte_freq[byte_freq > 0]
        entropy = -np.sum(non_zero * np.log2(non_zero))

        # 统计特征
        mean_val = np.mean(byte_array) / 255.0
        std_val = np.std(byte_array) / 255.0
        unique_ratio = len(np.unique(byte_array)) / 256.0

        # ASCII 可打印字符比例
        ascii_mask = (byte_array >= 32) & (byte_array <= 126)
        ascii_ratio = np.sum(ascii_mask) / len(data)

        # 拼接特征
        features = np.concatenate([
            byte_freq,
            np.array([entropy / 8.0, mean_val, std_val, unique_ratio, ascii_ratio],
                     dtype=np.float32)
        ])

        return features

    def extract_from_file(self, filepath: str, num_fragments: int = 3) -> List[np.ndarray]:
        """从文件提取多个片段的特征"""
        with open(filepath, 'rb') as f:
            data = f.read()

        if len(data) < self.fragment_size:
            return [self.extract(data)]

        features_list = []

        # 提取多个位置的片段
        positions = [
            0,                                    # 开头
            len(data) // 2,                       # 中间
            max(0, len(data) - self.fragment_size)  # 结尾
        ]

        for pos in positions[:num_fragments]:
            fragment = data[pos:pos + self.fragment_size]
            features_list.append(self.extract(fragment))

        return features_list
```

### 3.2 数据集构建

```python
# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import os

from feature_extractor import FileFeatureExtractor

class FileClassificationDataset(Dataset):
    """文件分类数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collect_and_process_data(
    data_dirs: list,
    file_types: Dict[str, list],
    max_samples_per_type: int = 500,
    fragment_size: int = 4096,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    收集文件并提取特征

    Args:
        data_dirs: 扫描目录列表
        file_types: {类型名: [扩展名列表]}
        max_samples_per_type: 每类最多样本数
        fragment_size: 片段大小
        output_path: 保存路径 (可选)

    Returns:
        features, labels, label_map
    """
    extractor = FileFeatureExtractor(fragment_size)

    # 建立扩展名到类型的映射
    ext_to_type = {}
    for type_name, extensions in file_types.items():
        for ext in extensions:
            ext_to_type[ext.lower()] = type_name

    # 收集文件
    type_files = {t: [] for t in file_types.keys()}

    print("扫描文件...")
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            continue

        for root, dirs, files in os.walk(data_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in ext_to_type:
                    continue

                type_name = ext_to_type[ext]
                if len(type_files[type_name]) >= max_samples_per_type:
                    continue

                fpath = os.path.join(root, fname)
                try:
                    if os.path.getsize(fpath) >= fragment_size:
                        type_files[type_name].append(fpath)
                except:
                    continue

    # 统计
    print("\n收集结果:")
    for type_name, files in sorted(type_files.items()):
        print(f"  {type_name}: {len(files)} 个文件")

    # 提取特征
    features_list = []
    labels_list = []
    label_map = {i: name for i, name in enumerate(sorted(file_types.keys()))}
    name_to_label = {v: k for k, v in label_map.items()}

    print("\n提取特征...")
    for type_name, files in tqdm(type_files.items()):
        label = name_to_label[type_name]

        for fpath in files:
            try:
                feats = extractor.extract_from_file(fpath)
                for feat in feats:
                    features_list.append(feat)
                    labels_list.append(label)
            except Exception as e:
                continue

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    print(f"\n总样本数: {len(labels)}")
    print(f"特征维度: {features.shape[1]}")

    # 保存
    if output_path:
        np.savez(output_path, features=features, labels=labels,
                 label_map=np.array(list(label_map.items())))
        print(f"已保存到: {output_path}")

    return features, labels, label_map


def create_data_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )

    # 标准化 (使用训练集的统计量)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_dataset = FileClassificationDataset(X_train, y_train)
    val_dataset = FileClassificationDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下设为 0
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, {'mean': mean, 'std': std}
```

---

## 四、模型设计

### 4.1 模型选择建议

| 模型 | 复杂度 | 参数量 | 适用场景 | 建议 |
|-----|-------|-------|---------|-----|
| **SimpleNet** | 低 | ~50K | 入门学习 | 第一个尝试 |
| **DeepNet** | 中 | ~200K | 追求精度 | 主力模型 |
| **ResNet-style** | 高 | ~500K | 深度探索 | 可选进阶 |

### 4.2 模型定义

```python
# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """
    简单的三层全连接网络
    适合入门和快速验证
    """
    def __init__(self, input_dim: int = 261, num_classes: int = 15, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DeepNet(nn.Module):
    """
    更深的网络，带 BatchNorm
    推荐作为主力模型
    """
    def __init__(self, input_dim: int = 261, num_classes: int = 15, dropout: float = 0.4):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Output
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNet(nn.Module):
    """
    带残差连接的网络
    适合更深的探索
    """
    def __init__(self, input_dim: int = 261, num_classes: int = 15,
                 hidden_dim: int = 256, num_blocks: int = 3, dropout: float = 0.3):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """模型工厂函数"""
    models = {
        'simple': SimpleNet,
        'deep': DeepNet,
        'resnet': ResNet,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 4.3 模型选择建议

```
初学者路线：
SimpleNet (理解基础) → DeepNet (加入 BatchNorm) → ResNet (理解残差)

实用路线：
直接用 DeepNet，参数量适中，效果稳定
```

---

## 五、训练流程

### 5.1 训练脚本

```python
# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from model import get_model, count_parameters
from dataset import collect_and_process_data, create_data_loaders


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = 'models/checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_acc = 0.0

    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
        }

        # 保存最新
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def train(self, epochs: int, early_stopping_patience: int = 15):
        """完整训练流程"""
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training on: {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print("-" * 60)

        no_improve_count = 0

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 更新学习率
            self.scheduler.step(val_acc)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 检查是否最佳
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 保存检查点
            self.save_checkpoint(epoch, is_best)

            # 打印进度
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2%} | "
                  f"LR: {lr:.2e} | "
                  f"{'*BEST*' if is_best else ''}")

            # 早停
            if no_improve_count >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print("-" * 60)
        print(f"Best validation accuracy: {self.best_acc:.2%}")

        return self.history


def main():
    """主函数"""
    # ========== 配置 ==========
    config = {
        # 数据
        'data_dirs': [
            r"C:\Windows\System32",
            r"C:\Program Files",
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
        ],
        'file_types': {
            'jpg': ['.jpg', '.jpeg'],
            'png': ['.png'],
            'gif': ['.gif'],
            'pdf': ['.pdf'],
            'zip': ['.zip'],
            'exe': ['.exe'],
            'dll': ['.dll'],
            'txt': ['.txt'],
            'doc': ['.doc', '.docx'],
            'mp3': ['.mp3'],
        },
        'max_samples_per_type': 300,

        # 模型
        'model_name': 'deep',  # 'simple', 'deep', 'resnet'
        'dropout': 0.4,

        # 训练
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
    }

    # ========== 设备 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========== 数据 ==========
    print("\n" + "=" * 60)
    print("数据准备")
    print("=" * 60)

    features, labels, label_map = collect_and_process_data(
        data_dirs=config['data_dirs'],
        file_types=config['file_types'],
        max_samples_per_type=config['max_samples_per_type'],
    )

    train_loader, val_loader, norm_params = create_data_loaders(
        features, labels,
        batch_size=config['batch_size'],
    )

    # 保存标准化参数和标签映射（推理时需要）
    np.savez('models/preprocessing.npz',
             mean=norm_params['mean'],
             std=norm_params['std'],
             label_map=np.array(list(label_map.items())))

    # ========== 模型 ==========
    print("\n" + "=" * 60)
    print("模型构建")
    print("=" * 60)

    num_classes = len(label_map)
    model = get_model(
        config['model_name'],
        input_dim=261,
        num_classes=num_classes,
        dropout=config['dropout']
    )

    print(f"Model: {config['model_name']}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Classes: {num_classes}")

    # ========== 训练 ==========
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    history = trainer.train(
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )

    # 保存配置
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n训练完成!")
    print(f"最佳模型保存在: models/checkpoints/best.pt")


if __name__ == '__main__':
    import os
    main()
```

### 5.2 训练技巧

#### 学习率调度策略

```python
# 方案1: ReduceLROnPlateau（推荐入门）
# 当验证指标停滞时自动降低学习率
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# 方案2: CosineAnnealingLR（适合固定epoch）
# 余弦退火，周期性调整
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# 方案3: OneCycleLR（追求极致性能）
# 先升后降的学习率策略
scheduler = OneCycleLR(optimizer, max_lr=1e-2, epochs=epochs,
                       steps_per_epoch=len(train_loader))
```

#### 防止过拟合

```python
# 1. Dropout（已在模型中使用）
nn.Dropout(0.3)  # 随机丢弃 30% 的神经元

# 2. Weight Decay（L2 正则化）
optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)

# 3. 数据增强（对于字节特征，可以添加噪声）
def add_noise(features, noise_level=0.01):
    noise = torch.randn_like(features) * noise_level
    return features + noise

# 4. Early Stopping（已在 Trainer 中实现）
```

#### 类别不平衡处理

```python
# 方案1: 加权损失函数
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# 方案2: 过采样少数类
from torch.utils.data import WeightedRandomSampler

sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

---

## 六、ONNX 导出

### 6.1 导出脚本

```python
# src/export_onnx.py
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from model import get_model


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: str = 'deep',
    input_dim: int = 261,
    num_classes: int = 10,
    opset_version: int = 17,
):
    """
    导出 PyTorch 模型为 ONNX 格式

    Args:
        checkpoint_path: PyTorch 检查点路径
        output_path: ONNX 输出路径
        model_name: 模型类型
        input_dim: 输入维度
        num_classes: 类别数
        opset_version: ONNX opset 版本
    """
    # 加载模型
    model = get_model(model_name, input_dim=input_dim, num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Best accuracy: {checkpoint.get('best_acc', 'N/A')}")

    # 创建示例输入
    dummy_input = torch.randn(1, input_dim)

    # 导出
    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # 常量折叠优化
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},   # 支持动态 batch
            'output': {0: 'batch_size'}
        }
    )

    print(f"Exported to: {output_path}")

    # 验证 ONNX 模型
    print("\nValidating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # 打印模型信息
    print(f"\nModel info:")
    print(f"  Input: {onnx_model.graph.input[0].name}, shape: dynamic x {input_dim}")
    print(f"  Output: {onnx_model.graph.output[0].name}, shape: dynamic x {num_classes}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return output_path


def verify_onnx_output(
    pytorch_checkpoint: str,
    onnx_path: str,
    model_name: str = 'deep',
    input_dim: int = 261,
    num_classes: int = 10,
    num_tests: int = 10,
    tolerance: float = 1e-5
):
    """验证 ONNX 输出与 PyTorch 一致"""

    # 加载 PyTorch 模型
    model = get_model(model_name, input_dim=input_dim, num_classes=num_classes)
    checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(onnx_path)

    print(f"Verifying ONNX output consistency...")

    max_diff = 0.0
    for i in range(num_tests):
        # 随机输入
        test_input = np.random.randn(1, input_dim).astype(np.float32)

        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = model(torch.from_numpy(test_input)).numpy()

        # ONNX 推理
        onnx_output = ort_session.run(None, {'input': test_input})[0]

        # 计算差异
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)

    print(f"Max difference: {max_diff:.2e}")

    if max_diff < tolerance:
        print(f"Verification PASSED (tolerance: {tolerance})")
        return True
    else:
        print(f"Verification FAILED (tolerance: {tolerance})")
        return False


def benchmark_onnx(onnx_path: str, input_dim: int = 261, num_iterations: int = 1000):
    """性能基准测试"""
    import time

    # CPU 推理
    ort_session_cpu = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    test_input = np.random.randn(1, input_dim).astype(np.float32)

    # 预热
    for _ in range(100):
        ort_session_cpu.run(None, {'input': test_input})

    # 计时
    start = time.perf_counter()
    for _ in range(num_iterations):
        ort_session_cpu.run(None, {'input': test_input})
    cpu_time = (time.perf_counter() - start) / num_iterations * 1000

    print(f"\nONNX Runtime Performance:")
    print(f"  CPU inference: {cpu_time:.3f} ms per sample")
    print(f"  Throughput: {1000/cpu_time:.0f} samples/sec")

    # GPU 推理 (如果可用)
    try:
        ort_session_gpu = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider']
        )

        # 预热
        for _ in range(100):
            ort_session_gpu.run(None, {'input': test_input})

        # 计时
        start = time.perf_counter()
        for _ in range(num_iterations):
            ort_session_gpu.run(None, {'input': test_input})
        gpu_time = (time.perf_counter() - start) / num_iterations * 1000

        print(f"  GPU inference: {gpu_time:.3f} ms per sample")
        print(f"  Throughput: {1000/gpu_time:.0f} samples/sec")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    except:
        print("  GPU inference: N/A")


def main():
    import json

    # 读取配置
    with open('models/config.json') as f:
        config = json.load(f)

    # 读取标签映射
    preprocessing = np.load('models/preprocessing.npz', allow_pickle=True)
    label_map = dict(preprocessing['label_map'])
    num_classes = len(label_map)

    # 导出
    onnx_path = export_to_onnx(
        checkpoint_path='models/checkpoints/best.pt',
        output_path='models/exported/file_classifier.onnx',
        model_name=config['model_name'],
        input_dim=261,
        num_classes=num_classes,
    )

    # 验证
    verify_onnx_output(
        pytorch_checkpoint='models/checkpoints/best.pt',
        onnx_path=onnx_path,
        model_name=config['model_name'],
        input_dim=261,
        num_classes=num_classes,
    )

    # 性能测试
    benchmark_onnx(onnx_path)


if __name__ == '__main__':
    main()
```

### 6.2 导出注意事项

```python
# 1. BatchNorm 的处理
# 导出前必须调用 model.eval()，否则 BN 会使用训练时的 running stats
model.eval()  # 关键！

# 2. 动态 batch size
dynamic_axes={
    'input': {0: 'batch_size'},
    'output': {0: 'batch_size'}
}

# 3. opset 版本选择
# opset 11: 广泛兼容
# opset 17: 较新特性，性能更好
# 建议: 使用 17，如果目标环境不支持再降级

# 4. 优化选项
do_constant_folding=True  # 常量折叠，减少计算

# 5. 模型简化（可选）
# pip install onnx-simplifier
import onnxsim
model_simplified, check = onnxsim.simplify(onnx_model)
```

---

## 七、C++ 集成

### 7.1 ONNX Runtime 安装

```bash
# 方法1: vcpkg
vcpkg install onnxruntime-gpu:x64-windows

# 方法2: 直接下载
# https://github.com/microsoft/onnxruntime/releases
# 下载 onnxruntime-win-x64-gpu-1.x.x.zip
```

### 7.2 C++ 推理代码

```cpp
// ONNXFileClassifier.h
#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <algorithm>

class ONNXFileClassifier {
public:
    struct Prediction {
        std::string fileType;
        float confidence;
        std::vector<std::pair<std::string, float>> topK;
    };

    ONNXFileClassifier() : env_(ORT_LOGGING_LEVEL_WARNING, "FileClassifier") {}

    bool LoadModel(const std::wstring& modelPath, const std::wstring& preprocessingPath) {
        try {
            // 创建会话选项
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(4);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // 尝试使用 GPU
            try {
                OrtCUDAProviderOptions cudaOptions;
                sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            } catch (...) {
                // GPU 不可用，使用 CPU
            }

            // 创建会话
            session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);

            // 加载预处理参数
            if (!LoadPreprocessing(preprocessingPath)) {
                return false;
            }

            isLoaded_ = true;
            return true;
        }
        catch (const Ort::Exception& e) {
            lastError_ = e.what();
            return false;
        }
    }

    Prediction Classify(const BYTE* data, size_t size) {
        Prediction result;

        if (!isLoaded_ || size == 0) {
            result.fileType = "unknown";
            result.confidence = 0.0f;
            return result;
        }

        // 1. 特征提取
        std::vector<float> features = ExtractFeatures(data, size);

        // 2. 标准化
        for (size_t i = 0; i < features.size(); i++) {
            features[i] = (features[i] - mean_[i]) / std_[i];
        }

        // 3. 推理
        std::vector<float> output = RunInference(features);

        // 4. Softmax
        std::vector<float> probs = Softmax(output);

        // 5. 获取预测结果
        int maxIdx = std::max_element(probs.begin(), probs.end()) - probs.begin();

        result.fileType = labelMap_[maxIdx];
        result.confidence = probs[maxIdx];

        // Top-K
        std::vector<std::pair<float, int>> indexed;
        for (size_t i = 0; i < probs.size(); i++) {
            indexed.emplace_back(probs[i], i);
        }
        std::sort(indexed.rbegin(), indexed.rend());

        for (int i = 0; i < std::min(5, (int)indexed.size()); i++) {
            result.topK.emplace_back(labelMap_[indexed[i].second], indexed[i].first);
        }

        return result;
    }

    std::string GetLastError() const { return lastError_; }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    bool isLoaded_ = false;
    std::string lastError_;

    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<std::string> labelMap_;

    std::vector<float> ExtractFeatures(const BYTE* data, size_t size) {
        std::vector<float> features(261, 0.0f);

        // 字节频率 (0-255)
        std::array<int, 256> freq = {0};
        for (size_t i = 0; i < size; i++) {
            freq[data[i]]++;
        }
        for (int i = 0; i < 256; i++) {
            features[i] = static_cast<float>(freq[i]) / size;
        }

        // Shannon 熵
        float entropy = 0.0f;
        for (int i = 0; i < 256; i++) {
            if (features[i] > 0) {
                entropy -= features[i] * std::log2(features[i]);
            }
        }
        features[256] = entropy / 8.0f;

        // 均值
        float mean = 0.0f;
        for (size_t i = 0; i < size; i++) {
            mean += data[i];
        }
        mean /= (size * 255.0f);
        features[257] = mean;

        // 标准差
        float variance = 0.0f;
        for (size_t i = 0; i < size; i++) {
            float diff = data[i] / 255.0f - mean;
            variance += diff * diff;
        }
        features[258] = std::sqrt(variance / size);

        // 唯一字节比例
        int uniqueCount = 0;
        for (int i = 0; i < 256; i++) {
            if (freq[i] > 0) uniqueCount++;
        }
        features[259] = static_cast<float>(uniqueCount) / 256.0f;

        // ASCII 可打印字符比例
        int asciiCount = 0;
        for (size_t i = 0; i < size; i++) {
            if (data[i] >= 32 && data[i] <= 126) {
                asciiCount++;
            }
        }
        features[260] = static_cast<float>(asciiCount) / size;

        return features;
    }

    std::vector<float> RunInference(const std::vector<float>& input) {
        // 准备输入
        std::vector<int64_t> inputShape = {1, 261};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(input.data()), input.size(),
            inputShape.data(), inputShape.size());

        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session_->GetInputNameAllocated(0, allocator);
        auto outputName = session_->GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        // 运行推理
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1);

        // 获取输出
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int outputSize = static_cast<int>(outputShape[1]);

        return std::vector<float>(outputData, outputData + outputSize);
    }

    std::vector<float> Softmax(const std::vector<float>& logits) {
        std::vector<float> probs(logits.size());

        float maxVal = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;

        for (size_t i = 0; i < logits.size(); i++) {
            probs[i] = std::exp(logits[i] - maxVal);
            sum += probs[i];
        }

        for (auto& p : probs) {
            p /= sum;
        }

        return probs;
    }

    bool LoadPreprocessing(const std::wstring& path) {
        // 这里需要实现 .npz 文件读取
        // 或者将预处理参数保存为简单的二进制/JSON格式
        // 简化示例：假设已经硬编码或从其他格式加载

        // 实际实现中，可以：
        // 1. 使用 cnpy 库读取 .npz
        // 2. 将 Python 导出为 JSON/二进制格式
        // 3. 在 C++ 中读取该格式

        return true;
    }
};
```

### 7.3 与 FileCarver 集成

```cpp
// 在 FileCarver.cpp 中

#include "ONNXFileClassifier.h"

class FileCarver {
private:
    std::unique_ptr<ONNXFileClassifier> onnxClassifier_;
    bool useNNClassification_ = false;

public:
    bool InitNNClassifier(const std::wstring& modelPath,
                          const std::wstring& preprocessingPath) {
        onnxClassifier_ = std::make_unique<ONNXFileClassifier>();
        if (onnxClassifier_->LoadModel(modelPath, preprocessingPath)) {
            useNNClassification_ = true;
            return true;
        }
        return false;
    }

    // 在签名匹配失败时使用神经网络分类
    bool ClassifyWithNN(const BYTE* data, size_t size,
                        std::string& outType, float& confidence) {
        if (!useNNClassification_ || !onnxClassifier_) {
            return false;
        }

        auto prediction = onnxClassifier_->Classify(data, size);

        if (prediction.confidence > 0.7f) {  // 置信度阈值
            outType = prediction.fileType;
            confidence = prediction.confidence;
            return true;
        }

        return false;
    }
};
```

---

## 八、调优建议

### 8.1 模型调优清单

```
□ 数据质量
  □ 每类样本数量均衡
  □ 样本质量良好（无损坏文件）
  □ 包含各种子类型（如 jpg: 手机照片、网页图片、扫描件）

□ 特征工程
  □ 尝试添加 bigram 特征
  □ 尝试分块熵（多个位置的熵值）
  □ 特征标准化（零均值，单位方差）

□ 模型结构
  □ 从 SimpleNet 开始，逐步增加复杂度
  □ 使用 BatchNorm 稳定训练
  □ Dropout 防止过拟合

□ 训练参数
  □ 学习率: 从 1e-3 开始
  □ Batch size: 32-128
  □ 早停耐心: 10-20 epochs

□ 验证策略
  □ 使用分层划分 (stratify)
  □ 考虑 K-Fold 交叉验证
```

### 8.2 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 验证准确率不提升 | 模型太简单 | 增加层数/宽度 |
| 训练损失震荡 | 学习率过高 | 降低学习率 |
| 训练准确率高，验证差 | 过拟合 | 增加 Dropout/数据 |
| 训练非常慢 | CPU 训练 | 检查 CUDA 是否可用 |
| ONNX 导出失败 | 不支持的操作 | 检查 opset 版本 |
| C++ 推理结果不同 | 预处理不一致 | 检查标准化参数 |

### 8.3 性能优化

```python
# 1. 混合精度训练 (可加速 2x)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. 编译模型 (PyTorch 2.0+)
model = torch.compile(model)

# 3. DataLoader 优化
DataLoader(...,
    num_workers=4,      # 多进程加载
    pin_memory=True,    # 固定内存
    prefetch_factor=2   # 预取
)
```

---

## 九、完整工作流程

```bash
# 1. 环境准备
python -m venv pytorch_env
pytorch_env\Scripts\activate
pip install -r requirements.txt

# 2. 数据准备
python src/dataset.py

# 3. 训练模型
python src/train.py

# 4. 导出 ONNX
python src/export_onnx.py

# 5. 验证 ONNX
python src/export_onnx.py --verify

# 6. C++ 集成测试
# 编译并运行 C++ 测试程序

# 预期输出:
# - models/checkpoints/best.pt    (PyTorch 模型)
# - models/exported/file_classifier.onnx (ONNX 模型)
# - models/preprocessing.npz      (预处理参数)
# - models/config.json            (配置)
```

---

## 十、进阶方向

完成基础版本后，可以探索：

1. **1D-CNN**: 直接处理原始字节，不需要手工特征
2. **Attention**: 学习哪些字节位置更重要
3. **对比学习**: 学习更好的文件表示
4. **模型蒸馏**: 用大模型教小模型
5. **量化**: INT8 量化加速推理

---

## 附录：快速启动脚本

```python
# quick_start.py - 一键训练和导出
import subprocess
import sys

def main():
    steps = [
        ("数据准备", [sys.executable, "src/dataset.py"]),
        ("模型训练", [sys.executable, "src/train.py"]),
        ("ONNX导出", [sys.executable, "src/export_onnx.py"]),
    ]

    for name, cmd in steps:
        print(f"\n{'='*60}")
        print(f"步骤: {name}")
        print('='*60)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"错误: {name} 失败")
            return

    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)
    print("输出文件:")
    print("  - models/exported/file_classifier.onnx")
    print("  - models/preprocessing.npz")

if __name__ == '__main__':
    main()
```
