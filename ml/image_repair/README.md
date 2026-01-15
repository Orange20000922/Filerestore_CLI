# 图像头部修复 - 训练指南

## 概述

本目录包含用于训练图像头部修复模型的脚本和工具。

## 文件说明

| 文件 | 用途 |
|------|------|
| `generate_training_data.py` | 从正常图像生成训练数据 |
| `train_model.py` | 训练头部修复模型 |
| `README.md` | 本文档 |

## 环境要求

```bash
pip install torch torchvision numpy pillow onnx onnxruntime
```

## 使用步骤

### 1. 收集训练图像

准备大量正常的 JPEG/PNG 图像（建议每类 10,000+ 张）：

```
images/
├── jpeg/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
└── png/
    ├── image1.png
    ├── image2.png
    └── ...
```

推荐来源：
- ImageNet 数据集
- COCO 数据集
- Flickr 公开图像
- 自己收集的图像库

### 2. 生成训练数据

```bash
# 处理所有类型
python generate_training_data.py /path/to/images --output training_data --type both --max-samples 10000

# 仅处理 JPEG
python generate_training_data.py /path/to/images --output training_data --type jpeg

# 仅处理 PNG
python generate_training_data.py /path/to/images --output training_data --type png
```

输出结构：
```
training_data/
├── jpeg/
│   ├── sample_000000.npz
│   ├── sample_000001.npz
│   ├── ...
│   └── metadata.json
└── png/
    ├── sample_000000.npz
    ├── sample_000001.npz
    ├── ...
    └── metadata.json
```

### 3. 训练模型

```bash
# 使用 GPU 训练（推荐）
python train_model.py training_data --epochs 50 --batch-size 64 --device cuda

# 使用 CPU 训练
python train_model.py training_data --epochs 50 --batch-size 32 --device cpu
```

训练参数：
- `--epochs`: 训练轮数（默认 50）
- `--batch-size`: 批大小（默认 64）
- `--device`: 设备 (cuda/cpu)
- `--output`: 模型输出目录（默认 models/）

### 4. 模型输出

训练完成后，会生成以下文件：

```
models/
└── image_type_classifier.onnx  # 图像类型分类器（JPEG/PNG）
```

### 5. 部署模型

将生成的 `.onnx` 模型文件复制到项目的模型目录：

```bash
# 复制到项目
cp models/*.onnx ../../Filerestore_CLI/models/
```

在 C++ 代码中加载：

```cpp
MLFileRepair repair;
repair.Initialize("path/to/models/");
```

## 训练数据说明

### 特征提取

从每个图像提取以下特征：

| 特征 | 说明 | 维度 |
|------|------|------|
| 均值 | 字节值平均值 | 1 |
| 标准差 | 字节值标准差 | 1 |
| 熵 | 数据随机性 | 1 |
| 零值计数 | 0x00 字节数 | 1 |
| 0xFF 计数 | 0xFF 字节数 | 1 |
| 直方图 | 16 区间分布 | 16 |

**总特征维度**: 21

### 标签

| 类型 | 标签 |
|------|------|
| JPEG | 0 |
| PNG | 1 |

## 模型架构

### 图像类型分类器

```
Input (21) → Dense(64) + ReLU + Dropout(0.3)
           → Dense(32) + ReLU + Dropout(0.2)
           → Dense(2)  → Softmax
```

**输出**: [JPEG 概率, PNG 概率]

## 预期性能

### 分类器

| 指标 | 预期值 |
|------|--------|
| 验证准确率 | > 95% |
| 推理时间 | < 1ms |
| 模型大小 | ~50KB |

## 注意事项

### 1. 数据质量

- 确保图像完整无损
- 避免重复图像
- 包含多样化的图像内容和尺寸

### 2. 训练资源

- GPU 推荐: NVIDIA GTX 1060 或更高
- 内存: 至少 8GB RAM
- 磁盘: 至少 10GB 空闲空间

### 3. 过拟合防范

- 使用 Dropout
- 数据增强
- 早停策略（已实现）

## 高级用法

### 自定义特征提取

修改 `generate_training_data.py` 中的 `extract_statistical_features()` 方法：

```python
def extract_statistical_features(self, data):
    # 添加自定义特征
    features = {
        'mean': float(np.mean(arr)),
        'custom_feature': your_calculation(arr)
    }
    return features
```

### 修改模型架构

修改 `train_model.py` 中的 `ImageTypeClassifier` 类：

```python
class ImageTypeClassifier(nn.Module):
    def __init__(self, input_size=21):
        super().__init__()
        # 自定义网络结构
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # ...
        )
```

## 故障排除

### 问题：训练准确率不提升

**解决方案**:
- 增加训练数据量
- 调整学习率 `lr=0.0001`
- 增加网络层数

### 问题：ONNX 导出失败

**解决方案**:
- 检查 PyTorch 和 ONNX 版本兼容性
- 简化模型结构（避免动态操作）

### 问题：内存不足

**解决方案**:
- 减小 `batch_size`
- 减少训练数据量
- 使用梯度累积

## 参考资源

- [PyTorch 文档](https://pytorch.org/docs/)
- [ONNX 文档](https://onnx.ai/onnx/)
- [图像格式规范](https://www.w3.org/Graphics/)
