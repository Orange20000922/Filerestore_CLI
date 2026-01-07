# ML 文件类型分类器

基于 PyTorch 的文件类型分类器，用于数据恢复场景。

## 目录结构

```
ml/
├── data/               # 数据集目录
│   └── govdocs/        # Govdocs1 数据集
├── models/
│   ├── checkpoints/    # 训练检查点
│   └── exported/       # 导出的 ONNX 模型
├── src/
│   ├── config.py       # 配置文件
│   ├── dataset.py      # 数据集处理
│   ├── model.py        # 模型定义
│   ├── train.py        # 训练脚本
│   └── export_onnx.py  # ONNX 导出工具
└── requirements.txt    # 依赖列表
```

## 环境配置

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 安装 PyTorch (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
# 使用默认配置训练
python src/train.py

# 指定模型类型
python src/train.py --model simple   # 简单网络
python src/train.py --model deep     # 深层网络 (默认)
python src/train.py --model resnet   # 残差网络

# 自定义参数
python src/train.py --epochs 50 --batch-size 128 --lr 0.001

# 从检查点恢复训练
python src/train.py --resume models/checkpoints/best_deep.pt
```

### 2. 导出 ONNX

```bash
# 从检查点导出
python src/export_onnx.py export models/checkpoints/best_deep.pt

# 指定输出路径
python src/export_onnx.py export models/checkpoints/best_deep.pt -o output/model.onnx

# 验证 ONNX 模型
python src/export_onnx.py verify models/exported/file_classifier_deep.onnx

# 使用真实文件测试
python src/export_onnx.py test models/exported/file_classifier_deep.onnx test.pdf

# 列出可用模型
python src/export_onnx.py list
```

## 模型架构

### SimpleNet
- 2 层全连接网络
- 参数量约 70K
- 适合快速验证

### DeepNet (默认)
- 3 层全连接 + BatchNorm
- 参数量约 300K
- 平衡性能和效率

### ResNet
- 带残差连接的网络
- 参数量约 200K
- 适合更深的网络

## 特征说明

输入特征为 261 维向量:
- 256 维: 字节频率分布 (0-255 每个字节的出现频率)
- 1 维: Shannon 熵 (归一化到 0-1)
- 1 维: 均值 (归一化到 0-1)
- 1 维: 标准差 (归一化到 0-1)
- 1 维: 唯一字节比例
- 1 维: ASCII 可打印字符比例

## 数据集

默认使用 Govdocs1 数据集的子集:
- 自动下载前 5 个 zip 文件 (约 2.5GB)
- 支持 10 种文件类型: pdf, doc, xls, ppt, html, txt, xml, jpg, gif, png
- 每类最多 500 个样本

## C++ 集成

导出的 ONNX 模型可以使用 ONNX Runtime 在 C++ 中推理:

```cpp
#include <onnxruntime_cxx_api.h>

// 加载模型
Ort::Session session(env, L"file_classifier.onnx", session_options);

// 准备输入 (需要先标准化)
std::vector<float> input_data = extract_features(file_data);
normalize(input_data, mean, std);

// 推理
auto output = session.Run(...);
int predicted_class = argmax(output);
```

详细集成指南请参考 `document/PYTORCH_ONNX_GUIDE.md`。
