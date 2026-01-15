# ML块连续性检测论文评估指南

本文档描述了用于评估ML块连续性检测模型性能和可用性的完整方案，包括指标体系、实验设计和Python自动化验证框架。

---

## 一、评估指标体系

### 1.1 模型分类性能指标 (Model Performance)

| 指标 | 公式/说明 | 意义 | 期望值 |
|------|-----------|------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 整体正确率 | >90% |
| **Precision** | TP/(TP+FP) | 预测连续时的可信度 | >85% |
| **Recall** | TP/(TP+FN) | 实际连续块的检出率 | >90% |
| **F1-Score** | 2×P×R/(P+R) | 精确率和召回率的平衡 | >87% |
| **AUC-ROC** | ROC曲线下面积 | 阈值无关的分类能力 | >0.95 |
| **FPR@95%TPR** | TPR=95%时的FPR | 高召回下的误报率 | <10% |
| **AUC-PR** | PR曲线下面积 | 不平衡数据下的性能 | >0.90 |

**说明：**
- TP (True Positive): 正确识别为连续的块对
- TN (True Negative): 正确识别为不连续的块对
- FP (False Positive): 错误识别为连续（实际不连续）
- FN (False Negative): 错误识别为不连续（实际连续）

### 1.2 文件恢复效果指标 (Recovery Quality)

| 指标 | 计算方法 | 意义 | 期望值 |
|------|----------|------|--------|
| **File Recovery Rate (FRR)** | 成功恢复文件数/总文件数 | 文件级恢复成功率 | >80% |
| **Byte Accuracy** | 正确字节数/总字节数 | 字节级准确性 | >95% |
| **Structural Integrity** | 通过格式验证的文件/恢复文件数 | 文件结构完整性 | >90% |
| **Content Fidelity** | Hash完全匹配的文件/恢复文件数 | 内容保真度 | >70% |
| **Block Boundary Accuracy** | 正确检测的边界数/总边界数 | 边界检测准确性 | >85% |
| **Fragment Reassembly Rate** | 正确重组的碎片文件/碎片文件总数 | 碎片重组能力 | >75% |

### 1.3 实用性指标 (Practicality)

| 指标 | 测量方法 | 意义 | 参考值 |
|------|----------|------|--------|
| **Inference Latency** | 单次块对推理耗时 (ms) | 实时性 | <1ms |
| **Throughput** | 每秒处理数据量 (MB/s) | 处理效率 | >100MB/s |
| **Memory Footprint** | 峰值内存占用 (MB) | 资源消耗 | <500MB |
| **Model Size** | ONNX文件大小 (KB) | 部署便利性 | <500KB |
| **CPU Utilization** | 多线程扩展效率 | 并行能力 | >80% |
| **Batch Efficiency** | 批处理vs单条的加速比 | 批处理优化 | >5x |

### 1.4 对比基线方法 (Baselines)

| 方法 | 说明 | 预期表现 |
|------|------|----------|
| **Random** | 随机猜测连续性 | ~50% Accuracy |
| **Entropy-only** | 仅用熵差判断 | ~65% Accuracy |
| **Byte-histogram** | 字节分布相似度 | ~70% Accuracy |
| **Signature-only** | 仅用文件签名匹配 | 高Precision低Recall |
| **Sequential** | 假设所有块都连续 | 无碎片时100%，碎片时差 |
| **Bifragment Gap Carving** | 传统双碎片恢复 | 有限碎片场景有效 |
| **SmartCarving** | 基于内容的启发式 | 特定格式有效 |

---

## 二、实验设计

### 2.1 实验1：模型分类性能评估

**目的：** 评估模型在标准测试集上的分类能力

**测试集构成：**
```
总样本: 20,000 块对
├── 正样本 (连续): 10,000
│   ├── ZIP: 2,500
│   ├── MP3: 2,500
│   ├── MP4: 2,500
│   └── JPEG: 2,500
└── 负样本 (不连续): 10,000
    ├── 不同文件拼接: 4,000
    ├── 随机数据: 2,000
    ├── 损坏数据: 2,000
    └── 不同类型混合: 2,000
```

**输出：**
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- 各文件类型的分类报告

### 2.2 实验2：碎片化影响评估

**目的：** 评估不同碎片化程度下的恢复能力

**实验变量：**
```
碎片化程度: [0%, 25%, 50%, 75%, 100%]
文件类型: [ZIP, MP3, MP4, JPEG]
测试文件数: 每类型100个文件
```

**碎片化定义：**
- 0%: 所有块连续存储
- 50%: 一半的块被打乱位置
- 100%: 完全随机分布

**输出：**
- Recovery Rate vs Fragmentation 曲线
- 各文件类型的恢复率对比

### 2.3 实验3：损坏鲁棒性评估

**目的：** 评估模型对数据损坏的容忍度

**损坏类型：**
| 类型 | 说明 | 测试级别 |
|------|------|----------|
| Bit-flip | 随机比特翻转 | 0.1%, 0.5%, 1%, 2%, 5% |
| Zero-fill | 区域清零 | 1%, 5%, 10%, 20% |
| Overwrite | 随机数据覆盖 | 5%, 10%, 20%, 30% |
| Truncation | 块截断 | 10%, 25%, 50% |

**输出：**
- Accuracy vs Corruption Level 曲线
- 不同损坏类型的影响对比

### 2.4 实验4：性能基准测试

**目的：** 评估实际部署场景下的性能表现

**测试场景：**
```
磁盘镜像大小: [1GB, 10GB, 100GB, 1TB]
线程数: [1, 2, 4, 8, 16]
批大小: [1, 32, 64, 128, 256]
```

**输出：**
- Throughput vs Image Size
- Throughput vs Thread Count
- Latency Distribution (P50, P95, P99)

### 2.5 实验5：与基线方法对比

**目的：** 证明ML方法相对传统方法的优势

**对比维度：**
- 恢复成功率
- 处理速度
- 碎片场景表现
- 损坏场景表现

**输出：**
- 方法对比表格
- 各场景下的性能对比图

### 2.6 实验6：真实场景案例研究

**目的：** 验证在真实磁盘镜像上的效果

**测试数据：**
- 模拟删除后的NTFS分区镜像
- 格式化后的USB驱动器镜像
- 部分覆写的存储设备镜像

**输出：**
- 案例描述和恢复结果
- 与商业工具的对比（如有）

---

## 三、Python自动化评估框架

### 3.1 目录结构

```
eval/
├── config.py              # 配置文件
├── data_generator.py      # 测试数据生成
├── cli_wrapper.py         # CLI调用封装
├── metrics.py             # 指标计算
├── evaluator.py           # 主评估流程
├── visualizer.py          # 结果可视化
├── run_evaluation.py      # 入口脚本
└── requirements.txt       # 依赖
```

### 3.2 完整代码

```python
#!/usr/bin/env python3
"""
ML Block Continuity Detection - Automated Evaluation Framework

用于自动化评估ML块连续性检测模型的性能和可用性。
生成论文所需的数据和图表。

Usage:
    python run_evaluation.py --config eval_config.json
    python run_evaluation.py --quick-test  # 快速测试模式
"""

import subprocess
import hashlib
import json
import time
import tempfile
import shutil
import struct
import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    # 路径配置
    cli_path: str = r"D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe"
    source_files_dir: str = ""          # 源文件目录（用于生成测试数据）
    output_dir: str = "./eval_results"  # 输出目录
    temp_dir: str = ""                  # 临时目录（默认系统临时目录）

    # 测试参数
    file_types: List[str] = field(default_factory=lambda: ["zip", "mp3", "mp4", "jpg"])
    fragmentation_levels: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    corruption_levels: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2])
    corruption_types: List[str] = field(default_factory=lambda: ["bitflip", "zero_fill", "overwrite"])

    # 性能测试参数
    image_sizes_mb: List[int] = field(default_factory=lambda: [100, 500, 1000])
    thread_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # 样本数量
    files_per_type: int = 50            # 每种类型的测试文件数
    classification_samples: int = 10000  # 分类测试样本数

    # 块大小
    block_size: int = 8192              # 8KB

    # 运行控制
    skip_slow_tests: bool = False       # 跳过耗时测试
    parallel_workers: int = 4           # 并行工作线程

    def __post_init__(self):
        if not self.temp_dir:
            self.temp_dir = str(Path(tempfile.gettempdir()) / "ml_continuity_eval")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 测试数据生成器
# ============================================================================

class TestDataGenerator:
    """生成已知ground truth的测试数据"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.rng = np.random.default_rng(42)  # 固定种子保证可重复性

    def create_synthetic_file(self, file_type: str, size_kb: int = 100) -> bytes:
        """
        创建合成的测试文件

        根据文件类型生成具有正确签名和结构的测试数据
        """
        size = size_kb * 1024

        if file_type == "zip":
            return self._create_synthetic_zip(size)
        elif file_type in ["jpg", "jpeg"]:
            return self._create_synthetic_jpeg(size)
        elif file_type == "mp3":
            return self._create_synthetic_mp3(size)
        elif file_type == "mp4":
            return self._create_synthetic_mp4(size)
        else:
            # 通用二进制数据
            return bytes(self.rng.integers(0, 256, size, dtype=np.uint8))

    def _create_synthetic_zip(self, size: int) -> bytes:
        """创建合成ZIP文件"""
        # ZIP Local File Header
        header = b'PK\x03\x04'  # 签名
        header += b'\x14\x00'   # 版本
        header += b'\x00\x00'   # 标志
        header += b'\x08\x00'   # 压缩方法 (DEFLATE)
        header += b'\x00\x00\x00\x00'  # 修改时间/日期
        header += b'\x00\x00\x00\x00'  # CRC32
        header += b'\x00\x00\x00\x00'  # 压缩大小
        header += b'\x00\x00\x00\x00'  # 原始大小
        header += b'\x08\x00'   # 文件名长度
        header += b'\x00\x00'   # 额外字段长度
        header += b'test.txt'  # 文件名

        # 填充数据（模拟DEFLATE压缩数据）
        remaining = size - len(header) - 22  # 减去EOCD大小
        data = bytes(self.rng.integers(0, 256, max(0, remaining), dtype=np.uint8))

        # End of Central Directory
        eocd = b'PK\x05\x06'
        eocd += b'\x00\x00' * 4
        eocd += struct.pack('<H', 1)  # 条目数
        eocd += struct.pack('<H', 1)
        eocd += b'\x00\x00\x00\x00' * 2
        eocd += b'\x00\x00'

        return header + data + eocd

    def _create_synthetic_jpeg(self, size: int) -> bytes:
        """创建合成JPEG文件"""
        # JPEG SOI + APP0
        header = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'

        # SOF0 (Start of Frame)
        sof = b'\xFF\xC0\x00\x11\x08\x00\x10\x00\x10\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01'

        # DHT (Huffman Table) - 简化版
        dht = b'\xFF\xC4\x00\x1F\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00'
        dht += bytes(range(12))

        # SOS (Start of Scan)
        sos = b'\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00'

        # 图像数据
        remaining = size - len(header) - len(sof) - len(dht) - len(sos) - 2
        # 避免在数据中出现 0xFF 后跟标记字节
        img_data = bytes(self.rng.integers(0, 255, max(0, remaining), dtype=np.uint8))

        # EOI
        eoi = b'\xFF\xD9'

        return header + sof + dht + sos + img_data + eoi

    def _create_synthetic_mp3(self, size: int) -> bytes:
        """创建合成MP3文件"""
        # ID3v2 Header
        id3 = b'ID3\x04\x00\x00\x00\x00\x00\x00'

        # MP3 Frame Header (MPEG1 Layer3, 128kbps, 44100Hz)
        frame_header = b'\xFF\xFB\x90\x00'

        # 填充帧数据
        frame_size = 417  # 128kbps @ 44100Hz 的帧大小
        num_frames = (size - len(id3)) // frame_size

        data = id3
        for _ in range(num_frames):
            data += frame_header
            data += bytes(self.rng.integers(0, 256, frame_size - 4, dtype=np.uint8))

        # 填充剩余
        remaining = size - len(data)
        if remaining > 0:
            data += bytes(self.rng.integers(0, 256, remaining, dtype=np.uint8))

        return data[:size]

    def _create_synthetic_mp4(self, size: int) -> bytes:
        """创建合成MP4文件"""
        # ftyp box
        ftyp = struct.pack('>I', 20)  # box size
        ftyp += b'ftyp'
        ftyp += b'isom'  # major brand
        ftyp += struct.pack('>I', 512)  # minor version
        ftyp += b'isom'  # compatible brand

        # mdat box (媒体数据)
        mdat_size = size - len(ftyp) - 8
        mdat = struct.pack('>I', mdat_size + 8)
        mdat += b'mdat'
        mdat += bytes(self.rng.integers(0, 256, max(0, mdat_size), dtype=np.uint8))

        return ftyp + mdat

    def create_fragmented_image(
        self,
        output_path: str,
        files: List[Tuple[str, bytes]],
        fragmentation: float = 0.5,
        image_size_mb: int = 100
    ) -> Dict:
        """
        创建碎片化的磁盘镜像

        Args:
            output_path: 输出镜像路径
            files: [(filename, data), ...] 文件列表
            fragmentation: 碎片化程度 0.0-1.0
            image_size_mb: 镜像大小(MB)

        Returns:
            ground_truth: {
                "files": [...],
                "block_sequence": [...],
                "fragmentation": float
            }
        """
        block_size = self.config.block_size
        image_size = image_size_mb * 1024 * 1024

        # 将文件切分成块
        all_blocks = []
        ground_truth = {
            "files": [],
            "block_sequence": [],
            "fragmentation": fragmentation,
            "block_size": block_size
        }

        for file_idx, (filename, data) in enumerate(files):
            file_hash = hashlib.sha256(data).hexdigest()
            num_blocks = (len(data) + block_size - 1) // block_size

            for block_idx in range(num_blocks):
                start = block_idx * block_size
                end = min(start + block_size, len(data))
                block_data = data[start:end]

                # 填充到完整块大小
                if len(block_data) < block_size:
                    block_data = block_data + b'\x00' * (block_size - len(block_data))

                all_blocks.append({
                    "file_idx": file_idx,
                    "block_idx": block_idx,
                    "data": block_data,
                    "is_last": block_idx == num_blocks - 1
                })

            ground_truth["files"].append({
                "name": filename,
                "original_hash": file_hash,
                "total_size": len(data),
                "num_blocks": num_blocks
            })

        # 根据碎片化程度打乱块顺序
        num_blocks = len(all_blocks)
        indices = list(range(num_blocks))

        # 选择要打乱的块
        num_to_shuffle = int(num_blocks * fragmentation)
        if num_to_shuffle > 1:
            shuffle_indices = self.rng.choice(num_blocks, num_to_shuffle, replace=False)
            shuffled = shuffle_indices.copy()
            self.rng.shuffle(shuffled)

            # 创建映射
            new_order = indices.copy()
            for orig, new in zip(shuffle_indices, shuffled):
                new_order[orig] = indices[new]

            # 应用新顺序
            shuffled_blocks = [all_blocks[i] for i in new_order]
        else:
            shuffled_blocks = all_blocks

        # 写入镜像文件
        with open(output_path, "wb") as f:
            for idx, block in enumerate(shuffled_blocks):
                f.write(block["data"])

                ground_truth["block_sequence"].append({
                    "file_idx": block["file_idx"],
                    "block_idx": block["block_idx"],
                    "offset": idx * block_size,
                    "is_last": block["is_last"]
                })

        # 填充到指定大小
        current_size = len(shuffled_blocks) * block_size
        if current_size < image_size:
            with open(output_path, "ab") as f:
                padding = bytes(self.rng.integers(0, 256, image_size - current_size, dtype=np.uint8))
                f.write(padding)

        return ground_truth

    def apply_corruption(
        self,
        image_path: str,
        corruption_type: str,
        severity: float
    ) -> Dict:
        """
        对镜像应用损坏

        Args:
            image_path: 镜像文件路径
            corruption_type: 损坏类型 (bitflip/zero_fill/overwrite)
            severity: 损坏程度 0.0-1.0

        Returns:
            corruption_info: 损坏详情
        """
        data = bytearray(Path(image_path).read_bytes())
        corruption_info = {
            "type": corruption_type,
            "severity": severity,
            "affected_regions": []
        }

        if corruption_type == "bitflip":
            # 随机比特翻转
            num_flips = int(len(data) * 8 * severity)
            for _ in range(num_flips):
                byte_idx = self.rng.integers(0, len(data))
                bit_idx = self.rng.integers(0, 8)
                data[byte_idx] ^= (1 << bit_idx)
            corruption_info["num_flips"] = num_flips

        elif corruption_type == "zero_fill":
            # 区域清零
            num_regions = max(1, int(severity * 100))
            for _ in range(num_regions):
                start = self.rng.integers(0, len(data) - 1024)
                length = self.rng.integers(512, 4096)
                end = min(start + length, len(data))
                data[start:end] = b'\x00' * (end - start)
                corruption_info["affected_regions"].append({
                    "start": int(start),
                    "length": int(end - start)
                })

        elif corruption_type == "overwrite":
            # 随机数据覆盖
            num_regions = max(1, int(severity * 50))
            for _ in range(num_regions):
                start = self.rng.integers(0, len(data) - 8192)
                length = self.rng.integers(4096, 16384)
                end = min(start + length, len(data))
                random_data = bytes(self.rng.integers(0, 256, end - start, dtype=np.uint8))
                data[start:end] = random_data
                corruption_info["affected_regions"].append({
                    "start": int(start),
                    "length": int(end - start)
                })

        Path(image_path).write_bytes(bytes(data))
        return corruption_info

    def generate_classification_samples(
        self,
        output_csv: str,
        num_samples: int = 10000
    ) -> str:
        """
        生成分类测试样本CSV

        生成正负样本对，用于评估模型分类性能
        """
        samples = []
        num_positive = num_samples // 2
        num_negative = num_samples - num_positive

        logger.info(f"Generating {num_samples} classification samples...")

        # 为每种文件类型生成测试文件
        test_files = {}
        for file_type in self.config.file_types:
            files = []
            for i in range(20):  # 每种类型20个文件
                size_kb = self.rng.integers(50, 500)
                data = self.create_synthetic_file(file_type, size_kb)
                files.append(data)
            test_files[file_type] = files

        # 生成正样本（同一文件的连续块）
        for i in range(num_positive):
            file_type = self.rng.choice(self.config.file_types)
            file_data = self.rng.choice(test_files[file_type])

            # 选择连续的两个块
            num_blocks = len(file_data) // self.config.block_size
            if num_blocks < 2:
                continue

            block_idx = self.rng.integers(0, num_blocks - 1)
            block1_start = block_idx * self.config.block_size
            block2_start = (block_idx + 1) * self.config.block_size

            block1 = file_data[block1_start:block1_start + self.config.block_size]
            block2 = file_data[block2_start:block2_start + self.config.block_size]

            samples.append({
                "block1_hash": hashlib.md5(block1).hexdigest()[:16],
                "block2_hash": hashlib.md5(block2).hexdigest()[:16],
                "file_type": file_type,
                "is_continuous": 1,
                "sample_type": "same_file"
            })

        # 生成负样本（不同文件/随机数据）
        for i in range(num_negative):
            sample_type = self.rng.choice([
                "different_files", "random_data", "different_type"
            ])

            if sample_type == "different_files":
                # 同类型不同文件
                file_type = self.rng.choice(self.config.file_types)
                files = test_files[file_type]
                if len(files) < 2:
                    continue
                file1, file2 = self.rng.choice(files, 2, replace=False)

                idx1 = self.rng.integers(0, len(file1) // self.config.block_size)
                idx2 = self.rng.integers(0, len(file2) // self.config.block_size)

                block1 = file1[idx1*self.config.block_size:(idx1+1)*self.config.block_size]
                block2 = file2[idx2*self.config.block_size:(idx2+1)*self.config.block_size]

            elif sample_type == "random_data":
                # 文件块 + 随机数据
                file_type = self.rng.choice(self.config.file_types)
                file_data = self.rng.choice(test_files[file_type])

                idx = self.rng.integers(0, len(file_data) // self.config.block_size)
                block1 = file_data[idx*self.config.block_size:(idx+1)*self.config.block_size]
                block2 = bytes(self.rng.integers(0, 256, self.config.block_size, dtype=np.uint8))

            else:  # different_type
                # 不同类型文件
                types = self.rng.choice(self.config.file_types, 2, replace=False)
                file1 = self.rng.choice(test_files[types[0]])
                file2 = self.rng.choice(test_files[types[1]])

                idx1 = self.rng.integers(0, max(1, len(file1) // self.config.block_size))
                idx2 = self.rng.integers(0, max(1, len(file2) // self.config.block_size))

                block1 = file1[idx1*self.config.block_size:(idx1+1)*self.config.block_size]
                block2 = file2[idx2*self.config.block_size:(idx2+1)*self.config.block_size]
                file_type = types[0]

            # 确保块大小正确
            if len(block1) < self.config.block_size:
                block1 = block1 + b'\x00' * (self.config.block_size - len(block1))
            if len(block2) < self.config.block_size:
                block2 = block2 + b'\x00' * (self.config.block_size - len(block2))

            samples.append({
                "block1_hash": hashlib.md5(block1).hexdigest()[:16],
                "block2_hash": hashlib.md5(block2).hexdigest()[:16],
                "file_type": file_type,
                "is_continuous": 0,
                "sample_type": sample_type
            })

        # 保存CSV
        df = pd.DataFrame(samples)
        df.to_csv(output_csv, index=False)

        logger.info(f"Saved {len(samples)} samples to {output_csv}")
        return output_csv


# ============================================================================
# CLI 调用封装
# ============================================================================

class CLIWrapper:
    """封装 Filerestore_CLI 调用"""

    def __init__(self, cli_path: str):
        self.cli_path = cli_path

        if not Path(cli_path).exists():
            raise FileNotFoundError(f"CLI not found: {cli_path}")

    def run_command(self, command: str, timeout: int = 3600) -> Dict:
        """
        运行CLI命令

        Args:
            command: CLI内部命令（如 "carvepool D: zip output/ 0 continuity"）
            timeout: 超时时间(秒)

        Returns:
            {
                "stdout": str,
                "stderr": str,
                "return_code": int,
                "elapsed_time": float
            }
        """
        # 构建命令 - 通过stdin发送命令
        full_command = f'echo {command} | "{self.cli_path}"'

        start_time = time.time()
        try:
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            elapsed = time.time() - start_time

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "elapsed_time": elapsed
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Command timed out",
                "return_code": -1,
                "elapsed_time": timeout
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "elapsed_time": time.time() - start_time
            }

    def run_carve(
        self,
        image_path: str,
        file_type: str,
        output_dir: str,
        mode: str = "continuity",
        start_offset: int = 0
    ) -> Dict:
        """
        运行文件恢复

        Args:
            image_path: 磁盘镜像路径
            file_type: 文件类型
            output_dir: 输出目录
            mode: 恢复模式 (continuity/sequential/0)
            start_offset: 起始偏移

        Returns:
            {
                "files_recovered": [...],
                "elapsed_time": float,
                "cli_output": str
            }
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 清空输出目录
        for f in Path(output_dir).glob("*"):
            f.unlink()

        # 运行carve命令
        command = f"carvepool {image_path} {file_type} {output_dir} {start_offset} {mode}"
        result = self.run_command(command)

        # 收集恢复的文件
        recovered_files = []
        for f in Path(output_dir).glob("*"):
            if f.is_file():
                recovered_files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "hash": hashlib.sha256(f.read_bytes()).hexdigest()
                })

        return {
            "files_recovered": recovered_files,
            "elapsed_time": result["elapsed_time"],
            "cli_output": result["stdout"],
            "return_code": result["return_code"]
        }

    def run_mlscan(
        self,
        source_dir: str,
        output_csv: str,
        mode: str = "continuity",
        samples_per_file: int = 10,
        incremental: bool = False
    ) -> Dict:
        """
        运行ML数据集生成
        """
        command = f"mlscan {source_dir} --{mode} --output={output_csv} --samples-per-file={samples_per_file}"
        if incremental:
            command += " --incremental"

        result = self.run_command(command)

        # 统计生成的样本数
        sample_count = 0
        if Path(output_csv).exists():
            df = pd.read_csv(output_csv)
            sample_count = len(df)

        return {
            "sample_count": sample_count,
            "elapsed_time": result["elapsed_time"],
            "cli_output": result["stdout"]
        }


# ============================================================================
# 指标计算器
# ============================================================================

class MetricsCalculator:
    """计算各项评估指标"""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        计算分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率（可选，用于计算AUC）

        Returns:
            metrics dict
        """
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix,
                precision_recall_curve, roc_curve, average_precision_score
            )
        except ImportError:
            logger.error("sklearn not installed. Run: pip install scikit-learn")
            return {}

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["true_negatives"] = int(cm[0, 0])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["true_positives"] = int(cm[1, 1])

        # AUC metrics (需要概率)
        if y_prob is not None:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
                metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))

                # FPR at 95% TPR
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                idx = np.argmin(np.abs(tpr - 0.95))
                metrics["fpr_at_95tpr"] = float(fpr[idx])
                metrics["threshold_at_95tpr"] = float(thresholds[idx])

                # ROC curve data (for plotting)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }

                # PR curve data
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                metrics["pr_curve"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist()
                }
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")

        return metrics

    @staticmethod
    def calculate_recovery_metrics(
        ground_truth: Dict,
        recovered_dir: str
    ) -> Dict:
        """
        计算文件恢复质量指标

        Args:
            ground_truth: 原始文件信息 {"files": [...]}
            recovered_dir: 恢复文件目录

        Returns:
            recovery metrics dict
        """
        recovered_path = Path(recovered_dir)

        total_files = len(ground_truth.get("files", []))
        if total_files == 0:
            return {"error": "No files in ground truth"}

        recovered_count = 0
        hash_matches = 0
        structure_valid = 0
        total_original_bytes = 0
        total_recovered_bytes = 0

        for gt_file in ground_truth["files"]:
            total_original_bytes += gt_file["total_size"]

            # 查找可能匹配的恢复文件
            # 按大小和扩展名匹配
            ext = Path(gt_file["name"]).suffix
            candidates = list(recovered_path.glob(f"*{ext}"))

            best_match = None
            best_match_score = 0

            for candidate in candidates:
                try:
                    rec_data = candidate.read_bytes()
                    rec_hash = hashlib.sha256(rec_data).hexdigest()

                    # 完全匹配
                    if rec_hash == gt_file["original_hash"]:
                        best_match = candidate
                        best_match_score = 1.0
                        break

                    # 部分匹配（按大小相似度）
                    size_ratio = min(len(rec_data), gt_file["total_size"]) / max(len(rec_data), gt_file["total_size"])
                    if size_ratio > best_match_score:
                        best_match = candidate
                        best_match_score = size_ratio

                except Exception:
                    continue

            if best_match is not None and best_match_score > 0.5:
                recovered_count += 1
                rec_data = best_match.read_bytes()
                total_recovered_bytes += len(rec_data)

                # Hash完全匹配
                rec_hash = hashlib.sha256(rec_data).hexdigest()
                if rec_hash == gt_file["original_hash"]:
                    hash_matches += 1

                # 结构验证
                if MetricsCalculator._validate_file_structure(best_match):
                    structure_valid += 1

        return {
            "file_recovery_rate": recovered_count / total_files,
            "hash_match_rate": hash_matches / total_files,
            "structure_validity_rate": structure_valid / recovered_count if recovered_count > 0 else 0,
            "byte_recovery_rate": total_recovered_bytes / total_original_bytes if total_original_bytes > 0 else 0,
            "files_recovered": recovered_count,
            "files_total": total_files,
            "hash_matches": hash_matches,
            "structure_valid": structure_valid
        }

    @staticmethod
    def _validate_file_structure(file_path: Path) -> bool:
        """验证文件结构完整性"""
        ext = file_path.suffix.lower()

        try:
            data = file_path.read_bytes()

            if ext == ".zip":
                import zipfile
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        return zf.testzip() is None
                except zipfile.BadZipFile:
                    return False

            elif ext in [".jpg", ".jpeg"]:
                # 检查JPEG标记
                if len(data) < 4:
                    return False
                return data[:2] == b'\xFF\xD8' and data[-2:] == b'\xFF\xD9'

            elif ext == ".mp3":
                # 检查ID3或帧同步
                if len(data) < 3:
                    return False
                return data[:3] == b'ID3' or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)

            elif ext == ".mp4":
                # 检查ftyp box
                if len(data) < 8:
                    return False
                return data[4:8] in [b'ftyp', b'moov', b'mdat', b'free']

            else:
                return True  # 未知类型默认通过

        except Exception:
            return False

    @staticmethod
    def calculate_performance_metrics(
        elapsed_times: List[float],
        data_sizes: List[int]
    ) -> Dict:
        """
        计算性能指标

        Args:
            elapsed_times: 耗时列表(秒)
            data_sizes: 数据大小列表(字节)

        Returns:
            performance metrics dict
        """
        if not elapsed_times:
            return {}

        elapsed_times = np.array(elapsed_times)
        data_sizes = np.array(data_sizes)

        # 避免除零
        elapsed_times = np.maximum(elapsed_times, 1e-6)

        throughputs = data_sizes / elapsed_times  # bytes per second

        return {
            "avg_latency_ms": float(np.mean(elapsed_times) * 1000),
            "std_latency_ms": float(np.std(elapsed_times) * 1000),
            "min_latency_ms": float(np.min(elapsed_times) * 1000),
            "max_latency_ms": float(np.max(elapsed_times) * 1000),
            "p50_latency_ms": float(np.percentile(elapsed_times, 50) * 1000),
            "p95_latency_ms": float(np.percentile(elapsed_times, 95) * 1000),
            "p99_latency_ms": float(np.percentile(elapsed_times, 99) * 1000),
            "avg_throughput_mbps": float(np.mean(throughputs) / (1024 * 1024)),
            "std_throughput_mbps": float(np.std(throughputs) / (1024 * 1024)),
            "total_data_mb": float(np.sum(data_sizes) / (1024 * 1024)),
            "total_time_s": float(np.sum(elapsed_times))
        }


# ============================================================================
# 主评估器
# ============================================================================

class Evaluator:
    """主评估流程"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.data_gen = TestDataGenerator(config)
        self.cli = CLIWrapper(config.cli_path)
        self.metrics = MetricsCalculator()

        self.results = {
            "config": asdict(config),
            "timestamp": datetime.now().isoformat(),
            "experiments": {}
        }

    def run_full_evaluation(self) -> Dict:
        """运行完整评估流程"""
        logger.info("=" * 60)
        logger.info("ML Block Continuity Detection - Full Evaluation")
        logger.info("=" * 60)

        try:
            # 实验1: 碎片化影响
            logger.info("\n[Experiment 1] Fragmentation Impact")
            self.results["experiments"]["fragmentation"] = self._eval_fragmentation()

            # 实验2: 损坏鲁棒性
            logger.info("\n[Experiment 2] Corruption Robustness")
            self.results["experiments"]["corruption"] = self._eval_corruption()

            # 实验3: 文件类型性能
            logger.info("\n[Experiment 3] File Type Performance")
            self.results["experiments"]["file_types"] = self._eval_file_types()

            # 实验4: 方法对比
            logger.info("\n[Experiment 4] Method Comparison")
            self.results["experiments"]["comparison"] = self._eval_method_comparison()

            # 实验5: 性能基准
            if not self.config.skip_slow_tests:
                logger.info("\n[Experiment 5] Performance Benchmark")
                self.results["experiments"]["performance"] = self._eval_performance()

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.results["error"] = str(e)

        # 保存结果
        self._save_results()

        return self.results

    def _eval_fragmentation(self) -> Dict:
        """评估碎片化影响"""
        results = []

        for frag_level in self.config.fragmentation_levels:
            logger.info(f"  Testing fragmentation={frag_level:.0%}...")

            try:
                # 准备测试文件
                test_files = []
                for file_type in self.config.file_types[:2]:  # 限制类型数量加速测试
                    for i in range(5):
                        size_kb = 100 + i * 50
                        data = self.data_gen.create_synthetic_file(file_type, size_kb)
                        test_files.append((f"test_{file_type}_{i}.{file_type}", data))

                # 创建测试镜像
                image_path = Path(self.config.temp_dir) / f"frag_{frag_level:.2f}.img"
                gt = self.data_gen.create_fragmented_image(
                    str(image_path),
                    test_files,
                    fragmentation=frag_level,
                    image_size_mb=50
                )

                # 保存ground truth
                gt_path = Path(self.config.temp_dir) / f"frag_{frag_level:.2f}_gt.json"
                with open(gt_path, "w") as f:
                    json.dump(gt, f)

                # 运行恢复 (ML模式)
                output_dir = Path(self.config.temp_dir) / f"output_frag_{frag_level:.2f}"
                carve_result = self.cli.run_carve(
                    str(image_path),
                    self.config.file_types[0],
                    str(output_dir),
                    mode="continuity"
                )

                # 计算指标
                recovery_metrics = self.metrics.calculate_recovery_metrics(gt, str(output_dir))

                results.append({
                    "fragmentation": frag_level,
                    **recovery_metrics,
                    "elapsed_time": carve_result["elapsed_time"]
                })

                # 清理
                shutil.rmtree(output_dir, ignore_errors=True)
                image_path.unlink(missing_ok=True)
                gt_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"  Failed for fragmentation={frag_level}: {e}")
                results.append({
                    "fragmentation": frag_level,
                    "error": str(e)
                })

        return {"results": results}

    def _eval_corruption(self) -> Dict:
        """评估损坏鲁棒性"""
        results = []

        for corr_type in self.config.corruption_types:
            for corr_level in self.config.corruption_levels:
                logger.info(f"  Testing {corr_type} @ {corr_level:.1%}...")

                try:
                    # 准备测试数据
                    test_files = []
                    for i in range(10):
                        file_type = self.config.file_types[i % len(self.config.file_types)]
                        data = self.data_gen.create_synthetic_file(file_type, 100)
                        test_files.append((f"test_{i}.{file_type}", data))

                    # 创建镜像（中等碎片化）
                    image_path = Path(self.config.temp_dir) / f"corr_{corr_type}_{corr_level:.2f}.img"
                    gt = self.data_gen.create_fragmented_image(
                        str(image_path),
                        test_files,
                        fragmentation=0.3,
                        image_size_mb=30
                    )

                    # 应用损坏
                    if corr_level > 0:
                        self.data_gen.apply_corruption(str(image_path), corr_type, corr_level)

                    # 运行恢复
                    output_dir = Path(self.config.temp_dir) / f"output_corr_{corr_type}_{corr_level:.2f}"
                    carve_result = self.cli.run_carve(
                        str(image_path),
                        self.config.file_types[0],
                        str(output_dir),
                        mode="continuity"
                    )

                    # 计算指标
                    recovery_metrics = self.metrics.calculate_recovery_metrics(gt, str(output_dir))

                    results.append({
                        "corruption_type": corr_type,
                        "corruption_level": corr_level,
                        **recovery_metrics,
                        "elapsed_time": carve_result["elapsed_time"]
                    })

                    # 清理
                    shutil.rmtree(output_dir, ignore_errors=True)
                    image_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.warning(f"  Failed for {corr_type}@{corr_level}: {e}")
                    results.append({
                        "corruption_type": corr_type,
                        "corruption_level": corr_level,
                        "error": str(e)
                    })

        return {"results": results}

    def _eval_file_types(self) -> Dict:
        """评估不同文件类型"""
        results = []

        for file_type in self.config.file_types:
            logger.info(f"  Testing file type: {file_type}...")

            try:
                # 只用该类型的文件
                test_files = []
                for i in range(20):
                    size_kb = 50 + i * 25
                    data = self.data_gen.create_synthetic_file(file_type, size_kb)
                    test_files.append((f"test_{i}.{file_type}", data))

                # 创建镜像（50%碎片化）
                image_path = Path(self.config.temp_dir) / f"type_{file_type}.img"
                gt = self.data_gen.create_fragmented_image(
                    str(image_path),
                    test_files,
                    fragmentation=0.5,
                    image_size_mb=30
                )

                # 运行恢复
                output_dir = Path(self.config.temp_dir) / f"output_type_{file_type}"
                carve_result = self.cli.run_carve(
                    str(image_path),
                    file_type,
                    str(output_dir),
                    mode="continuity"
                )

                # 计算指标
                recovery_metrics = self.metrics.calculate_recovery_metrics(gt, str(output_dir))

                results.append({
                    "file_type": file_type,
                    **recovery_metrics,
                    "elapsed_time": carve_result["elapsed_time"]
                })

                # 清理
                shutil.rmtree(output_dir, ignore_errors=True)
                image_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"  Failed for {file_type}: {e}")
                results.append({
                    "file_type": file_type,
                    "error": str(e)
                })

        return {"results": results}

    def _eval_method_comparison(self) -> Dict:
        """对比不同恢复方法"""
        results = []
        methods = ["continuity", "0"]  # ML模式 vs 传统顺序模式

        # 创建统一测试集
        test_files = []
        for file_type in self.config.file_types:
            for i in range(10):
                data = self.data_gen.create_synthetic_file(file_type, 150)
                test_files.append((f"test_{file_type}_{i}.{file_type}", data))

        # 不同碎片化程度
        for frag_level in [0.0, 0.5, 1.0]:
            image_path = Path(self.config.temp_dir) / f"compare_frag_{frag_level:.1f}.img"
            gt = self.data_gen.create_fragmented_image(
                str(image_path),
                test_files,
                fragmentation=frag_level,
                image_size_mb=50
            )

            for method in methods:
                logger.info(f"  Testing method={method}, frag={frag_level:.0%}...")

                try:
                    output_dir = Path(self.config.temp_dir) / f"output_compare_{method}_{frag_level:.1f}"
                    carve_result = self.cli.run_carve(
                        str(image_path),
                        self.config.file_types[0],
                        str(output_dir),
                        mode=method
                    )

                    recovery_metrics = self.metrics.calculate_recovery_metrics(gt, str(output_dir))

                    results.append({
                        "method": "ML-Continuity" if method == "continuity" else "Sequential",
                        "fragmentation": frag_level,
                        **recovery_metrics,
                        "elapsed_time": carve_result["elapsed_time"]
                    })

                    shutil.rmtree(output_dir, ignore_errors=True)

                except Exception as e:
                    logger.warning(f"  Failed for method={method}: {e}")
                    results.append({
                        "method": method,
                        "fragmentation": frag_level,
                        "error": str(e)
                    })

            image_path.unlink(missing_ok=True)

        return {"results": results}

    def _eval_performance(self) -> Dict:
        """性能基准测试"""
        results = []

        for size_mb in self.config.image_sizes_mb:
            logger.info(f"  Testing image size: {size_mb}MB...")

            try:
                # 创建测试数据
                test_files = []
                num_files = size_mb // 2  # 平均每个文件2MB
                for i in range(num_files):
                    file_type = self.config.file_types[i % len(self.config.file_types)]
                    data = self.data_gen.create_synthetic_file(file_type, 2048)  # 2MB
                    test_files.append((f"test_{i}.{file_type}", data))

                # 创建镜像
                image_path = Path(self.config.temp_dir) / f"perf_{size_mb}mb.img"
                gt = self.data_gen.create_fragmented_image(
                    str(image_path),
                    test_files,
                    fragmentation=0.5,
                    image_size_mb=size_mb
                )

                # 多次运行取平均
                elapsed_times = []
                for run in range(3):
                    output_dir = Path(self.config.temp_dir) / f"output_perf_{size_mb}_{run}"
                    carve_result = self.cli.run_carve(
                        str(image_path),
                        self.config.file_types[0],
                        str(output_dir),
                        mode="continuity"
                    )
                    elapsed_times.append(carve_result["elapsed_time"])
                    shutil.rmtree(output_dir, ignore_errors=True)

                avg_time = np.mean(elapsed_times)
                throughput = (size_mb * 1024 * 1024) / avg_time if avg_time > 0 else 0

                results.append({
                    "image_size_mb": size_mb,
                    "avg_elapsed_time": float(avg_time),
                    "std_elapsed_time": float(np.std(elapsed_times)),
                    "throughput_mbps": float(throughput / (1024 * 1024)),
                    "runs": len(elapsed_times)
                })

                image_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"  Failed for {size_mb}MB: {e}")
                results.append({
                    "image_size_mb": size_mb,
                    "error": str(e)
                })

        return {"results": results}

    def _save_results(self):
        """保存评估结果"""
        output_path = Path(self.config.output_dir) / "evaluation_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"\nResults saved to: {output_path}")

        # 同时保存CSV格式（方便导入Excel）
        self._export_csv_summaries()

    def _export_csv_summaries(self):
        """导出CSV格式的汇总数据"""
        output_dir = Path(self.config.output_dir)

        for exp_name, exp_data in self.results.get("experiments", {}).items():
            if "results" in exp_data:
                df = pd.DataFrame(exp_data["results"])
                csv_path = output_dir / f"{exp_name}_results.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"  Exported: {csv_path}")


# ============================================================================
# 结果可视化
# ============================================================================

class Visualizer:
    """生成论文用图表"""

    def __init__(self, results: Dict, output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 尝试导入绘图库
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            logger.warning("matplotlib not installed. Skipping visualizations.")
            self.has_matplotlib = False

    def generate_all_plots(self):
        """生成所有图表"""
        if not self.has_matplotlib:
            return

        logger.info("\nGenerating visualizations...")

        self._plot_fragmentation_impact()
        self._plot_corruption_robustness()
        self._plot_file_type_comparison()
        self._plot_method_comparison()
        self._plot_performance_scaling()

        logger.info(f"Plots saved to: {self.output_dir}")

    def _plot_fragmentation_impact(self):
        """图1: 碎片化影响"""
        exp = self.results.get("experiments", {}).get("fragmentation", {})
        data = exp.get("results", [])

        if not data:
            return

        # 过滤掉错误数据
        data = [d for d in data if "error" not in d]
        if not data:
            return

        fig, ax = self.plt.subplots(figsize=(8, 5))

        frags = [d["fragmentation"] for d in data]
        rates = [d.get("file_recovery_rate", 0) for d in data]

        ax.plot(frags, rates, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax.fill_between(frags, rates, alpha=0.3, color='#2ecc71')

        ax.set_xlabel("Fragmentation Level", fontsize=12)
        ax.set_ylabel("File Recovery Rate", fontsize=12)
        ax.set_title("Recovery Performance vs Disk Fragmentation", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

        fig.tight_layout()
        fig.savefig(self.output_dir / "fig1_fragmentation_impact.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig1_fragmentation_impact.png", dpi=300, bbox_inches='tight')
        self.plt.close(fig)

    def _plot_corruption_robustness(self):
        """图2: 损坏鲁棒性"""
        exp = self.results.get("experiments", {}).get("corruption", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]
        if not data:
            return

        fig, ax = self.plt.subplots(figsize=(10, 6))

        colors = {'bitflip': '#e74c3c', 'zero_fill': '#3498db', 'overwrite': '#9b59b6'}
        markers = {'bitflip': 'o', 'zero_fill': 's', 'overwrite': '^'}

        for corr_type in set(d["corruption_type"] for d in data):
            type_data = [d for d in data if d["corruption_type"] == corr_type]
            type_data.sort(key=lambda x: x["corruption_level"])

            levels = [d["corruption_level"] for d in type_data]
            rates = [d.get("file_recovery_rate", 0) for d in type_data]

            ax.plot(levels, rates, marker=markers.get(corr_type, 'o'),
                   linewidth=2, markersize=8, label=corr_type.replace('_', ' ').title(),
                   color=colors.get(corr_type, '#333'))

        ax.set_xlabel("Corruption Level", fontsize=12)
        ax.set_ylabel("File Recovery Rate", fontsize=12)
        ax.set_title("Recovery Robustness Against Data Corruption", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "fig2_corruption_robustness.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig2_corruption_robustness.png", dpi=300, bbox_inches='tight')
        self.plt.close(fig)

    def _plot_file_type_comparison(self):
        """图3: 文件类型对比"""
        exp = self.results.get("experiments", {}).get("file_types", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]
        if not data:
            return

        fig, ax = self.plt.subplots(figsize=(8, 5))

        types = [d["file_type"].upper() for d in data]
        rates = [d.get("file_recovery_rate", 0) for d in data]

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(types, rates, color=colors[:len(types)])

        ax.set_ylabel("File Recovery Rate", fontsize=12)
        ax.set_title("Recovery Performance by File Type", fontsize=14)
        ax.set_ylim(0, 1.1)

        # 在柱子上显示数值
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.1%}', ha='center', fontsize=10)

        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        fig.savefig(self.output_dir / "fig3_file_type_comparison.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig3_file_type_comparison.png", dpi=300, bbox_inches='tight')
        self.plt.close(fig)

    def _plot_method_comparison(self):
        """图4: 方法对比"""
        exp = self.results.get("experiments", {}).get("comparison", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]
        if not data:
            return

        fig, ax = self.plt.subplots(figsize=(10, 6))

        # 按碎片化程度分组
        frags = sorted(set(d["fragmentation"] for d in data))
        methods = sorted(set(d["method"] for d in data))

        x = np.arange(len(frags))
        width = 0.35

        colors = {'ML-Continuity': '#2ecc71', 'Sequential': '#e74c3c'}

        for i, method in enumerate(methods):
            method_data = [d for d in data if d["method"] == method]
            method_data.sort(key=lambda x: x["fragmentation"])
            rates = [d.get("file_recovery_rate", 0) for d in method_data]

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, rates, width, label=method,
                         color=colors.get(method, '#333'))

        ax.set_xlabel("Fragmentation Level", fontsize=12)
        ax.set_ylabel("File Recovery Rate", fontsize=12)
        ax.set_title("ML-based vs Traditional Carving Methods", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f:.0%}' for f in frags])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        fig.savefig(self.output_dir / "fig4_method_comparison.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig4_method_comparison.png", dpi=300, bbox_inches='tight')
        self.plt.close(fig)

    def _plot_performance_scaling(self):
        """图5: 性能扩展性"""
        exp = self.results.get("experiments", {}).get("performance", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]
        if not data:
            return

        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 5))

        sizes = [d["image_size_mb"] for d in data]
        times = [d.get("avg_elapsed_time", 0) for d in data]
        throughputs = [d.get("throughput_mbps", 0) for d in data]

        # 处理时间
        ax1.plot(sizes, times, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax1.set_xlabel("Image Size (MB)", fontsize=12)
        ax1.set_ylabel("Processing Time (seconds)", fontsize=12)
        ax1.set_title("Processing Time Scaling", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 吞吐量
        ax2.bar(range(len(sizes)), throughputs, color='#2ecc71')
        ax2.set_xlabel("Image Size (MB)", fontsize=12)
        ax2.set_ylabel("Throughput (MB/s)", fontsize=12)
        ax2.set_title("Processing Throughput", fontsize=14)
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels([str(s) for s in sizes])
        ax2.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        fig.savefig(self.output_dir / "fig5_performance_scaling.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig5_performance_scaling.png", dpi=300, bbox_inches='tight')
        self.plt.close(fig)


# ============================================================================
# LaTeX表格生成
# ============================================================================

class LaTeXTableGenerator:
    """生成论文用LaTeX表格"""

    def __init__(self, results: Dict, output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)

    def generate_all_tables(self):
        """生成所有表格"""
        logger.info("\nGenerating LaTeX tables...")

        self._generate_fragmentation_table()
        self._generate_comparison_table()
        self._generate_performance_table()

        logger.info(f"Tables saved to: {self.output_dir}")

    def _generate_fragmentation_table(self):
        """生成碎片化实验表格"""
        exp = self.results.get("experiments", {}).get("fragmentation", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Recovery Performance at Different Fragmentation Levels}",
            r"\label{tab:fragmentation}",
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"Fragmentation & Recovery Rate & Hash Match & Structure Valid & Time (s) \\",
            r"\midrule"
        ]

        for d in data:
            frag = f"{d['fragmentation']:.0%}"
            recovery = f"{d.get('file_recovery_rate', 0):.1%}"
            hash_match = f"{d.get('hash_match_rate', 0):.1%}"
            structure = f"{d.get('structure_validity_rate', 0):.1%}"
            time = f"{d.get('elapsed_time', 0):.2f}"

            lines.append(f"{frag} & {recovery} & {hash_match} & {structure} & {time} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        output_path = self.output_dir / "table_fragmentation.tex"
        output_path.write_text("\n".join(lines))

    def _generate_comparison_table(self):
        """生成方法对比表格"""
        exp = self.results.get("experiments", {}).get("comparison", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison of ML-based and Traditional Carving Methods}",
            r"\label{tab:comparison}",
            r"\begin{tabular}{cccc}",
            r"\toprule",
            r"Method & Fragmentation & Recovery Rate & Time (s) \\",
            r"\midrule"
        ]

        for d in sorted(data, key=lambda x: (x['method'], x['fragmentation'])):
            method = d['method']
            frag = f"{d['fragmentation']:.0%}"
            recovery = f"{d.get('file_recovery_rate', 0):.1%}"
            time = f"{d.get('elapsed_time', 0):.2f}"

            lines.append(f"{method} & {frag} & {recovery} & {time} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        output_path = self.output_dir / "table_comparison.tex"
        output_path.write_text("\n".join(lines))

    def _generate_performance_table(self):
        """生成性能表格"""
        exp = self.results.get("experiments", {}).get("performance", {})
        data = exp.get("results", [])

        if not data:
            return

        data = [d for d in data if "error" not in d]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Processing Performance at Different Image Sizes}",
            r"\label{tab:performance}",
            r"\begin{tabular}{cccc}",
            r"\toprule",
            r"Image Size (MB) & Avg Time (s) & Std Time (s) & Throughput (MB/s) \\",
            r"\midrule"
        ]

        for d in data:
            size = d['image_size_mb']
            avg_time = f"{d.get('avg_elapsed_time', 0):.2f}"
            std_time = f"{d.get('std_elapsed_time', 0):.3f}"
            throughput = f"{d.get('throughput_mbps', 0):.1f}"

            lines.append(f"{size} & {avg_time} & {std_time} & {throughput} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        output_path = self.output_dir / "table_performance.tex"
        output_path.write_text("\n".join(lines))


# ============================================================================
# 入口函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML Block Continuity Detection - Automated Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                           # 运行完整评估
  python run_evaluation.py --quick                   # 快速测试模式
  python run_evaluation.py --config my_config.json   # 使用自定义配置
  python run_evaluation.py --output ./results        # 指定输出目录
  python run_evaluation.py --plots-only results.json # 仅生成图表
        """
    )

    parser.add_argument("--config", type=str, help="Configuration JSON file")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (fewer samples)")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--plots-only", type=str, help="Only generate plots from existing results JSON")
    parser.add_argument("--source-dir", type=str, help="Source files directory for test data")
    parser.add_argument("--cli-path", type=str, help="Path to Filerestore_CLI.exe")

    args = parser.parse_args()

    # 仅生成图表模式
    if args.plots_only:
        with open(args.plots_only, "r") as f:
            results = json.load(f)

        output_dir = args.output or Path(args.plots_only).parent

        visualizer = Visualizer(results, str(output_dir))
        visualizer.generate_all_plots()

        latex_gen = LaTeXTableGenerator(results, str(output_dir))
        latex_gen.generate_all_tables()

        return

    # 加载或创建配置
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = EvalConfig(**config_dict)
    else:
        config = EvalConfig()

    # 应用命令行参数
    config.output_dir = args.output
    if args.quick:
        config.fragmentation_levels = [0.0, 0.5, 1.0]
        config.corruption_levels = [0.0, 0.1]
        config.corruption_types = ["bitflip"]
        config.image_sizes_mb = [100]
        config.files_per_type = 10
    if args.skip_slow:
        config.skip_slow_tests = True
    if args.source_dir:
        config.source_files_dir = args.source_dir
    if args.cli_path:
        config.cli_path = args.cli_path

    # 运行评估
    evaluator = Evaluator(config)
    results = evaluator.run_full_evaluation()

    # 生成可视化
    visualizer = Visualizer(results, config.output_dir)
    visualizer.generate_all_plots()

    # 生成LaTeX表格
    latex_gen = LaTeXTableGenerator(results, config.output_dir)
    latex_gen.generate_all_tables()

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Results: {config.output_dir}/evaluation_results.json")
    logger.info(f"Plots:   {config.output_dir}/*.pdf")
    logger.info(f"Tables:  {config.output_dir}/*.tex")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

### 3.3 依赖文件 (requirements.txt)

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

---

## 四、使用说明

### 4.1 快速开始

```bash
# 1. 安装依赖
pip install -r eval/requirements.txt

# 2. 快速测试（验证环境）
python eval/run_evaluation.py --quick --output ./quick_results

# 3. 完整评估
python eval/run_evaluation.py --output ./full_results

# 4. 仅生成图表（从已有结果）
python eval/run_evaluation.py --plots-only ./full_results/evaluation_results.json
```

### 4.2 自定义配置

创建 `eval_config.json`:

```json
{
    "cli_path": "D:/path/to/Filerestore_CLI.exe",
    "source_files_dir": "D:/test_files",
    "output_dir": "./eval_results",
    "file_types": ["zip", "mp3", "mp4", "jpg"],
    "fragmentation_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
    "corruption_levels": [0.0, 0.01, 0.05, 0.1, 0.2],
    "image_sizes_mb": [100, 500, 1000],
    "files_per_type": 50,
    "skip_slow_tests": false
}
```

```bash
python eval/run_evaluation.py --config eval_config.json
```

### 4.3 输出文件

```
eval_results/
├── evaluation_results.json     # 完整结果（JSON）
├── fragmentation_results.csv   # 碎片化实验（CSV）
├── corruption_results.csv      # 损坏实验（CSV）
├── file_types_results.csv      # 文件类型实验（CSV）
├── comparison_results.csv      # 方法对比（CSV）
├── performance_results.csv     # 性能实验（CSV）
├── fig1_fragmentation_impact.pdf   # 图1
├── fig2_corruption_robustness.pdf  # 图2
├── fig3_file_type_comparison.pdf   # 图3
├── fig4_method_comparison.pdf      # 图4
├── fig5_performance_scaling.pdf    # 图5
├── table_fragmentation.tex     # LaTeX表格
├── table_comparison.tex
└── table_performance.tex
```

---

## 五、建议的CLI扩展

为更好支持自动化评估，建议在CLI中添加以下功能：

### 5.1 JSON输出模式

```cpp
// 命令格式
carvepool D: zip output/ 0 continuity --json

// 输出
{
    "files_recovered": 15,
    "bytes_processed": 104857600,
    "elapsed_time_ms": 1234,
    "blocks_evaluated": 12800,
    "continuity_decisions": {
        "positive": 1200,
        "negative": 11600
    }
}
```

### 5.2 单块对推理命令

```cpp
// 命令格式
mlpredict block1.bin block2.bin --model=continuity

// 输出
{
    "is_continuous": true,
    "confidence": 0.95,
    "inference_time_ms": 0.5
}
```

### 5.3 批量推理命令

```cpp
// 命令格式
mlbatch pairs.csv --model=continuity --output=predictions.csv

// pairs.csv 格式
block1_path,block2_path
block_0000.bin,block_0001.bin
block_0001.bin,block_0002.bin
...
```

---

## 六、参考文献格式

论文中引用本工具时建议使用：

```bibtex
@software{filerestore_cli,
    title = {Filerestore-CLI: ML-based Block Continuity Detection for File Carving},
    author = {Your Name},
    year = {2025},
    url = {https://github.com/your-repo/filerestore-cli}
}
```

---

## 七、版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2025-01-11 | 初始版本，包含完整评估框架 |
