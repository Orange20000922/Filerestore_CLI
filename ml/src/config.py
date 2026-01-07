"""
配置文件 - 文件分类器训练配置
"""
import os
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_DIR = MODELS_DIR / "exported"

# 确保目录存在
for dir_path in [DATA_DIR, CHECKPOINTS_DIR, EXPORTED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据集配置
DATASET_CONFIG = {
    # Govdocs1 下载设置
    "govdocs_num_zips": 40,  # 下载的 zip 数量（每个约 500MB，40个约20GB）

    # 目标文件类型（Govdocs1 中常见的类型）
    "file_types": {
        "pdf": [".pdf"],
        "doc": [".doc"],
        "xls": [".xls"],
        "ppt": [".ppt"],
        "html": [".html", ".htm"],
        "txt": [".txt"],
        "xml": [".xml"],
        "jpg": [".jpg", ".jpeg"],
        "gif": [".gif"],
        "png": [".png"],
    },

    # 每类最大样本数
    "max_samples_per_type": 500,

    # 每类最小样本数（少于此数的类别将被排除）
    "min_samples_per_type": 20,

    # 片段大小（字节）
    "fragment_size": 4096,

    # 每个文件提取的片段数
    "fragments_per_file": 3,

    # 最小文件大小
    "min_file_size": 4096,
}

# 模型配置
MODEL_CONFIG = {
    "input_dim": 261,       # 特征维度：256(字节频率) + 5(统计特征)
    "hidden_dims": [512, 256, 128],  # 隐藏层维度
    "dropout": 0.4,
    "use_batch_norm": True,
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 15,
    "val_split": 0.2,
    "random_state": 42,

    # 学习率调度
    "lr_scheduler": "plateau",  # "plateau" 或 "cosine"
    "lr_factor": 0.5,
    "lr_patience": 5,
}

# 设备配置
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    print(f"[GPU] 使用 {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("[WARNING] CUDA 不可用，使用 CPU 训练")
