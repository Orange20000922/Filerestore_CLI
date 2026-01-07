"""
数据集处理模块 - 特征提取、数据加载
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import hashlib
from tqdm import tqdm
import os
import zipfile
import urllib.request

from config import DATA_DIR, DATASET_CONFIG


class FileFeatureExtractor:
    """文件特征提取器"""

    def __init__(self, fragment_size: int = 4096):
        self.fragment_size = fragment_size

    def extract(self, data: bytes) -> np.ndarray:
        """
        提取特征向量（261维）

        特征组成:
        - 256维: 字节频率分布
        - 1维: Shannon熵（归一化到0-1）
        - 1维: 均值（归一化到0-1）
        - 1维: 标准差（归一化到0-1）
        - 1维: 唯一字节比例
        - 1维: ASCII可打印字符比例
        """
        if len(data) == 0:
            return np.zeros(261, dtype=np.float32)

        # 转换为 numpy 数组
        byte_array = np.frombuffer(data, dtype=np.uint8)

        # 1. 字节频率 (256维)
        byte_freq = np.bincount(byte_array, minlength=256).astype(np.float32)
        byte_freq /= len(data)

        # 2. Shannon 熵
        non_zero = byte_freq[byte_freq > 0]
        entropy = -np.sum(non_zero * np.log2(non_zero))
        entropy_normalized = entropy / 8.0  # 最大熵为 8

        # 3. 统计特征
        mean_val = np.mean(byte_array) / 255.0
        std_val = np.std(byte_array) / 255.0

        # 4. 唯一字节比例
        unique_ratio = len(np.unique(byte_array)) / 256.0

        # 5. ASCII 可打印字符比例 (32-126)
        ascii_mask = (byte_array >= 32) & (byte_array <= 126)
        ascii_ratio = np.sum(ascii_mask) / len(data)

        # 拼接特征
        features = np.concatenate([
            byte_freq,
            np.array([entropy_normalized, mean_val, std_val, unique_ratio, ascii_ratio],
                     dtype=np.float32)
        ])

        return features

    def extract_from_file(self, filepath: str, num_fragments: int = 3) -> List[np.ndarray]:
        """从文件提取多个片段的特征"""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
        except Exception:
            return []

        if len(data) < self.fragment_size:
            # 文件太小，直接使用全部数据
            return [self.extract(data)]

        features_list = []
        file_size = len(data)

        # 提取不同位置的片段
        positions = [
            0,                                          # 开头
            file_size // 2 - self.fragment_size // 2,   # 中间
            file_size - self.fragment_size,             # 结尾
        ]

        for pos in positions[:num_fragments]:
            pos = max(0, pos)
            fragment = data[pos:pos + self.fragment_size]
            if len(fragment) >= self.fragment_size // 2:  # 至少一半大小
                features_list.append(self.extract(fragment))

        return features_list


class FileClassificationDataset(Dataset):
    """文件分类数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DownloadProgressBar:
    """下载进度条"""
    def __init__(self, filename: str):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True,
                           desc=f"下载 {self.filename}")
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_govdocs_subset(num_zips: int = DATASET_CONFIG.get("num_zips", 5), output_dir: Optional[Path] = None) -> Path:
    """
    下载 Govdocs1 数据集子集

    Args:
        num_zips: 下载的 zip 文件数量（每个约 500MB）
        output_dir: 输出目录

    Returns:
        数据目录路径
    """
    if output_dir is None:
        output_dir = DATA_DIR / "govdocs"

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    base_url = "https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles"

    print(f"下载 Govdocs1 子集 ({num_zips} 个压缩包，每个约 500MB)...")
    print(f"目标目录: {output_dir}")
    print("提示: 如果下载太慢，可手动下载后放到 raw 目录")
    print(f"下载地址: {base_url}/000.zip ~ {base_url}/{num_zips-1:03d}.zip\n")

    success_count = 0
    for i in range(num_zips):
        zip_name = f"{i:03d}.zip"
        zip_path = raw_dir / zip_name
        extract_dir = output_dir / f"{i:03d}"

        # 检查是否已解压
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[跳过] {zip_name} 已解压")
            success_count += 1
            continue

        # 检查是否已下载
        if not zip_path.exists():
            url = f"{base_url}/{zip_name}"
            max_retries = 3
            for retry in range(max_retries):
                try:
                    progress = DownloadProgressBar(zip_name)
                    urllib.request.urlretrieve(url, zip_path, progress)
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"下载失败，重试 ({retry+2}/{max_retries}): {e}")
                    else:
                        print(f"[错误] 下载 {zip_name} 失败: {e}")
                        continue

        # 解压
        if zip_path.exists():
            try:
                print(f"解压 {zip_name}...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(output_dir)
                success_count += 1
                print(f"[完成] {zip_name}")
            except Exception as e:
                print(f"[错误] 解压 {zip_name} 失败: {e}")

    print(f"\n下载完成: {success_count}/{num_zips} 个压缩包")
    return output_dir


def collect_files(
    data_dirs: List[Path],
    file_types: Dict[str, List[str]],
    max_samples_per_type: int = 500,
    min_file_size: int = 4096,
) -> Dict[str, List[Path]]:
    """
    收集文件样本

    Args:
        data_dirs: 扫描的目录列表
        file_types: {类型名: [扩展名列表]}
        max_samples_per_type: 每类最大样本数
        min_file_size: 最小文件大小

    Returns:
        {类型名: [文件路径列表]}
    """
    # 建立扩展名到类型的映射
    ext_to_type = {}
    for type_name, extensions in file_types.items():
        for ext in extensions:
            ext_to_type[ext.lower()] = type_name

    # 收集文件
    type_files = defaultdict(list)
    seen_hashes = set()

    print("扫描文件...")
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue

        for root, dirs, files in os.walk(data_dir):
            # 跳过隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in ext_to_type:
                    continue

                type_name = ext_to_type[ext]
                if len(type_files[type_name]) >= max_samples_per_type:
                    continue

                fpath = Path(root) / fname
                try:
                    size = fpath.stat().st_size
                    if size < min_file_size:
                        continue

                    # 简单去重（使用文件头哈希）
                    with open(fpath, 'rb') as f:
                        header = f.read(1024)
                    file_hash = hashlib.md5(header + str(size).encode()).hexdigest()

                    if file_hash in seen_hashes:
                        continue
                    seen_hashes.add(file_hash)

                    type_files[type_name].append(fpath)

                except Exception:
                    continue

    return dict(type_files)


def extract_dataset(
    type_files: Dict[str, List[Path]],
    fragment_size: int = 4096,
    fragments_per_file: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    从文件中提取特征数据集

    Returns:
        features, labels, label_map
    """
    extractor = FileFeatureExtractor(fragment_size)

    # 建立标签映射
    sorted_types = sorted(type_files.keys())
    label_map = {i: name for i, name in enumerate(sorted_types)}
    name_to_label = {v: k for k, v in label_map.items()}

    features_list = []
    labels_list = []

    print("\n提取特征...")
    for type_name in tqdm(sorted_types, desc="处理类型"):
        label = name_to_label[type_name]
        files = type_files[type_name]

        for fpath in tqdm(files, desc=f"  {type_name}", leave=False):
            feats = extractor.extract_from_file(str(fpath), fragments_per_file)
            for feat in feats:
                features_list.append(feat)
                labels_list.append(label)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    return features, labels, label_map


def create_data_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练和验证数据加载器

    Returns:
        train_loader, val_loader, norm_params
    """
    from sklearn.model_selection import train_test_split

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )

    # 计算标准化参数（仅使用训练集）
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # 防止除零

    # 标准化
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # 创建数据集
    train_dataset = FileClassificationDataset(X_train, y_train)
    val_dataset = FileClassificationDataset(X_val, y_val)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    norm_params = {'mean': mean, 'std': std}

    return train_loader, val_loader, norm_params


def prepare_dataset(config: dict = None, use_local: bool = False, local_dirs: List[Path] = None) -> Tuple[DataLoader, DataLoader, Dict, Dict[int, str]]:
    """
    准备完整数据集（一站式函数）

    Args:
        config: 配置字典
        use_local: 使用本地文件而非下载 Govdocs
        local_dirs: 本地文件目录列表

    Returns:
        train_loader, val_loader, norm_params, label_map
    """
    if config is None:
        config = DATASET_CONFIG

    data_dirs = []

    if use_local and local_dirs:
        # 使用本地文件
        print("使用本地文件目录...")
        for d in local_dirs:
            p = Path(d)
            if p.exists():
                data_dirs.append(p)
                print(f"  添加目录: {p}")
    else:
        # 下载/检查 Govdocs 数据（函数内部会跳过已下载的）
        govdocs_dir = DATA_DIR / "govdocs"

        # 检查已有多少个解压目录
        existing_dirs = sum(1 for i in range(config["govdocs_num_zips"])
                          if (govdocs_dir / f"{i:03d}").exists())
        needed = config["govdocs_num_zips"]

        if existing_dirs < needed:
            print(f"已有 {existing_dirs}/{needed} 个数据包，继续下载...")
            download_govdocs_subset(config["govdocs_num_zips"], govdocs_dir)
        else:
            print(f"数据集完整: {existing_dirs}/{needed} 个数据包")

        # 收集所有数据目录
        for i in range(config["govdocs_num_zips"]):
            subdir = govdocs_dir / f"{i:03d}"
            if subdir.exists():
                data_dirs.append(subdir)

    if not data_dirs:
        print("\n" + "=" * 60)
        print("错误: 没有找到数据目录！")
        print("=" * 60)
        print("\n可选方案:")
        print("1. 等待自动下载 Govdocs1 数据集 (约 2.5GB)")
        print("2. 手动下载并解压到: ml/data/govdocs/")
        print("   下载地址: https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/")
        print("3. 使用本地文件测试:")
        print("   python src/train.py --local C:\\path\\to\\your\\files")
        print("=" * 60)
        raise RuntimeError("没有找到数据目录")

    # 收集文件
    type_files = collect_files(
        data_dirs,
        config["file_types"],
        config["max_samples_per_type"],
        config["min_file_size"]
    )

    # 过滤样本数过少的类别
    min_samples = config.get("min_samples_per_type", 20)  # 默认最少20个样本
    filtered_types = {}
    excluded_types = []
    for type_name, files in type_files.items():
        if len(files) >= min_samples:
            filtered_types[type_name] = files
        else:
            excluded_types.append((type_name, len(files)))

    if excluded_types:
        print(f"\n[警告] 以下类别因样本不足(<{min_samples})被排除:")
        for name, count in excluded_types:
            print(f"  {name}: {count} 个文件")

    type_files = filtered_types

    # 打印统计
    print("\n数据集统计:")
    total_files = 0
    for type_name, files in sorted(type_files.items()):
        print(f"  {type_name}: {len(files)} 个文件")
        total_files += len(files)
    print(f"  总计: {total_files} 个文件")

    # 提取特征
    features, labels, label_map = extract_dataset(
        type_files,
        config["fragment_size"],
        config["fragments_per_file"]
    )

    print(f"\n特征数据集:")
    print(f"  样本数: {len(labels)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  类别数: {len(label_map)}")

    # 创建数据加载器
    from config import TRAIN_CONFIG
    train_loader, val_loader, norm_params = create_data_loaders(
        features, labels,
        TRAIN_CONFIG["batch_size"],
        TRAIN_CONFIG["val_split"],
        TRAIN_CONFIG["random_state"]
    )

    print(f"\n数据加载器:")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")

    return train_loader, val_loader, norm_params, label_map


if __name__ == "__main__":
    # 测试数据准备
    train_loader, val_loader, norm_params, label_map = prepare_dataset()
    print("\n标签映射:", label_map)
