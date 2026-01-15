"""
图像头部修复训练数据生成器

从大量正常的 JPEG/PNG 图像生成训练数据：
- 提取图像主体特征
- 保存对应的文件头部
- 用于训练头部重建模型
"""

import os
import glob
import numpy as np
import struct
from pathlib import Path
from PIL import Image
import json

class ImageHeaderTrainingDataGenerator:
    def __init__(self, output_dir="training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/jpeg", exist_ok=True)
        os.makedirs(f"{output_dir}/png", exist_ok=True)

    def extract_jpeg_features(self, file_path):
        """从 JPEG 文件提取特征"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            if len(data) < 100:
                return None

            # 查找 SOF (Start of Frame) 标记
            sof_pos = self.find_jpeg_marker(data, 0xC0)
            if sof_pos == -1:
                sof_pos = self.find_jpeg_marker(data, 0xC2)

            if sof_pos == -1 or sof_pos < 20:
                return None

            # 提取头部（SOI 到 SOF 之前）
            header = data[:sof_pos]

            # 提取主体数据特征（从 SOF 开始的 4KB）
            body_start = sof_pos
            body_data = data[body_start:body_start + 4096]

            if len(body_data) < 1000:
                return None

            features = self.extract_statistical_features(body_data)

            # 尝试获取图像尺寸
            try:
                img = Image.open(file_path)
                width, height = img.size
                features['width'] = width
                features['height'] = height
            except:
                features['width'] = 0
                features['height'] = 0

            return {
                'header': header,
                'features': features,
                'file_type': 'jpeg',
                'header_size': len(header),
                'file_size': len(data)
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def extract_png_features(self, file_path):
        """从 PNG 文件提取特征"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            if len(data) < 100:
                return None

            # PNG 签名 + IHDR chunk
            if data[:8] != b'\x89PNG\r\n\x1a\n':
                return None

            # 查找 IDAT chunk
            idat_pos = self.find_png_chunk(data, b'IDAT')
            if idat_pos == -1:
                return None

            # 提取头部（签名 + IHDR + 其他辅助 chunks）
            header = data[:idat_pos]

            # 提取主体数据特征
            body_data = data[idat_pos:idat_pos + 4096]

            if len(body_data) < 1000:
                return None

            features = self.extract_statistical_features(body_data)

            # 提取 IHDR 信息
            try:
                width = struct.unpack('>I', data[16:20])[0]
                height = struct.unpack('>I', data[20:24])[0]
                bit_depth = data[24]
                color_type = data[25]

                features['width'] = width
                features['height'] = height
                features['bit_depth'] = bit_depth
                features['color_type'] = color_type
            except:
                features['width'] = 0
                features['height'] = 0

            return {
                'header': header,
                'features': features,
                'file_type': 'png',
                'header_size': len(header),
                'file_size': len(data)
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def extract_statistical_features(self, data):
        """提取数据统计特征"""
        arr = np.frombuffer(data, dtype=np.uint8)

        features = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'entropy': float(self.calculate_entropy(arr)),
            'zero_count': int(np.sum(arr == 0)),
            'ff_count': int(np.sum(arr == 255))
        }

        # 字节分布（16个区间）
        hist, _ = np.histogram(arr, bins=16, range=(0, 256))
        features['histogram'] = hist.tolist()

        return features

    def calculate_entropy(self, data):
        """计算熵"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        prob = hist / len(data)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    def find_jpeg_marker(self, data, marker):
        """查找 JPEG 标记"""
        for i in range(len(data) - 1):
            if data[i] == 0xFF and data[i+1] == marker:
                return i
        return -1

    def find_png_chunk(self, data, chunk_type):
        """查找 PNG chunk"""
        for i in range(8, len(data) - 12):
            if data[i+4:i+8] == chunk_type:
                return i
        return -1

    def process_directory(self, input_dir, file_type='jpeg', max_samples=10000):
        """处理目录中的所有图像"""
        patterns = {
            'jpeg': ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG'],
            'png': ['*.png', '*.PNG']
        }

        samples = []
        count = 0

        for pattern in patterns.get(file_type, []):
            for file_path in glob.glob(os.path.join(input_dir, '**', pattern), recursive=True):
                if count >= max_samples:
                    break

                if file_type == 'jpeg':
                    result = self.extract_jpeg_features(file_path)
                else:
                    result = self.extract_png_features(file_path)

                if result:
                    # 保存样本
                    sample_file = f"{self.output_dir}/{file_type}/sample_{count:06d}.npz"
                    np.savez_compressed(
                        sample_file,
                        header=np.frombuffer(result['header'], dtype=np.uint8),
                        features=result['features'],
                        header_size=result['header_size'],
                        file_size=result['file_size']
                    )

                    samples.append({
                        'sample_id': count,
                        'source_file': file_path,
                        'header_size': result['header_size'],
                        'file_size': result['file_size']
                    })

                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} {file_type} files...")

        # 保存元数据
        with open(f"{self.output_dir}/{file_type}/metadata.json", 'w') as f:
            json.dump({
                'total_samples': count,
                'file_type': file_type,
                'samples': samples
            }, f, indent=2)

        print(f"Total {file_type} samples: {count}")
        return count

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate training data for image header repair')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('--output', default='training_data', help='Output directory')
    parser.add_argument('--type', choices=['jpeg', 'png', 'both'], default='both',
                       help='Image type to process')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum samples per type')

    args = parser.parse_args()

    generator = ImageHeaderTrainingDataGenerator(args.output)

    if args.type in ['jpeg', 'both']:
        print(f"\nProcessing JPEG files from {args.input_dir}...")
        generator.process_directory(args.input_dir, 'jpeg', args.max_samples)

    if args.type in ['png', 'both']:
        print(f"\nProcessing PNG files from {args.input_dir}...")
        generator.process_directory(args.input_dir, 'png', args.max_samples)

    print("\nTraining data generation complete!")
    print(f"Output directory: {args.output}")

if __name__ == '__main__':
    main()
