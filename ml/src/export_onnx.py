"""
ONNX 导出工具 - 独立的模型导出脚本
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import CHECKPOINTS_DIR, EXPORTED_DIR, CLASSIFICATION_MODELS_DIR, MODEL_CONFIG, DEVICE
from model import create_model


def export_checkpoint_to_onnx(
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    optimize: bool = True,
) -> Path:
    """
    从检查点导出 ONNX 模型

    Args:
        checkpoint_path: 检查点文件路径
        output_path: 输出路径（默认自动生成）
        optimize: 是否优化 ONNX 模型

    Returns:
        导出的 ONNX 文件路径
    """
    print(f"加载检查点: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 获取配置
    model_config = checkpoint.get("model_config", MODEL_CONFIG)
    label_map = checkpoint["label_map"]
    norm_params = checkpoint["norm_params"]

    input_dim = model_config["input_dim"]
    num_classes = len(label_map)

    # 重建模型
    model = create_model(
        model_type="deep",  # 默认使用 deep 模型
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=model_config.get("hidden_dims", [512, 256, 128]),
        dropout=model_config.get("dropout", 0.4),
        use_batch_norm=model_config.get("use_batch_norm", True),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 设置输出路径
    if output_path is None:
        output_path = CLASSIFICATION_MODELS_DIR / f"{checkpoint_path.stem}.onnx"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建示例输入
    dummy_input = torch.randn(1, input_dim)

    # 导出 ONNX
    print(f"导出 ONNX 模型: {output_path}")
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

    # 优化 ONNX（如果安装了 onnx）
    if optimize:
        try:
            import onnx
            from onnx import optimizer

            print("优化 ONNX 模型...")
            onnx_model = onnx.load(str(output_path))
            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "fuse_consecutive_transposes",
                "fuse_transpose_into_gemm",
            ]
            optimized_model = optimizer.optimize(onnx_model, passes)
            onnx.save(optimized_model, str(output_path))
        except ImportError:
            print("onnx 优化器未安装，跳过优化")
        except Exception as e:
            print(f"优化失败: {e}")

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过")
    except ImportError:
        print("onnx 未安装，跳过验证")
    except Exception as e:
        print(f"ONNX 验证警告: {e}")

    # 保存元数据
    metadata = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "label_map": {str(k): v for k, v in label_map.items()},
        "norm_params": {
            "mean": norm_params["mean"] if isinstance(norm_params["mean"], list) else norm_params["mean"].tolist(),
            "std": norm_params["std"] if isinstance(norm_params["std"], list) else norm_params["std"].tolist(),
        },
        "model_config": model_config,
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"元数据已保存: {metadata_path}")

    # 打印模型信息
    file_size = output_path.stat().st_size / 1024
    print(f"\n导出完成:")
    print(f"  模型大小: {file_size:.1f} KB")
    print(f"  输入维度: {input_dim}")
    print(f"  输出类别: {num_classes}")
    print(f"  标签: {list(label_map.values())}")

    return output_path


def verify_onnx_inference(
    onnx_path: Path,
    test_input: Optional[np.ndarray] = None,
) -> Dict:
    """
    验证 ONNX 模型推理

    Args:
        onnx_path: ONNX 模型路径
        test_input: 测试输入（可选）

    Returns:
        推理结果
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime 未安装，无法验证推理")
        return {}

    # 加载元数据
    metadata_path = onnx_path.with_suffix(".json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    input_dim = metadata["input_dim"]
    label_map = {int(k): v for k, v in metadata["label_map"].items()}
    norm_mean = np.array(metadata["norm_params"]["mean"], dtype=np.float32)
    norm_std = np.array(metadata["norm_params"]["std"], dtype=np.float32)

    # 创建会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    print(f"使用执行提供者: {session.get_providers()}")

    # 创建测试输入
    if test_input is None:
        test_input = np.random.randn(1, input_dim).astype(np.float32)
        # 标准化
        test_input = (test_input - norm_mean) / norm_std

    # 推理
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: test_input})
    logits = outputs[0]

    # 计算概率
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_class = np.argmax(probs, axis=1)[0]
    pred_prob = probs[0, pred_class]

    result = {
        "predicted_class": pred_class,
        "predicted_label": label_map[pred_class],
        "confidence": float(pred_prob),
        "all_probabilities": {label_map[i]: float(probs[0, i]) for i in range(len(label_map))},
    }

    print(f"\n推理测试结果:")
    print(f"  预测类别: {result['predicted_label']}")
    print(f"  置信度: {result['confidence']*100:.2f}%")

    return result


def test_with_real_file(
    onnx_path: Path,
    file_path: Path,
) -> Dict:
    """
    使用真实文件测试 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
        file_path: 测试文件路径

    Returns:
        预测结果
    """
    from dataset import FileFeatureExtractor

    # 加载元数据
    metadata_path = onnx_path.with_suffix(".json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    norm_mean = np.array(metadata["norm_params"]["mean"], dtype=np.float32)
    norm_std = np.array(metadata["norm_params"]["std"], dtype=np.float32)
    label_map = {int(k): v for k, v in metadata["label_map"].items()}

    # 提取特征
    extractor = FileFeatureExtractor()
    features_list = extractor.extract_from_file(str(file_path))

    if not features_list:
        print(f"无法从文件提取特征: {file_path}")
        return {}

    # 使用第一个片段的特征
    features = features_list[0].reshape(1, -1).astype(np.float32)

    # 标准化
    features = (features - norm_mean) / norm_std

    # 推理
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime 未安装")
        return {}

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: features})
    logits = outputs[0]

    # 计算概率
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_class = np.argmax(probs, axis=1)[0]

    result = {
        "file": str(file_path),
        "predicted_label": label_map[pred_class],
        "confidence": float(probs[0, pred_class]),
        "probabilities": {label_map[i]: float(probs[0, i]) for i in range(len(label_map))},
    }

    print(f"\n文件分类结果: {file_path.name}")
    print(f"  预测类型: {result['predicted_label']}")
    print(f"  置信度: {result['confidence']*100:.2f}%")
    print("  各类别概率:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1])[:5]:
        print(f"    {label}: {prob*100:.2f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="ONNX 模型导出工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 导出命令
    export_parser = subparsers.add_parser("export", help="从检查点导出 ONNX")
    export_parser.add_argument("checkpoint", type=str, help="检查点文件路径")
    export_parser.add_argument("-o", "--output", type=str, default=None, help="输出路径")
    export_parser.add_argument("--no-optimize", action="store_true", help="不优化模型")

    # 验证命令
    verify_parser = subparsers.add_parser("verify", help="验证 ONNX 模型")
    verify_parser.add_argument("onnx", type=str, help="ONNX 模型路径")

    # 测试命令
    test_parser = subparsers.add_parser("test", help="使用文件测试模型")
    test_parser.add_argument("onnx", type=str, help="ONNX 模型路径")
    test_parser.add_argument("file", type=str, help="测试文件路径")

    # 列出命令
    list_parser = subparsers.add_parser("list", help="列出可用的检查点")

    args = parser.parse_args()

    if args.command == "export":
        checkpoint_path = Path(args.checkpoint)
        output_path = Path(args.output) if args.output else None
        export_checkpoint_to_onnx(checkpoint_path, output_path, optimize=not args.no_optimize)

    elif args.command == "verify":
        onnx_path = Path(args.onnx)
        verify_onnx_inference(onnx_path)

    elif args.command == "test":
        onnx_path = Path(args.onnx)
        file_path = Path(args.file)
        test_with_real_file(onnx_path, file_path)

    elif args.command == "list":
        print("可用的检查点:")
        for pt_file in CHECKPOINTS_DIR.glob("*.pt"):
            size = pt_file.stat().st_size / 1024 / 1024
            print(f"  {pt_file.name} ({size:.1f} MB)")

        print("\n已导出的 ONNX 模型:")
        for onnx_file in EXPORTED_DIR.glob("*.onnx"):
            size = onnx_file.stat().st_size / 1024
            print(f"  {onnx_file.name} ({size:.1f} KB)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
