#!/usr/bin/env python3
"""
可行性诊断脚本 - 判断是方法问题还是数据量问题

通过数学分析回答：
1. 特征是否有区分能力？（可分离性分析）
2. 模型是否在学习？（学习曲线）
3. 理论上限是多少？（贝叶斯错误率）
4. 需要多少数据？（数据效率分析）
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import sys
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from dataset import ContinuityDataset
from model_cnn import get_cnn_model


def analyze_feature_separability(csv_path: str) -> Dict:
    """
    分析1: 特征可分离性

    判断标准:
    - Fisher判别比 > 1.0: 特征有区分能力
    - t-test p-value < 0.05: 统计显著
    - LDA准确率 > 70%: 线性可分

    结论:
    - 如果LDA > 85%: 特征好，增加数据有用
    - 如果LDA < 65%: 特征差，增加数据无用
    """
    print("\n" + "="*60)
    print("分析1: 特征可分离性（Feature Separability）")
    print("="*60)

    df = pd.read_csv(csv_path)
    feature_cols = [f"f{i}" for i in range(64)]
    X = df[feature_cols].values
    y = df['is_continuous'].values

    X_pos = X[y == 1]
    X_neg = X[y == 0]

    # 1. Fisher判别比 (Between-class variance / Within-class variance)
    mean_pos = X_pos.mean(axis=0)
    mean_neg = X_neg.mean(axis=0)
    mean_all = X.mean(axis=0)

    # Between-class scatter
    S_B = len(X_pos) * np.outer(mean_pos - mean_all, mean_pos - mean_all) + \
          len(X_neg) * np.outer(mean_neg - mean_all, mean_neg - mean_all)

    # Within-class scatter
    S_W = np.cov(X_pos.T) * len(X_pos) + np.cov(X_neg.T) * len(X_neg)

    # Fisher判别比 (取特征值的迹作为度量)
    fisher_ratio = np.trace(S_B) / (np.trace(S_W) + 1e-8)

    print(f"\nFisher判别比: {fisher_ratio:.4f}")
    if fisher_ratio > 1.0:
        print("  ✓ Fisher比 > 1.0: 特征有区分能力")
    else:
        print("  ✗ Fisher比 < 1.0: 特征区分能力极弱")

    # 2. 逐特征t-test
    significant_features = 0
    for i in range(64):
        t_stat, p_value = stats.ttest_ind(X_pos[:, i], X_neg[:, i])
        if p_value < 0.05:
            significant_features += 1

    print(f"\n统计显著特征数: {significant_features}/64 (p < 0.05)")
    if significant_features > 30:
        print(f"  ✓ 超过一半特征显著: 特征有效")
    else:
        print(f"  ✗ 少于一半特征显著: 特征设计可能有问题")

    # 3. 线性判别分析 (LDA)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    lda_score = lda.score(X, y)

    print(f"\nLDA准确率（线性可分性）: {lda_score:.2%}")
    if lda_score > 0.85:
        print("  ✓ LDA > 85%: 特征线性可分，增加数据会有效")
    elif lda_score > 0.70:
        print("  ⚠ LDA 70-85%: 特征有一定区分能力，可能需要非线性模型")
    else:
        print("  ✗ LDA < 70%: 特征线性不可分，增加数据帮助有限")

    # 4. Logistic Regression (简单非线性)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X[:, :10])  # 只用前10个特征避免维度爆炸

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_poly, y)
    lr_score = lr.score(X_poly, y)

    print(f"\n多项式Logistic准确率: {lr_score:.2%}")
    if lr_score - lda_score > 0.1:
        print("  → 非线性特征组合有帮助，CNN可能有效")
    else:
        print("  → 非线性提升有限，问题可能在特征本身")

    return {
        "fisher_ratio": float(fisher_ratio),
        "significant_features": int(significant_features),
        "lda_accuracy": float(lda_score),
        "lr_accuracy": float(lr_score),
        "separable": lda_score > 0.70
    }


def analyze_learning_curve(csv_path: str, checkpoint_path: str) -> Dict:
    """
    分析2: 学习曲线

    判断标准:
    - Training acc >> Test acc: 过拟合，需要正则化
    - Training acc ≈ Test acc 且都低: 欠拟合，特征不足
    - 曲线上升: 增加数据有用
    - 曲线平坦: 增加数据无用
    """
    print("\n" + "="*60)
    print("分析2: 学习曲线（Learning Curve）")
    print("="*60)

    df = pd.read_csv(csv_path)
    feature_cols = [f"f{i}" for i in range(64)]
    X = df[feature_cols].values
    y = df['is_continuous'].values

    # 使用简单的Logistic Regression画学习曲线
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=1000, random_state=42),
        X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    print(f"\n数据量 vs 性能:")
    for i, size in enumerate(train_sizes_abs):
        print(f"  {size:6d} samples: train F1={train_mean[i]:.2%}, val F1={val_mean[i]:.2%}")

    # 判断趋势
    gap = train_mean[-1] - val_mean[-1]
    slope = (val_mean[-1] - val_mean[0]) / (train_sizes_abs[-1] - train_sizes_abs[0])

    print(f"\n训练-验证差距: {gap:.2%}")
    if gap > 0.15:
        print("  ✗ 过拟合严重: 模型记忆训练集，泛化能力差")
    elif gap > 0.05:
        print("  ⚠ 轻微过拟合: 可能需要正则化")
    else:
        print("  ✓ 拟合良好: 训练验证接近")

    print(f"\n学习曲线斜率: {slope:.6f}")
    if slope > 0.00001:
        print("  ✓ 曲线上升: 增加数据会提升性能")
    else:
        print("  ✗ 曲线平坦: 增加数据帮助有限")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', label='Training F1', linewidth=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes_abs, val_mean, 'o-', label='Validation F1', linewidth=2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_results/learning_curve.png', dpi=150)
    print("\n学习曲线已保存到: experiment_results/learning_curve.png")

    return {
        "train_val_gap": float(gap),
        "learning_curve_slope": float(slope),
        "final_val_f1": float(val_mean[-1]),
        "data_helps": slope > 0.00001
    }


def estimate_bayes_error(csv_path: str) -> Dict:
    """
    分析3: 贝叶斯错误率估计

    理论上界: 即使完美模型也无法突破的准确率

    方法: k-NN估计（k=5）
    - k-NN接近100%: 数据可分，问题在模型
    - k-NN < 80%: 数据本身有噪声或标注错误
    """
    print("\n" + "="*60)
    print("分析3: 贝叶斯错误率（理论上界）")
    print("="*60)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    df = pd.read_csv(csv_path)
    feature_cols = [f"f{i}" for i in range(64)]
    X = df[feature_cols].values
    y = df['is_continuous'].values

    # 使用k-NN估计贝叶斯错误率
    knn_scores = []
    for k in [1, 3, 5, 10, 20]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='f1', n_jobs=-1)
        knn_scores.append((k, scores.mean()))
        print(f"  k-NN (k={k:2d}): F1 = {scores.mean():.2%}")

    best_knn = max(knn_scores, key=lambda x: x[1])
    print(f"\n最佳k-NN性能: {best_knn[1]:.2%} (k={best_knn[0]})")

    if best_knn[1] > 0.90:
        print("  ✓ k-NN > 90%: 数据高度可分，理论上限高")
        print("    → 当前CNN性能低是模型问题，可以改进")
    elif best_knn[1] > 0.75:
        print("  ⚠ k-NN 75-90%: 数据中等可分")
        print("    → 有改进空间，但上限有限")
    else:
        print("  ✗ k-NN < 75%: 数据不可分或标注有问题")
        print("    → 即使完美模型也难以超过此性能")

    return {
        "best_knn_f1": float(best_knn[1]),
        "best_k": int(best_knn[0]),
        "theoretical_limit": float(best_knn[1])
    }


def analyze_data_quality(csv_path: str) -> Dict:
    """
    分析4: 数据质量检查

    - 标注一致性
    - 样本多样性
    - 类别平衡
    """
    print("\n" + "="*60)
    print("分析4: 数据质量")
    print("="*60)

    df = pd.read_csv(csv_path)
    feature_cols = [f"f{i}" for i in range(64)]
    X = df[feature_cols].values
    y = df['is_continuous'].values

    # 1. 类别平衡
    pos_ratio = y.mean()
    print(f"\n类别平衡: {pos_ratio:.2%} positive")
    if 0.4 < pos_ratio < 0.6:
        print("  ✓ 平衡良好")
    else:
        print("  ⚠ 类别不平衡，可能影响训练")

    # 2. 特征方差
    feature_std = X.std(axis=0)
    zero_variance = (feature_std < 1e-6).sum()
    print(f"\n零方差特征: {zero_variance}/64")
    if zero_variance > 0:
        print(f"  ✗ {zero_variance}个特征无变化，需要移除")

    # 3. 样本相似度（检测重复）
    from sklearn.metrics.pairwise import cosine_similarity
    sample_indices = np.random.choice(len(X), min(1000, len(X)), replace=False)
    X_sample = X[sample_indices]
    sim_matrix = cosine_similarity(X_sample)
    np.fill_diagonal(sim_matrix, 0)

    high_sim = (sim_matrix > 0.99).sum() / 2  # 除以2因为对称
    print(f"\n高相似样本对 (sim > 0.99): {high_sim}")
    if high_sim > len(X_sample) * 0.1:
        print("  ⚠ 可能有大量重复样本")

    # 4. 检查明显错误的标注
    # 如果两个样本几乎完全相同但标签不同，可能是标注错误
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    mislabeled = 0
    for i in range(len(X)):
        nearest_idx = indices[i, 1]  # 最近邻（排除自己）
        if distances[i, 1] < 0.01 and y[i] != y[nearest_idx]:
            mislabeled += 1

    print(f"\n可疑标注 (最近邻相似但标签不同): {mislabeled}")
    if mislabeled > len(X) * 0.05:
        print("  ✗ 可能有大量标注错误")

    return {
        "class_balance": float(pos_ratio),
        "zero_variance_features": int(zero_variance),
        "high_similarity_pairs": int(high_sim),
        "suspicious_labels": int(mislabeled)
    }


def generate_final_report(results: Dict) -> str:
    """生成最终诊断报告"""

    report = "\n" + "="*60
    report += "\n最终诊断报告\n"
    report += "="*60 + "\n"

    # 特征可分离性
    sep = results['separability']
    report += f"\n【特征可分离性】\n"
    report += f"  Fisher判别比: {sep['fisher_ratio']:.4f}\n"
    report += f"  显著特征数: {sep['significant_features']}/64\n"
    report += f"  LDA准确率: {sep['lda_accuracy']:.2%}\n"
    report += f"  Logistic准确率: {sep['lr_accuracy']:.2%}\n"

    # 学习曲线
    lc = results['learning_curve']
    report += f"\n【学习曲线】\n"
    report += f"  训练-验证差距: {lc['train_val_gap']:.2%}\n"
    report += f"  曲线斜率: {lc['learning_curve_slope']:.6f}\n"
    report += f"  最终验证F1: {lc['final_val_f1']:.2%}\n"

    # 理论上界
    bayes = results['bayes_error']
    report += f"\n【理论上界】\n"
    report += f"  最佳k-NN F1: {bayes['best_knn_f1']:.2%}\n"
    report += f"  理论极限: {bayes['theoretical_limit']:.2%}\n"

    # 数据质量
    quality = results['data_quality']
    report += f"\n【数据质量】\n"
    report += f"  类别平衡: {quality['class_balance']:.2%}\n"
    report += f"  零方差特征: {quality['zero_variance_features']}\n"
    report += f"  可疑标注: {quality['suspicious_labels']}\n"

    # 最终结论
    report += "\n" + "="*60
    report += "\n最终结论\n"
    report += "="*60 + "\n"

    # 判断逻辑
    if bayes['best_knn_f1'] > 0.85 and sep['lda_accuracy'] > 0.75:
        conclusion = "✓ 方法可行，增加数据有用"
        recommendation = """
建议行动:
1. 增加训练数据到10万+样本
2. 保持当前特征和模型架构
3. 可以考虑集成学习进一步提升
4. 理论上限约 {:.0%}，有很大改进空间
        """.format(bayes['theoretical_limit'])

    elif bayes['best_knn_f1'] > 0.75 and lc['data_helps']:
        conclusion = "⚠ 方法有一定可行性，但上限有限"
        recommendation = """
建议行动:
1. 可以尝试增加数据，但收益可能有限
2. 重点改进特征工程
3. 尝试特征选择，去除无效特征
4. 理论上限约 {:.0%}，改进空间中等
        """.format(bayes['theoretical_limit'])

    else:
        conclusion = "✗ 当前方法不可行"
        recommendation = """
建议行动:
1. 停止增加数据（不会有明显改善）
2. 重新设计特征提取算法
3. 检查数据标注是否正确
4. 考虑其他方法（如传统签名验证）
5. 或将此作为negative result发表
        """

    report += f"\n{conclusion}\n"
    report += recommendation

    # 数学依据
    report += "\n" + "="*60
    report += "\n数学依据\n"
    report += "="*60 + "\n"
    report += f"""
1. Fisher判别分析: {sep['fisher_ratio']:.4f}
   - > 1.0 表示类间差异 > 类内差异
   - 你的数据: {'合格' if sep['fisher_ratio'] > 1.0 else '不合格'}

2. 线性可分性 (LDA): {sep['lda_accuracy']:.2%}
   - > 85% 表示特征线性可分，深度学习有效
   - 你的数据: {'合格' if sep['lda_accuracy'] > 0.75 else '不合格'}

3. 理论上界 (k-NN): {bayes['best_knn_f1']:.2%}
   - 即使完美模型也难以超过此值
   - 当前CNN ({lc['final_val_f1']:.2%}) vs 理论极限: 差距 {bayes['best_knn_f1'] - lc['final_val_f1']:.2%}

4. 学习曲线斜率: {lc['learning_curve_slope']:.6f}
   - > 0 表示增加数据有用
   - 你的数据: {'有用' if lc['data_helps'] else '无用'}
"""

    return report


def main():
    results_dir = Path("./experiment_results")

    # 检查文件
    train_csv = results_dir / "dataset_train.csv"
    test_csv = results_dir / "dataset_test.csv"
    checkpoint = results_dir / "best_model.pt"

    if not train_csv.exists():
        print("错误: 找不到 dataset_train.csv")
        print("请先运行: python validate_continuity.py run")
        return

    results = {}

    # 分析1: 特征可分离性
    results['separability'] = analyze_feature_separability(str(train_csv))

    # 分析2: 学习曲线
    results['learning_curve'] = analyze_learning_curve(str(train_csv), str(checkpoint))

    # 分析3: 贝叶斯错误率
    results['bayes_error'] = estimate_bayes_error(str(train_csv))

    # 分析4: 数据质量
    results['data_quality'] = analyze_data_quality(str(train_csv))

    # 生成报告
    report = generate_final_report(results)
    print(report)

    # 保存结果
    with open(results_dir / "feasibility_diagnosis.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(results_dir / "feasibility_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n诊断结果已保存:")
    print(f"  - {results_dir / 'feasibility_diagnosis.json'}")
    print(f"  - {results_dir / 'feasibility_report.txt'}")
    print(f"  - {results_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
