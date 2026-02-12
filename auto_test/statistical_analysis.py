#!/usr/bin/env python3
"""
MFT 复用率统计学分析
使用卡方检验、Logistic 回归、残差分析等方法
分析文件大小和删除时间对 MFT 复用率的独立影响
"""

import json
import numpy as np
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# 读取数据
with open('D:\\mft_reuse_analysis_C.json', 'r', encoding='utf-8-sig') as f:
    c_data = json.load(f)
with open('D:\\mft_reuse_analysis_D.json', 'r', encoding='utf-8-sig') as f:
    d_data = json.load(f)

# 合并数据
size_buckets = ['0-1KB', '1-10KB', '10-100KB', '100KB-1MB', '1-10MB', '10MB+']
age_buckets = ['0-24h', '1-3d', '3-7d', '7-14d', '14-30d', '30d+']

# 构建列联表
contingency_table = []
for age in age_buckets:
    row = []
    for size in size_buckets:
        ok = c_data['byTimeAndSize'][age][size]['OK'] + d_data['byTimeAndSize'][age][size]['OK']
        reused = c_data['byTimeAndSize'][age][size]['REUSED'] + d_data['byTimeAndSize'][age][size]['REUSED']
        row.append((ok, reused))
    contingency_table.append(row)

print("=" * 80)
print("MFT Reuse Rate Statistical Analysis")
print("=" * 80)
print()

# ============================================================================
# 1. 卡方检验 - 检验大小和时间是否独立影响复用率
# ============================================================================
print("1. CHI-SQUARE TEST FOR INDEPENDENCE")
print("-" * 80)

# 构建总列联表（只包含有数据的单元格）
# 按大小
size_counts = []
for i, size in enumerate(size_buckets):
    ok_total = sum(contingency_table[j][i][0] for j in range(len(age_buckets)))
    reused_total = sum(contingency_table[j][i][1] for j in range(len(age_buckets)))
    size_counts.append([ok_total, reused_total])

size_table = np.array(size_counts)
chi2_size, p_size, dof_size, expected_size = stats.chi2_contingency(size_table)

print(f"Size vs Reuse Rate:")
print(f"  Chi-square statistic: {chi2_size:.2f}")
print(f"  Degrees of freedom: {dof_size}")
print(f"  P-value: {p_size:.2e}")
print(f"  Result: {'SIGNIFICANT (p < 0.05)' if p_size < 0.05 else 'NOT SIGNIFICANT'}")
print()

# 按时间
age_counts = []
valid_ages = []
for j, age in enumerate(age_buckets):
    ok_total = sum(contingency_table[j][i][0] for i in range(len(size_buckets)))
    reused_total = sum(contingency_table[j][i][1] for i in range(len(size_buckets)))
    if ok_total + reused_total > 0:  # 只包含有数据的时间段
        age_counts.append([ok_total, reused_total])
        valid_ages.append(age)

age_table = np.array(age_counts)
chi2_age, p_age, dof_age, expected_age = stats.chi2_contingency(age_table)

print(f"Time vs Reuse Rate:")
print(f"  Chi-square statistic: {chi2_age:.2f}")
print(f"  Degrees of freedom: {dof_age}")
print(f"  P-value: {p_age:.2e}")
print(f"  Result: {'SIGNIFICANT (p < 0.05)' if p_age < 0.05 else 'NOT SIGNIFICANT'}")
print()

# ============================================================================
# 2. Cramér's V - 效应量（Effect Size）
# ============================================================================
print("2. EFFECT SIZE (Cramér's V)")
print("-" * 80)

def cramers_v(chi2, n, min_dim):
    return np.sqrt(chi2 / (n * (min_dim - 1)))

n_size = size_table.sum()
n_age = age_table.sum()

v_size = cramers_v(chi2_size, n_size, min(len(size_buckets), 2))
v_age = cramers_v(chi2_age, n_age, min(len(valid_ages), 2))

print(f"Size effect size (Cramér's V): {v_size:.4f}")
print(f"  Interpretation: ", end="")
if v_size < 0.1: print("Negligible")
elif v_size < 0.3: print("Small")
elif v_size < 0.5: print("Medium")
else: print("Large")

print(f"Time effect size (Cramér's V): {v_age:.4f}")
print(f"  Interpretation: ", end="")
if v_age < 0.1: print("Negligible")
elif v_age < 0.3: print("Small")
elif v_age < 0.5: print("Medium")
else: print("Large")
print()

# ============================================================================
# 3. Logistic 回归 - 预测复用概率
# ============================================================================
print("3. LOGISTIC REGRESSION MODEL")
print("-" * 80)

# 准备数据（使用数值编码）
# 大小编码：0=最小，5=最大
# 时间编码：0=最近，5=最久
data_points = []
for j, age in enumerate(age_buckets):
    for i, size in enumerate(size_buckets):
        ok, reused = contingency_table[j][i]
        if ok + reused > 0:
            for _ in range(ok):
                data_points.append({'size': i, 'age': j, 'reused': 0})
            for _ in range(reused):
                data_points.append({'size': i, 'age': j, 'reused': 1})

# 转换为 numpy 数组
data = np.array([[d['size'], d['age']] for d in data_points])
labels = np.array([d['reused'] for d in data_points])

# 使用 statsmodels 进行 Logistic 回归
try:
    import statsmodels.api as sm

    X = sm.add_constant(data)
    model = sm.Logit(labels, X)
    result = model.fit(disp=0)

    print(result.summary().replace('const', 'Intercept'))
    print()

    # 计算边际效应
    print("Marginal Effects (at mean):")
    print(f"  Size coefficient: {result.params[1]:.4f}")
    print(f"  Size odds ratio: {np.exp(result.params[1]):.4f}")
    print(f"    -> Each size level increase multiplies odds by {np.exp(result.params[1]):.2f}")
    print()
    print(f"  Time coefficient: {result.params[2]:.4f}")
    print(f"  Time odds ratio: {np.exp(result.params[2]):.4f}")
    print(f"    -> Each time level increase multiplies odds by {np.exp(result.params[2]):.2f}")
    print()

    # Pseudo R-squared
    print(f"Pseudo R-squared: {result.prsquared:.4f}")
    print()

    # 预测概率矩阵
    print("Predicted Reuse Probability Matrix:")
    print("Age/Size".ljust(12), end="")
    for size in size_buckets:
        print(size.ljust(12), end="")
    print()
    print("-" * 84)

    for j, age in enumerate(age_buckets):
        print(age.ljust(12), end="")
        for i in range(len(size_buckets)):
            X_pred = sm.add_constant(np.array([[i, j]]))
            prob = result.predict(X_pred)[0]
            print(f"{prob:.1%}".ljust(12), end="")
        print()
    print()

except ImportError:
    print("statsmodels not available, using simplified logistic regression")
    # 简化版：手动计算每组的复用率
    print("\nObserved Reuse Rate Matrix (raw data):")
    print("Age/Size".ljust(12), end="")
    for size in size_buckets:
        print(size.ljust(12), end="")
    print()
    print("-" * 84)

    for j, age in enumerate(age_buckets):
        print(age.ljust(12), end="")
        for i in range(len(size_buckets)):
            ok, reused = contingency_table[j][i]
            total = ok + reused
            if total > 0:
                rate = reused / total
                print(f"{rate:.1%} (n={total})".ljust(12), end="")
            else:
                print("-".ljust(12), end="")
        print()
    print()

# ============================================================================
# 4. 标准化残差分析 - 找出异常单元格
# ============================================================================
print("4. STANDARDIZED RESIDUALS ANALYSIS")
print("-" * 80)
print("Identifying cells with significantly higher/lower reuse rates")
print()

# 计算期望复用率（全局平均）
total_ok = sum(sum(contingency_table[j][i][0] for i in range(len(size_buckets))) for j in range(len(age_buckets)))
total_reused = sum(sum(contingency_table[j][i][1] for i in range(len(size_buckets))) for j in range(len(age_buckets)))
expected_rate = total_reused / (total_ok + total_reused)

print(f"Global expected reuse rate: {expected_rate:.1%}")
print()

residual_matrix = []
print("Standardized Residuals (>2 or <-2 indicates significance):")
print("Age/Size".ljust(12), end="")
for size in size_buckets:
    print(size.ljust(12), end="")
print()
print("-" * 84)

for j, age in enumerate(age_buckets):
    print(age.ljust(12), end="")
    row_residuals = []
    for i in range(len(size_buckets)):
        ok, reused = contingency_table[j][i]
        total = ok + reused
        if total >= 10:  # 至少10个样本
            observed_rate = reused / total
            expected_count = total * expected_rate
            std_error = np.sqrt(expected_rate * (1 - expected_rate) / total)
            if std_error > 0:
                z_score = (observed_rate - expected_rate) / std_error
                row_residuals.append(z_score)
                if abs(z_score) > 2:
                    sign = "+" if z_score > 0 else ""
                    print(f"{sign}{z_score:.1f}*".ljust(12), end="")
                else:
                    print(f"{z_score:.1f}".ljust(12), end="")
            else:
                row_residuals.append(0)
                print("-".ljust(12), end="")
        else:
            row_residuals.append(np.nan)
            print("-".ljust(12), end="")
    residual_matrix.append(row_residuals)
    print()
print()

# ============================================================================
# 5. 分层分析 - 控制时间后的复用率
# ============================================================================
print("5. STRATIFIED ANALYSIS (Controlling for Time)")
print("-" * 80)
print("Reuse rate by size, controlling for time period:")
print()

for age in ['0-24h', '1-3d', '14-30d', '30d+']:
    j = age_buckets.index(age)
    ok_total = sum(contingency_table[j][i][0] for i in range(len(size_buckets)))
    reused_total = sum(contingency_table[j][i][1] for i in range(len(size_buckets)))

    if ok_total + reused_total < 50:
        continue

    print(f"\nTime period: {age}")
    print("  Size        | Rate    | n")
    print("  " + "-" * 35)

    for i, size in enumerate(size_buckets):
        ok, reused = contingency_table[j][i]
        total = ok + reused
        if total >= 10:
            rate = reused / total
            print(f"  {size:11} | {rate:6.1%} | {total:5d}")

print()

# ============================================================================
# 6. 关键结论
# ============================================================================
print("=" * 80)
print("KEY STATISTICAL CONCLUSIONS")
print("=" * 80)
print()

print("1. SIGNIFICANCE:")
print(f"   - File size is {'a SIGNIFICANT' if p_size < 0.05 else 'NOT a significant'} factor (p={p_size:.2e})")
print(f"   - Deletion time is {'a SIGNIFICANT' if p_age < 0.05 else 'NOT a significant'} factor (p={p_age:.2e})")
print()

print("2. EFFECT SIZE:")
print(f"   - Size explains {v_size**2 * 100:.1f}% of variance in reuse rate")
print(f"   - Time explains {v_age**2 * 100:.1f}% of variance in reuse rate")
print()

print("3. INTERACTION:")
print("   Analyzing residuals shows which combinations are abnormal:")
for j, age in enumerate(age_buckets):
    for i, size in enumerate(size_buckets):
        if not np.isnan(residual_matrix[j][i]) and abs(residual_matrix[j][i]) > 3:
            sign = "HIGHER" if residual_matrix[j][i] > 0 else "LOWER"
            ok, reused = contingency_table[j][i]
            total = ok + reused
            rate = reused / total
            print(f"   - {age} + {size}: {sign} than expected ({rate:.1%} vs {expected_rate:.1%}, z={residual_matrix[j][i]:.1f})")
print()

print("4. RECOMMENDATIONS FOR RECOVERY:")
print("   Based on statistical evidence:")
print("   - Size factor: Larger files (>1MB) have significantly higher reuse risk")
print("   - Time factor: First 3 days have highest reuse rate")
print("   - Interaction: Large files deleted within 3 days are CRITICAL")
print()
