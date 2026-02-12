#!/usr/bin/env python3
"""
MFT 复用率加权统计学分析
考虑样本量不均衡和幸存者偏差
"""

import json
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 读取数据
with open('D:\\mft_reuse_analysis_C.json', 'r', encoding='utf-8-sig') as f:
    c_data = json.load(f)
with open('D:\\mft_reuse_analysis_D.json', 'r', encoding='utf-8-sig') as f:
    d_data = json.load(f)

size_buckets = ['0-1KB', '1-10KB', '10-100KB', '100KB-1MB', '1-10MB', '10MB+']
age_buckets = ['0-24h', '1-3d', '3-7d', '7-14d', '14-30d', '30d+']

# 构建数据矩阵
contingency_table = []
for age in age_buckets:
    row = []
    for size in size_buckets:
        ok = c_data['byTimeAndSize'][age][size]['OK'] + d_data['byTimeAndSize'][age][size]['OK']
        reused = c_data['byTimeAndSize'][age][size]['REUSED'] + d_data['byTimeAndSize'][age][size]['REUSED']
        row.append((ok, reused))
    contingency_table.append(row)

print("=" * 90)
print("MFT REUSE RATE ANALYSIS - WEIGHTED AND SAMPLE BIAS CORRECTED")
print("=" * 90)
print()

# ============================================================================
# 1. 样本量分布分析
# ============================================================================
print("1. SAMPLE DISTRIBUTION ANALYSIS")
print("-" * 90)

total_samples = sum(
    sum(contingency_table[j][i][0] + contingency_table[j][i][1]
        for i in range(len(size_buckets)))
    for j in range(len(age_buckets))
)

print(f"Total samples: {total_samples:,}")
print()

# 按时间段统计
print("By Time Period:")
print(f"{'Age':<12} {'Samples':>8} {'% of Total':>12} {'Valid Cells':>12} {'Notes':<30}")
print("-" * 90)

age_stats = []
for j, age in enumerate(age_buckets):
    samples = sum(contingency_table[j][i][0] + contingency_table[j][i][1]
                  for i in range(len(size_buckets)))
    valid_cells = sum(1 for i in range(len(size_buckets))
                     if contingency_table[j][i][0] + contingency_table[j][i][1] > 0)
    pct = samples / total_samples * 100

    notes = ""
    if samples == 0:
        notes = "NO DATA (USN limit)"
    elif pct < 5:
        notes = "LOW SAMPLE (<5%)"
    elif pct > 30:
        notes = "HIGH SAMPLE (>30%)"

    age_stats.append({
        'age': age,
        'samples': samples,
        'pct': pct,
        'valid_cells': valid_cells
    })

    print(f"{age:<12} {samples:>8,} {pct:>11.1f}% {valid_cells:>12} {notes:<30}")

print()

# 按大小统计
print("By Size Category:")
print(f"{'Size':<12} {'Samples':>8} {'% of Total':>12}")
print("-" * 40)

size_stats = []
for i, size in enumerate(size_buckets):
    samples = sum(contingency_table[j][i][0] + contingency_table[j][i][1]
                  for j in range(len(age_buckets)))
    pct = samples / total_samples * 100
    size_stats.append({'size': size, 'samples': samples, 'pct': pct})
    print(f"{size:<12} {samples:>8,} {pct:>11.1f}%")

print()

# ============================================================================
# 2. 样本选择偏差检测
# ============================================================================
print("2. SAMPLE SELECTION BIAS DETECTION")
print("-" * 90)

# 检查 USN 日志的时间范围限制
print("USN Journal Coverage Check:")
print("  - USN 日志有固定大小限制，旧记录会被覆盖")
print("  - 3-7d 和 7-14d 数据缺失可能是 USN 滚动覆盖导致")
print()

# 计算期望分布（假设均匀删除）
print("Expected vs Observed Distribution:")
print(f"{'Age':<12} {'Expected %':>12} {'Observed %':>12} {'Deviation':>12}")
print("-" * 60)

expected_pct = 100 / len([a for a in age_stats if a['samples'] > 0])  # 只计算有数据的时间段
cumulative_deviation = 0

for stat in age_stats:
    if stat['samples'] > 0:
        deviation = stat['pct'] - expected_pct
        cumulative_deviation += abs(deviation)
        print(f"{stat['age']:<12} {expected_pct:>11.1f}% {stat['pct']:>11.1f}% {deviation:>+11.1f}%")

print()
print(f"Total deviation: {cumulative_deviation:.1f}%")
if cumulative_deviation > 50:
    print("  WARNING: HIGH sample imbalance detected - weighted analysis required")
else:
    print("  OK: Sample distribution is relatively balanced")
print()

# ============================================================================
# 3. 加权复用率分析（消除样本量偏差）
# ============================================================================
print("3. WEIGHTED REUSE RATE ANALYSIS")
print("-" * 90)
print("Calculating weighted averages to eliminate sample size bias")
print()

# 方法 1: 按大小加权（每个时间段权重相等）
print("Method 1: Size Effect (Controlling for Time)")
print("Weighted by inverse of time period sample count")
print()

size_weighted_rates = []
for i, size in enumerate(size_buckets):
    weighted_sum = 0
    weight_total = 0

    for j, age in enumerate(age_buckets):
        ok, reused = contingency_table[j][i]
        total = ok + reused

        if total >= 10:  # 至少10个样本
            rate = reused / total
            # 权重 = 1 / 该时间段的样本总数（给予小样本时段更高权重）
            age_total = sum(contingency_table[j][k][0] + contingency_table[j][k][1]
                          for k in range(len(size_buckets)))
            if age_total > 0:
                weight = 1 / age_total
                weighted_sum += rate * weight
                weight_total += weight

    if weight_total > 0:
        weighted_rate = weighted_sum / weight_total
        size_weighted_rates.append((size, weighted_rate))
        print(f"  {size:<12}: {weighted_rate:>6.1%}")

print()

# 方法 2: 按时间加权（每个大小类别权重相等）
print("Method 2: Time Effect (Controlling for Size)")
print("Weighted by inverse of size category sample count")
print()

age_weighted_rates = []
for j, age in enumerate(age_buckets):
    weighted_sum = 0
    weight_total = 0

    for i, size in enumerate(size_buckets):
        ok, reused = contingency_table[j][i]
        total = ok + reused

        if total >= 10:
            rate = reused / total
            # 权重 = 1 / 该大小类别的样本总数
            size_total = sum(contingency_table[k][i][0] + contingency_table[k][i][1]
                           for k in range(len(age_buckets)))
            if size_total > 0:
                weight = 1 / size_total
                weighted_sum += rate * weight
                weight_total += weight

    if weight_total > 0:
        weighted_rate = weighted_sum / weight_total
        age_weighted_rates.append((age, weighted_rate))
        print(f"  {age:<12}: {weighted_rate:>6.1%}")

print()

# ============================================================================
# 4. 幸存者偏差分析
# ============================================================================
print("4. SURVIVOR BIAS ANALYSIS")
print("-" * 90)
print()

# 计算"幸存者"比例（未复用的比例）
print("Survival Rate by Time (NOT reused = survived):")
print(f"{'Age':<12} {'Survived':>10} {'Total':>10} {'Survival %':>12} {'Interpretation':<30}")
print("-" * 90)

for j, age in enumerate(age_buckets):
    ok_total = sum(contingency_table[j][i][0] for i in range(len(size_buckets)))
    reused_total = sum(contingency_table[j][i][1] for i in range(len(size_buckets)))
    total = ok_total + reused_total

    if total > 0:
        survival_rate = ok_total / total

        if age in ['3-7d', '7-14d']:
            interp = "N/A (no data)"
        elif survival_rate > 0.7:
            interp = "HIGH survival (biased sample)"
        elif survival_rate > 0.5:
            interp = "Moderate survival"
        else:
            interp = "Normal turnover"

        print(f"{age:<12} {ok_total:>10,} {total:>10,} {survival_rate:>11.1%} {interp:<30}")

print()

print("Interpretation:")
print("  - 30d+ 的高存活率 (87.7%) 是幸存者偏差")
print("  - 这些文件之所以还在 USN 日志中，正是因为它们的 MFT 没被复用")
print("  - MFT 已复用的文件，其 USN 记录可能更早被滚动覆盖")
print("  - 因此 30d+ 的低复用率不能代表真实情况")
print()

# ============================================================================
# 5. 仅分析样本充足且无偏差的组合
# ============================================================================
print("5. UNBIASED ANALYSIS (Adequate Sample, Minimal Bias)")
print("-" * 90)
print("Only analyzing time periods with adequate samples and minimal survivor bias")
print()

# 只分析 0-24h 和 1-3d（样本充足，幸存者偏差较小）
valid_ages = ['0-24h', '1-3d']

print(f"Analyzing: {', '.join(valid_ages)} (n={sum(age_stats[j]['samples'] for j, age in enumerate(age_buckets) if age in valid_ages):,})")
print()

print("Reuse Rate by Size (0-3 days combined):")
print(f"{'Size':<12} {'0-24h':>10} {'1-3d':>10} {'Combined':>12} {'n':>8}")
print("-" * 60)

for i, size in enumerate(size_buckets):
    rates = []
    samples = []

    for age in valid_ages:
        j = age_buckets.index(age)
        ok, reused = contingency_table[j][i]
        total = ok + reused
        if total >= 10:
            rates.append(reused / total)
            samples.append(total)

    if rates:
        avg_rate = np.mean(rates)
        total_n = sum(samples)
        print(f"{size:<12} {rates[0]:>9.1%} {rates[1]:>9.1%} {avg_rate:>11.1%} {total_n:>8,}")

print()

# ============================================================================
# 6. 最终结论
# ============================================================================
print("=" * 90)
print("FINAL CONCLUSIONS (Statistically Valid)")
print("=" * 90)
print()

print("1. SAMPLE BIAS CORRECTED:")
print("   - 3-7d and 7-14d: EXCLUDED (no USN data)")
print("   - 30d+: EXCLUDED (survivor bias, 87.7% survival rate)")
print("   - Valid data: 0-24h and 1-3d only (n=6,890)")
print()

print("2. SIZE EFFECT (0-3 days, bias-corrected):")
size_effects = []
for i, size in enumerate(size_buckets):
    rates = []
    for age in valid_ages:
        j = age_buckets.index(age)
        ok, reused = contingency_table[j][i]
        total = ok + reused
        if total >= 10:
            rates.append(reused / total)
    if rates:
        avg_rate = np.mean(rates)
        size_effects.append((size, avg_rate))

for size, rate in size_effects:
    if rate > 0.7:
        risk = "VERY HIGH"
    elif rate > 0.5:
        risk = "HIGH"
    elif rate > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    print(f"   {size:<12}: {rate:>6.1%} - {risk} RISK")

print()

print("3. TIME EFFECT:")
print("   - Cannot reliably measure (survivor bias in older data)")
print("   - Only conclusion: 0-3 days show consistent patterns")
print()

print("4. INTERACTION EFFECT:")
print("   Critical combinations (0-3d + >1MB):")
for size in ['100KB-1MB', '1-10MB', '10MB+']:
    i = size_buckets.index(size)
    rates = []
    for age in valid_ages:
        j = age_buckets.index(age)
        ok, reused = contingency_table[j][i]
        total = ok + reused
        if total >= 10:
            rates.append(reused / total)
    if rates:
        avg_rate = np.mean(rates)
        print(f"   - {size}: {avg_rate:.1%} reuse rate")

print()
print("=" * 90)
