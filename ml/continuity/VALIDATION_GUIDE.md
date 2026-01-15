# 块连续性检测验证实验指南

## 实验目的

验证 ML 模型是否能准确判断两个数据块是否连续，为文件恢复中的碎片重组提供技术可行性证明。

## 实验设计

### 核心问题
**模型能否准确判断两个 8KB 数据块是否来自同一文件的相邻位置？**

### 验证方法

1. **分类任务**：判断块对是否连续
   - 正样本：同一文件的相邻块 (label=1)
   - 负样本：不相邻的块 (label=0)
   - 评估指标：Accuracy, Precision, Recall, F1-score

2. **重组任务**（进阶）：打乱文件块后重新排序
   - 将文件切成 N 个块
   - 随机打乱顺序
   - 用模型预测所有块对的连续性分数
   - 贪心算法重建顺序
   - 评估指标：块对正确率、完美重组率

## 快速开始

### 环境准备

```bash
cd ml/continuity
pip install torch numpy tqdm scikit-learn
```

### 数据准备

将测试数据放在 `D:\temp\ml_training\` 目录下：
- ZIP 文件（Govdocs）
- MP3 文件（FMA）
- MP4 文件（Pexels）

建议至少准备：
- 50 个 ZIP 文件
- 100 个 MP3 文件
- 50 个 MP4 文件

### 运行完整实验

```bash
python validate_continuity.py run --data-dir D:\temp\ml_training
```

这将自动完成：
1. 生成训练/验证/测试数据集
2. 训练 CNN 模型
3. 在测试集上评估
4. 运行碎片重组实验

结果保存在 `./experiment_results/` 目录。

## 分步执行

### 1. 仅生成数据集

```bash
python validate_continuity.py generate \
    --data-dir D:\temp\ml_training \
    --output dataset.csv
```

生成文件：
- `dataset_train.csv` - 训练集 (70%)
- `dataset_val.csv` - 验证集 (15%)
- `dataset_test.csv` - 测试集 (15%)
- `dataset_stats.json` - 统计信息

### 2. 训练模型

```bash
python validate_continuity.py train \
    --train-csv dataset_train.csv \
    --val-csv dataset_val.csv \
    --output ./models/ \
    --epochs 50
```

### 3. 评估模型

```bash
python validate_continuity.py evaluate \
    --checkpoint ./models/best_model.pt \
    --test-csv dataset_test.csv
```

### 4. 碎片重组实验

```bash
python validate_continuity.py reassembly \
    --checkpoint ./models/best_model.pt \
    --data-dir D:\temp\ml_training \
    --num-files 10
```

## 预期结果

### 良好结果（可行性证明成功）

```
Classification Task:
  Test Accuracy: 90%+
  Test F1 Score: 88%+

Reassembly Task:
  Avg Pair Accuracy: 85%+
  Perfect Rate: 30%+
```

**结论**：ML 方法对块连续性检测有效，可用于文件恢复。

### 中等结果（需要改进）

```
Classification Task:
  Test Accuracy: 75-85%
  Test F1 Score: 70-85%

Reassembly Task:
  Avg Pair Accuracy: 65-80%
  Perfect Rate: 10-25%
```

**结论**：方法有一定效果，但需要：
- 增加训练数据
- 优化特征提取
- 尝试更复杂的模型

### 较差结果（需要重新考虑）

```
Classification Task:
  Test Accuracy: <70%
  Test F1 Score: <65%

Reassembly Task:
  Avg Pair Accuracy: <60%
  Perfect Rate: <5%
```

**结论**：当前方法可能不适合，需要：
- 重新设计特征
- 考虑其他方法（如基于签名的启发式）
- 分析失败案例

## 数据集说明

### 样本类型

1. **same_file** (正样本)
   - 同一文件的相邻块
   - 例如：file.zip 的第 5 块和第 6 块

2. **same_file_non_adjacent** (负样本)
   - 同一文件的非相邻块
   - 例如：file.zip 的第 5 块和第 20 块

3. **different_files** (负样本)
   - 不同文件的块
   - 例如：file1.zip 的第 5 块和 file2.zip 的第 10 块

### 自适应采样

脚本默认使用自适应采样：
- 小文件（<10MB）：生成 10-50 个样本
- 中等文件（10-100MB）：生成 50-200 个样本
- 大文件（>100MB）：生成 200-500 个样本

这确保了不同大小文件在数据集中的平衡。

## 输出文件说明

### experiment_results/

```
experiment_results/
├── dataset_train.csv          # 训练集
├── dataset_val.csv            # 验证集
├── dataset_test.csv           # 测试集
├── dataset_stats.json         # 数据集统计
├── best_model.pt              # 最佳模型权重
└── experiment_results.json    # 完整实验结果
```

### experiment_results.json 结构

```json
{
  "timestamp": "2026-01-14T...",
  "dataset_stats": {
    "total_samples": 50000,
    "positive_samples": 25000,
    "negative_samples": 25000,
    "train_samples": 35000,
    "val_samples": 7500,
    "test_samples": 7500
  },
  "test_metrics": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1": 0.92,
    "auc": 0.96
  },
  "reassembly_results": {
    "avg_pair_accuracy": 0.87,
    "perfect_rate": 0.35,
    "md5_match_rate": 0.35
  }
}
```

## 常见问题

### Q: 训练需要多长时间？

**A**: 取决于数据集大小和硬件：
- 5万样本 + GPU：10-20 分钟
- 5万样本 + CPU：1-2 小时
- 20万样本 + GPU：30-60 分钟

### Q: 需要多少数据？

**A**: 建议：
- 最小：200 个文件，2万样本
- 推荐：500 个文件，5万样本
- 理想：1000+ 个文件，10万+ 样本

### Q: 为什么碎片重组准确率低于分类准确率？

**A**: 这是正常的：
- 分类任务：判断单个块对（独立决策）
- 重组任务：需要对所有块对排序（累积误差）
- 即使分类准确率 90%，重组 10 个块也可能出错

### Q: 如何提高性能？

**A**: 尝试：
1. 增加训练数据（最有效）
2. 调整特征提取（修改 `feature_extractor.py`）
3. 尝试不同模型（`--model cnn_residual`）
4. 增加训练轮数（`--epochs 100`）
5. 调整学习率

### Q: 可以用于论文/比赛吗？

**A**: 可以，建议包含：
1. 实验设计说明
2. 数据集构建方法
3. 对比实验（与 baseline 比较）
4. 失败案例分析
5. 实际应用场景讨论

## 下一步

### 如果结果良好

1. **扩大数据集**：增加到 10 万+ 样本
2. **云端训练**：使用 Colab/学校 GPU 训练更大模型
3. **写论文/准备比赛材料**：
   - 实验设计
   - 对比实验
   - 结果分析
4. **集成到 C++ 工具**：导出 ONNX 模型

### 如果结果一般

1. **分析失败案例**：哪些文件类型效果差？
2. **特征工程**：是否需要更多特征？
3. **数据增强**：增加负样本多样性
4. **尝试其他方法**：Transformer、图神经网络等

### 如果结果较差

1. **检查数据质量**：是否有标注错误？
2. **简化问题**：先只做单一文件类型（如 ZIP）
3. **对比 baseline**：与简单启发式方法比较
4. **重新评估可行性**：是否需要调整研究方向

## 论文/比赛建议

### 创新点

1. **ML 用于文件碎片检测**：相对新颖的应用
2. **自适应采样策略**：平衡不同大小文件
3. **多格式支持**：ZIP/MP3/MP4 统一框架

### 对比实验

建议对比：
1. **Random baseline**：随机猜测（50% 准确率）
2. **Entropy-based**：基于熵值的启发式
3. **Signature-based**：基于文件签名
4. **Your ML method**：你的方法

### 评估指标

- 分类任务：Accuracy, Precision, Recall, F1, AUC
- 重组任务：块对准确率、完美重组率、部分正确率
- 效率：推理时间、内存占用

### 论文结构建议

1. **Introduction**：文件恢复挑战、碎片化问题
2. **Related Work**：现有文件恢复方法
3. **Method**：特征设计、模型架构、训练策略
4. **Experiments**：数据集、实验设置、结果
5. **Analysis**：失败案例、特征重要性分析
6. **Conclusion**：贡献、局限性、未来工作

## 技术支持

如有问题，检查：
1. `experiment_results/experiment_results.json` - 完整结果
2. `experiment_results/dataset_stats.json` - 数据集统计
3. 控制台输出 - 训练过程日志

祝实验顺利！
