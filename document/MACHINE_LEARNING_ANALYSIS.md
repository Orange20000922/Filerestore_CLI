# 机器学习在数据恢复中的应用分析

> 文档创建时间: 2026-01-06
> 项目: Filerestore_CLI
> 目的: 记录ML技术在数据恢复领域的可行性分析、数学基础和库推荐

---

## 一、可行性评估

### 1.1 结论：完全可行

机器学习在数据恢复领域有明确的应用价值，主要体现在以下场景：

| 应用场景 | 传统方法 | ML方法优势 |
|---------|---------|-----------|
| 文件类型识别 | 签名匹配 | 识别无头/损坏文件 |
| 文件片段重组 | 顺序假设 | 智能匹配碎片 |
| 损坏检测 | 固定熵阈值 | 自适应异常检测 |
| 恢复优先级 | 人工判断 | 预测可恢复性 |

### 1.2 现有研究与工具实例

**1. Sceadan (2013, Simson Garfinkel)**
- 使用字节频率分布进行文件类型分类
- 不依赖文件头签名，可识别片段
- 准确率 > 90%

**2. File Fragment Classification Research**
- Conti et al. 使用 SVM 和决策树
- 基于 n-gram 特征的文件类型识别
- 可处理 4KB 的小片段

**3. Deep Learning Approaches (2018+)**
- CNN 用于原始字节序列分类
- LSTM 用于序列模式识别
- 可达到 95%+ 的片段分类准确率

**4. 商业工具中的应用**
- Autopsy/Sleuth Kit 的 PhotoRec 扩展
- 部分企业级恢复工具已集成 ML 模块

---

## 二、数学基础

### 2.1 信息论基础（项目已在使用）

**Shannon 熵公式：**
```
H(X) = -Σ p(x) · log₂(p(x))
```

项目中 `FileIntegrityValidator::CalculateEntropy()` 的实现：

```cpp
// FileIntegrityValidator.cpp
for (int i = 0; i < 256; i++) {
    if (frequency[i] > 0) {
        double p = (double)frequency[i] / size;
        entropy -= p * log2(p);  // Shannon熵公式
    }
}
```

**不同文件类型的特征熵值：**
| 文件类型 | 熵值范围 (bits/byte) |
|---------|---------------------|
| 纯文本 | 4.5 - 5.5 |
| HTML/XML | 5.0 - 6.0 |
| 可执行文件 | 5.5 - 6.5 |
| 图片 (未压缩) | 6.0 - 7.5 |
| 压缩文件 | 7.8 - 8.0 |
| 加密文件 | 7.9 - 8.0 |

### 2.2 贝叶斯分类

**后验概率公式：**
```
P(FileType | Features) = P(Features | FileType) · P(FileType) / P(Features)
```

**特征向量选择：**
- 字节频率分布 (256维向量)
- n-gram 频率 (如 bigram: 65536维)
- 熵值序列 (分块计算)
- 字节转换概率矩阵

### 2.3 神经网络结构

**用于文件分类的 CNN 架构：**
```
Input: 原始字节序列 [4096 bytes]
    ↓
Conv1D(filters=64, kernel=8) + ReLU
    ↓
MaxPooling1D(pool_size=4)
    ↓
Conv1D(filters=128, kernel=4) + ReLU
    ↓
GlobalAveragePooling1D
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(num_classes) + Softmax
    ↓
Output: 文件类型概率分布
```

**关键数学表达：**
```
卷积层: y = σ(W * x + b)
ReLU: f(x) = max(0, x)
Softmax: P(class_i) = exp(z_i) / Σ exp(z_j)
交叉熵损失: L = -Σ y_true · log(y_pred)
```

### 2.4 聚类算法（用于片段重组）

**K-Means 目标函数：**
```
J = Σ Σ ||x - μ_k||²
```

**中心更新：**
```
μ_k = (1/|C_k|) Σ x
```

**图匹配用于 Fragment Reassembly：**
```
相似度矩阵: S[i][j] = similarity(fragment_i_end, fragment_j_start)
最优路径: 使用动态规划或匈牙利算法求解
```

### 2.5 SVM 支持向量机

**决策函数：**
```
f(x) = sign(Σ αᵢyᵢK(xᵢ, x) + b)
```

**常用核函数：**
- 线性核: K(x, y) = x · y
- RBF核: K(x, y) = exp(-γ||x - y||²)
- 多项式核: K(x, y) = (x · y + c)^d

---

## 三、与本项目的结合点

### 3.1 可增强的模块

```
┌─────────────────────────────────────────────────────────┐
│                    现有模块                              │
├─────────────────────────────────────────────────────────┤
│  FileCarver.cpp          →  ML文件类型分类器             │
│  (签名匹配)                 (无签名片段识别)             │
├─────────────────────────────────────────────────────────┤
│  FileIntegrityValidator  →  ML异常检测模型               │
│  (固定熵阈值)               (自适应损坏检测)             │
├─────────────────────────────────────────────────────────┤
│  OverwriteDetector       →  ML可恢复性预测               │
│  (簇状态检查)               (恢复成功率预估)             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 建议的接口设计

```cpp
// IFileClassifier.h - ML分类器抽象接口
class IFileClassifier {
public:
    virtual ~IFileClassifier() = default;

    // 分类预测
    virtual FileTypePrediction Classify(
        const BYTE* data,
        size_t size
    ) = 0;

    // 获取置信度
    virtual double GetConfidence() = 0;

    // 加载模型
    virtual bool LoadModel(const std::string& modelPath) = 0;

    // 批量预测
    virtual std::vector<FileTypePrediction> ClassifyBatch(
        const std::vector<std::pair<const BYTE*, size_t>>& samples
    ) = 0;
};

// 预测结果结构
struct FileTypePrediction {
    std::string fileType;       // 预测的文件类型
    double confidence;          // 置信度 0-1
    std::vector<std::pair<std::string, double>> topK;  // Top-K 预测
};
```

---

## 四、C++ 机器学习库推荐

### 4.1 轻量级推理库

| 库 | 特点 | 模型格式 | 跨平台 | 集成难度 |
|---|------|---------|--------|---------|
| **ONNX Runtime** | 微软出品，性能优秀 | .onnx | ✅ | 中 |
| **TensorFlow Lite** | 移动端优化，体积小 | .tflite | ✅ | 中 |
| **ncnn** | 腾讯出品，极致轻量 | 自有格式 | ✅ | 低 |

### 4.2 传统机器学习库

| 库 | 算法支持 | 适合场景 | 依赖 |
|---|---------|---------|-----|
| **dlib** | SVM/决策树/聚类 | 成熟稳定 | 无 |
| **mlpack** | SVM/RF/KNN/聚类 | 完整ML管道 | Armadillo |
| **Shark** | SVM/RF/神经网络 | 学术研究 | Boost |

### 4.3 深度学习框架

| 库 | 特点 | 体积 | 适用 |
|---|------|-----|-----|
| **LibTorch** | PyTorch C++版 | 大 (~500MB) | 复杂模型 |
| **OpenCV DNN** | 推理为主 | 中 | 已用OpenCV时 |
| **tiny-dnn** | 纯头文件 | 极小 | 简单网络 |

### 4.4 推荐方案

#### 方案 A：快速起步（1-2周）
```
dlib (SVM分类器)
- 头文件库，直接 #include
- 字节频率 → SVM → 文件类型
- 无运行时依赖
```

#### 方案 B：最佳平衡（3-4周）
```
Python训练 + ONNX Runtime推理
- sklearn/PyTorch 训练模型
- 导出为 .onnx 格式
- C++ 端用 ONNX Runtime 推理
```

#### 方案 C：零依赖（2-3周）
```
tiny-dnn 或 自实现
- 纯头文件神经网络
- 完全可控
- 适合简单分类任务
```

### 4.5 安装方式

```bash
# vcpkg 安装
vcpkg install dlib:x64-windows
vcpkg install onnxruntime-gpu:x64-windows
vcpkg install mlpack:x64-windows

# 或直接下载头文件库
# dlib: https://github.com/davisking/dlib
# tiny-dnn: https://github.com/tiny-dnn/tiny-dnn
```

---

## 五、实现示例代码

### 5.1 dlib SVM 分类器

```cpp
// MLFileClassifier.h
#pragma once
#include <dlib/svm.h>
#include <vector>
#include <map>
#include <string>

class MLFileClassifier {
public:
    using SampleType = dlib::matrix<double, 256, 1>;
    using KernelType = dlib::radial_basis_kernel<SampleType>;

    // 从字节数据提取特征（256维字节频率）
    SampleType ExtractFeatures(const BYTE* data, size_t size) {
        SampleType sample;
        int freq[256] = {0};

        for (size_t i = 0; i < size; i++) {
            freq[data[i]]++;
        }

        for (int i = 0; i < 256; i++) {
            sample(i) = (double)freq[i] / size;  // 归一化到 [0, 1]
        }
        return sample;
    }

    // 预测文件类型
    std::string Predict(const BYTE* data, size_t size) {
        auto features = ExtractFeatures(data, size);
        int label = static_cast<int>(classifier(features));

        auto it = labelToType.find(label);
        if (it != labelToType.end()) {
            return it->second;
        }
        return "unknown";
    }

    // 获取预测置信度
    double GetConfidence() const {
        return lastConfidence;
    }

    // 加载预训练模型
    bool LoadModel(const std::string& path) {
        try {
            dlib::deserialize(path) >> classifier >> labelToType;
            return true;
        } catch (...) {
            return false;
        }
    }

    // 保存模型
    bool SaveModel(const std::string& path) {
        try {
            dlib::serialize(path) << classifier << labelToType;
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    dlib::decision_function<KernelType> classifier;
    std::map<int, std::string> labelToType;
    double lastConfidence = 0.0;
};
```

### 5.2 tiny-dnn 神经网络

```cpp
// NeuralFileClassifier.h
#pragma once
#include <tiny_dnn/tiny_dnn.h>
#include <vector>
#include <string>

class NeuralFileClassifier {
public:
    NeuralFileClassifier() {
        // 构建网络：256 → 128 → 64 → 15
        net << tiny_dnn::fc(256, 128) << tiny_dnn::relu()
            << tiny_dnn::fc(128, 64) << tiny_dnn::relu()
            << tiny_dnn::fc(64, 15) << tiny_dnn::softmax();
    }

    // 提取特征
    tiny_dnn::vec_t ExtractFeatures(const BYTE* data, size_t size) {
        tiny_dnn::vec_t features(256, 0.0);

        for (size_t i = 0; i < size; i++) {
            features[data[i]] += 1.0;
        }

        // 归一化
        for (auto& f : features) {
            f /= size;
        }

        return features;
    }

    // 预测
    std::string Predict(const BYTE* data, size_t size) {
        auto features = ExtractFeatures(data, size);
        auto result = net.predict(features);

        // 找最大概率
        int maxIdx = 0;
        double maxProb = result[0];
        for (size_t i = 1; i < result.size(); i++) {
            if (result[i] > maxProb) {
                maxProb = result[i];
                maxIdx = i;
            }
        }

        lastConfidence = maxProb;
        return fileTypes[maxIdx];
    }

    // 训练
    void Train(const std::vector<tiny_dnn::vec_t>& inputs,
               const std::vector<tiny_dnn::label_t>& labels,
               int epochs = 30) {
        tiny_dnn::adam optimizer;
        net.train<tiny_dnn::cross_entropy>(
            optimizer, inputs, labels, 32, epochs);
    }

    // 保存/加载
    void Save(const std::string& path) { net.save(path); }
    void Load(const std::string& path) { net.load(path); }

    double GetConfidence() const { return lastConfidence; }

private:
    tiny_dnn::network<tiny_dnn::sequential> net;
    double lastConfidence = 0.0;

    std::vector<std::string> fileTypes = {
        "zip", "pdf", "jpg", "png", "gif", "bmp",
        "mp3", "mp4", "avi", "exe", "dll", "doc",
        "xls", "ppt", "txt"
    };
};
```

### 5.3 与 FileCarver 集成

```cpp
// FileCarver.cpp 中添加

#include "MLFileClassifier.h"

// 成员变量
private:
    std::unique_ptr<MLFileClassifier> mlClassifier;
    bool useMLClassification = false;

// 初始化
bool FileCarver::InitMLClassifier(const std::string& modelPath) {
    mlClassifier = std::make_unique<MLFileClassifier>();
    if (mlClassifier->LoadModel(modelPath)) {
        useMLClassification = true;
        return true;
    }
    return false;
}

// 使用ML分类（当签名匹配失败时）
bool FileCarver::ClassifyWithML(const BYTE* data, size_t size,
                                 std::string& outType, double& confidence) {
    if (!mlClassifier || !useMLClassification) {
        return false;
    }

    outType = mlClassifier->Predict(data, size);
    confidence = mlClassifier->GetConfidence();

    // 置信度阈值
    return confidence > 0.7;
}

// 在 ScanBufferMultiSignature 中使用
void FileCarver::ScanBufferMultiSignature(...) {
    // ... 现有签名匹配逻辑 ...

    // 如果签名匹配失败，尝试ML分类
    if (!signatureMatched && useMLClassification) {
        std::string mlType;
        double mlConfidence;

        if (ClassifyWithML(data + offset, 4096, mlType, mlConfidence)) {
            CarvedFileInfo info;
            info.extension = mlType;
            info.confidence = mlConfidence * 0.8;  // ML结果稍微降权
            info.description = "ML-classified " + mlType;
            // ... 填充其他字段 ...
            results.push_back(info);
        }
    }
}
```

---

## 六、实施路线图

### 阶段 1：基础验证（2周）
- [ ] 收集已知文件类型样本（每类100+个）
- [ ] 实现字节频率特征提取
- [ ] 使用 dlib SVM 训练分类器
- [ ] 在离线数据集上测试准确率

### 阶段 2：集成测试（2周）
- [ ] 将 MLFileClassifier 集成到 FileCarver
- [ ] 实现签名匹配 + ML 备选策略
- [ ] 对比测试恢复效果
- [ ] 性能优化（批处理、缓存）

### 阶段 3：深度模型（3周，可选）
- [ ] 使用 PyTorch 训练 CNN 模型
- [ ] 导出为 ONNX 格式
- [ ] 替换 SVM 为神经网络推理
- [ ] A/B 测试对比效果

### 阶段 4：高级功能（4周，可选）
- [ ] 实现文件碎片重组
- [ ] 添加损坏程度预测
- [ ] 支持更多文件类型
- [ ] 模型持续更新机制

---

## 七、参考资源

### 论文
1. Garfinkel, S. "Sceadan: Using Byte Frequency Analysis for File Type Detection" (2013)
2. Conti, G. et al. "Automated Classification of File Fragments" (2010)
3. Chen, Q. et al. "File Fragment Classification Using Deep Learning" (2018)

### 开源项目
- https://github.com/davisking/dlib
- https://github.com/tiny-dnn/tiny-dnn
- https://github.com/microsoft/onnxruntime
- https://github.com/sleuthkit/sleuthkit

### 数据集
- Govdocs1: 公开政府文档数据集
- NIST CFReDS: 数字取证测试数据集
- Custom: 自行收集的文件样本

---

## 八、总结

机器学习在数据恢复中**完全可行且有实际价值**，特别适合：
- 无签名文件片段的类型识别
- 文件损坏程度的智能评估
- 碎片化文件的重组

本项目已经有了良好的基础（熵计算、统计分析），添加 ML 支持是自然的演进方向。

**推荐起步方案：字节频率 + dlib SVM**
- 快速验证可行性
- 零运行时依赖
- 预期准确率 > 85%

验证效果后再考虑深度学习方案以提升性能。
