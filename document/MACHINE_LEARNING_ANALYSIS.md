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

### 2.5 SVM 支持向量机（详细数学原理）

支持向量机 (Support Vector Machine) 是一种强大的监督学习算法，其核心思想是找到一个最优超平面，使得不同类别的数据点之间的"间隔"最大化。

---

#### 2.5.1 直觉理解：为什么要最大间隔？

考虑二维平面上两类点的分类问题：

```
        |
    ×   |   ○
   ×    |    ○ ○
    × × |  ○
   ×    |   ○
        |
```

存在无数条直线可以分开两类点，但哪条最好？

```
方案1（贴近×）          方案2（居中）           方案3（贴近○）
    ×  /  ○             ×   |   ○              ×  \  ○
   × /   ○ ○           ×    |    ○ ○          ×   \  ○ ○
    /× ○               × ×  |  ○               × × \ ○
   /×   ○               ×   |   ○               ×   \ ○
```

**直觉上，方案2最好**——它离两边的点都最远，对新数据的"容错能力"最强。

SVM 的目标就是找到这条"离两边都最远"的分界线，即**最大间隔超平面**。

---

#### 2.5.2 数学建模：线性可分情况

##### 超平面的表示

在 n 维空间中，超平面可以表示为：

```
w · x + b = 0
```

其中：
- `w` 是法向量（垂直于超平面的向量），维度为 n
- `b` 是偏置项（bias），决定超平面到原点的距离
- `x` 是数据点

##### 分类规则

对于二分类问题（标签 y ∈ {-1, +1}）：

```
若 w · x + b > 0，则预测 y = +1
若 w · x + b < 0，则预测 y = -1
```

可以统一写成：**正确分类的点满足 y(w · x + b) > 0**

##### 间隔的定义

点 x 到超平面的距离（几何间隔）：

```
距离 = |w · x + b| / ||w||
```

其中 `||w|| = √(w₁² + w₂² + ... + wₙ²)` 是 w 的欧几里得范数。

对于正确分类的点，`y(w · x + b) > 0`，所以：

```
几何间隔 = y(w · x + b) / ||w||
```

##### 最大间隔优化问题

我们希望最小间隔（距离超平面最近的点）尽可能大：

```
max   min  y_i(w · x_i + b) / ||w||
w,b    i
```

这个问题难以直接求解。我们做一个关键变换：

**函数间隔归一化**：由于 (w, b) 可以任意缩放而不改变超平面位置，我们约定让最近的点满足：

```
min  y_i(w · x_i + b) = 1
 i
```

这样几何间隔就变成了 `1 / ||w||`，最大化间隔等价于：

```
max  1 / ||w||   ⟺   min  ||w||   ⟺   min  (1/2)||w||²
```

##### 最终优化问题（原问题）

```
min   (1/2)||w||²
w,b

subject to:  y_i(w · x_i + b) ≥ 1,  对所有 i = 1, ..., N
```

这是一个**凸二次规划问题**（Convex Quadratic Programming），有唯一全局最优解。

---

#### 2.5.3 拉格朗日对偶与 KKT 条件

##### 为什么要用对偶？

1. 对偶问题更容易求解（只需优化拉格朗日乘子）
2. 对偶形式自然引出**核技巧**
3. 只有少数样本（支持向量）参与最终决策

##### 拉格朗日函数

引入拉格朗日乘子 α_i ≥ 0，构建拉格朗日函数：

```
L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w · xᵢ + b) - 1]
```

原问题等价于：

```
min  max  L(w, b, α)
w,b  α≥0
```

##### 对偶问题推导

对 L 分别对 w 和 b 求偏导并令其为零：

**对 w 求导：**
```
∂L/∂w = w - Σᵢ αᵢyᵢxᵢ = 0

⟹ w = Σᵢ αᵢyᵢxᵢ    ← 关键结论：w 是支持向量的线性组合
```

**对 b 求导：**
```
∂L/∂b = -Σᵢ αᵢyᵢ = 0

⟹ Σᵢ αᵢyᵢ = 0       ← 约束条件
```

将 w = Σᵢ αᵢyᵢxᵢ 代入 L：

```
L = (1/2)||Σᵢ αᵢyᵢxᵢ||² - Σᵢ αᵢyᵢ(Σⱼ αⱼyⱼxⱼ) · xᵢ - b·Σᵢ αᵢyᵢ + Σᵢ αᵢ

  = (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ) - Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ) + Σᵢ αᵢ

  = Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ)
```

##### 对偶问题（最终形式）

```
max   Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ)
 α

subject to:
    αᵢ ≥ 0,  对所有 i
    Σᵢ αᵢyᵢ = 0
```

**核心观察：对偶问题只涉及样本之间的内积 (xᵢ · xⱼ)！** 这为核技巧铺平了道路。

##### KKT 条件

最优解必须满足 Karush-Kuhn-Tucker 条件：

```
αᵢ ≥ 0                           (对偶可行)
yᵢ(w · xᵢ + b) - 1 ≥ 0           (原始可行)
αᵢ[yᵢ(w · xᵢ + b) - 1] = 0       (互补松弛)
```

**互补松弛条件的含义：**
- 若 αᵢ > 0，则 yᵢ(w · xᵢ + b) = 1（该点恰好在间隔边界上）
- 若 yᵢ(w · xᵢ + b) > 1，则 αᵢ = 0（该点在间隔外部，不影响决策）

**αᵢ > 0 的点就是"支持向量"**——只有它们决定了分类超平面！

---

#### 2.5.4 软间隔 SVM：处理线性不可分

现实数据往往无法完美线性分开。软间隔 SVM 引入**松弛变量** ξᵢ 允许部分点违反约束：

```
min   (1/2)||w||² + C · Σᵢ ξᵢ
w,b,ξ

subject to:
    yᵢ(w · xᵢ + b) ≥ 1 - ξᵢ,  对所有 i
    ξᵢ ≥ 0
```

**参数 C 的含义：**
- C 越大：对误分类惩罚越重，间隔越小，容易过拟合
- C 越小：允许更多误分类，间隔越大，可能欠拟合

```
C = 0.1（宽容）           C = 10（严格）           C = 1000（极严格）
    ×    |    ○              ×   |   ○              ×  | ○
   × ×   |  ○ ○             × ×  |  ○ ○            × × |○ ○
    ×  ○ | ○               ×  ○ |  ○              ×  ○| ○
   ×     |  ○               ×   |   ○              ×  | ○
 (允许一个○越界)          (边界更紧)            (过拟合风险)
```

##### 软间隔的对偶问题

```
max   Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ(xᵢ · xⱼ)
 α

subject to:
    0 ≤ αᵢ ≤ C,  对所有 i    ← 唯一区别：α 有上界
    Σᵢ αᵢyᵢ = 0
```

**软间隔的支持向量类型：**
- αᵢ = 0：普通点，在间隔外
- 0 < αᵢ < C：间隔边界上的支持向量
- αᵢ = C：违反间隔的点（可能被误分类）

---

#### 2.5.5 核技巧：处理非线性问题

##### 问题：线性不够怎么办？

考虑 XOR 问题，线性不可分：

```
    ○   ×
    ×   ○
```

**思路：将数据映射到高维空间，在高维空间线性可分！**

定义映射函数 φ: Rⁿ → Rᵐ（m >> n），在高维空间中：

```
f(x) = w · φ(x) + b
```

对偶问题变成：

```
max   Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ · φ(xᵢ) · φ(xⱼ)
```

##### 核函数的魔法

直接计算 φ(x) 可能很昂贵（甚至无穷维），但我们只需要**内积** φ(xᵢ) · φ(xⱼ)！

**核函数定义：**
```
K(x, y) = φ(x) · φ(y)
```

核函数直接计算高维内积，无需显式映射！

##### 常用核函数

**1. 线性核**
```
K(x, y) = x · y
```
等价于不做映射，用于线性可分数据。

**2. 多项式核**
```
K(x, y) = (γ · x · y + r)^d
```
- d: 多项式次数
- γ: 缩放系数
- r: 自由项

隐式映射到 C(n+d, d) 维空间（组合数）。

**例：二维二次多项式核**
```
x = (x₁, x₂), y = (y₁, y₂)

K(x, y) = (x · y)² = (x₁y₁ + x₂y₂)²
        = x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂²

对应映射：φ(x) = (x₁², √2·x₁x₂, x₂²)

验证：φ(x) · φ(y) = x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂² = K(x, y) ✓
```

**3. RBF（高斯）核** ⭐ 最常用
```
K(x, y) = exp(-γ||x - y||²)
```
其中 γ > 0 是带宽参数。

**RBF 核的特殊性质：**
- 隐式映射到**无穷维空间**
- 可以拟合任意复杂的决策边界
- γ 越大，决策边界越复杂（越容易过拟合）

```
γ = 0.1（平滑）          γ = 1（适中）           γ = 10（复杂）
   ___________           ___/\___               _/\/\/\_
  /           \         /        \             /        \
（简单边界）          （适度弯曲）           （过度拟合）
```

**为什么 RBF 核是无穷维？**

通过泰勒展开可以证明：

```
exp(-γ||x - y||²) = exp(-γ||x||²) · exp(-γ||y||²) · exp(2γ x · y)

exp(2γ x · y) = Σₖ (2γ)ᵏ (x · y)ᵏ / k!
              = 1 + 2γ(x·y) + 2γ²(x·y)²/2! + ...

这是无穷级数，对应无穷维特征空间！
```

**4. Sigmoid 核**
```
K(x, y) = tanh(γ · x · y + r)
```
类似神经网络的激活函数，但使用较少。

##### 核函数的选择

| 核函数 | 适用场景 | 优点 | 缺点 |
|-------|---------|-----|-----|
| 线性核 | 线性可分、高维稀疏 | 快速、可解释 | 表达能力有限 |
| 多项式核 | 特征交互重要 | 可控复杂度 | 参数多 |
| RBF核 | 通用场景 | 强大灵活 | 需调参、易过拟合 |

**经验法则：先试 RBF，不行再换。**

---

#### 2.5.6 决策函数

训练完成后，对新样本 x 的预测：

```
f(x) = sign(Σᵢ αᵢyᵢK(xᵢ, x) + b)
```

其中求和只需遍历**支持向量**（αᵢ > 0 的样本），通常只占训练集的一小部分！

**偏置 b 的计算：**

选择任意一个满足 0 < αⱼ < C 的支持向量 xⱼ：

```
b = yⱼ - Σᵢ αᵢyᵢK(xᵢ, xⱼ)
```

实践中取所有这类支持向量计算的 b 的平均值。

---

#### 2.5.7 多分类扩展

SVM 原生是二分类器，多分类需要扩展：

**1. One-vs-Rest (OvR)**
- 训练 K 个分类器（K 为类别数）
- 第 k 个分类器：类别 k vs 其他所有类别
- 预测时选择得分最高的类别

```
训练：K 个二分类器
预测：O(K) 次决策
```

**2. One-vs-One (OvO)**
- 训练 K(K-1)/2 个分类器（每对类别一个）
- 预测时投票决定

```
训练：K(K-1)/2 个二分类器
预测：投票制
优点：每个分类器只用两类数据，训练更快
```

**sklearn 默认使用 OvO 策略。**

---

#### 2.5.8 与文件分类任务的结合

##### 为什么 SVM 适合文件分类？

1. **高维数据**：字节频率特征是 256 维，SVM 擅长高维
2. **样本不多**：SVM 在小样本上表现好（依赖支持向量）
3. **特征相关性**：核技巧能捕捉特征间的非线性关系
4. **类别明确**：文件类型边界相对清晰

##### 参数选择建议

```python
# sklearn 中的 SVM 配置
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',      # RBF核，处理非线性
    C=10.0,            # 惩罚系数，经验值
    gamma='scale',     # γ = 1/(n_features * X.var())
    class_weight='balanced',  # 处理类别不平衡
    probability=True,  # 启用概率估计（用于置信度）
)
```

##### 特征空间可视化（概念图）

```
原始空间（字节频率256维 → 降维到2D示意）：

  高熵 ↑
       |    ·zip ·rar    ·encrypted
       |      ·7z
       |              ·jpg ·png
       |         ·exe
       |    ·pdf
       |         ·doc
       | ·txt ·html
  低熵 +------------------------→
      低ASCII比例        高ASCII比例

经过 RBF 核映射后，不同类型在高维空间中更易分离。
```

##### 训练流程图

```
    原始字节数据
         ↓
    ┌─────────────┐
    │  特征提取   │
    │ (字节频率)  │
    └─────────────┘
         ↓
    256维特征向量
         ↓
    ┌─────────────┐
    │  标准化     │
    │ (零均值)    │
    └─────────────┘
         ↓
    ┌─────────────┐
    │  SVM训练    │  ← 核技巧在此生效
    │ (RBF核)     │
    └─────────────┘
         ↓
    决策函数 f(x) = Σαᵢyᵢ K(xᵢ,x) + b
         ↓
    ┌─────────────┐
    │   预测      │
    │ (多分类)    │
    └─────────────┘
         ↓
    文件类型 + 置信度
```

---

#### 2.5.9 数学公式汇总

| 概念 | 公式 |
|-----|------|
| 超平面 | w · x + b = 0 |
| 几何间隔 | γ = y(w · x + b) / ‖w‖ |
| 原问题 | min ½‖w‖², s.t. yᵢ(w·xᵢ+b) ≥ 1 |
| 对偶问题 | max Σαᵢ - ½ΣᵢⱼαᵢαⱼyᵢyⱼK(xᵢ,xⱼ), s.t. Σαᵢyᵢ=0, 0≤αᵢ≤C |
| 最优 w | w* = Σαᵢyᵢxᵢ |
| 决策函数 | f(x) = sign(Σαᵢyᵢ K(xᵢ,x) + b) |
| RBF核 | K(x,y) = exp(-γ‖x-y‖²) |

---

### 2.6 核技巧深入解析

核技巧（Kernel Trick）是机器学习中最优雅的数学思想之一。本节将深入探讨其数学本质、理论基础和实际应用。

---

#### 2.6.1 核技巧的本质：为什么它有效？

##### 问题的起源

许多现实问题在原始特征空间中是非线性可分的：

```
原始空间（线性不可分）          高维空间（线性可分）

    ○ ○                           ·  ·
  ○ × × ○                       ·      ·
  ○ × × ○        φ(x)          ×        ×
    ○ ○          ───→         ×    ×    ×
                              ×        ×
 (圆形分布)                    (抛物面上)
```

**核心思想**：通过映射 φ: X → H 将数据从原始空间 X 映射到高维特征空间 H，使得在 H 中线性可分。

##### 维度灾难

直接计算高维映射面临的问题：

| 原始维度 | 多项式次数 | 映射后维度 | 计算复杂度 |
|---------|-----------|-----------|-----------|
| 100 | 2 | 5,150 | O(5150²) |
| 100 | 3 | 176,851 | O(10¹⁰) |
| 256 | 2 | 33,024 | O(10⁹) |
| 256 | 3 | 2,829,056 | O(10¹³) |

对于 RBF 核，映射维度是**无穷大**！

##### 核技巧的突破

观察 SVM 对偶问题：

```
max Σᵢ αᵢ - (1/2) Σᵢⱼ αᵢαⱼyᵢyⱼ · φ(xᵢ)·φ(xⱼ)
```

我们只需要计算 **φ(xᵢ)·φ(xⱼ)**（内积），而不需要 φ(x) 本身！

**核函数的定义**：
```
K(x, y) = ⟨φ(x), φ(y)⟩ = φ(x)·φ(y)
```

如果存在一个函数 K 能直接计算这个内积，就不需要显式构造 φ！

---

#### 2.6.2 具体例子：理解核函数如何工作

##### 例1：二次多项式核

考虑二维输入 x = (x₁, x₂)，使用核函数：

```
K(x, y) = (x·y)² = (x₁y₁ + x₂y₂)²
```

**展开计算**：
```
K(x, y) = (x₁y₁)² + 2(x₁y₁)(x₂y₂) + (x₂y₂)²
        = x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂²
```

**对应的特征映射**：
```
φ(x) = (x₁², √2·x₁x₂, x₂²)
φ(y) = (y₁², √2·y₁y₂, y₂²)
```

**验证**：
```
φ(x)·φ(y) = x₁²y₁² + 2x₁x₂y₁y₂ + x₂²y₂² = K(x, y) ✓
```

**计算复杂度对比**：

| 方法 | 计算步骤 | 复杂度 |
|-----|---------|-------|
| 显式映射 | 计算 φ(x), φ(y), 再内积 | O(d²) |
| 核函数 | 直接计算 (x·y)² | O(d) |

##### 例2：带参数的多项式核

```
K(x, y) = (x·y + 1)²
```

**展开**：
```
K(x, y) = (x₁y₁ + x₂y₂ + 1)²
        = x₁²y₁² + x₂²y₂² + 1 + 2x₁y₁ + 2x₂y₂ + 2x₁x₂y₁y₂
```

**对应映射**（6维）：
```
φ(x) = (x₁², x₂², 1, √2·x₁, √2·x₂, √2·x₁x₂)
```

常数项 1 使得映射包含了原始特征（一次项），提供了更丰富的表达能力。

##### 例3：XOR 问题的核解法

XOR 问题（线性不可分的经典例子）：

```
输入          标签
(0, 0)   →    -1
(0, 1)   →    +1
(1, 0)   →    +1
(1, 1)   →    -1
```

使用核 K(x, y) = (x·y + 1)²，计算核矩阵：

```
       (0,0)  (0,1)  (1,0)  (1,1)
(0,0)    1      1      1      1
(0,1)    1      2      1      2
(1,0)    1      1      2      2
(1,1)    1      2      2      4
```

在这个核诱导的特征空间中，四个点变成线性可分！

---

#### 2.6.3 Mercer 定理：什么函数可以作为核？

##### 核函数的条件

不是任意函数都能作为核函数。**Mercer 定理**给出了充分必要条件。

**定义（正定核）**：

函数 K: X × X → R 是正定核，当且仅当对于任意 n 个点 {x₁, ..., xₙ} 和任意实数 {c₁, ..., cₙ}：

```
Σᵢⱼ cᵢcⱼ K(xᵢ, xⱼ) ≥ 0
```

等价地，**核矩阵（Gram 矩阵）总是半正定的**。

**Mercer 定理**：

K 是有效的核函数 ⟺ 存在特征映射 φ 使得 K(x,y) = ⟨φ(x), φ(y)⟩

##### 核矩阵（Gram 矩阵）

给定 n 个样本，核矩阵定义为：

```
      ┌                                    ┐
      │ K(x₁,x₁)  K(x₁,x₂)  ...  K(x₁,xₙ) │
  G = │ K(x₂,x₁)  K(x₂,x₂)  ...  K(x₂,xₙ) │
      │    ⋮         ⋮       ⋱      ⋮     │
      │ K(xₙ,x₁)  K(xₙ,x₂)  ...  K(xₙ,xₙ) │
      └                                    ┘
```

**性质**：
- 对称性：G = Gᵀ（因为 K(x,y) = K(y,x)）
- 半正定性：所有特征值 ≥ 0
- 大小：n × n（只依赖样本数，不依赖特征维度！）

##### 验证一个函数是否为有效核

**方法1：构造特征映射**
如果能显式写出 φ，使得 K(x,y) = φ(x)·φ(y)，则 K 是有效核。

**方法2：验证核矩阵半正定**
对于给定数据，计算 G 的特征值，检查是否都 ≥ 0。

**方法3：使用核的封闭性**（见下节）

---

#### 2.6.4 核函数的构造与组合

有效核函数在某些运算下保持有效性，这提供了构造新核的方法。

##### 基本构造规则

设 K₁, K₂ 是有效核，a > 0，则以下也是有效核：

| 规则 | 新核 | 说明 |
|-----|-----|-----|
| 常数倍 | a·K₁(x,y) | 缩放 |
| 加法 | K₁(x,y) + K₂(x,y) | 特征拼接 |
| 乘法 | K₁(x,y) · K₂(x,y) | 特征张量积 |
| 多项式 | (K₁(x,y))ⁿ | 高阶交互 |
| 指数 | exp(K₁(x,y)) | 无穷级数 |
| f变换 | f(x)·K₁(x,y)·f(y) | 特征加权 |

##### 加法核：特征拼接

```
K(x,y) = K₁(x,y) + K₂(x,y)
```

对应特征映射：
```
φ(x) = [φ₁(x), φ₂(x)]  （拼接两个特征向量）
```

**应用**：组合不同类型的特征

```python
# 例：同时使用字节频率和熵特征
K_combined = K_byte_freq + K_entropy
```

##### 乘法核：张量积

```
K(x,y) = K₁(x,y) · K₂(x,y)
```

对应特征映射：
```
φ(x) = φ₁(x) ⊗ φ₂(x)  （张量积/外积）
```

维度变化：如果 φ₁ 是 m 维，φ₂ 是 n 维，则 φ 是 m×n 维。

##### RBF 核的构造

RBF 核可以通过多项式核的无穷和构造：

```
exp(x·y) = Σₙ (x·y)ⁿ / n!
         = 1 + (x·y) + (x·y)²/2! + (x·y)³/3! + ...
```

每一项 (x·y)ⁿ 都对应一个多项式核，所以 exp(x·y) 是有效核。

RBF 核可以改写为：

```
K(x,y) = exp(-γ‖x-y‖²)
       = exp(-γ‖x‖²) · exp(2γ x·y) · exp(-γ‖y‖²)
       = f(x) · exp(2γ x·y) · f(y)
```

其中 f(x) = exp(-γ‖x‖²)，这符合 f 变换规则，所以是有效核。

---

#### 2.6.5 常用核函数深入分析

##### 1. 线性核

```
K(x, y) = x · y = Σᵢ xᵢyᵢ
```

**特征映射**：φ(x) = x（恒等映射）

**特点**：
- 最简单，计算最快
- 无超参数
- 适用于线性可分问题
- 高维稀疏数据（如文本 TF-IDF）效果好

**决策边界**：超平面（线性）

```
     |        /
  ×  |  ○   /  ← 线性核的决策边界
 × × | ○ ○ /
  ×  |  ○ /
     |   /
```

##### 2. 多项式核

```
K(x, y) = (γ·x·y + r)^d
```

**参数含义**：
- d（degree）：多项式次数，控制复杂度
- γ（gamma）：内积缩放因子
- r（coef0）：自由项，控制高阶项 vs 低阶项的影响

**特征维度**：C(n+d, d) = (n+d)! / (d!·n!)

**d 的影响**：

```
d=1:        d=2:           d=3:
 线性        二次曲线        三次曲线
   /        ___           ___/\___
  /        /   \         /        \
```

**为什么需要 r ≠ 0？**

当 r = 0 时，K(x,y) = (x·y)^d 只包含 d 次齐次项。
当 r > 0 时，展开式包含 0 到 d 次的所有项：

```
(x·y + 1)² = 1 + 2(x·y) + (x·y)²
             ↑     ↑        ↑
           常数   一次     二次
```

##### 3. RBF（高斯）核 ⭐

```
K(x, y) = exp(-γ‖x - y‖²) = exp(-γ Σᵢ(xᵢ - yᵢ)²)
```

**直观理解**：
- 衡量两点之间的"相似度"
- 距离越近，K 值越接近 1
- 距离越远，K 值越接近 0

```
K(x,y)
  1 |  *
    |   *
    |    *
    |     * *
  0 |________* * * ___
    0              ‖x-y‖
```

**γ（gamma）的影响**：

```
γ 控制"影响范围"的大小：

γ 小（宽松）              γ 大（紧凑）
每个点影响范围大          每个点影响范围小
决策边界平滑              决策边界复杂
可能欠拟合                可能过拟合

   ___________              /\/\/\/\
  /           \            /        \
```

**γ 的选择**：
- sklearn 默认：γ = 1 / (n_features × X.var())
- 经验范围：通常在 [0.001, 1000] 之间网格搜索
- 交叉验证选择最优值

**RBF 核对应的无穷维空间**：

展开推导：
```
K(x,y) = exp(-γ‖x-y‖²)
       = exp(-γ(‖x‖² - 2x·y + ‖y‖²))
       = exp(-γ‖x‖²) · exp(2γ x·y) · exp(-γ‖y‖²)
```

令 σ² = 1/(2γ)，考虑 exp(x·y/σ²) 的泰勒展开：

```
exp(x·y/σ²) = Σₙ₌₀^∞ (x·y)ⁿ / (n! · σ^(2n))
```

每个 (x·y)ⁿ 项对应一个 n 次多项式核，所以 RBF 核等价于**所有多项式核的加权和**，对应无穷维特征空间！

**形式化的特征映射**（一维情况）：

```
φ(x) = exp(-γx²) · [1, √(2γ/1!)·x, √((2γ)²/2!)·x², √((2γ)³/3!)·x³, ...]
```

这是一个无穷维向量！

##### 4. Sigmoid 核

```
K(x, y) = tanh(γ·x·y + r)
```

**与神经网络的联系**：
- tanh 是神经网络常用的激活函数
- Sigmoid 核模拟了两层神经网络

**注意**：Sigmoid 核不总是正定的！只有当参数满足特定条件时才是有效核。实践中较少使用。

##### 5. Laplacian 核

```
K(x, y) = exp(-γ‖x - y‖₁) = exp(-γ Σᵢ|xᵢ - yᵢ|)
```

使用 L1 范数而非 L2 范数，对异常值更鲁棒。

##### 核函数对比

| 核 | 参数 | 复杂度 | 适用场景 |
|---|-----|-------|---------|
| 线性 | 无 | O(d) | 高维稀疏、线性可分 |
| 多项式 | d, γ, r | O(d) | 已知多项式关系 |
| RBF | γ | O(d) | 通用、非线性 |
| Laplacian | γ | O(d) | 有异常值 |

---

#### 2.6.6 核 PCA：核技巧的另一个应用

核技巧不仅用于 SVM，还可以用于许多基于内积的算法。

##### 传统 PCA

主成分分析寻找数据的主方向，通过特征值分解协方差矩阵：

```
C = (1/n) Σᵢ (xᵢ - μ)(xᵢ - μ)ᵀ
```

##### 核 PCA

在特征空间 H 中做 PCA：

```
C_φ = (1/n) Σᵢ φ(xᵢ)φ(xᵢ)ᵀ
```

由于 φ(x) 可能是无穷维，直接计算不可行。但可以证明：

核 PCA 只需要核矩阵 G，无需显式计算 φ！

**算法**：
1. 计算核矩阵 G，其中 Gᵢⱼ = K(xᵢ, xⱼ)
2. 中心化：G̃ = G - 1ₙG - G1ₙ + 1ₙG1ₙ
3. 对 G̃ 做特征值分解
4. 取前 k 个特征向量

**效果**：

```
原始空间（不可分）          核 PCA 后（可分）
    ○ ○                        ○ ○ ○
  ○ × × ○         →              ×
  ○ × × ○                      × × ×
    ○ ○                          ○
```

---

#### 2.6.7 核参数选择

##### 网格搜索 + 交叉验证

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# 网格搜索
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,              # 5折交叉验证
    scoring='accuracy',
    n_jobs=-1          # 并行
)

grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
```

##### 参数影响可视化

```
           C 小                 C 大
         (欠拟合)              (过拟合)
        ┌─────────┐         ┌─────────┐
γ 小    │ 非常平滑 │         │  较平滑  │
(欠拟合) │ × ○混杂 │         │ 基本分开 │
        └─────────┘         └─────────┘
        ┌─────────┐         ┌─────────┐
γ 大    │  较复杂  │         │ 非常复杂 │
(过拟合) │ 基本分开 │         │ 完美分开 │ ← 可能过拟合
        └─────────┘         └─────────┘
```

##### 经验法则

1. **先用默认参数**：sklearn 的默认值通常合理
2. **对数尺度搜索**：C 和 γ 通常在对数尺度变化
3. **先粗后细**：先大范围搜索，再在最优附近细化
4. **使用交叉验证**：避免在测试集上调参

---

#### 2.6.8 实际计算示例

##### 完整的核矩阵计算（Python）

```python
import numpy as np

def compute_kernel_matrix(X, kernel='rbf', gamma=1.0, degree=3, coef0=1):
    """
    计算核矩阵
    X: (n_samples, n_features)
    """
    n = X.shape[0]
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if kernel == 'linear':
                # 线性核: K(x,y) = x·y
                K[i, j] = np.dot(X[i], X[j])

            elif kernel == 'poly':
                # 多项式核: K(x,y) = (γ·x·y + r)^d
                K[i, j] = (gamma * np.dot(X[i], X[j]) + coef0) ** degree

            elif kernel == 'rbf':
                # RBF核: K(x,y) = exp(-γ‖x-y‖²)
                diff = X[i] - X[j]
                K[i, j] = np.exp(-gamma * np.dot(diff, diff))

    return K

# 示例：3个二维样本
X = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
])

print("线性核矩阵:")
print(compute_kernel_matrix(X, kernel='linear'))
# [[0 0 0]
#  [0 1 0]
#  [0 0 1]]

print("\nRBF核矩阵 (γ=1):")
print(compute_kernel_matrix(X, kernel='rbf', gamma=1))
# [[1.    0.368 0.368]
#  [0.368 1.    0.135]
#  [0.368 0.135 1.   ]]
```

##### 验证核矩阵半正定性

```python
def is_valid_kernel(K, tol=1e-10):
    """检查核矩阵是否半正定"""
    eigenvalues = np.linalg.eigvals(K)
    return np.all(eigenvalues >= -tol)

# 验证 RBF 核矩阵
K_rbf = compute_kernel_matrix(X, kernel='rbf', gamma=1)
print(f"RBF核矩阵是有效核: {is_valid_kernel(K_rbf)}")  # True

# 尝试一个无效的"核"
def invalid_kernel(X):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = np.sin(np.dot(X[i], X[j]))  # sin 不是有效核！
    return K

K_invalid = invalid_kernel(X)
print(f"sin核矩阵是有效核: {is_valid_kernel(K_invalid)}")  # 可能 False
```

---

#### 2.6.9 核技巧在文件分类中的应用

##### 为什么文件分类适合用核方法？

1. **字节频率是高维数据**（256维）
   - 高维空间中线性方法可能失效
   - 核方法天然适合高维

2. **类别之间存在非线性边界**
   - 压缩文件和加密文件熵值相近但字节分布不同
   - 核方法可以学习这种非线性关系

3. **样本量有限**
   - 核方法计算复杂度主要取决于样本数，而非特征维度
   - SVM 的泛化能力在小样本上表现好

##### 自定义核函数

可以设计专门针对文件分类的核：

```python
def file_kernel(x, y, gamma_freq=1.0, gamma_entropy=0.5):
    """
    自定义文件分类核
    x, y: 261维特征向量 (256维字节频率 + 5维统计特征)
    """
    # 分离特征
    freq_x, stat_x = x[:256], x[256:]
    freq_y, stat_y = y[:256], y[256:]

    # 字节频率部分：RBF核
    diff_freq = freq_x - freq_y
    K_freq = np.exp(-gamma_freq * np.dot(diff_freq, diff_freq))

    # 统计特征部分：RBF核（不同的gamma）
    diff_stat = stat_x - stat_y
    K_stat = np.exp(-gamma_entropy * np.dot(diff_stat, diff_stat))

    # 组合（加法核）
    return K_freq + K_stat
```

##### 使用自定义核

```python
from sklearn.svm import SVC

# 预计算核矩阵
def compute_custom_kernel_matrix(X):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = file_kernel(X[i], X[j])
    return K

K_train = compute_custom_kernel_matrix(X_train)
K_test = compute_custom_kernel_matrix_vs(X_test, X_train)  # 测试集 vs 训练集

# 使用预计算核
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)
y_pred = svm.predict(K_test)
```

---

#### 2.6.10 核技巧总结

| 概念 | 要点 |
|-----|-----|
| **核心思想** | 用核函数 K(x,y) 替代高维内积 φ(x)·φ(y)，避免显式计算 |
| **数学基础** | Mercer 定理：K 有效 ⟺ 核矩阵半正定 |
| **核的构造** | 加法、乘法、指数变换保持有效性 |
| **RBF 核** | 最常用，隐式映射到无穷维，参数 γ 控制复杂度 |
| **核矩阵** | n×n 对称半正定矩阵，复杂度只依赖样本数 |
| **参数选择** | 网格搜索 + 交叉验证 |

**核技巧的优雅之处**：
```
在无穷维空间中计算，却只付出有限的代价。
```

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
