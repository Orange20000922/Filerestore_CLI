# 数据集策略分析

## 背景

Govdocs1 数据集文件类型单一（以 zip, pdf, docx, ppt, xls, jpg 为主），缺少 exe, mp4, mp3 等文件类型。本机有 500GB 各式文件数据，当前方案是用 C++ 扫描整个卷生成数据集。

---

## 本机数据 vs Govdocs1 对比

| 维度 | 本机 500GB | Govdocs1 |
|------|-----------|----------|
| **文件多样性** | 高（exe, mp4, mp3, 游戏文件等） | 低（政府文档为主） |
| **真实性** | 真实使用场景 | 人工收集，偏学术 |
| **数据量** | 500GB | ~500GB |
| **隐私风险** | 有（不能公开） | 无（可公开） |
| **偏差** | 有（个人使用习惯） | 有（政府文档偏向） |
| **可复现性** | 差（别人无法复现） | 好（公开数据集） |

**结论：本机数据更适合文件恢复任务**，因为用户场景就是"普通人的电脑"。

---

## 其他公开数据集选择

### 多媒体文件（mp4, mp3, mkv 等）

| 数据集 | 内容 | 大小 | 获取方式 |
|--------|------|------|----------|
| **Pexels/Pixabay** | 免费视频素材 | 可批量下载 | API |
| **Free Music Archive** | mp3 音乐 | 几十GB | 直接下载 |
| **Librivox** | mp3 有声书 | 100GB+ | Archive.org |
| **Internet Archive Video** | 各种视频 | TB级 | 按需下载 |

```bash
# 批量下载 Internet Archive 上的 mp3
ia download --glob="*.mp3" librivox_audiobooks
```

### 可执行文件（exe, dll, so）

| 数据集 | 内容 | 说明 |
|--------|------|------|
| **NSRL (NIST)** | 合法软件哈希库 | 只有哈希，无原始文件 |
| **VirusTotal** | 恶意+正常软件 | 需要 API 权限 |
| **PortableApps** | 免安装软件 | 可批量下载 exe |
| **GitHub Releases** | 开源软件 release | 大量 exe/zip |
| **Chocolatey packages** | Windows 软件包 | 可脚本下载 |

```python
# 从 GitHub 批量下载 release 文件
import requests

repos = [
    "microsoft/vscode",
    "git-for-windows/git",
    "notepad-plus-plus/notepad-plus-plus",
    # ...
]

for repo in repos:
    releases = requests.get(f"https://api.github.com/repos/{repo}/releases").json()
    for release in releases[:3]:  # 最近3个版本
        for asset in release["assets"]:
            if asset["name"].endswith((".exe", ".zip", ".msi")):
                # 下载
                pass
```

### 综合数据集

| 数据集 | 特点 | 链接 |
|--------|------|------|
| **Digital Corpora** | 多种取证数据集 | digitalcorpora.org |
| **Kaggle 文件类型数据集** | 社区贡献 | 搜索 "file type classification" |

---

## 推荐方案：混合数据集

```
┌─────────────────────────────────────────────────────────────┐
│                     混合数据集策略                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   本机数据 (500GB)                                          │
│   ├── 优势：多样性高，真实场景                               │
│   ├── 用于：私有训练，不公开                                 │
│   └── 占比：70%                                             │
│                                                             │
│   公开数据集补充                                             │
│   ├── Govdocs1: PDF, DOC, XLS (已有)                        │
│   ├── Internet Archive: MP3, MP4, MKV                       │
│   ├── GitHub Releases: EXE, ZIP, MSI                        │
│   ├── Free Music Archive: MP3, FLAC                         │
│   └── 占比：30%                                             │
│                                                             │
│   最终数据集                                                │
│   ├── 文件类型覆盖：20+ 种                                  │
│   ├── 总样本量：100万+ 块对                                  │
│   └── 用途：训练 + 发布匿名化版本                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 本机数据最佳实践

### 1. 只提取特征，不保存原始数据

```cpp
// 当前做法是对的：只保存统计特征
// CSV 里不包含原始字节，只有熵、频率分布等
// 这样即使公开也不泄露隐私
```

### 2. 排除敏感目录

```cpp
// 在 DatasetGenerator 中添加排除列表
std::vector<std::wstring> excludePaths = {
    L"\\Users\\*\\AppData\\",
    L"\\Users\\*\\Documents\\Personal\\",
    L"\\$Recycle.Bin\\",
    L"\\Windows\\",  // 系统文件单独处理
};
```

### 3. 按文件类型平衡采样

```cpp
// 避免某类文件过多导致偏差
struct SamplingConfig {
    size_t maxSamplesPerType = 10000;  // 每种类型最多1万样本
    size_t minSamplesPerType = 1000;   // 每种类型至少1000样本
};
```

---

## 当前方案优化建议

### 问题 1：类型分布不均

```
典型个人电脑可能：
├── JPG: 50000 个（照片多）
├── MP4: 2000 个
├── EXE: 500 个
├── ZIP: 300 个
└── FLAC: 50 个

训练时 JPG 会主导模型
```

### 解决：分层采样

```cpp
// 在 DatasetGenerator 中添加
class StratifiedSampler {
public:
    void addSample(const std::string& fileType, const Sample& sample) {
        auto& bucket = samples_[fileType];
        if (bucket.size() < maxPerType_) {
            bucket.push_back(sample);
        } else {
            // 随机替换（reservoir sampling）
            size_t idx = rand() % totalSeen_[fileType];
            if (idx < maxPerType_) {
                bucket[idx] = sample;
            }
        }
        totalSeen_[fileType]++;
    }

private:
    std::map<std::string, std::vector<Sample>> samples_;
    std::map<std::string, size_t> totalSeen_;
    size_t maxPerType_ = 10000;
};
```

### 问题 2：缺少某些文件类型

### 解决：分析并定向补充

```bash
# 识别缺失的文件类型
python analyze_dataset.py --csv my_dataset.csv --show-distribution

# 输出示例：
# JPG: 45000 (45%)  ← 过多
# PDF: 20000 (20%)
# MP4: 5000 (5%)
# EXE: 500 (0.5%)   ← 过少
# MP3: 200 (0.2%)   ← 过少
# MKV: 50 (0.05%)   ← 过少
```

然后从公开数据集补充缺失类型。

---

## 总结

| 建议 | 说明 |
|------|------|
| **继续用本机数据** | 500GB 真实数据比 Govdocs1 更适合 |
| **添加分层采样** | 避免 JPG 等常见类型主导 |
| **公开数据集补充** | 补充 EXE、MP4、MP3 等缺失类型 |
| **只保存特征** | CSV 只含统计特征，可安全公开 |
| **排除敏感路径** | 跳过个人文档、AppData 等 |

---

## 待实现

- [ ] DatasetGenerator 添加分层采样功能
- [ ] 添加敏感目录排除配置
- [ ] 编写数据集分布分析脚本
- [ ] 从 GitHub Releases 批量下载 EXE 补充数据
- [ ] 从 Internet Archive 下载 MP3/MP4 补充数据
