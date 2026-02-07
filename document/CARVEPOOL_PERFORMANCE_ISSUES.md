# carvepool 性能问题排查记录

## 基准对比

| 版本 | 吞吐量 | 备注 |
|------|--------|------|
| `029d399` (原始线程池) | ~2.3 GB/s | 基线 |
| `6cab623` (最近大更新) | ~2.3 GB/s | 正常 |
| `84ed807` (SIMD默认开启) | ~700 MB/s | **根因: SIMD两阶段设计** |
| `84ed807` + SIMD默认关闭 | ~2.3 GB/s | **已恢复** |
| **流式SIMD重新设计** | **平均 2.7 GB/s, 峰值 3.0-4.5 GB/s** | **+17% 平均, +30-96% 峰值** |

**测试环境**: PCIe 4.0 x4 NVMe (~6 GB/s 物理上限)

**根本原因**: `ScanChunkSimd` 的两阶段扫描设计在磁盘数据场景下是反优化（详见已修复问题 #6）。流式 SIMD 重新设计解决了这个问题（详见已修复问题 #7）。

---

## 已修复的问题

### 1. EstimateFileSize 包装函数调用重量级实现
- **文件**: `SignatureScanThreadPool.cpp`
- **问题**: `EstimateFileSize()` 原本是轻量级 O(1) 函数（只读头部字段），但被改成了直接调用 `FileCarver::EstimateFileSizeStatic()`，后者对 ZIP 做 EOCD 逆向搜索（最多65KB）、对 PDF 做 %%EOF 逆向搜索、对 PNG 做 chunk 遍历等
- **影响**: 每个签名命中都触发重量级计算，工作线程处理变慢，队列堆积，I/O 线程被 `queueNotFull.wait()` 阻塞
- **修复**: 恢复为原始 O(1) 实现：BMP 读 offset 2 的 DWORD、AVI/WAV 读 RIFF size、MP4 读 ftyp atom、其他返回 `min(dataSize, maxSize)`

### 2. ScanChunk 对所有格式调用 EstimateFileSizeStatic
- **文件**: `SignatureScanThreadPool.cpp` - `ScanChunk()` 和 `ScanChunkSimd()`
- **问题**: 扫描核心函数对所有格式统一调用 `FileCarver::EstimateFileSizeStatic()`
- **修复**: 分路径处理——ZIP 保留 `EstimateFileSizeStatic`（EOCD 搜索对 ZIP 恢复至关重要），其他格式使用轻量级 `EstimateFileSize()`

### 3. 高频首字节签名污染扫描热循环
- **文件**: `FileCarver.cpp` - `InitializeSignatures()`
- **问题**: 新增的签名引入了高频首字节：
  - `html` (0x3C `<`) — 磁盘数据中极其常见
  - `xml` (0x3C `<`) — 同上，与 html 共享首字节
  - `rtf` (0x7B `{`) — 很常见
  - `ole` (0xD0) — 较常见
- **影响**: 扫描热循环中 `signatureIndex->find(currentByte)` 对这些字节频繁命中，导致大量无意义的签名匹配尝试
- **修复**: 从 `InitializeSignatures()` 中移除这四个签名。html/xml/rtf 可由 ML 分类处理，ole 通常被解析为 zip

### 4. sig 模式下 ML 推理仍在运行
- **文件**: `CarveCommands.cpp`
- **问题**: `FileCarver` 构造函数调用 `AutoLoadMLModel()` 自动加载 ONNX 模型。`ScanForFileTypesThreadPool()` 内部检测到模型已加载就设置 `threadPoolConfig.useMLClassification = true`，导致每个工作线程在每次签名命中时调用 `EnhanceWithML()` → ONNX 推理
- **影响**: 12个工作线程并发执行 ONNX 推理，造成内存飙升（最高2GB）和吞吐剧烈波动（2000-500 MB/s），内存占用与磁盘吞吐成反比
- **修复**: `sig` 模式下在调用 `ScanForFileTypesThreadPool` 前显式 `carver.SetMLClassification(false)`

### 5. SIMD 相关声明丢失（git stash 事故）
- **文件**: `SignatureScanThreadPool.h` / `.cpp`
- **问题**: 之前的 `git stash` 操作导致 SIMD 相关的成员变量和方法声明丢失，编译报错
- **修复**: 重新添加 `SimdSignatureScanner` 成员、`ScanChunkSimd` 方法、`SetSimdEnabled`/`IsSimdEnabled`/`GetSimdInfo` 公有接口

### 6. [根因] ScanChunkSimd 两阶段设计导致性能崩塌
- **文件**: `SignatureScanThreadPool.cpp`
- **问题**: `ScanChunkSimd` 采用两阶段设计：
  1. SIMD 扫描整个 8MB chunk，将所有首字节匹配位置收集到 `vector<MatchResult>`
  2. 遍历 vector 逐个验证完整签名
- **根因**: 磁盘原始数据中 `0x00`（mp4签名首字节）和 `0xFF`（jpg签名首字节）极其密集，一个 8MB chunk 中可能有**数百万**个 0x00 字节。`matches.reserve(dataSize/512)` 只预分配 16K 项，实际可能需要 100x+，导致：
  - vector 反复 realloc（每次拷贝+释放旧内存）
  - 内存占用飙升至 GB 级别
  - CPU cache 全面失效（巨大 vector 无法放入 L1/L2）
  - 12个工作线程同时竞争内存分配器
- **对比**: 原始 `ScanChunk` 是单遍逐字节扫描，命中时就地处理，零额外内存分配，cache 友好
- **临时修复**: 将 `useSimdScan` 默认值改为 `false`，默认走标量 `ScanChunk` 路径
- **后续**: 见已修复问题 #7（流式 SIMD 重新设计）

### 7. 流式 SIMD 重新设计
- **文件**: `SignatureScanThreadPool.h` / `.cpp`
- **实现日期**: 2026-02-07
- **设计思路**: 保留 SIMD 的并行字节比较能力，但**立即就地处理匹配**，不收集到中间 vector
- **核心改动**:
  1. **移除 `SimdSignatureScanner` 依赖**，改用内联 SIMD 指令
  2. **预计算目标字节向量**（栈上分配）：`vector<__m256i> targetVecs`（AVX2）或 `vector<__m128i>`（SSE2）
  3. **流式扫描循环**：
     - AVX2: 32 字节步进，mask == 0 时直接跳过（快速路径）
     - SSE2: 16 字节步进，同样逻辑
     - mask != 0: 对每个命中位置，立即执行完整签名验证 + 文件处理
  4. **剩余字节标量处理**：使用 256-bit bitmap 快速查找
  5. **零额外内存分配**：不再需要 `vector<MatchResult>`
- **性能测试结果**（PCIe 4.0 x4 NVMe）:
  - 平均吞吐: 2.7 GB/s（标量版 2.3 GB/s，**+17%**）
  - 峰值吞吐: 3.0-4.5 GB/s（数据分布影响显著，**+30-96%**）
  - 峰值出现在未分配空间、零填充区域（目标首字节稀疏）
  - 低吞吐区域：已有文件数据区域、高频首字节密集区
- **为什么有效**:
  - **无匹配区域加速**: 标量版逐字节 hash lookup，SIMD 版直接跳 32 字节
  - **零内存分配**: 消除了 vector realloc、cache 污染、分配器竞争
  - **cache 友好**: 流式处理，工作集小，L1/L2 命中率高
- **I/O 瓶颈分析**:
  - PCIe 4.0 x4 NVMe 物理上限 ~6 GB/s
  - 当前平均 2.7 GB/s，峰值 4.5 GB/s，仍未达到 I/O 上限
  - CPU 处理速度仍是主要瓶颈，但已大幅缓解

---

## 未修复 / 待排查的问题

### 6. [已确认非问题] 6cab623 版 SignatureScanThreadPool 实现差异
- **状态**: 已通过 git diff 完整对比，确认差异仅为我们的修复内容
- **结论**: 扫描热循环的差异已全部修复，剩余性能问题定位为 SIMD 路径（见已修复 #6）

### 7. carvepool 默认 hybrid 模式问题
- **状态**: 已发现，未修复
- **问题**: `CarveCommands.cpp` 第682行 `string scanMode = "hybrid"` — 默认走 `ScanHybridMode`，而非 `ScanForFileTypesThreadPool`
- **`ScanHybridMode` 流程**:
  1. Phase 1: 调用 `ScanForFileTypesThreadPool` 做签名扫描（全盘）
  2. Phase 2: 如果 ML 模型已加载 + 有 mlOnlyTypes，再次全盘扫描做 ML 分类
- **影响**: 即使用户只扫 zip/jpg 等纯签名类型，hybrid 模式仍会尝试 Phase 2（如果 ML 可用）
- **建议**: 考虑将默认模式改为 `sig`，或在 hybrid 模式下检查是否确实有 mlOnlyTypes 需要扫描

### 8. CarvedFileInfo 结构体膨胀
- **状态**: 已知，影响程度待评估
- **对比**:
  - 旧版: ~100 字节（7 个字段 + 2 个 string）
  - 新版: ~300 字节（22 个字段 + 5 个 string）
- **影响**: 3x 内存占用、更多 cache miss、vector 扩容时更大的拷贝开销
- **评估**: 文件命中数量通常在几百到几千级别，结构体膨胀对逐字节扫描热循环本身无影响，但对结果收集和合并阶段有影响
- **建议**: 如果排查完其他问题后仍有差距，可考虑使用轻量级扫描结构体 + 延迟填充

### 9. CarvingStats 使用 atomic 类型
- **状态**: 已知，影响程度低
- **问题**: `CarvingStats` 所有字段从 `ULONGLONG` 改为 `atomic<ULONGLONG>`，但只在单一 I/O 线程中使用
- **影响**: 轻微性能开销（x86 上 atomic load/store 在单线程场景下接近零开销）

### 10. [已解决] SIMD 扫描路径需要重新设计
- **状态**: ✅ **已完成**（见已修复问题 #7）
- **解决方案**: 流式 SIMD 设计，零额外内存分配，命中后立即就地处理
- **性能验证**: 平均 +17%，峰值 +30-96%
- **可用性**: 默认禁用，可通过 `carvepool ... simd` 手动启用

---

## 当前文件修改状态

### 流式 SIMD 实现 (2026-02-07):
- `SignatureScanThreadPool.h`:
  - 移除 `SimdSignatureScanner simdScanner` 依赖
  - 添加 `simdLevel_`、`simdTargetBytes_`、`targetByteBitmap_[4]` 成员
  - 添加 `IsTargetByte()` 内联位图查找
  - 改用 `#include <immintrin.h>` 和 `CpuFeatures.h`
- `SignatureScanThreadPool.cpp`:
  - 构造函数：预计算目标字节列表和 256-bit 位图
  - **完全重写 `ScanChunkSimd`**：
    - AVX2 路径：32 字节步进，mask == 0 直接跳过
    - SSE2 路径：16 字节步进
    - 流式处理：命中后立即就地验证，零额外内存分配
    - 标量尾部：使用 bitmap 快速查找
  - `GetSimdInfo` 改用 `simdLevel_` 查询

### 之前的修复 (已完成):
- `SignatureScanThreadPool.cpp`: EstimateFileSize 恢复为 O(1)、ScanChunk 分路径处理
- `FileCarver.cpp`: 移除 ole/rtf/html/xml 签名
- `CarveCommands.cpp`: sig 模式下 `carver.SetMLClassification(false)`

---

## 旧版 worktree

`6cab623` 版本的 worktree 位于:
```
D:\Users\21405\source\repos\Filerestore_CLI_old\
```
编译产物:
```
D:\Users\21405\source\repos\Filerestore_CLI_old\x64\Release\Filerestore_CLI.exe
```
用完后清理: `git worktree remove Filerestore_CLI_old`

---

## 性能优化总结

### ✅ 已完成
- 标量扫描性能恢复到基线 2.3 GB/s
- 流式 SIMD 实现，平均 2.7 GB/s，峰值 3.0-4.5 GB/s
- 消除了旧 SIMD 设计的灾难性性能问题

### 📊 当前瓶颈分析（PCIe 4.0 x4 NVMe）
- 磁盘物理上限: ~6 GB/s
- 当前平均吞吐: 2.7 GB/s（占 45%）
- 当前峰值吞吐: 4.5 GB/s（占 75%）
- **主要瓶颈**: CPU 处理（签名验证、ZIP EOCD 搜索、ML 推理）

### 🔧 可选的进一步优化方向
1. **Issue #8（结构体膨胀）**: 如果需要进一步提升，可考虑轻量级扫描结构体
2. **签名匹配 SIMD 化**: 使用 SIMD 加速 `memcmp` 签名验证（当前仍是标量）
3. **ZIP EOCD 搜索优化**: 使用 SIMD 加速逆向搜索
4. **ML 推理优化**: 批量推理或 GPU 加速（如果 ML 是瓶颈）
