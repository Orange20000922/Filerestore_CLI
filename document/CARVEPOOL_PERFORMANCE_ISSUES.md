# carvepool 性能问题排查记录

## 基准对比

| 版本 | 吞吐量 | 备注 |
|------|--------|------|
| `029d399` (原始线程池) | ~2.3 GB/s | 基线 |
| `6cab623` (最近大更新) | ~2.3 GB/s | 正常 |
| `3b409b3` HEAD + staged/unstaged | ~700 MB/s | 异常 |

结论：性能回退发生在 **staged + unstaged changes** 中，不是已提交的代码。

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

---

## 未修复 / 待排查的问题

### 6. [核心] 6cab623 版 SignatureScanThreadPool::EstimateFileSize 实现差异
- **状态**: 未排查完成
- **线索**: `6cab623` 版本能跑满 2.3GB/s，当前版本不能。两者之间的 `SignatureScanThreadPool.cpp` 存在 staged 差异。需要对比 `6cab623` 版的 `EstimateFileSize` 实现与当前版本的差异
- **可能性**: `6cab623` 版的 `EstimateFileSize` 可能本身就是轻量级的（因为它是在 staged changes 中被替换为调用 `EstimateFileSizeStatic` 的包装），而我们的修复虽然恢复了轻量级实现，但可能与原始实现有细微差异
- **下一步**: 直接读取 `D:\Users\21405\source\repos\Filerestore_CLI_old\Filerestore_CLI\src\fileRestore\SignatureScanThreadPool.cpp` 中的 `EstimateFileSize` 实现进行对比

### 7. [核心] carvepool 默认 hybrid 模式问题
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

### 10. SIMD 扫描路径的额外内存分配
- **状态**: 已知，影响程度待评估
- **问题**: `ScanChunkSimd` 每次调用都 `matches.reserve(dataSize / 512)` 分配 MatchResult 向量
- **建议**: 可考虑使用线程局部预分配的向量

---

## 当前文件修改状态

### Staged (已暂存):
- `FileCarver.cpp`: +228 行（RefineCarvedFileInfo、SIMD 接口、EstimateFileSizeStatic ZIP 返回0 修改）
- `FileCarver.h`: +10 行（RefineCarvedFileInfo 声明、SIMD 声明）
- `CarveCommands.cpp`: +133 / -340 行（重构为使用 SaveScanResults、添加 SIMD/sig 支持、移除扫描后自动恢复流程）
- `UsnRecoverCommands.cpp`: +376 行（recover 命令集成 RefineCarvedFileInfo）
- 其他: CommandUtils.h、各 Commands 文件的小修改

### Unstaged (未暂存):
- `SignatureScanThreadPool.cpp`: EstimateFileSize 恢复为 O(1)、ScanChunk 分路径、ScanChunkSimd 实现、SIMD 初始化
- `SignatureScanThreadPool.h`: 添加 SIMD 成员和方法声明
- `FileCarver.cpp`: 移除 ole/rtf/html/xml 签名、SIMD 接口恢复原实现

### Unstaged (CarveCommands.cpp via agent-dir-1):
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

## 下次继续的优先事项

1. **对比 `6cab623` 版的 `SignatureScanThreadPool.cpp` 完整实现**，特别是 `EstimateFileSize`、`ScanChunk`、`WorkerFunction`，找出仍存在的差异
2. **构建并测试**当前修改版本，验证吞吐是否恢复
3. 如果仍有差距，考虑 Issue #8（结构体膨胀）和 Issue #10（SIMD 内存分配）
