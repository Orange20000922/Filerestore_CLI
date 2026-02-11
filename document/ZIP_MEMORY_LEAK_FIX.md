# ZIP 内存泄漏修复 - 流式写入实现

## 问题
ZIP 扫描时，如果找不到 EOCD 头，`fileBuffer` 会无限增长（每次 +4MB），导致：
- 单线程：最多 4GB 内存占用 → OOM
- 8 线程：32GB 内存占用 → 系统崩溃

**根源**：`RecoverZipWithEOCDScan()` (FileCarver.cpp:3715)

---

## 解决方案

### 智能自适应策略

```
├─ expectedSize > 0 且 ≤ 256MB → 内存缓冲（快速）
├─ expectedSize > 256MB → 流式写入（32MB 滚动缓冲）
└─ expectedSize = 0 (未知) → 流式写入 + 512MB 硬限制
```

### 修改内容

#### 1. FileCarver.h
- 添加常量：`MEMORY_BUFFER_THRESHOLD = 256MB`, `MAX_STREAMING_LIMIT = 512MB`
- 添加两个私有函数：
  - `RecoverZipWithEOCDScan_MemoryBuffer()` - 内存模式
  - `RecoverZipWithEOCDScan_Streaming()` - 流式模式

#### 2. FileCarver.cpp
- 改造 `RecoverZipWithEOCDScan()` 为智能路由器
- 重命名原实现为 `_MemoryBuffer` + 添加安全检查
- 新增 `_Streaming` 实现（32MB 滚动缓冲，自动刷新）

---

## 性能对比

| 场景 | 大小 | 模式 | 峰值内存 | 速度 |
|------|------|------|---------|------|
| 正常 ZIP | 100MB | 内存 | 100MB | ⚡⚡⚡ |
| 大 ZIP | 1GB | 流式 | 32MB | ⚡⚡ |
| 损坏 ZIP（无 EOCD） | 未知 | 流式 + 限制 | 32MB | ⚡⚡ |
| 8 线程损坏 ZIP | 未知 | 流式 | **256MB** | ✅ 安全 |

**修复前**：8 线程 × 4GB = 32GB → OOM 崩溃
**修复后**：8 线程 × 32MB = 256MB → 稳定运行

---

## 使用建议

### 如果知道文件大小（推荐）
```cpp
ZipRecoveryConfig config;
config.expectedSize = mftRecord.dataSize;  // 从 MFT 获取
carver.RecoverZipWithEOCDScan(startLCN, outputPath, config);
// 自动选择最优模式
```

### 未知大小
```cpp
ZipRecoveryConfig config;
// config.expectedSize = 0;  // 默认
carver.RecoverZipWithEOCDScan(startLCN, outputPath, config);
// 自动使用流式 + 512MB 限制
```

---

**修复日期**: 2026-02-08
**影响版本**: v0.3.3+
**状态**: ✅ 已修复
