# 对话摘要：大文件连续性检测集成

## 背景

用户发现 `FileCarver.cpp` 中的 `EstimateFileSize` 函数没有应用之前实现的大文件连续性检测，对于大文件（超出64MB缓冲区）直接返回保守估计。

## 已完成的修改

### 1. FileCarver.h - 添加新方法声明 ✅

在第172-178行添加了：
```cpp
// 使用连续性检测估算大文件大小（当缓冲区内找不到文件尾时）
// 返回估算的文件大小，如果无法估算则返回 0
ULONGLONG EstimateFileSizeWithContinuity(
    ULONGLONG startLCN,                     // 文件起始簇号
    const FileSignature& sig,               // 文件签名
    ULONGLONG maxSize = 0                   // 最大搜索大小 (0 = 使用 sig.maxSize)
);
```

### 2. FileCarver.cpp - 实现 EstimateFileSizeWithContinuity ✅

在第536-680行添加了完整实现：
- 使用4MB块逐步读取
- 调用 `ValidateBlockContinuity` 进行ML连续性检测
- 在找到文件尾或连续性中断时返回估算大小
- 支持 zip, mp4, avi, pdf, 7z, rar 格式

### 3. FileCarver.cpp - 修改 ScanBufferMultiSignature ✅

在第818-851行修改了扫描逻辑：
```cpp
// 如果没有找到文件尾，且是大文件格式，尝试使用连续性检测
bool usedContinuityDetection = false;
if (footerPos == 0 && continuityDetectionEnabled && IsContinuityModelLoaded()) {
    // 检查是否是支持连续性检测的大文件格式
    if (sig->extension == "zip" || sig->extension == "mp4" ||
        sig->extension == "avi" || sig->extension == "pdf" ||
        sig->extension == "7z" || sig->extension == "rar") {
        // 检查估算大小是否接近缓冲区边界（可能是大文件）
        if (estimatedSize >= remaining * 0.9) {
            ULONGLONG absoluteLCN = baseLCN + (offset / bytesPerCluster);
            ULONGLONG continuitySize = EstimateFileSizeWithContinuity(absoluteLCN, *sig);
            if (continuitySize > estimatedSize) {
                estimatedSize = continuitySize;
                usedContinuityDetection = true;
            }
        }
    }
}
```

### 4. CarveCommands.cpp - 添加连续性检测选项 ✅

#### CarveCommand（第229-293行）
- 添加 `continuity` 选项支持
- 用法：`carve D zip D:\recovered\ continuity`

#### CarveCommandThreadPool（第530-620行）
- 添加 `continuity` 选项支持
- 用法：`carvepool D zip D:\recovered\ 0 continuity`

## 待完成

### 5. 构建并验证 ❌

需要执行：
```powershell
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI\Filerestore_CLI.vcxproj' /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal
```

## 使用方法

启用连续性检测后，对于大文件（ZIP、MP4等）：

1. 首先尝试在缓冲区内查找文件尾
2. 如果找不到，且文件大小接近缓冲区边界
3. 使用ML连续性检测器逐块扫描，判断相邻块是否属于同一文件
4. 当检测到连续性中断时，确定文件边界

## 命令示例

```bash
# 启用连续性检测扫描ZIP文件
carve D zip D:\recovered\ continuity

# 线程池模式，启用连续性检测
carvepool D zip D:\recovered\ 0 continuity

# 同时启用多个选项
carvepool D zip,mp4 D:\recovered\ 8 hybrid continuity
```
