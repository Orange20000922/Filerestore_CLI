# 测试脚本使用说明
# Test Script Usage Guide

## 📁 创建的脚本文件

项目中现在有两个测试脚本：

1. **test_file_recovery.ps1** - PowerShell 版本（功能更强大）
2. **test_file_recovery.bat** - 批处理版本（兼容性更好）

---

## 🚀 快速开始

### 方法 1：使用批处理脚本（推荐，最简单）

```cmd
# 直接双击运行
test_file_recovery.bat

# 或在命令行中运行
.\test_file_recovery.bat
```

### 方法 2：使用 PowerShell 脚本（更多选项）

```powershell
# 基础用法
.\test_file_recovery.ps1

# 创建多个不同类型的测试文件
.\test_file_recovery.ps1 -MultipleFiles

# 自定义文件名和大小
.\test_file_recovery.ps1 -TestFileName "my_test.xml" -FileSizeKB 50

# 跳过确认提示（自动运行）
.\test_file_recovery.ps1 -SkipPrompt

# 指定自定义路径
.\test_file_recovery.ps1 -TestFilePath "D:\TestFiles" -ProgramPath ".\x64\Release\ConsoleApplication5.exe"
```

---

## 📋 脚本功能

### 自动化流程

1. ✅ **创建测试目录** - 在 `C:\Temp` 创建测试文件夹（可自定义）
2. ✅ **生成测试文件** - 创建指定大小的测试文件（默认 10KB）
3. ✅ **显示文件信息** - 显示文件名、路径、大小、创建时间
4. ✅ **永久删除文件** - 绕过回收站直接删除（模拟 Shift+Delete）
5. ✅ **等待文件系统刷新** - 确保 MFT 更新
6. ✅ **自动启动程序** - 启动文件恢复工具
7. ✅ **显示测试命令** - 提示应该使用的恢复命令

---

## 🎯 PowerShell 脚本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-TestFileName` | string | `test_recovery_file.txt` | 测试文件名 |
| `-TestFilePath` | string | `C:\Temp` | 测试文件路径 |
| `-FileSizeKB` | int | `10` | 文件大小（KB） |
| `-ProgramPath` | string | `.\x64\Debug\ConsoleApplication5.exe` | 程序路径 |
| `-MultipleFiles` | switch | - | 创建多个不同类型的文件 |
| `-SkipPrompt` | switch | - | 跳过确认提示 |

---

## 📝 使用示例

### 示例 1：测试单个 TXT 文件恢复

```powershell
.\test_file_recovery.ps1
```

**脚本会：**
1. 创建 `C:\Temp\test_recovery_file.txt`（10KB）
2. 等待您按键后删除文件
3. 启动恢复程序

**建议在程序中执行：**
```
listdeleted C none
searchdeleted C test .txt
```

### 示例 2：测试多种文件类型

```powershell
.\test_file_recovery.ps1 -MultipleFiles
```

**脚本会创建并删除：**
- `test_document.txt`（5KB）
- `test_data.xml`（3KB）
- `test_image.png`（20KB）
- `test_config.json`（2KB）

**建议在程序中执行：**
```
searchdeleted C test .xml
searchdeleted C test .png
searchdeleted C test .json
```

### 示例 3：创建大文件测试

```powershell
.\test_file_recovery.ps1 -FileSizeKB 1024 -TestFileName "large_file.bin"
```

**脚本会：**
1. 创建 1MB 的测试文件
2. 删除并测试恢复

### 示例 4：自定义路径测试

```powershell
.\test_file_recovery.ps1 -TestFilePath "D:\MyTests" -TestFileName "important.docx"
```

### 示例 5：自动化测试（无需手动确认）

```powershell
.\test_file_recovery.ps1 -SkipPrompt -MultipleFiles
```

---

## ⚠️ 注意事项

### PowerShell 执行策略

如果遇到 "无法加载脚本" 错误，需要设置执行策略：

```powershell
# 临时允许（仅当前会话）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 然后运行脚本
.\test_file_recovery.ps1

# 或者一次性运行
powershell -ExecutionPolicy Bypass -File .\test_file_recovery.ps1
```

### 管理员权限

- 文件恢复工具需要**管理员权限**才能访问 MFT
- 如果程序启动后报错，请**右键 → 以管理员身份运行**

### 文件系统延迟

- 脚本会等待 2 秒确保文件系统刷新
- 如果仍然找不到文件，可能需要：
  - 等待更长时间（文件系统缓存）
  - 重启系统（彻底刷新 MFT）
  - 检查 MFT 缓存机制

---

## 🔍 测试场景

### 场景 1：验证"最近删除文件"问题

**目的**：测试刚删除的文件能否被扫描到

**步骤**：
1. 运行 `.\test_file_recovery.ps1`
2. 立即在恢复程序中执行 `listdeleted C none`
3. 查找 `test_recovery_file.txt`

**预期结果**：应该能找到刚删除的文件

### 场景 2：验证扩展名过滤

**目的**：测试不同扩展名的搜索功能

**步骤**：
1. 运行 `.\test_file_recovery.ps1 -MultipleFiles`
2. 在恢复程序中分别执行：
   - `searchdeleted C test .txt`
   - `searchdeleted C test .xml`
   - `searchdeleted C test .png`

**预期结果**：每个命令只返回对应扩展名的文件

### 场景 3：验证文件名搜索

**目的**：测试按文件名模糊搜索

**步骤**：
1. 运行 `.\test_file_recovery.ps1 -TestFileName "important_document.docx"`
2. 在恢复程序中执行 `searchdeleted C important`

**预期结果**：应该找到包含 "important" 的文件

### 场景 4：验证大文件恢复

**目的**：测试大文件的覆盖检测性能

**步骤**：
1. 运行 `.\test_file_recovery.ps1 -FileSizeKB 10240`（创建 10MB 文件）
2. 获取文件的 MFT 记录号
3. 执行 `detectoverwrite C <record_number> balanced`

**预期结果**：显示覆盖检测结果和性能指标

---

## 🛠️ 自定义脚本

### 修改默认测试路径

编辑脚本，更改默认参数：

**PowerShell 版本：**
```powershell
param(
    [string]$TestFileName = "my_custom_file.txt",    # 改这里
    [string]$TestFilePath = "D:\MyTestFolder",       # 改这里
    [int]$FileSizeKB = 50,                           # 改这里
    ...
)
```

**批处理版本：**
```batch
set TEST_DIR=D:\MyTestFolder
set TEST_FILE=my_custom_file.txt
set PROGRAM_PATH=x64\Release\ConsoleApplication5.exe
```

---

## 📊 输出示例

### PowerShell 脚本输出

```
========================================
  文件恢复工具测试脚本
  File Recovery Tool Test Script
========================================

========== Step 1: Creating Test Files ==========
[+] Created: test_document.txt (5273 bytes)
[+] Created: test_data.xml (3189 bytes)
[+] Created: test_image.png (20736 bytes)
[+] Created: test_config.json (2156 bytes)

[*] File Information:
    Name:     test_document.txt
    Path:     C:\Temp\test_document.txt
    Size:     5273 bytes
    Created:  2025-12-31 14:30:25
    Modified: 2025-12-31 14:30:25
    ---
    ...

[!] Press ANY KEY to DELETE the test file(s)...

========== Step 2: Deleting Test Files ==========
[*] Deleting files (bypassing Recycle Bin - permanent deletion)...
[+] Deleted: test_document.txt
[+] Deleted: test_data.xml
[+] Deleted: test_image.png
[+] Deleted: test_config.json

[*] Deletion Summary:
    Files created: 4
    Files deleted: 4
    Deleted at: 2025-12-31 14:30:30

[*] Waiting for filesystem to flush metadata...

========== Step 3: Launching Recovery Program ==========
[+] Found program: .\x64\Debug\ConsoleApplication5.exe

[*] Suggested test commands:
    listdeleted C none
    searchdeleted C test .txt
    searchdeleted C test .xml
    searchdeleted C test .png
    searchdeleted C test .json

[*] Test file details:
    - test_document.txt (deleted from C:\Temp)
    - test_data.xml (deleted from C:\Temp)
    - test_image.png (deleted from C:\Temp)
    - test_config.json (deleted from C:\Temp)

[*] Launching program in 3 seconds...
[+] Program launched!

========== Test Setup Complete ==========

Test completed at: 2025-12-31 14:30:33
```

---

## 🐛 故障排除

### 问题 1：找不到程序

**错误**：`[-] ERROR: Program not found!`

**解决**：
1. 确保已编译项目（按 F7 或 Ctrl+Shift+B）
2. 检查编译配置（Debug/Release, x64/x86）
3. 手动指定路径：`-ProgramPath "完整路径"`

### 问题 2：PowerShell 脚本无法运行

**错误**：`无法加载，因为在此系统上禁止运行脚本`

**解决**：
```powershell
# 方法1：临时绕过
powershell -ExecutionPolicy Bypass -File .\test_file_recovery.ps1

# 方法2：使用批处理版本
.\test_file_recovery.bat
```

### 问题 3：测试文件找不到

**可能原因**：
1. 文件系统缓存未刷新
2. MFT 缓存未更新
3. 扫描过滤级别过高

**解决**：
1. 等待更长时间再扫描
2. 使用 `listdeleted C none` 查看所有文件
3. 检查文件是否真的被删除

### 问题 4：权限不足

**错误**：程序启动但无法访问 MFT

**解决**：
1. 关闭程序
2. 右键程序 → 以管理员身份运行
3. 或设置程序始终以管理员运行（见上文"VS调试自动管理员权限"）

---

## 📚 相关文档

- **FINAL_STATUS_REPORT.md** - 项目完整功能说明
- **SEARCH_BUG_ANALYSIS.md** - 搜索功能问题分析
- **PERFORMANCE_OPTIMIZATION.md** - 性能优化详情

---

**脚本版本**：1.0
**创建日期**：2025-12-31
**更新日期**：2025-12-31
