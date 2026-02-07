# Filerestore_CLI 自动化测试能力

## 概述

通过 `--cmd` 选项和日志系统，现在可以实现完整的自动化测试流程：

**核心能力**：
- ✅ 命令行接口（--cmd）- 无需交互直接执行
- ✅ 日志系统（debug.log）- 记录执行细节和性能指标
- ✅ 退出码（Exit Code）- 成功/失败状态
- ✅ 标准输出（stdout）- 捕获命令结果

---

## 测试工具

### 1. PowerShell 测试套件（推荐）

**文件**: `document/auto_test.ps1`

**功能**：
- 完整的测试框架（参数验证、功能测试、性能分析）
- 自动生成 JSON 测试报告
- 从日志提取性能指标
- 彩色输出和进度提示

**使用方法**：
```powershell
cd document
.\auto_test.ps1

# 指定测试驱动器
.\auto_test.ps1 -TestDrive "E:"

# 跳过慢速测试
.\auto_test.ps1 -SkipSlowTests

# 自定义可执行文件路径
.\auto_test.ps1 -ExePath "C:\path\to\Filerestore_CLI.exe"
```

**输出示例**：
```
========================================
Basic Functionality Tests
========================================
[INFO] Running: Help Command
[PASS] Help Command (0.12s)
[INFO] Running: Invalid Command
[PASS] Invalid Command (0.08s)

========================================
Test Summary
========================================
Total tests:   8
Passed:        7
Failed:        1
Avg Duration:  1.43s

Detailed report saved to: test_report_20260207_213045.json
```

---

### 2. 批处理快速测试

**文件**: `document/quick_test.bat`

**功能**：
- 快速基础测试（4个测试用例）
- 验证核心功能
- 检查日志文件
- 适合 CI/CD 集成

**使用方法**：
```batch
cd document
quick_test.bat

REM 指定测试驱动器
quick_test.bat E:
```

---

## 测试场景

### 基础功能测试

```powershell
# 帮助系统
Filerestore_CLI.exe --cmd "help"
Filerestore_CLI.exe --cmd "help listdeleted"

# 命令验证
Filerestore_CLI.exe --cmd "exit"
Filerestore_CLI.exe --cmd "invalid_command"  # 应返回错误
```

### 参数验证测试

```powershell
# 缺少必填参数
Filerestore_CLI.exe --cmd "listdeleted"  # 缺少 drive
Filerestore_CLI.exe --cmd "carvepool"    # 缺少所有参数

# 无效参数
Filerestore_CLI.exe --cmd "recover XYZ: test.txt"  # 无效驱动器
Filerestore_CLI.exe --cmd "carvepool D: invalid_type out"  # 无效类型
```

### 功能测试（需要有效驱动器）

```powershell
# 驱动器诊断
Filerestore_CLI.exe --cmd "checkdrive D:"

# 列出已删除文件（快速）
Filerestore_CLI.exe --cmd "listdeleted D: all"

# 深度扫描（慢）
Filerestore_CLI.exe --cmd "carvepool D: jpg D:\out 8"

# 智能恢复
Filerestore_CLI.exe --cmd "recover D: document.docx D:\out"

# 浏览结果
Filerestore_CLI.exe --cmd "carvelist"
```

### 性能测试

```powershell
# 使用 Measure-Command 测量执行时间
$time = Measure-Command {
    .\Filerestore_CLI.exe --cmd "listdeleted D:"
}
Write-Host "Execution time: $($time.TotalSeconds)s"

# 批量测试
1..10 | ForEach-Object {
    Measure-Command {
        .\Filerestore_CLI.exe --cmd "checkdrive D:"
    }
} | Measure-Object -Property TotalSeconds -Average
```

---

## 日志分析

### 从日志提取性能指标

```powershell
# 读取最新日志
$log = Get-Content ..\x64\Release\debug.log -Tail 200 -Encoding UTF8

# 扫描统计
$log | Select-String "已扫描.*找到"

# 缓存命中率
$log | Select-String "命中率"

# 执行时间
$log | Select-String "Total records written"

# 警告/错误
$log | Select-String "\[WARNING\]|\[ERROR\]"
```

### 日志中的关键指标

从 `debug.log` 可以提取：

1. **扫描性能**：
   ```
   已扫描: 718832, 找到: 34 个已删除文件
   ```

2. **缓存效率**：
   ```
   命中率=25.93%
   ```

3. **内存/存储**：
   ```
   保存 34 个文件到缓存
   ```

4. **异常情况**：
   ```
   [WARNING] Suspiciously large data run: 1090941983232 clusters
   ```

---

## 自动化测试示例

### CI/CD 集成（GitHub Actions 示例）

```yaml
name: Filerestore_CLI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build
        run: |
          msbuild Filerestore_CLI.vcxproj /p:Configuration=Release /p:Platform=x64

      - name: Run Tests
        run: |
          cd document
          .\auto_test.ps1 -SkipSlowTests

      - name: Upload Test Report
        uses: actions/upload-artifact@v2
        with:
          name: test-report
          path: document/test_report_*.json

      - name: Upload Logs
        uses: actions/upload-artifact@v2
        with:
          name: debug-logs
          path: x64/Release/debug.log
```

### 回归测试脚本

```powershell
# regression_test.ps1
$baseline = Import-Csv baseline_performance.csv

$currentTests = @(
    @{Cmd="listdeleted D:"; ExpectedTime=5.0},
    @{Cmd="checkdrive D:"; ExpectedTime=0.5},
    @{Cmd="carvelist"; ExpectedTime=0.1}
)

foreach ($test in $currentTests) {
    $time = Measure-Command {
        .\Filerestore_CLI.exe --cmd $test.Cmd
    }

    $baseline = $test.ExpectedTime
    $actual = $time.TotalSeconds
    $ratio = $actual / $baseline

    if ($ratio -gt 1.2) {
        Write-Warning "Performance regression detected: $($test.Cmd)"
        Write-Warning "Expected: ${baseline}s, Actual: ${actual}s (${ratio}x slower)"
    }
}
```

---

## 测试驱动开发流程

### 1. 编写测试用例
```powershell
# test_new_feature.ps1
$result = & .\Filerestore_CLI.exe --cmd "newfeature param1 param2" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Feature test failed"
}
```

### 2. 运行失败（红）
```
[FAIL] New Feature - Exit Code 1
```

### 3. 实现功能

### 4. 测试通过（绿）
```
[PASS] New Feature (1.23s)
```

### 5. 重构优化
```
[PASS] New Feature (0.87s)  # 性能提升 29%
```

---

## 我可以做的测试

作为 Claude Code，我现在可以：

### ✅ 已验证能力

1. **执行命令并捕获输出**
   ```powershell
   Filerestore_CLI.exe --cmd "help"
   ```

2. **读取和分析日志**
   ```
   Direct command mode: help
   Command execution completed.
   ```

3. **验证退出码**
   ```
   Exit Code: 0 (Success)
   Exit Code: 1 (Failure)
   ```

### ⚠️ 限制

1. **需要管理员权限** - MFT/USN 访问需要提升权限
2. **需要真实数据** - 某些测试需要已删除文件
3. **PowerShell 执行策略** - 可能需要 `Set-ExecutionPolicy`

### 🎯 推荐测试流程

**你提供**：
- 测试驱动器（如 D:）
- 预期行为描述

**我执行**：
1. 运行命令：`Filerestore_CLI.exe --cmd "..."`
2. 读取日志：`debug.log` 最新条目
3. 分析结果：性能指标、错误、警告
4. 验证功能：对比预期输出
5. 生成报告：汇总测试结果

---

## 下一步

1. **运行快速测试**：
   ```batch
   cd document
   quick_test.bat D:
   ```

2. **查看测试报告**：
   检查生成的 `test_report_*.json`

3. **分析性能**：
   ```powershell
   .\auto_test.ps1 -TestDrive "D:"
   ```

4. **集成 CI/CD**：
   将测试脚本加入持续集成流程

---

## 总结

通过 `--cmd` + 日志 + 退出码，你成功打造了一个**可测试、可自动化、可分析**的命令行工具。现在我可以：

- ✅ 自动执行测试用例
- ✅ 验证功能正确性
- ✅ 分析性能指标
- ✅ 检测回归问题
- ✅ 生成测试报告

**这是软件工程最佳实践的完美体现！** 🎉
