# Filerestore_CLI_Tests

[中文](#中文) | [English](#english)

---

<a name="中文"></a>

## 中文

### 概述

Filerestore_CLI 的 Google Test 单元测试项目。

**最新进展** (2026-02-07):
- ✅ 完成 Google Test 1.14.0 集成（通过 NuGet）
- ✅ 创建 CLI 参数解析测试套件（26 个测试）
- ✅ 创建 SIMD 签名匹配测试套件（19 个测试）
- ✅ 配置自动化构建脚本 (`build_and_test.ps1`)
- ✅ 总计 45 个单元测试覆盖核心功能

### 测试套件

#### 1. CLI 参数解析测试 (`cli_test.cpp`)

测试命令行界面的参数解析和命令匹配功能：

- **基础命令测试** (5个)：help, exit, 无效命令, 空命令, 额外空格
- **参数验证** (7个)：缺少必填参数, 无效驱动器格式, 参数验证
- **命令匹配** (2个)：前缀匹配, 大小写不敏感
- **复杂命令** (3个)：多参数, 带空格路径, 特殊字符
- **CommandHelper** (6个)：命令元数据, 参数信息, 命令组装
- **边界条件** (3个)：超长命令, 大量参数, Unicode 字符

**总计**: 26 个测试

#### 2. 签名匹配测试 (`signature_scanner_test.cpp`)

测试 SIMD 优化的签名匹配功能（验证 SSE2/AVX2 加速正确性）：

- **基础签名匹配** (7个)：ZIP, PDF, JPG, PNG, GIF, RAR, 7z
- **不匹配测试** (2个)：错误签名, 部分匹配
- **边界条件** (6个)：
  - 数据大小 = 签名大小
  - 数据 < 签名
  - 空签名
  - 极短签名 (1-2 字节)
  - 16 字节边界 (SSE2 边界)
- **SIMD 优化验证** (3个)：
  - 短签名 (4 字节) - 触发 SSE2
  - 中等签名 (8 字节, PNG) - SSE2 优化路径
  - 长签名 (12+ 字节) - AVX2 或分段处理
- **特殊模式** (3个)：全 0, 全 1, 交替模式 (0xAA/0x55)
- **内存对齐** (2个)：非对齐访问测试（验证 `_mm_loadu_si128` 正确性）

**总计**: 19 个测试

### 构建和运行

#### 前置条件

1. **Visual Studio 2022** (带 C++ 工作负载)
2. **NuGet 包管理器** (集成在 VS 中)
3. **nuget.exe** (用于命令行包管理)

#### 快速开始（推荐）

使用自动化脚本一键构建和测试：

```powershell
# 进入测试目录
cd D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests

# 运行所有测试（Debug）
.\build_and_test.ps1

# 运行 Release 配置
.\build_and_test.ps1 -Configuration Release

# 只运行 CLI 测试
.\build_and_test.ps1 -TestFilter "CLITest.*"

# 只运行签名匹配测试
.\build_and_test.ps1 -TestFilter "SignatureScannerTest.*"

# 运行特定测试
.\build_and_test.ps1 -TestFilter "SignatureScannerTest.MatchZipSignature"
```

#### 手动构建

##### 步骤 1: 安装 Google Test

```bash
# 首次构建需要安装 NuGet 包
nuget restore Filerestore_CLI_Tests.vcxproj
```

##### 步骤 2: 使用 MSBuild 构建

```powershell
# Debug 版本
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' `
  'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj' `
  /p:Configuration=Debug /p:Platform=x64 /t:Build /v:minimal

# Release 版本
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' `
  'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj' `
  /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal
```

##### 步骤 3: 运行测试

```powershell
# 运行所有测试
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe

# 运行特定测试套件
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_filter=CLITest.*
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_filter=SignatureScannerTest.*

# 彩色输出
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_color=yes

# 生成 XML 报告
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_output=xml:test_results.xml

# 列出所有测试（不运行）
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_list_tests
```

#### 使用 Visual Studio 运行

1. 在 Visual Studio 中打开解决方案
2. 右键点击 `Filerestore_CLI_Tests` 项目
3. 选择 "设为启动项目"
4. 按 **F5** 运行测试（调试模式）或 **Ctrl+F5**（非调试模式）
5. 使用 **测试资源管理器** (Test Explorer, `Ctrl+E, T`) 查看结果

### 输出示例

```
========================================
  Filerestore_CLI Unit Test Runner
========================================
Configuration: Debug
Test Filter:   *

[1/3] Restoring NuGet packages...
  Google Test already installed

[2/3] Building test project...
  Build succeeded

[3/3] Running tests...

Executing: D:\...\Filerestore_CLI_Tests.exe --gtest_color=yes

[==========] Running 45 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 16 tests from CLITest
[ RUN      ] CLITest.HelpCommand
[       OK ] CLITest.HelpCommand (12 ms)
[ RUN      ] CLITest.ExitCommand
[       OK ] CLITest.ExitCommand (3 ms)
[ RUN      ] CLITest.InvalidCommand
[       OK ] CLITest.InvalidCommand (5 ms)
...
[----------] 16 tests from CLITest (187 ms total)

[----------] 10 tests from CommandHelperTest
[ RUN      ] CommandHelperTest.GetAllCommandNames
[       OK ] CommandHelperTest.GetAllCommandNames (1 ms)
[ RUN      ] CommandHelperTest.MatchCommandsPrefix
[       OK ] CommandHelperTest.MatchCommandsPrefix (2 ms)
...
[----------] 10 tests from CommandHelperTest (23 ms total)

[----------] 19 tests from SignatureScannerTest
[ RUN      ] SignatureScannerTest.MatchZipSignature
[       OK ] SignatureScannerTest.MatchZipSignature (0 ms)
[ RUN      ] SignatureScannerTest.MatchPngSignature
[       OK ] SignatureScannerTest.MatchPngSignature (0 ms)
[ RUN      ] SignatureScannerTest.SimdEquivalenceShort
[       OK ] SignatureScannerTest.SimdEquivalenceShort (1 ms)
...
[----------] 19 tests from SignatureScannerTest (34 ms total)

[----------] Global test environment tear-down
[==========] 45 tests from 3 test suites ran. (244 ms total)
[  PASSED  ] 45 tests.

========================================
  All tests PASSED!
========================================
```

### 测试覆盖率

#### 当前覆盖模块

- ✅ **CLI 参数解析** (`cli.cpp`, `CommandHelper.cpp`)
  - 命令解析和匹配
  - 参数验证
  - 命令元数据管理

- ✅ **签名匹配优化** (`SignatureScanThreadPool.cpp`)
  - SIMD 加速验证 (SSE2/AVX2)
  - 标量回退路径
  - 边界条件和内存安全

#### 待添加测试

- ⏳ **MFT 解析** (`MFTReader.cpp`)
  - 记录解析正确性
  - 属性提取
  - 文件名编码

- ⏳ **USN 日志解析** (`UsnJournalParser.cpp`)
  - 日志记录解析
  - 时间戳处理
  - 变更原因判断

- ⏳ **文件修复** (`FileRepair.cpp`)
  - ZIP 修复算法
  - Office 文档修复
  - PNG 修复

- ⏳ **缓存系统** (`FileCache.cpp`)
  - 序列化/反序列化
  - 缓存命中率
  - 并发安全

### 持续集成

#### GitHub Actions 示例

```yaml
name: Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup MSBuild
        uses: microsoft/setup-msbuild@v1

      - name: Setup NuGet
        uses: nuget/setup-nuget@v1

      - name: Restore NuGet packages
        run: nuget restore Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj

      - name: Build Tests
        run: |
          msbuild Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj `
            /p:Configuration=Release /p:Platform=x64 /t:Build

      - name: Run Tests
        run: |
          .\x64\Release\Tests\Filerestore_CLI_Tests.exe --gtest_output=xml:test_results.xml

      - name: Publish Test Results
        uses: EnricoMi/publish-unit-test-result-action/composite@v1
        if: always()
        with:
          files: test_results.xml
```

### 故障排除

#### 问题：NuGet 包无法下载

```bash
# 手动下载 Google Test
nuget install gtest -Version 1.14.0 -OutputDirectory ..\packages

# 或使用 Visual Studio 包管理器控制台
Install-Package gtest -Version 1.14.0
```

#### 问题：链接错误 (unresolved external symbol)

确保包含路径正确：

```xml
<AdditionalIncludeDirectories>
  $(SolutionDir)Filerestore_CLI\src;
  $(SolutionDir)packages\gtest.1.14.0\build\native\include;
</AdditionalIncludeDirectories>
```

检查库路径：

```xml
<AdditionalLibraryDirectories>
  $(SolutionDir)packages\gtest.1.14.0\build\native\lib\x64\v143\$(Configuration);
</AdditionalLibraryDirectories>
```

#### 问题：测试运行时崩溃

1. 检查 DLL 依赖：
   ```powershell
   dumpbin /dependents .\x64\Debug\Tests\Filerestore_CLI_Tests.exe
   ```

2. 确保测试 fixture 正确清理：
   ```cpp
   void TearDown() override {
       // 清理资源
   }
   ```

3. 检查静态变量初始化顺序

#### 问题：某些测试在 CI 中失败

- 文件路径硬编码：使用相对路径或环境变量
- 权限问题：某些测试可能需要管理员权限（MFT/USN 访问）
- 时区/语言依赖：使用固定的 locale 设置

### 最佳实践

1. **每次提交前运行测试**
   ```bash
   # 在 git commit 前执行
   .\build_and_test.ps1
   ```

2. **TDD (测试驱动开发) 流程**
   - 🔴 编写失败的测试
   - 🟢 实现最小功能使测试通过
   - 🔵 重构优化代码
   - 🔁 重复

3. **保持测试独立**
   - 每个测试应该独立运行
   - 不依赖其他测试的状态
   - 使用 `SetUp()` 和 `TearDown()` 管理资源

4. **使用有意义的测试名称**
   - ✅ `MatchZipSignature` - 清晰描述测试内容
   - ❌ `Test1`, `TestCase2` - 无意义

5. **覆盖边界条件**
   - 空输入
   - 极大/极小值
   - 非法参数
   - 内存边界（对齐/非对齐）

6. **性能测试使用 DISABLED_ 前缀**
   ```cpp
   TEST_F(MyTest, DISABLED_PerformanceBenchmark) {
       // 仅在需要时手动运行
   }
   ```

7. **Mock 外部依赖**
   - 对于需要管理员权限的测试，创建 Mock 类
   - 对于文件系统访问，使用虚拟文件系统

### 项目结构

```
Filerestore_CLI_Tests/
├── tests/
│   ├── cli_test.cpp                  # CLI 参数解析测试 (26 个)
│   └── signature_scanner_test.cpp    # SIMD 签名匹配测试 (19 个)
├── mocks/                             # Mock 类（待添加）
├── Filerestore_CLI_Tests.vcxproj     # Visual Studio 项目文件
├── packages.config                    # NuGet 包配置
├── build_and_test.ps1                # 自动化构建脚本
└── README.md                         # 本文档
```

### 相关文档

- [Google Test 官方文档](https://google.github.io/googletest/)
- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [AUTO_TEST_GUIDE.md](../document/AUTO_TEST_GUIDE.md) - 自动化测试指南（集成测试）
- [CLAUDE.md](../CLAUDE.md) - 项目构建配置

### 贡献

#### 添加新测试

1. 在 `tests/` 目录创建 `<module>_test.cpp`
2. 编写测试用例：
   ```cpp
   #include <gtest/gtest.h>
   #include "../../Filerestore_CLI/src/<module>.h"

   TEST(ModuleTest, TestName) {
       // Arrange
       // Act
       // Assert
   }
   ```
3. 在 `Filerestore_CLI_Tests.vcxproj` 添加：
   ```xml
   <ClCompile Include="tests\<module>_test.cpp" />
   ```
4. 重新构建并运行：
   ```powershell
   .\build_and_test.ps1
   ```
5. 更新本 README 的测试覆盖率部分

#### 代码风格

- 遵循 Google C++ Style Guide
- 测试类名：`<Module>Test`
- 测试用例名：描述性驼峰命名，如 `MatchZipSignature`
- 使用 `EXPECT_*` 进行非致命断言，`ASSERT_*` 进行致命断言

---

<a name="english"></a>

## English

### Overview

Google Test unit testing project for Filerestore_CLI.

**Latest Progress** (2026-02-07):
- ✅ Completed Google Test 1.14.0 integration (via NuGet)
- ✅ Created CLI argument parsing test suite (26 tests)
- ✅ Created SIMD signature matching test suite (19 tests)
- ✅ Configured automated build script (`build_and_test.ps1`)
- ✅ Total 45 unit tests covering core functionality

### Test Suites

#### 1. CLI Argument Parsing Tests (`cli_test.cpp`)

Tests command-line interface argument parsing and command matching:

- **Basic Command Tests** (5): help, exit, invalid commands, empty commands, extra spaces
- **Argument Validation** (7): missing required arguments, invalid drive formats, parameter validation
- **Command Matching** (2): prefix matching, case insensitivity
- **Complex Commands** (3): multi-parameter, paths with spaces, special characters
- **CommandHelper** (6): command metadata, parameter info, command assembly
- **Boundary Conditions** (3): very long commands, many parameters, Unicode characters

**Total**: 26 tests

#### 2. Signature Matching Tests (`signature_scanner_test.cpp`)

Tests SIMD-optimized signature matching (validates SSE2/AVX2 acceleration correctness):

- **Basic Signature Matching** (7): ZIP, PDF, JPG, PNG, GIF, RAR, 7z
- **No-Match Tests** (2): wrong signature, partial match
- **Boundary Conditions** (6):
  - Data size = signature size
  - Data < signature
  - Empty signature
  - Very short signatures (1-2 bytes)
  - 16-byte boundary (SSE2 boundary)
- **SIMD Optimization Validation** (3):
  - Short signature (4 bytes) - triggers SSE2
  - Medium signature (8 bytes, PNG) - SSE2 optimized path
  - Long signature (12+ bytes) - AVX2 or chunked processing
- **Special Patterns** (3): all zeros, all ones, alternating pattern (0xAA/0x55)
- **Memory Alignment** (2): unaligned access tests (validates `_mm_loadu_si128` correctness)

**Total**: 19 tests

### Build and Run

#### Prerequisites

1. **Visual Studio 2022** (with C++ workload)
2. **NuGet Package Manager** (integrated in VS)
3. **nuget.exe** (for command-line package management)

#### Quick Start (Recommended)

Use the automated script for one-click build and test:

```powershell
# Navigate to test directory
cd D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests

# Run all tests (Debug)
.\build_and_test.ps1

# Run Release configuration
.\build_and_test.ps1 -Configuration Release

# Run only CLI tests
.\build_and_test.ps1 -TestFilter "CLITest.*"

# Run only signature matching tests
.\build_and_test.ps1 -TestFilter "SignatureScannerTest.*"

# Run specific test
.\build_and_test.ps1 -TestFilter "SignatureScannerTest.MatchZipSignature"
```

#### Manual Build

##### Step 1: Install Google Test

```bash
# First-time build requires NuGet package installation
nuget restore Filerestore_CLI_Tests.vcxproj
```

##### Step 2: Build with MSBuild

```powershell
# Debug build
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' `
  'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj' `
  /p:Configuration=Debug /p:Platform=x64 /t:Build /v:minimal

# Release build
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' `
  'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj' `
  /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal
```

##### Step 3: Run Tests

```powershell
# Run all tests
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe

# Run specific test suite
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_filter=CLITest.*
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_filter=SignatureScannerTest.*

# Colored output
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_color=yes

# Generate XML report
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_output=xml:test_results.xml

# List all tests (without running)
.\x64\Debug\Tests\Filerestore_CLI_Tests.exe --gtest_list_tests
```

#### Using Visual Studio

1. Open solution in Visual Studio
2. Right-click `Filerestore_CLI_Tests` project
3. Select "Set as Startup Project"
4. Press **F5** to run tests (debug mode) or **Ctrl+F5** (non-debug)
5. Use **Test Explorer** (`Ctrl+E, T`) to view results

### Output Example

```
========================================
  Filerestore_CLI Unit Test Runner
========================================
Configuration: Debug
Test Filter:   *

[1/3] Restoring NuGet packages...
  Google Test already installed

[2/3] Building test project...
  Build succeeded

[3/3] Running tests...

Executing: D:\...\Filerestore_CLI_Tests.exe --gtest_color=yes

[==========] Running 45 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 16 tests from CLITest
[ RUN      ] CLITest.HelpCommand
[       OK ] CLITest.HelpCommand (12 ms)
[ RUN      ] CLITest.ExitCommand
[       OK ] CLITest.ExitCommand (3 ms)
[ RUN      ] CLITest.InvalidCommand
[       OK ] CLITest.InvalidCommand (5 ms)
...
[----------] 16 tests from CLITest (187 ms total)

[----------] 10 tests from CommandHelperTest
[ RUN      ] CommandHelperTest.GetAllCommandNames
[       OK ] CommandHelperTest.GetAllCommandNames (1 ms)
[ RUN      ] CommandHelperTest.MatchCommandsPrefix
[       OK ] CommandHelperTest.MatchCommandsPrefix (2 ms)
...
[----------] 10 tests from CommandHelperTest (23 ms total)

[----------] 19 tests from SignatureScannerTest
[ RUN      ] SignatureScannerTest.MatchZipSignature
[       OK ] SignatureScannerTest.MatchZipSignature (0 ms)
[ RUN      ] SignatureScannerTest.MatchPngSignature
[       OK ] SignatureScannerTest.MatchPngSignature (0 ms)
[ RUN      ] SignatureScannerTest.SimdEquivalenceShort
[       OK ] SignatureScannerTest.SimdEquivalenceShort (1 ms)
...
[----------] 19 tests from SignatureScannerTest (34 ms total)

[----------] Global test environment tear-down
[==========] 45 tests from 3 test suites ran. (244 ms total)
[  PASSED  ] 45 tests.

========================================
  All tests PASSED!
========================================
```

### Test Coverage

#### Currently Covered Modules

- ✅ **CLI Argument Parsing** (`cli.cpp`, `CommandHelper.cpp`)
  - Command parsing and matching
  - Argument validation
  - Command metadata management

- ✅ **Signature Matching Optimization** (`SignatureScanThreadPool.cpp`)
  - SIMD acceleration validation (SSE2/AVX2)
  - Scalar fallback path
  - Boundary conditions and memory safety

#### Tests To Be Added

- ⏳ **MFT Parsing** (`MFTReader.cpp`)
  - Record parsing correctness
  - Attribute extraction
  - Filename encoding

- ⏳ **USN Journal Parsing** (`UsnJournalParser.cpp`)
  - Journal record parsing
  - Timestamp handling
  - Change reason detection

- ⏳ **File Repair** (`FileRepair.cpp`)
  - ZIP repair algorithms
  - Office document repair
  - PNG repair

- ⏳ **Cache System** (`FileCache.cpp`)
  - Serialization/deserialization
  - Cache hit rate
  - Concurrency safety

### Continuous Integration

#### GitHub Actions Example

```yaml
name: Unit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup MSBuild
        uses: microsoft/setup-msbuild@v1

      - name: Setup NuGet
        uses: nuget/setup-nuget@v1

      - name: Restore NuGet packages
        run: nuget restore Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj

      - name: Build Tests
        run: |
          msbuild Filerestore_CLI_Tests\Filerestore_CLI_Tests.vcxproj `
            /p:Configuration=Release /p:Platform=x64 /t:Build

      - name: Run Tests
        run: |
          .\x64\Release\Tests\Filerestore_CLI_Tests.exe --gtest_output=xml:test_results.xml

      - name: Publish Test Results
        uses: EnricoMi/publish-unit-test-result-action/composite@v1
        if: always()
        with:
          files: test_results.xml
```

### Troubleshooting

#### Issue: NuGet Package Download Fails

```bash
# Manually download Google Test
nuget install gtest -Version 1.14.0 -OutputDirectory ..\packages

# Or use Visual Studio Package Manager Console
Install-Package gtest -Version 1.14.0
```

#### Issue: Linker Error (unresolved external symbol)

Ensure include paths are correct:

```xml
<AdditionalIncludeDirectories>
  $(SolutionDir)Filerestore_CLI\src;
  $(SolutionDir)packages\gtest.1.14.0\build\native\include;
</AdditionalIncludeDirectories>
```

Check library paths:

```xml
<AdditionalLibraryDirectories>
  $(SolutionDir)packages\gtest.1.14.0\build\native\lib\x64\v143\$(Configuration);
</AdditionalLibraryDirectories>
```

#### Issue: Tests Crash at Runtime

1. Check DLL dependencies:
   ```powershell
   dumpbin /dependents .\x64\Debug\Tests\Filerestore_CLI_Tests.exe
   ```

2. Ensure test fixtures clean up properly:
   ```cpp
   void TearDown() override {
       // Clean up resources
   }
   ```

3. Check static variable initialization order

#### Issue: Some Tests Fail in CI

- Hardcoded file paths: Use relative paths or environment variables
- Permission issues: Some tests may require admin rights (MFT/USN access)
- Timezone/locale dependency: Use fixed locale settings

### Best Practices

1. **Run Tests Before Each Commit**
   ```bash
   # Execute before git commit
   .\build_and_test.ps1
   ```

2. **TDD (Test-Driven Development) Workflow**
   - 🔴 Write a failing test
   - 🟢 Implement minimum code to pass
   - 🔵 Refactor and optimize
   - 🔁 Repeat

3. **Keep Tests Independent**
   - Each test should run independently
   - Don't depend on other test states
   - Use `SetUp()` and `TearDown()` to manage resources

4. **Use Meaningful Test Names**
   - ✅ `MatchZipSignature` - clearly describes test content
   - ❌ `Test1`, `TestCase2` - meaningless

5. **Cover Boundary Conditions**
   - Empty input
   - Maximum/minimum values
   - Invalid arguments
   - Memory boundaries (aligned/unaligned)

6. **Use DISABLED_ Prefix for Performance Tests**
   ```cpp
   TEST_F(MyTest, DISABLED_PerformanceBenchmark) {
       // Only run manually when needed
   }
   ```

7. **Mock External Dependencies**
   - For tests requiring admin privileges, create mock classes
   - For file system access, use virtual file systems

### Project Structure

```
Filerestore_CLI_Tests/
├── tests/
│   ├── cli_test.cpp                  # CLI argument parsing tests (26)
│   └── signature_scanner_test.cpp    # SIMD signature matching tests (19)
├── mocks/                             # Mock classes (to be added)
├── Filerestore_CLI_Tests.vcxproj     # Visual Studio project file
├── packages.config                    # NuGet package configuration
├── build_and_test.ps1                # Automated build script
└── README.md                         # This document
```

### Related Documentation

- [Google Test Official Documentation](https://google.github.io/googletest/)
- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [AUTO_TEST_GUIDE.md](../document/AUTO_TEST_GUIDE.md) - Automated Testing Guide (Integration Tests)
- [CLAUDE.md](../CLAUDE.md) - Project Build Configuration

### Contributing

#### Adding New Tests

1. Create `<module>_test.cpp` in `tests/` directory
2. Write test cases:
   ```cpp
   #include <gtest/gtest.h>
   #include "../../Filerestore_CLI/src/<module>.h"

   TEST(ModuleTest, TestName) {
       // Arrange
       // Act
       // Assert
   }
   ```
3. Add to `Filerestore_CLI_Tests.vcxproj`:
   ```xml
   <ClCompile Include="tests\<module>_test.cpp" />
   ```
4. Rebuild and run:
   ```powershell
   .\build_and_test.ps1
   ```
5. Update test coverage section in this README

#### Code Style

- Follow Google C++ Style Guide
- Test class name: `<Module>Test`
- Test case name: descriptive camelCase, e.g., `MatchZipSignature`
- Use `EXPECT_*` for non-fatal assertions, `ASSERT_*` for fatal assertions

---

**Version**: 1.0.0
**Last Updated**: 2026-02-07
