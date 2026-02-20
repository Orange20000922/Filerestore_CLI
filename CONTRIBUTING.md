# Contributing to Filerestore_CLI

**简体中文** | [English](#english)

感谢你对 Filerestore_CLI 的关注！欢迎提交 Bug 报告、功能建议和代码贡献。

## 开始之前

1. 阅读 [README.md](README.md) 了解项目概况
2. 查看 [Issues](https://github.com/Orange20000922/Filerestore_CLI/issues) 确认是否已有相关讨论
3. 对于大型改动，请先开 Issue 讨论方案

## 开发环境搭建

### 前置条件

- Windows 10/11 (x64)
- Visual Studio 2022（v143 工具集，C++20）
- CMake 3.20+（用于构建 FTXUI 依赖）
- 管理员权限（运行时需要，磁盘原始读取）

### 构建步骤

```bash
# 克隆
git clone https://github.com/Orange20000922/Filerestore_CLI.git
cd Filerestore_CLI

# 设置 FTXUI
git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui
cd Filerestore_CLI/deps/ftxui && mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ../../../..

# 构建主项目
msbuild Filerestore_CLI/Filerestore_CLI.vcxproj /p:Configuration=Release /p:Platform=x64
```

### 可选：ONNX Runtime（ML 功能）

下载 [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) 并解压到 `Filerestore_CLI/deps/onnxruntime/`。

## 提交 Pull Request

### 分支策略

- `master` — 稳定分支，所有 PR 合入此分支
- `feature/*` — 功能开发分支
- `fix/*` — Bug 修复分支

### PR 流程

1. Fork 并创建功能分支：`git checkout -b feature/my-feature`
2. 编写代码并确保 Release x64 编译通过（0 error）
3. 提交并推送：`git push origin feature/my-feature`
4. 创建 Pull Request，描述改动内容和测试方法

### 代码规范

- C++20 标准
- 文件编码：UTF-8 with BOM（Visual Studio 默认）
- 缩进：4 空格（.cpp/.h）、制表符（vcxproj）
- 命名：
  - 类名：`PascalCase`（如 `ClusterFilteredReader`）
  - 方法名：`PascalCase`（如 `ReadFromDataRuns`）
  - 成员变量：`camelCase_` 尾部下划线（如 `reader_`）
  - 局部变量：`camelCase`
- 新增 `.cpp`/`.h` 文件必须同步更新 `.vcxproj` 和 `.vcxproj.filters`
- 注释：代码自解释为主，复杂逻辑用中文或英文注释均可

### 提交消息格式

```
<类型>: <简短描述>

<详细说明（可选）>
```

类型：`feat`（新功能）、`fix`（修复）、`refactor`（重构）、`docs`（文档）、`perf`（性能）

## 模块说明

| 目录 | 职责 | 改动注意事项 |
|------|------|-------------|
| `src/core/` | CLI 入口、API 层 | 改动 `FileRestoreAPI` 需同步更新头文件接口 |
| `src/commands/` | 命令实现 | 新命令需在 `climodule.cpp` 中注册 |
| `src/fileRestore/` | 恢复核心引擎 | 底层模块被多路径复用，改动需回归测试所有路径 |
| `src/tui/` | TUI 界面 | 依赖 FTXUI，改动需测试 `--tui` 模式 |
| `src/utils/` | 工具类 | `Logger` 全局使用，接口改动影响面大 |

## Bug 报告

提交 Issue 时请包含：

1. 操作系统版本和磁盘类型（HDD/SSD/NVMe）
2. 复现步骤
3. 预期行为 vs 实际行为
4. 错误日志（如果有 `debug.log`）

## 安全问题

安全漏洞请不要通过公开 Issue 报告，请参见 [SECURITY.md](SECURITY.md)。

---

<a name="english"></a>

[简体中文](#contributing-to-filerestore_cli) | **English**

# Contributing to Filerestore_CLI

Thank you for your interest in Filerestore_CLI! Bug reports, feature requests, and code contributions are welcome.

## Before You Start

1. Read [README.md](README.md) for project overview
2. Check [Issues](https://github.com/Orange20000922/Filerestore_CLI/issues) for existing discussions
3. For major changes, please open an Issue first to discuss the approach

## Development Setup

### Prerequisites

- Windows 10/11 (x64)
- Visual Studio 2022 (v143 toolset, C++20)
- CMake 3.20+ (for building FTXUI dependency)
- Administrator privileges (required at runtime for raw disk access)

### Build Steps

```bash
# Clone
git clone https://github.com/Orange20000922/Filerestore_CLI.git
cd Filerestore_CLI

# Setup FTXUI
git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui
cd Filerestore_CLI/deps/ftxui && mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ../../../..

# Build main project
msbuild Filerestore_CLI/Filerestore_CLI.vcxproj /p:Configuration=Release /p:Platform=x64
```

### Optional: ONNX Runtime (ML features)

Download [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) and extract to `Filerestore_CLI/deps/onnxruntime/`.

## Pull Requests

### Branch Strategy

- `master` — Stable branch, all PRs merge here
- `feature/*` — Feature development
- `fix/*` — Bug fixes

### PR Workflow

1. Fork and create a feature branch: `git checkout -b feature/my-feature`
2. Write code and ensure Release x64 builds with 0 errors
3. Push: `git push origin feature/my-feature`
4. Create a Pull Request describing changes and how to test

### Code Style

- C++20 standard
- Encoding: UTF-8 with BOM (Visual Studio default)
- Indentation: 4 spaces (.cpp/.h), tabs (vcxproj)
- Naming:
  - Classes: `PascalCase` (e.g. `ClusterFilteredReader`)
  - Methods: `PascalCase` (e.g. `ReadFromDataRuns`)
  - Member variables: `camelCase_` with trailing underscore (e.g. `reader_`)
  - Local variables: `camelCase`
- New `.cpp`/`.h` files must be added to both `.vcxproj` and `.vcxproj.filters`
- Comments: Self-documenting code preferred; Chinese or English comments both acceptable

### Commit Message Format

```
<type>: <short description>

<detailed explanation (optional)>
```

Types: `feat` (feature), `fix` (bugfix), `refactor`, `docs`, `perf` (performance)

## Module Overview

| Directory | Responsibility | Notes |
|-----------|---------------|-------|
| `src/core/` | CLI entry, API layer | Changes to `FileRestoreAPI` require header sync |
| `src/commands/` | Command implementations | New commands must be registered in `climodule.cpp` |
| `src/fileRestore/` | Core recovery engine | Low-level modules are shared across recovery paths; test all paths on changes |
| `src/tui/` | TUI interface | Depends on FTXUI; test with `--tui` flag |
| `src/utils/` | Utilities | `Logger` is used globally; interface changes have wide impact |

## Bug Reports

When filing an Issue, please include:

1. OS version and disk type (HDD/SSD/NVMe)
2. Steps to reproduce
3. Expected vs actual behavior
4. Error logs (if `debug.log` exists)

## Security Issues

Do not report security vulnerabilities via public Issues. See [SECURITY.md](SECURITY.md).
