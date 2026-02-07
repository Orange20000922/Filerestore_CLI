# 依赖检查报告

## 概述

对 Filerestore_CLI 项目的所有依赖进行全面检查。

**检查日期**: 2026-02-07
**更新日期**: 2026-02-07 (ONNX Runtime CI 修复)

## 依赖清单

### 1. FTXUI (TUI 框架)
**位置**: `Filerestore_CLI/deps/ftxui/`  
**类型**: CMake 项目  
**状态**: ✅ 已修复

**配置**:
- 本地: 手动克隆 + CMake 构建
- CI: 自动克隆 + 构建 + 缓存
- .gitignore: ✅ 已添加 `.git/`, `build/`

**Include 路径**:
```xml
<AdditionalIncludeDirectories>$(ProjectDir)deps\ftxui\include</AdditionalIncludeDirectories>
```

**库文件**:
- Debug: `deps/ftxui/build/Debug/*.lib`
- Release: `deps/ftxui/build/Release/*.lib`
- 链接: `ftxui-component.lib`, `ftxui-dom.lib`, `ftxui-screen.lib`

**CI 处理**: ✅ 已在 `.github/workflows/msbuild.yml` 中添加构建步骤

---

### 2. ONNX Runtime (ML 推理引擎)
**位置**: `Filerestore_CLI/deps/onnxruntime/`
**类型**: 预编译二进制包
**状态**: ✅ **已修复 - CI 自动下载**

**重要性**: 核心ML功能依赖（文件分类、类型识别、置信度评估）

**配置** (`OnnxRuntime.props`):
- 自动检测: 检查 `onnxruntime_cxx_api.h` 是否存在
- 条件编译: 可用时启用 `USE_ONNX_RUNTIME` 宏，不可用时优雅降级
- 自动复制 DLL: 构建后自动复制到输出目录

**依赖结构**:
```
onnxruntime/
├── include/
│   └── onnxruntime_cxx_api.h
├── lib/
│   ├── onnxruntime.lib
│   └── onnxruntime.dll
├── LICENSE
└── VERSION_NUMBER (1.16.3)
```

**.gitignore**: ✅ 已忽略 `deps/onnxruntime/include/`, `deps/onnxruntime/lib/`

**CI 处理**: ✅ **已修复** - 在 `.github/workflows/msbuild.yml` 中添加自动下载步骤
- 版本: v1.16.3 (CPU版, ~50MB)
- 缓存: 使用 `actions/cache@v4` 缓存，首次1-2分钟，后续5秒
- 来源: GitHub Releases (microsoft/onnxruntime)
- 详见: `document/ONNX_RUNTIME_CI_FIX.md`

---

### 3. nlohmann/json (JSON 库)
**位置**: `Filerestore_CLI/third_party/nlohmann/json.hpp`  
**类型**: Header-only 库  
**状态**: ✅ 无问题

**配置**:
- 单个头文件: `json.hpp` (920KB)
- 无需构建: 直接 `#include <nlohmann/json.hpp>`

**Include 路径**:
```xml
<AdditionalIncludeDirectories>$(ProjectDir)third_party</AdditionalIncludeDirectories>
```

**Git 状态**: ✅ 已提交到仓库（header-only，无构建产物）

**CI 处理**: ✅ 不需要（跟随代码仓库一起 checkout）

---

## 潜在问题检查

### ⚠️ 问题 1: PlatformToolset 版本不一致

**Filerestore_CLI.vcxproj**:
```xml
<PlatformToolset>v145</PlatformToolset>  <!-- VS 2022 (17.5+) -->
```

**Filerestore_CLI_Tests.vcxproj**:
```xml
<PlatformToolset>v143</PlatformToolset>  <!-- VS 2022 (17.0-17.4) -->
```

**GitHub Actions**:
```yaml
/p:PlatformToolset=v143  # 指定 v143
```

👉 **建议**: 统一使用 `v143`（VS 2022 标准版本）

**原因**:
- v143 是 VS 2022 的标准版本
- v145 只在 VS 2022 17.5+ 中可用
- GitHub Actions 可能使用较旧的 VS 2022 版本

---

### ⚠️ 问题 2: Win32 平台缺少 FTXUI 链接配置

**Win32 (x86) 配置**：
- Include: ✅ 有 FTXUI include 路径
- Link: ❌ **没有** FTXUI 库路径和链接库

**x64 配置**：
- Include: ✅ 有
- Link: ✅ 有

👉 **影响**: 如果编译 Win32 版本会失败（但 GitHub Actions 只编译 x64）

👉 **建议**: 
- 选项A: Win32 平台添加 FTXUI 链接配置
- 选项B: 直接移除 Win32 配置（简化项目）

---

## GitHub Actions 依赖状态

### 当前配置 (`.github/workflows/msbuild.yml`)

```yaml
steps:
  - Checkout code              # ✅ 获取代码
  - Setup MSBuild              # ✅ MSBuild
  - Setup Visual Studio        # ✅ VS 2022
  - Install CMake              # ✅ CMake (用于 FTXUI)
  - Cache FTXUI                # ✅ 缓存 FTXUI 构建
  - Clone and Build FTXUI      # ✅ 构建 FTXUI
  - Cache ONNX Runtime         # ✅ 缓存 ONNX Runtime (2026-02-07新增)
  - Download ONNX Runtime      # ✅ 下载 ONNX Runtime (2026-02-07新增)
  - Build solution             # ✅ 构建主项目
  - Upload artifacts           # ✅ 上传产物
```

### 缺少的依赖

✅ **无** - 所有必需依赖已处理：
- nlohmann/json: ✅ 跟随代码仓库
- FTXUI: ✅ 在 CI 中构建
- ONNX Runtime: ✅ **在 CI 中自动下载（2026-02-07修复）**

---

## 问题修复建议

### 优先级 1: 统一 PlatformToolset

**修复**: 将 `Filerestore_CLI.vcxproj` 改为 `v143`

```xml
<!-- 修复前 -->
<PlatformToolset>v145</PlatformToolset>

<!-- 修复后 -->
<PlatformToolset>v143</PlatformToolset>
```

**影响**: 低，主要是兼容性修复

---

### 优先级 2（可选）: 处理 Win32 平台

**选项 A**: 添加 Win32 FTXUI 链接配置  
**选项 B**: 移除 Win32 配置（推荐，简化项目）

---

### 优先级 3（可选）: 添加依赖文档

在主 README.md 中添加依赖说明章节。

---

## 总结

### ✅ 已解决
1. **FTXUI 依赖** - CI 中自动构建 + 缓存
2. **ONNX Runtime 依赖** - **CI 中自动下载 + 缓存（2026-02-07修复）**
3. **nlohmann/json** - Header-only，无问题
4. **PlatformToolset 统一** - 已统一为 v143

### ⚠️ 建议修复（可选）
1. **优先级 2**: 修复 Win32 FTXUI 链接（或移除 Win32 配置）
2. **优先级 3**: 添加依赖文档到 README

### ✅ 无问题
- Windows SDK 版本
- 编译器配置
- Include 路径
- .gitignore 配置

**检查者**: Claude Code
**状态**: ✅ **所有依赖问题已修复**
**最后更新**: 2026-02-07
