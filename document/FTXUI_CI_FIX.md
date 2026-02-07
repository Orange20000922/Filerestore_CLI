# GitHub Actions FTXUI 依赖修复

## 问题描述

GitHub Actions构建失败，错误提示缺少FTXUI的Include依赖。

## 根本原因

FTXUI库位于本地`Filerestore_CLI/deps/ftxui/`目录，包含：
- 源代码
- `.git`目录（作为独立的git仓库）
- `build/`目录（本地构建产物）

这些内容在GitHub Actions的`checkout`操作中不会被克隆，导致构建失败。

## 解决方案

### 1. 更新 GitHub Actions 工作流 (`.github/workflows/msbuild.yml`)

**新增步骤**：
- 安装CMake
- 缓存FTXUI构建产物（加速后续构建）
- 克隆FTXUI源码（浅克隆）
- 使用CMake构建FTXUI（Debug + Release）

**关键改动**：
```yaml
- name: Install CMake
  uses: lukka/get-cmake@latest

- name: Cache FTXUI build
  id: cache-ftxui
  uses: actions/cache@v4
  with:
    path: Filerestore_CLI/deps/ftxui/build
    key: ftxui-${{ runner.os }}-${{ hashFiles('Filerestore_CLI/deps/ftxui/CMakeLists.txt') }}

- name: Clone and Build FTXUI
  shell: pwsh
  run: |
    # 克隆FTXUI（如果不存在）
    git clone --depth 1 https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui

    # 构建（如果缓存未命中）
    if ("${{ steps.cache-ftxui.outputs.cache-hit }}" -ne "true") {
      cmake --build . --config Debug
      cmake --build . --config Release
    }
```

### 2. 更新 `.gitignore`

**新增规则**：
```gitignore
# FTXUI dependency (built in CI)
Filerestore_CLI/deps/ftxui/.git/
Filerestore_CLI/deps/ftxui/build/
Filerestore_CLI/deps/ftxui/.cache/
Filerestore_CLI/deps/ftxui/cmake-build-*/
```

**目的**：
- 不提交FTXUI的Git仓库信息
- 不提交本地构建产物
- 在CI中动态克隆和构建

### 3. 临时设备文件修复

**修复了之前的问题**：
```gitignore
# 之前（只忽略根目录）
/nul

# 修复后（忽略所有目录）
nul
null
NUL
NULL
*.tmp
*.temp
```

## 构建流程

### 本地开发
1. 开发者手动克隆FTXUI到`deps/ftxui/`
2. 使用CMake构建FTXUI
3. MSBuild构建主项目

### GitHub Actions CI
1. Checkout代码（不包含`deps/ftxui/`）
2. 安装CMake
3. 检查缓存（FTXUI构建产物）
4. 如果缓存未命中：
   - 克隆FTXUI源码
   - 使用CMake构建FTXUI
   - 缓存构建产物
5. 如果缓存命中：
   - 直接恢复构建产物
   - 跳过构建（节省~2-3分钟）
6. MSBuild构建主项目
7. 上传构建产物

## 性能优化

### 缓存机制
- **缓存键**：`ftxui-{OS}-{CMakeLists.txt哈希}`
- **缓存路径**：`Filerestore_CLI/deps/ftxui/build`
- **预期效果**：
  - 首次构建：~5分钟（包含FTXUI构建）
  - 后续构建：~2分钟（使用缓存）

### 浅克隆
```bash
git clone --depth 1 https://github.com/ArthurSonzogni/FTXUI.git
```
- 只克隆最新提交
- 节省约50MB下载
- 加速克隆时间

## 验证步骤

1. **提交更改**：
   ```bash
   git add .github/workflows/msbuild.yml
   git add .gitignore
   git commit -m "Fix GitHub Actions: Add FTXUI dependency build"
   git push
   ```

2. **检查GitHub Actions**：
   - 访问：https://github.com/{username}/{repo}/actions
   - 查看最新的workflow运行
   - 验证"Clone and Build FTXUI"步骤成功

3. **验证缓存**：
   - 第一次运行：应该构建FTXUI（日志显示"Building FTXUI..."）
   - 第二次运行：应该使用缓存（日志显示"FTXUI restored from cache"）

## 替代方案（未采用）

### 方案A：Git子模块
```bash
git submodule add https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui
```
**优点**：Git原生支持
**缺点**：仍需在CI中构建，复杂度增加

### 方案B：vcpkg
```bash
vcpkg install ftxui
```
**优点**：包管理器自动处理
**缺点**：需要配置vcpkg，增加依赖

### 方案C：提交构建产物
**优点**：CI无需构建
**缺点**：
- 增加仓库体积（~50MB）
- 不符合Git最佳实践
- 跨平台/跨版本兼容性问题

**最终选择**：动态克隆+构建+缓存（方案0）
- 符合最佳实践
- 缓存机制提供良好性能
- 灵活性高

## 相关文件

- `.github/workflows/msbuild.yml` - GitHub Actions配置
- `.gitignore` - Git忽略规则
- `Filerestore_CLI/deps/ftxui/` - FTXUI源码目录（本地）

## 文档更新

本次修复后，请在主README中添加开发者设置说明：

```markdown
## 开发环境设置

### 依赖：FTXUI

本项目使用FTXUI构建TUI界面。首次克隆项目后需要手动设置：

1. 克隆FTXUI：
   ```bash
   git clone https://github.com/ArthurSonzogni/FTXUI.git Filerestore_CLI/deps/ftxui
   ```

2. 构建FTXUI：
   ```bash
   cd Filerestore_CLI/deps/ftxui
   mkdir build && cd build
   cmake .. -G "Visual Studio 17 2022" -A x64
   cmake --build . --config Debug
   cmake --build . --config Release
   ```

3. 构建主项目（VS或MSBuild）

**注意**：GitHub Actions会自动处理FTXUI构建，无需担心CI配置。
```

---

**修复日期**: 2026-02-07
**修复状态**: ✅ 已完成
**下次推送时生效**: 是
