# TUI 设计方案：三层架构（向导 + 仪表盘 + CLI）

## 1. 总体定位

**三层交互架构，最大化降低使用门槛，同时保留高级用户的灵活性。**

```
┌────────────────────────────────────┐
│  Layer 1: 向导/菜单 (新用户友好)    │  ← 引导式交互，零学习成本
├────────────────────────────────────┤
│  Layer 2: 仪表盘 (可视化)          │  ← 状态、进度、结果的实时展示
├────────────────────────────────────┤
│  Layer 3: 命令输入 (高级用户)       │  ← 直接输入CLI命令，完全控制
├────────────────────────────────────┤
│  CLI::Run() (命令引擎)             │  ← 现有代码，不动
└────────────────────────────────────┘
```

**核心原则：**
- 向导和命令输入最终都调用 `CLI::Run()`，零重构现有命令系统
- 新用户通过菜单/向导操作，不需要记任何命令
- 高级用户随时可以切到命令输入框直接打命令
- 向导只是UI组件，底层都是已有命令

**提供的能力：**
- 引导式向导（选驱动器 → 选操作 → 选参数 → 执行）
- 实时状态仪表盘
- 可视化扫描进度
- 可滚动/可筛选的结果表格
- 结构化日志查看
- 命令行输入（回退到传统CLI体验）

## 2. 技术选型：FTXUI

**库**: [FTXUI](https://github.com/ArthurSonzogni/FTXUI) (Functional Terminal User Interface)

**选择理由：**
- 纯 C++（无外部依赖），与项目技术栈完全匹配
- Header-only 可选，集成简单
- 支持 C++17/20
- 支持 Windows Terminal（你的目标平台）
- 组件化设计（Input, Menu, Table, Gauge 等）
- 支持异步刷新（扫描进度实时更新）
- MIT 许可证，无商业风险

**集成方式：**
- 通过 vcpkg 安装（推荐，与 Visual Studio 集成最好）
- 或 git submodule + CMake/直接编译源码

## 3. 界面布局

```
╔══════════════════════════════════════════════════════════════════╗
║  Filerestore CLI v0.3.2                 [D:]  [Cache: 45 MB]   ║
╠════════════╦═════════════════════════════════════════════════════╣
║  STATUS    ║  MAIN VIEW                                        ║
║            ║                                                    ║
║  Drive: D  ║  (根据当前操作动态切换内容)                           ║
║  MFT:  ✓   ║                                                    ║
║  USN:  ✓   ║  · 空闲时: 欢迎页 + 快速操作菜单                     ║
║  Cache: ✓  ║  · 扫描时: 实时进度条 + 统计数据                     ║
║            ║  · 结果时: 可滚动表格                                ║
║ ────────── ║  · 恢复时: 文件详情 + 验证信息                       ║
║  SCAN      ║                                                    ║
║  Speed: -  ║                                                    ║
║  Found: 0  ║                                                    ║
║  Time:  -  ║                                                    ║
║            ║                                                    ║
║ ────────── ║                                                    ║
║  QUICK     ║                                                    ║
║  [F1] Help ║                                                    ║
║  [F2] Scan ║                                                    ║
║  [F5] List ║                                                    ║
║  [Esc]Quit ║                                                    ║
╠════════════╩═════════════════════════════════════════════════════╣
║  Command> _                                                     ║
╠═════════════════════════════════════════════════════════════════╣
║  [LOG] MFT cache loaded: 1,234,567 records                     ║
║  [LOG] Ready.                                                   ║
╚═════════════════════════════════════════════════════════════════╝
```

### 区域说明

| 区域 | 功能 | FTXUI 组件 |
|------|------|-----------|
| **标题栏** | 版本号、当前驱动器、缓存状态 | `hbox + text + separator` |
| **状态面板** (左侧) | 驱动器/MFT/USN/Cache 状态指示 | `vbox + text + color` |
| **扫描面板** (左侧中) | 实时速度、发现文件数、耗时 | `vbox + text` (异步更新) |
| **快捷键面板** (左侧下) | 常用功能键提示 | `vbox + text` |
| **主视图** (右侧) | 动态内容区，根据操作切换 | `Container::Tab` |
| **命令输入** | CLI命令输入框 | `Input` 组件 |
| **日志面板** | 最近日志消息（可滚动） | `vbox + frame` |

## 4. 向导系统（Layer 1 - 核心降门槛设计）

### 4.1 设计原则

向导系统是 TUI 的核心差异化功能。目标用户画像：
- **不知道有哪些命令**的新用户
- **不想记参数格式**的普通用户
- **想快速完成常见操作**的所有用户

**向导的本质：通过菜单选择 → 构造CLI命令字符串 → 调用 CLI::Run()**

### 4.2 主菜单（欢迎页即主菜单）

程序启动后直接显示功能菜单，用户用方向键选择：

```
╭──────────────────────────────────────────╮
│    NTFS File Recovery Tool v0.3.2        │
│                                          │
│    What do you want to do?               │
│                                          │
│    > [1] Smart Recovery (推荐)           │  ← recover 命令
│      [2] Scan for Deleted Files          │  ← listdeleted 命令
│      [3] Deep Scan (Signature Carving)   │  ← carvepool 命令
│      [4] Repair Corrupted Files          │  ← repair 命令
│      [5] Browse Previous Results         │  ← carvelist 命令
│      [6] Advanced (Command Line)         │  ← 切到命令输入模式
│                                          │
│    [↑↓] Select  [Enter] Confirm          │
╰──────────────────────────────────────────╯
```

### 4.3 向导流程示例

#### A. Smart Recovery 向导 (选项1)

```
Step 1/3: Select Drive
┌────────────────────────────────────┐
│  Available NTFS Drives:            │
│                                    │
│  > [C:]  Windows (120 GB)          │
│    [D:]  Data (500 GB)             │
│    [E:]  Backup (1 TB)             │
│                                    │
│  [↑↓] Select  [Enter] Next        │
└────────────────────────────────────┘

Step 2/3: Enter Filename (optional)
┌────────────────────────────────────┐
│  File to recover (leave empty for  │
│  interactive mode):                │
│                                    │
│  > [report.docx              ]     │
│                                    │
│  [Enter] Next  [Esc] Back          │
└────────────────────────────────────┘

Step 3/3: Output Directory
┌────────────────────────────────────┐
│  Save recovered files to:          │
│                                    │
│  > [D:\Recovery\             ]     │
│                                    │
│  [Enter] Start  [Esc] Back         │
└────────────────────────────────────┘
```

→ 自动生成命令: `recover D report.docx D:\Recovery`
→ 调用 `CLI::Run("recover D report.docx D:\\Recovery")`

#### B. Deep Scan 向导 (选项3)

```
Step 1/4: Select Drive
  > [D:]  Data (500 GB)

Step 2/4: File Types
┌────────────────────────────────────┐
│  Select file types to scan:        │
│                                    │
│  ☑ Images (jpg, png, bmp, gif)     │
│  ☑ Documents (pdf, docx, xlsx)     │
│  ☐ Archives (zip, rar, 7z)        │
│  ☐ Media (mp3, mp4, avi)          │
│  ☐ All types                       │
│                                    │
│  [Space] Toggle  [Enter] Next      │
└────────────────────────────────────┘

Step 3/4: Thread Count
┌────────────────────────────────────┐
│  Parallel threads:                 │
│                                    │
│    [4]  [8]  > [12]  [16]          │
│                                    │
│  Recommended: 8 (your CPU: 8C/16T) │
│  [Enter] Next  [Esc] Back          │
└────────────────────────────────────┘

Step 4/4: Output Directory
  > [D:\Recovery\]

  Ready to scan:
  Drive: D:  Types: jpg,png,bmp,gif,pdf,docx,xlsx
  Threads: 12  Output: D:\Recovery\

  [Enter] Start Scan  [Esc] Back
```

→ 自动生成: `carvepool D jpg,png,bmp,gif,pdf,docx,xlsx D:\Recovery 12`

#### C. Scan Deleted Files 向导 (选项2)

```
Step 1/2: Select Drive
  > [D:]

Step 2/2: Options
┌────────────────────────────────────┐
│  ☑ Use MFT Cache (faster)          │
│  ☐ Show active files too           │
│  ☐ Filter by extension             │
│                                    │
│  [Space] Toggle  [Enter] Start     │
└────────────────────────────────────┘
```

→ 自动生成: `listdeleted D cache`

### 4.4 向导的代码架构

```cpp
// 向导基类
class WizardBase {
protected:
    int currentStep = 0;
    int totalSteps = 0;
    vector<string> collectedParams;  // 收集的参数

    // 每个向导子类实现：
    virtual Component RenderStep(int step) = 0;
    virtual string BuildCommand() = 0;  // 构造CLI命令

public:
    // 完成后自动调用 CLI::Run(BuildCommand())
    void Execute() {
        string cmd = BuildCommand();
        CLI cli;
        cli.Run(cmd);
    }
};

// 具体向导
class SmartRecoveryWizard : public WizardBase {
    string BuildCommand() override {
        // 从收集的参数构造命令
        return "recover " + drive + " " + filename + " " + outputDir;
    }
};

class DeepScanWizard : public WizardBase {
    string BuildCommand() override {
        return "carvepool " + drive + " " + types + " " + outputDir + " " + threads;
    }
};
```

### 4.5 向导与命令输入的无缝切换

用户在任何向导步骤中按 **`/`** 或选择 "Advanced" 可以切换到命令输入模式：

```
┌────────────────────────────────────────────┐
│  Switched to command mode.                 │
│  Type any command directly:                │
│                                            │
│  Command> _                                │
│                                            │
│  [Esc] Back to wizard                      │
└────────────────────────────────────────────┘
```

## 5. 主视图的多种模式（Layer 2 - 仪表盘）

### 5.1 欢迎页 = 主菜单 (Welcome/Menu View)

空闲时的默认页面（即上面的主菜单）：

```
  ╭──────────────────────────────────────╮
  │    NTFS File Recovery Tool           │
  │    Version 0.3.2                     │
  │                                      │
  │    Quick Start:                      │
  │    1. recover D               智能恢复│
  │    2. carvepool D all out     全盘扫描│
  │    3. listdeleted D           查看删除│
  │    4. help                    查看帮助│
  ╰──────────────────────────────────────╯
```

### 4.2 扫描进度视图 (Scan View)

执行 `carvepool` 或 `recover` 时自动切换：

```
  Scanning Drive D: (Signature Scan - Thread Pool)

  Progress: ████████████░░░░░░░░  58.3%

  Speed:     2.71 GB/s
  Scanned:   186.4 GB / 320.0 GB
  Found:     1,247 files
  Elapsed:   1m 08s
  ETA:       0m 49s

  ┌─────────────────────────────────┐
  │ Type      │ Count │ Total Size │
  ├───────────┼───────┼────────────┤
  │ .jpg      │   523 │   2.1 GB   │
  │ .pdf      │   312 │   1.8 GB   │
  │ .docx     │   198 │   456 MB   │
  │ .png      │   127 │   890 MB   │
  │ .zip      │    87 │   3.2 GB   │
  └─────────────────────────────────┘
```

### 4.3 结果表格视图 (Results View)

扫描完成或执行 `carvelist` 时：

```
  Carved Files: 1,247 found (8.4 GB total)     Page 1/25

  ┌────┬─────────┬──────────┬────────────┬─────────┬───────────┐
  │ #  │ Type    │ Size     │ Offset     │ Valid   │ Score     │
  ├────┼─────────┼──────────┼────────────┼─────────┼───────────┤
  │  1 │ .jpg    │ 2.3 MB   │ 0x1A30000  │ ✓ Footer│ 0.92      │
  │  2 │ .pdf    │ 156 KB   │ 0x1B40000  │ ✓ Footer│ 0.88      │
  │  3 │ .docx   │ 45 KB    │ 0x1C00000  │ ~ Est.  │ 0.71      │
  │> 4 │ .png    │ 890 KB   │ 0x1D20000  │ ✓ Footer│ 0.95      │ ← 当前选中
  │  5 │ .zip    │ 12.3 MB  │ 0x1E00000  │ ✓ Footer│ 0.85      │
  └────┴─────────┴──────────┴────────────┴─────────┴───────────┘

  [↑↓] Navigate  [Enter] Details  [R] Recover  [F] Filter  [/] Search
```

### 4.4 文件详情视图 (Detail View)

选中文件后按 Enter：

```
  File Detail: #4 - PNG Image

  ┌ Location ──────────────────────────────┐
  │ Start LCN:    0x1D200                  │
  │ Start Offset: 0x1D20000               │
  │ File Size:    890 KB                   │
  │ Has Footer:   Yes (IEND)              │
  ├ Validation ────────────────────────────┤
  │ Confidence:   0.95 (High)             │
  │ Integrity:    92%                      │
  │ MFT Match:    Record #45678           │
  │ USN Match:    screenshot_2026.png      │
  │ Triple Val:   ✓ TRIPLE (USN+MFT+Sig)  │
  ├ Timestamps ────────────────────────────┤
  │ Created:      2026-01-15 14:32:10     │
  │ Modified:     2026-01-15 14:32:10     │
  │ Source:       MFT                      │
  ├ ML Classification ────────────────────┤
  │ Class:        screenshot              │
  │ ML Score:     0.87                     │
  └────────────────────────────────────────┘

  [R] Recover this file  [Esc] Back to list
```

### 4.5 Recover 智能恢复视图

执行 `recover D filename` 时：

```
  Smart Recovery: "report.docx"

  Step 1/4: USN Journal Search          ✓ Found 3 matches
  Step 2/4: MFT Enhancement             ✓ Record #12345
  Step 3/4: Signature Scan              ████████████████████ 100%
  Step 4/4: Triple Validation           ✓ Confidence: 0.94

  ┌ Best Match ────────────────────────────┐
  │ File:       report.docx                │
  │ Deleted:    2026-02-05 09:15:23       │
  │ Size:       2.3 MB                     │
  │ Validation: TRIPLE (USN+MFT+Sig)      │
  │ Confidence: 94%                        │
  │ Overwrite:  None detected              │
  └────────────────────────────────────────┘

  Recover to: D:\Recovery\report.docx? [Y/n]
```

## 6. 架构设计

### 6.1 模块结构

```
src/
├── tui/                          # 新增 TUI 模块
│   ├── TuiApp.h / .cpp           # TUI 应用主类（入口）
│   ├── TuiLayout.h / .cpp        # 整体布局管理
│   ├── components/               # UI 组件
│   │   ├── StatusPanel.h/.cpp    # 左侧状态面板
│   │   ├── MainView.h/.cpp       # 右侧主视图（Tab容器）
│   │   ├── CommandInput.h/.cpp   # 底部命令输入
│   │   ├── LogPanel.h/.cpp       # 日志面板
│   │   ├── ScanView.h/.cpp       # 扫描进度视图
│   │   ├── ResultsTable.h/.cpp   # 结果表格视图
│   │   ├── DetailView.h/.cpp     # 文件详情视图
│   │   └── WelcomeMenu.h/.cpp    # 欢迎页 = 主菜单
│   ├── wizards/                  # 向导系统
│   │   ├── WizardBase.h/.cpp     # 向导基类（分步收集参数 → 构造命令）
│   │   ├── SmartRecoveryWizard.h/.cpp   # recover 向导
│   │   ├── DeepScanWizard.h/.cpp        # carvepool 向导
│   │   ├── DeletedFilesWizard.h/.cpp    # listdeleted 向导
│   │   └── RepairWizard.h/.cpp          # repair 向导
│   └── TuiProgressBridge.h/.cpp  # ProgressBar → TUI 的桥接
```

### 6.2 与现有代码的集成策略

**核心原则：最小侵入，最大复用。**

```
                    ┌─────────────────┐
                    │    Main.cpp     │
                    │  (启动入口)      │
                    └────────┬────────┘
                             │
                    选择模式（--tui 或 --cli）
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌─────────▼────────┐
     │  传统 CLI 模式   │          │   TUI 模式        │
     │  (现有代码不变)   │          │  (TuiApp 驱动)    │
     │  while(getline)  │          │  FTXUI Screen    │
     └─────────────────┘          └────────┬─────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  CommandInput    │
                                  │  用户输入命令     │
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │    CLI::Run()    │  ← 复用现有命令解析
                                  │  命令解析和执行   │
                                  └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │  OutputCapture   │  ← 新增：捕获 cout 输出
                                  │  重定向到 TUI    │
                                  └─────────────────┘
```

### 6.3 关键集成点

#### A. Main.cpp 修改（最小化）

```cpp
int main(int argc, char* argv[]) {
    // ... 现有初始化代码 ...

    bool useTui = false;
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--tui") useTui = true;
    }

    if (useTui) {
        TuiApp app;
        app.Run();  // TUI 模式
    } else {
        // 现有 CLI 模式（完全不变）
        while (!CLI::ShouldExit()) {
            // ...
        }
    }
}
```

#### B. 输出捕获（OutputCapture）

**问题：** 现有命令通过 `cout` 输出结果，TUI 需要捕获这些输出并重定向到主视图。

**方案：** 使用 `std::streambuf` 重定向：

```cpp
class TuiOutputCapture : public std::streambuf {
    // 捕获所有 cout 输出
    // 转发到 TUI 的 MainView 或 LogPanel
    // 解析特殊标记（进度更新、表格数据等）
};
```

#### C. 进度桥接（TuiProgressBridge）

**问题：** 现有 `ProgressBar` 直接写控制台，TUI 中需要更新 FTXUI 的 Gauge 组件。

**方案：** 提供替代接口：

```cpp
class TuiProgressBridge {
    // 注册回调，当进度更新时通知 TUI
    static function<void(double progress, string status)> onProgress;

    // 替代 ProgressBar::Update()
    static void UpdateProgress(double percent, const string& info);
};
```

### 6.4 TuiApp 主循环

```cpp
void TuiApp::Run() {
    auto screen = ScreenInteractive::Fullscreen();

    // 构建布局
    auto layout = TuiLayout::Create();

    // 命令输入处理
    auto commandInput = CommandInput::Create([&](const string& cmd) {
        // 在独立线程执行命令（避免阻塞UI）
        thread([cmd]() {
            CLI cli;
            cli.Run(cmd);
        }).detach();
    });

    // 异步刷新（扫描进度等）
    auto refreshLoop = std::async(std::launch::async, [&]() {
        while (running) {
            screen.PostEvent(Event::Custom);
            this_thread::sleep_for(100ms);
        }
    });

    screen.Loop(layout);
}
```

## 7. 实施计划（分阶段）

### Phase 1：基础框架 + 主菜单（最小可用）
**目标：** TUI 框架跑起来，主菜单可交互，能输入命令、显示输出

1. 集成 FTXUI 到项目（vcpkg 或源码编译）
2. 实现 `TuiApp` 主框架
3. 实现 `WelcomeMenu`（主菜单，方向键选择操作）
4. 实现 `CommandInput`（底部命令输入）
5. 实现 `OutputCapture`（cout 重定向到主视图）
6. 实现基本布局（标题栏 + 主视图 + 命令输入 + 日志）
7. `--tui` 启动参数

**交付物：** 一个能跑的 TUI，主菜单可选择操作，命令输入框可用

### Phase 2：向导系统（核心降门槛）
**目标：** 主要操作可通过向导完成，零命令记忆

1. 实现 `WizardBase`（向导基类：分步收集参数 → BuildCommand → CLI::Run）
2. 实现 `SmartRecoveryWizard`（recover 向导：选驱动器 → 输文件名 → 选输出）
3. 实现 `DeepScanWizard`（carvepool 向导：选驱动器 → 选类型 → 选线程 → 选输出）
4. 实现 `DeletedFilesWizard`（listdeleted 向导）
5. 主菜单选项绑定到对应向导
6. 向导中可按 `/` 切换到命令输入模式

### Phase 3：状态面板 + 进度可视化
**目标：** 左侧状态面板、扫描进度条

1. 实现 `StatusPanel`（驱动器/缓存状态）
2. 实现 `ScanView`（进度条 + 统计）
3. 实现 `TuiProgressBridge`（替代 ProgressBar）
4. 扫描时自动切换到 ScanView

### Phase 4：结果表格 + 交互
**目标：** 可滚动、可筛选的结果表格

1. 实现 `ResultsTable`（可滚动表格）
2. 键盘导航（↑↓翻页、Enter 查看详情）
3. 实现 `DetailView`（文件详情）
4. 筛选和搜索功能
5. 结果表格中直接按 R 恢复文件（调用 `carverecover` 命令）

### Phase 5：智能恢复可视化
**目标：** recover 命令的分步可视化

1. 实现 Recover 专用视图
2. 4 步流程可视化（USN → MFT → Scan → Validate）
3. 三角验证结果展示

### Phase 6：快捷键 + 主题 + 打磨
**目标：** 快捷键操作、配色方案、用户体验打磨

1. F1-F5 快捷键绑定（任何页面可用）
2. 配色主题（暗色/亮色）
3. 自适应终端大小
4. 向导中的自动补全（驱动器列表自动检测、历史输出目录记忆）
5. 错误提示友好化（命令失败时在TUI中显示可读错误信息）

## 8. 风险和注意事项

### 8.1 cout 重定向的复杂性
现有命令大量使用 cout/printf/wcout 混合输出。需要：
- 同时重定向 cout 和 wcout
- printf 需要重定向 stdout（更底层）
- 某些命令可能直接调用 Windows Console API

**缓解：** Phase 1 先做 cout 重定向，printf/wcout 在后续阶段处理。

### 8.2 多线程安全
扫描在独立线程运行，UI 在主线程。FTXUI 的 `screen.PostEvent()` 是线程安全的，但需要注意：
- 共享数据用 mutex 保护
- 进度更新频率控制（不要每个扇区都刷新UI）

### 8.3 Windows Terminal 兼容性
FTXUI 在 Windows Terminal 和新版 CMD 中效果好，但旧版 CMD (conhost) 可能有渲染问题。
- 建议在 README 中说明推荐使用 Windows Terminal
- 保留 `--cli` 作为回退

### 8.4 向后兼容
- 默认仍然启动 CLI 模式（不破坏现有用户习惯）
- `--tui` 显式启用 TUI
- 所有命令在两种模式下行为一致

## 9. 用户体验对比

### 9.1 新用户（不了解命令）

**Before (CLI):**
```
Command> ???
Command not found.
Command> help
... 30+ 命令列表 ...
Command> ??? 还是不知道该用哪个
```

**After (TUI):**
```
> [1] Smart Recovery (推荐)     ← 方向键选择，Enter确认
  [2] Scan for Deleted Files
  [3] Deep Scan
  ...

Step 1/3: Select Drive
  > [D:]                       ← 方向键选择
Step 2/3: Enter Filename
  > [report.docx]              ← 直接输入
Step 3/3: Output Directory
  > [D:\Recovery\]

Recovering... ████████ 100%     ← 自动执行，可视化进度
Done! File saved to D:\Recovery\report.docx
```

### 9.2 高级用户

TUI 中随时按 `/` 或选择 "Advanced" 切换到命令输入：
```
Command> carvepool D jpg,png D:\out 16 --validate
```

功能完全不变，只是多了一个可视化壳。

