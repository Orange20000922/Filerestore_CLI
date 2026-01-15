# Qt GUI 迁移分析

## 概述

本文档分析将 Filerestore_CLI 从命令行工具迁移到 Qt GUI 应用所需的工作量和架构设计。

---

## 当前代码统计

| 模块 | 文件数 | 预计行数 | 可复用性 | 说明 |
|------|--------|----------|----------|------|
| **Core** | 10 | ~3,000 | ✅ 100% | CLI框架、错误码、API封装 |
| **FileRestore** | 28 | ~8,000 | ✅ 100% | MFT、Carving、ML、检测 |
| **Commands** | 16 | ~4,000 | ❌ 0% | CLI命令实现，需重写为UI |
| **Utils** | 8 | ~1,500 | ⚠️ 50% | Logger可复用，ProgressBar需替换 |
| **总计** | 72 | ~16,500 | ~70% | |

---

## Qt GUI 新增代码估算

### UI 组件

| 组件 | 新代码行数 | 说明 |
|------|------------|------|
| **MainWindow** | ~500 | 主窗口、菜单栏、工具栏、状态栏 |
| **ScanResultsWidget** | ~800 | 文件列表表格、排序、筛选、多选 |
| **RecoveryProgressDialog** | ~300 | 进度条、取消按钮、详情日志 |
| **SettingsDialog** | ~200 | 线程数、ML开关、语言选择 |
| **FilePreviewWidget** | ~400 | 图片预览、十六进制查看器 |
| **DriveSelectionWidget** | ~150 | 磁盘列表、刷新、权限检测 |
| **CarveConfigDialog** | ~250 | 文件类型选择、扫描模式配置 |
| **MLConfidenceChart** | ~200 | 置信度饼图/柱状图可视化 |

### 基础设施

| 组件 | 新代码行数 | 说明 |
|------|------------|------|
| **Qt Model 类** | ~600 | QAbstractTableModel 子类 |
| **Worker 线程类** | ~400 | QThread/QRunnable 封装 |
| **信号槽胶水代码** | ~600 | 连接 UI 和核心逻辑 |
| **资源文件** | ~100 | .qrc、图标、QSS样式表 |

### 总计

| 类别 | 行数 |
|------|------|
| UI 组件 | ~2,800 |
| 基础设施 | ~1,700 |
| **新增总计** | **~4,500** |

---

## 架构设计

### 目录结构

```
Filerestore_GUI/
├── CMakeLists.txt              # CMake 构建配置
├── src/
│   ├── main.cpp                # Qt 应用入口
│   │
│   ├── core/                   # [复用] 核心逻辑
│   │   ├── FileRestore.h/cpp
│   │   ├── FileCarver.h/cpp
│   │   ├── MLClassifier.h/cpp
│   │   ├── MFTReader.h/cpp
│   │   ├── MFTParser.h/cpp
│   │   ├── SignatureScanThreadPool.h/cpp
│   │   ├── OverwriteDetector.h/cpp
│   │   └── ...
│   │
│   ├── ui/                     # [新建] Qt 界面
│   │   ├── MainWindow.h/cpp
│   │   ├── MainWindow.ui       # Qt Designer 文件
│   │   ├── ScanResultsWidget.h/cpp
│   │   ├── RecoveryProgressDialog.h/cpp
│   │   ├── SettingsDialog.h/cpp
│   │   ├── FilePreviewWidget.h/cpp
│   │   ├── DriveSelectionWidget.h/cpp
│   │   └── CarveConfigDialog.h/cpp
│   │
│   ├── models/                 # [新建] Qt 数据模型
│   │   ├── DeletedFileModel.h/cpp
│   │   ├── CarvedFileModel.h/cpp
│   │   └── DriveListModel.h/cpp
│   │
│   ├── workers/                # [新建] 后台任务
│   │   ├── MFTScanWorker.h/cpp
│   │   ├── CarvingWorker.h/cpp
│   │   ├── RecoveryWorker.h/cpp
│   │   └── OverwriteCheckWorker.h/cpp
│   │
│   └── utils/                  # [部分复用] 工具类
│       ├── Logger.h/cpp        # 复用，输出到 QTextEdit
│       └── LocalizationManager.h/cpp  # 复用或迁移到 QTranslator
│
├── resources/
│   ├── resources.qrc           # Qt 资源文件
│   ├── icons/                  # 图标资源
│   │   ├── app.ico
│   │   ├── scan.png
│   │   ├── recover.png
│   │   └── ...
│   ├── styles/                 # QSS 样式表
│   │   ├── light.qss
│   │   └── dark.qss
│   └── translations/           # Qt 翻译文件
│       ├── app_zh_CN.ts
│       └── app_en_US.ts
│
└── deps/                       # 依赖库
    └── onnxruntime/            # ML 推理引擎
```

### 类图

```
┌─────────────────────────────────────────────────────────────┐
│                        MainWindow                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │DriveSelect│ │ScanResults│ │FilePreview│ │ToolBar  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────┬───────────────────────────────────┘
                          │ signals/slots
┌─────────────────────────▼───────────────────────────────────┐
│                      Workers (QThread)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │MFTScanWorker│CarvingWorker│RecoveryWorker│CheckWorker│    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────┬───────────────────────────────────┘
                          │ 调用
┌─────────────────────────▼───────────────────────────────────┐
│                    Core Logic [复用]                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │FileRestore│ │FileCarver │ │MLClassifier│ │MFTReader │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心适配工作

### 1. 进度回调改为 Qt 信号

**当前 CLI 方式**：
```cpp
// SignatureScanThreadPool.cpp
void SignatureScanThreadPool::UpdateProgress() {
    // 直接输出到控制台
    printf("\rProgress: %.1f%%", progress);
}
```

**Qt GUI 方式**：
```cpp
// CarvingWorker.h
class CarvingWorker : public QObject {
    Q_OBJECT
signals:
    void progressUpdated(int percent, QString status);
    void fileFound(CarvedFileInfo info);
    void scanCompleted(bool success, QString message);
};

// CarvingWorker.cpp
void CarvingWorker::onProgress(double progress) {
    emit progressUpdated(static_cast<int>(progress * 100),
                         tr("Scanning: %1%").arg(progress * 100, 0, 'f', 1));
}
```

### 2. 数据模型封装

**DeletedFileModel.h**：
```cpp
class DeletedFileModel : public QAbstractTableModel {
    Q_OBJECT
public:
    enum Column { Name, Path, Size, DeleteTime, RecordNumber, ColumnCount };

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

    void setFiles(const std::vector<DeletedFileInfo>& files);
    DeletedFileInfo getFile(int row) const;

private:
    std::vector<DeletedFileInfo> m_files;
};
```

### 3. 异步任务封装

**MFTScanWorker.h**：
```cpp
class MFTScanWorker : public QThread {
    Q_OBJECT
public:
    MFTScanWorker(char driveLetter, QObject* parent = nullptr);

signals:
    void started();
    void progress(int percent);
    void fileFound(DeletedFileInfo info);
    void finished(bool success, int totalFiles);
    void error(QString message);

protected:
    void run() override;

private:
    char m_driveLetter;
    FileRestore m_restorer;
};
```

---

## 界面设计草图

### 主窗口布局

```
┌────────────────────────────────────────────────────────────┐
│ [File] [Scan] [Recovery] [Tools] [Help]          [Settings]│
├────────────────────────────────────────────────────────────┤
│ [Drive: C: ▼] [Scan MFT] [Carve] [Stop]    [Filter: ____] │
├──────────────────────────────┬─────────────────────────────┤
│                              │                             │
│  File List                   │  Preview                    │
│  ┌────────────────────────┐  │  ┌───────────────────────┐  │
│  │ Name    Size    Date   │  │  │                       │  │
│  │─────────────────────── │  │  │    [Image Preview]    │  │
│  │ doc.docx  125KB  1/5   │  │  │         or            │  │
│  │ img.jpg   2.1MB  1/4   │  │  │    [Hex Viewer]       │  │
│  │ data.xlsx 89KB   1/3   │  │  │                       │  │
│  │ ...                    │  │  └───────────────────────┘  │
│  └────────────────────────┘  │  Confidence: 95% ████████░  │
│                              │  Type: JPEG (ML)            │
├──────────────────────────────┴─────────────────────────────┤
│ [Recover Selected] [Recover All]     Status: Ready    0/0 │
├────────────────────────────────────────────────────────────┤
│ Progress: ████████████░░░░░░░░ 60%   Speed: 2.5 GB/s      │
└────────────────────────────────────────────────────────────┘
```

---

## 依赖项

### Qt 模块

| 模块 | 用途 |
|------|------|
| Qt6::Core | 基础类型、容器、信号槽 |
| Qt6::Widgets | GUI 组件 |
| Qt6::Concurrent | 线程池、异步任务 |
| Qt6::Charts | ML 置信度可视化（可选） |

### 构建配置

**CMakeLists.txt**：
```cmake
cmake_minimum_required(VERSION 3.20)
project(Filerestore_GUI VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets Concurrent)
find_package(onnxruntime REQUIRED)

add_executable(Filerestore_GUI
    src/main.cpp
    src/ui/MainWindow.cpp
    # ... 其他源文件
    resources/resources.qrc
)

target_link_libraries(Filerestore_GUI PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::Concurrent
    onnxruntime::onnxruntime
)
```

---

## 工作量总结

| 任务 | 代码行数 | 复杂度 |
|------|----------|--------|
| 复用核心逻辑 | 0（已有） | - |
| MainWindow + 菜单 | ~500 | 低 |
| 文件列表组件 | ~800 | 中 |
| 数据模型类 | ~600 | 中 |
| Worker 线程类 | ~400 | 中 |
| 进度/恢复对话框 | ~500 | 低 |
| 文件预览组件 | ~400 | 中 |
| 配置/设置界面 | ~450 | 低 |
| 信号槽连接 | ~600 | 低 |
| 资源/样式 | ~250 | 低 |
| **新增总计** | **~4,500** | |

### 可复用代码

| 模块 | 行数 |
|------|------|
| FileRestore 引擎 | ~2,500 |
| FileCarver + ThreadPool | ~3,000 |
| MLClassifier | ~800 |
| MFT 操作 | ~2,500 |
| 覆盖检测 | ~1,200 |
| 工具类 | ~500 |
| **复用总计** | **~10,500** |

---

## 迁移路线图

### Phase 1: 基础框架
- [ ] 创建 Qt 项目结构
- [ ] 复制核心逻辑代码
- [ ] 实现 MainWindow 基础布局
- [ ] 实现磁盘选择功能

### Phase 2: MFT 扫描
- [ ] 实现 MFTScanWorker
- [ ] 实现 DeletedFileModel
- [ ] 实现文件列表显示
- [ ] 实现筛选/排序功能

### Phase 3: File Carving
- [ ] 实现 CarvingWorker
- [ ] 实现 CarvedFileModel
- [ ] 实现扫描配置对话框
- [ ] 实现进度显示

### Phase 4: 恢复功能
- [ ] 实现 RecoveryWorker
- [ ] 实现批量恢复
- [ ] 实现恢复进度对话框
- [ ] 实现覆盖检测集成

### Phase 5: 增强功能
- [ ] 实现文件预览
- [ ] 实现 ML 置信度可视化
- [ ] 实现设置对话框
- [ ] 实现多语言支持

### Phase 6: 打磨
- [ ] UI 美化（QSS 样式）
- [ ] 图标资源
- [ ] 打包发布
- [ ] 文档更新

---

## 风险与注意事项

1. **管理员权限**：Qt 应用需要以管理员身份运行才能访问原始磁盘
2. **线程安全**：确保 UI 更新在主线程，耗时操作在 Worker 线程
3. **内存管理**：Qt 对象树自动管理，但核心逻辑使用 unique_ptr
4. **ONNX Runtime**：需要正确配置 DLL 部署路径
5. **编码问题**：确保 UTF-8/UTF-16 转换正确（文件名显示）
