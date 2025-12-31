# USN Journal 和文件诊断功能实现方案

## 📋 概述

本文档说明如何添加两个新功能来解决"最近删除文件扫描不到"的问题：

1. **scanusn** - 使用 USN Change Journal 追踪最近删除的文件
2. **diagnosefile** - 诊断特定文件是否存在于 MFT 中

---

## 🔧 已创建的文件

### 1. UsnJournalReader.h / UsnJournalReader.cpp

**功能**：
- 读取 Windows USN Change Journal
- 追踪文件系统的所有变化（包括删除操作）
- 可以检测 MFT 扫描方法遗漏的文件

**主要方法**：
```cpp
// 打开指定驱动器的 USN Journal
bool Open(char driveLetter);

// 扫描最近删除的文件
vector<UsnDeletedFileInfo> ScanRecentlyDeletedFiles(
    int maxTimeSeconds = 3600,   // 最大回溯时间（秒）
    size_t maxRecords = 10000    // 最大返回记录数
);

// 按文件名搜索删除记录
vector<UsnDeletedFileInfo> SearchDeletedByName(
    const wstring& fileName,
    bool exactMatch = false
);
```

---

## 📝 需要添加的命令实现

### 命令 1: scanusn

**命令格式**：
```
scanusn <drive> [max_hours]
```

**示例**：
```bash
scanusn C           # 扫描C盘最近1小时删除的文件
scanusn C 24        # 扫描C盘最近24小时删除的文件
```

**实现代码**（添加到 cmd.cpp）：

```cpp
// ==================== ScanUsnCommand ====================

ScanUsnCommand::ScanUsnCommand() {
    FlagHasArgs = TRUE;
}

ScanUsnCommand::~ScanUsnCommand() {
}

void ScanUsnCommand::AcceptArgs(vector<LPVOID> argslist) {
    ScanUsnCommand::ArgsList = argslist;
}

void ScanUsnCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (ArgsList.size() < 1 || ArgsList.size() > 2) {
        cout << "Invalid Args! Usage: scanusn <drive_letter> [max_hours]" << endl;
        cout << "Examples:" << endl;
        cout << "  scanusn C         - Scan C: for files deleted in the last hour" << endl;
        cout << "  scanusn C 24      - Scan C: for files deleted in the last 24 hours" << endl;
        cout << "  scanusn C 168     - Scan C: for files deleted in the last week" << endl;
        return;
    }

    try {
        string& driveStr = *(string*)ArgsList[0];
        int maxHours = 1;  // Default: 1 hour

        if (ArgsList.size() >= 2) {
            string& hoursStr = *(string*)ArgsList[1];
            try {
                maxHours = stoi(hoursStr);
                if (maxHours <= 0) {
                    cout << "Invalid hours value. Using default (1 hour)." << endl;
                    maxHours = 1;
                }
            } catch (...) {
                cout << "Invalid hours value. Using default (1 hour)." << endl;
            }
        }

        if (driveStr.empty()) {
            cout << "Invalid drive letter." << endl;
            return;
        }

        char driveLetter = driveStr[0];

        cout << "\n========== USN Journal Scanner ==========\n" << endl;
        cout << "Drive: " << driveLetter << ":" << endl;
        cout << "Time range: Last " << maxHours << " hour(s)" << endl;
        cout << endl;

        // 创建 USN Journal 读取器
        UsnJournalReader usnReader;

        if (!usnReader.Open(driveLetter)) {
            cout << "ERROR: " << usnReader.GetLastError() << endl;
            cout << "\nNote: USN Journal requires:" << endl;
            cout << "  1. Administrator privileges" << endl;
            cout << "  2. USN Journal enabled on the volume" << endl;
            return;
        }

        // 获取并显示 USN Journal 统计信息
        UsnJournalStats stats;
        if (usnReader.GetJournalStats(stats)) {
            cout << "USN Journal Information:" << endl;
            cout << "  Journal ID: " << stats.UsnJournalID << endl;
            cout << "  Maximum Size: " << (stats.MaximumSize / 1024 / 1024) << " MB" << endl;
            cout << "  First USN: " << stats.FirstUsn << endl;
            cout << "  Next USN: " << stats.NextUsn << endl;
            cout << endl;
        }

        // 扫描删除的文件
        int maxTimeSeconds = maxHours * 3600;
        vector<UsnDeletedFileInfo> deletedFiles = usnReader.ScanRecentlyDeletedFiles(
            maxTimeSeconds, 10000);

        if (deletedFiles.empty()) {
            cout << "\nNo deleted files found in the specified time range." << endl;
            return;
        }

        cout << "\n===== Recently Deleted Files (from USN Journal) =====\n" << endl;
        cout << "Found: " << deletedFiles.size() << " deleted files" << endl;
        cout << "\nFormat: [MFT#] Filename | Parent MFT# | Time" << endl;
        cout << "----------------------------------------------" << endl;

        // 显示结果
        size_t displayLimit = min(deletedFiles.size(), (size_t)100);
        for (size_t i = 0; i < displayLimit; i++) {
            const auto& info = deletedFiles[i];

            // 转换时间戳
            SYSTEMTIME st;
            FILETIME ft;
            ft.dwLowDateTime = info.TimeStamp.LowPart;
            ft.dwHighDateTime = info.TimeStamp.HighPart;
            FileTimeToSystemTime(&ft, &st);

            cout << "[" << info.FileReferenceNumber << "] ";
            wcout << info.FileName << " | ";
            cout << "Parent: " << info.ParentFileReferenceNumber << " | ";
            printf("%04d-%02d-%02d %02d:%02d:%02d\n",
                   st.wYear, st.wMonth, st.wDay,
                   st.wHour, st.wMinute, st.wSecond);
        }

        cout << "\n----------------------------------------------" << endl;
        if (deletedFiles.size() > 100) {
            cout << "Note: Showing first 100 of " << deletedFiles.size() << " files." << endl;
        }

        cout << "\nTip: Use 'diagnosefile <drive> <filename>' to check if a file exists in MFT" << endl;

    } catch (const exception& e) {
        cout << "[ERROR] Exception: " << e.what() << endl;
    } catch (...) {
        cout << "[ERROR] Unknown exception in ScanUsnCommand::Execute" << endl;
    }
}

BOOL ScanUsnCommand::HasArgs() {
    return FlagHasArgs;
}

BOOL ScanUsnCommand::CheckName(string input) {
    if (input.compare(name) == 0) {
        return TRUE;
    }
    return FALSE;
}
```

---

### 命令 2: diagnosefile

**命令格式**：
```
diagnosefile <drive> <filename>
```

**示例**：
```bash
diagnosefile C test_recovery_file.txt
diagnosefile C test                    # 搜索包含"test"的所有文件
```

**实现代码**（添加到 cmd.cpp）：

```cpp
// ==================== DiagnoseFileCommand ====================

DiagnoseFileCommand::DiagnoseFileCommand() {
    FlagHasArgs = TRUE;
}

DiagnoseFileCommand::~DiagnoseFileCommand() {
}

void DiagnoseFileCommand::AcceptArgs(vector<LPVOID> argslist) {
    DiagnoseFileCommand::ArgsList = argslist;
}

void DiagnoseFileCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (ArgsList.size() != 2) {
        cout << "Invalid Args! Usage: diagnosefile <drive_letter> <filename>" << endl;
        cout << "Examples:" << endl;
        cout << "  diagnosefile C test.txt          - Search for exact filename" << endl;
        cout << "  diagnosefile C test              - Search for files containing 'test'" << endl;
        return;
    }

    try {
        string& driveStr = *(string*)ArgsList[0];
        string& fileNameStr = *(string*)ArgsList[1];

        if (driveStr.empty() || fileNameStr.empty()) {
            cout << "Invalid arguments." << endl;
            return;
        }

        char driveLetter = driveStr[0];
        wstring searchName(fileNameStr.begin(), fileNameStr.end());

        cout << "\n========== File Diagnostic Tool ==========\n" << endl;
        cout << "Drive: " << driveLetter << ":" << endl;
        wcout << L"Searching for: " << searchName << endl;
        cout << endl;

        // 创建 MFT 读取器和解析器
        MFTReader reader;
        if (!reader.OpenVolume(driveLetter)) {
            cout << "ERROR: Failed to open volume " << driveLetter << ":" << endl;
            cout << "Administrator privileges are required." << endl;
            return;
        }

        MFTParser parser(&reader);
        PathResolver pathResolver(&reader, &parser);

        ULONGLONG totalRecords = reader.GetTotalMFTRecords();
        cout << "Total MFT records: " << totalRecords << endl;
        cout << "Scanning..." << endl;
        cout << endl;

        vector<BYTE> record;
        ULONGLONG foundCount = 0;
        ULONGLONG scannedCount = 0;
        ULONGLONG activeFiles = 0;
        ULONGLONG deletedFiles = 0;

        // 转换为小写进行不区分大小写的搜索
        wstring searchNameLower = searchName;
        transform(searchNameLower.begin(), searchNameLower.end(),
                  searchNameLower.begin(), ::towlower);

        // 扫描所有 MFT 记录
        for (ULONGLONG i = 16; i < totalRecords; i++) {
            if (!reader.ReadMFTRecord(i, record)) {
                continue;
            }

            scannedCount++;

            // 解析文件名
            ULONGLONG parentDir;
            wstring fileName = parser.GetFileNameFromRecord(record, parentDir);

            if (fileName.empty()) {
                continue;
            }

            // 转换为小写进行比较
            wstring fileNameLower = fileName;
            transform(fileNameLower.begin(), fileNameLower.end(),
                      fileNameLower.begin(), ::towlower);

            // 检查是否匹配
            if (fileNameLower.find(searchNameLower) != wstring::npos) {
                FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
                bool isDeleted = ((header->Flags & 0x01) == 0);
                bool isDirectory = ((header->Flags & 0x02) != 0);

                if (isDeleted) {
                    deletedFiles++;
                } else {
                    activeFiles++;
                }

                foundCount++;

                // 显示找到的文件
                cout << "\n[" << foundCount << "] MFT Record #" << i << endl;
                wcout << L"  Name: " << fileName << endl;
                cout << "  Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
                cout << "  Type: " << (isDirectory ? "Directory" : "File") << endl;
                cout << "  Parent MFT#: " << parentDir << endl;

                // 尝试重建路径
                try {
                    wstring fullPath = pathResolver.ReconstructPath(i);
                    if (!fullPath.empty()) {
                        wcout << L"  Full Path: " << fullPath << endl;
                    }
                } catch (...) {
                    cout << "  Full Path: (unable to reconstruct)" << endl;
                }

                if (foundCount >= 50) {
                    cout << "\n(Limiting results to first 50 matches)" << endl;
                    break;
                }
            }

            // 显示进度
            if (scannedCount % 100000 == 0) {
                cout << "\r  Progress: " << scannedCount << " / " << totalRecords
                     << " (" << (scannedCount * 100 / totalRecords) << "%)" << flush;
            }
        }

        cout << "\r                                                                " << flush;
        cout << "\r";

        // 显示统计信息
        cout << "\n========== Scan Results ==========\n" << endl;
        cout << "Total MFT records scanned: " << scannedCount << endl;
        cout << "Total matches found: " << foundCount << endl;
        cout << "  - Active files: " << activeFiles << endl;
        cout << "  - Deleted files: " << deletedFiles << endl;

        if (foundCount == 0) {
            cout << "\nNo files matching '" << fileNameStr << "' were found." << endl;
            cout << "\nPossible reasons:" << endl;
            cout << "  1. File was never created on this volume" << endl;
            cout << "  2. MFT record was reused (old data overwritten)" << endl;
            cout << "  3. Try using USN Journal: scanusn " << driveLetter << endl;
        } else {
            cout << "\nNote: If your target file is not in the list above:" << endl;
            cout << "  - It may have been created with a different name" << endl;
            cout << "  - Try USN Journal for recently deleted files: scanusn " << driveLetter << endl;
        }

    } catch (const exception& e) {
        cout << "[ERROR] Exception: " << e.what() << endl;
    } catch (...) {
        cout << "[ERROR] Unknown exception in DiagnoseFileCommand::Execute" << endl;
    }
}

BOOL DiagnoseFileCommand::HasArgs() {
    return FlagHasArgs;
}

BOOL DiagnoseFileCommand::CheckName(string input) {
    if (input.compare(name) == 0) {
        return TRUE;
    }
    return FALSE;
}
```

---

## 🔨 集成步骤

### 1. 添加命令名称到 Main.cpp

在 Main.cpp 中添加（在其他命令名称定义之后）：

```cpp
string ScanUsnCommand::name = "scanusn |name |name";
string DiagnoseFileCommand::name = "diagnosefile |name |name";
```

### 2. 注册命令到 CLI

在 cli.cpp 的构造函数中添加（在其他命令注册之后）：

```cpp
ParseCommands(ScanUsnCommand::name, ScanUsnCommand::GetInstancePtr());
ParseCommands(DiagnoseFileCommand::name, DiagnoseFileCommand::GetInstancePtr());
```

### 3. 添加命令执行检查到 cli.cpp

在 cli.cpp 的 Run 方法中添加（在其他命令执行检查之后）：

```cpp
if (ScanUsnCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
    ScanUsnCommand* scanusncommand = (ScanUsnCommand*)commandclassptr;
    scanusncommand->AcceptArgs(argsinstances);
    scanusncommand->Execute(currectcommandname);
}
if (DiagnoseFileCommand::CheckName(currectcommandname) && climodule->GetModuleFlagByName(currectcommandname)) {
    DiagnoseFileCommand* diagnosefilecommand = (DiagnoseFileCommand*)commandclassptr;
    diagnosefilecommand->AcceptArgs(argsinstances);
    diagnosefilecommand->Execute(currectcommandname);
}
```

### 4. 添加到项目文件 (.vcxproj)

在 Visual Studio 中：
1. 右键项目 → 添加 → 现有项
2. 选择 `UsnJournalReader.h` 和 `UsnJournalReader.cpp`
3. 或者手动编辑 .vcxproj 文件添加：

```xml
<ClCompile Include="UsnJournalReader.cpp" />
<ClInclude Include="UsnJournalReader.h" />
```

---

## 🧪 测试流程

### 测试 1：USN Journal 扫描

```bash
# 扫描最近1小时删除的文件
scanusn C

# 扫描最近24小时删除的文件
scanusn C 24
```

**预期结果**：
- 显示 USN Journal 统计信息
- 列出最近删除的文件及其删除时间
- 包括 MFT 记录号

### 测试 2：文件诊断

```bash
# 运行测试脚本创建并删除文件
.\test_file_recovery.ps1

# 然后诊断测试文件
diagnosefile C test_recovery_file.txt
```

**预期结果**：
- 显示所有匹配的 MFT 记录
- 区分 ACTIVE 和 DELETED 状态
- 显示完整路径（如果可以重建）

---

## 📊 预期优势

1. **USN Journal 的优势**：
   - ✅ 追踪最近的删除操作
   - ✅ 不依赖 MFT Flags
   - ✅ 包含精确的删除时间
   - ✅ 可以检测 MFT 扫描遗漏的文件

2. **文件诊断的优势**：
   - ✅ 全面扫描整个 MFT
   - ✅ 找出所有匹配的记录
   - ✅ 区分活动和删除状态
   - ✅ 帮助定位问题

---

## ⚠️ 注意事项

1. **USN Journal 要求**：
   - 需要管理员权限
   - 需要在卷上启用 USN Journal
   - 只能追踪 Journal 启用后的变化

2. **性能考虑**：
   - USN Journal 扫描比 MFT 扫描更快
   - 但仅限于最近的变化
   - diagnosefile 扫描整个 MFT 较慢

3. **局限性**：
   - 如果 MFT 记录被重用，旧文件信息将丢失
   - USN Journal 有大小限制，旧记录会被覆盖

---

**实现日期**：2025-12-31
**版本**：1.0
**下一步**：完成代码集成并测试
