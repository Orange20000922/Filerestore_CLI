// UsnRecoverCommands.cpp - USN 定点恢复命令实现
// 包含: UsnListCommand, UsnRecoverCommand, RecoverCommand

#include "cmd.h"
#include "CommandUtils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <algorithm>
#include <filesystem>
#include "MFTReader.h"
#include "MFTParser.h"
#include "UsnTargetedRecovery.h"
#include "LocalizationManager.h"
#include "FileCarver.h"
#include "FileCarverRecovery.h"
#include "TripleValidator.h"
#include "MFTCache.h"
#include "MFTSnapshotStore.h"
#include "UsnDeleteMonitor.h"
#include "MonitorDaemon.h"
#include "components/TuiInputBridge.h"
#include "CarvedResultsCache.h"
namespace fs = std::filesystem;

using namespace std;

// ============================================================================
// UsnListCommand - 列出 USN 删除记录（带验证）
// ============================================================================
// 用法: usnlist <drive_letter> [hours] [--validate] [--pattern=<name>]
// 示例: usnlist C 24
//       usnlist C 48 --validate
//       usnlist C 24 --pattern=document
// ============================================================================
DEFINE_COMMAND_BASE(UsnListCommand, "usnlist |name |name |name |name", TRUE)
REGISTER_COMMAND(UsnListCommand);

void UsnListCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (GET_ARG_COUNT() < 1) {
        cout << LOC_STR("usnlist.usage") << endl;
        cout << "  usnlist <drive> [hours] [--validate] [--pattern=<name>] [--limit=<n>]" << endl;
        cout << LOC_STR("usnlist.examples") << endl;
        cout << "  usnlist C 24" << endl;
        cout << "  usnlist C 48 --validate" << endl;
        cout << "  usnlist C 24 --pattern=document" << endl;
        cout << "  usnlist C 168 --validate --limit=5000" << endl;
        return;
    }

    // 解析参数
    string& driveStr = GET_ARG_STRING(0);
    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << LOC_STR("error.invalid_drive") << endl;
        return;
    }

    int maxHours = 24;
    bool doValidate = false;
    wstring pattern = L"";
    size_t maxResults = 1000;  // 默认限制

    for (int i = 1; i < GET_ARG_COUNT(); i++) {
        string& arg = GET_ARG_STRING(i);
        if (arg == "--validate" || arg == "-v") {
            doValidate = true;
        }
        else if (arg.substr(0, 10) == "--pattern=" || arg.substr(0, 2) == "-p") {
            size_t pos = arg.find('=');
            if (pos != string::npos) {
                string patternStr = arg.substr(pos + 1);
                pattern = UsnTargetedRecovery::NarrowToWide(patternStr);
            }
        }
        else if (arg.substr(0, 8) == "--limit=") {
            try {
                maxResults = stoul(arg.substr(8));
                if (maxResults == 0) maxResults = 1000;
            }
            catch (...) {}
        }
        else {
            try {
                maxHours = stoi(arg);
                if (maxHours <= 0) maxHours = 24;
            }
            catch (...) {}
        }
    }

    // 初始化组件
    MFTReader reader;
    if (!reader.OpenVolume(driveLetter)) {
        cout << LOC_STR("error.open_volume_failed") << ": " << driveLetter << ":/" << endl;
        return;
    }

    // 加载 MFT data runs（支持碎片化 MFT 的正确记录定位）
    reader.GetTotalMFTRecords();

    MFTParser parser(&reader);
    UsnTargetedRecovery recovery(&reader, &parser);

    cout << "\n" << LOC_STR("usnlist.title") << endl;
    cout << "========================================" << endl;
    cout << LOC_STR("usnlist.drive") << ": " << driveLetter << ":/" << endl;
    cout << LOC_STR("usnlist.time_range") << ": " << maxHours << " " << LOC_STR("usnlist.hours") << endl;
    if (!pattern.empty()) {
        cout << LOC_STR("usnlist.pattern") << ": ";
        wcout << pattern << endl;
    }
    cout << LOC_STR("usnlist.validate") << ": " << (doValidate ? LOC_STR("common.yes") : LOC_STR("common.no")) << endl;
    cout << "========================================\n" << endl;

    // 搜索并验证
    cout << LOC_STR("usnlist.scanning") << "..." << endl;

    vector<UsnFileListItem> results = recovery.SearchAndValidate(
        driveLetter, maxHours, pattern, maxResults);

    if (results.empty()) {
        cout << LOC_STR("usnlist.no_results") << endl;
        return;
    }

    // 显示结果
    cout << "\n" << LOC_STR("usnlist.found") << ": " << results.size() << " " << LOC_STR("usnlist.files") << "\n" << endl;

    // 表头
    cout << left << setw(6) << LOC_STR("usnlist.col_idx")
         << setw(40) << LOC_STR("usnlist.col_name")
         << setw(12) << LOC_STR("usnlist.col_size")
         << setw(20) << LOC_STR("usnlist.col_time");

    if (doValidate) {
        cout << setw(12) << LOC_STR("usnlist.col_status")
             << setw(8) << LOC_STR("usnlist.col_conf");
    }
    cout << endl;

    cout << string(doValidate ? 98 : 78, '-') << endl;

    int idx = 0;
    int recoverableCount = 0;

    for (const auto& item : results) {
        // 跳过目录
        if (item.usnInfo.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }

        // 文件名（截断）
        wstring fileName = item.usnInfo.FileName;
        if (fileName.length() > 38) {
            fileName = fileName.substr(0, 35) + L"...";
        }

        // 时间戳
        wstring timeStr = UsnTargetedRecovery::FormatTimestamp(item.usnInfo.TimeStamp);

        cout << left << setw(6) << idx;
        // 输出宽字符文件名
        string fileNameNarrow = UsnTargetedRecovery::WideToNarrow(fileName);
        cout << setw(40) << fileNameNarrow;

        // 文件大小（从 MFT 获取）
        if (item.usnInfo.MftInfoValid && item.usnInfo.FileSize > 0) {
            string sizeStr = UsnTargetedRecovery::WideToNarrow(
                UsnTargetedRecovery::FormatFileSize(item.usnInfo.FileSize));
            cout << setw(12) << sizeStr;
        } else {
            cout << setw(12) << "-";
        }
        string timeNarrow = UsnTargetedRecovery::WideToNarrow(timeStr);
        cout << setw(20) << timeNarrow;

        if (doValidate) {
            // 状态
            string statusStr;
            if (item.canRecover) {
                statusStr = "[OK]";
                recoverableCount++;
            }
            else {
                statusStr = "[" + UsnTargetedRecovery::GetStatusString(item.status) + "]";
            }
            cout << setw(12) << statusStr;

            // 置信度
            if (item.confidence > 0) {
                cout << setw(8) << fixed << setprecision(0) << (item.confidence * 100) << "%";
            }
            else {
                cout << setw(8) << "-";
            }
        }
        cout << endl;
        idx++;
    }

    cout << string(doValidate ? 98 : 78, '-') << endl;
    cout << LOC_STR("usnlist.total") << ": " << idx << " " << LOC_STR("usnlist.files");
    if (doValidate) {
        cout << ", " << LOC_STR("usnlist.recoverable") << ": " << recoverableCount;
    }
    cout << "\n" << endl;

    cout << LOC_STR("usnlist.hint") << ":" << endl;
    cout << "  usnrecover " << driveLetter << " <index> <output_dir>" << endl;
    cout << "  usnrecover " << driveLetter << " <filename> <output_dir>" << endl;
}

// ============================================================================
// UsnRecoverCommand - USN 定点恢复
// ============================================================================
// 用法: usnrecover <drive_letter> <index|filename|record> <output_dir> [--force]
// 示例: usnrecover C 0 D:\recovered\
//       usnrecover C document.docx D:\recovered\
//       usnrecover C 0x12345 D:\recovered\ --force
// ============================================================================
DEFINE_COMMAND_BASE(UsnRecoverCommand, "usnrecover |name |name |file |name", TRUE)
REGISTER_COMMAND(UsnRecoverCommand);

// 保存上次搜索结果（用于按索引恢复）
static vector<UsnDeletedFileInfo> g_lastUsnSearchResults;
static char g_lastUsnDrive = 0;

void UsnRecoverCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (GET_ARG_COUNT() < 1) {
        cout << LOC_STR("usnrecover.usage") << endl;
        cout << "  usnrecover <drive> <index|filename|record> <output_dir> [--force]" << endl;
        cout << "  usnrecover <drive> <hours> --pattern=<name> --output=<dir> [--batch] [--force]" << endl;
        cout << LOC_STR("usnrecover.examples") << endl;
        cout << "  usnrecover C 0 D:\\recovered\\" << endl;
        cout << "  usnrecover C document.docx D:\\recovered\\" << endl;
        cout << "  usnrecover C 1 --pattern=test_ --output=D:\\out --batch" << endl;
        cout << "  usnrecover C 0x12345 D:\\recovered\\ --force" << endl;
        return;
    }

    // ========== 统一解析位置参数和命名参数 ==========
    string driveStr;
    string targetStr;
    string outputStr;
    string patternStr;
    bool forceRecover = false;
    bool batchMode = false;
    int hoursArg = 1;  // 默认搜索最近1小时

    // 收集位置参数
    vector<string> positionalArgs;

    for (size_t i = 0; i < GET_ARG_COUNT(); i++) {
        string arg = GET_ARG_STRING(i);

        if (arg.find("--output=") == 0) {
            outputStr = arg.substr(9);
        } else if (arg.find("--pattern=") == 0) {
            patternStr = arg.substr(10);
        } else if (arg == "--force" || arg == "-f") {
            forceRecover = true;
        } else if (arg == "--batch" || arg == "-b") {
            batchMode = true;
        } else if (arg.find("--hours=") == 0) {
            hoursArg = stoi(arg.substr(8));
        } else if (arg[0] != '-') {
            // 位置参数
            positionalArgs.push_back(arg);
        }
    }

    // 从位置参数中提取值
    if (positionalArgs.size() >= 1) {
        driveStr = positionalArgs[0];
    }
    if (positionalArgs.size() >= 2) {
        targetStr = positionalArgs[1];
    }
    if (positionalArgs.size() >= 3 && outputStr.empty()) {
        outputStr = positionalArgs[2];
    }

    // 验证必要参数
    if (driveStr.empty() || outputStr.empty()) {
        cout << "Error: Missing required arguments (drive and output_dir)" << endl;
        cout << "Usage: usnrecover <drive> <index|filename|hours> <output_dir>" << endl;
        return;
    }

    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << LOC_STR("error.invalid_drive") << endl;
        return;
    }

    wstring outputDir = UsnTargetedRecovery::NarrowToWide(outputStr);

    // 初始化组件
    MFTReader reader;
    if (!reader.OpenVolume(driveLetter)) {
        cout << LOC_STR("error.open_volume_failed") << ": " << driveLetter << ":/" << endl;
        return;
    }

    // 加载 MFT data runs（支持碎片化 MFT 的正确记录定位）
    reader.GetTotalMFTRecords();

    MFTParser parser(&reader);
    UsnTargetedRecovery recovery(&reader, &parser);

    cout << "\n" << LOC_STR("usnrecover.title") << endl;
    cout << "========================================" << endl;

    UsnTargetedRecoveryResult result;

    // ========== 模式判断 ==========
    // 如果指定了 --pattern，使用批量模式搜索
    if (!patternStr.empty()) {
        cout << LOC_STR("usnrecover.mode_batch") << ": pattern=" << patternStr << ", hours=" << hoursArg << endl;
        cout << LOC_STR("usnrecover.searching") << "..." << endl;

        // 搜索匹配的文件
        wstring pattern = UsnTargetedRecovery::NarrowToWide(patternStr);
        auto matchedFiles = recovery.SearchAndValidate(driveLetter, hoursArg * 3600, pattern, 1000);

        if (matchedFiles.empty()) {
            cout << LOC_STR("usnrecover.no_match") << endl;
            return;
        }

        cout << "Found " << matchedFiles.size() << " matching files" << endl;

        // 批量恢复
        size_t successCount = 0;
        size_t failCount = 0;

        for (const auto& item : matchedFiles) {
            if (item.canRecover || forceRecover) {
                UsnTargetedRecoveryResult recResult = recovery.Recover(item.usnInfo, outputDir, forceRecover);

                if (recResult.status == UsnRecoveryStatus::SUCCESS ||
                    recResult.status == UsnRecoveryStatus::RESIDENT_DATA ||
                    recResult.status == UsnRecoveryStatus::PARTIAL_RECOVERY ||
                    recResult.status == UsnRecoveryStatus::MFT_REUSED_DATA_VALID) {
                    successCount++;
                    string fname = UsnTargetedRecovery::WideToNarrow(item.usnInfo.FileName);
                    cout << "  [OK] " << fname << endl;
                } else {
                    failCount++;
                }
            } else {
                failCount++;
            }
        }

        cout << "\n========================================" << endl;
        cout << "Batch recovery complete:" << endl;
        cout << "  Success: " << successCount << endl;
        cout << "  Failed:  " << failCount << endl;
        cout << "========================================\n" << endl;
        return;
    }

    // 判断目标类型：索引、MFT记录号、或文件名
    bool isIndex = true;
    bool isRecordNumber = false;
    int index = -1;
    ULONGLONG recordNumber = 0;

    // 尝试解析为数字（索引）
    try {
        if (targetStr.substr(0, 2) == "0x" || targetStr.substr(0, 2) == "0X") {
            // 十六进制 MFT 记录号
            recordNumber = stoull(targetStr, nullptr, 16);
            isRecordNumber = true;
            isIndex = false;
        }
        else {
            index = stoi(targetStr);
            if (index < 0) {
                isIndex = false;
            }
        }
    }
    catch (...) {
        isIndex = false;
    }

    if (isRecordNumber) {
        // 按 MFT 记录号恢复
        cout << LOC_STR("usnrecover.mode_record") << ": 0x" << hex << recordNumber << dec << endl;

        // 需要先从 USN 搜索该记录
        UsnJournalReader usnReader;
        if (!usnReader.Open(driveLetter)) {
            cout << LOC_STR("error.usn_open_failed") << endl;
            return;
        }

        // 扫描最近删除的文件
        auto deletedFiles = usnReader.ScanRecentlyDeletedFiles(24 * 3600, 10000);

        // 查找匹配的记录
        UsnDeletedFileInfo* targetInfo = nullptr;
        for (auto& info : deletedFiles) {
            if (info.GetMftRecordNumber() == recordNumber) {
                targetInfo = &info;
                break;
            }
        }

        if (!targetInfo) {
            cout << LOC_STR("usnrecover.record_not_found") << endl;
            return;
        }

        result = recovery.Recover(*targetInfo, outputDir, forceRecover);
    }
    else if (isIndex) {
        // 按索引恢复（需要先执行 usnlist）
        cout << LOC_STR("usnrecover.mode_index") << ": " << index << endl;

        // 如果没有缓存或驱动器不同，重新搜索
        if (g_lastUsnSearchResults.empty() || g_lastUsnDrive != driveLetter) {
            cout << LOC_STR("usnrecover.searching") << "..." << endl;

            UsnJournalReader usnReader;
            if (!usnReader.Open(driveLetter)) {
                cout << LOC_STR("error.usn_open_failed") << endl;
                return;
            }

            g_lastUsnSearchResults = usnReader.ScanRecentlyDeletedFiles(24 * 3600, 1000);
            g_lastUsnDrive = driveLetter;

            // 过滤掉目录
            vector<UsnDeletedFileInfo> filtered;
            for (const auto& info : g_lastUsnSearchResults) {
                if (!(info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    filtered.push_back(info);
                }
            }
            g_lastUsnSearchResults = filtered;
        }

        if (index >= (int)g_lastUsnSearchResults.size()) {
            cout << LOC_STR("usnrecover.index_out_of_range") << ": " << index << endl;
            cout << LOC_STR("usnrecover.valid_range") << ": 0-" << (g_lastUsnSearchResults.size() - 1) << endl;
            return;
        }

        result = recovery.Recover(g_lastUsnSearchResults[index], outputDir, forceRecover);
    }
    else {
        // 按文件名搜索并恢复
        wstring fileName = UsnTargetedRecovery::NarrowToWide(targetStr);
        cout << LOC_STR("usnrecover.mode_name") << ": " << targetStr << endl;

        result = recovery.SearchAndRecover(driveLetter, fileName, outputDir, forceRecover);
    }

    // 显示结果
    cout << "\n" << LOC_STR("usnrecover.result") << endl;
    cout << "----------------------------------------" << endl;

    string fileNameNarrow = UsnTargetedRecovery::WideToNarrow(result.usnInfo.FileName);
    cout << LOC_STR("usnrecover.filename") << ": " << fileNameNarrow << endl;
    cout << LOC_STR("usnrecover.mft_record") << ": " << result.mftRecordNumber << endl;

    if (result.dataRuns.size() > 0) {
        cout << LOC_STR("usnrecover.data_runs") << ": " << result.dataRuns.size() << " " << LOC_STR("usnrecover.fragments") << endl;
        cout << LOC_STR("usnrecover.total_clusters") << ": " << result.totalClusters << endl;
    }

    if (result.isResident) {
        cout << LOC_STR("usnrecover.resident") << ": " << LOC_STR("common.yes") << endl;
    }

    cout << LOC_STR("usnrecover.sequence_match") << ": "
         << (result.sequenceMatched ? LOC_STR("common.yes") : LOC_STR("common.no"))
         << " (" << result.expectedSequence << " vs " << result.actualSequence << ")" << endl;

    cout << LOC_STR("usnrecover.detected_type") << ": " << result.detectedType << endl;
    cout << LOC_STR("usnrecover.confidence") << ": " << fixed << setprecision(1) << (result.confidence * 100) << "%" << endl;

    cout << "\n" << LOC_STR("usnrecover.status") << ": ";

    // 根据状态显示不同颜色/标记
    switch (result.status) {
        case UsnRecoveryStatus::SUCCESS:
            cout << "[SUCCESS] " << LOC_STR("usnrecover.status_success") << endl;
            break;
        case UsnRecoveryStatus::RESIDENT_DATA:
            cout << "[SUCCESS] " << LOC_STR("usnrecover.status_resident") << endl;
            break;
        case UsnRecoveryStatus::PARTIAL_RECOVERY:
            cout << "[WARNING] " << LOC_STR("usnrecover.status_partial") << endl;
            break;
        case UsnRecoveryStatus::MFT_RECORD_REUSED:
            cout << "[FAILED] " << LOC_STR("usnrecover.status_reused") << endl;
            break;
        case UsnRecoveryStatus::SIGNATURE_MISMATCH:
            cout << "[FAILED] " << LOC_STR("usnrecover.status_mismatch") << endl;
            break;
        case UsnRecoveryStatus::NO_DATA_ATTRIBUTE:
            cout << "[FAILED] " << LOC_STR("usnrecover.status_no_data") << endl;
            break;
        default:
            cout << "[FAILED] " << result.statusMessage << endl;
            break;
    }

    if (!result.recoveredPath.empty()) {
        string pathNarrow = UsnTargetedRecovery::WideToNarrow(result.recoveredPath);
        string sizeStr = UsnTargetedRecovery::WideToNarrow(UsnTargetedRecovery::FormatFileSize(result.recoveredSize));
        cout << "\n" << LOC_STR("usnrecover.output_path") << ": " << pathNarrow << endl;
        cout << LOC_STR("usnrecover.recovered_size") << ": " << sizeStr << endl;
    }

    cout << "========================================\n" << endl;

    if (!result.canRecover && !forceRecover) {
        cout << LOC_STR("usnrecover.hint_force") << ":" << endl;
        cout << "  usnrecover " << driveStr << " " << targetStr << " " << outputStr << " --force" << endl;
    }
}

// ============================================================================
// RecoverCommand - 智能文件恢复向导（USN + 签名扫描联合）
// ============================================================================
// 用法: recover <drive> [filename] [output_dir]
//       recover <drive> [hours] --pattern=<name> --output=<dir>
// 示例: recover C
//       recover C document.docx
//       recover C document.docx D:\recovered
//       recover C 1 --pattern=test_ --output=D:\out
// ============================================================================

DEFINE_COMMAND_BASE(RecoverCommand, "recover |name |name |name", TRUE)
REGISTER_COMMAND(RecoverCommand);

void RecoverCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (GET_ARG_COUNT() < 1) {
        cout << "\n=== 智能文件恢复 ===" << endl;
        cout << "用法: recover <drive> [filename] [output_dir]" << endl;
        cout << "      recover <drive> [hours] --pattern=<name> --output=<dir>" << endl;
        cout << "\n示例:" << endl;
        cout << "  recover C                          # 交互式搜索" << endl;
        cout << "  recover C document.docx            # 搜索指定文件" << endl;
        cout << "  recover C document.docx D:\\out     # 直接恢复到指定目录" << endl;
        cout << "  recover C 1 --pattern=test_ --output=D:\\out  # 批量模式" << endl;
        return;
    }

    // ========== 统一解析位置参数和命名参数 ==========
    string driveStr;
    string targetStr;
    string outputDir;
    string patternStr;
    bool hascarveresult = false;
    int hoursArg = 1;  // 默认搜索最近1小时

    // 收集位置参数
    vector<string> positionalArgs;

    for (size_t i = 0; i < GET_ARG_COUNT(); i++) {
        string arg = GET_ARG_STRING(i);

        if (arg.find("--output=") == 0) {
            outputDir = arg.substr(9);
        } else if (arg.find("--pattern=") == 0) {
            patternStr = arg.substr(10);
        } else if (arg.find("--hours=") == 0) {
            hoursArg = stoi(arg.substr(8));
        } else if (arg[0] != '-') {
            // 位置参数
            positionalArgs.push_back(arg);
        }
    }

    // 从位置参数中提取值
    if (positionalArgs.size() >= 1) {
        driveStr = positionalArgs[0];
    }
    if (positionalArgs.size() >= 2 && patternStr.empty()) {
        targetStr = positionalArgs[1];
    } else if (positionalArgs.size() >= 2) {
        // 如果有 --pattern，第二个参数可能是 hours
        try {
            hoursArg = stoi(positionalArgs[1]);
        } catch (...) {}
    }
    if (positionalArgs.size() >= 3 && outputDir.empty()) {
        outputDir = positionalArgs[2];
    }

    // 解析驱动器
    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << "错误: 无效的驱动器字母" << endl;
        return;
    }

    // 解析文件名（可选）
    wstring targetFileName = L"";
    if (!targetStr.empty()) {
        targetFileName = UsnTargetedRecovery::NarrowToWide(targetStr);
    } else if (!patternStr.empty()) {
        targetFileName = UsnTargetedRecovery::NarrowToWide(patternStr);
    }

    // 初始化组件
    MFTReader reader;
    if (!reader.OpenVolume(driveLetter)) {
        cout << "错误: 无法打开卷 " << driveLetter << ":/" << endl;
        return;
    }

    // 加载 MFT data runs（支持碎片化 MFT 的正确记录定位）
    reader.GetTotalMFTRecords();

    MFTParser parser(&reader);
    UsnTargetedRecovery recovery(&reader, &parser);

    // 尝试加载快照存储（用于 MFT 已复用时的回退恢复）
    MFTSnapshotStore snapshotStore;
    string snapshotPath = MFTSnapshotStore::GenerateStorePath(driveLetter);
    if (snapshotStore.LoadFromFile(snapshotPath)) {
        recovery.SetSnapshotStore(&snapshotStore);
        cout << "  快照存储已加载 (" << snapshotStore.GetCount() << " 个快照)" << endl;
    }

    cout << "\n=== 智能文件恢复 ===" << endl;
    cout << "驱动器: " << driveLetter << ":/" << endl;

    // ========== 第1步：如果没有指定文件名，进入交互式搜索 ==========
    if (targetFileName.empty()) {
        string input;
        if (!TuiInputBridge::Instance().GetLine("\n请输入要恢复的文件名（支持部分匹配）: ", input) || input.empty()) {
            cout << "已取消" << endl;
            return;
        }
        targetFileName = UsnTargetedRecovery::NarrowToWide(input);
    }

    cout << "\n正在搜索: ";
    wcout << targetFileName << endl;

    // ========== 第1步：USN 搜索 + MFT 验证 ==========
    cout << "\n[1/4] 搜索 USN 删除记录并验证 MFT..." << endl;

    vector<UsnFileListItem> usnResults = recovery.SearchAndValidate(
        driveLetter, 168, targetFileName, 100);  // 搜索最近7天

    // 分类：可直接恢复 vs 需要签名扫描
    vector<size_t> recoverableIndices;  // usnResults 中可直接恢复的索引
    vector<UsnDeletedFileInfo> matchedUsn;
    for (size_t i = 0; i < usnResults.size(); i++) {
        auto& item = usnResults[i];
        if (item.usnInfo.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        matchedUsn.push_back(item.usnInfo);
        if (item.canRecover) {
            recoverableIndices.push_back(i);
        }
    }

    if (matchedUsn.empty()) {
        cout << "  未在 USN 日志中找到匹配的删除记录" << endl;
    } else {
        cout << "  找到 " << matchedUsn.size() << " 条 USN 删除记录" << endl;
        cout << "  其中 " << recoverableIndices.size() << " 个文件的 MFT 数据可用" << endl;

        // 填充 MFT 信息获取文件大小（用于显示）
        cout << "\n[2/4] 从 MFT 获取文件信息..." << endl;
        size_t enriched = recovery.EnrichWithMFTBatch(matchedUsn);
        cout << "  成功获取 " << enriched << " 个文件的大小信息" << endl;
    }

    // ========== MFT 直接恢复（快速路径）==========
    if (!recoverableIndices.empty()) {
        cout << "\n========================================" << endl;
        cout << "MFT 直接恢复（无需全盘扫描）" << endl;
        cout << "========================================" << endl;

        // 显示可直接恢复的文件列表
        cout << "\n以下文件可通过 MFT 数据直接恢复:" << endl;
        cout << string(70, '-') << endl;
        cout << left << setw(6) << "编号" << setw(40) << "文件名"
             << setw(12) << "大小" << "状态" << endl;
        cout << string(70, '-') << endl;

        size_t displayLimit = min(recoverableIndices.size(), (size_t)20);
        for (size_t i = 0; i < displayLimit; i++) {
            auto& item = usnResults[recoverableIndices[i]];
            string fname = UsnTargetedRecovery::WideToNarrow(item.usnInfo.FileName);
            if (fname.length() > 38) fname = fname.substr(0, 35) + "...";

            string sizeStr = "未知";
            if (item.usnInfo.MftInfoValid && item.usnInfo.FileSize > 0) {
                sizeStr = UsnTargetedRecovery::WideToNarrow(
                    UsnTargetedRecovery::FormatFileSize(item.usnInfo.FileSize));
            }

            cout << left << setw(6) << i
                 << setw(40) << fname
                 << setw(12) << sizeStr
                 << "[" << UsnTargetedRecovery::GetStatusString(item.status) << "]"
                 << endl;
        }
        if (recoverableIndices.size() > 20) {
            cout << "  ... 还有 " << (recoverableIndices.size() - 20) << " 个文件" << endl;
        }
        cout << endl;

        if (outputDir.empty()) {
            // 交互模式：让用户选择
            string input;
            if (!TuiInputBridge::Instance().GetLine("输入编号直接恢复，'s' 全盘签名扫描，'q' 退出: ", input)) {
                cout << "已取消" << endl;
                return;
            }

            if (input == "q" || input == "Q" || input.empty()) {
                cout << "已取消" << endl;
                return;
            }

            if (input != "s" && input != "S") {
                // 用户选择了一个文件编号
                size_t selectedIndex;
                try {
                    selectedIndex = stoul(input);
                } catch (...) {
                    cout << "无效的输入" << endl;
                    return;
                }

                if (selectedIndex >= min(recoverableIndices.size(), (size_t)20)) {
                    cout << "编号超出范围" << endl;
                    return;
                }

                // 询问输出目录
                string outDir;
                TuiInputBridge::Instance().GetLine("输入输出目录 (直接回车使用当前目录): ", outDir);
                if (outDir.empty()) outDir = ".";

                auto& selectedItem = usnResults[recoverableIndices[selectedIndex]];
                wstring wOutputDir = UsnTargetedRecovery::NarrowToWide(outDir);
                string fname = UsnTargetedRecovery::WideToNarrow(selectedItem.usnInfo.FileName);

                cout << "\n正在通过 MFT 数据直接恢复: " << fname << " ..." << endl;

                UsnTargetedRecoveryResult recResult = recovery.Recover(selectedItem.usnInfo, wOutputDir);

                if (recResult.status == UsnRecoveryStatus::SUCCESS ||
                    recResult.status == UsnRecoveryStatus::RESIDENT_DATA ||
                    recResult.status == UsnRecoveryStatus::PARTIAL_RECOVERY) {
                    string path = UsnTargetedRecovery::WideToNarrow(recResult.recoveredPath);
                    if (recResult.usedClusterFiltering && recResult.clusterHealth.overwrittenClusters > 0) {
                        cout << "\n=== 部分恢复 ===" << endl;
                    } else {
                        cout << "\n=== 恢复成功 ===" << endl;
                    }
                    cout << "文件大小: " << recResult.recoveredSize << " bytes" << endl;
                    cout << "已保存到: " << path << endl;
                    if (recResult.signatureMatched) {
                        cout << "签名验证: 通过 (" << recResult.detectedType << ")" << endl;
                    }
                    if (recResult.usedClusterFiltering && recResult.clusterHealth.overwrittenClusters > 0) {
                        cout << "簇健康: " << recResult.clusterHealth.goodClusters << "/"
                             << recResult.clusterHealth.totalClusters << " ("
                             << fixed << setprecision(1) << recResult.clusterHealth.healthPercentage
                             << "%) | 覆写簇: " << recResult.clusterHealth.overwrittenClusters
                             << " | 检测: " << fixed << setprecision(1)
                             << recResult.clusterHealth.detectionTimeMs << "ms" << endl;
                        if (recResult.clusterHealth.formatTruncated) {
                            cout << "格式截断: 移除了 " << recResult.clusterHealth.truncatedBytes << " bytes" << endl;
                        }
                    }
                    return;
                } else {
                    cout << "\nMFT 直接恢复失败: " << recResult.statusMessage << endl;
                    // 询问是否回退到签名扫描
                    string fallbackChoice;
                    TuiInputBridge::Instance().GetLine("是否尝试签名扫描恢复? (y/n): ", fallbackChoice);
                    if (fallbackChoice != "y" && fallbackChoice != "Y") {
                        return;
                    }
                    cout << "\n回退到签名扫描..." << endl;
                    // 继续执行签名扫描（不 return）
                }
            } else {
                // input == "s" → 用户选择全盘扫描
                cout << "\n用户选择全盘签名扫描..." << endl;
            }
        } else {
            // 自动模式：直接恢复最佳匹配
            auto& bestItem = usnResults[recoverableIndices[0]];
            wstring wOutputDir = UsnTargetedRecovery::NarrowToWide(outputDir);

            string fname = UsnTargetedRecovery::WideToNarrow(bestItem.usnInfo.FileName);
            cout << "\n正在通过 MFT 数据直接恢复: " << fname << " ..." << endl;

            UsnTargetedRecoveryResult recResult = recovery.Recover(bestItem.usnInfo, wOutputDir);

            if (recResult.status == UsnRecoveryStatus::SUCCESS ||
                recResult.status == UsnRecoveryStatus::RESIDENT_DATA ||
                recResult.status == UsnRecoveryStatus::PARTIAL_RECOVERY) {
                string path = UsnTargetedRecovery::WideToNarrow(recResult.recoveredPath);
                if (recResult.usedClusterFiltering && recResult.clusterHealth.overwrittenClusters > 0) {
                    cout << "\n=== 部分恢复 ===" << endl;
                } else {
                    cout << "\n=== 恢复成功 ===" << endl;
                }
                cout << "文件大小: " << recResult.recoveredSize << " bytes" << endl;
                cout << "已保存到: " << path << endl;
                if (recResult.usedClusterFiltering && recResult.clusterHealth.overwrittenClusters > 0) {
                    cout << "簇健康: " << recResult.clusterHealth.goodClusters << "/"
                         << recResult.clusterHealth.totalClusters << " ("
                         << fixed << setprecision(1) << recResult.clusterHealth.healthPercentage
                         << "%) | 覆写簇: " << recResult.clusterHealth.overwrittenClusters
                         << " | 检测: " << fixed << setprecision(1)
                         << recResult.clusterHealth.detectionTimeMs << "ms" << endl;
                }
                return;
            }

            // MFT 恢复失败，回退到签名扫描
            cout << "\nMFT 直接恢复失败: " << recResult.statusMessage << endl;
            cout << "回退到签名扫描..." << endl;
        }
    } else if (!matchedUsn.empty()) {
        cout << "\n  MFT 数据不可用，将使用签名扫描恢复" << endl;
    }

    // ========== 第3步：签名扫描（回退路径）==========
    cout << "\n[3/4] 签名扫描磁盘..." << endl;
    cout << " 尝试加载签名扫描缓存" << endl;
    FileCarver carver(&reader);
    FileCarverRecovery carveRecovery(&reader, carver.GetSignatures());
    HybridScanConfig hybridconfig;
    vector<CarvedFileInfo> carveResults;
    if (!CarvedResultsCache::HasValidCache(driveLetter)) {
        // 根据文件扩展名确定要扫描的类型
        cout << " 不存在有效缓存，将执行完整扫描 " << endl;
        wstring ext = UsnTargetedRecovery::GetExtension(targetFileName);
        string extNarrow = UsnTargetedRecovery::WideToNarrow(ext);
        transform(extNarrow.begin(), extNarrow.end(), extNarrow.begin(), ::tolower);

        vector<string> scanTypes;
        if (!extNarrow.empty()) {
            // 映射扩展名到签名类型
            if (extNarrow == "docx" || extNarrow == "xlsx" || extNarrow == "pptx") {
                scanTypes.push_back("zip");  // Office 文档是 ZIP 格式
            }
            else {
                scanTypes.push_back(extNarrow);
            }
        }
        else {
            // 没有扩展名，扫描常见类型
            scanTypes = { "zip", "pdf", "jpg", "png" };
        }
        carveResults = carver.ScanHybridMode(scanTypes, hybridconfig, CARVE_SMART, 1000);
        cout << "  找到 " << carveResults.size() << " 个候选文件" << endl;
    }
    else {
        cout << " 使用缓存的签名扫描结果" << endl;
        CarvedResultsCache carvecache;
        if (carvecache.InitFromDrive(driveLetter) &&
            carvecache.LoadAllResults(carveResults, driveLetter)) {
            cout << "  加载到 " << carveResults.size() << " 个候选文件" << endl;
        } else {
            cout << "  缓存加载失败，执行完整扫描..." << endl;
            wstring ext = UsnTargetedRecovery::GetExtension(targetFileName);
            string extNarrow = UsnTargetedRecovery::WideToNarrow(ext);
            transform(extNarrow.begin(), extNarrow.end(), extNarrow.begin(), ::tolower);

            vector<string> scanTypes;
            if (!extNarrow.empty()) {
                if (extNarrow == "docx" || extNarrow == "xlsx" || extNarrow == "pptx") {
                    scanTypes.push_back("zip");
                } else {
                    scanTypes.push_back(extNarrow);
                }
            } else {
                scanTypes = { "zip", "pdf", "jpg", "png" };
            }
			carver.SetSimdEnabled(true);
            carveResults = carver.ScanHybridMode(scanTypes, hybridconfig, CARVE_SMART, 1000);
            cout << "  找到 " << carveResults.size() << " 个候选文件" << endl;
        }
    }

    // ========== 第4步：三角交叉验证 ==========
    cout << "\n[4/4] 执行三角交叉验证 (USN + MFT + 签名)..." << endl;

    TripleValidator validator(&reader, &parser);

    // 尝试使用 MFT 缓存（如果可用）
    cout << "  加载 MFT 缓存..." << endl;
    MFTCache* cache = MFTCacheManager::GetCache(driveLetter, false);
    if (cache && cache->IsValid()) {
        cout << "  使用已缓存的 MFT 数据 (" << cache->GetTotalCount() << " 条记录)" << endl;
        // 使用缓存填充 CarvedFileInfo 的 MFT 信息
        size_t enriched = cache->EnrichCarvedInfoBatch(carveResults);
        cout << "  关联到 MFT 记录: " << enriched << " 个文件" << endl;
    } else {
        // 没有缓存，回退到传统方式构建 LCN 索引
        cout << "  未找到缓存，正在构建 MFT LCN 索引..." << endl;
        cout << "  提示: 使用 'listdeleted " << driveLetter << " cache' 预先构建缓存以加速恢复" << endl;
        validator.BuildLcnIndex(true, false);
    }
    
    // 加载 USN 记录
    validator.LoadUsnDeletedRecords(matchedUsn);

    // 加载签名扫描结果
    validator.LoadCarvedResults(carveResults);

    // 执行批量验证
    vector<TripleValidationResult> validationResults = validator.ValidateCarvedFiles(carveResults, false);

    // 结构体存储匹配结果
    struct MatchResult {
        size_t carveIndex;
        CarvedFileInfo carveInfo;
        TripleValidationResult validation;
        double score;
    };
    vector<MatchResult> matches;

    for (size_t i = 0; i < carveResults.size(); i++) {
        MatchResult match;
        match.carveIndex = i;
        match.carveInfo = carveResults[i];
        match.validation = validationResults[i];
        match.score = validationResults[i].confidence;
        matches.push_back(match);
    }

    // 按置信度排序
    sort(matches.begin(), matches.end(), [](const MatchResult& a, const MatchResult& b) {
        return a.score > b.score;
    });

    // 统计验证结果
    size_t tripleCount = 0, doubleCount = 0, singleCount = 0;
    for (const auto& v : validationResults) {
        if (v.level == VAL_TRIPLE) tripleCount++;
        else if (v.level == VAL_MFT_SIGNATURE || v.level == VAL_USN_SIGNATURE || v.level == VAL_USN_MFT) doubleCount++;
        else if (v.level == VAL_SIGNATURE_ONLY) singleCount++;
    }

    cout << "\n========================================" << endl;
    cout << "验证结果统计" << endl;
    cout << "========================================" << endl;
    cout << "  三角验证通过: " << tripleCount << endl;
    cout << "  双重验证通过: " << doubleCount << endl;
    cout << "  仅签名验证:   " << singleCount << endl;

    cout << "\n========================================" << endl;
    cout << "搜索结果" << endl;
    cout << "========================================\n" << endl;

    // 显示 USN 记录
    if (!matchedUsn.empty()) {
        cout << "USN 删除记录:" << endl;
        cout << string(60, '-') << endl;
        for (size_t i = 0; i < min(matchedUsn.size(), (size_t)5); i++) {
            const auto& usn = matchedUsn[i];
            string fname = UsnTargetedRecovery::WideToNarrow(usn.FileName);
            cout << "  " << fname;
            if (usn.MftInfoValid) {
                string sizeStr = UsnTargetedRecovery::WideToNarrow(
                    UsnTargetedRecovery::FormatFileSize(usn.FileSize));
                cout << "  [" << sizeStr << "]";
                if (usn.MftRecordReused) {
                    cout << " (MFT已复用)";
                }
            } else {
                cout << "  [大小未知]";
            }
            cout << endl;
        }
        cout << endl;
    }

    // 显示候选文件
    if (matches.empty()) {
        cout << "未找到可恢复的文件" << endl;
        return;
    }

    cout << "候选文件 (按置信度排序):" << endl;
    cout << string(75, '-') << endl;
    cout << left << setw(6) << "编号"
         << setw(12) << "大小"
         << setw(10) << "置信度"
         << setw(8) << "类型"
         << setw(20) << "验证级别"
         << "状态" << endl;
    cout << string(75, '-') << endl;

    size_t displayCount = min(matches.size(), (size_t)10);
    for (size_t i = 0; i < displayCount; i++) {
        const auto& m = matches[i];
        string sizeStr = UsnTargetedRecovery::WideToNarrow(
            UsnTargetedRecovery::FormatFileSize(m.carveInfo.fileSize));

        cout << left << setw(6) << i
             << setw(12) << sizeStr
             << setw(10) << fixed << setprecision(0) << (m.score * 100) << "%"
             << setw(8) << m.carveInfo.extension
             << setw(20) << TripleValidator::ValidationLevelToString(m.validation.level);

        if (m.validation.level == VAL_TRIPLE) {
            cout << "*** 最佳";
        } else if (m.validation.sequenceValid) {
            cout << "MFT有效";
        } else if (m.validation.signatureValid) {
            cout << "签名有效";
        } else {
            cout << "-";
        }
        cout << endl;
    }

    if (matches.size() > displayCount) {
        cout << "  ... 还有 " << (matches.size() - displayCount) << " 个结果" << endl;
    }

    cout << "\n========================================" << endl;

    // ========== 第5步：用户选择 ==========
    if (outputDir.empty()) {
        string input;
        TuiInputBridge::Instance().GetLine("输入编号恢复文件，或输入 'q' 退出: ", input);

        if (input.empty() || input == "q" || input == "Q") {
            cout << "已取消" << endl;
            return;
        }

        size_t selectedIndex;
        try {
            selectedIndex = stoul(input);
        } catch (...) {
            cout << "无效的输入" << endl;
            return;
        }

        if (selectedIndex >= matches.size()) {
            cout << "编号超出范围" << endl;
            return;
        }

        // 询问输出目录
        TuiInputBridge::Instance().GetLine("输入输出目录 (直接回车使用当前目录): ", outputDir);
        if (outputDir.empty()) {
            outputDir = ".";
        }

        // 恢复文件
        auto selected = matches[selectedIndex];  // 复制，精细化需要修改
        string outputFileName = targetFileName.empty() ?
            ("recovered_" + to_string(selectedIndex) + "." + selected.carveInfo.extension) :
            UsnTargetedRecovery::WideToNarrow(targetFileName);

        // 恢复前精细化：精确大小计算 + 完整性验证
        cout << "\n[精细化] 正在对候选文件进行恢复前分析..." << endl;
        bool isHealthy = carveRecovery.RefineCarvedFileInfo(selected.carveInfo);

        if (!isHealthy) {
            string confirm;
            TuiInputBridge::Instance().GetLine("\n警告: 文件可能已损坏，是否仍然恢复? (y/n): ", confirm);
            if (confirm != "y" && confirm != "Y") {
                cout << "已取消" << endl;
                return;
            }
        }

        // 精细化后文件名可能需要更新（如 zip -> docx）
        if (targetFileName.empty()) {
            outputFileName = "recovered_" + to_string(selectedIndex) + "." + selected.carveInfo.extension;
        }

        string outputPath = outputDir + "\\" + outputFileName;

        cout << "\n正在恢复到: " << outputPath << " ..." << endl;

        // ZIP/OOXML 使用智能恢复（EOCD扫描 + CRC校验）
        bool isZipType = (selected.carveInfo.extension == "zip" || selected.carveInfo.extension == "docx" ||
                          selected.carveInfo.extension == "xlsx" || selected.carveInfo.extension == "pptx" ||
                          selected.carveInfo.extension == "ooxml");

        bool recovered = false;
        if (isZipType) {
            FileCarverRecovery::ZipRecoveryConfig config;
            config.verifyCRC = true;
            config.stopOnFirstEOCD = true;
            if (selected.carveInfo.fileSize > 0) {
                config.expectedSize = selected.carveInfo.fileSize;
                config.expectedSizeTolerance = selected.carveInfo.fileSize / 5;
            }
            auto result = carveRecovery.RecoverZipWithEOCDScan(selected.carveInfo.startLCN, outputPath, config);
            if (result.success) {
                recovered = true;
                cout << "恢复成功!" << endl;
                cout << "文件大小: " << result.actualSize << " bytes" << endl;
                cout << "CRC校验: " << (result.crcValid ? "通过" : "警告") << endl;
            } else {
                // 回退到普通恢复
                ClusterHealthReport carveHealth;
                recovered = carveRecovery.RecoverCarvedFile(selected.carveInfo, outputPath, &carveHealth);
                if (recovered) {
                    cout << "恢复成功 (无EOCD，使用估算大小)" << endl;
                    if (carveHealth.overwrittenClusters > 0) {
                        cout << "簇健康: " << fixed << setprecision(1)
                             << carveHealth.healthPercentage << "%" << endl;
                    }
                }
            }
        } else {
            ClusterHealthReport carveHealth;
            recovered = carveRecovery.RecoverCarvedFile(selected.carveInfo, outputPath, &carveHealth);
            if (recovered) {
                cout << "恢复成功!" << endl;
                cout << "文件大小: " << selected.carveInfo.fileSize << " bytes" << endl;
                if (carveHealth.overwrittenClusters > 0) {
                    cout << "簇健康: " << fixed << setprecision(1)
                         << carveHealth.healthPercentage << "%" << endl;
                }
            }
        }

        if (!recovered) {
            cout << "恢复失败" << endl;
        }
    } else {
        // 直接恢复第一个（最高置信度）
        auto best = matches[0];  // 复制，精细化需要修改
        string outputFileName = UsnTargetedRecovery::WideToNarrow(targetFileName);
        if (outputFileName.empty()) {
            outputFileName = "recovered." + best.carveInfo.extension;
        }

        // 恢复前精细化：精确大小计算 + 完整性验证
        cout << "\n[精细化] 正在对候选文件进行恢复前分析..." << endl;
        bool isHealthy = carveRecovery.RefineCarvedFileInfo(best.carveInfo);

        if (!isHealthy) {
            cout << "警告: 文件可能已损坏，仍尝试恢复..." << endl;
        }

        // 精细化后文件名可能需要更新（如 zip -> docx）
        if (UsnTargetedRecovery::WideToNarrow(targetFileName).empty()) {
            outputFileName = "recovered." + best.carveInfo.extension;
        }

        string outputPath = outputDir + "\\" + outputFileName;

        cout << "\n正在恢复最佳匹配到: " << outputPath << " ..." << endl;

        // ZIP/OOXML 使用智能恢复（EOCD扫描 + CRC校验）
        bool isZipType = (best.carveInfo.extension == "zip" || best.carveInfo.extension == "docx" ||
                          best.carveInfo.extension == "xlsx" || best.carveInfo.extension == "pptx" ||
                          best.carveInfo.extension == "ooxml");

        bool recovered = false;
        if (isZipType) {
            FileCarverRecovery::ZipRecoveryConfig config;
            config.verifyCRC = true;
            config.stopOnFirstEOCD = true;
            if (best.carveInfo.fileSize > 0) {
                config.expectedSize = best.carveInfo.fileSize;
                config.expectedSizeTolerance = best.carveInfo.fileSize / 5;
            }
            auto result = carveRecovery.RecoverZipWithEOCDScan(best.carveInfo.startLCN, outputPath, config);
            if (result.success) {
                recovered = true;
                cout << "恢复成功!" << endl;
                cout << "文件大小: " << result.actualSize << " bytes" << endl;
                cout << "CRC校验: " << (result.crcValid ? "通过" : "警告") << endl;
            } else {
                ClusterHealthReport carveHealth;
                recovered = carveRecovery.RecoverCarvedFile(best.carveInfo, outputPath, &carveHealth);
                if (recovered) {
                    cout << "恢复成功 (无EOCD，使用估算大小)" << endl;
                    if (carveHealth.overwrittenClusters > 0) {
                        cout << "簇健康: " << fixed << setprecision(1)
                             << carveHealth.healthPercentage << "%" << endl;
                    }
                }
            }
        } else {
            ClusterHealthReport carveHealth;
            recovered = carveRecovery.RecoverCarvedFile(best.carveInfo, outputPath, &carveHealth);
            if (recovered) {
                cout << "恢复成功!" << endl;
                cout << "文件大小: " << best.carveInfo.fileSize << " bytes" << endl;
                if (carveHealth.overwrittenClusters > 0) {
                    cout << "簇健康: " << fixed << setprecision(1)
                         << carveHealth.healthPercentage << "%" << endl;
                }
            }
        }

        if (!recovered) {
            cout << "恢复失败" << endl;
        }
    }
}

// ============================================================================
// SnapshotCommand - 立即扫描 USN 删除记录并捕获 MFT 快照
// ============================================================================
// 用法: snapshot <drive> [hours]
// 示例: snapshot D
//       snapshot D 72
// ============================================================================
DEFINE_COMMAND_BASE(SnapshotCommand, "snapshot |name |name", TRUE)
REGISTER_COMMAND(SnapshotCommand);

void SnapshotCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (GET_ARG_COUNT() < 1) {
        cout << "\n=== MFT 快照捕获 ===" << endl;
        cout << "用法: snapshot <drive> [hours]" << endl;
        cout << "功能: 扫描 USN 删除记录，为每个被删文件捕获 MFT 元数据快照" << endl;
        cout << "\n示例:" << endl;
        cout << "  snapshot D          # 捕获最近 24 小时内删除文件的快照" << endl;
        cout << "  snapshot D 72       # 捕获最近 72 小时内删除文件的快照" << endl;
        cout << "\n说明:" << endl;
        cout << "  快照保存了被删文件的 Data Run (LCN 映射) 信息。" << endl;
        cout << "  即使 MFT 记录被复用，也可以通过快照直接定位文件数据。" << endl;
        cout << "  建议在发现误删后立即运行此命令。" << endl;
        return;
    }

    string driveStr = GET_ARG_STRING(0);
    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << "错误: 无效的驱动器字母" << endl;
        return;
    }

    int maxHours = 24;
    if (GET_ARG_COUNT() >= 2) {
        try {
            maxHours = stoi(GET_ARG_STRING(1));
        } catch (...) {
            cout << "警告: 无效的小时数，使用默认值 24" << endl;
        }
    }

    cout << "\n=== MFT 快照捕获 ===" << endl;
    cout << "驱动器: " << driveLetter << ":/" << endl;
    cout << "回溯时间: " << maxHours << " 小时" << endl;

    // 加载已有快照
    MFTSnapshotStore store;
    store.SetDriveLetter(driveLetter);
    string storePath = MFTSnapshotStore::GenerateStorePath(driveLetter);
    store.LoadFromFile(storePath);
    size_t existingCount = store.GetCount();

    // 使用 UsnDeleteMonitor 的一次性扫描功能
    UsnDeleteMonitor monitor(driveLetter);

    // 将已有快照传递给 monitor（通过直接操作 store）
    // 注意: monitor 内部有自己的 store，这里单独使用 CaptureExistingDeleted
    cout << "\n正在扫描 USN 日志..." << endl;
    size_t captured = monitor.CaptureExistingDeleted(maxHours);

    // 合并到持久化存储
    // 从 monitor 的 store 获取所有新捕获的快照
    auto& monitorStore = monitor.GetSnapshotStore();

    // 保存 monitor 的快照（直接保存即可，因为是独立 store）
    monitorStore.SaveToFile(storePath);

    cout << "\n=== 快照捕获完成 ===" << endl;
    cout << "新捕获: " << captured << " 个快照" << endl;
    cout << "总计: " << monitorStore.GetCount() << " 个快照" << endl;
    cout << "存储位置: " << storePath << endl;

    auto& stats = monitor.GetStats();
    if (stats.missedCount > 0) {
        cout << "未能捕获: " << stats.missedCount.load() << " 个 (MFT 记录无法解析)" << endl;
    }
    if (stats.skippedCount > 0) {
        cout << "已跳过: " << stats.skippedCount.load() << " 个 (目录/系统文件)" << endl;
    }
}

// ============================================================================
// MonitorCommand - 启动/停止后台 USN 删除监控
// ============================================================================
// 用法: monitor <drive> [start|stop|status]
// ============================================================================
DEFINE_COMMAND_BASE(MonitorCommand, "monitor |name |name", TRUE)
REGISTER_COMMAND(MonitorCommand);

void MonitorCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (GET_ARG_COUNT() < 1) {
        cout << "\n=== USN 删除监控守护进程 ===" << endl;
        cout << "用法: monitor <drive> [start|stop|status|autostart|unautostart]" << endl;
        cout << "功能: 启动独立后台守护进程，实时监控文件删除并自动捕获 MFT 快照" << endl;
        cout << "\n示例:" << endl;
        cout << "  monitor D start         # 启动 D 盘监控守护进程" << endl;
        cout << "  monitor D stop          # 停止守护进程" << endl;
        cout << "  monitor D status        # 查看守护进程状态" << endl;
        cout << "  monitor D autostart     # 启用开机自启动" << endl;
        cout << "  monitor D unautostart   # 禁用开机自启动" << endl;
        return;
    }

    string driveStr = GET_ARG_STRING(0);
    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << "错误: 无效的驱动器字母" << endl;
        return;
    }

    string action = "start";
    if (GET_ARG_COUNT() >= 2) {
        action = GET_ARG_STRING(1);
        transform(action.begin(), action.end(), action.begin(), ::tolower);
    }

    MonitorDaemon daemon;

    if (action == "start") {
        if (daemon.IsDaemonRunning(driveLetter)) {
            DWORD pid = daemon.GetDaemonPID(driveLetter);
            cout << "监控守护进程已在运行中 (PID: " << pid << ")" << endl;
            return;
        }

        cout << "正在启动监控守护进程..." << endl;
        if (daemon.StartDaemon(driveLetter)) {
            DWORD pid = daemon.GetDaemonPID(driveLetter);
            cout << "USN 删除监控守护进程已启动: 驱动器 " << driveLetter << ":/" << endl;
            cout << "PID: " << pid << endl;
            cout << "守护进程在后台独立运行，CLI 退出后仍会继续监控。" << endl;
            cout << "使用 'monitor " << driveLetter << " status' 查看状态，'monitor " << driveLetter << " stop' 停止。" << endl;
        } else {
            cout << "错误: 无法启动监控守护进程" << endl;
        }
    }
    else if (action == "stop") {
        if (!daemon.IsDaemonRunning(driveLetter)) {
            cout << "监控守护进程未在运行" << endl;
            return;
        }

        // 先读取统计信息
        MonitorSharedState state = {};
        if (daemon.AttachSharedMemory(driveLetter)) {
            daemon.ReadState(state);
            daemon.DetachSharedMemory();
        }

        cout << "正在停止监控守护进程..." << endl;
        if (daemon.StopDaemon(driveLetter)) {
            cout << "USN 删除监控守护进程已停止" << endl;
            cout << "  检测到的删除事件: " << state.totalEvents << endl;
            cout << "  成功捕获快照: " << state.capturedCount << endl;
            cout << "  未能捕获: " << state.missedCount << endl;
            cout << "  快照总数: " << state.snapshotCount << endl;
        } else {
            cout << "错误: 无法停止守护进程（可能需要手动结束进程）" << endl;
        }
    }
    else if (action == "status") {
        if (!daemon.IsDaemonRunning(driveLetter)) {
            cout << "监控守护进程未在运行" << endl;

            // 检查是否有已保存的快照
            string path = MFTSnapshotStore::GenerateStorePath(driveLetter);
            MFTSnapshotStore store;
            if (store.LoadFromFile(path)) {
                cout << "已保存的快照: " << store.GetCount() << " 个" << endl;
            }

            cout << "自启动: " << (MonitorDaemon::IsAutoStartInstalled(driveLetter) ? "已启用" : "未启用") << endl;
            return;
        }

        if (!daemon.AttachSharedMemory(driveLetter)) {
            cout << "错误: 无法读取守护进程状态" << endl;
            return;
        }

        MonitorSharedState state;
        if (!daemon.ReadState(state)) {
            cout << "错误: 共享内存数据无效" << endl;
            daemon.DetachSharedMemory();
            return;
        }

        // 格式化启动时间
        SYSTEMTIME st;
        FileTimeToSystemTime(&state.startTime, &st);

        cout << "=== 监控守护进程状态 ===" << endl;
        cout << "驱动器: " << state.driveLetter << ":/" << endl;
        cout << "状态: 运行中" << endl;
        cout << "PID: " << state.pid << endl;
        cout << "启动时间: " << st.wYear << "-"
             << setfill('0') << setw(2) << st.wMonth << "-"
             << setw(2) << st.wDay << " "
             << setw(2) << st.wHour << ":"
             << setw(2) << st.wMinute << ":"
             << setw(2) << st.wSecond << setfill(' ') << endl;
        cout << "轮询间隔: " << state.pollIntervalMs << "ms" << endl;
        cout << "自启动: " << (state.autoStartEnabled ? "已启用" : "未启用") << endl;
        cout << "========================================" << endl;
        cout << "删除事件: " << state.totalEvents << endl;
        cout << "捕获快照: " << state.capturedCount << endl;
        cout << "未能捕获: " << state.missedCount << endl;
        cout << "已跳过: " << state.skippedCount << endl;
        cout << "快照总数: " << state.snapshotCount << endl;

        // 显示最近事件
        LONG eventCount = state.recentEventCount;
        if (eventCount > 0) {
            cout << "========================================" << endl;
            cout << "最近事件:" << endl;
            int displayCount = min((int)eventCount, MONITOR_RECENT_EVENT_MAX);
            LONG head = state.recentEventHead;
            for (int i = displayCount - 1; i >= 0; i--) {
                LONG idx = (head - 1 - i + MONITOR_RECENT_EVENT_MAX * 100) % MONITOR_RECENT_EVENT_MAX;
                auto& evt = state.recentEvents[idx];
                // wide-to-narrow for console output
                string narrowName;
                for (size_t c = 0; c < wcslen(evt.fileName); c++) {
                    wchar_t wc = evt.fileName[c];
                    if (wc < 128) narrowName += (char)wc;
                    else narrowName += '?';
                }
                cout << "  " << (evt.captured ? "[OK]" : "[MISS]")
                     << " MFT#" << evt.mftRecord
                     << " " << narrowName << endl;
            }
        }

        daemon.DetachSharedMemory();
    }
    else if (action == "autostart") {
        if (MonitorDaemon::InstallAutoStart(driveLetter)) {
            cout << "已启用开机自启动: 驱动器 " << driveLetter << ":/" << endl;
        } else {
            cout << "错误: 无法设置自启动（请检查权限）" << endl;
        }
    }
    else if (action == "unautostart") {
        if (MonitorDaemon::UninstallAutoStart(driveLetter)) {
            cout << "已禁用开机自启动: 驱动器 " << driveLetter << ":/" << endl;
        } else {
            cout << "错误: 无法移除自启动设置" << endl;
        }
    }
    else {
        cout << "错误: 未知操作 '" << action << "'，可用: start, stop, status, autostart, unautostart" << endl;
    }
}

// ============================================================================
// SnapshotQueryCommand - 查询快照存储中的文件
// ============================================================================
// 用法: snapshotquery <drive> [pattern]
// ============================================================================
DEFINE_COMMAND_BASE(SnapshotQueryCommand, "snapshotquery |name |name", TRUE)
REGISTER_COMMAND(SnapshotQueryCommand);

void SnapshotQueryCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    if (GET_ARG_COUNT() < 1) {
        cout << "\n=== 快照查询 ===" << endl;
        cout << "用法: snapshotquery <drive> [pattern]" << endl;
        cout << "功能: 查询快照存储中的文件" << endl;
        cout << "\n示例:" << endl;
        cout << "  snapshotquery D              # 列出所有快照" << endl;
        cout << "  snapshotquery D .cpp          # 搜索 .cpp 文件" << endl;
        cout << "  snapshotquery D document      # 搜索含 'document' 的文件" << endl;
        return;
    }

    string driveStr = GET_ARG_STRING(0);
    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << "错误: 无效的驱动器字母" << endl;
        return;
    }

    // 加载快照
    MFTSnapshotStore store;
    string storePath = MFTSnapshotStore::GenerateStorePath(driveLetter);
    if (!store.LoadFromFile(storePath)) {
        cout << "未找到快照存储文件。" << endl;
        cout << "提示: 使用 'snapshot " << driveLetter << "' 先捕获快照。" << endl;
        return;
    }

    cout << "\n=== 快照查询 ===" << endl;
    cout << "快照总数: " << store.GetCount() << endl;

    // 查询
    wstring pattern = L"";
    if (GET_ARG_COUNT() >= 2) {
        pattern = UsnTargetedRecovery::NarrowToWide(GET_ARG_STRING(1));
    }

    vector<const MFTSnapshot*> results;
    if (pattern.empty()) {
        // 列出所有（限制 100 条）
        results = store.SearchByName(L"");
    } else {
        results = store.SearchByName(pattern);
    }

    if (results.empty()) {
        cout << "未找到匹配的快照" << endl;
        return;
    }

    // 显示结果
    size_t displayCount = min(results.size(), (size_t)100);
    cout << "\n找到 " << results.size() << " 个快照";
    if (results.size() > 100) {
        cout << " (显示前 100 条)";
    }
    cout << "\n" << endl;

    cout << left << setw(8) << "MFT#"
         << setw(6) << "Seq"
         << setw(12) << "大小"
         << setw(8) << "Runs"
         << setw(10) << "类型"
         << "文件名" << endl;
    cout << string(70, '-') << endl;

    for (size_t i = 0; i < displayCount; i++) {
        const MFTSnapshot* snap = results[i];

        // 格式化大小
        string sizeStr;
        if (snap->fileSize < 1024) {
            sizeStr = to_string(snap->fileSize) + " B";
        } else if (snap->fileSize < 1024 * 1024) {
            sizeStr = to_string(snap->fileSize / 1024) + " KB";
        } else {
            sizeStr = to_string(snap->fileSize / (1024 * 1024)) + " MB";
        }

        string typeStr = snap->isResident ? "常驻" : "非常驻";

        cout << left << setw(8) << snap->recordNumber
             << setw(6) << snap->sequenceNumber
             << setw(12) << sizeStr
             << setw(8) << snap->dataRuns.size()
             << setw(10) << typeStr
             << UsnTargetedRecovery::WideToNarrow(snap->fileName) << endl;
    }
}

