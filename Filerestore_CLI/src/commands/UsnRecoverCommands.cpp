// UsnRecoverCommands.cpp - USN 定点恢复命令实现
// 包含: UsnListCommand, UsnRecoverCommand

#include "cmd.h"
#include "CommandUtils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include "MFTReader.h"
#include "MFTParser.h"
#include "UsnTargetedRecovery.h"
#include "LocalizationManager.h"

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
        cout << "  usnlist <drive> [hours] [--validate] [--pattern=<name>]" << endl;
        cout << LOC_STR("usnlist.examples") << endl;
        cout << "  usnlist C 24" << endl;
        cout << "  usnlist C 48 --validate" << endl;
        cout << "  usnlist C 24 --pattern=document" << endl;
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
        driveLetter, maxHours, pattern, 1000);

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
        cout << setw(12) << "-";  // USN 不包含文件大小
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

    if (GET_ARG_COUNT() < 3) {
        cout << LOC_STR("usnrecover.usage") << endl;
        cout << "  usnrecover <drive> <index|filename|record> <output_dir> [--force]" << endl;
        cout << LOC_STR("usnrecover.examples") << endl;
        cout << "  usnrecover C 0 D:\\recovered\\" << endl;
        cout << "  usnrecover C document.docx D:\\recovered\\" << endl;
        cout << "  usnrecover C 0x12345 D:\\recovered\\ --force" << endl;
        return;
    }

    // 解析参数
    string& driveStr = GET_ARG_STRING(0);
    string& targetStr = GET_ARG_STRING(1);
    string& outputStr = GET_ARG_STRING(2);

    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << LOC_STR("error.invalid_drive") << endl;
        return;
    }

    bool forceRecover = false;
    if (GET_ARG_COUNT() >= 4) {
        string& forceStr = GET_ARG_STRING(3);
        if (forceStr == "--force" || forceStr == "-f") {
            forceRecover = true;
        }
    }

    wstring outputDir = UsnTargetedRecovery::NarrowToWide(outputStr);

    // 初始化组件
    MFTReader reader;
    if (!reader.OpenVolume(driveLetter)) {
        cout << LOC_STR("error.open_volume_failed") << ": " << driveLetter << ":/" << endl;
        return;
    }

    MFTParser parser(&reader);
    UsnTargetedRecovery recovery(&reader, &parser);

    cout << "\n" << LOC_STR("usnrecover.title") << endl;
    cout << "========================================" << endl;

    UsnTargetedRecoveryResult result;

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
