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
#include "components/TuiInputBridge.h"

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

    // 加载 MFT data runs（支持碎片化 MFT 的正确记录定位）
    reader.GetTotalMFTRecords();

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

// ============================================================================
// RecoverCommand - 智能文件恢复向导（USN + 签名扫描联合）
// ============================================================================
// 用法: recover <drive> [filename] [output_dir]
// 示例: recover C
//       recover C document.docx
//       recover C document.docx D:\recovered
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
        cout << "\n示例:" << endl;
        cout << "  recover C                          # 交互式搜索" << endl;
        cout << "  recover C document.docx            # 搜索指定文件" << endl;
        cout << "  recover C document.docx D:\\out     # 直接恢复到指定目录" << endl;
        return;
    }

    // 解析驱动器
    string& driveStr = GET_ARG_STRING(0);
    char driveLetter;
    if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
        cout << "错误: 无效的驱动器字母" << endl;
        return;
    }

    // 解析文件名（可选）
    wstring targetFileName = L"";
    if (GET_ARG_COUNT() >= 2) {
        string& targetStr = GET_ARG_STRING(1);
        targetFileName = UsnTargetedRecovery::NarrowToWide(targetStr);
    }

    // 解析输出目录（可选）
    string outputDir = "";
    if (GET_ARG_COUNT() >= 3) {
        outputDir = GET_ARG_STRING(2);
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
                    cout << "\n=== 恢复成功 ===" << endl;
                    cout << "文件大小: " << recResult.recoveredSize << " bytes" << endl;
                    cout << "已保存到: " << path << endl;
                    if (recResult.signatureMatched) {
                        cout << "签名验证: 通过 (" << recResult.detectedType << ")" << endl;
                    }
                } else {
                    cout << "\nMFT 直接恢复失败: " << recResult.statusMessage << endl;
                    cout << "提示: 重新运行并输入 's' 可进行全盘签名扫描" << endl;
                }
                return;
            }
            // input == "s" → 用户选择全盘扫描，继续执行签名扫描
            cout << "\n用户选择全盘签名扫描..." << endl;
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
                cout << "\n=== 恢复成功 ===" << endl;
                cout << "文件大小: " << recResult.recoveredSize << " bytes" << endl;
                cout << "已保存到: " << path << endl;
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

    // 根据文件扩展名确定要扫描的类型
    wstring ext = UsnTargetedRecovery::GetExtension(targetFileName);
    string extNarrow = UsnTargetedRecovery::WideToNarrow(ext);
    transform(extNarrow.begin(), extNarrow.end(), extNarrow.begin(), ::tolower);

    vector<string> scanTypes;
    if (!extNarrow.empty()) {
        // 映射扩展名到签名类型
        if (extNarrow == "docx" || extNarrow == "xlsx" || extNarrow == "pptx") {
            scanTypes.push_back("zip");  // Office 文档是 ZIP 格式
        } else {
            scanTypes.push_back(extNarrow);
        }
    } else {
        // 没有扩展名，扫描常见类型
        scanTypes = {"zip", "pdf", "jpg", "png"};
    }

    FileCarver carver(&reader);
    FileCarverRecovery carveRecovery(&reader, carver.GetSignatures());
    vector<CarvedFileInfo> carveResults = carver.ScanForFileTypes(scanTypes, CARVE_SMART, 200);

    cout << "  找到 " << carveResults.size() << " 个候选文件" << endl;

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
                recovered = carveRecovery.RecoverCarvedFile(selected.carveInfo, outputPath);
                if (recovered) {
                    cout << "恢复成功 (无EOCD，使用估算大小)" << endl;
                }
            }
        } else {
            recovered = carveRecovery.RecoverCarvedFile(selected.carveInfo, outputPath);
            if (recovered) {
                cout << "恢复成功!" << endl;
                cout << "文件大小: " << selected.carveInfo.fileSize << " bytes" << endl;
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
                recovered = carveRecovery.RecoverCarvedFile(best.carveInfo, outputPath);
                if (recovered) {
                    cout << "恢复成功 (无EOCD，使用估算大小)" << endl;
                }
            }
        } else {
            recovered = carveRecovery.RecoverCarvedFile(best.carveInfo, outputPath);
            if (recovered) {
                cout << "恢复成功!" << endl;
                cout << "文件大小: " << best.carveInfo.fileSize << " bytes" << endl;
            }
        }

        if (!recovered) {
            cout << "恢复失败" << endl;
        }
    }
}
