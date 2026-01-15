/**
 * RepairCommands.cpp
 * 文件修复 CLI 命令实现
 *
 * 命令:
 * - repair: 修复单个损坏的文件
 * - repair-batch: 批量修复目录中的损坏文件
 */

#include "cmd.h"
#include "CommandMacros.h"
#include "CommandUtils.h"
#include "../fileRestore/MLFileRepair.h"
#include "../fileRestore/ImageHeaderRepairer.h"
#include "../utils/LocalizationManager.h"
#include "../utils/Logger.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <set>

namespace fs = std::filesystem;

using namespace std;

// ============================================================================
// 辅助函数
// ============================================================================

namespace {

// 读取文件到字节数组
bool ReadFileToBytes(const string& path, vector<BYTE>& data) {
    ifstream file(path, ios::binary | ios::ate);
    if (!file) return false;

    size_t size = (size_t)file.tellg();
    file.seekg(0, ios::beg);

    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    return file.good();
}

// 写入字节数组到文件
bool WriteBytesToFile(const string& path, const vector<BYTE>& data) {
    // 确保目录存在
    fs::path filePath(path);
    if (filePath.has_parent_path()) {
        fs::create_directories(filePath.parent_path());
    }

    ofstream file(path, ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<const char*>(data.data()), data.size());

    return file.good();
}

// 从文件扩展名推断类型
string InferTypeFromExtension(const string& filename) {
    fs::path path(filename);
    string ext = path.extension().string();

    if (ext.empty()) return "";

    // 移除点号并转小写
    ext = ext.substr(1);
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "jpg") return "jpeg";
    return ext;
}

// 格式化文件大小
string FormatSize(size_t bytes) {
    if (bytes < 1024) return to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return to_string(bytes / 1024) + " KB";
    return to_string(bytes / (1024 * 1024)) + " MB";
}

// 获取损坏类型描述
string GetDamageTypeString(DamageType type) {
    switch (type) {
        case DamageType::NONE: return "无损坏";
        case DamageType::HEADER_CORRUPTED: return "头部损坏";
        case DamageType::STRUCTURE_CORRUPTED: return "结构损坏";
        case DamageType::CONTENT_CORRUPTED: return "内容损坏";
        default: return "未知";
    }
}

// 获取修复结果描述
string GetRepairResultString(RepairResult result) {
    switch (result) {
        case RepairResult::SUCCESS: return "成功";
        case RepairResult::PARTIAL: return "部分成功";
        case RepairResult::FAILED: return "失败";
        case RepairResult::NOT_APPLICABLE: return "不适用";
        default: return "未知";
    }
}

} // namespace

// ============================================================================
// repair 命令 - 修复单个文件
// ============================================================================
// 用法: repair <input_file> [output_file] [--type <jpeg|png>] [--analyze] [--no-ml] [--force]
// 示例: repair corrupted.jpg
//       repair corrupted.jpg fixed.jpg --type jpeg
//       repair image.png --analyze
// ============================================================================
DEFINE_COMMAND_BASE(RepairCommand, "repair |file |file |name |name |name |name", TRUE)
REGISTER_COMMAND(RepairCommand);

void RepairCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    auto& loc = LocalizationManager::Instance();

    // 解析参数
    string inputPath;
    string outputPath;
    string fileType;
    bool useML = true;
    bool analyze = false;
    bool force = false;

    for (int i = 0; i < GET_ARG_COUNT(); i++) {
        string& arg = GET_ARG_STRING(i);

        if (arg == "--type" && i + 1 < GET_ARG_COUNT()) {
            fileType = GET_ARG_STRING(++i);
        }
        else if (arg == "--no-ml") {
            useML = false;
        }
        else if (arg == "--analyze" || arg == "-a") {
            analyze = true;
        }
        else if (arg == "--force" || arg == "-f") {
            force = true;
        }
        else if (inputPath.empty()) {
            inputPath = arg;
        }
        else if (outputPath.empty()) {
            outputPath = arg;
        }
    }

    // 检查参数
    if (inputPath.empty()) {
        cout << LOC_STR("repair.usage") << endl;
        cout << "  repair <input_file> [output_file] [--type <jpeg|png>] [--analyze] [--no-ml] [--force]" << endl;
        cout << endl;
        cout << LOC_STR("repair.options") << ":" << endl;
        cout << "  --type <type>   " << LOC_STR("repair.option_type") << endl;
        cout << "  --analyze, -a   " << LOC_STR("repair.option_analyze") << endl;
        cout << "  --no-ml         " << LOC_STR("repair.option_no_ml") << endl;
        cout << "  --force, -f     " << LOC_STR("repair.option_force") << endl;
        return;
    }

    // 检查输入文件是否存在
    if (!fs::exists(inputPath)) {
        cout << LOC_STR("error.file_not_found") << ": " << inputPath << endl;
        return;
    }

    // 如果未指定输出路径，生成默认路径
    if (outputPath.empty()) {
        fs::path inPath(inputPath);
        string stem = inPath.stem().string();
        string ext = inPath.extension().string();
        fs::path parent = inPath.parent_path();
        outputPath = (parent / (stem + "_repaired" + ext)).string();
    }

    // 读取输入文件
    vector<BYTE> fileData;
    if (!ReadFileToBytes(inputPath, fileData)) {
        cout << LOC_STR("error.read_failed") << ": " << inputPath << endl;
        return;
    }

    cout << LOC_STR("repair.input_file") << ": " << inputPath << endl;
    cout << LOC_STR("repair.file_size") << ": " << FormatSize(fileData.size()) << endl;

    // 推断文件类型
    if (fileType.empty()) {
        fileType = InferTypeFromExtension(inputPath);
    }
    if (!fileType.empty()) {
        cout << LOC_STR("repair.file_type") << ": " << fileType << endl;
    }

    // 创建修复器
    ImageHeaderRepairer repairer;

    // 分析模式
    if (analyze) {
        cout << endl << "=== " << LOC_STR("repair.damage_analysis") << " ===" << endl;

        // 提取特征
        ImageFeatureVector features = repairer.ExtractFullFeatures(fileData);

        // 显示特征
        cout << LOC_STR("repair.feature_mean") << ": " << fixed << setprecision(3) << features.mean << endl;
        cout << LOC_STR("repair.feature_stddev") << ": " << features.stddev << endl;
        cout << LOC_STR("repair.feature_entropy") << ": " << features.entropy << endl;
        cout << LOC_STR("repair.feature_zero_ratio") << ": " << features.zeroRatio << endl;
        cout << LOC_STR("repair.feature_ff_ratio") << ": " << features.ffRatio << endl;

        // JPEG 特征
        if (ImageHeaderRepairer::IsLikelyJPEG(fileData) || fileType == "jpeg" || fileType == "jpg") {
            cout << endl << "JPEG " << LOC_STR("repair.features") << ":" << endl;
            cout << "  FF " << LOC_STR("repair.marker_density") << ": " << features.ffMarkerDensity << endl;
            cout << "  DQT: " << (features.quantTablePresence > 0.5f ? LOC_STR("common.yes") : LOC_STR("common.no")) << endl;
            cout << "  DHT: " << (features.huffmanTablePresence > 0.5f ? LOC_STR("common.yes") : LOC_STR("common.no")) << endl;
        }

        // PNG 特征
        if (ImageHeaderRepairer::IsLikelyPNG(fileData) || fileType == "png") {
            cout << endl << "PNG " << LOC_STR("repair.features") << ":" << endl;
            cout << "  DEFLATE: " << features.deflateSignature << endl;
            cout << "  Chunk " << LOC_STR("repair.score") << ": " << features.chunkBoundaryScore << endl;
        }

        // 预测类型
        string predictedType = repairer.PredictImageType(features);
        cout << endl << LOC_STR("repair.predicted_type") << ": " << predictedType << endl;

        // 分析损坏
        DamageAnalysis damage = repairer.AnalyzeDamage(fileData);
        cout << LOC_STR("repair.damage_type") << ": " << GetDamageTypeString(damage.type) << endl;
        cout << LOC_STR("repair.damage_severity") << ": " << fixed << setprecision(1) << (damage.severity * 100) << "%" << endl;
        cout << LOC_STR("repair.can_repair") << ": " << (damage.isRepairable ? LOC_STR("common.yes") : LOC_STR("common.no")) << endl;
        cout << LOC_STR("repair.description") << ": " << damage.description << endl;

        if (!damage.isRepairable && !force) {
            cout << endl << LOC_STR("repair.hint_force") << endl;
            return;
        }
    }

    // 执行修复
    cout << endl << "=== " << LOC_STR("repair.repairing") << " ===" << endl;

    RepairReport report;
    if (useML) {
        report = repairer.TryRepairML(fileData, fileType);
    }
    else {
        report = repairer.TryRepair(fileData, RepairType::HEADER_REBUILD);
    }

    // 显示结果
    cout << LOC_STR("repair.result") << ": " << GetRepairResultString(report.result) << endl;
    cout << LOC_STR("repair.confidence") << ": " << fixed << setprecision(1) << (report.confidence * 100) << "%" << endl;

    if (!report.repairActions.empty()) {
        cout << LOC_STR("repair.actions") << ":" << endl;
        for (const auto& action : report.repairActions) {
            cout << "  - " << action << endl;
        }
    }

    if (!report.message.empty()) {
        cout << LOC_STR("repair.message") << ": " << report.message << endl;
    }

    // 保存修复后的文件
    if (report.result == RepairResult::SUCCESS || report.result == RepairResult::PARTIAL || force) {
        if (WriteBytesToFile(outputPath, fileData)) {
            cout << endl << LOC_STR("repair.output_file") << ": " << outputPath << endl;
            cout << LOC_STR("repair.output_size") << ": " << FormatSize(fileData.size()) << endl;
            cout << LOC_STR("repair.success") << endl;
        }
        else {
            cout << LOC_STR("error.write_failed") << ": " << outputPath << endl;
        }
    }
    else {
        cout << endl << LOC_STR("repair.failed") << endl;
        if (!force) {
            cout << LOC_STR("repair.hint_force") << endl;
        }
    }
}

// ============================================================================
// repair-batch 命令 - 批量修复文件
// ============================================================================
// 用法: repair-batch <input_dir> <output_dir> [--type <jpeg|png>] [--recursive] [--no-ml]
// 示例: repair-batch ./damaged ./fixed
//       repair-batch ./damaged ./fixed --type jpeg --recursive
// ============================================================================
DEFINE_COMMAND_BASE(RepairBatchCommand, "repair-batch |file |file |name |name |name", TRUE)
REGISTER_COMMAND(RepairBatchCommand);

void RepairBatchCommand::Execute(string command) {
    if (!CheckName(command)) {
        return;
    }

    auto& loc = LocalizationManager::Instance();

    // 解析参数
    string inputDir;
    string outputDir;
    string fileType;
    bool useML = true;
    bool recursive = false;
    int repaired = 0, failed = 0, skipped = 0;

    for (int i = 0; i < GET_ARG_COUNT(); i++) {
        string& arg = GET_ARG_STRING(i);

        if (arg == "--type" && i + 1 < GET_ARG_COUNT()) {
            fileType = GET_ARG_STRING(++i);
        }
        else if (arg == "--no-ml") {
            useML = false;
        }
        else if (arg == "--recursive" || arg == "-r") {
            recursive = true;
        }
        else if (inputDir.empty()) {
            inputDir = arg;
        }
        else if (outputDir.empty()) {
            outputDir = arg;
        }
    }

    // 检查参数
    if (inputDir.empty()) {
        cout << LOC_STR("repair_batch.usage") << endl;
        cout << "  repair-batch <input_dir> <output_dir> [--type <jpeg|png>] [--recursive] [--no-ml]" << endl;
        cout << endl;
        cout << LOC_STR("repair_batch.options") << ":" << endl;
        cout << "  --type <type>     " << LOC_STR("repair.option_type") << endl;
        cout << "  --recursive, -r   " << LOC_STR("repair_batch.option_recursive") << endl;
        cout << "  --no-ml           " << LOC_STR("repair.option_no_ml") << endl;
        return;
    }

    if (outputDir.empty()) {
        outputDir = inputDir + "_repaired";
    }

    // 检查输入目录
    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        cout << LOC_STR("error.dir_not_found") << ": " << inputDir << endl;
        return;
    }

    // 创建输出目录
    fs::create_directories(outputDir);

    cout << LOC_STR("repair_batch.input_dir") << ": " << inputDir << endl;
    cout << LOC_STR("repair_batch.output_dir") << ": " << outputDir << endl;
    cout << endl;

    // 创建修复器
    ImageHeaderRepairer repairer;

    // 支持的扩展名
    set<string> supportedExts = {".jpg", ".jpeg", ".png"};

    // 遍历文件
    function<void(const fs::path&)> iterate = [&](const fs::path& path) {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                string ext = entry.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (supportedExts.find(ext) == supportedExts.end()) {
                    continue;
                }

                // 如果指定了类型，只处理该类型
                if (!fileType.empty()) {
                    string expectedExt = "." + fileType;
                    if (fileType == "jpeg") expectedExt = ".jpg";
                    if (ext != expectedExt && !(ext == ".jpeg" && fileType == "jpeg")) {
                        continue;
                    }
                }

                string inputPath = entry.path().string();
                string filename = entry.path().filename().string();

                cout << LOC_STR("repair_batch.processing") << ": " << filename << " ... ";

                // 读取文件
                vector<BYTE> fileData;
                if (!ReadFileToBytes(inputPath, fileData)) {
                    cout << LOC_STR("error.read_failed") << endl;
                    failed++;
                    continue;
                }

                // 分析是否需要修复
                DamageAnalysis damage = repairer.AnalyzeDamage(fileData);
                if (damage.type == DamageType::NONE) {
                    cout << LOC_STR("repair_batch.no_damage") << endl;
                    skipped++;
                    continue;
                }

                if (!damage.isRepairable) {
                    cout << LOC_STR("repair_batch.not_repairable") << endl;
                    failed++;
                    continue;
                }

                // 执行修复
                RepairReport report;
                string type = InferTypeFromExtension(inputPath);
                if (useML) {
                    report = repairer.TryRepairML(fileData, type);
                }
                else {
                    report = repairer.TryRepair(fileData, RepairType::HEADER_REBUILD);
                }

                if (report.result == RepairResult::SUCCESS || report.result == RepairResult::PARTIAL) {
                    // 保存
                    string outputPath = (fs::path(outputDir) / filename).string();
                    if (WriteBytesToFile(outputPath, fileData)) {
                        cout << LOC_STR("common.success") << " (" << fixed << setprecision(0) << (report.confidence * 100) << "%)" << endl;
                        repaired++;
                    }
                    else {
                        cout << LOC_STR("error.write_failed") << endl;
                        failed++;
                    }
                }
                else {
                    cout << LOC_STR("common.failed") << endl;
                    failed++;
                }
            }
            else if (entry.is_directory() && recursive) {
                iterate(entry.path());
            }
        }
    };

    iterate(inputDir);

    // 显示统计
    cout << endl << "=== " << LOC_STR("repair_batch.summary") << " ===" << endl;
    cout << LOC_STR("repair_batch.repaired") << ": " << repaired << endl;
    cout << LOC_STR("repair_batch.failed") << ": " << failed << endl;
    cout << LOC_STR("repair_batch.skipped") << ": " << skipped << endl;
}
