/**
 * @file MLCommands.cpp
 * @brief ML 相关命令实现
 *
 * 包含 ML 数据集生成等命令
 */

#include "cmd.h"
#include "CommandMacros.h"
#include "DatasetGenerator.h"
#include "Logger.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace std;

// ============================================================================
// MLScanCommand - 扫描目录/卷生成 ML 训练数据集
// ============================================================================
// 用法: mlscan <路径/驱动器> [--types=type1,type2,...] [--max=N] [--output=file.csv]
//       mlscan D:                                    # 扫描D盘（分类模式）
//       mlscan C:\Users\Documents                    # 扫描指定目录
//       mlscan D: --types=pdf,jpg,png --max=1000    # 指定类型和最大数量
//       mlscan D: --output=dataset.csv              # 指定输出文件
//       mlscan D: --repair                          # 修复模式：收集图像修复训练数据

IMPLEMENT_COMMAND(MLScanCommand, "mlscan |name |name |name |name |name |name |name |name", TRUE);

BEGIN_EXECUTE(MLScanCommand)
{
    if (!HAS_ARG(0)) {
        cout << "=== ML Dataset Generator ===" << endl;
        cout << "Usage: mlscan <path/drive> [options]" << endl;
        cout << endl;
        cout << "Options:" << endl;
        cout << "  --types=type1,type2,...  File types to scan (default: all supported)" << endl;
        cout << "  --max=N                  Max samples per type (default: 2000)" << endl;
        cout << "  --output=file.csv        Output file (default: ml_dataset.csv)" << endl;
        cout << "  --binary                 Export as binary format instead of CSV" << endl;
        cout << "  --threads=N              Worker threads (default: 8)" << endl;
        cout << "  --include-path           Include file paths in output" << endl;
        cout << endl;
        cout << "Repair Mode (for image repair model training):" << endl;
        cout << "  --repair                 Enable repair mode (31-dim image features)" << endl;
        cout << "  --damage-ratio=N         Ratio of damaged samples (default: 0.7)" << endl;
        cout << "  --no-damage              Only collect normal samples (no damage simulation)" << endl;
        cout << endl;
        cout << "Examples:" << endl;
        cout << "  mlscan D:" << endl;
        cout << "  mlscan C:\\Users\\Documents --types=pdf,doc,jpg --max=500" << endl;
        cout << "  mlscan D: --output=dataset.csv --threads=12" << endl;
        cout << "  mlscan D: --repair --output=repair_dataset.csv" << endl;
        cout << endl;
        cout << "Supported types (Classification mode):" << endl;
        cout << "  pdf, doc, xls, ppt, html, txt, xml, jpg, gif, png" << endl;
        cout << endl;
        cout << "Supported types (Repair mode):" << endl;
        cout << "  jpg, jpeg, png" << endl;
        cout << endl;
        return;
    }

    // 解析参数
    string pathArg = GET_ARG_STRING(0);
    string outputFile = "ml_dataset.csv";
    bool useBinary = false;
    bool repairMode = false;
    bool noDamage = false;
    float damageRatio = 0.7f;


    // 用户指定的选项（需要保留）
    bool userSpecifiedTypes = false;
    bool userSpecifiedMax = false;
    bool userSpecifiedThreads = false;
    bool userSpecifiedIncludePath = false;

    ML::DatasetGeneratorConfig config;

    // 解析选项
    for (size_t i = 1; i < GET_ARG_COUNT(); i++) {
        string arg = GET_ARG_STRING(i);
        try {
            if (arg.find("--types=") == 0) {
                // 解析类型列表
                string typeList = arg.substr(8);
                config.targetTypes.clear();

                stringstream ss(typeList);
                string type;
                while (getline(ss, type, ',')) {
                    // 去除空格并转小写
                    type.erase(remove_if(type.begin(), type.end(), ::isspace), type.end());
                    transform(type.begin(), type.end(), type.begin(), ::tolower);
                    if (!type.empty()) {
                        config.targetTypes.insert(type);
                    }
                }
                userSpecifiedTypes = true;
            }
            else if (arg.find("--max=") == 0) {
                config.maxSamplesPerType = stoul(arg.substr(6));
                userSpecifiedMax = true;
            }
            else if (arg.find("--output=") == 0) {
                outputFile = arg.substr(9);
                LOG_DEBUG_FMT("Output file:", outputFile);
            }
            else if (arg == "--binary") {
                useBinary = true;
                if (outputFile == "ml_dataset.csv") {
                    outputFile = "ml_dataset.bin";
                }
            }
            else if (arg.find("--threads=") == 0) {
                config.workerThreads = stoi(arg.substr(10));
                userSpecifiedThreads = true;
                LOG_DEBUG_FMT("Count of thread:", config.workerThreads);
            }
            else if (arg == "--include-path") {
                config.includeFilePath = true;
                userSpecifiedIncludePath = true;
            }
            else if (arg == "--repair") {
                repairMode = true;
                if (outputFile == "ml_dataset.csv") {
                    outputFile = "repair_dataset.csv";
                }
            }
            else if (arg.find("--damage-ratio=") == 0) {
                damageRatio = stof(arg.substr(15));
                LOG_DEBUG_FMT("Damage Ratio is", damageRatio);
                if (damageRatio < 0.0f) damageRatio = 0.0f;
                if (damageRatio > 1.0f) damageRatio = 1.0f;
            }
            else if (arg == "--no-damage") {
                noDamage = true;
            }
            else {
                // 未识别的参数，忽略
            }
        }
        catch(std::invalid_argument e){
            cout << "参数格式不正确" << endl;
            return;
        }
    }
    
    // 配置模式
    if (repairMode) {
        // 保存用户指定的选项
        auto savedTypes = config.targetTypes;
        size_t savedMax = config.maxSamplesPerType;
        int savedThreads = config.workerThreads;
        bool savedIncludePath = config.includeFilePath;

        // 应用修复模式预设
        config = ML::DatasetGeneratorConfig::RepairModePreset();
        config.generateDamagedSamples = !noDamage;
        config.damageRatio = damageRatio;

        // 恢复用户指定的选项
        if (userSpecifiedTypes) {
            config.targetTypes = savedTypes;
        }
        if (userSpecifiedMax) {
            config.maxSamplesPerType = savedMax;
        }
        if (userSpecifiedThreads) {
            config.workerThreads = savedThreads;
        }
        if (userSpecifiedIncludePath) {
            config.includeFilePath = savedIncludePath;
        }
    }

    // 转换路径为宽字符
    wstring widePath;
    widePath.assign(pathArg.begin(), pathArg.end());


    // 打印配置
    cout << endl;
    if (repairMode) {
        cout << "=== ML Repair Dataset Generation ===" << endl;
    } else {
        cout << "=== ML Classification Dataset Generation ===" << endl;
    }
    cout << "Target path: " << pathArg << endl;
    cout << "Target types: ";
    for (const auto& t : config.targetTypes) {
        cout << t << " ";
    }
    cout << endl;
    cout << "Max samples per type: " << config.maxSamplesPerType << endl;
    cout << "Output file: " << outputFile << endl;
    cout << "Output format: " << (useBinary ? "Binary" : "CSV") << endl;
    cout << "Worker threads: " << config.workerThreads << endl;

    if (repairMode) {
        cout << "Mode: Repair (31-dim image features)" << endl;
        cout << "Generate damaged samples: " << (config.generateDamagedSamples ? "Yes" : "No") << endl;
        if (config.generateDamagedSamples) {
            cout << "Damage ratio: " << fixed << setprecision(1) << (config.damageRatio * 100) << "%" << endl;
        }
    } else {
        cout << "Mode: Classification (261-dim features)" << endl;
    }
    cout << endl;

    // 创建生成器
    ML::DatasetGenerator generator(config);

    // 设置进度回调
    generator.SetProgressCallback([](size_t current, size_t total, const string& status) {
        if (total > 0) {
            double progress = 100.0 * current / total;
            cout << "\r[" << fixed << setprecision(1) << progress << "%] "
                 << status << "                    " << flush;
        }
    });

    cout << "Scanning..." << endl;

    // 执行扫描
    bool success = false;
    if (pathArg.length() == 2 && pathArg[1] == ':') {
        // 卷扫描 (e.g., "D:")
        success = generator.ScanVolume(toupper(pathArg[0]));
    } else {
        // 目录扫描
        success = generator.ScanDirectory(widePath);
    }

    cout << endl << endl;

    if (!success) {
        cout << "Scan failed or was interrupted." << endl;
        return;
    }

    // 获取统计信息
    ML::DatasetStats stats = generator.GetStats();

    // 打印统计
    cout << "=== Dataset Statistics ===" << endl;

    if (repairMode) {
        // 修复模式统计
        cout << "Sample breakdown:" << endl;
        cout << "  Normal samples:      " << stats.normalSamples << endl;
        cout << "  Damaged samples:     " << stats.damagedSamples << endl;
        cout << "    - Repairable:      " << stats.repairableSamples << endl;
        cout << "    - Not repairable:  " << stats.unrepairableSamples << endl;
        cout << endl;

        cout << "Damage type distribution:" << endl;
        for (const auto& [damageType, count] : stats.damageTypeCounts) {
            if (count > 0) {
                cout << "  " << setw(20) << left
                     << ML::DatasetGenerator::GetDamageTypeName(static_cast<ML::SimulatedDamageType>(damageType))
                     << ": " << count << endl;
            }
        }
        cout << endl;
        cout << "Feature dimension: " << ImageFeatureVector::FEATURE_DIM << endl;
    } else {
        // 分类模式统计
        // 按类型排序输出
        vector<pair<string, size_t>> sortedTypes(stats.typeCounts.begin(), stats.typeCounts.end());
        sort(sortedTypes.begin(), sortedTypes.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [type, count] : sortedTypes) {
            if (count > 0) {
                cout << "  " << setw(8) << left << type << ": " << count << " samples" << endl;
            }
        }
        cout << "  -------------------------" << endl;
        cout << "  " << setw(8) << left << "Total" << ": " << stats.totalSamples << " samples" << endl;
        cout << endl;
        cout << "Feature dimension: " << ML::FileFeatures::FEATURE_DIM << endl;
    }

    cout << "Files scanned: " << stats.totalFilesScanned << endl;
    cout << "Skipped (too small): " << stats.skippedTooSmall << endl;
    cout << "Skipped (read error): " << stats.skippedReadError << endl;
    cout << "Skipped (quota): " << stats.skippedQuotaReached << endl;
    cout << "Time elapsed: " << fixed << setprecision(1) << stats.elapsedSeconds << " seconds" << endl;
    cout << endl;

    size_t totalSamples;
    if (repairMode) {
        totalSamples = generator.GetRepairSampleCount();
    } else {
        totalSamples = stats.totalSamples;
    }

    if (totalSamples == 0) {
        cout << "No samples collected. Nothing to export." << endl;
        return;
    }

    // 导出数据集
    cout << "Exporting dataset..." << endl;
    bool exportSuccess = false;

    if (repairMode) {
        // 修复模式导出
        exportSuccess = generator.ExportRepairCSV(outputFile);
    } else {
        // 分类模式导出
        if (useBinary) {
            exportSuccess = generator.ExportBinary(outputFile);
        } else {
            exportSuccess = generator.ExportCSV(outputFile);
        }
    }

    if (exportSuccess) {
        cout << "Dataset saved to: " << outputFile << endl;
        cout << endl;
        cout << "Next steps:" << endl;
        if (repairMode) {
            cout << "  1. Copy " << outputFile << " to ml/data/" << endl;
            cout << "  2. Run: python ml/image_repair/train_model.py " << outputFile << endl;
            cout << "  3. Model will be saved to: ml/models/repair/image_type_classifier.onnx" << endl;
            cout << "  4. Copy model to exe directory or models/repair/ subdirectory" << endl;
        } else {
            cout << "  1. Copy " << outputFile << " to ml/data/" << endl;
            cout << "  2. Run: python ml/src/train.py --csv " << outputFile << endl;
            cout << "  3. Run: python ml/src/export_onnx.py export <checkpoint.pt>" << endl;
            cout << "  4. Model will be saved to: ml/models/classification/*.onnx" << endl;
            cout << "  5. Copy model to exe directory or models/classification/ subdirectory" << endl;
        }
    } else {
        cout << "Failed to export dataset." << endl;
    }
}
END_EXECUTE
