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
//       mlscan D:                                    # 扫描D盘
//       mlscan C:\Users\Documents                    # 扫描指定目录
//       mlscan D: --types=pdf,jpg,png --max=1000    # 指定类型和最大数量
//       mlscan D: --output=dataset.csv              # 指定输出文件

IMPLEMENT_COMMAND(MLScanCommand, "mlscan |name", TRUE);

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
        cout << "Examples:" << endl;
        cout << "  mlscan D:" << endl;
        cout << "  mlscan C:\\Users\\Documents --types=pdf,doc,jpg --max=500" << endl;
        cout << "  mlscan D: --output=dataset.csv --threads=12" << endl;
        cout << endl;
        cout << "Supported types:" << endl;
        cout << "  pdf, doc, xls, ppt, html, txt, xml, jpg, gif, png" << endl;
        cout << "  (Add more types in config as needed)" << endl;
        return;
    }

    // 解析参数
    string pathArg = GET_ARG_STRING(0);
    string outputFile = "ml_dataset.csv";
    bool useBinary = false;

    ML::DatasetGeneratorConfig config;

    // 解析选项
    for (size_t i = 1; i < GET_ARG_COUNT(); i++) {
        string arg = GET_ARG_STRING(i);

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
        }
        else if (arg.find("--max=") == 0) {
            config.maxSamplesPerType = stoul(arg.substr(6));
        }
        else if (arg.find("--output=") == 0) {
            outputFile = arg.substr(9);
        }
        else if (arg == "--binary") {
            useBinary = true;
            if (outputFile == "ml_dataset.csv") {
                outputFile = "ml_dataset.bin";
            }
        }
        else if (arg.find("--threads=") == 0) {
            config.workerThreads = stoi(arg.substr(10));
        }
        else if (arg == "--include-path") {
            config.includeFilePath = true;
        }
    }

    // 转换路径为宽字符
    wstring widePath;
    widePath.assign(pathArg.begin(), pathArg.end());

    // 打印配置
    cout << endl;
    cout << "=== ML Dataset Generation ===" << endl;
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
    cout << "Files scanned: " << stats.totalFilesScanned << endl;
    cout << "Skipped (too small): " << stats.skippedTooSmall << endl;
    cout << "Skipped (read error): " << stats.skippedReadError << endl;
    cout << "Skipped (quota): " << stats.skippedQuotaReached << endl;
    cout << "Time elapsed: " << fixed << setprecision(1) << stats.elapsedSeconds << " seconds" << endl;
    cout << endl;

    if (stats.totalSamples == 0) {
        cout << "No samples collected. Nothing to export." << endl;
        return;
    }

    // 导出数据集
    cout << "Exporting dataset..." << endl;
    bool exportSuccess = false;

    if (useBinary) {
        exportSuccess = generator.ExportBinary(outputFile);
    } else {
        exportSuccess = generator.ExportCSV(outputFile);
    }

    if (exportSuccess) {
        cout << "Dataset saved to: " << outputFile << endl;
        cout << endl;
        cout << "Next steps:" << endl;
        cout << "  1. Copy " << outputFile << " to ml/data/" << endl;
        cout << "  2. Run: python ml/src/train.py --csv " << outputFile << endl;
    } else {
        cout << "Failed to export dataset." << endl;
    }
}
END_EXECUTE
