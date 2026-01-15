#include "MLFileRepair.h"
#include "ImageHeaderRepairer.h"
#include "../utils/Logger.h"
#include <algorithm>

// ============================================================================
// 构造函数 / 析构函数
// ============================================================================

MLFileRepair::MLFileRepair() {
    // 注册图像修复器
    RegisterRepairer(make_unique<ImageHeaderRepairer>());
}

MLFileRepair::~MLFileRepair() {
}

// ============================================================================
// 初始化
// ============================================================================

bool MLFileRepair::Initialize(const string& modelsDir) {
    LOG_INFO_FMT("初始化 ML 文件修复器，模型目录: %s", modelsDir.c_str());

    bool allSuccess = true;

    // 初始化所有修复器
    for (auto& pair : repairers) {
        try {
            // ImageHeaderRepairer 等会实现自己的初始化
            LOG_INFO_FMT("初始化修复器: %s", pair.first.c_str());
        }
        catch (const exception& e) {
            LOG_ERROR_FMT("初始化修复器失败 (%s): %s", pair.first.c_str(), e.what());
            allSuccess = false;
        }
    }

    return allSuccess;
}

// ============================================================================
// 修复器管理
// ============================================================================

void MLFileRepair::RegisterRepairer(unique_ptr<FileTypeRepairer> repairer) {
    if (!repairer) return;

    auto types = repairer->GetSupportedTypes();
    for (const auto& type : types) {
        repairers[type] = move(repairer);
        LOG_DEBUG_FMT("注册文件类型修复器: %s", type.c_str());
        break; // 因为 unique_ptr 只能 move 一次，这里简化处理
    }
}

// ============================================================================
// 核心功能
// ============================================================================

DamageAnalysis MLFileRepair::AnalyzeDamage(const vector<BYTE>& data,
                                           const string& expectedType) {
    DamageAnalysis analysis;

    // 查找对应的修复器
    auto it = repairers.find(expectedType);
    if (it == repairers.end()) {
        analysis.type = DamageType::UNKNOWN;
        analysis.isRepairable = false;
        analysis.description = "不支持的文件类型: " + expectedType;
        return analysis;
    }

    // 调用修复器分析
    return it->second->AnalyzeDamage(data);
}

RepairReport MLFileRepair::TryRepair(vector<BYTE>& data,
                                     const string& expectedType,
                                     RepairType repairType) {
    RepairReport report;

    // 查找对应的修复器
    auto it = repairers.find(expectedType);
    if (it == repairers.end()) {
        report.result = RepairResult::NOT_APPLICABLE;
        report.message = "不支持的文件类型: " + expectedType;
        LOG_WARNING_FMT("尝试修复不支持的文件类型: %s", expectedType.c_str());
        return report;
    }

    LOG_INFO_FMT("尝试修复 %s 文件", expectedType.c_str());

    // 调用修复器修复
    report = it->second->TryRepair(data, repairType);

    if (report.result == RepairResult::SUCCESS) {
        LOG_INFO_FMT("文件修复成功 (置信度: %.1f%%)", report.confidence * 100);
    }
    else if (report.result == RepairResult::PARTIAL) {
        LOG_WARNING_FMT("文件部分修复 (置信度: %.1f%%)", report.confidence * 100);
    }
    else {
        LOG_ERROR("文件修复失败");
    }

    return report;
}

bool MLFileRepair::IsSupportedType(const string& fileType) const {
    return repairers.find(fileType) != repairers.end();
}

vector<string> MLFileRepair::GetSupportedTypes() const {
    vector<string> types;
    for (const auto& pair : repairers) {
        types.push_back(pair.first);
    }
    return types;
}

// ============================================================================
// 静态工具方法
// ============================================================================

bool MLFileRepair::IsFileCorrupted(const vector<BYTE>& data, const string& fileType) {
    if (data.empty()) return true;

    // 简单的签名验证
    if (fileType == "jpeg" || fileType == "jpg") {
        if (data.size() < 2) return true;
        return !(data[0] == 0xFF && data[1] == 0xD8);
    }
    else if (fileType == "png") {
        if (data.size() < 8) return true;
        const BYTE png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
        return memcmp(data.data(), png_sig, 8) != 0;
    }

    return false;
}

string MLFileRepair::GetExtension(const string& fileType) {
    if (fileType == "jpeg") return "jpg";
    return fileType;
}
