#include "ImageHeaderRepairer.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>

// ONNX Runtime
#include <onnxruntime_cxx_api.h>

// JSON 解析（简单实现，用于加载 metadata）
#include <sstream>
#include <regex>

namespace fs = std::filesystem;

// 静态成员初始化
std::shared_ptr<Ort::Env> ImageHeaderRepairer::s_ortEnv = nullptr;

// ============================================================================
// 构造函数 / 析构函数
// ============================================================================

ImageHeaderRepairer::ImageHeaderRepairer() {
    // 尝试自动加载 ML 模型
    LoadRepairModels();
}

ImageHeaderRepairer::~ImageHeaderRepairer() {
    // unique_ptr 会自动释放 ONNX Session
}

// ============================================================================
// ML 模型管理
// ============================================================================

vector<wstring> ImageHeaderRepairer::GetRepairModelSearchPaths() {
    vector<wstring> paths;

    // 1. exe 所在目录
    wchar_t exePath[MAX_PATH];
    if (GetModuleFileNameW(NULL, exePath, MAX_PATH)) {
        fs::path exeDir = fs::path(exePath).parent_path();
        paths.push_back(exeDir.wstring());
        paths.push_back((exeDir / L"models" / L"repair").wstring());
    }

    // 2. 当前工作目录
    paths.push_back(L".");
    paths.push_back(L"models\\repair");
    paths.push_back(L"ml\\models\\repair");

    return paths;
}

bool ImageHeaderRepairer::LoadRepairModels() {
    if (m_mlModelsLoaded) {
        return true;
    }

    try {
        // 初始化 ONNX Runtime 环境（全局共享）
        if (!s_ortEnv) {
            s_ortEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ImageRepair");
        }

        // 搜索模型文件
        vector<wstring> searchPaths = GetRepairModelSearchPaths();

        wstring typeModelPath;
        wstring repairModelPath;
        wstring metadataPath;

        for (const auto& basePath : searchPaths) {
            fs::path base(basePath);

            // 检查类型分类器
            fs::path typePath = base / L"image_type_classifier.onnx";
            if (fs::exists(typePath) && typeModelPath.empty()) {
                typeModelPath = typePath.wstring();
            }

            // 检查可修复性预测器
            fs::path repairPath = base / L"repairability_predictor.onnx";
            if (fs::exists(repairPath) && repairModelPath.empty()) {
                repairModelPath = repairPath.wstring();
            }

            // 检查 metadata
            fs::path metaPath = base / L"image_type_classifier_metadata.json";
            if (fs::exists(metaPath) && metadataPath.empty()) {
                metadataPath = metaPath.wstring();
            }
        }

        // 检查是否找到模型
        if (typeModelPath.empty() && repairModelPath.empty()) {
            LOG_DEBUG("修复模型未找到，将使用规则方法");
            return false;
        }

        // 创建 Session 选项
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // 加载类型分类器
        if (!typeModelPath.empty()) {
            m_typeClassifier = std::make_unique<Ort::Session>(*s_ortEnv, typeModelPath.c_str(), sessionOptions);
            LOG_INFO_FMT("已加载图像类型分类器: %ls", typeModelPath.c_str());
        }

        // 加载可修复性预测器
        if (!repairModelPath.empty()) {
            m_repairabilityPredictor = std::make_unique<Ort::Session>(*s_ortEnv, repairModelPath.c_str(), sessionOptions);
            LOG_INFO_FMT("已加载可修复性预测器: %ls", repairModelPath.c_str());
        }

        // 加载标准化参数
        if (!metadataPath.empty()) {
            LoadNormalizationParams(metadataPath);
        }

        m_mlModelsLoaded = (m_typeClassifier != nullptr);
        return m_mlModelsLoaded;
    }
    catch (const Ort::Exception& e) {
        LOG_ERROR_FMT("ONNX 加载错误: %s", e.what());
        return false;
    }
    catch (const std::exception& e) {
        LOG_ERROR_FMT("模型加载错误: %s", e.what());
        return false;
    }
}

bool ImageHeaderRepairer::LoadNormalizationParams(const wstring& metadataPath) {
    try {
        // 读取 JSON 文件
        std::ifstream file(metadataPath);
        if (!file.is_open()) {
            LOG_WARNING("无法打开 metadata 文件");
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        string jsonStr = buffer.str();

        // 简单的 JSON 解析（查找 mean 和 std 数组）
        // 格式: "mean": [v1, v2, ...], "std": [v1, v2, ...]

        auto parseFloatArray = [](const string& json, const string& key) -> vector<float> {
            vector<float> result;
            string searchKey = "\"" + key + "\"";
            size_t pos = json.find(searchKey);
            if (pos == string::npos) return result;

            size_t start = json.find('[', pos);
            size_t end = json.find(']', start);
            if (start == string::npos || end == string::npos) return result;

            string arrayStr = json.substr(start + 1, end - start - 1);

            // 解析逗号分隔的浮点数
            std::stringstream ss(arrayStr);
            string token;
            while (std::getline(ss, token, ',')) {
                try {
                    result.push_back(std::stof(token));
                }
                catch (...) {}
            }
            return result;
        };

        m_normParams.mean = parseFloatArray(jsonStr, "mean");
        m_normParams.std = parseFloatArray(jsonStr, "std");

        if (m_normParams.mean.size() == ImageFeatureVector::FEATURE_DIM &&
            m_normParams.std.size() == ImageFeatureVector::FEATURE_DIM) {
            m_normParams.loaded = true;
            LOG_INFO_FMT("已加载标准化参数 (%zu 维)", m_normParams.mean.size());
            return true;
        }
        else {
            LOG_WARNING_FMT("标准化参数维度不匹配: mean=%zu, std=%zu, 期望=%zu",
                           m_normParams.mean.size(), m_normParams.std.size(),
                           ImageFeatureVector::FEATURE_DIM);
            return false;
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR_FMT("解析 metadata 错误: %s", e.what());
        return false;
    }
}

vector<float> ImageHeaderRepairer::NormalizeFeatures(const ImageFeatureVector& features) {
    vector<float> raw = features.ToVector();
    vector<float> normalized(raw.size());

    if (m_normParams.loaded && m_normParams.mean.size() == raw.size()) {
        for (size_t i = 0; i < raw.size(); i++) {
            float std_val = m_normParams.std[i];
            if (std_val < 1e-7f) std_val = 1.0f;  // 避免除零
            normalized[i] = (raw[i] - m_normParams.mean[i]) / std_val;
        }
    }
    else {
        // 无标准化参数，使用原始值
        normalized = raw;
    }

    return normalized;
}

vector<float> ImageHeaderRepairer::RunInference(Ort::Session* session, const vector<float>& input) {
    if (!session) {
        return {};
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // 获取输入输出信息
        auto inputName = session->GetInputNameAllocated(0, allocator);
        auto outputName = session->GetOutputNameAllocated(0, allocator);

        // 创建输入张量
        std::array<int64_t, 2> inputShape = {1, (int64_t)input.size()};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(input.data()), input.size(),
            inputShape.data(), inputShape.size());

        // 运行推理
        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        auto outputTensors = session->Run(Ort::RunOptions{nullptr},
                                          inputNames, &inputTensor, 1,
                                          outputNames, 1);

        // 获取输出
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        size_t outputSize = outputInfo.GetElementCount();

        return vector<float>(outputData, outputData + outputSize);
    }
    catch (const Ort::Exception& e) {
        LOG_ERROR_FMT("ONNX 推理错误: %s", e.what());
        return {};
    }
}

bool ImageHeaderRepairer::PredictImageTypeML(const ImageFeatureVector& features,
                                              string& predictedType, float& confidence) {
    if (!m_typeClassifier) {
        return false;
    }

    // 标准化特征
    vector<float> normalized = NormalizeFeatures(features);

    // 运行推理
    vector<float> output = RunInference(m_typeClassifier.get(), normalized);

    if (output.size() < 2) {
        return false;
    }

    // Softmax 并获取预测结果
    // output[0] = jpeg 概率, output[1] = png 概率
    float maxVal = max(output[0], output[1]);
    float expSum = exp(output[0] - maxVal) + exp(output[1] - maxVal);
    float jpegProb = exp(output[0] - maxVal) / expSum;
    float pngProb = exp(output[1] - maxVal) / expSum;

    if (jpegProb > pngProb) {
        predictedType = "jpeg";
        confidence = jpegProb;
    }
    else {
        predictedType = "png";
        confidence = pngProb;
    }

    LOG_DEBUG_FMT("ML 类型预测: %s (%.1f%%)", predictedType.c_str(), confidence * 100);
    return true;
}

bool ImageHeaderRepairer::PredictRepairability(const ImageFeatureVector& features, float& confidence) {
    if (!m_repairabilityPredictor) {
        // 无模型时使用启发式判断
        // 如果能检测到关键标记，则认为可修复
        confidence = 0.5f;
        return true;
    }

    // 标准化特征
    vector<float> normalized = NormalizeFeatures(features);

    // 运行推理
    vector<float> output = RunInference(m_repairabilityPredictor.get(), normalized);

    if (output.size() < 2) {
        confidence = 0.5f;
        return true;  // 默认认为可修复
    }

    // Softmax
    // output[0] = 不可修复概率, output[1] = 可修复概率
    float maxVal = max(output[0], output[1]);
    float expSum = exp(output[0] - maxVal) + exp(output[1] - maxVal);
    float repairableProb = exp(output[1] - maxVal) / expSum;

    confidence = repairableProb;

    LOG_DEBUG_FMT("ML 可修复性预测: %.1f%%", confidence * 100);
    return (repairableProb > 0.5f);
}

// ============================================================================
// 基类接口实现
// ============================================================================

vector<string> ImageHeaderRepairer::GetSupportedTypes() const {
    return {"jpeg", "jpg", "png"};
}

DamageAnalysis ImageHeaderRepairer::AnalyzeDamage(const vector<BYTE>& data) {
    DamageAnalysis analysis;

    if (data.size() < 100) {
        analysis.type = DamageType::UNKNOWN;
        analysis.severity = 1.0;
        analysis.isRepairable = false;
        analysis.description = "文件太小，无法分析";
        return analysis;
    }

    // 检测是否为 JPEG
    if (data.size() >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
        return AnalyzeJPEGDamage(data);
    }
    // 检测是否可能是 JPEG（头部损坏）
    else if (FindJPEGMarker(data, 0xC0, 0) != string::npos) {
        analysis.type = DamageType::HEADER_CORRUPTED;
        analysis.severity = 0.5;
        analysis.isRepairable = true;
        analysis.description = "JPEG 头部损坏，但找到 SOF 标记";
        analysis.damagedRanges.push_back({0, 2});
        return analysis;
    }

    // 检测是否为 PNG
    const BYTE png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (data.size() >= 8 && memcmp(data.data(), png_sig, 8) == 0) {
        return AnalyzePNGDamage(data);
    }
    // 检测是否可能是 PNG（头部损坏）
    else if (FindPNGChunk(data, "IDAT", 0) != string::npos) {
        analysis.type = DamageType::HEADER_CORRUPTED;
        analysis.severity = 0.5;
        analysis.isRepairable = true;
        analysis.description = "PNG 头部损坏，但找到 IDAT chunk";
        analysis.damagedRanges.push_back({0, 8});
        return analysis;
    }

    analysis.type = DamageType::UNKNOWN;
    analysis.severity = 1.0;
    analysis.isRepairable = false;
    analysis.description = "无法识别的图像格式";
    return analysis;
}

RepairReport ImageHeaderRepairer::TryRepair(vector<BYTE>& data, RepairType type) {
    RepairReport report;

    // 分析损坏类型
    DamageAnalysis analysis = AnalyzeDamage(data);

    if (!analysis.isRepairable) {
        report.result = RepairResult::NOT_APPLICABLE;
        report.message = analysis.description;
        return report;
    }

    // 尝试修复
    bool success = false;

    if (analysis.description.find("JPEG") != string::npos) {
        LOG_INFO("尝试修复 JPEG 头部");
        success = RebuildJPEGHeader(data);
        if (success) {
            report.repairActions.push_back("重建 JPEG SOI 和 JFIF APP0 标记");
        }
    }
    else if (analysis.description.find("PNG") != string::npos) {
        LOG_INFO("尝试修复 PNG 头部");
        success = RebuildPNGHeader(data);
        if (success) {
            report.repairActions.push_back("重建 PNG 文件签名和 IHDR chunk");
        }
    }

    if (success) {
        report.result = RepairResult::SUCCESS;
        report.confidence = 0.85;
        report.bytesModified = analysis.damagedRanges.empty() ? 0 :
            analysis.damagedRanges[0].second;
        report.modifiedRanges = analysis.damagedRanges;
        report.message = "头部修复成功";
    }
    else {
        report.result = RepairResult::FAILED;
        report.confidence = 0.0;
        report.message = "头部修复失败";
    }

    return report;
}

// ============================================================================
// JPEG 修复
// ============================================================================

DamageAnalysis ImageHeaderRepairer::AnalyzeJPEGDamage(const vector<BYTE>& data) {
    DamageAnalysis analysis;
    analysis.type = DamageType::NONE;
    analysis.severity = 0.0;
    analysis.isRepairable = true;
    analysis.description = "JPEG 文件完整";

    // 检查 SOI (Start of Image)
    if (data[0] != 0xFF || data[1] != 0xD8) {
        analysis.type = DamageType::HEADER_CORRUPTED;
        analysis.severity = 0.3;
        analysis.description = "JPEG SOI 标记损坏";
        analysis.damagedRanges.push_back({0, 2});
    }

    // 检查是否有 APP0/APP1 标记
    bool hasAPP = false;
    for (size_t i = 2; i < min(data.size(), (size_t)100); i++) {
        if (data[i] == 0xFF && (data[i + 1] == 0xE0 || data[i + 1] == 0xE1)) {
            hasAPP = true;
            break;
        }
    }

    if (!hasAPP) {
        if (analysis.type == DamageType::NONE) {
            analysis.type = DamageType::HEADER_CORRUPTED;
        }
        analysis.severity = max(analysis.severity, 0.4);
        analysis.description += " (缺少 APP 标记)";
    }

    return analysis;
}

size_t ImageHeaderRepairer::FindJPEGMarker(const vector<BYTE>& data,
                                           BYTE marker,
                                           size_t startOffset) {
    for (size_t i = startOffset; i < data.size() - 1; i++) {
        if (data[i] == 0xFF && data[i + 1] == marker) {
            return i;
        }
    }
    return string::npos;
}

bool ImageHeaderRepairer::RebuildJPEGHeader(vector<BYTE>& data) {
    if (data.size() < 100) return false;

    // 查找 SOF (Start of Frame) 标记来定位实际图像数据
    size_t sofPos = FindJPEGMarker(data, 0xC0, 0); // SOF0 (Baseline DCT)
    if (sofPos == string::npos) {
        sofPos = FindJPEGMarker(data, 0xC2, 0); // SOF2 (Progressive DCT)
    }

    if (sofPos == string::npos) {
        LOG_ERROR("未找到 JPEG SOF 标记");
        return false;
    }

    // 构建标准 JPEG 头部
    vector<BYTE> newHeader;

    // 1. SOI (Start of Image)
    newHeader.push_back(0xFF);
    newHeader.push_back(0xD8);

    // 2. JFIF APP0 标记
    const BYTE jfif[] = {
        0xFF, 0xE0,             // APP0 marker
        0x00, 0x10,             // Length (16 bytes)
        0x4A, 0x46, 0x49, 0x46, // "JFIF"
        0x00,                   // Null terminator
        0x01, 0x01,             // Version 1.1
        0x00,                   // Density units (no units)
        0x00, 0x01,             // X density (1)
        0x00, 0x01,             // Y density (1)
        0x00, 0x00              // Thumbnail (0x0)
    };
    newHeader.insert(newHeader.end(), jfif, jfif + sizeof(jfif));

    // 3. 保留从 SOF 开始的原始数据
    if (sofPos >= 2) {
        // 检查 SOF 之前是否有 DQT (Define Quantization Table)
        size_t dqtPos = FindJPEGMarker(data, 0xDB, 0);
        size_t dataStart = (dqtPos != string::npos && dqtPos < sofPos) ? dqtPos : sofPos;

        // 替换头部
        data.erase(data.begin(), data.begin() + dataStart);
        data.insert(data.begin(), newHeader.begin(), newHeader.end());

        LOG_INFO_FMT("JPEG 头部重建成功，新头部大小: %zu 字节", newHeader.size());
        return true;
    }

    return false;
}

bool ImageHeaderRepairer::ExtractJPEGDimensions(const vector<BYTE>& data,
                                                uint16_t& width,
                                                uint16_t& height) {
    size_t sofPos = FindJPEGMarker(data, 0xC0, 0);
    if (sofPos == string::npos) {
        sofPos = FindJPEGMarker(data, 0xC2, 0);
    }

    if (sofPos == string::npos || sofPos + 9 >= data.size()) {
        return false;
    }

    // SOF 结构: FF C0 [length:2] [precision:1] [height:2] [width:2] ...
    height = (data[sofPos + 5] << 8) | data[sofPos + 6];
    width = (data[sofPos + 7] << 8) | data[sofPos + 8];

    return true;
}

// ============================================================================
// PNG 修复
// ============================================================================

DamageAnalysis ImageHeaderRepairer::AnalyzePNGDamage(const vector<BYTE>& data) {
    DamageAnalysis analysis;
    analysis.type = DamageType::NONE;
    analysis.severity = 0.0;
    analysis.isRepairable = true;
    analysis.description = "PNG 文件完整";

    // 检查文件签名
    const BYTE png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (memcmp(data.data(), png_sig, 8) != 0) {
        analysis.type = DamageType::HEADER_CORRUPTED;
        analysis.severity = 0.3;
        analysis.description = "PNG 文件签名损坏";
        analysis.damagedRanges.push_back({0, 8});
    }

    // 检查 IHDR chunk
    if (data.size() >= 33) {
        if (memcmp(&data[12], "IHDR", 4) != 0) {
            if (analysis.type == DamageType::NONE) {
                analysis.type = DamageType::HEADER_CORRUPTED;
            }
            analysis.severity = max(analysis.severity, 0.5);
            analysis.description += " (IHDR chunk 损坏)";
        }
    }

    return analysis;
}

size_t ImageHeaderRepairer::FindPNGChunk(const vector<BYTE>& data,
                                         const char* chunkType,
                                         size_t startOffset) {
    for (size_t i = startOffset; i < data.size() - 12; i++) {
        if (memcmp(&data[i + 4], chunkType, 4) == 0) {
            return i;
        }
    }
    return string::npos;
}

bool ImageHeaderRepairer::RebuildPNGHeader(vector<BYTE>& data) {
    if (data.size() < 100) return false;

    // 查找 IDAT chunk 来定位实际图像数据
    size_t idatPos = FindPNGChunk(data, "IDAT", 0);
    if (idatPos == string::npos) {
        LOG_ERROR("未找到 PNG IDAT chunk");
        return false;
    }

    // 尝试从数据推断图像尺寸（简化版：使用默认值）
    uint32_t width = 800;   // 默认宽度
    uint32_t height = 600;  // 默认高度

    // 构建 PNG 头部
    vector<BYTE> newHeader;

    // 1. PNG 文件签名
    const BYTE png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    newHeader.insert(newHeader.end(), png_sig, png_sig + 8);

    // 2. IHDR chunk
    vector<BYTE> ihdr;

    // Chunk length (13 bytes for IHDR data)
    ihdr.push_back(0x00);
    ihdr.push_back(0x00);
    ihdr.push_back(0x00);
    ihdr.push_back(0x0D);

    // Chunk type "IHDR"
    ihdr.push_back(0x49);
    ihdr.push_back(0x48);
    ihdr.push_back(0x44);
    ihdr.push_back(0x52);

    // Width (4 bytes, big-endian)
    ihdr.push_back((width >> 24) & 0xFF);
    ihdr.push_back((width >> 16) & 0xFF);
    ihdr.push_back((width >> 8) & 0xFF);
    ihdr.push_back(width & 0xFF);

    // Height (4 bytes, big-endian)
    ihdr.push_back((height >> 24) & 0xFF);
    ihdr.push_back((height >> 16) & 0xFF);
    ihdr.push_back((height >> 8) & 0xFF);
    ihdr.push_back(height & 0xFF);

    // Bit depth (8)
    ihdr.push_back(0x08);

    // Color type (6 = Truecolor with alpha)
    ihdr.push_back(0x06);

    // Compression method (0 = deflate)
    ihdr.push_back(0x00);

    // Filter method (0 = adaptive)
    ihdr.push_back(0x00);

    // Interlace method (0 = no interlace)
    ihdr.push_back(0x00);

    // CRC32
    uint32_t crc = CalculateCRC32(&ihdr[4], 17); // Type + Data
    ihdr.push_back((crc >> 24) & 0xFF);
    ihdr.push_back((crc >> 16) & 0xFF);
    ihdr.push_back((crc >> 8) & 0xFF);
    ihdr.push_back(crc & 0xFF);

    newHeader.insert(newHeader.end(), ihdr.begin(), ihdr.end());

    // 3. 保留从 IDAT 开始的原始数据
    data.erase(data.begin(), data.begin() + idatPos);
    data.insert(data.begin(), newHeader.begin(), newHeader.end());

    LOG_INFO_FMT("PNG 头部重建成功，新头部大小: %zu 字节", newHeader.size());
    return true;
}

bool ImageHeaderRepairer::ExtractPNGDimensions(const vector<BYTE>& data,
                                               uint32_t& width,
                                               uint32_t& height) {
    if (data.size() < 24) return false;

    // PNG IHDR 在偏移 8-24
    if (memcmp(&data[12], "IHDR", 4) != 0) {
        return false;
    }

    width = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
    height = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];

    return true;
}

uint32_t ImageHeaderRepairer::CalculateCRC32(const BYTE* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;

    // CRC32 表
    static uint32_t crc_table[256];
    static bool table_computed = false;

    if (!table_computed) {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) {
                if (c & 1)
                    c = 0xEDB88320 ^ (c >> 1);
                else
                    c = c >> 1;
            }
            crc_table[n] = c;
        }
        table_computed = true;
    }

    for (size_t i = 0; i < length; i++) {
        crc = crc_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }

    return crc ^ 0xFFFFFFFF;
}

// ============================================================================
// 特征提取
// ============================================================================

vector<float> ImageHeaderRepairer::ExtractImageFeatures(const vector<BYTE>& data,
                                                        size_t offset,
                                                        size_t length) {
    vector<float> features;

    if (offset + length > data.size()) {
        length = data.size() - offset;
    }

    // 1. 熵
    double entropy = CalculateEntropy(data, offset, length);
    features.push_back((float)entropy);

    // 2. 字节值统计
    vector<int> histogram(256, 0);
    for (size_t i = offset; i < offset + length; i++) {
        histogram[data[i]]++;
    }

    // 计算均值和方差
    double mean = 0.0;
    for (int count : histogram) {
        mean += count;
    }
    mean /= 256;

    double variance = 0.0;
    for (int count : histogram) {
        double diff = count - mean;
        variance += diff * diff;
    }
    variance /= 256;

    features.push_back((float)mean);
    features.push_back((float)sqrt(variance));

    return features;
}

double ImageHeaderRepairer::CalculateEntropy(const vector<BYTE>& data,
                                            size_t offset,
                                            size_t length) {
    if (length == 0) return 0.0;

    vector<int> histogram(256, 0);
    for (size_t i = offset; i < offset + length && i < data.size(); i++) {
        histogram[data[i]]++;
    }

    double entropy = 0.0;
    for (int count : histogram) {
        if (count > 0) {
            double p = (double)count / length;
            entropy -= p * log2(p);
        }
    }

    return entropy;
}

// ============================================================================
// 工具方法
// ============================================================================

bool ImageHeaderRepairer::IsJPEG(const vector<BYTE>& data) {
    return data.size() >= 2 && data[0] == 0xFF && data[1] == 0xD8;
}

bool ImageHeaderRepairer::IsPNG(const vector<BYTE>& data) {
    if (data.size() < 8) return false;
    const BYTE png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    return memcmp(data.data(), png_sig, 8) == 0;
}

bool ImageHeaderRepairer::IsLikelyJPEG(const vector<BYTE>& data) {
    // 宽松检测：查找 JPEG 标记
    if (IsJPEG(data)) return true;

    // 查找 SOF 标记
    if (FindJPEGMarker(data, 0xC0, 0) != string::npos) return true;
    if (FindJPEGMarker(data, 0xC2, 0) != string::npos) return true;

    // 查找 DQT 标记
    if (FindJPEGMarker(data, 0xDB, 0) != string::npos) return true;

    return false;
}

bool ImageHeaderRepairer::IsLikelyPNG(const vector<BYTE>& data) {
    // 宽松检测：查找 PNG chunk
    if (IsPNG(data)) return true;

    // 查找 IDAT chunk
    if (FindPNGChunk(data, "IDAT", 0) != string::npos) return true;

    // 查找 IEND chunk
    if (FindPNGChunk(data, "IEND", 0) != string::npos) return true;

    return false;
}

// ============================================================================
// ImageFeatureVector 实现
// ============================================================================

ImageFeatureVector::ImageFeatureVector()
    : mean(0), stddev(0), entropy(0), zeroRatio(0), ffRatio(0),
      ffMarkerDensity(0), dctBlockRegularity(0), quantTablePresence(0), huffmanTablePresence(0),
      deflateSignature(0), filterBytePattern(0), chunkBoundaryScore(0), colorTypeIndicator(0),
      predictedWidth(0), predictedHeight(0) {
    histogram.fill(0);
}

vector<float> ImageFeatureVector::ToVector() const {
    vector<float> vec;
    vec.reserve(FEATURE_DIM);

    // 基础特征
    vec.push_back(mean);
    vec.push_back(stddev);
    vec.push_back(entropy);
    vec.push_back(zeroRatio);
    vec.push_back(ffRatio);

    // 直方图
    for (float h : histogram) {
        vec.push_back(h);
    }

    // JPEG 特征
    vec.push_back(ffMarkerDensity);
    vec.push_back(dctBlockRegularity);
    vec.push_back(quantTablePresence);
    vec.push_back(huffmanTablePresence);

    // PNG 特征
    vec.push_back(deflateSignature);
    vec.push_back(filterBytePattern);
    vec.push_back(chunkBoundaryScore);
    vec.push_back(colorTypeIndicator);

    // 尺寸特征
    vec.push_back(predictedWidth);
    vec.push_back(predictedHeight);

    return vec;
}

// ============================================================================
// ML 特征提取实现
// ============================================================================

void ImageHeaderRepairer::ExtractBasicFeatures(const vector<BYTE>& data,
                                                size_t offset, size_t length,
                                                ImageFeatureVector& features) {
    if (offset >= data.size()) return;
    length = min(length, data.size() - offset);
    if (length == 0) return;

    // 统计计算
    vector<int> histogram(256, 0);
    double sum = 0;
    int zeroCount = 0;
    int ffCount = 0;

    for (size_t i = offset; i < offset + length; i++) {
        BYTE b = data[i];
        histogram[b]++;
        sum += b;
        if (b == 0) zeroCount++;
        if (b == 0xFF) ffCount++;
    }

    // 均值
    features.mean = (float)(sum / length / 255.0);

    // 标准差
    double variance = 0;
    double mean = sum / length;
    for (size_t i = offset; i < offset + length; i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    features.stddev = (float)(sqrt(variance / length) / 255.0);

    // 熵
    double entropy = 0;
    for (int count : histogram) {
        if (count > 0) {
            double p = (double)count / length;
            entropy -= p * log2(p);
        }
    }
    features.entropy = (float)(entropy / 8.0);  // 归一化到 [0, 1]

    // 特殊字节比例
    features.zeroRatio = (float)zeroCount / length;
    features.ffRatio = (float)ffCount / length;

    // 16 区间直方图
    for (int i = 0; i < 16; i++) {
        int binSum = 0;
        for (int j = i * 16; j < (i + 1) * 16; j++) {
            binSum += histogram[j];
        }
        features.histogram[i] = (float)binSum / length;
    }
}

void ImageHeaderRepairer::ExtractJPEGFeatures(const vector<BYTE>& data,
                                               ImageFeatureVector& features) {
    // 0xFF 标记密度（JPEG 特征）
    int ffMarkerCount = 0;
    for (size_t i = 0; i < data.size() - 1; i++) {
        if (data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF) {
            ffMarkerCount++;
        }
    }
    features.ffMarkerDensity = (float)ffMarkerCount / max((size_t)1, data.size() / 1000);

    // DQT（量化表）存在指示
    features.quantTablePresence = (FindJPEGMarker(data, 0xDB, 0) != string::npos) ? 1.0f : 0.0f;

    // DHT（霍夫曼表）存在指示
    features.huffmanTablePresence = (FindJPEGMarker(data, 0xC4, 0) != string::npos) ? 1.0f : 0.0f;

    // DCT 块规律性（检测 8x8 块模式）
    // 简化实现：检测是否有规律的字节模式
    int regularPatterns = 0;
    for (size_t i = 64; i < min(data.size(), (size_t)4096); i += 64) {
        // 检查 64 字节块的相似性
        bool similar = true;
        for (int j = 0; j < 8 && similar; j++) {
            if (abs((int)data[i + j] - (int)data[i - 64 + j]) > 50) {
                similar = false;
            }
        }
        if (similar) regularPatterns++;
    }
    features.dctBlockRegularity = (float)regularPatterns / 64.0f;
}

void ImageHeaderRepairer::ExtractPNGFeatures(const vector<BYTE>& data,
                                              ImageFeatureVector& features) {
    // DEFLATE 签名强度（检测 zlib 头部）
    int deflateSignatures = 0;
    for (size_t i = 0; i < data.size() - 1; i++) {
        // zlib 头部: 78 01, 78 5E, 78 9C, 78 DA
        if (data[i] == 0x78 &&
            (data[i + 1] == 0x01 || data[i + 1] == 0x5E ||
             data[i + 1] == 0x9C || data[i + 1] == 0xDA)) {
            deflateSignatures++;
        }
    }
    features.deflateSignature = min(1.0f, (float)deflateSignatures / 10.0f);

    // 滤波字节模式（PNG 行滤波器在每行开头）
    // 滤波类型: 0(None), 1(Sub), 2(Up), 3(Average), 4(Paeth)
    int filterBytes = 0;
    for (size_t i = 0; i < min(data.size(), (size_t)10000); i++) {
        if (data[i] <= 4) filterBytes++;
    }
    features.filterBytePattern = (float)filterBytes / min(data.size(), (size_t)10000);

    // Chunk 边界得分（查找类似 chunk 头的模式）
    int chunkPatterns = 0;
    for (size_t i = 0; i < data.size() - 8; i++) {
        // 检查是否像 chunk 类型（4 个 ASCII 字母）
        bool isChunkLike = true;
        for (int j = 4; j < 8 && isChunkLike; j++) {
            BYTE b = data[i + j];
            if (!((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z'))) {
                isChunkLike = false;
            }
        }
        if (isChunkLike) chunkPatterns++;
    }
    features.chunkBoundaryScore = min(1.0f, (float)chunkPatterns / 100.0f);

    // 颜色类型指示（基于数据复杂度推断）
    // 简化：使用熵作为颜色类型的粗略估计
    features.colorTypeIndicator = features.entropy;
}

bool ImageHeaderRepairer::InferPNGDimensionsFromIDAT(const vector<BYTE>& data,
                                                      uint32_t& width,
                                                      uint32_t& height) {
    // 启发式方法：从 IDAT 数据大小推断尺寸
    // 假设: 原始像素数据 ≈ IDAT 压缩后大小 * 压缩比(约3-5)
    // 对于 RGBA (4字节/像素): width * height * 4 ≈ idat_size * 4

    size_t idatPos = FindPNGChunk(data, "IDAT", 0);
    if (idatPos == string::npos) return false;

    // 计算所有 IDAT chunk 的总大小
    size_t totalIdatSize = 0;
    size_t pos = idatPos;
    while (pos != string::npos && pos + 8 < data.size()) {
        uint32_t chunkLen = (data[pos] << 24) | (data[pos + 1] << 16) |
                           (data[pos + 2] << 8) | data[pos + 3];
        totalIdatSize += chunkLen;
        pos = FindPNGChunk(data, "IDAT", pos + 12 + chunkLen);
    }

    if (totalIdatSize == 0) return false;

    // 估算原始像素数据大小（假设压缩比 4:1）
    size_t estimatedRawSize = totalIdatSize * 4;

    // 假设 RGBA (4 字节/像素)
    size_t estimatedPixels = estimatedRawSize / 4;

    // 常见宽高比: 4:3, 16:9, 1:1
    // 尝试找到最接近的常见尺寸
    struct CommonSize { uint32_t w; uint32_t h; };
    CommonSize commonSizes[] = {
        {1920, 1080}, {1280, 720}, {1024, 768}, {800, 600},
        {640, 480}, {1600, 900}, {2560, 1440}, {3840, 2160},
        {512, 512}, {256, 256}, {1024, 1024}, {2048, 2048}
    };

    size_t minDiff = SIZE_MAX;
    uint32_t bestW = 800, bestH = 600;

    for (const auto& size : commonSizes) {
        size_t pixels = (size_t)size.w * size.h;
        size_t diff = (pixels > estimatedPixels) ?
                      (pixels - estimatedPixels) : (estimatedPixels - pixels);
        if (diff < minDiff) {
            minDiff = diff;
            bestW = size.w;
            bestH = size.h;
        }
    }

    width = bestW;
    height = bestH;

    LOG_DEBUG_FMT("PNG 尺寸推断: IDAT 大小=%zu, 估算像素=%zu, 推断尺寸=%ux%u",
                  totalIdatSize, estimatedPixels, width, height);

    return true;
}

ImageFeatureVector ImageHeaderRepairer::ExtractFullFeatures(const vector<BYTE>& data) {
    ImageFeatureVector features;

    if (data.empty()) return features;

    // 提取基础特征（从前 8KB 数据）
    size_t analyzeSize = min(data.size(), (size_t)8192);
    ExtractBasicFeatures(data, 0, analyzeSize, features);

    // 检测可能的图像类型并提取特定特征
    bool likelyJPEG = IsLikelyJPEG(data);
    bool likelyPNG = IsLikelyPNG(data);

    if (likelyJPEG) {
        ExtractJPEGFeatures(data, features);

        // 尝试提取 JPEG 尺寸
        uint16_t w, h;
        if (ExtractJPEGDimensions(data, w, h)) {
            features.predictedWidth = (float)w / 10000.0f;
            features.predictedHeight = (float)h / 10000.0f;
        }
    }

    if (likelyPNG) {
        ExtractPNGFeatures(data, features);

        // 尝试推断 PNG 尺寸
        uint32_t w, h;
        if (InferPNGDimensionsFromIDAT(data, w, h)) {
            features.predictedWidth = (float)w / 10000.0f;
            features.predictedHeight = (float)h / 10000.0f;
        }
    }

    return features;
}

string ImageHeaderRepairer::PredictImageType(const ImageFeatureVector& features) {
    // 优先使用 ML 模型预测
    if (m_mlModelsLoaded && m_typeClassifier) {
        string predictedType;
        float confidence;
        if (PredictImageTypeML(features, predictedType, confidence)) {
            if (confidence > 0.6f) {  // 置信度阈值
                LOG_INFO_FMT("ML 预测图像类型: %s (置信度: %.1f%%)",
                             predictedType.c_str(), confidence * 100);
                return predictedType;
            }
            LOG_DEBUG_FMT("ML 预测置信度过低 (%.1f%%)，回退到规则方法", confidence * 100);
        }
    }

    // 回退: 基于特征的简单分类规则
    // JPEG 特征: 高 FF 标记密度，有量化表和霍夫曼表
    float jpegScore = features.ffMarkerDensity * 0.3f +
                      features.quantTablePresence * 0.3f +
                      features.huffmanTablePresence * 0.3f +
                      (1.0f - features.filterBytePattern) * 0.1f;

    // PNG 特征: DEFLATE 签名，chunk 模式
    float pngScore = features.deflateSignature * 0.3f +
                     features.chunkBoundaryScore * 0.3f +
                     features.filterBytePattern * 0.2f +
                     (1.0f - features.ffMarkerDensity) * 0.2f;

    if (jpegScore > 0.5f && jpegScore > pngScore) {
        return "jpeg";
    }
    else if (pngScore > 0.5f && pngScore > jpegScore) {
        return "png";
    }

    return "unknown";
}

bool ImageHeaderRepairer::PredictPNGDimensions(const ImageFeatureVector& features,
                                                uint32_t& width, uint32_t& height) {
    // 从特征中恢复尺寸（反归一化）
    if (features.predictedWidth > 0 && features.predictedHeight > 0) {
        width = (uint32_t)(features.predictedWidth * 10000.0f);
        height = (uint32_t)(features.predictedHeight * 10000.0f);
        return true;
    }

    // 默认尺寸
    width = 800;
    height = 600;
    return false;
}

bool ImageHeaderRepairer::RebuildPNGHeaderWithSize(vector<BYTE>& data,
                                                    uint32_t width, uint32_t height) {
    // 查找 IDAT chunk
    size_t idatPos = FindPNGChunk(data, "IDAT", 0);
    if (idatPos == string::npos) {
        LOG_ERROR("未找到 PNG IDAT chunk");
        return false;
    }

    // 构建 PNG 头部
    vector<BYTE> newHeader;

    // 1. PNG 文件签名
    const BYTE png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    newHeader.insert(newHeader.end(), png_sig, png_sig + 8);

    // 2. IHDR chunk
    vector<BYTE> ihdr;

    // Chunk length (13 bytes for IHDR data)
    ihdr.push_back(0x00);
    ihdr.push_back(0x00);
    ihdr.push_back(0x00);
    ihdr.push_back(0x0D);

    // Chunk type "IHDR"
    ihdr.push_back(0x49);
    ihdr.push_back(0x48);
    ihdr.push_back(0x44);
    ihdr.push_back(0x52);

    // Width (4 bytes, big-endian)
    ihdr.push_back((width >> 24) & 0xFF);
    ihdr.push_back((width >> 16) & 0xFF);
    ihdr.push_back((width >> 8) & 0xFF);
    ihdr.push_back(width & 0xFF);

    // Height (4 bytes, big-endian)
    ihdr.push_back((height >> 24) & 0xFF);
    ihdr.push_back((height >> 16) & 0xFF);
    ihdr.push_back((height >> 8) & 0xFF);
    ihdr.push_back(height & 0xFF);

    // Bit depth (8)
    ihdr.push_back(0x08);

    // Color type (6 = Truecolor with alpha)
    ihdr.push_back(0x06);

    // Compression method (0 = deflate)
    ihdr.push_back(0x00);

    // Filter method (0 = adaptive)
    ihdr.push_back(0x00);

    // Interlace method (0 = no interlace)
    ihdr.push_back(0x00);

    // CRC32
    uint32_t crc = CalculateCRC32(&ihdr[4], 17);
    ihdr.push_back((crc >> 24) & 0xFF);
    ihdr.push_back((crc >> 16) & 0xFF);
    ihdr.push_back((crc >> 8) & 0xFF);
    ihdr.push_back(crc & 0xFF);

    newHeader.insert(newHeader.end(), ihdr.begin(), ihdr.end());

    // 3. 保留从 IDAT 开始的原始数据
    data.erase(data.begin(), data.begin() + idatPos);
    data.insert(data.begin(), newHeader.begin(), newHeader.end());

    LOG_INFO_FMT("PNG 头部重建成功 (尺寸: %ux%u)", width, height);
    return true;
}

RepairReport ImageHeaderRepairer::TryRepairML(vector<BYTE>& data, const string& expectedType) {
    RepairReport report;

    if (data.size() < 100) {
        report.result = RepairResult::NOT_APPLICABLE;
        report.message = "文件太小";
        return report;
    }

    // 提取特征
    ImageFeatureVector features = ExtractFullFeatures(data);

    // 使用 ML 预测可修复性
    float repairConfidence = 0.5f;
    bool mlPredictRepairable = true;
    if (m_mlModelsLoaded && m_repairabilityPredictor) {
        mlPredictRepairable = PredictRepairability(features, repairConfidence);
        LOG_INFO_FMT("ML 可修复性预测: %s (置信度: %.1f%%)",
                     mlPredictRepairable ? "可修复" : "不可修复",
                     repairConfidence * 100);

        if (!mlPredictRepairable && repairConfidence > 0.8f) {
            report.result = RepairResult::NOT_APPLICABLE;
            report.confidence = 1.0f - repairConfidence;
            report.message = "ML 模型预测此文件不可修复";
            return report;
        }
    }

    // 预测类型（如果未指定）
    string fileType = expectedType;
    if (fileType.empty()) {
        fileType = PredictImageType(features);  // 内部会使用 ML 模型
        if (fileType == "unknown") {
            // 尝试宽松检测
            if (IsLikelyJPEG(data)) fileType = "jpeg";
            else if (IsLikelyPNG(data)) fileType = "png";
        }
    }

    LOG_INFO_FMT("ML 辅助修复，检测类型: %s, 使用ML模型: %s",
                 fileType.c_str(), m_mlModelsLoaded ? "是" : "否(规则方法)");

    bool success = false;

    if (fileType == "jpeg" || fileType == "jpg") {
        success = RebuildJPEGHeader(data);
        if (success) {
            report.repairActions.push_back("重建 JPEG 头部" +
                string(m_mlModelsLoaded ? " (ML 辅助类型识别)" : ""));
        }
    }
    else if (fileType == "png") {
        // 使用 ML 预测的尺寸
        uint32_t width, height;
        if (PredictPNGDimensions(features, width, height)) {
            success = RebuildPNGHeaderWithSize(data, width, height);
            if (success) {
                report.repairActions.push_back("重建 PNG 头部 (尺寸: " +
                    to_string(width) + "x" + to_string(height) + ")");
            }
        }
        else {
            success = RebuildPNGHeader(data);
            if (success) {
                report.repairActions.push_back("重建 PNG 头部 (默认尺寸)");
            }
        }
    }

    if (success) {
        report.result = RepairResult::SUCCESS;
        // 置信度结合 ML 预测和修复成功
        report.confidence = m_mlModelsLoaded ?
            min(0.95, 0.7 + repairConfidence * 0.25) : 0.75;
        report.message = m_mlModelsLoaded ?
            "ML 辅助修复成功" : "规则方法修复成功";
    }
    else {
        report.result = RepairResult::FAILED;
        report.confidence = 0.0;
        report.message = "无法修复";
    }

    return report;
}
