#include "BlockContinuityDetector.h"
#include "Logger.h"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <regex>

// Windows min/max macros conflict with std::min/std::max
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace fs = std::filesystem;

namespace Continuity {

// ============================================================================
// PIMPL 实现类（隐藏 ONNX Runtime 细节）
// ============================================================================
class BlockContinuityDetector::Impl {
public:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions sessionOptions;
    Ort::MemoryInfo memoryInfo;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::string> inputNameStrings;
    std::vector<std::string> outputNameStrings;

    bool modelLoaded = false;

    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "ContinuityDetector"),
             memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    bool LoadModel(const std::wstring& modelPath) {
        try {
            session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;

            size_t numInputs = session->GetInputCount();
            inputNameStrings.clear();
            inputNames.clear();
            for (size_t i = 0; i < numInputs; i++) {
                auto name = session->GetInputNameAllocated(i, allocator);
                inputNameStrings.push_back(name.get());
            }
            for (const auto& name : inputNameStrings) {
                inputNames.push_back(name.c_str());
            }

            size_t numOutputs = session->GetOutputCount();
            outputNameStrings.clear();
            outputNames.clear();
            for (size_t i = 0; i < numOutputs; i++) {
                auto name = session->GetOutputNameAllocated(i, allocator);
                outputNameStrings.push_back(name.get());
            }
            for (const auto& name : outputNameStrings) {
                outputNames.push_back(name.c_str());
            }

            modelLoaded = true;
            return true;
        }
        catch (const Ort::Exception& e) {
            LOG_ERROR("Failed to load ONNX model: " + std::string(e.what()));
            modelLoaded = false;
            return false;
        }
    }

    std::vector<float> RunInference(const std::vector<float>& input) {
        if (!modelLoaded || !session) {
            return {};
        }

        try {
            std::vector<int64_t> inputShape = { 1, static_cast<int64_t>(input.size()) };
            auto inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, const_cast<float*>(input.data()), input.size(),
                inputShape.data(), inputShape.size()
            );

            auto outputTensors = session->Run(
                Ort::RunOptions{ nullptr },
                inputNames.data(), &inputTensor, 1,
                outputNames.data(), outputNames.size()
            );

            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

            size_t outputSize = 1;
            for (auto dim : outputShape) {
                outputSize *= static_cast<size_t>(dim);
            }

            return std::vector<float>(outputData, outputData + outputSize);
        }
        catch (const Ort::Exception& e) {
            LOG_ERROR("ONNX inference failed: " + std::string(e.what()));
            return {};
        }
    }
};

// ============================================================================
// 构造函数和析构函数
// ============================================================================
BlockContinuityDetector::BlockContinuityDetector()
    : pImpl(std::make_unique<Impl>()) {
}

BlockContinuityDetector::~BlockContinuityDetector() = default;

// ============================================================================
// 模型加载
// ============================================================================
bool BlockContinuityDetector::LoadModel(const std::wstring& modelPath) {
    if (!fs::exists(modelPath)) {
        LOG_ERROR("Model file not found");
        return false;
    }

    if (!pImpl->LoadModel(modelPath)) {
        return false;
    }

    // 加载标准化参数
    std::wstring metadataPath = modelPath;
    size_t dotPos = metadataPath.rfind(L'.');
    if (dotPos != std::wstring::npos) {
        metadataPath = metadataPath.substr(0, dotPos) + L"_metadata.json";
    }

    if (fs::exists(metadataPath)) {
        LoadNormalizationParams(metadataPath);
    }

    LOG_INFO("Continuity model loaded successfully");
    return true;
}

bool BlockContinuityDetector::IsModelLoaded() const {
    return pImpl && pImpl->modelLoaded;
}

std::vector<std::wstring> BlockContinuityDetector::GetDefaultModelPaths() {
    std::vector<std::wstring> paths;

    // 获取可执行文件路径
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    fs::path exeDir = fs::path(exePath).parent_path();

    // 搜索路径
    paths.push_back(exeDir / L"continuity_classifier.onnx");
    paths.push_back(exeDir / L"models" / L"continuity_classifier.onnx");
    paths.push_back(exeDir / L"models" / L"continuity" / L"continuity_classifier.onnx");
    paths.push_back(exeDir.parent_path() / L"ml" / L"models" / L"continuity" / L"continuity_classifier.onnx");

    return paths;
}

bool BlockContinuityDetector::AutoLoadModel() {
    for (const auto& path : GetDefaultModelPaths()) {
        if (fs::exists(path)) {
            if (LoadModel(path)) {
                return true;
            }
        }
    }
    LOG_WARNING("No continuity model found in default paths");
    return false;
}

// 辅助函数：从JSON字符串中提取float数组
static bool ParseFloatArray(const std::string& content, const std::string& key,
                            std::vector<float>& result) {
    // 查找 "key": [...]
    std::string pattern = "\"" + key + "\"\\s*:\\s*\\[";
    std::regex re(pattern);
    std::smatch match;

    if (!std::regex_search(content, match, re)) {
        return false;
    }

    size_t startPos = match.position() + match.length();
    size_t endPos = content.find(']', startPos);
    if (endPos == std::string::npos) {
        return false;
    }

    std::string arrayStr = content.substr(startPos, endPos - startPos);

    // 解析逗号分隔的浮点数
    result.clear();
    std::istringstream iss(arrayStr);
    std::string token;
    while (std::getline(iss, token, ',')) {
        try {
            // 移除空白
            size_t first = token.find_first_not_of(" \t\n\r");
            size_t last = token.find_last_not_of(" \t\n\r");
            if (first != std::string::npos && last != std::string::npos) {
                token = token.substr(first, last - first + 1);
            }
            if (!token.empty()) {
                result.push_back(std::stof(token));
            }
        }
        catch (...) {
            // 跳过无效数值
        }
    }

    return !result.empty();
}

bool BlockContinuityDetector::LoadNormalizationParams(const std::wstring& metadataPath) {
    try {
        std::ifstream file(metadataPath);
        if (!file.is_open()) {
            return false;
        }

        // 读取整个文件内容
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        // 检查是否包含 norm_params
        if (content.find("\"norm_params\"") == std::string::npos) {
            return false;
        }

        // 解析 mean 和 std 数组
        std::vector<float> meanArray, stdArray;

        if (!ParseFloatArray(content, "mean", meanArray) ||
            !ParseFloatArray(content, "std", stdArray)) {
            return false;
        }

        if (meanArray.size() == ContinuityFeatures::FEATURE_DIM &&
            stdArray.size() == ContinuityFeatures::FEATURE_DIM) {
            std::copy(meanArray.begin(), meanArray.end(), normParams.mean.begin());
            std::copy(stdArray.begin(), stdArray.end(), normParams.std.begin());
            normParams.loaded = true;
            LOG_INFO("Loaded normalization params from metadata file");
            return true;
        }
    }
    catch (const std::exception& e) {
        LOG_WARNING("Failed to load normalization params: " + std::string(e.what()));
    }
    return false;
}

// ============================================================================
// 特征提取 - 主函数
// ============================================================================
ContinuityFeatures BlockContinuityDetector::ExtractFeatures(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    const std::string& fileType
) {
    ContinuityFeatures features;

    if (!block1_tail || !block2_head || tail_size == 0 || head_size == 0) {
        return features;
    }

    // 提取块1特征 (0-15)
    ExtractBlockFeatures(block1_tail, tail_size, &features.data[0], fileType);

    // 提取块2特征 (16-31)
    ExtractBlockFeatures(block2_head, head_size, &features.data[16], fileType);

    // 提取边界特征 (32-47)
    ExtractBoundaryFeatures(block1_tail, tail_size, block2_head, head_size,
                            &features.data[32], fileType);

    // 提取格式特定特征 (48-63)
    if (fileType == "zip" || fileType == "docx" || fileType == "xlsx" ||
        fileType == "pptx" || fileType == "jar" || fileType == "apk") {
        ExtractZIPFeatures(block1_tail, tail_size, block2_head, head_size,
                           &features.data[48]);
    }
    else if (fileType == "mp3" || fileType == "mp2" || fileType == "mp1") {
        ExtractMP3Features(block1_tail, tail_size, block2_head, head_size,
                           &features.data[48]);
    }
    else if (fileType == "mp4" || fileType == "mov" || fileType == "m4a" ||
             fileType == "m4v" || fileType == "3gp") {
        ExtractMP4Features(block1_tail, tail_size, block2_head, head_size,
                           &features.data[48]);
    }
    else {
        // 通用媒体格式 (wav, avi, flac, etc.)
        ExtractGenericMediaFeatures(block1_tail, tail_size, block2_head, head_size,
                                    &features.data[48]);
    }

    return features;
}

// ============================================================================
// 特征提取 - 单块特征 (16维)
// ============================================================================
void BlockContinuityDetector::ExtractBlockFeatures(
    const BYTE* data, size_t size, float* features, const std::string& fileType
) {
    if (!data || size == 0 || !features) {
        return;
    }

    // 字节统计
    std::array<size_t, 256> byteCount = {0};
    size_t zeroCount = 0;
    size_t highByteCount = 0;
    size_t printableCount = 0;
    double sum = 0;
    double sumSq = 0;

    for (size_t i = 0; i < size; i++) {
        BYTE b = data[i];
        byteCount[b]++;

        if (b == 0) zeroCount++;
        if (b >= 0x80) highByteCount++;
        if (b >= 32 && b <= 126) printableCount++;

        sum += b;
        sumSq += static_cast<double>(b) * b;
    }

    // 熵值
    features[0] = CalculateEntropy(data, size);

    // 均值 [0-1]
    double mean = sum / size;
    features[1] = static_cast<float>(mean / 255.0);

    // 标准差 [0-1]
    double variance = (sumSq / size) - (mean * mean);
    double stddev = std::sqrt(std::max(0.0, variance));
    features[2] = static_cast<float>(stddev / 128.0);  // 归一化

    // 零字节比例
    features[3] = static_cast<float>(zeroCount) / size;

    // 高字节比例
    features[4] = static_cast<float>(highByteCount) / size;

    // 可打印字符比例
    features[5] = static_cast<float>(printableCount) / size;

    // 8区间直方图 (features[6-13])
    CalculateHistogram8(data, size, &features[6]);

    // PK签名检测分数
    features[14] = DetectPKSignature(data, size);

    // 压缩数据特征分数
    features[15] = DetectCompressionScore(data, size);
}

// ============================================================================
// 特征提取 - 边界特征 (16维)
// ============================================================================
void BlockContinuityDetector::ExtractBoundaryFeatures(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    float* features, const std::string& fileType
) {
    if (!block1_tail || !block2_head || !features) {
        return;
    }

    // 计算两块的熵值
    float entropy1 = CalculateEntropy(block1_tail, tail_size);
    float entropy2 = CalculateEntropy(block2_head, head_size);

    // 熵值差异
    features[0] = std::abs(entropy1 - entropy2);

    // 熵值变化梯度（归一化）
    features[1] = (entropy2 - entropy1 + 1.0f) / 2.0f;  // [-1,1] -> [0,1]

    // 均值差异
    double mean1 = 0, mean2 = 0;
    for (size_t i = 0; i < tail_size; i++) mean1 += block1_tail[i];
    for (size_t i = 0; i < head_size; i++) mean2 += block2_head[i];
    mean1 /= tail_size;
    mean2 /= head_size;
    features[2] = static_cast<float>(std::abs(mean1 - mean2) / 255.0);

    // 字节分布相似度（余弦相似度）
    features[3] = CalculateDistributionSimilarity(block1_tail, tail_size,
                                                   block2_head, head_size);

    // 边界平滑度
    features[4] = CalculateBoundarySmoothness(block1_tail, tail_size,
                                               block2_head, head_size);

    // 跨边界相关性
    // 使用边界附近的字节计算相关性
    size_t boundarySize = std::min(size_t(64), std::min(tail_size, head_size));
    const BYTE* tail_end = block1_tail + tail_size - boundarySize;
    double corr = 0;
    for (size_t i = 0; i < boundarySize; i++) {
        corr += static_cast<double>(tail_end[i]) * block2_head[i];
    }
    corr /= (boundarySize * 255.0 * 255.0);
    features[5] = static_cast<float>(corr);

    // 边界字节转移直方图 (features[6-13])
    // 统计块1末尾到块2开头的字节值转移
    std::array<size_t, 8> transitionCount = {0};
    size_t transitionSamples = std::min(size_t(256), std::min(tail_size, head_size));
    for (size_t i = 0; i < transitionSamples; i++) {
        BYTE from = block1_tail[tail_size - transitionSamples + i];
        BYTE to = block2_head[i];
        int diff = static_cast<int>(to) - static_cast<int>(from);
        // 将差异映射到8个区间
        int bin = (diff + 256) * 8 / 512;
        bin = std::clamp(bin, 0, 7);
        transitionCount[bin]++;
    }
    for (int i = 0; i < 8; i++) {
        features[6 + i] = static_cast<float>(transitionCount[i]) / transitionSamples;
    }

    // 边界处是否有本地文件头 (PK\x03\x04)
    features[14] = 0.0f;
    if (head_size >= 4) {
        if (block2_head[0] == 0x50 && block2_head[1] == 0x4B &&
            block2_head[2] == 0x03 && block2_head[3] == 0x04) {
            features[14] = 1.0f;
        }
    }

    // EOCD 接近度
    features[15] = DetectEOCDProximity(block1_tail, tail_size);
}

// ============================================================================
// 特征提取 - ZIP 特定特征 (16维)
// ============================================================================
void BlockContinuityDetector::ExtractZIPFeatures(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    float* features
) {
    std::fill(features, features + 16, 0.0f);

    // DEFLATE 流连续性
    // 检查压缩数据的熵值稳定性
    float entropy1 = CalculateEntropy(block1_tail, tail_size);
    float entropy2 = CalculateEntropy(block2_head, head_size);
    // 压缩数据熵值通常在 7.5-8.0 之间
    bool highEntropy1 = (entropy1 > 0.9f);
    bool highEntropy2 = (entropy2 > 0.9f);
    features[0] = (highEntropy1 && highEntropy2) ? 1.0f : 0.0f;

    // 块边界对齐分数
    // ZIP 本地文件头通常在块边界对齐
    features[1] = 0.0f;
    if (head_size >= 30) {
        if (block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
            features[1] = 1.0f;
        }
    }

    // 估算压缩率
    // 高熵值 = 高压缩
    features[2] = (entropy1 + entropy2) / 2.0f;

    // ZIP 结构完整性分数
    float structureScore = 0.0f;
    // 检查块2是否以有效的 ZIP 结构开始
    if (head_size >= 4) {
        BYTE sig0 = block2_head[0];
        BYTE sig1 = block2_head[1];
        if (sig0 == 0x50 && sig1 == 0x4B) {
            BYTE sig2 = block2_head[2];
            BYTE sig3 = block2_head[3];
            // Local file header
            if (sig2 == 0x03 && sig3 == 0x04) structureScore = 1.0f;
            // Central directory
            else if (sig2 == 0x01 && sig3 == 0x02) structureScore = 0.8f;
            // EOCD
            else if (sig2 == 0x05 && sig3 == 0x06) structureScore = 0.6f;
            // Data descriptor
            else if (sig2 == 0x07 && sig3 == 0x08) structureScore = 0.7f;
        }
    }
    features[3] = structureScore;

    // 中央目录签名检测
    features[4] = 0.0f;
    for (size_t i = 0; i + 4 <= head_size; i++) {
        if (block2_head[i] == 0x50 && block2_head[i+1] == 0x4B &&
            block2_head[i+2] == 0x01 && block2_head[i+3] == 0x02) {
            features[4] = 1.0f;
            break;
        }
    }

    // 数据描述符检测分数
    features[5] = 0.0f;
    for (size_t i = 0; i + 4 <= tail_size; i++) {
        if (block1_tail[i] == 0x50 && block1_tail[i+1] == 0x4B &&
            block1_tail[i+2] == 0x07 && block1_tail[i+3] == 0x08) {
            features[5] = 1.0f;
            break;
        }
    }

    // 文件头链完整性
    // 检查是否有连续的本地文件头
    int localHeaderCount = 0;
    for (size_t i = 0; i + 30 <= head_size; i++) {
        if (block2_head[i] == 0x50 && block2_head[i+1] == 0x4B &&
            block2_head[i+2] == 0x03 && block2_head[i+3] == 0x04) {
            localHeaderCount++;
            // 跳过这个头部
            if (i + 30 <= head_size) {
                WORD nameLen = *reinterpret_cast<const WORD*>(&block2_head[i + 26]);
                WORD extraLen = *reinterpret_cast<const WORD*>(&block2_head[i + 28]);
                i += 30 + nameLen + extraLen - 1;  // -1 因为循环会 +1
            }
        }
    }
    features[6] = std::min(1.0f, localHeaderCount / 3.0f);

    // 压缩大小字段有效性
    features[7] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B &&
        block2_head[2] == 0x03 && block2_head[3] == 0x04) {
        DWORD compSize = *reinterpret_cast<const DWORD*>(&block2_head[18]);
        DWORD uncompSize = *reinterpret_cast<const DWORD*>(&block2_head[22]);
        // 合理性检查
        if (compSize > 0 && compSize <= 500 * 1024 * 1024 &&
            uncompSize > 0 && uncompSize <= 1024 * 1024 * 1024) {
            features[7] = 1.0f;
        }
    }

    // CRC 模式检测
    features[8] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
        DWORD crc = *reinterpret_cast<const DWORD*>(&block2_head[14]);
        // CRC 不应该全是 0 或全是 1
        if (crc != 0 && crc != 0xFFFFFFFF) {
            features[8] = 1.0f;
        }
    }

    // 扩展字段有效性
    features[9] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
        WORD extraLen = *reinterpret_cast<const WORD*>(&block2_head[28]);
        // 扩展字段通常不会太大
        if (extraLen < 1024) {
            features[9] = 1.0f;
        }
    }

    // 文件名有效性分数
    features[10] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B &&
        block2_head[2] == 0x03 && block2_head[3] == 0x04) {
        WORD nameLen = *reinterpret_cast<const WORD*>(&block2_head[26]);
        if (nameLen > 0 && nameLen < 256 && 30 + nameLen <= head_size) {
            const BYTE* filename = &block2_head[30];
            int validChars = 0;
            for (WORD i = 0; i < nameLen; i++) {
                BYTE c = filename[i];
                // 允许的文件名字符
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                    c == '-' || c == '/' || c == '\\' || c > 127) {
                    validChars++;
                }
            }
            features[10] = static_cast<float>(validChars) / nameLen;
        }
    }

    // 解压版本合理性
    features[11] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
        WORD version = *reinterpret_cast<const WORD*>(&block2_head[4]);
        // 常见版本: 10, 20, 45, 63
        if (version <= 63) {
            features[11] = 1.0f;
        }
    }

    // 通用标志合理性
    features[12] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
        WORD flags = *reinterpret_cast<const WORD*>(&block2_head[6]);
        // 检查保留位是否为 0
        if ((flags & 0xF800) == 0) {
            features[12] = 1.0f;
        }
    }

    // 压缩方法检测
    features[13] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
        WORD method = *reinterpret_cast<const WORD*>(&block2_head[8]);
        // 常见压缩方法: 0 (存储), 8 (Deflate), 9 (Deflate64), 12 (BZIP2), 14 (LZMA)
        if (method == 0 || method == 8 || method == 9 || method == 12 || method == 14) {
            features[13] = 1.0f;
        }
    }

    // 最后修改时间合理性
    features[14] = 0.0f;
    if (head_size >= 30 && block2_head[0] == 0x50 && block2_head[1] == 0x4B) {
        WORD modTime = *reinterpret_cast<const WORD*>(&block2_head[10]);
        WORD modDate = *reinterpret_cast<const WORD*>(&block2_head[12]);
        // DOS 时间格式检查
        int hour = (modTime >> 11) & 0x1F;
        int minute = (modTime >> 5) & 0x3F;
        int second = (modTime & 0x1F) * 2;
        int year = ((modDate >> 9) & 0x7F) + 1980;
        int month = (modDate >> 5) & 0x0F;
        int day = modDate & 0x1F;

        if (hour < 24 && minute < 60 && second < 60 &&
            year >= 1980 && year <= 2100 &&
            month >= 1 && month <= 12 &&
            day >= 1 && day <= 31) {
            features[14] = 1.0f;
        }
    }

    // 保留特征
    features[15] = 0.0f;
}

// ============================================================================
// 特征提取 - MP3 特定特征 (16维)
// ============================================================================
void BlockContinuityDetector::ExtractMP3Features(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    float* features
) {
    std::fill(features, features + 16, 0.0f);

    // MP3 帧同步检测 (0xFF 0xFB/0xFA/0xF3/0xF2)
    // 帧同步字节: 11个1 = 0xFF + 0xE0 以上
    auto detectFrameSync = [](const BYTE* data, size_t size) -> int {
        int syncCount = 0;
        for (size_t i = 0; i + 1 < size; i++) {
            if (data[i] == 0xFF && (data[i + 1] & 0xE0) == 0xE0) {
                syncCount++;
                // 验证帧头有效性
                BYTE b = data[i + 1];
                int version = (b >> 3) & 0x03;
                int layer = (b >> 1) & 0x03;
                // version 01 是保留的, layer 00 是保留的
                if (version != 1 && layer != 0) {
                    i += 1;  // 跳过这个同步
                }
            }
        }
        return syncCount;
    };

    int sync1 = detectFrameSync(block1_tail, tail_size);
    int sync2 = detectFrameSync(block2_head, head_size);

    // 帧同步检测分数
    features[0] = std::min(1.0f, (sync1 + sync2) / 20.0f);

    // 帧头有效性分数
    features[1] = 0.0f;
    if (head_size >= 4 && block2_head[0] == 0xFF && (block2_head[1] & 0xE0) == 0xE0) {
        BYTE b1 = block2_head[1];
        BYTE b2 = block2_head[2];
        int version = (b1 >> 3) & 0x03;
        int layer = (b1 >> 1) & 0x03;
        int bitrateIdx = (b2 >> 4) & 0x0F;
        int sampleRateIdx = (b2 >> 2) & 0x03;

        // 有效性检查
        if (version != 1 && layer != 0 &&
            bitrateIdx != 0 && bitrateIdx != 15 &&
            sampleRateIdx != 3) {
            features[1] = 1.0f;
        }
    }

    // 比特率一致性 (连续帧应有相似的比特率)
    features[2] = (sync1 > 0 && sync2 > 0) ? 1.0f : 0.0f;

    // 采样率一致性
    features[3] = features[2];  // 简化：使用相同逻辑

    // 帧长度一致性
    // MP3 帧通常是 417 或 418 字节 (128kbps) 或 1044/1045 字节 (320kbps)
    features[4] = (sync1 >= 2 || sync2 >= 2) ? 1.0f : 0.0f;

    // ID3v2 标签检测
    features[5] = 0.0f;
    if (head_size >= 10) {
        if (block2_head[0] == 'I' && block2_head[1] == 'D' && block2_head[2] == '3') {
            features[5] = 1.0f;
        }
    }
    // ID3v1 检测 (在文件末尾)
    if (tail_size >= 3) {
        if (block1_tail[tail_size - 128] == 'T' &&
            block1_tail[tail_size - 127] == 'A' &&
            block1_tail[tail_size - 126] == 'G') {
            features[5] = std::max(features[5], 0.8f);
        }
    }

    // Xing/VBRI VBR 头检测
    features[6] = 0.0f;
    for (size_t i = 0; i + 4 <= head_size; i++) {
        if ((block2_head[i] == 'X' && block2_head[i+1] == 'i' &&
             block2_head[i+2] == 'n' && block2_head[i+3] == 'g') ||
            (block2_head[i] == 'V' && block2_head[i+1] == 'B' &&
             block2_head[i+2] == 'R' && block2_head[i+3] == 'I') ||
            (block2_head[i] == 'I' && block2_head[i+1] == 'n' &&
             block2_head[i+2] == 'f' && block2_head[i+3] == 'o')) {
            features[6] = 1.0f;
            break;
        }
    }

    // 音频数据熵值 (MP3压缩数据有高熵)
    float entropy1 = CalculateEntropy(block1_tail, tail_size);
    float entropy2 = CalculateEntropy(block2_head, head_size);
    features[7] = (entropy1 > 0.85f && entropy2 > 0.85f) ? 1.0f : 0.0f;

    // 熵值稳定性
    features[8] = 1.0f - std::abs(entropy1 - entropy2);

    // 保留
    features[9] = 0.0f;
    features[10] = 0.0f;
    features[11] = 0.0f;
    features[12] = 0.0f;
    features[13] = 0.0f;
    features[14] = 0.0f;
    features[15] = 0.0f;
}

// ============================================================================
// 特征提取 - MP4/MOV 特定特征 (16维)
// ============================================================================
void BlockContinuityDetector::ExtractMP4Features(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    float* features
) {
    std::fill(features, features + 16, 0.0f);

    // MP4 Box (atom) 检测
    // Box 结构: 4字节大小 + 4字节类型
    auto detectBox = [](const BYTE* data, size_t size) -> std::pair<bool, std::string> {
        if (size < 8) return {false, ""};

        // 大端序读取大小
        uint32_t boxSize = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];

        // 检查类型是否为有效的 ASCII
        bool validType = true;
        char type[5] = {0};
        for (int i = 0; i < 4; i++) {
            char c = static_cast<char>(data[4 + i]);
            type[i] = c;
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == ' ' || c == '-' || c == '_')) {
                validType = false;
            }
        }

        // 大小合理性检查
        bool validSize = (boxSize >= 8 && boxSize < 0x80000000) || boxSize == 0 || boxSize == 1;

        return {validType && validSize, std::string(type, 4)};
    };

    // Box 头检测分数
    auto [valid1, type1] = detectBox(block2_head, head_size);
    features[0] = valid1 ? 1.0f : 0.0f;

    // Box 大小有效性
    if (head_size >= 8) {
        uint32_t boxSize = (block2_head[0] << 24) | (block2_head[1] << 16) |
                          (block2_head[2] << 8) | block2_head[3];
        if (boxSize >= 8 && boxSize <= 500 * 1024 * 1024) {
            features[1] = 1.0f;
        } else if (boxSize == 0 || boxSize == 1) {
            features[1] = 0.8f;  // 延续到文件末尾或64位大小
        }
    }

    // mdat 连续性分数 (mdat 是主要的媒体数据容器)
    features[2] = 0.0f;
    if (type1 == "mdat") {
        features[2] = 1.0f;
    }
    // 检查块1是否在 mdat 内部 (高熵数据)
    float entropy1 = CalculateEntropy(block1_tail, tail_size);
    if (entropy1 > 0.9f) {
        features[2] = std::max(features[2], 0.7f);
    }

    // moov 结构检测
    features[3] = 0.0f;
    if (type1 == "moov" || type1 == "trak" || type1 == "mdia" ||
        type1 == "minf" || type1 == "stbl") {
        features[3] = 1.0f;
    }

    // ftyp 检测 (文件类型标识，通常在文件开头)
    features[4] = 0.0f;
    if (type1 == "ftyp") {
        features[4] = 1.0f;
    }

    // 原子边界对齐
    features[5] = valid1 ? 1.0f : 0.0f;

    // 常见 box 类型检测
    static const char* commonBoxes[] = {
        "ftyp", "moov", "mdat", "free", "skip", "wide",
        "pnot", "uuid", "moof", "mfra", "sidx", "styp"
    };
    features[6] = 0.0f;
    for (const char* box : commonBoxes) {
        if (type1 == box) {
            features[6] = 1.0f;
            break;
        }
    }

    // 媒体数据特征 (mdat 内容通常高熵)
    float entropy2 = CalculateEntropy(block2_head, head_size);
    features[7] = (entropy1 + entropy2) / 2.0f;

    // 熵值稳定性
    features[8] = 1.0f - std::abs(entropy1 - entropy2);

    // 保留
    features[9] = 0.0f;
    features[10] = 0.0f;
    features[11] = 0.0f;
    features[12] = 0.0f;
    features[13] = 0.0f;
    features[14] = 0.0f;
    features[15] = 0.0f;
}

// ============================================================================
// 特征提取 - 通用媒体特征 (16维)
// ============================================================================
void BlockContinuityDetector::ExtractGenericMediaFeatures(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    float* features
) {
    std::fill(features, features + 16, 0.0f);

    // 熵值特征
    float entropy1 = CalculateEntropy(block1_tail, tail_size);
    float entropy2 = CalculateEntropy(block2_head, head_size);

    // 高熵数据检测 (压缩或加密数据)
    features[0] = (entropy1 > 0.9f) ? 1.0f : entropy1;
    features[1] = (entropy2 > 0.9f) ? 1.0f : entropy2;

    // 熵值稳定性
    features[2] = 1.0f - std::abs(entropy1 - entropy2);

    // 字节分布相似度
    features[3] = CalculateDistributionSimilarity(block1_tail, tail_size,
                                                   block2_head, head_size);

    // 边界平滑度
    features[4] = CalculateBoundarySmoothness(block1_tail, tail_size,
                                               block2_head, head_size);

    // WAV RIFF 检测
    features[5] = 0.0f;
    if (head_size >= 12) {
        if (block2_head[0] == 'R' && block2_head[1] == 'I' &&
            block2_head[2] == 'F' && block2_head[3] == 'F') {
            features[5] = 1.0f;
        }
    }

    // AVI 检测
    features[6] = 0.0f;
    if (head_size >= 12) {
        if (block2_head[0] == 'R' && block2_head[1] == 'I' &&
            block2_head[2] == 'F' && block2_head[3] == 'F' &&
            block2_head[8] == 'A' && block2_head[9] == 'V' &&
            block2_head[10] == 'I' && block2_head[11] == ' ') {
            features[6] = 1.0f;
        }
    }

    // FLAC 检测
    features[7] = 0.0f;
    if (head_size >= 4) {
        if (block2_head[0] == 'f' && block2_head[1] == 'L' &&
            block2_head[2] == 'a' && block2_head[3] == 'C') {
            features[7] = 1.0f;
        }
    }

    // OGG 检测
    features[8] = 0.0f;
    if (head_size >= 4) {
        if (block2_head[0] == 'O' && block2_head[1] == 'g' &&
            block2_head[2] == 'g' && block2_head[3] == 'S') {
            features[8] = 1.0f;
        }
    }

    // PNG 检测
    features[9] = 0.0f;
    if (head_size >= 8) {
        if (block2_head[0] == 0x89 && block2_head[1] == 'P' &&
            block2_head[2] == 'N' && block2_head[3] == 'G') {
            features[9] = 1.0f;
        }
    }

    // JPEG 检测
    features[10] = 0.0f;
    if (head_size >= 3) {
        if (block2_head[0] == 0xFF && block2_head[1] == 0xD8 &&
            block2_head[2] == 0xFF) {
            features[10] = 1.0f;
        }
    }

    // 保留
    features[11] = 0.0f;
    features[12] = 0.0f;
    features[13] = 0.0f;
    features[14] = 0.0f;
    features[15] = 0.0f;
}

// ============================================================================
// 推理
// ============================================================================
ContinuityResult BlockContinuityDetector::Predict(const ContinuityFeatures& features) {
    ContinuityResult result;

    if (!IsModelLoaded()) {
        result.reason = "Model not loaded";
        return result;
    }

    // 复制并标准化特征
    ContinuityFeatures normalizedFeatures = features;
    NormalizeFeatures(normalizedFeatures);

    // 运行推理
    auto output = RunInference(normalizedFeatures.ToVector());
    if (output.empty()) {
        result.reason = "Inference failed";
        return result;
    }

    // 解析输出 (假设输出是 2 个 logits: [不连续, 连续])
    if (output.size() >= 2) {
        // Softmax
        float maxLogit = std::max(output[0], output[1]);
        float exp0 = std::exp(output[0] - maxLogit);
        float exp1 = std::exp(output[1] - maxLogit);
        float sum = exp0 + exp1;

        result.score = exp1 / sum;  // 连续的概率
        result.confidence = std::abs(result.score - 0.5f) * 2.0f;  // 置信度
        result.isContinuous = (result.score > defaultThreshold);
        result.reason = result.isContinuous ? "High continuity score" : "Low continuity score";
    }

    return result;
}

ContinuityResult BlockContinuityDetector::PredictContinuity(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size,
    const std::string& fileType, float threshold
) {
    auto features = ExtractFeatures(block1_tail, tail_size, block2_head, head_size, fileType);

    float oldThreshold = defaultThreshold;
    defaultThreshold = threshold;
    auto result = Predict(features);
    defaultThreshold = oldThreshold;

    return result;
}

std::vector<ContinuityResult> BlockContinuityDetector::PredictBatch(
    const std::vector<ContinuityFeatures>& featuresBatch, float threshold
) {
    std::vector<ContinuityResult> results;
    results.reserve(featuresBatch.size());

    float oldThreshold = defaultThreshold;
    defaultThreshold = threshold;

    for (const auto& features : featuresBatch) {
        results.push_back(Predict(features));
    }

    defaultThreshold = oldThreshold;
    return results;
}

void BlockContinuityDetector::NormalizeFeatures(ContinuityFeatures& features) const {
    if (!normParams.loaded) {
        return;
    }

    for (size_t i = 0; i < ContinuityFeatures::FEATURE_DIM; i++) {
        if (normParams.std[i] > 1e-8f) {
            features.data[i] = (features.data[i] - normParams.mean[i]) / normParams.std[i];
        }
    }
}

std::vector<float> BlockContinuityDetector::RunInference(const std::vector<float>& input) {
    return pImpl->RunInference(input);
}

// ============================================================================
// 工具函数实现
// ============================================================================
float BlockContinuityDetector::CalculateEntropy(const BYTE* data, size_t size) {
    if (!data || size == 0) return 0.0f;

    std::array<size_t, 256> counts = {0};
    for (size_t i = 0; i < size; i++) {
        counts[data[i]]++;
    }

    double entropy = 0.0;
    double logSize = std::log2(static_cast<double>(size));

    for (int i = 0; i < 256; i++) {
        if (counts[i] > 0) {
            double p = static_cast<double>(counts[i]) / size;
            entropy -= p * std::log2(p);
        }
    }

    // 归一化到 [0, 1]
    return static_cast<float>(entropy / 8.0);
}

void BlockContinuityDetector::CalculateHistogram8(const BYTE* data, size_t size, float* histogram) {
    std::array<size_t, 8> bins = {0};

    for (size_t i = 0; i < size; i++) {
        int bin = data[i] / 32;  // 256 / 8 = 32
        bins[bin]++;
    }

    for (int i = 0; i < 8; i++) {
        histogram[i] = static_cast<float>(bins[i]) / size;
    }
}

float BlockContinuityDetector::CalculateDistributionSimilarity(
    const BYTE* data1, size_t size1,
    const BYTE* data2, size_t size2
) {
    std::array<float, 256> dist1 = {0}, dist2 = {0};

    for (size_t i = 0; i < size1; i++) dist1[data1[i]]++;
    for (size_t i = 0; i < size2; i++) dist2[data2[i]]++;

    for (int i = 0; i < 256; i++) {
        dist1[i] /= size1;
        dist2[i] /= size2;
    }

    // 余弦相似度
    double dot = 0, norm1 = 0, norm2 = 0;
    for (int i = 0; i < 256; i++) {
        dot += dist1[i] * dist2[i];
        norm1 += dist1[i] * dist1[i];
        norm2 += dist2[i] * dist2[i];
    }

    if (norm1 < 1e-10 || norm2 < 1e-10) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(norm1) * std::sqrt(norm2)));
}

float BlockContinuityDetector::CalculateBoundarySmoothness(
    const BYTE* block1_tail, size_t tail_size,
    const BYTE* block2_head, size_t head_size
) {
    // 计算边界处的字节变化平滑度
    // 使用最后几个字节和开头几个字节的差异
    size_t sampleSize = std::min(size_t(16), std::min(tail_size, head_size));

    double totalDiff = 0;
    const BYTE* tail_end = block1_tail + tail_size - sampleSize;

    for (size_t i = 0; i < sampleSize; i++) {
        int diff = std::abs(static_cast<int>(tail_end[i]) - static_cast<int>(block2_head[i]));
        totalDiff += diff;
    }

    double avgDiff = totalDiff / sampleSize;
    // 归一化：差异越小越平滑，返回值越高
    return static_cast<float>(1.0 - avgDiff / 255.0);
}

float BlockContinuityDetector::DetectPKSignature(const BYTE* data, size_t size) {
    if (!data || size < 4) return 0.0f;

    int signatureCount = 0;
    for (size_t i = 0; i + 4 <= size; i++) {
        if (data[i] == 0x50 && data[i+1] == 0x4B) {
            // 检查是否是有效的 ZIP 签名
            BYTE sig2 = data[i+2];
            BYTE sig3 = data[i+3];
            if ((sig2 == 0x03 && sig3 == 0x04) ||  // Local file header
                (sig2 == 0x01 && sig3 == 0x02) ||  // Central directory
                (sig2 == 0x05 && sig3 == 0x06) ||  // EOCD
                (sig2 == 0x07 && sig3 == 0x08)) {  // Data descriptor
                signatureCount++;
            }
        }
    }

    return std::min(1.0f, signatureCount / 5.0f);
}

float BlockContinuityDetector::DetectCompressionScore(const BYTE* data, size_t size) {
    // 压缩数据通常具有高熵值
    float entropy = CalculateEntropy(data, size);

    // 压缩数据的熵值通常在 0.9-1.0 之间
    if (entropy > 0.9f) {
        return 1.0f;
    } else if (entropy > 0.7f) {
        return (entropy - 0.7f) / 0.2f;
    }
    return 0.0f;
}

float BlockContinuityDetector::DetectEOCDProximity(const BYTE* data, size_t size) {
    if (!data || size < 22) return 0.0f;

    // 从后向前搜索 EOCD 签名
    for (size_t i = size - 22; i > 0 && i > size - 65536; i--) {
        if (data[i] == 0x50 && data[i+1] == 0x4B &&
            data[i+2] == 0x05 && data[i+3] == 0x06) {
            // 发现 EOCD，计算接近度
            float proximity = 1.0f - static_cast<float>(size - i) / size;
            return proximity;
        }
    }
    return 0.0f;
}

} // namespace Continuity
