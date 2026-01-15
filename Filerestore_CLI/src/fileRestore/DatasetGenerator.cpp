/**
 * @file DatasetGenerator.cpp
 * @brief ML数据集生成器的实现
 */

#include "DatasetGenerator.h"
#include "Logger.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace ML {

// 辅助函数：宽字符转UTF-8
static std::string WideToUtf8(const std::wstring& wstr) {
    if (wstr.empty()) return "";

    int size = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(),
                                    nullptr, 0, nullptr, nullptr);
    if (size <= 0) return "";

    std::string result(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(),
                        &result[0], size, nullptr, nullptr);
    return result;
}

// ============================================================================
// 构造函数和析构函数
// ============================================================================

DatasetGenerator::DatasetGenerator(const DatasetGeneratorConfig& config)
    : m_config(config)
    , m_totalScanned(0)
    , m_skippedSmall(0)
    , m_skippedError(0)
    , m_skippedQuota(0)
    , m_running(false)
    , m_stopFlag(false)
    , m_totalFiles(0)
    , m_processedFiles(0)
    , m_normalSamples(0)
    , m_damagedSamples(0)
    , m_repairableSamples(0)
    , m_unrepairableSamples(0)
    , m_positiveSamples(0)
    , m_negativeSamples(0)
    , m_rng(std::random_device{}())
{
    QueryPerformanceFrequency(&m_frequency);

    // 初始化每个文件类型的计数器
    for (const auto& type : m_config.targetTypes) {
        m_typeCounts[type] = 0;
    }

    // 初始化损坏类型计数器
    for (int i = 0; i <= static_cast<int>(SimulatedDamageType::RANDOM_CORRUPTION); i++) {
        m_damageTypeCounts[i] = 0;
    }

    // 初始化连续性样本类型计数器（包括损坏样本类型）
    for (int i = 0; i <= static_cast<int>(ContinuitySampleType::CORRUPTED_PARTIAL); i++) {
        m_sampleTypeCounts[i] = 0;
    }
}

DatasetGenerator::~DatasetGenerator() {
    Stop();
}

// ============================================================================
// 扫描方法
// ============================================================================

bool DatasetGenerator::ScanDirectory(const std::wstring& path) {
    return ScanDirectories({ path });
}

bool DatasetGenerator::ScanDirectories(const std::vector<std::wstring>& paths) {
    if (m_running.load()) {
        LOG_WARNING("DatasetGenerator is already running");
        return false;
    }

    // 记录当前模式
    if (m_config.mode == DatasetMode::REPAIR) {
        LOG_INFO("DatasetGenerator starting in REPAIR mode (31-dim features)");
    } else if (m_config.mode == DatasetMode::CONTINUITY) {
        LOG_INFO("DatasetGenerator starting in CONTINUITY mode (64-dim features)");
    } else {
        LOG_INFO("DatasetGenerator starting in CLASSIFICATION mode (261-dim features)");
    }

    // 增量模式：加载之前的进度
    if (m_config.incrementalMode && !m_config.progressFilePath.empty()) {
        if (LoadProgress(m_config.progressFilePath)) {
            LOG_INFO_FMT("Incremental mode: Resuming from %zu processed files",
                        m_progress.processedFiles.size());
        } else {
            LOG_INFO("Incremental mode: Starting fresh (no previous progress)");
            // 初始化新的会话ID
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            m_progress.sessionId = std::to_string(time_t);
            m_progress.mode = m_config.mode;
        }
    }

    m_running = true;
    m_stopFlag = false;
    QueryPerformanceCounter(&m_startTime);

    // 重置统计（增量模式不重置已处理计数）
    m_totalScanned = 0;
    m_skippedSmall = 0;
    m_skippedError = 0;
    m_skippedQuota = 0;
    m_totalFiles = 0;
    m_processedFiles = 0;

    size_t skippedAlreadyProcessed = 0;

    try {
        // 阶段1：收集候选文件
        ReportProgress("Collecting candidate files...");  // 保持英文日志
        std::vector<std::pair<std::wstring, std::string>> candidateFiles;

        for (const auto& basePath : paths) {
            if (m_stopFlag.load()) break;

            if (!fs::exists(basePath)) {
                LOG_WARNING_FMT("Path does not exist: %S", basePath.c_str());
                continue;
            }

            std::error_code ec;
            for (auto it = fs::recursive_directory_iterator(basePath,
                          fs::directory_options::skip_permission_denied, ec);
                 it != fs::recursive_directory_iterator(); ) {

                if (m_stopFlag.load()) break;

                try {
                    const auto& entry = *it;

                    if (entry.is_regular_file(ec) && !ec) {
                        std::string ext = GetLowerExtension(entry.path().wstring());

                        if (ShouldProcessType(ext)) {
                            // 增量模式：跳过已处理的文件
                            if (m_config.incrementalMode && IsFileProcessed(entry.path().wstring())) {
                                skippedAlreadyProcessed++;
                                ++it;
                                continue;
                            }

                            // 检查文件大小是否满足最小要求
                            auto fileSize = entry.file_size(ec);
                            if (!ec && fileSize >= m_config.minFileSize) {
                                // 检查该类型的样本配额是否已满
                                if (m_typeCounts[ext].load() < m_config.maxSamplesPerType) {
                                    candidateFiles.emplace_back(entry.path().wstring(), ext);
                                }
                            }
                        }
                    }

                    ++it;
                } catch (const std::exception&) {
                    ++it;
                }
            }
        }

        if (m_config.incrementalMode && skippedAlreadyProcessed > 0) {
            LOG_INFO_FMT("Incremental mode: Skipped %zu already processed files", skippedAlreadyProcessed);
        }

        if (m_stopFlag.load()) {
            // 保存进度后退出
            if (m_config.incrementalMode && !m_config.progressFilePath.empty()) {
                SaveProgress(m_config.progressFilePath);
            }
            m_running = false;
            return false;
        }

        m_totalFiles = candidateFiles.size();
        LOG_INFO_FMT("Found %zu new candidate files to process", candidateFiles.size());

        if (candidateFiles.empty()) {
            m_running = false;
            return true;
        }

        // 阶段2：启动工作线程进行特征提取
        ReportProgress("Starting feature extraction...");  // 保持英文日志

        int numWorkers = (std::min)(m_config.workerThreads,
                                   (int)std::thread::hardware_concurrency());
        if (numWorkers < 1) numWorkers = 1;

        // 将候选文件添加到任务队列
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            for (auto& file : candidateFiles) {
                m_taskQueue.push(std::move(file));
            }
        }

        // 创建并启动工作线程
        for (int i = 0; i < numWorkers; i++) {
            m_workers.emplace_back(&DatasetGenerator::WorkerFunction, this);
        }

        // 等待所有工作线程完成
        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        m_workers.clear();

        // 计算总耗时
        LARGE_INTEGER endTime;
        QueryPerformanceCounter(&endTime);
        double elapsed = (double)(endTime.QuadPart - m_startTime.QuadPart) / m_frequency.QuadPart;

        LOG_INFO_FMT("Dataset generation completed in %.2f seconds", elapsed);

        // 根据模式记录正确的样本数
        if (m_config.mode == DatasetMode::REPAIR) {
            LOG_INFO_FMT("Repair mode: Total repair samples collected: %zu", m_repairSamples.size());
            LOG_INFO_FMT("  Normal: %zu, Damaged: %zu", m_normalSamples.load(), m_damagedSamples.load());
        } else if (m_config.mode == DatasetMode::CONTINUITY) {
            LOG_INFO_FMT("Continuity mode: Samples this batch: %zu", m_continuitySamples.size());
            LOG_INFO_FMT("  Positive: %zu, Negative: %zu", m_positiveSamples.load(), m_negativeSamples.load());

            // 增量模式：追加到CSV并保存进度
            if (m_config.incrementalMode) {
                if (!m_progress.outputPath.empty()) {
                    AppendToContinuityCSV(m_progress.outputPath);
                }
                if (!m_config.progressFilePath.empty()) {
                    SaveProgress(m_config.progressFilePath);
                }
                LOG_INFO_FMT("Incremental mode: Total samples written: %zu", m_progress.totalSamplesWritten);
            }
        } else {
            LOG_INFO_FMT("Classification mode: Total samples collected: %zu", m_samples.size());
        }

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error during scan: %s", e.what());
        // 增量模式：即使出错也保存进度
        if (m_config.incrementalMode && !m_config.progressFilePath.empty()) {
            SaveProgress(m_config.progressFilePath);
        }
        m_running = false;
        return false;
    }

    m_running = false;
    return true;
}

bool DatasetGenerator::ScanVolume(char driveLetter) {
    std::wstring volumePath = std::wstring(1, driveLetter) + L":\\";
    return ScanDirectory(volumePath);
}

void DatasetGenerator::Stop() {
    m_stopFlag = true;

    // 通知所有等待中的工作线程退出
    m_taskAvailable.notify_all();

    // 等待所有工作线程结束
    for (auto& worker : m_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    m_workers.clear();

    m_running = false;
}

// ============================================================================
// 工作线程
// ============================================================================

void DatasetGenerator::WorkerFunction() {
    while (!m_stopFlag.load()) {
        std::pair<std::wstring, std::string> task;

        // 从任务队列获取下一个任务
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);

            if (m_taskQueue.empty()) {
                // 任务队列已空，线程退出
                break;
            }

            task = std::move(m_taskQueue.front());
            m_taskQueue.pop();
        }

        // 处理当前文件
        ProcessFile(task.first, task.second);

        // 更新处理进度
        m_processedFiles++;
        if (m_processedFiles % 100 == 0) {
            std::ostringstream ss;
            ss << "Processing: " << m_processedFiles.load() << "/" << m_totalFiles.load();
            ReportProgress(ss.str());
        }
    }
}

void DatasetGenerator::ProcessFile(const std::wstring& filePath, const std::string& extension) {
    m_totalScanned++;

    // 再次检查配额（多线程环境下配额可能已被其他线程填满）
    size_t currentCount = m_typeCounts[extension].load();
    if (currentCount >= m_config.maxSamplesPerType) {
        m_skippedQuota++;
        return;
    }

    // 使用CAS操作原子地增加计数，确保线程安全
    size_t expected = currentCount;
    while (!m_typeCounts[extension].compare_exchange_weak(expected, expected + 1)) {
        if (expected >= m_config.maxSamplesPerType) {
            m_skippedQuota++;
            return;
        }
    }

    // 根据模式分派处理
    if (m_config.mode == DatasetMode::REPAIR) {
        ProcessImageFile(filePath, extension);
    } else if (m_config.mode == DatasetMode::CONTINUITY) {
        // 连续性模式：处理 ZIP 文件
        ProcessZipFileForContinuity(filePath, extension);
    } else {
        // 分类模式：提取文件特征
        SampleInfo sample;
        sample.filePath = filePath;
        sample.extension = extension;

        if (ExtractFileFeatures(filePath, sample.features)) {
            sample.valid = true;

            // 将有效样本添加到样本列表
            {
                std::lock_guard<std::mutex> lock(m_samplesMutex);
                m_samples.push_back(std::move(sample));
            }
        } else {
            // 特征提取失败，回退计数器
            m_typeCounts[extension]--;
            m_skippedError++;
        }
    }
}

bool DatasetGenerator::ExtractFileFeatures(const std::wstring& filePath, FileFeatures& outFeatures) {
    try {
        // 以二进制模式打开文件
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // 读取指定大小的数据片段
        std::vector<uint8_t> buffer(m_config.fragmentSize);
        file.read(reinterpret_cast<char*>(buffer.data()), m_config.fragmentSize);
        std::streamsize bytesRead = file.gcount();

        // 检查读取的字节数是否足够
        if (bytesRead < (std::streamsize)(m_config.minFileSize / 2)) {
            return false;
        }

        // 调用MLClassifier的特征提取函数
        outFeatures = MLClassifier::extractFeatures(buffer.data(), static_cast<size_t>(bytesRead));
        return true;

    } catch (const std::exception&) {
        return false;
    }
}

// ============================================================================
// 导出方法
// ============================================================================

bool DatasetGenerator::ExportCSV(const std::string& outputPath) {
    if (m_samples.empty()) {
        LOG_WARNING("No samples to export");
        return false;
    }

    try {
        std::ofstream file(outputPath,ios::app);
        if (!file.is_open()) {
            LOG_ERROR_FMT("Cannot open file for writing: %s", outputPath.c_str());
            return false;
        }

        // 写入CSV表头
        for (size_t i = 0; i < FileFeatures::FEATURE_DIM; i++) {
            file << "f" << i;
            if (i < FileFeatures::FEATURE_DIM - 1) file << ",";
        }
        file << ",extension";
        if (m_config.includeFilePath) {
            file << ",file_path";
        }
        file << "\n";

        // 设置浮点数输出精度
        file << std::fixed << std::setprecision(8);

        // 写入所有样本数据
        std::lock_guard<std::mutex> lock(m_samplesMutex);
        for (const auto& sample : m_samples) {
            if (!sample.valid) continue;

            // 写入261维特征向量
            for (size_t i = 0; i < FileFeatures::FEATURE_DIM; i++) {
                file << sample.features.data[i];
                if (i < FileFeatures::FEATURE_DIM - 1) file << ",";
            }

            // 写入文件类型标签
            file << "," << sample.extension;

            // 可选：写入文件路径
            if (m_config.includeFilePath) {
                // 将宽字符路径转换为UTF-8编码
                std::string utf8Path = WideToUtf8(sample.filePath);
                // 对包含逗号或引号的路径进行转义处理
                if (utf8Path.find(',') != std::string::npos ||
                    utf8Path.find('"') != std::string::npos) {
                    // 用双引号包围字符串，并将内部的引号转义为双引号
                    std::string escaped;
                    escaped.push_back('"');
                    for (char c : utf8Path) {
                        if (c == '"') escaped.push_back('"');
                        escaped.push_back(c);
                    }
                    escaped.push_back('"');
                    utf8Path = escaped;
                }
                file << "," << utf8Path;
            }

            file << "\n";
        }

        file.close();

        LOG_INFO_FMT("Exported %zu samples to CSV: %s", m_samples.size(), outputPath.c_str());
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error exporting CSV: %s", e.what());
        return false;
    }
}

bool DatasetGenerator::ExportBinary(const std::string& outputPath) {
    if (m_samples.empty()) {
        LOG_WARNING("No samples to export");
        return false;
    }

    try {
        std::ofstream file(outputPath, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR_FMT("Cannot open file for writing: %s", outputPath.c_str());
            return false;
        }

        // 写入二进制文件头部
        BinaryDatasetHeader header;
        header.sampleCount = static_cast<uint32_t>(m_samples.size());
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // 写入所有样本数据
        std::lock_guard<std::mutex> lock(m_samplesMutex);
        for (const auto& sample : m_samples) {
            if (!sample.valid) continue;

            // 写入特征向量 (261维 * float32)
            file.write(reinterpret_cast<const char*>(sample.features.data.data()),
                      FileFeatures::FEATURE_DIM * sizeof(float));

            // 写入扩展名：先写长度(1字节)，再写内容
            uint8_t extLen = static_cast<uint8_t>(sample.extension.length());
            file.write(reinterpret_cast<const char*>(&extLen), 1);
            file.write(sample.extension.c_str(), extLen);

            // 可选：写入文件路径
            if (m_config.includeFilePath) {
                std::string utf8Path = WideToUtf8(sample.filePath);
                // 路径长度限制为65535字节
                uint16_t pathLen = static_cast<uint16_t>(
                    (std::min)(utf8Path.length(), (size_t)65535));
                file.write(reinterpret_cast<const char*>(&pathLen), 2);
                file.write(utf8Path.c_str(), pathLen);
            }
        }

        file.close();

        LOG_INFO_FMT("Exported %zu samples to binary: %s", m_samples.size(), outputPath.c_str());
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error exporting binary: %s", e.what());
        return false;
    }
}

// ============================================================================
// 状态查询
// ============================================================================

DatasetStats DatasetGenerator::GetStats() const {
    DatasetStats stats;

    // 统计每个文件类型的样本数量
    for (const auto& [type, count] : m_typeCounts) {
        stats.typeCounts[type] = count.load();
        stats.totalSamples += count.load();
    }

    stats.totalFilesScanned = m_totalScanned.load();
    stats.skippedTooSmall = m_skippedSmall.load();
    stats.skippedReadError = m_skippedError.load();
    stats.skippedQuotaReached = m_skippedQuota.load();

    // 修复模式统计
    stats.normalSamples = m_normalSamples.load();
    stats.damagedSamples = m_damagedSamples.load();
    stats.repairableSamples = m_repairableSamples.load();
    stats.unrepairableSamples = m_unrepairableSamples.load();
    //连续模式统计
    stats.positiveSamples = m_positiveSamples.load();
    stats.negativeSamples = m_negativeSamples.load();

    for (const auto& [damageType, count] : m_damageTypeCounts) {
        stats.damageTypeCounts[damageType] = count.load();
    }

    // 计算已用时间
    LARGE_INTEGER currentTime;
    QueryPerformanceCounter(&currentTime);
    stats.elapsedSeconds = (double)(currentTime.QuadPart - m_startTime.QuadPart) / m_frequency.QuadPart;

    return stats;
}

double DatasetGenerator::GetProgress() const {
    size_t total = m_totalFiles.load();
    if (total == 0) return 0.0;
    return 100.0 * m_processedFiles.load() / total;
}

void DatasetGenerator::Clear() {
    // 清空分类样本列表
    {
        std::lock_guard<std::mutex> lock(m_samplesMutex);
        m_samples.clear();
    }

    // 清空修复样本列表
    {
        std::lock_guard<std::mutex> lock(m_repairSamplesMutex);
        m_repairSamples.clear();
    }

    // 重置所有类型的计数器
    for (auto& [type, count] : m_typeCounts) {
        count = 0;
    }

    // 重置统计计数器
    m_totalScanned = 0;
    m_skippedSmall = 0;
    m_skippedError = 0;
    m_skippedQuota = 0;

    // 重置修复模式计数器
    m_normalSamples = 0;
    m_damagedSamples = 0;
    m_repairableSamples = 0;
    m_unrepairableSamples = 0;

    for (auto& [damageType, count] : m_damageTypeCounts) {
        count = 0;
    }
}

// ============================================================================
// 辅助方法
// ============================================================================

// 检查给定扩展名是否在目标类型集合中
bool DatasetGenerator::ShouldProcessType(const std::string& ext) const {
    return m_config.targetTypes.find(ext) != m_config.targetTypes.end();
}

// 从文件路径中提取小写扩展名
std::string DatasetGenerator::GetLowerExtension(const std::wstring& path) {
    // 查找最后一个点号的位置
    size_t dotPos = path.rfind(L'.');
    if (dotPos == std::wstring::npos || dotPos == path.length() - 1) {
        return "";
    }

    // 提取扩展名并转换为小写ASCII字符串
    std::wstring wext = path.substr(dotPos + 1);
    std::string ext;
    ext.reserve(wext.length());

    for (wchar_t wc : wext) {
        if (wc < 128) {
            ext.push_back(static_cast<char>(tolower(static_cast<int>(wc))));
        }
    }

    return ext;
}

// 报告当前处理进度
void DatasetGenerator::ReportProgress(const std::string& status) {
    // 调用进度回调函数（如果已设置）
    if (m_progressCallback) {
        m_progressCallback(m_processedFiles.load(), m_totalFiles.load(), status);
    }

    // 详细模式下输出日志
    if (m_config.verbose) {
        double progress = GetProgress();
        LOG_INFO_FMT("[%.1f%%] %s", progress, status.c_str());
    }
}

// ============================================================================
// 修复模式实现
// ============================================================================

void DatasetGenerator::ProcessImageFile(const std::wstring& filePath, const std::string& extension) {
    try {
        // 读取整个文件（修复模式需要更多数据）
        std::ifstream file(filePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            m_typeCounts[extension]--;
            m_skippedError++;
            return;
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        // 限制读取大小（最多 64KB 用于特征提取）
        size_t readSize = (std::min)(fileSize, (size_t)(64 * 1024));
        std::vector<uint8_t> originalData(readSize);
        file.read(reinterpret_cast<char*>(originalData.data()), readSize);

        if (file.gcount() < (std::streamsize)m_config.minFileSize) {
            m_typeCounts[extension]--;
            m_skippedError++;
            return;
        }

        // 确定图像类型
        std::string imageType;
        if (extension == "jpg" || extension == "jpeg") {
            imageType = "jpeg";
        } else if (extension == "png") {
            imageType = "png";
        } else {
            m_typeCounts[extension]--;
            m_skippedError++;
            return;
        }

        // 验证是否为有效图像（检查魔数）
        bool isValidImage = false;
        if (imageType == "jpeg" && originalData.size() >= 2) {
            isValidImage = (originalData[0] == 0xFF && originalData[1] == 0xD8);
        } else if (imageType == "png" && originalData.size() >= 8) {
            const uint8_t png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
            isValidImage = (memcmp(originalData.data(), png_sig, 8) == 0);
        }

        if (!isValidImage) {
            m_typeCounts[extension]--;
            m_skippedError++;
            return;
        }

        // 创建 ImageHeaderRepairer 用于特征提取
        ImageHeaderRepairer repairer;

        // 决定是否生成损坏样本
        bool shouldDamage = false;
        {
            std::lock_guard<std::mutex> lock(m_rngMutex);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            shouldDamage = m_config.generateDamagedSamples && (dist(m_rng) < m_config.damageRatio);
        }

        if (shouldDamage) {
            // 生成损坏样本
            std::vector<uint8_t> damagedData = originalData;
            SimulatedDamageType damageType = GetRandomDamageType();
            float severity = GetRandomSeverity();
            size_t damageOffset = 0;
            size_t damageSize = 0;

            SimulateDamage(damagedData, damageType, severity, damageOffset, damageSize);

            // 提取损坏后的特征
            RepairSampleInfo sample;
            sample.filePath = filePath;
            sample.imageType = imageType;
            sample.damageType = damageType;
            sample.damageSeverity = severity;
            sample.damageOffset = damageOffset;
            sample.damageSize = damageSize;
            sample.fileSize = fileSize;

            if (ExtractRepairFeatures(damagedData, sample.features)) {
                // 评估是否可修复
                sample.isRepairable = EvaluateRepairability(originalData, damagedData, imageType);
                sample.valid = true;

                {
                    std::lock_guard<std::mutex> lock(m_repairSamplesMutex);
                    m_repairSamples.push_back(std::move(sample));
                }

                m_damagedSamples++;
                m_damageTypeCounts[static_cast<int>(damageType)]++;
                if (sample.isRepairable) {
                    m_repairableSamples++;
                } else {
                    m_unrepairableSamples++;
                }
            } else {
                m_typeCounts[extension]--;
                m_skippedError++;
            }
        } else {
            // 生成正常样本（无损坏）
            RepairSampleInfo sample;
            sample.filePath = filePath;
            sample.imageType = imageType;
            sample.damageType = SimulatedDamageType::NONE;
            sample.damageSeverity = 0.0f;
            sample.damageOffset = 0;
            sample.damageSize = 0;
            sample.isRepairable = true;
            sample.fileSize = fileSize;

            if (ExtractRepairFeatures(originalData, sample.features)) {
                sample.valid = true;

                {
                    std::lock_guard<std::mutex> lock(m_repairSamplesMutex);
                    m_repairSamples.push_back(std::move(sample));
                }

                m_normalSamples++;
                m_damageTypeCounts[static_cast<int>(SimulatedDamageType::NONE)]++;
            } else {
                m_typeCounts[extension]--;
                m_skippedError++;
            }
        }

    } catch (const std::exception&) {
        m_typeCounts[extension]--;
        m_skippedError++;
    }
}

bool DatasetGenerator::ExtractRepairFeatures(const std::vector<uint8_t>& data, ImageFeatureVector& outFeatures) {
    if (data.size() < 100) return false;

    // 使用 ImageHeaderRepairer 的特征提取
    ImageHeaderRepairer repairer;
    outFeatures = repairer.ExtractFullFeatures(data);
    return true;
}

void DatasetGenerator::SimulateDamage(std::vector<uint8_t>& data, SimulatedDamageType type,
                                       float severity, size_t& damageOffset, size_t& damageSize) {
    if (data.empty()) return;

    std::lock_guard<std::mutex> lock(m_rngMutex);

    // 计算损坏区域大小（基于严重程度）
    size_t maxDamage = (std::min)(m_config.maxDamageSize, data.size() / 4);
    damageSize = static_cast<size_t>(maxDamage * severity);
    if (damageSize < 1) damageSize = 1;

    switch (type) {
        case SimulatedDamageType::HEADER_ZEROED: {
            // 头部清零：将文件开头的字节设置为0
            damageOffset = 0;
            damageSize = (std::min)(damageSize, (size_t)64);  // 头部最多64字节
            std::fill(data.begin(), data.begin() + damageSize, 0);
            break;
        }

        case SimulatedDamageType::HEADER_RANDOM: {
            // 头部随机字节：用随机数据覆盖文件头部
            damageOffset = 0;
            damageSize = (std::min)(damageSize, (size_t)64);
            std::uniform_int_distribution<int> byteDist(0, 255);
            for (size_t i = 0; i < damageSize; i++) {
                data[i] = static_cast<uint8_t>(byteDist(m_rng));
            }
            break;
        }

        case SimulatedDamageType::PARTIAL_OVERWRITE: {
            // 部分覆盖：在文件中间某处用随机数据覆盖
            std::uniform_int_distribution<size_t> offsetDist(data.size() / 4, data.size() / 2);
            damageOffset = offsetDist(m_rng);
            damageSize = (std::min)(damageSize, data.size() - damageOffset);

            std::uniform_int_distribution<int> byteDist(0, 255);
            for (size_t i = 0; i < damageSize; i++) {
                data[damageOffset + i] = static_cast<uint8_t>(byteDist(m_rng));
            }
            break;
        }

        case SimulatedDamageType::TRUNCATED: {
            // 截断：模拟文件被截断（用零填充末尾）
            size_t truncPoint = static_cast<size_t>(data.size() * (1.0f - severity * 0.5f));
            truncPoint = (std::max)(truncPoint, data.size() / 2);
            damageOffset = truncPoint;
            damageSize = data.size() - truncPoint;
            std::fill(data.begin() + truncPoint, data.end(), 0);
            break;
        }

        case SimulatedDamageType::RANDOM_CORRUPTION: {
            // 随机损坏：在随机位置插入随机字节
            std::uniform_int_distribution<size_t> offsetDist(0, data.size() - damageSize);
            damageOffset = offsetDist(m_rng);

            std::uniform_int_distribution<int> byteDist(0, 255);
            // 随机翻转一些位
            for (size_t i = 0; i < damageSize; i++) {
                if (i % 3 == 0) {  // 每3个字节修改一个
                    data[damageOffset + i] = static_cast<uint8_t>(byteDist(m_rng));
                }
            }
            break;
        }

        default:
            damageOffset = 0;
            damageSize = 0;
            break;
    }
}

SimulatedDamageType DatasetGenerator::GetRandomDamageType() {
    std::lock_guard<std::mutex> lock(m_rngMutex);

    // 从启用的损坏类型中随机选择
    std::vector<SimulatedDamageType> enabledTypes(
        m_config.enabledDamageTypes.begin(),
        m_config.enabledDamageTypes.end()
    );

    if (enabledTypes.empty()) {
        return SimulatedDamageType::HEADER_ZEROED;
    }

    std::uniform_int_distribution<size_t> dist(0, enabledTypes.size() - 1);
    return enabledTypes[dist(m_rng)];
}

float DatasetGenerator::GetRandomSeverity() {
    std::lock_guard<std::mutex> lock(m_rngMutex);
    std::uniform_real_distribution<float> dist(m_config.minDamageSeverity, m_config.maxDamageSeverity);
    return dist(m_rng);
}

bool DatasetGenerator::EvaluateRepairability(const std::vector<uint8_t>& originalData,
                                              const std::vector<uint8_t>& damagedData,
                                              const std::string& imageType) {
    // 使用 ImageHeaderRepairer 评估是否可修复
    ImageHeaderRepairer repairer;

    // 尝试检测损坏后是否仍能识别图像类型
    if (imageType == "jpeg") {
        // JPEG: 检查是否能找到 SOF 标记
        return ImageHeaderRepairer::IsLikelyJPEG(damagedData);
    } else if (imageType == "png") {
        // PNG: 检查是否能找到 IDAT chunk
        return ImageHeaderRepairer::IsLikelyPNG(damagedData);
    }

    return false;
}

bool DatasetGenerator::ExportRepairCSV(const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(m_repairSamplesMutex);

    if (m_repairSamples.empty()) {
        LOG_WARNING("No repair samples to export");
        return false;
    }

    try {
        std::ofstream file(outputPath);
        if (!file.is_open()) {
            LOG_ERROR_FMT("Cannot open file for writing: %s", outputPath.c_str());
            return false;
        }

        // 写入CSV表头（31维特征 + 元数据）
        for (size_t i = 0; i < ImageFeatureVector::FEATURE_DIM; i++) {
            file << "f" << i << ",";
        }
        file << "image_type,damage_type,damage_severity,damage_offset,damage_size,is_repairable";
        if (m_config.includeFilePath) {
            file << ",file_path";
        }
        file << "\n";

        // 设置浮点数精度
        file << std::fixed << std::setprecision(8);

        // 写入样本数据
        for (const auto& sample : m_repairSamples) {
            if (!sample.valid) continue;

            // 写入31维特征
            std::vector<float> featureVec = sample.features.ToVector();
            for (size_t i = 0; i < featureVec.size(); i++) {
                file << featureVec[i] << ",";
            }

            // 写入元数据
            file << sample.imageType << ",";
            file << static_cast<int>(sample.damageType) << ",";
            file << sample.damageSeverity << ",";
            file << sample.damageOffset << ",";
            file << sample.damageSize << ",";
            file << (sample.isRepairable ? 1 : 0);

            // 可选：写入文件路径
            if (m_config.includeFilePath) {
                std::string utf8Path = WideToUtf8(sample.filePath);
                if (utf8Path.find(',') != std::string::npos || utf8Path.find('"') != std::string::npos) {
                    std::string escaped = "\"";
                    for (char c : utf8Path) {
                        if (c == '"') escaped += "\"";
                        escaped += c;
                    }
                    escaped += "\"";
                    utf8Path = escaped;
                }
                file << "," << utf8Path;
            }

            file << "\n";
        }

        file.close();

        LOG_INFO_FMT("Exported %zu repair samples to CSV: %s", m_repairSamples.size(), outputPath.c_str());
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error exporting repair CSV: %s", e.what());
        return false;
    }
}

size_t DatasetGenerator::GetRepairSampleCount() const {
    return m_repairSamples.size();
}

std::string DatasetGenerator::GetDamageTypeName(SimulatedDamageType type) {
    switch (type) {
        case SimulatedDamageType::NONE:              return "none";
        case SimulatedDamageType::HEADER_ZEROED:     return "header_zeroed";
        case SimulatedDamageType::HEADER_RANDOM:     return "header_random";
        case SimulatedDamageType::PARTIAL_OVERWRITE: return "partial_overwrite";
        case SimulatedDamageType::TRUNCATED:         return "truncated";
        case SimulatedDamageType::RANDOM_CORRUPTION: return "random_corruption";
        default:                                     return "unknown";
    }
}

// ============================================================================
// 连续性模式实现
// ============================================================================

std::string DatasetGenerator::GetSampleTypeName(ContinuitySampleType type) {
    switch (type) {
        case ContinuitySampleType::SAME_FILE:            return "same_file";
        case ContinuitySampleType::DIFFERENT_FILES:      return "different_files";
        case ContinuitySampleType::FILE_BOUNDARY:        return "file_boundary";
        case ContinuitySampleType::RANDOM_DATA:          return "random_data";
        case ContinuitySampleType::DIFFERENT_TYPE:       return "different_type";
        // 损坏样本类型
        case ContinuitySampleType::CORRUPTED_TRUNCATION:   return "corrupted_truncation";
        case ContinuitySampleType::CORRUPTED_BITFLIP:      return "corrupted_bitflip";
        case ContinuitySampleType::CORRUPTED_ZERO_FILL:    return "corrupted_zero_fill";
        case ContinuitySampleType::CORRUPTED_RANDOM_FILL:  return "corrupted_random_fill";
        case ContinuitySampleType::CORRUPTED_HEADER_DAMAGE:return "corrupted_header_damage";
        case ContinuitySampleType::CORRUPTED_PARTIAL:      return "corrupted_partial";
        default:                                         return "unknown";
    }
}

size_t DatasetGenerator::GetContinuitySampleCount() const {
    return m_continuitySamples.size();
}

size_t DatasetGenerator::CalculateAdaptiveSamplesPerFile(size_t fileSize) const {
    if (!m_config.useAdaptiveSampling) {
        return m_config.samplesPerFile;
    }

    // 计算可用块数（文件需要至少2个块才能生成1个样本）
    size_t numBlocks = fileSize / m_config.continuityBlockSize;
    if (numBlocks < 2) {
        return 0;
    }

    // 可用的块对数 = numBlocks - 1
    size_t availablePairs = numBlocks - 1;

    // 根据采样率计算样本数
    size_t samples = static_cast<size_t>(availablePairs * m_config.adaptiveSamplingRate);

    // 限制在最小和最大范围内
    samples = (std::max)(m_config.minSamplesPerFile, samples);
    samples = (std::min)(m_config.maxSamplesPerFile, samples);

    // 不能超过可用块对数
    samples = (std::min)(samples, availablePairs);

    return samples;
}

bool DatasetGenerator::ScanLocalZipFiles(const std::vector<std::wstring>& directories) {
    if (m_config.mode != DatasetMode::CONTINUITY) {
        LOG_ERROR("ScanLocalZipFiles requires CONTINUITY mode");
        return false;
    }

    return ScanDirectories(directories);
}

bool DatasetGenerator::GenerateContinuityDataset(const std::string& outputPath) {
    if (m_config.mode != DatasetMode::CONTINUITY) {
        LOG_ERROR("GenerateContinuityDataset requires CONTINUITY mode");
        return false;
    }

    // 增量模式：设置输出路径（在扫描前设置，以便 AppendToContinuityCSV 知道写入位置）
    if (m_config.incrementalMode) {
        m_progress.outputPath = outputPath;
    }

    // 扫描本地目录
    if (m_config.useLocalFiles && !m_config.localPaths.empty()) {
        ScanDirectories(m_config.localPaths);
    }

    // 导出数据集（非增量模式，或增量模式首次运行）
    if (!m_config.incrementalMode) {
        return ExportContinuityCSV(outputPath);
    }

    // 增量模式：数据已在 ScanDirectories 中通过 AppendToContinuityCSV 写入
    return true;
}

// 验证文件格式签名
static bool ValidateFileSignature(const std::vector<uint8_t>& data, const std::string& extension) {
    if (data.size() < 8) return false;

    // ZIP-based formats (ZIP, DOCX, XLSX, PPTX, JAR, APK)
    if (extension == "zip" || extension == "docx" || extension == "xlsx" ||
        extension == "pptx" || extension == "jar" || extension == "apk") {
        return data[0] == 0x50 && data[1] == 0x4B &&
               data[2] == 0x03 && data[3] == 0x04;
    }

    // MP3 (ID3v2 or frame sync)
    if (extension == "mp3" || extension == "mp2" || extension == "mp1") {
        // ID3v2 tag
        if (data[0] == 'I' && data[1] == 'D' && data[2] == '3') {
            return true;
        }
        // Frame sync (0xFF + 0xE0 mask)
        if (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
            return true;
        }
        return false;
    }

    // MP4/MOV/M4A/3GP (ftyp box or mdat)
    if (extension == "mp4" || extension == "mov" || extension == "m4a" ||
        extension == "m4v" || extension == "3gp") {
        // Check for ftyp box
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            return true;
        }
        // Check for moov or mdat at start (some files)
        if ((data[4] == 'm' && data[5] == 'o' && data[6] == 'o' && data[7] == 'v') ||
            (data[4] == 'm' && data[5] == 'd' && data[6] == 'a' && data[7] == 't') ||
            (data[4] == 'f' && data[5] == 'r' && data[6] == 'e' && data[7] == 'e') ||
            (data[4] == 'w' && data[5] == 'i' && data[6] == 'd' && data[7] == 'e')) {
            return true;
        }
        return false;
    }

    // WAV (RIFF...WAVE)
    if (extension == "wav") {
        return data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F';
    }

    // AVI (RIFF...AVI )
    if (extension == "avi") {
        return data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F';
    }

    // FLAC
    if (extension == "flac") {
        return data[0] == 'f' && data[1] == 'L' && data[2] == 'a' && data[3] == 'C';
    }

    // OGG
    if (extension == "ogg") {
        return data[0] == 'O' && data[1] == 'g' && data[2] == 'g' && data[3] == 'S';
    }

    // MKV/WebM (EBML)
    if (extension == "mkv" || extension == "webm") {
        return data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3;
    }

    // JPEG
    if (extension == "jpg" || extension == "jpeg") {
        return data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
    }

    // PNG
    if (extension == "png") {
        return data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G';
    }

    // GIF
    if (extension == "gif") {
        return data[0] == 'G' && data[1] == 'I' && data[2] == 'F' && data[3] == '8';
    }

    // BMP
    if (extension == "bmp") {
        return data[0] == 'B' && data[1] == 'M';
    }

    // Unknown format - accept based on extension being in target list
    return true;
}

void DatasetGenerator::ProcessZipFileForContinuity(const std::wstring& filePath,
                                                   const std::string& extension) {
    // 读取文件
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        m_skippedError++;
        return;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize < m_config.minFileSize) {
        m_skippedSmall++;
        return;
    }

    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> fileData(fileSize);
    if (!file.read(reinterpret_cast<char*>(fileData.data()), fileSize)) {
        m_skippedError++;
        return;
    }
    file.close();

    // 验证文件格式签名
    if (!ValidateFileSignature(fileData, extension)) {
        m_skippedError++;
        return;
    }

    m_totalScanned++;

    // 生成正样本（同一文件的连续块）
    GeneratePositiveSamples(filePath, fileData, extension);

    // 生成损坏样本（负样本）- 从正常连续块创建损坏的负样本
    if (m_config.generateCorruptedSamples) {
        GenerateCorruptedSamples(filePath, fileData, extension);
    }

    // 缓存文件用于生成负样本
    {
        std::lock_guard<std::mutex> lock(m_fileCacheMutex);
        if (m_zipFileCache.size() < 100) {  // 最多缓存100个文件
            m_zipFileCache.emplace_back(filePath, std::move(fileData));
        }

        // 当有足够的缓存文件时，生成负样本
        if (m_zipFileCache.size() >= 2) {
            // 随机选择两个不同的文件生成负样本
            std::uniform_int_distribution<size_t> dist(0, m_zipFileCache.size() - 1);
            size_t idx1 = dist(m_rng);
            size_t idx2;
            do {
                idx2 = dist(m_rng);
            } while (idx2 == idx1 && m_zipFileCache.size() > 1);

            if (idx1 != idx2) {
                GenerateNegativeSamples(
                    m_zipFileCache[idx1].first, m_zipFileCache[idx1].second,
                    m_zipFileCache[idx2].first, m_zipFileCache[idx2].second,
                    extension
                );
            }
        }
    }

    // 增量模式：标记文件为已处理
    if (m_config.incrementalMode) {
        MarkFileProcessed(filePath);
    }
}

void DatasetGenerator::GeneratePositiveSamples(const std::wstring& filePath,
                                               const std::vector<uint8_t>& fileData,
                                               const std::string& fileType) {
    const size_t blockSize = m_config.continuityBlockSize;
    const size_t minBlocks = 4;  // 文件至少需要4个块

    if (fileData.size() < blockSize * minBlocks) {
        return;
    }

    size_t numBlocks = fileData.size() / blockSize;

    // 使用自适应采样计算样本数
    size_t samplesToGenerate = CalculateAdaptiveSamplesPerFile(fileData.size());
    samplesToGenerate = (std::min)(samplesToGenerate, numBlocks - 1);

    std::uniform_int_distribution<size_t> blockDist(0, numBlocks - 2);

    for (size_t i = 0; i < samplesToGenerate && !m_stopFlag; i++) {
        size_t blockIdx = blockDist(m_rng);
        size_t offset1 = blockIdx * blockSize;
        size_t offset2 = (blockIdx + 1) * blockSize;

        if (offset2 + blockSize > fileData.size()) {
            continue;
        }

        // 提取特征
        auto features = Continuity::BlockContinuityDetector::ExtractFeatures(
            fileData.data() + offset1, blockSize,
            fileData.data() + offset2, blockSize,
            fileType
        );

        // 创建样本
        ContinuitySampleInfo sample;
        sample.file1Path = filePath;
        sample.file2Path = filePath;
        sample.fileType = fileType;
        sample.features = features;
        sample.sampleType = ContinuitySampleType::SAME_FILE;
        sample.isContinuous = true;
        sample.block1Offset = offset1;
        sample.block2Offset = offset2;
        sample.valid = true;

        // 添加到样本列表
        {
            std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);
            m_continuitySamples.push_back(sample);
        }
        m_positiveSamples++;
        m_sampleTypeCounts[static_cast<int>(ContinuitySampleType::SAME_FILE)]++;
    }
}

void DatasetGenerator::GenerateNegativeSamples(const std::wstring& file1Path,
                                               const std::vector<uint8_t>& file1Data,
                                               const std::wstring& file2Path,
                                               const std::vector<uint8_t>& file2Data,
                                               const std::string& fileType) {
    const size_t blockSize = m_config.continuityBlockSize;

    if (file1Data.size() < blockSize * 2 || file2Data.size() < blockSize * 2) {
        return;
    }

    size_t numBlocks1 = file1Data.size() / blockSize;
    size_t numBlocks2 = file2Data.size() / blockSize;

    std::uniform_int_distribution<size_t> dist1(0, numBlocks1 - 1);
    std::uniform_int_distribution<size_t> dist2(0, numBlocks2 - 1);

    // 生成几个负样本
    size_t negSamples = (std::min)(size_t(3), m_config.samplesPerFile / 3);

    for (size_t i = 0; i < negSamples && !m_stopFlag; i++) {
        size_t blockIdx1 = dist1(m_rng);
        size_t blockIdx2 = dist2(m_rng);

        size_t offset1 = blockIdx1 * blockSize;
        size_t offset2 = blockIdx2 * blockSize;

        if (offset1 + blockSize > file1Data.size() ||
            offset2 + blockSize > file2Data.size()) {
            continue;
        }

        // 提取特征
        auto features = Continuity::BlockContinuityDetector::ExtractFeatures(
            file1Data.data() + offset1, blockSize,
            file2Data.data() + offset2, blockSize,
            fileType
        );

        // 创建样本
        ContinuitySampleInfo sample;
        sample.file1Path = file1Path;
        sample.file2Path = file2Path;
        sample.fileType = fileType;
        sample.features = features;
        sample.sampleType = ContinuitySampleType::DIFFERENT_FILES;
        sample.isContinuous = false;
        sample.block1Offset = offset1;
        sample.block2Offset = offset2;
        sample.valid = true;

        // 添加到样本列表
        {
            std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);
            m_continuitySamples.push_back(sample);
        }
        m_negativeSamples++;
        m_sampleTypeCounts[static_cast<int>(ContinuitySampleType::DIFFERENT_FILES)]++;
    }
}

void DatasetGenerator::GenerateRandomNegativeSamples(const std::wstring& filePath,
                                                     const std::vector<uint8_t>& fileData,
                                                     const std::string& fileType) {
    const size_t blockSize = m_config.continuityBlockSize;

    if (fileData.size() < blockSize) {
        return;
    }

    size_t numBlocks = fileData.size() / blockSize;
    std::uniform_int_distribution<size_t> blockDist(0, numBlocks - 1);

    // 生成随机数据
    std::vector<uint8_t> randomData(blockSize);

    for (size_t i = 0; i < 2 && !m_stopFlag; i++) {
        // 生成随机字节
        for (size_t j = 0; j < blockSize; j++) {
            randomData[j] = static_cast<uint8_t>(m_rng() & 0xFF);
        }

        size_t blockIdx = blockDist(m_rng);
        size_t offset = blockIdx * blockSize;

        if (offset + blockSize > fileData.size()) {
            continue;
        }

        // 提取特征
        auto features = Continuity::BlockContinuityDetector::ExtractFeatures(
            fileData.data() + offset, blockSize,
            randomData.data(), blockSize,
            fileType
        );

        // 创建样本
        ContinuitySampleInfo sample;
        sample.file1Path = filePath;
        sample.file2Path = L"<random>";
        sample.fileType = fileType;
        sample.features = features;
        sample.sampleType = ContinuitySampleType::RANDOM_DATA;
        sample.isContinuous = false;
        sample.block1Offset = offset;
        sample.block2Offset = 0;
        sample.valid = true;

        // 添加到样本列表
        {
            std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);
            m_continuitySamples.push_back(sample);
        }
        m_negativeSamples++;
        m_sampleTypeCounts[static_cast<int>(ContinuitySampleType::RANDOM_DATA)]++;
    }
}

void DatasetGenerator::GenerateDifferentTypeNegativeSamples(
    const std::wstring& zipFilePath,
    const std::vector<uint8_t>& zipData,
    const std::wstring& otherFilePath,
    const std::vector<uint8_t>& otherData
) {
    const size_t blockSize = m_config.continuityBlockSize;

    if (zipData.size() < blockSize || otherData.size() < blockSize) {
        return;
    }

    size_t numZipBlocks = zipData.size() / blockSize;
    size_t numOtherBlocks = otherData.size() / blockSize;

    std::uniform_int_distribution<size_t> zipDist(0, numZipBlocks - 1);
    std::uniform_int_distribution<size_t> otherDist(0, numOtherBlocks - 1);

    size_t zipBlockIdx = zipDist(m_rng);
    size_t otherBlockIdx = otherDist(m_rng);

    size_t zipOffset = zipBlockIdx * blockSize;
    size_t otherOffset = otherBlockIdx * blockSize;

    if (zipOffset + blockSize > zipData.size() ||
        otherOffset + blockSize > otherData.size()) {
        return;
    }

    // 提取特征
    auto features = Continuity::BlockContinuityDetector::ExtractFeatures(
        zipData.data() + zipOffset, blockSize,
        otherData.data() + otherOffset, blockSize,
        "zip"
    );

    // 创建样本
    ContinuitySampleInfo sample;
    sample.file1Path = zipFilePath;
    sample.file2Path = otherFilePath;
    sample.fileType = "zip";
    sample.features = features;
    sample.sampleType = ContinuitySampleType::DIFFERENT_TYPE;
    sample.isContinuous = false;
    sample.block1Offset = zipOffset;
    sample.block2Offset = otherOffset;
    sample.valid = true;

    // 添加到样本列表
    {
        std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);
        m_continuitySamples.push_back(sample);
    }
    m_negativeSamples++;
    m_sampleTypeCounts[static_cast<int>(ContinuitySampleType::DIFFERENT_TYPE)]++;
}

ContinuitySampleType DatasetGenerator::GetRandomNegativeSampleType() {
    std::vector<ContinuitySampleType> enabledTypes;
    for (const auto& type : m_config.enabledSampleTypes) {
        if (type != ContinuitySampleType::SAME_FILE) {
            enabledTypes.push_back(type);
        }
    }

    if (enabledTypes.empty()) {
        return ContinuitySampleType::DIFFERENT_FILES;
    }

    std::uniform_int_distribution<size_t> dist(0, enabledTypes.size() - 1);
    return enabledTypes[dist(m_rng)];
}

// ============================================================================
// 损坏样本生成方法
// ============================================================================

void DatasetGenerator::GenerateCorruptedSamples(const std::wstring& filePath,
                                                 const std::vector<uint8_t>& fileData,
                                                 const std::string& fileType) {
    if (!m_config.generateCorruptedSamples) {
        return;
    }

    const size_t blockSize = m_config.continuityBlockSize;
    const size_t minBlocks = 4;

    if (fileData.size() < blockSize * minBlocks) {
        return;
    }

    size_t numBlocks = fileData.size() / blockSize;

    // 使用自适应采样计算基础样本数，再按损坏样本比例计算
    size_t baseSamples = CalculateAdaptiveSamplesPerFile(fileData.size());
    size_t corruptedSamplesToGenerate = static_cast<size_t>(
        baseSamples * m_config.corruptedSampleRatio);
    if (corruptedSamplesToGenerate < 1) {
        corruptedSamplesToGenerate = 1;
    }

    std::uniform_int_distribution<size_t> blockDist(0, numBlocks - 2);

    for (size_t i = 0; i < corruptedSamplesToGenerate && !m_stopFlag; i++) {
        // 选择一个连续的块对
        size_t blockIdx = blockDist(m_rng);
        size_t offset1 = blockIdx * blockSize;
        size_t offset2 = (blockIdx + 1) * blockSize;

        if (offset2 + blockSize > fileData.size()) {
            continue;
        }

        // 复制块2用于损坏
        std::vector<uint8_t> corruptedBlock2(
            fileData.begin() + offset2,
            fileData.begin() + offset2 + blockSize
        );

        // 选择损坏类型
        ContinuitySampleType corruptionType = GetRandomCorruptionType();
        float severity = GetRandomCorruptionSeverity();
        size_t corruptionOffset = 0;
        size_t corruptionSize = 0;

        // 应用损坏
        switch (corruptionType) {
            case ContinuitySampleType::CORRUPTED_TRUNCATION:
                ApplyTruncationCorruption(corruptedBlock2, severity,
                                         corruptionOffset, corruptionSize);
                break;

            case ContinuitySampleType::CORRUPTED_BITFLIP:
                ApplyBitFlipCorruption(corruptedBlock2, severity,
                                      corruptionOffset, corruptionSize);
                break;

            case ContinuitySampleType::CORRUPTED_ZERO_FILL:
                ApplyZeroFillCorruption(corruptedBlock2, severity,
                                       corruptionOffset, corruptionSize);
                break;

            case ContinuitySampleType::CORRUPTED_RANDOM_FILL:
                ApplyRandomFillCorruption(corruptedBlock2, severity,
                                         corruptionOffset, corruptionSize);
                break;

            case ContinuitySampleType::CORRUPTED_PARTIAL:
                // 使用文件中其他位置的数据作为覆盖源
                if (numBlocks > 3) {
                    std::uniform_int_distribution<size_t> otherBlockDist(0, numBlocks - 1);
                    size_t otherBlockIdx = otherBlockDist(m_rng);
                    while (otherBlockIdx == blockIdx || otherBlockIdx == blockIdx + 1) {
                        otherBlockIdx = otherBlockDist(m_rng);
                    }
                    size_t otherOffset = otherBlockIdx * blockSize;
                    std::vector<uint8_t> otherData(
                        fileData.begin() + otherOffset,
                        fileData.begin() + otherOffset + blockSize
                    );
                    ApplyPartialOverwriteCorruption(corruptedBlock2, otherData, severity,
                                                   corruptionOffset, corruptionSize);
                } else {
                    // 如果块数不够，使用随机填充
                    ApplyRandomFillCorruption(corruptedBlock2, severity,
                                             corruptionOffset, corruptionSize);
                    corruptionType = ContinuitySampleType::CORRUPTED_RANDOM_FILL;
                }
                break;

            default:
                continue;
        }

        // 提取损坏后的特征
        auto features = Continuity::BlockContinuityDetector::ExtractFeatures(
            fileData.data() + offset1, blockSize,
            corruptedBlock2.data(), blockSize,
            fileType
        );

        // 创建样本（损坏样本标记为不连续）
        ContinuitySampleInfo sample;
        sample.file1Path = filePath;
        sample.file2Path = filePath;
        sample.fileType = fileType;
        sample.features = features;
        sample.sampleType = corruptionType;
        sample.isContinuous = false;  // 损坏导致不连续
        sample.block1Offset = offset1;
        sample.block2Offset = offset2;
        sample.corruptionSeverity = severity;
        sample.corruptionOffset = corruptionOffset;
        sample.corruptionSize = corruptionSize;
        sample.valid = true;

        // 添加到样本列表
        {
            std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);
            m_continuitySamples.push_back(sample);
        }
        m_negativeSamples++;
        m_sampleTypeCounts[static_cast<int>(corruptionType)]++;
    }
}

void DatasetGenerator::ApplyTruncationCorruption(std::vector<uint8_t>& block,
                                                  float severity,
                                                  size_t& corruptionOffset,
                                                  size_t& corruptionSize) {
    // severity 决定截断位置：severity越高，截断越靠前
    // severity = 0.2: 截断后80%
    // severity = 0.8: 截断后20%
    size_t truncatePoint = static_cast<size_t>(block.size() * (1.0f - severity));
    truncatePoint = (std::max)(truncatePoint, size_t(512));  // 至少保留512字节
    truncatePoint = (std::min)(truncatePoint, block.size() - 512);  // 至少截断512字节

    corruptionOffset = truncatePoint;
    corruptionSize = block.size() - truncatePoint;

    // 截断部分填充零
    std::fill(block.begin() + truncatePoint, block.end(), 0x00);
}

void DatasetGenerator::ApplyBitFlipCorruption(std::vector<uint8_t>& block,
                                               float severity,
                                               size_t& corruptionOffset,
                                               size_t& corruptionSize) {
    // severity 决定翻转的比特数量
    // 基础翻转率 + severity调整
    float flipRate = m_config.bitFlipRate * (0.5f + severity);
    size_t totalBits = block.size() * 8;
    size_t bitsToFlip = static_cast<size_t>(totalBits * flipRate);
    bitsToFlip = (std::max)(bitsToFlip, size_t(1));
    bitsToFlip = (std::min)(bitsToFlip, totalBits / 10);  // 最多翻转10%

    std::uniform_int_distribution<size_t> byteDist(0, block.size() - 1);
    std::uniform_int_distribution<int> bitDist(0, 7);

    size_t minOffset = block.size();
    size_t maxOffset = 0;

    for (size_t i = 0; i < bitsToFlip; i++) {
        size_t byteIdx = byteDist(m_rng);
        int bitIdx = bitDist(m_rng);
        block[byteIdx] ^= (1 << bitIdx);

        minOffset = (std::min)(minOffset, byteIdx);
        maxOffset = (std::max)(maxOffset, byteIdx);
    }

    corruptionOffset = minOffset;
    corruptionSize = maxOffset - minOffset + 1;
}

void DatasetGenerator::ApplyZeroFillCorruption(std::vector<uint8_t>& block,
                                                float severity,
                                                size_t& corruptionOffset,
                                                size_t& corruptionSize) {
    // severity 决定清零区域大小
    size_t maxFillSize = static_cast<size_t>(block.size() * severity * 0.5f);
    maxFillSize = (std::max)(maxFillSize, size_t(512));
    maxFillSize = (std::min)(maxFillSize, block.size() / 2);

    std::uniform_int_distribution<size_t> sizeDist(256, maxFillSize);
    size_t fillSize = sizeDist(m_rng);

    std::uniform_int_distribution<size_t> offsetDist(0, block.size() - fillSize);
    size_t fillOffset = offsetDist(m_rng);

    std::fill(block.begin() + fillOffset,
              block.begin() + fillOffset + fillSize,
              0x00);

    corruptionOffset = fillOffset;
    corruptionSize = fillSize;
}

void DatasetGenerator::ApplyRandomFillCorruption(std::vector<uint8_t>& block,
                                                  float severity,
                                                  size_t& corruptionOffset,
                                                  size_t& corruptionSize) {
    // severity 决定随机填充区域大小
    size_t maxFillSize = static_cast<size_t>(block.size() * severity * 0.5f);
    maxFillSize = (std::max)(maxFillSize, size_t(512));
    maxFillSize = (std::min)(maxFillSize, block.size() / 2);

    std::uniform_int_distribution<size_t> sizeDist(256, maxFillSize);
    size_t fillSize = sizeDist(m_rng);

    std::uniform_int_distribution<size_t> offsetDist(0, block.size() - fillSize);
    size_t fillOffset = offsetDist(m_rng);

    // 填充随机数据
    for (size_t i = 0; i < fillSize; i++) {
        block[fillOffset + i] = static_cast<uint8_t>(m_rng() & 0xFF);
    }

    corruptionOffset = fillOffset;
    corruptionSize = fillSize;
}

void DatasetGenerator::ApplyPartialOverwriteCorruption(std::vector<uint8_t>& block,
                                                        const std::vector<uint8_t>& otherData,
                                                        float severity,
                                                        size_t& corruptionOffset,
                                                        size_t& corruptionSize) {
    if (otherData.empty()) {
        ApplyRandomFillCorruption(block, severity, corruptionOffset, corruptionSize);
        return;
    }

    // severity 决定覆盖区域大小
    size_t maxOverwriteSize = static_cast<size_t>(block.size() * severity * 0.5f);
    maxOverwriteSize = (std::max)(maxOverwriteSize, size_t(512));
    maxOverwriteSize = (std::min)(maxOverwriteSize, block.size() / 2);
    maxOverwriteSize = (std::min)(maxOverwriteSize, otherData.size());

    std::uniform_int_distribution<size_t> sizeDist(256, maxOverwriteSize);
    size_t overwriteSize = sizeDist(m_rng);

    std::uniform_int_distribution<size_t> destOffsetDist(0, block.size() - overwriteSize);
    size_t destOffset = destOffsetDist(m_rng);

    std::uniform_int_distribution<size_t> srcOffsetDist(0, otherData.size() - overwriteSize);
    size_t srcOffset = srcOffsetDist(m_rng);

    // 复制其他文件的数据
    std::copy(otherData.begin() + srcOffset,
              otherData.begin() + srcOffset + overwriteSize,
              block.begin() + destOffset);

    corruptionOffset = destOffset;
    corruptionSize = overwriteSize;
}

ContinuitySampleType DatasetGenerator::GetRandomCorruptionType() {
    std::vector<ContinuitySampleType> enabledTypes;
    for (const auto& type : m_config.enabledCorruptionTypes) {
        enabledTypes.push_back(type);
    }

    if (enabledTypes.empty()) {
        return ContinuitySampleType::CORRUPTED_RANDOM_FILL;
    }

    std::uniform_int_distribution<size_t> dist(0, enabledTypes.size() - 1);
    return enabledTypes[dist(m_rng)];
}

float DatasetGenerator::GetRandomCorruptionSeverity() {
    std::uniform_real_distribution<float> dist(
        m_config.minCorruptionSeverity,
        m_config.maxCorruptionSeverity
    );
    return dist(m_rng);
}

bool DatasetGenerator::ExportContinuityCSV(const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);

    if (m_continuitySamples.empty()) {
        LOG_WARNING("No continuity samples to export");
        return false;
    }
    LARGE_INTEGER filesize;
    ZeroMemory(&filesize,sizeof(filesize));
	HANDLE hfile=CreateFileA(outputPath.c_str(),
        GENERIC_READ, 
        FILE_SHARE_READ,
        NULL, 
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);
    if (hfile==INVALID_HANDLE_VALUE) {
		LOG_ERROR_FMT("Failed to open output file for size check: %s", outputPath.c_str(),",Error Code:",GetLastError());
    }
    if (!GetFileSizeEx(hfile, &filesize)) {
        LOG_ERROR_FMT("Failed to get output file size: %s", outputPath.c_str(), ",Error Code:", GetLastError());
        CloseHandle(hfile);
    }
    else {
		LOG_DEBUG_FMT("Output file size: %lld bytes", filesize.QuadPart);
		CloseHandle(hfile);
    }
    std::ofstream file(outputPath,ios::app);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open output file: " + outputPath);
        return false;
    }
	// 写入头部,仅当文件为空时写入
    if (filesize.QuadPart==0) {
        for (size_t i = 0; i < Continuity::ContinuityFeatures::FEATURE_DIM; i++) {
            file << "f" << i << ",";
        }
        file << "is_continuous,file_type,sample_type";
        if (m_config.includeFilePath) {
            file << ",file1_path,file2_path";
        }
        file << "\n";
    }
    // 写入样本
    for (const auto& sample : m_continuitySamples) {
        if (!sample.valid) continue;

        // 写入64维特征
        for (size_t i = 0; i < Continuity::ContinuityFeatures::FEATURE_DIM; i++) {
            file << std::fixed << std::setprecision(6) << sample.features.data[i];
            file << ",";
        }

        // 写入标签
        file << (sample.isContinuous ? 1 : 0) << ",";
        file << sample.fileType << ",";
        file << GetSampleTypeName(sample.sampleType);

        if (m_config.includeFilePath) {
            file << "," << WideToUtf8(sample.file1Path);
            file << "," << WideToUtf8(sample.file2Path);
        }
        file << "\n";
    }

    file.close();
    LOG_INFO_FMT("Exported %zu continuity samples to %s",
             m_continuitySamples.size(), outputPath.c_str());
    LOG_INFO_FMT("  Positive samples: %zu", m_positiveSamples.load());
    LOG_INFO_FMT("  Negative samples: %zu", m_negativeSamples.load());
    return true;
}

// ============================================================================
// 增量处理实现
// ============================================================================

using json = nlohmann::json;

std::string DatasetGenerator::GetDefaultProgressPath(const std::string& outputCsvPath) {
    // 将 output.csv 转换为 output.progress.json
    fs::path csvPath(outputCsvPath);
    fs::path progressPath = csvPath.parent_path() / (csvPath.stem().string() + ".progress.json");
    return progressPath.string();
}

bool DatasetGenerator::LoadProgress(const std::string& progressPath) {
    std::lock_guard<std::mutex> lock(m_progressMutex);

    if (!fs::exists(progressPath)) {
        LOG_INFO_FMT("No existing progress file found at: %s", progressPath.c_str());
        return false;
    }

    try {
        std::ifstream file(progressPath);
        if (!file.is_open()) {
            LOG_ERROR_FMT("Cannot open progress file: %s", progressPath.c_str());
            return false;
        }

        json j;
        file >> j;
        file.close();

        // 解析进度信息
        m_progress.sessionId = j.value("sessionId", "");
        m_progress.outputPath = j.value("outputPath", "");
        m_progress.mode = static_cast<DatasetMode>(j.value("mode", 2));  // 默认CONTINUITY
        m_progress.totalSamplesWritten = j.value("totalSamplesWritten", 0);
        m_progress.totalFilesProcessed = j.value("totalFilesProcessed", 0);
        m_progress.positiveSamples = j.value("positiveSamples", 0);
        m_progress.negativeSamples = j.value("negativeSamples", 0);
        m_progress.lastUpdateTime = j.value("lastUpdateTime", "");
        m_progress.csvHeaderWritten = j.value("csvHeaderWritten", false);

        // 加载已处理文件列表
        m_progress.processedFiles.clear();
        if (j.contains("processedFiles") && j["processedFiles"].is_array()) {
            for (const auto& path : j["processedFiles"]) {
                m_progress.processedFiles.insert(path.get<std::string>());
            }
        }

        // 加载类型计数
        m_progress.typeCounts.clear();
        if (j.contains("typeCounts") && j["typeCounts"].is_object()) {
            for (auto& [key, value] : j["typeCounts"].items()) {
                m_progress.typeCounts[key] = value.get<size_t>();
            }
        }

        // 恢复原子计数器
        m_positiveSamples = m_progress.positiveSamples;
        m_negativeSamples = m_progress.negativeSamples;

        // 恢复类型计数
        for (const auto& [type, count] : m_progress.typeCounts) {
            if (m_typeCounts.find(type) != m_typeCounts.end()) {
                m_typeCounts[type] = count;
            }
        }

        LOG_INFO_FMT("Loaded progress: %zu files processed, %zu samples written",
                     m_progress.totalFilesProcessed, m_progress.totalSamplesWritten);
        LOG_INFO_FMT("  Positive: %zu, Negative: %zu",
                     m_progress.positiveSamples, m_progress.negativeSamples);

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error loading progress: %s", e.what());
        return false;
    }
}

bool DatasetGenerator::SaveProgress(const std::string& progressPath) {
    std::lock_guard<std::mutex> lock(m_progressMutex);

    try {
        // 更新进度信息
        m_progress.positiveSamples = m_positiveSamples.load();
        m_progress.negativeSamples = m_negativeSamples.load();
        m_progress.totalSamplesWritten = m_progress.positiveSamples + m_progress.negativeSamples;

        // 更新类型计数
        for (const auto& [type, count] : m_typeCounts) {
            m_progress.typeCounts[type] = count.load();
        }

        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_s(&tm_buf, &time_t);
        char timeStr[32];
        strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &tm_buf);
        m_progress.lastUpdateTime = timeStr;

        // 构建JSON对象
        json j;
        j["sessionId"] = m_progress.sessionId;
        j["outputPath"] = m_progress.outputPath;
        j["mode"] = static_cast<int>(m_progress.mode);
        j["totalSamplesWritten"] = m_progress.totalSamplesWritten;
        j["totalFilesProcessed"] = m_progress.totalFilesProcessed;
        j["positiveSamples"] = m_progress.positiveSamples;
        j["negativeSamples"] = m_progress.negativeSamples;
        j["lastUpdateTime"] = m_progress.lastUpdateTime;
        j["csvHeaderWritten"] = m_progress.csvHeaderWritten;

        // 保存已处理文件列表
        j["processedFiles"] = json::array();
        for (const auto& path : m_progress.processedFiles) {
            j["processedFiles"].push_back(path);
        }

        // 保存类型计数
        j["typeCounts"] = json::object();
        for (const auto& [type, count] : m_progress.typeCounts) {
            j["typeCounts"][type] = count;
        }

        // 写入文件
        std::ofstream file(progressPath);
        if (!file.is_open()) {
            LOG_ERROR_FMT("Cannot create progress file: %s", progressPath.c_str());
            return false;
        }

        file << j.dump(2);  // 缩进2空格，便于阅读
        file.close();

        LOG_INFO_FMT("Progress saved: %zu files, %zu samples",
                     m_progress.totalFilesProcessed, m_progress.totalSamplesWritten);

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error saving progress: %s", e.what());
        return false;
    }
}

bool DatasetGenerator::IsFileProcessed(const std::wstring& filePath) const {
    std::string utf8Path = WideToUtf8(filePath);
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_progressMutex));
    return m_progress.processedFiles.find(utf8Path) != m_progress.processedFiles.end();
}

void DatasetGenerator::MarkFileProcessed(const std::wstring& filePath) {
    std::string utf8Path = WideToUtf8(filePath);
    std::lock_guard<std::mutex> lock(m_progressMutex);
    m_progress.processedFiles.insert(utf8Path);
    m_progress.totalFilesProcessed = m_progress.processedFiles.size();
}

bool DatasetGenerator::AppendToContinuityCSV(const std::string& outputPath) {
    std::lock_guard<std::mutex> lock(m_continuitySamplesMutex);

    if (m_continuitySamples.empty()) {
        LOG_INFO("No new samples to append");
        return true;
    }

    try {
        // 检查文件是否存在，决定是否写入头部
        bool writeHeader = !m_progress.csvHeaderWritten;
        if (!writeHeader && !fs::exists(outputPath)) {
            writeHeader = true;
        }

        // 以追加模式打开文件
        std::ios_base::openmode mode = std::ios::out;
        if (!writeHeader) {
            mode |= std::ios::app;
        }

        std::ofstream file(outputPath, mode);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open output file: " + outputPath);
            return false;
        }

        // 写入头部（如果需要）
        if (writeHeader) {
            for (size_t i = 0; i < Continuity::ContinuityFeatures::FEATURE_DIM; i++) {
                file << "f" << i << ",";
            }
            file << "is_continuous,file_type,sample_type";
            if (m_config.includeFilePath) {
                file << ",file1_path,file2_path";
            }
            file << "\n";
            m_progress.csvHeaderWritten = true;
        }

        // 追加样本
        size_t appendedCount = 0;
        for (const auto& sample : m_continuitySamples) {
            if (!sample.valid) continue;

            // 写入64维特征
            for (size_t i = 0; i < Continuity::ContinuityFeatures::FEATURE_DIM; i++) {
                file << std::fixed << std::setprecision(6) << sample.features.data[i];
                file << ",";
            }

            // 写入标签
            file << (sample.isContinuous ? 1 : 0) << ",";
            file << sample.fileType << ",";
            file << GetSampleTypeName(sample.sampleType);

            if (m_config.includeFilePath) {
                file << "," << WideToUtf8(sample.file1Path);
                file << "," << WideToUtf8(sample.file2Path);
            }
            file << "\n";
            appendedCount++;
        }

        file.close();

        // 更新进度
        m_progress.totalSamplesWritten += appendedCount;

        LOG_INFO_FMT("Appended %zu samples to %s (total: %zu)",
                     appendedCount, outputPath.c_str(), m_progress.totalSamplesWritten);

        // 清空已写入的样本，释放内存
        m_continuitySamples.clear();

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error appending to CSV: %s", e.what());
        return false;
    }
}

} // namespace ML
