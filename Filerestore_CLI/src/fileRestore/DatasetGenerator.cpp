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
{
    QueryPerformanceFrequency(&m_frequency);

    // 初始化每个文件类型的计数器
    for (const auto& type : m_config.targetTypes) {
        m_typeCounts[type] = 0;
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

    m_running = true;
    m_stopFlag = false;
    QueryPerformanceCounter(&m_startTime);

    // 重置统计
    m_totalScanned = 0;
    m_skippedSmall = 0;
    m_skippedError = 0;
    m_skippedQuota = 0;
    m_totalFiles = 0;
    m_processedFiles = 0;

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

        if (m_stopFlag.load()) {
            m_running = false;
            return false;
        }

        m_totalFiles = candidateFiles.size();
        LOG_INFO_FMT("Found %zu candidate files", candidateFiles.size());

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
        LOG_INFO_FMT("Total samples collected: %zu", m_samples.size());

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("Error during scan: %s", e.what());
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

    // 提取文件特征
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
        std::ofstream file(outputPath);
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
    // 清空样本列表
    std::lock_guard<std::mutex> lock(m_samplesMutex);
    m_samples.clear();

    // 重置所有类型的计数器
    for (auto& [type, count] : m_typeCounts) {
        count = 0;
    }

    // 重置统计计数器
    m_totalScanned = 0;
    m_skippedSmall = 0;
    m_skippedError = 0;
    m_skippedQuota = 0;
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

} // namespace ML
