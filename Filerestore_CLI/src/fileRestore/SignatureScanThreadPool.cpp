#include "SignatureScanThreadPool.h"
#include "FileFormatUtils.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

using namespace std;
using namespace std::chrono;

// ============================================================================
// 构造函数
// ============================================================================
SignatureScanThreadPool::SignatureScanThreadPool(
    const unordered_map<BYTE, vector<const FileSignature*>>* sigIndex,
    const set<string>* activeSigs,
    const ScanThreadPoolConfig& cfg)
    : signatureIndex(sigIndex)
    , activeSignatures(activeSigs)
    , config(cfg)
    , stopFlag(false)
    , pauseFlag(false)
    , completedTasks(0)
    , totalTasks(0)
    , totalFilesFound(0)
    , totalBytesScanned(0)
    , mlClassifier(nullptr)
    , mlEnhancedCount(0)
    , mlMatchCount(0)
    , mlMismatchCount(0)
    , mlSkippedCount(0)
    , mlUnknownCount(0)
    , useSimdScan(false)
    , simdLevel_(CpuFeatures::SimdLevel::SCALAR)
{
    // 如果启用自动检测，计算最优线程数
    if (config.autoDetectThreads) {
        config.workerCount = GetOptimalThreadCount();
    }

    // 初始化流式SIMD扫描所需的预计算数据
    simdLevel_ = CpuFeatures::Instance().GetBestSimdLevel();
    if (signatureIndex) {
        // 收集目标首字节
        for (const auto& pair : *signatureIndex) {
            simdTargetBytes_.push_back(pair.first);
        }
        // 构建256-bit快速查找表
        for (BYTE b : simdTargetBytes_) {
            targetByteBitmap_[b / 64] |= (1ULL << (b % 64));
        }
    }

    LOG_INFO_FMT("SignatureScanThreadPool created with %d workers, chunk size: %zu MB",
                 config.workerCount, config.chunkSize / (1024 * 1024));
}

// ============================================================================
// 析构函数
// ============================================================================
SignatureScanThreadPool::~SignatureScanThreadPool() {
    Stop();
}

// ============================================================================
// 启动线程池
// ============================================================================
void SignatureScanThreadPool::Start() {
    if (!workers.empty()) {
        LOG_WARNING("Thread pool already started");
        return;
    }

    stopFlag = false;
    pauseFlag = false;
    completedTasks = 0;
    totalTasks = 0;

    LOG_INFO_FMT("Starting thread pool with %d workers", config.workerCount);

    for (int i = 0; i < config.workerCount; ++i) {
        workers.emplace_back(&SignatureScanThreadPool::WorkerFunction, this);
    }

    LOG_INFO("Thread pool started successfully");
}

// ============================================================================
// 停止线程池
// ============================================================================
void SignatureScanThreadPool::Stop() {
    if (stopFlag.load()) return;

    LOG_DEBUG("Stopping thread pool...");

    stopFlag = true;
    pauseFlag = false;

    // 唤醒所有等待的线程
    taskAvailable.notify_all();
    queueNotFull.notify_all();

    // 等待所有线程完成
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers.clear();
    LOG_DEBUG("Thread pool stopped");
}

// ============================================================================
// 暂停/恢复
// ============================================================================
void SignatureScanThreadPool::Pause() {
    pauseFlag = true;
    LOG_DEBUG("Thread pool paused");
}

void SignatureScanThreadPool::Resume() {
    pauseFlag = false;
    taskAvailable.notify_all();
    LOG_DEBUG("Thread pool resumed");
}

// ============================================================================
// 工作线程主函数
// ============================================================================
void SignatureScanThreadPool::WorkerFunction() {
    LOG_DEBUG("Worker thread started");

    while (true) {
        ScanTask task;

        // 从队列获取任务
        {
            unique_lock<mutex> lock(queueMutex);

            // 等待任务或停止信号
            taskAvailable.wait(lock, [this] {
                return stopFlag.load() || (!taskQueue.empty() && !pauseFlag.load());
            });

            // 检查停止条件
            if (stopFlag.load() && taskQueue.empty()) {
                LOG_DEBUG("Worker thread exiting");
                return;
            }

            // 如果暂停，继续等待
            if (pauseFlag.load()) {
                continue;
            }

            // 如果队列为空，继续等待
            if (taskQueue.empty()) {
                continue;
            }

            // 获取任务
            task = taskQueue.front();
            taskQueue.pop();
        }

        // 通知可能在等待队列空位的生产者
        queueNotFull.notify_one();

        // 内存超限检查：跳过扫描，只消费任务队列
        if (memoryLimitExceeded.load()) {
            completedTasks++;
            continue;
        }

        // 执行扫描任务
        vector<CarvedFileInfo> localResults;

        try {
            if (useSimdScan.load()) {
                ScanChunkSimd(task, localResults);
            } else {
                ScanChunk(task, localResults);
            }
        }
        catch (const exception& e) {
            LOG_ERROR_FMT("Exception in worker thread: %s", e.what());
        }
        catch (...) {
            LOG_ERROR("Unknown exception in worker thread");
        }

        // 保存结果（含内存限制检查）
        {
            // 估算本次结果内存占用
            ULONGLONG resultMemory = localResults.size() * sizeof(CarvedFileInfo);
            ULONGLONG currentTotal = totalResultsMemory.fetch_add(resultMemory) + resultMemory;

            // 检查是否超过 1GB 硬限制
            if (currentTotal > MAX_RESULTS_MEMORY) {
                if (!memoryLimitExceeded.exchange(true)) {
                    // 只输出一次警告
                    LOG_ERROR_FMT("Memory limit exceeded: results memory = %llu MB (limit: %llu MB). "
                                  "Stopping scan to prevent OOM.",
                                  currentTotal / (1024 * 1024),
                                  MAX_RESULTS_MEMORY / (1024 * 1024));
                    cerr << "\n[ERROR] Scan results memory exceeded 1GB limit ("
                         << (currentTotal / (1024 * 1024)) << " MB). "
                         << "Stopping scan. Found " << totalFilesFound.load()
                         << " files so far." << endl;
                }
                // 丢弃本次结果，回退内存计数
                totalResultsMemory.fetch_sub(resultMemory);
                localResults.clear();
                completedTasks++;
                totalBytesScanned += task.dataSize;
                continue;
            }

            lock_guard<mutex> lock(resultsMutex);
            ScanTaskResult result;
            result.taskId = task.taskId;
            result.files = move(localResults);
            result.bytesScanned = task.dataSize;
            result.filesFound = result.files.size();
            results.push_back(move(result));
        }

        // 更新统计
        totalFilesFound += localResults.size();
        totalBytesScanned += task.dataSize;
        completedTasks++;

        // 定期输出进度日志
        int completed = completedTasks.load();
        if (completed % 10 == 0) {
            LOG_DEBUG_FMT("Progress: %d/%d tasks (%.1f%%)",
                         completed, totalTasks.load(), GetProgress());
        }
    }
}

// ============================================================================
// 签名命中后的统一处理逻辑
// 返回跳过字节数（>0 = 有效匹配或 ZIP 跳过），0 = 不匹配（继续下一个签名）
// ============================================================================
size_t SignatureScanThreadPool::ProcessSignatureMatch(
    const BYTE* data, size_t dataSize,
    size_t matchOffset, const ScanTask& task,
    const FileSignature* sig,
    vector<CarvedFileInfo>& localResults)
{
    size_t remaining = dataSize - matchOffset;
    const BYTE* matchData = data + matchOffset;

    // AVI/WAV 子类型区分（都是 RIFF 容器）
    if (sig->extension == "avi" || sig->extension == "wav") {
        if (remaining >= 12) {
            bool isAvi = (matchData[8] == 'A' && matchData[9] == 'V' && matchData[10] == 'I');
            bool isWav = (matchData[8] == 'W' && matchData[9] == 'A' &&
                          matchData[10] == 'V' && matchData[11] == 'E');
            if ((sig->extension == "avi" && !isAvi) || (sig->extension == "wav" && !isWav)) {
                return 0;
            }
        } else {
            return 0;
        }
    }

    // MP4 ftyp 验证
    if (sig->extension == "mp4" && remaining >= 8) {
        if (!(matchData[4] == 'f' && matchData[5] == 't' &&
              matchData[6] == 'y' && matchData[7] == 'p')) {
            return 0;
        }
    }

    // 估算文件大小
    ULONGLONG footerPos = 0;
    bool isComplete = false;
    ULONGLONG estimatedSize;

    if (sig->extension == "zip") {
        estimatedSize = FileFormatUtils::EstimateFileSizeStatic(
            matchData, remaining, *sig, &footerPos, &isComplete);

        // ZIP 必须找到 EOCD 才有效
        if (estimatedSize == 0) {
            return sig->header.size();  // 跳过签名长度避免死循环
        }
    } else {
        estimatedSize = EstimateFileSize(matchData, remaining, *sig);
        isComplete = false;
    }

    // 验证文件
    double confidence = ValidateFile(matchData, min(remaining, (size_t)sig->maxSize), *sig);

    if (footerPos > 0) {
        confidence = min(1.0, confidence + 0.1);
    }
    if (!isComplete) {
        confidence *= 0.9;
    }

    if (confidence < 0.6) {
        return 0;
    }

    // 构建结果
    ULONGLONG absoluteLCN = task.baseLCN + (matchOffset / task.bytesPerCluster);
    ULONGLONG clusterOffset = matchOffset % task.bytesPerCluster;

    CarvedFileInfo info;
    info.startLCN = absoluteLCN;
    info.startOffset = clusterOffset;
    info.fileSize = estimatedSize;
    info.extension = sig->extension;
    info.description = sig->description;
    info.hasValidFooter = (footerPos > 0);
    info.sizeIsEstimated = !isComplete;
    info.confidence = confidence;

    // OOXML 检测
    if (sig->extension == "zip") {
        string ooxmlType = FileFormatUtils::DetectOOXMLTypeStatic(matchData, remaining);
        if (!ooxmlType.empty()) {
            info.extension = ooxmlType;
            if (ooxmlType == "docx") info.description = "Microsoft Word Document (OOXML)";
            else if (ooxmlType == "xlsx") info.description = "Microsoft Excel Spreadsheet (OOXML)";
            else if (ooxmlType == "pptx") info.description = "Microsoft PowerPoint Presentation (OOXML)";
            else if (ooxmlType == "ooxml") info.description = "Microsoft Office Document (OOXML)";
        }
    }

    // ML 增强
    if (IsMLEnabled()) {
        size_t mlDataSize = (size_t)(std::min)(remaining, (size_t)4096);
        EnhanceWithML(matchData, mlDataSize, info);
    }

    localResults.push_back(info);

    // 返回跳过大小
    return (estimatedSize > 0) ? (size_t)min(estimatedSize, (ULONGLONG)remaining) : sig->header.size();
}

// ============================================================================
// 核心扫描逻辑（扫描单个数据块）
// ============================================================================
void SignatureScanThreadPool::ScanChunk(const ScanTask& task,
                                         vector<CarvedFileInfo>& localResults) {
    if (!task.data || task.dataSize == 0) return;

    const BYTE* data = task.data;
    size_t dataSize = task.dataSize;
    size_t offset = 0;

    while (offset < dataSize && !stopFlag.load()) {
        BYTE currentByte = data[offset];

        auto it = signatureIndex->find(currentByte);
        if (it != signatureIndex->end()) {
            for (const FileSignature* sig : it->second) {
                if (activeSignatures->find(sig->extension) == activeSignatures->end()) {
                    continue;
                }

                size_t remaining = dataSize - offset;
                if (remaining < sig->header.size()) {
                    continue;
                }

                if (MatchSignature(data + offset, remaining, sig->header)) {
                    size_t skipSize = ProcessSignatureMatch(data, dataSize, offset, task, sig, localResults);
                    if (skipSize > 0) {
                        offset += skipSize;
                        goto next_position;
                    }
                }
            }
        }

        offset++;
        next_position:;
    }
}

// ============================================================================
// SIMD加速扫描（流式设计，零额外内存分配）
// 使用SIMD批量检测目标首字节，命中后立即就地验证签名
// ============================================================================
void SignatureScanThreadPool::ScanChunkSimd(const ScanTask& task,
                                              vector<CarvedFileInfo>& localResults) {
    if (!task.data || task.dataSize == 0) return;

    // 不支持 SSE2 时回退到标量
    if (simdLevel_ < CpuFeatures::SimdLevel::SSE2) {
        ScanChunk(task, localResults);
        return;
    }

    const BYTE* data = task.data;
    size_t dataSize = task.dataSize;
    size_t offset = 0;

    // 根据SIMD级别选择步长和处理方式
    const bool useAVX2 = (simdLevel_ >= CpuFeatures::SimdLevel::AVX2);
    const size_t simdWidth = useAVX2 ? 32 : 16;
    const size_t alignedEnd = dataSize & ~(simdWidth - 1);

    // 预计算 SIMD 目标字节向量（栈上分配，线程安全）
    if (useAVX2) {
        // ===== AVX2 路径：32 字节步进 =====
        const size_t targetCount = simdTargetBytes_.size();
        std::vector<__m256i> targetVecs;
        targetVecs.reserve(targetCount);
        for (BYTE b : simdTargetBytes_) {
            targetVecs.push_back(_mm256_set1_epi8(static_cast<char>(b)));
        }

        while (offset < alignedEnd && !stopFlag.load()) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset));

            // 对所有目标字节做并行比较
            __m256i anyMatch = _mm256_setzero_si256();
            for (const __m256i& target : targetVecs) {
                anyMatch = _mm256_or_si256(anyMatch, _mm256_cmpeq_epi8(chunk, target));
            }

            int mask = _mm256_movemask_epi8(anyMatch);
            if (mask == 0) {
                offset += 32;
                continue;
            }

            // 有匹配 → 逐位处理每个命中位置
            size_t windowEnd = offset + 32;
            while (mask) {
                unsigned long pos;
                _BitScanForward(&pos, mask);
                size_t matchOffset = offset + pos;
                mask &= (mask - 1);

                BYTE currentByte = data[matchOffset];
                auto it = signatureIndex->find(currentByte);
                if (it == signatureIndex->end()) continue;

                bool fileFound = false;
                for (const FileSignature* sig : it->second) {
                    if (activeSignatures->find(sig->extension) == activeSignatures->end()) {
                        continue;
                    }

                    size_t remaining = dataSize - matchOffset;
                    if (remaining < sig->header.size()) continue;

                    if (!MatchSignature(data + matchOffset, remaining, sig->header)) continue;

                    size_t skipSize = ProcessSignatureMatch(data, dataSize, matchOffset, task, sig, localResults);
                    if (skipSize > 0) {
                        offset = matchOffset + skipSize;
                        fileFound = true;
                        break;
                    }
                }

                if (fileFound) {
                    break;
                }
            }

            // 只有在没有文件跳跃的情况下才正常步进
            if (offset < windowEnd) {
                offset = windowEnd;
            }
        }

        _mm256_zeroupper();

    } else {
        // ===== SSE2 路径：16 字节步进 =====
        const size_t targetCount = simdTargetBytes_.size();
        std::vector<__m128i> targetVecs;
        targetVecs.reserve(targetCount);
        for (BYTE b : simdTargetBytes_) {
            targetVecs.push_back(_mm_set1_epi8(static_cast<char>(b)));
        }

        while (offset < alignedEnd && !stopFlag.load()) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + offset));

            __m128i anyMatch = _mm_setzero_si128();
            for (const __m128i& target : targetVecs) {
                anyMatch = _mm_or_si128(anyMatch, _mm_cmpeq_epi8(chunk, target));
            }

            int mask = _mm_movemask_epi8(anyMatch);
            if (mask == 0) {
                offset += 16;
                continue;
            }

            size_t windowEnd = offset + 16;
            while (mask) {
                unsigned long pos;
                _BitScanForward(&pos, mask);
                size_t matchOffset = offset + pos;
                mask &= (mask - 1);

                BYTE currentByte = data[matchOffset];
                auto it = signatureIndex->find(currentByte);
                if (it == signatureIndex->end()) continue;

                bool fileFound = false;
                for (const FileSignature* sig : it->second) {
                    if (activeSignatures->find(sig->extension) == activeSignatures->end()) {
                        continue;
                    }

                    size_t remaining = dataSize - matchOffset;
                    if (remaining < sig->header.size()) continue;

                    if (!MatchSignature(data + matchOffset, remaining, sig->header)) continue;

                    size_t skipSize = ProcessSignatureMatch(data, dataSize, matchOffset, task, sig, localResults);
                    if (skipSize > 0) {
                        offset = matchOffset + skipSize;
                        fileFound = true;
                        break;
                    }
                }

                if (fileFound) {
                    break;
                }
            }

            if (offset < windowEnd) {
                offset = windowEnd;
            }
        }
    }

    // ===== 剩余字节：标量处理（使用 bitmap 快速检查）=====
    while (offset < dataSize && !stopFlag.load()) {
        BYTE currentByte = data[offset];

        if (IsTargetByte(currentByte)) {
            auto it = signatureIndex->find(currentByte);
            if (it != signatureIndex->end()) {
                for (const FileSignature* sig : it->second) {
                    if (activeSignatures->find(sig->extension) == activeSignatures->end()) {
                        continue;
                    }

                    size_t remaining = dataSize - offset;
                    if (remaining < sig->header.size()) continue;

                    if (MatchSignature(data + offset, remaining, sig->header)) {
                        size_t skipSize = ProcessSignatureMatch(data, dataSize, offset, task, sig, localResults);
                        if (skipSize > 0) {
                            offset += skipSize;
                            goto simd_tail_next;
                        }
                    }
                }
            }
        }

        offset++;
        simd_tail_next:;
    }
}

// ============================================================================
// 获取SIMD信息
// ============================================================================
std::string SignatureScanThreadPool::GetSimdInfo() const {
    switch (simdLevel_) {
        case CpuFeatures::SimdLevel::AVX512: return "AVX-512";
        case CpuFeatures::SimdLevel::AVX2: return "AVX2";
        case CpuFeatures::SimdLevel::SSE42: return "SSE4.2";
        case CpuFeatures::SimdLevel::SSE2: return "SSE2";
        default: return "Scalar";
    }
}

// ============================================================================
// 签名匹配（统一入口）
// 根据签名长度和SIMD级别自动选择最优实现
// ============================================================================
bool SignatureScanThreadPool::MatchSignature(const BYTE* data, size_t dataSize,
                                              const vector<BYTE>& signature) {
    if (dataSize < signature.size()) return false;

    size_t sigLen = signature.size();

    // 短签名（< 4 字节）或不支持 SIMD：标量 memcmp 更快
    if (sigLen < 4 || simdLevel_ < CpuFeatures::SimdLevel::SSE2) {
        return MatchSignatureScalar(data, signature);
    }

    // SIMD 需要至少读取 16 字节，数据不足时回退标量
    if (dataSize < 16) {
        return MatchSignatureScalar(data, signature);
    }

    return MatchSignatureSimd(data, dataSize, signature);
}

// ============================================================================
// 标量签名匹配（memcmp 回退实现）
// ============================================================================
bool SignatureScanThreadPool::MatchSignatureScalar(const BYTE* data,
                                                     const vector<BYTE>& signature) {
    return memcmp(data, signature.data(), signature.size()) == 0;
}

// ============================================================================
// SIMD 优化签名匹配
// SSE2：单次 16 字节比较覆盖 4-16 字节签名
// AVX2：单次 32 字节比较覆盖 17-32 字节签名（预留扩展）
// ============================================================================
bool SignatureScanThreadPool::MatchSignatureSimd(const BYTE* data, size_t dataSize,
                                                   const vector<BYTE>& signature) {
    size_t sigLen = signature.size();
    const BYTE* sigData = signature.data();

    if (sigLen <= 16) {
        // SSE2 路径：单次 16 字节比较
        // data 侧：调用方已保证 dataSize >= 16
        __m128i data_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));

        // signature 侧：可能不足 16 字节，使用栈缓冲区安全加载
        alignas(16) BYTE sigBuf[16] = { 0 };
        memcpy(sigBuf, sigData, sigLen);
        __m128i sig_vec = _mm_load_si128(reinterpret_cast<const __m128i*>(sigBuf));

        __m128i cmp = _mm_cmpeq_epi8(data_vec, sig_vec);
        int mask = _mm_movemask_epi8(cmp);

        // 只检查签名长度对应的低位
        // 例如 sigLen=4 → expected_mask = 0x0F (0b00001111)
        int expected_mask = (1 << sigLen) - 1;
        return (mask & expected_mask) == expected_mask;
    }
    else if (simdLevel_ >= CpuFeatures::SimdLevel::AVX2 && sigLen <= 32) {
        // AVX2 路径：单次 32 字节比较（当前签名最长 8 字节，此分支预留扩展）
        if (dataSize < 32) {
            return MatchSignatureScalar(data, signature);
        }

        __m256i data_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));

        alignas(32) BYTE sigBuf[32] = { 0 };
        memcpy(sigBuf, sigData, sigLen);
        __m256i sig_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(sigBuf));

        __m256i cmp = _mm256_cmpeq_epi8(data_vec, sig_vec);
        int mask = _mm256_movemask_epi8(cmp);

        // 使用 unsigned 避免 sigLen >= 31 时的移位溢出
        unsigned int expected_mask = (1u << sigLen) - 1u;
        return (static_cast<unsigned int>(mask) & expected_mask) == expected_mask;
    }
    else {
        // 超长签名（> 32 字节）：回退标量
        return MatchSignatureScalar(data, signature);
    }
}

// ============================================================================
// 估算文件大小（轻量级，仅解析头部字段，用于扫描热循环）
// 详细的 footer 搜索和结构遍历推迟到 recover 阶段
// ============================================================================
ULONGLONG SignatureScanThreadPool::EstimateFileSize(const BYTE* data, size_t dataSize,
                                                     const FileSignature& sig) {
    // BMP: 头部包含精确大小
    if (sig.extension == "bmp" && dataSize >= 6) {
        DWORD size = *(DWORD*)(data + 2);
        if (size > sig.minSize && size <= sig.maxSize && size <= dataSize) {
            return size;
        }
    }

    // AVI: RIFF 头部包含精确大小
    if (sig.extension == "avi" && dataSize >= 12) {
        if (data[8] == 'A' && data[9] == 'V' && data[10] == 'I' && data[11] == ' ') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                return min((ULONGLONG)riffSize + 8, (ULONGLONG)dataSize);
            }
        }
    }

    // WAV: RIFF 头部包含精确大小
    if (sig.extension == "wav" && dataSize >= 12) {
        if (data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E') {
            DWORD riffSize = *(DWORD*)(data + 4);
            if (riffSize > 0 && riffSize <= sig.maxSize) {
                return min((ULONGLONG)riffSize + 8, (ULONGLONG)dataSize);
            }
        }
    }

    // MP4: 读取 ftyp atom 大小（但实际文件通常更大）
    if (sig.extension == "mp4" && dataSize >= 8) {
        DWORD atomSize = _byteswap_ulong(*(DWORD*)data);
        if (atomSize >= 8) {
            return min((ULONGLONG)dataSize, sig.maxSize);
        }
    }

    // 默认：保守估计
    return min((ULONGLONG)dataSize, sig.maxSize);
}

// ============================================================================
// 文件验证（简化版）
// ============================================================================
double SignatureScanThreadPool::ValidateFile(const BYTE* data, size_t dataSize,
                                              const FileSignature& sig) {
    double confidence = 0.8;  // 基础置信度（签名已匹配）

    // 特定格式额外验证（检查各格式特有结构）
    if (sig.extension == "jpg" && dataSize >= 10) {
        if ((data[3] == 0xE0 || data[3] == 0xE1) && data[6] == 0x4A) {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "png" && dataSize >= 24) {
        if (data[12] == 'I' && data[13] == 'H' && data[14] == 'D' && data[15] == 'R') {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "pdf" && dataSize >= 20) {
        if (data[5] == '-' && data[6] >= '1' && data[6] <= '9') {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "zip" && dataSize >= 30) {
        WORD version = *(WORD*)(data + 4);
        WORD flags = *(WORD*)(data + 6);
        if (version <= 63 && (flags & 0xFF00) == 0) {
            confidence += 0.1;
        }
    }
    else if (sig.extension == "mp4" && dataSize >= 12) {
        if (data[4] == 'f' && data[5] == 't' && data[6] == 'y' && data[7] == 'p') {
            confidence += 0.15;
        }
    }

    return min(1.0, confidence);
}

// ============================================================================
// ML增强验证
// ============================================================================
void SignatureScanThreadPool::EnhanceWithML(const BYTE* data, size_t dataSize,
                                             CarvedFileInfo& info) {
    if (!mlClassifier || !mlClassifier->isLoaded()) {
        return;
    }

    // 检查签名检测到的类型是否在ML模型支持范围内
    // 特殊处理：OLE文件(doc)需要检查xls/ppt是否也支持
    bool isOleFile = (info.extension == "doc");
    if (!isOleFile && !mlClassifier->isTypeSupported(info.extension)) {
        // 类型不被ML支持，跳过分类，不计入统计
        mlSkippedCount++;
        return;
    }

    try {
        auto result = mlClassifier->classify(data, dataSize);
        if (result && result->isValid()) {
            // 保存ML分类结果
            info.mlClassification = result->predictedType;
            info.mlConfidence = result->confidence;

            // 如果ML标记为unknown（低置信度），不参与匹配统计
            if (result->isUnknown) {
                mlUnknownCount++;
                return;
            }

            // 特殊处理：OLE文件类型细化
            // 如果签名检测为doc，但ML高置信度判断为xls/ppt，则更新类型
            if (isOleFile && result->confidence > config.mlConfidenceThreshold) {
                if (result->predictedType == "xls" || result->predictedType == "ppt") {
                    info.extension = result->predictedType;
                    info.description = (result->predictedType == "xls") ?
                        "Microsoft Excel Spreadsheet (ML refined)" :
                        "Microsoft PowerPoint Presentation (ML refined)";
                    info.confidence = (std::min)(1.0, info.confidence + 0.15 * result->confidence);
                    mlMatchCount++;
                    mlEnhancedCount++;
                    return;
                } else if (result->predictedType == "doc") {
                    info.confidence = (std::min)(1.0, info.confidence + 0.1 * result->confidence);
                    mlMatchCount++;
                    mlEnhancedCount++;
                    return;
                }
            }

            // 如果ML预测与签名检测一致，提高置信度
            if (result->predictedType == info.extension) {
                info.confidence = (std::min)(1.0, info.confidence + 0.1 * result->confidence);
                mlMatchCount++;
            }
            // 如果ML高置信度预测与签名不一致，轻微降低置信度
            else if (result->confidence > config.mlConfidenceThreshold) {
                info.confidence *= 0.9;  // 轻微降低置信度
                mlMismatchCount++;
            }

            mlEnhancedCount++;
        }
    }
    catch (const std::exception& e) {
        LOG_DEBUG_FMT("ML classification error: %s", e.what());
    }
}

// ============================================================================
// 提交任务（阻塞式）
// ============================================================================
void SignatureScanThreadPool::SubmitTask(const ScanTask& task) {
    {
        unique_lock<mutex> lock(queueMutex);

        // 等待队列有空位
        queueNotFull.wait(lock, [this] {
            return taskQueue.size() < config.maxQueueSize || stopFlag.load();
        });

        if (stopFlag.load()) return;

        taskQueue.push(task);
        totalTasks++;
    }

    taskAvailable.notify_one();
}

// ============================================================================
// 尝试提交任务（非阻塞式）
// ============================================================================
bool SignatureScanThreadPool::TrySubmitTask(const ScanTask& task) {
    {
        lock_guard<mutex> lock(queueMutex);

        if (taskQueue.size() >= config.maxQueueSize) {
            return false;
        }

        taskQueue.push(task);
        totalTasks++;
    }

    taskAvailable.notify_one();
    return true;
}

// ============================================================================
// 等待所有任务完成
// ============================================================================
void SignatureScanThreadPool::WaitForCompletion() {
    while (completedTasks.load() < totalTasks.load()) {
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}

// ============================================================================
// 获取合并后的结果
// ============================================================================
vector<CarvedFileInfo> SignatureScanThreadPool::GetMergedResults() {
    lock_guard<mutex> lock(resultsMutex);

    // 按 taskId 排序以保持顺序
    sort(results.begin(), results.end(),
         [](const ScanTaskResult& a, const ScanTaskResult& b) {
             return a.taskId < b.taskId;
         });

    // 合并所有结果
    vector<CarvedFileInfo> merged;
    for (const auto& r : results) {
        merged.insert(merged.end(), r.files.begin(), r.files.end());
    }

    return merged;
}

// ============================================================================
// 清空结果
// ============================================================================
void SignatureScanThreadPool::ClearResults() {
    lock_guard<mutex> lock(resultsMutex);
    results.clear();
    completedTasks = 0;
    totalTasks = 0;
    totalFilesFound = 0;
    totalBytesScanned = 0;
}

// ============================================================================
// 获取进度
// ============================================================================
double SignatureScanThreadPool::GetProgress() const {
    int total = totalTasks.load();
    if (total == 0) return 0.0;
    return (double)completedTasks.load() / total * 100.0;
}

// ============================================================================
// 获取最优线程数
// ============================================================================
int SignatureScanThreadPool::GetOptimalThreadCount() {
    int cores = ThreadPoolUtils::GetLogicalCoreCount();

    // 高端CPU（16线程及以上）：使用 cores - 4，留4个给系统和I/O
    // 中端CPU（8-15线程）：使用 cores - 2
    // 低端CPU（4-7线程）：使用 cores - 1
    // 最少使用2个工作线程

    int optimalCount;
    if (cores >= 16) {
        optimalCount = cores - 4;  // 留4个给系统和I/O
    } else if (cores >= 8) {
        optimalCount = cores - 2;
    } else if (cores >= 4) {
        optimalCount = cores - 1;
    } else {
        optimalCount = max(2, cores);
    }

    LOG_INFO_FMT("Detected %d logical cores, using %d worker threads", cores, optimalCount);
    return optimalCount;
}

// ============================================================================
// 辅助函数实现
// ============================================================================
namespace ThreadPoolUtils {

int GetLogicalCoreCount() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    int cores = (int)sysInfo.dwNumberOfProcessors;
    return (cores > 0) ? cores : 4;  // 默认4核
}

int GetPhysicalCoreCount() {
    // 简化实现：假设超线程比例为2:1（物理核心数约为逻辑核心数的一半）
    return max(1, GetLogicalCoreCount() / 2);
}

ULONGLONG GetAvailableMemoryMB() {
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        return memStatus.ullAvailPhys / (1024 * 1024);
    }
    return 4096;  // 获取失败时默认返回4GB
}

ScanThreadPoolConfig GetOptimalConfig() {
    ScanThreadPoolConfig config;

    int cores = GetLogicalCoreCount();
    ULONGLONG availMem = GetAvailableMemoryMB();

    // 根据CPU核心数设置工作线程数和相关参数
    if (cores >= 16) {
        config.workerCount = 12;
        config.chunkSize = 8 * 1024 * 1024;   // 8MB
        config.maxQueueSize = 48;
    } else if (cores >= 8) {
        config.workerCount = 6;
        config.chunkSize = 8 * 1024 * 1024;   // 8MB
        config.maxQueueSize = 24;
    } else if (cores >= 4) {
        config.workerCount = 3;
        config.chunkSize = 4 * 1024 * 1024;   // 4MB
        config.maxQueueSize = 12;
    } else {
        config.workerCount = 2;
        config.chunkSize = 4 * 1024 * 1024;   // 4MB
        config.maxQueueSize = 8;
    }

    // 根据可用内存调整队列大小
    // 每个任务最多占用 chunkSize 内存，限制使用25%可用内存
    ULONGLONG maxMemoryForQueue = availMem / 4;
    size_t maxQueueByMem = (size_t)(maxMemoryForQueue * 1024 * 1024 / config.chunkSize);
    config.maxQueueSize = min(config.maxQueueSize, maxQueueByMem);
    config.maxQueueSize = max(config.maxQueueSize, (size_t)4);  // 队列至少保持4个任务

    LOG_INFO_FMT("Optimal config: %d workers, %zu MB chunks, %zu max queue (Cores: %d, Mem: %llu MB)",
                 config.workerCount, config.chunkSize / (1024 * 1024),
                 config.maxQueueSize, cores, availMem);

    return config;
}

} // namespace ThreadPoolUtils
