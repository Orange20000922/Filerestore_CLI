#include "ClusterFilteredReader.h"
#include "MFTReader.h"
#include "OverwriteDetector.h"
#include "FileFormatUtils.h"
#include "Logger.h"
#include <algorithm>
#include <chrono>

using namespace std;

// ============================================================================
// 构造 / 析构
// ============================================================================

ClusterFilteredReader::ClusterFilteredReader(MFTReader* reader)
    : reader_(reader) {
}

ClusterFilteredReader::~ClusterFilteredReader() {
    // unique_ptr 自动释放 detector_
}

// ============================================================================
// 延迟创建 OverwriteDetector（复用 $Bitmap 缓存）
// ============================================================================

OverwriteDetector& ClusterFilteredReader::GetDetector() {
    if (!detector_) {
        detector_ = make_unique<OverwriteDetector>(reader_);
        detector_->SetDetectionMode(MODE_THOROUGH);
    }
    return *detector_;
}

// ============================================================================
// 构建簇健康报告
// ============================================================================

ClusterHealthReport ClusterFilteredReader::BuildHealthReport(
    const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns) {

    ClusterHealthReport report;

    // 计算总簇数
    for (const auto& run : dataRuns) {
        report.totalClusters += run.second;
    }

    if (report.totalClusters == 0) {
        return report;
    }

    // 调用 OverwriteDetector 批量检测
    OverwriteDetectionResult detResult = GetDetector().CheckDataRuns(dataRuns);

    report.detectionTimeMs = detResult.detectionTimeMs;

    // 构建逐簇布尔向量
    report.clusterIsGood.resize((size_t)report.totalClusters, true);

    size_t idx = 0;
    for (const auto& status : detResult.clusterStatuses) {
        if (idx < report.clusterIsGood.size()) {
            report.clusterIsGood[idx] = !status.isOverwritten;
            idx++;
        }
    }

    // 统计
    report.goodClusters = 0;
    report.overwrittenClusters = 0;
    for (size_t i = 0; i < report.clusterIsGood.size(); i++) {
        if (report.clusterIsGood[i]) {
            report.goodClusters++;
        } else {
            report.overwrittenClusters++;
        }
    }

    // 计算健康百分比
    if (report.totalClusters > 0) {
        report.healthPercentage = (double)report.goodClusters / report.totalClusters * 100.0;
    }

    LOG_INFO_FMT("ClusterHealthReport: %llu/%llu good (%.1f%%), detection %.2f ms",
                 report.goodClusters, report.totalClusters,
                 report.healthPercentage, report.detectionTimeMs);

    return report;
}

// ============================================================================
// 带过滤的读取
// ============================================================================

bool ClusterFilteredReader::ReadWithFilter(
    const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
    ULONGLONG fileSize,
    const ClusterHealthReport& report,
    vector<BYTE>& fileData) {

    DWORD clusterSize = (DWORD)reader_->GetBytesPerCluster();
    if (clusterSize == 0) return false;

    // 预分配
    ULONGLONG totalBytes = report.totalClusters * clusterSize;
    fileData.clear();
    fileData.reserve((size_t)min(totalBytes, fileSize));

    ULONGLONG clusterIndex = 0;
    ULONGLONG bytesCollected = 0;

    for (const auto& run : dataRuns) {
        ULONGLONG lcn = run.first;
        ULONGLONG count = run.second;

        // 批量读取整个 run
        vector<BYTE> batchData;
        if (!reader_->ReadClusters(lcn, count, batchData)) {
            LOG_WARNING_FMT("ClusterFilteredReader: 无法读取 LCN=%llu count=%llu", lcn, count);
            // 填零保持偏移对齐
            for (ULONGLONG i = 0; i < count; i++) {
                ULONGLONG remaining = fileSize - bytesCollected;
                ULONGLONG chunkSize = min((ULONGLONG)clusterSize, remaining);
                if (chunkSize == 0) break;
                fileData.insert(fileData.end(), (size_t)chunkSize, 0);
                bytesCollected += chunkSize;
                clusterIndex++;
            }
            continue;
        }

        // 逐簇处理
        for (ULONGLONG i = 0; i < count; i++) {
            if (bytesCollected >= fileSize) break;

            ULONGLONG remaining = fileSize - bytesCollected;
            ULONGLONG chunkSize = min((ULONGLONG)clusterSize, remaining);
            size_t srcOffset = (size_t)(i * clusterSize);

            if (clusterIndex < report.clusterIsGood.size() && report.clusterIsGood[(size_t)clusterIndex]) {
                // Good cluster - 复制实际数据
                if (srcOffset + chunkSize <= batchData.size()) {
                    fileData.insert(fileData.end(),
                                    batchData.begin() + srcOffset,
                                    batchData.begin() + srcOffset + (size_t)chunkSize);
                } else {
                    // 数据不足，补零
                    fileData.insert(fileData.end(), (size_t)chunkSize, 0);
                }
            } else {
                // Bad cluster - 插入零字节（保持偏移对齐）
                fileData.insert(fileData.end(), (size_t)chunkSize, 0);
            }

            bytesCollected += chunkSize;
            clusterIndex++;
        }

        if (bytesCollected >= fileSize) break;
    }

    // 截断到精确 fileSize
    if (fileData.size() > (size_t)fileSize) {
        fileData.resize((size_t)fileSize);
    }

    return !fileData.empty();
}

// ============================================================================
// 格式感知截断
// ============================================================================

ULONGLONG ClusterFilteredReader::TryFormatTruncation(
    vector<BYTE>& fileData,
    const wstring& fileName) {

    if (fileData.empty()) return 0;

    // 提取扩展名
    wstring ext;
    size_t dotPos = fileName.rfind(L'.');
    if (dotPos != wstring::npos && dotPos < fileName.length() - 1) {
        ext = fileName.substr(dotPos + 1);
    }
    // 转小写
    for (auto& c : ext) {
        c = towlower(c);
    }

    ULONGLONG endPos = 0;
    const BYTE* data = fileData.data();
    size_t dataSize = fileData.size();

    if (ext == L"png") {
        endPos = FileFormatUtils::FindPngEndByChunksStatic(data, dataSize);
    }
    else if (ext == L"zip" || ext == L"docx" || ext == L"xlsx" || ext == L"pptx") {
        endPos = FileFormatUtils::FindZipEndOfCentralDirectoryStatic(data, dataSize);
    }
    else if (ext == L"jpg" || ext == L"jpeg") {
        // JPEG: 必须用反向搜索找最后一个 FFD9
        // 前向搜索会命中 EXIF 缩略图的 FFD9，导致整个文件被截断到只剩缩略图
        vector<BYTE> jpegFooter = { 0xFF, 0xD9 };
        endPos = FileFormatUtils::FindFooterReverseStatic(data, dataSize, jpegFooter, dataSize);
    }
    else if (ext == L"pdf") {
        // PDF %%EOF — 反向搜索找最后一个
        vector<BYTE> pdfFooter = { '%', '%', 'E', 'O', 'F' };
        endPos = FileFormatUtils::FindFooterReverseStatic(data, dataSize, pdfFooter, dataSize);
    }

    if (endPos > 0 && endPos < fileData.size()) {
        ULONGLONG truncated = fileData.size() - endPos;

        // 安全阈值：截断不得移除超过原文件 50% 的数据
        // 如果格式解析器因坏簇零字节误判了结束位置，这能防止灾难性截断
        if (endPos < fileData.size() / 2) {
            LOG_WARNING_FMT("格式截断被安全阈值阻止: 截断点 %llu < 文件大小 %zu 的 50%%",
                            endPos, fileData.size());
            return 0;
        }

        fileData.resize((size_t)endPos);
        LOG_INFO_FMT("格式截断: %ls -> %llu bytes removed (end at %llu)",
                     ext.c_str(), truncated, endPos);
        return truncated;
    }

    return 0;
}

// ============================================================================
// 公开方法: ReadFromDataRuns（USN 定点恢复路径）
// ============================================================================

bool ClusterFilteredReader::ReadFromDataRuns(
    const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
    ULONGLONG fileSize,
    const wstring& fileName,
    vector<BYTE>& fileData,
    ClusterHealthReport& report) {

    if (dataRuns.empty() || fileSize == 0) {
        return false;
    }

    // 1. 构建健康报告
    report = BuildHealthReport(dataRuns);

    // 2. 全部坏簇，直接失败
    if (report.goodClusters == 0) {
        LOG_WARNING("ClusterFilteredReader: 所有簇均被覆盖");
        return false;
    }

    // 3. 带过滤读取
    if (!ReadWithFilter(dataRuns, fileSize, report, fileData)) {
        return false;
    }

    // 4. 如果有坏簇，尝试格式截断
    if (report.overwrittenClusters > 0) {
        ULONGLONG truncated = TryFormatTruncation(fileData, fileName);
        if (truncated > 0) {
            report.formatTruncated = true;
            report.truncatedBytes = truncated;
        }
    }

    return true;
}

// ============================================================================
// 公开方法: ReadContiguous（签名雕刻 / API 路径）
// ============================================================================

bool ClusterFilteredReader::ReadContiguous(
    ULONGLONG startLCN,
    ULONGLONG startOffset,
    ULONGLONG fileSize,
    const wstring& fileName,
    vector<BYTE>& fileData,
    ClusterHealthReport& report) {

    if (fileSize == 0) {
        return false;
    }

    ULONGLONG bytesPerCluster = reader_->GetBytesPerCluster();
    if (bytesPerCluster == 0) return false;

    // 计算需要的总簇数（包含 startOffset）
    ULONGLONG totalBytes = startOffset + fileSize;
    ULONGLONG clusterCount = (totalBytes + bytesPerCluster - 1) / bytesPerCluster;

    // 构造单元素 data runs
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;
    dataRuns.push_back(make_pair(startLCN, clusterCount));

    // 构建健康报告
    report = BuildHealthReport(dataRuns);

    if (report.goodClusters == 0) {
        LOG_WARNING("ClusterFilteredReader: 连续区域所有簇均被覆盖");
        return false;
    }

    // 读取全部数据（含 offset 前的部分）
    vector<BYTE> rawData;
    if (!ReadWithFilter(dataRuns, totalBytes, report, rawData)) {
        return false;
    }

    // 截取 startOffset 之后的 fileSize 字节
    if (startOffset > 0) {
        if (rawData.size() > (size_t)startOffset) {
            size_t available = rawData.size() - (size_t)startOffset;
            size_t copySize = min(available, (size_t)fileSize);
            fileData.assign(rawData.begin() + (size_t)startOffset,
                            rawData.begin() + (size_t)startOffset + copySize);
        } else {
            return false;
        }
    } else {
        fileData = move(rawData);
        if (fileData.size() > (size_t)fileSize) {
            fileData.resize((size_t)fileSize);
        }
    }

    // 如果有坏簇，尝试格式截断
    if (report.overwrittenClusters > 0) {
        ULONGLONG truncated = TryFormatTruncation(fileData, fileName);
        if (truncated > 0) {
            report.formatTruncated = true;
            report.truncatedBytes = truncated;
        }
    }

    return true;
}
