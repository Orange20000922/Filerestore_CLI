#include "MemoryMappedResults.h"
#include "CarvedResultsCache.h"
#include "Logger.h"
#include <fstream>

using namespace std;

// ============================================================================
// 构造和析构
// ============================================================================
MemoryMappedResults::MemoryMappedResults()
    : hFile(INVALID_HANDLE_VALUE), hMapping(NULL), pMappedView(NULL),
      fileSize(0), totalRecords(0), currentViewStart(0), currentViewSize(0),
      isValid(false), driveLetter(0) {
}

MemoryMappedResults::~MemoryMappedResults() {
    Close();
}

// ============================================================================
// 映射特定范围到内存
// ============================================================================
bool MemoryMappedResults::MapViewRange(ULONGLONG offset, ULONGLONG size) {
    // 如果当前视图已经包含所需范围，直接返回
    if (pMappedView != NULL &&
        offset >= currentViewStart &&
        offset + size <= currentViewStart + currentViewSize) {
        return true;
    }

    // 取消当前映射
    UnmapCurrentView();

    // 确保偏移对齐到分配粒度（64KB）
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    ULONGLONG alignedOffset = (offset / sysInfo.dwAllocationGranularity) *
                               sysInfo.dwAllocationGranularity;

    // 调整大小以包含原始偏移
    ULONGLONG adjustedSize = size + (offset - alignedOffset);

    // 限制视图大小
    adjustedSize = min(adjustedSize, (ULONGLONG)MMAP_VIEW_SIZE);
    adjustedSize = min(adjustedSize, fileSize - alignedOffset);

    // 创建映射视图
    pMappedView = MapViewOfFile(
        hMapping,
        FILE_MAP_READ,
        (DWORD)(alignedOffset >> 32),
        (DWORD)(alignedOffset & 0xFFFFFFFF),
        (SIZE_T)adjustedSize
    );

    if (pMappedView == NULL) {
        LOG_ERROR_FMT("MapViewOfFile failed: error %u", GetLastError());
        return false;
    }

    currentViewStart = alignedOffset;
    currentViewSize = adjustedSize;

    LOG_DEBUG_FMT("Mapped view: offset=%llu, size=%llu KB",
                 alignedOffset, adjustedSize / 1024);

    return true;
}

// ============================================================================
// 取消当前视图映射
// ============================================================================
void MemoryMappedResults::UnmapCurrentView() {
    if (pMappedView != NULL) {
        UnmapViewOfFile(pMappedView);
        pMappedView = NULL;
        currentViewStart = 0;
        currentViewSize = 0;
    }
}

// ============================================================================
// 从缓存文件创建内存映射
// ============================================================================
bool MemoryMappedResults::OpenFromCache(const string& cachePath) {
    Close();

    filePath = cachePath;

    // 打开文件
    hFile = CreateFileA(
        cachePath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        LOG_ERROR_FMT("Failed to open cache file: %s (error %u)",
                     cachePath.c_str(), GetLastError());
        return false;
    }

    // 获取文件大小
    LARGE_INTEGER liFileSize;
    if (!GetFileSizeEx(hFile, &liFileSize)) {
        LOG_ERROR_FMT("GetFileSizeEx failed: error %u", GetLastError());
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        return false;
    }

    fileSize = liFileSize.QuadPart;

    // 读取文件头获取记录数
    CarvedCacheHeader header;
    DWORD bytesRead = 0;
    if (!ReadFile(hFile, &header, sizeof(header), &bytesRead, NULL) ||
        bytesRead != sizeof(header)) {
        LOG_ERROR("Failed to read cache header");
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        return false;
    }

    if (header.magic != CARVED_CACHE_MAGIC) {
        LOG_ERROR("Invalid cache file magic");
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        return false;
    }

    totalRecords = header.fileCount;
    driveLetter = header.driveLetter;

    // 创建文件映射对象
    hMapping = CreateFileMapping(
        hFile,
        NULL,
        PAGE_READONLY,
        (DWORD)(fileSize >> 32),
        (DWORD)(fileSize & 0xFFFFFFFF),
        NULL
    );

    if (hMapping == NULL) {
        LOG_ERROR_FMT("CreateFileMapping failed: error %u", GetLastError());
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        return false;
    }

    isValid = true;

    LOG_INFO_FMT("Memory-mapped cache opened: %llu records, %llu MB",
                 totalRecords, fileSize / (1024 * 1024));

    return true;
}

// ============================================================================
// 关闭内存映射
// ============================================================================
void MemoryMappedResults::Close() {
    UnmapCurrentView();

    if (hMapping != NULL) {
        CloseHandle(hMapping);
        hMapping = NULL;
    }

    if (hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
    }

    isValid = false;
    fileSize = 0;
    totalRecords = 0;
}

// ============================================================================
// 获取单个记录
// ============================================================================
bool MemoryMappedResults::GetRecord(size_t index, CarvedFileInfo& outInfo) {
    if (!isValid || index >= totalRecords) {
        LOG_ERROR_FMT("Invalid index: %zu (total: %llu)", index, totalRecords);
        return false;
    }

    // 注意：由于每个记录大小不固定（有字符串），我们无法直接计算偏移
    // 需要顺序读取到目标记录（或者使用索引表，这里简化实现）

    // 为了性能，这里我们批量读取并缓存
    // 实际应用中应该维护一个记录偏移索引表
    vector<CarvedFileInfo> batch;
    if (!GetRecordBatch(index, 1, batch)) {
        return false;
    }

    if (batch.empty()) {
        return false;
    }

    outInfo = batch[0];
    return true;
}

// ============================================================================
// 获取一批记录（分页）
// ============================================================================
bool MemoryMappedResults::GetRecordBatch(size_t startIndex, size_t count,
                                          vector<CarvedFileInfo>& outResults) {
    if (!isValid || startIndex >= totalRecords) {
        return false;
    }

    // 限制批量大小
    size_t actualCount = min(count, (size_t)(totalRecords - startIndex));

    // 映射整个文件头部（包含所有记录）
    // 对于超大文件，这里应该优化为只映射需要的部分
    ULONGLONG estimatedOffset = sizeof(CarvedCacheHeader);
    ULONGLONG estimatedSize = fileSize - estimatedOffset;

    if (!MapViewRange(estimatedOffset, min(estimatedSize, (ULONGLONG)MMAP_VIEW_SIZE))) {
        return false;
    }

    // 从映射的内存中反序列化记录
    // 这里需要手动解析，因为我们不能直接用 ifstream
    const BYTE* pData = (const BYTE*)pMappedView;
    size_t offset = estimatedOffset - currentViewStart;  // 视图内的偏移

    outResults.clear();
    outResults.reserve(actualCount);

    // 创建一个临时的内存流来模拟文件读取
    // 注意：这是简化实现，实际应该直接操作字节

    // 跳过前面的记录
    for (size_t i = 0; i < startIndex; i++) {
        // 读取字段跳过记录
        // startLCN, startOffset, fileSize, hasValidFooter, confidence
        offset += sizeof(ULONGLONG) * 3 + sizeof(bool) + sizeof(double);

        // 跳过字符串（extension）
        if (offset + sizeof(DWORD) > currentViewSize) {
            LOG_ERROR("View overflow during skip");
            return false;
        }
        DWORD strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD) + strLen;

        // 跳过字符串（description）
        if (offset + sizeof(DWORD) > currentViewSize) {
            LOG_ERROR("View overflow during skip");
            return false;
        }
        strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD) + strLen;

        // 跳过时间戳和其他字段
        offset += sizeof(FILETIME) * 3 + sizeof(TimestampSource) + sizeof(ULONGLONG);

        // 跳过 embeddedInfo
        if (offset + sizeof(DWORD) > currentViewSize) {
            LOG_ERROR("View overflow during skip");
            return false;
        }
        strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD) + strLen;

        // 跳过完整性信息
        offset += sizeof(double) + sizeof(bool);

        // 跳过 integrityDiagnosis
        if (offset + sizeof(DWORD) > currentViewSize) {
            LOG_ERROR("View overflow during skip");
            return false;
        }
        strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD) + strLen;
    }

    // 读取所需记录
    for (size_t i = 0; i < actualCount; i++) {
        CarvedFileInfo info;

        // 读取基本字段
        if (offset + sizeof(ULONGLONG) * 3 + sizeof(bool) + sizeof(double) > currentViewSize) {
            LOG_WARNING("Reached view boundary, stopping batch read");
            break;
        }

        info.startLCN = *(ULONGLONG*)(pData + offset);
        offset += sizeof(ULONGLONG);

        info.startOffset = *(ULONGLONG*)(pData + offset);
        offset += sizeof(ULONGLONG);

        info.fileSize = *(ULONGLONG*)(pData + offset);
        offset += sizeof(ULONGLONG);

        info.hasValidFooter = *(bool*)(pData + offset);
        offset += sizeof(bool);

        info.confidence = *(double*)(pData + offset);
        offset += sizeof(double);

        // 读取 extension
        DWORD strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD);
        if (strLen > 0 && strLen < 256 && offset + strLen <= currentViewSize) {
            info.extension = string((const char*)(pData + offset), strLen);
            offset += strLen;
        }

        // 读取 description
        strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD);
        if (strLen > 0 && strLen < 1024 && offset + strLen <= currentViewSize) {
            info.description = string((const char*)(pData + offset), strLen);
            offset += strLen;
        }

        // 读取时间戳
        info.creationTime = *(FILETIME*)(pData + offset);
        offset += sizeof(FILETIME);

        info.modificationTime = *(FILETIME*)(pData + offset);
        offset += sizeof(FILETIME);

        info.accessTime = *(FILETIME*)(pData + offset);
        offset += sizeof(FILETIME);

        info.tsSource = *(TimestampSource*)(pData + offset);
        offset += sizeof(TimestampSource);

        info.matchedMftRecord = *(ULONGLONG*)(pData + offset);
        offset += sizeof(ULONGLONG);

        // 读取 embeddedInfo
        strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD);
        if (strLen > 0 && strLen < 2048 && offset + strLen <= currentViewSize) {
            info.embeddedInfo = string((const char*)(pData + offset), strLen);
            offset += strLen;
        }

        // 读取完整性信息
        info.integrityScore = *(double*)(pData + offset);
        offset += sizeof(double);

        info.integrityValidated = *(bool*)(pData + offset);
        offset += sizeof(bool);

        // 读取 integrityDiagnosis
        strLen = *(DWORD*)(pData + offset);
        offset += sizeof(DWORD);
        if (strLen > 0 && strLen < 1024 && offset + strLen <= currentViewSize) {
            info.integrityDiagnosis = string((const char*)(pData + offset), strLen);
            offset += strLen;
        }

        outResults.push_back(info);
    }

    LOG_DEBUG_FMT("Loaded batch: %zu records (index %zu-%zu)",
                 outResults.size(), startIndex, startIndex + outResults.size());

    return !outResults.empty();
}

// ============================================================================
// 预取记录
// ============================================================================
void MemoryMappedResults::PrefetchRange(size_t startIndex, size_t count) {
    // 预取功能：提前映射所需区域到内存
    // Windows 会自动缓存映射的页面，提高后续访问速度

    if (!isValid || startIndex >= totalRecords) {
        return;
    }

    // 估算偏移（简化实现）
    ULONGLONG estimatedOffset = sizeof(CarvedCacheHeader) + startIndex * 300;  // 假设每条记录约300字节
    ULONGLONG estimatedSize = count * 300;

    MapViewRange(estimatedOffset, estimatedSize);

    LOG_DEBUG_FMT("Prefetched range: %zu-%zu", startIndex, startIndex + count);
}
