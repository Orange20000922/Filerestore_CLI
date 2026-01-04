#include "OverwriteDetector.h"
#include "OverwriteDetectionThreadPool.h"
#include "Logger.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <winioctl.h>
#include <ntddstor.h>
using namespace std;
using namespace std::chrono;

OverwriteDetector::OverwriteDetector(MFTReader* mftReader)
    : reader(mftReader), detectionMode(MODE_BALANCED),
      threadingStrategy(THREADING_AUTO),
      cachedStorageType(STORAGE_UNKNOWN), storageTypeDetected(false),
      threadPool(nullptr) {
    bytesPerCluster = reader->GetBytesPerSector() * reader->GetSectorsPerCluster();
    threadPool = new OverwriteDetectionThreadPool(this);
    LOG_DEBUG("OverwriteDetector initialized with multi-threading support");
}

OverwriteDetector::~OverwriteDetector() {
    if (threadPool) {
        delete threadPool;
        threadPool = nullptr;
    }
}

// 计算数据熵（用于判断数据是否为随机数据或被覆盖）
double OverwriteDetector::CalculateEntropy(const vector<BYTE>& data) {
    if (data.empty()) return 0.0;

    // 统计每个字节值的频率
    int frequency[256] = { 0 };
    for (BYTE b : data) {
        frequency[b]++;
    }

    // 计算香农熵
    double entropy = 0.0;
    size_t dataSize = data.size();

    for (int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            double probability = (double)frequency[i] / dataSize;
            entropy -= probability * log2(probability);
        }
    }

    return entropy;
}

// 检查数据是否全为0
bool OverwriteDetector::IsAllZeros(const vector<BYTE>& data) {
    return all_of(data.begin(), data.end(), [](BYTE b) { return b == 0; });
}

// 检查数据是否全为同一个值
bool OverwriteDetector::IsAllSameValue(const vector<BYTE>& data) {
    if (data.empty()) return true;
    BYTE firstByte = data[0];
    return all_of(data.begin(), data.end(), [firstByte](BYTE b) { return b == firstByte; });
}

// 检查数据是否包含有效的文件结构特征
bool OverwriteDetector::HasValidFileStructure(const vector<BYTE>& data) {
    if (data.size() < 16) return false;

    // 检查常见文件签名
    // JPEG: FF D8 FF
    if (data.size() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
        return true;
    }

    // PNG: 89 50 4E 47
    if (data.size() >= 4 && data[0] == 0x89 && data[1] == 0x50 &&
        data[2] == 0x4E && data[3] == 0x47) {
        return true;
    }

    // PDF: 25 50 44 46 (%PDF)
    if (data.size() >= 4 && data[0] == 0x25 && data[1] == 0x50 &&
        data[2] == 0x44 && data[3] == 0x46) {
        return true;
    }

    // ZIP/DOCX/XLSX/PPTX: 50 4B 03 04
    if (data.size() >= 4 && data[0] == 0x50 && data[1] == 0x4B &&
        data[2] == 0x03 && data[3] == 0x04) {
        return true;
    }

    // 7z: 37 7A BC AF 27 1C
    if (data.size() >= 6 && data[0] == 0x37 && data[1] == 0x7A &&
        data[2] == 0xBC && data[3] == 0xAF && data[4] == 0x27 && data[5] == 0x1C) {
        return true;
    }

    // RAR: 52 61 72 21 1A 07
    if (data.size() >= 6 && data[0] == 0x52 && data[1] == 0x61 &&
        data[2] == 0x72 && data[3] == 0x21 && data[4] == 0x1A && data[5] == 0x07) {
        return true;
    }

    // GZ: 1F 8B
    if (data.size() >= 2 && data[0] == 0x1F && data[1] == 0x8B) {
        return true;
    }

    // BZ2: 42 5A 68 (BZh)
    if (data.size() >= 3 && data[0] == 0x42 && data[1] == 0x5A && data[2] == 0x68) {
        return true;
    }

    // XZ: FD 37 7A 58 5A 00
    if (data.size() >= 6 && data[0] == 0xFD && data[1] == 0x37 &&
        data[2] == 0x7A && data[3] == 0x58 && data[4] == 0x5A && data[5] == 0x00) {
        return true;
    }

    // EXE/DLL: 4D 5A (MZ)
    if (data.size() >= 2 && data[0] == 0x4D && data[1] == 0x5A) {
        return true;
    }

    // MP4/MOV: 00 00 00 xx 66 74 79 70 (ftyp)
    if (data.size() >= 8 && data[4] == 0x66 && data[5] == 0x74 &&
        data[6] == 0x79 && data[7] == 0x70) {
        return true;
    }

    // AVI: 52 49 46 46 ... 41 56 49 20 (RIFF...AVI )
    if (data.size() >= 12 && data[0] == 0x52 && data[1] == 0x49 &&
        data[2] == 0x46 && data[3] == 0x46 && data[8] == 0x41 &&
        data[9] == 0x56 && data[10] == 0x49 && data[11] == 0x20) {
        return true;
    }

    // MP3: FF FB, FF FA, FF F3, FF F2, or ID3
    if (data.size() >= 3) {
        if ((data[0] == 0xFF && (data[1] & 0xE0) == 0xE0)) {
            return true;
        }
        if (data[0] == 0x49 && data[1] == 0x44 && data[2] == 0x33) { // ID3
            return true;
        }
    }

    // 检查是否包含可打印的ASCII文本（可能是文本文件）
    int printableCount = 0;
    int sampleSize = min((size_t)256, data.size());
    for (int i = 0; i < sampleSize; i++) {
        BYTE b = data[i];
        if ((b >= 0x20 && b <= 0x7E) || b == 0x09 || b == 0x0A || b == 0x0D) {
            printableCount++;
        }
    }

    // 如果超过70%是可打印字符，认为是文本文件
    if (printableCount > sampleSize * 0.7) {
        return true;
    }

    return false;
}

// 检查簇是否已分配给其他文件（通过读取$Bitmap）
bool OverwriteDetector::CheckClusterAllocation(ULONGLONG clusterNumber) {
    // TODO: 实现$Bitmap读取功能
    // 这需要读取MFT记录#6 ($Bitmap)，检查对应位是否为1
    // 暂时返回false（假设未分配）
    return false;
}

// 读取可变长度的无符号整数
ULONGLONG OverwriteDetector::ReadVariableLength(const BYTE* data, int length) {
    ULONGLONG result = 0;
    for (int i = 0; i < length && i < 8; i++) {
        result |= ((ULONGLONG)data[i]) << (i * 8);
    }
    return result;
}

// 读取可变长度的有符号整数
LONGLONG OverwriteDetector::ReadVariableLengthSigned(const BYTE* data, int length) {
    LONGLONG result = 0;
    for (int i = 0; i < length && i < 8; i++) {
        result |= ((LONGLONG)data[i]) << (i * 8);
    }

    // 符号扩展
    if (length > 0 && length < 8) {
        if (data[length - 1] & 0x80) {
            for (int i = length; i < 8; i++) {
                result |= ((LONGLONG)0xFF) << (i * 8);
            }
        }
    }

    return result;
}

// 从MFT记录中提取Data Runs
bool OverwriteDetector::ExtractDataRuns(const vector<BYTE>& mftRecord,
                                       vector<pair<ULONGLONG, ULONGLONG>>& runs) {
    runs.clear();

    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        LOG_ERROR("MFT record too small");
        return false;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // 遍历属性，查找$DATA属性
    BYTE* attrPtr = (BYTE*)mftRecord.data() + header->FirstAttributeOffset;
    BYTE* recordEnd = (BYTE*)mftRecord.data() + header->UsedSize;

    while (attrPtr < recordEnd) {
        if (attrPtr + sizeof(ATTRIBUTE_HEADER) > recordEnd) {
            break;
        }

        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            break;
        }

        if (attr->Length == 0 || attr->Length > (DWORD)(recordEnd - attrPtr)) {
            break;
        }

        // 找到$DATA属性且为非常驻
        if (attr->Type == AttributeData && attr->NonResident) {
            PNONRESIDENT_ATTRIBUTE nonResAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));

            // 解析Data Runs
            BYTE* runList = attrPtr + sizeof(ATTRIBUTE_HEADER) + nonResAttr->DataRunOffset;
            ULONGLONG currentLCN = 0;

            while (runList < recordEnd && *runList != 0) {
                BYTE header = *runList++;

                int lengthBytes = header & 0x0F;
                int offsetBytes = (header >> 4) & 0x0F;

                if (lengthBytes == 0 || lengthBytes > 8 || offsetBytes > 8) {
                    break;
                }

                if (runList + lengthBytes + offsetBytes > recordEnd) {
                    break;
                }

                // 读取簇数量
                ULONGLONG runLength = ReadVariableLength(runList, lengthBytes);
                runList += lengthBytes;

                // 读取LCN偏移
                LONGLONG lcnOffset = ReadVariableLengthSigned(runList, offsetBytes);
                runList += offsetBytes;

                currentLCN += lcnOffset;

                runs.push_back(make_pair(currentLCN, runLength));
            }

            return true;
        }

        attrPtr += attr->Length;
    }

    return false;
}

// 检测单个簇是否被覆盖
ClusterStatus OverwriteDetector::CheckCluster(ULONGLONG clusterNumber) {
    ClusterStatus status;
    status.clusterNumber = clusterNumber;
    status.isOverwritten = false;
    status.isAllocated = false;
    status.dataEntropy = 0.0;
    status.overwriteReason = "Unknown";

    // 读取簇数据
    vector<BYTE> clusterData;
    if (!reader->ReadClusters(clusterNumber, 1, clusterData)) {
        status.isOverwritten = true;
        status.overwriteReason = "Failed to read cluster";
        LOG_WARNING_FMT("Failed to read cluster %llu", clusterNumber);
        return status;
    }

    // 检查1: 全为0（快速格式化或被清零）
    if (IsAllZeros(clusterData)) {
        status.isOverwritten = true;
        status.dataEntropy = 0.0;
        status.overwriteReason = "All zeros (formatted or wiped)";
        return status;
    }

    // 检查2: 全为同一个值（某些擦除工具）
    if (IsAllSameValue(clusterData)) {
        status.isOverwritten = true;
        status.dataEntropy = 0.0;
        status.overwriteReason = "All same value (wiped)";
        return status;
    }

    // 检查3: 计算熵值
    status.dataEntropy = CalculateEntropy(clusterData);

    // 高熵值（>7.5）可能表示：
    // - 压缩数据
    // - 加密数据
    // - 随机覆盖
    // 低熵值（<3.0）可能表示：
    // - 重复数据
    // - 简单文本
    // - 部分清零

    // 检查4: 是否包含有效的文件结构
    bool hasValidStructure = HasValidFileStructure(clusterData);

    // 检查5: 是否已分配给其他文件
    status.isAllocated = CheckClusterAllocation(clusterNumber);

    // ========== 改进的综合判断逻辑 ==========
    // 对于高熵文件（压缩/加密），提高阈值以减少误判

    if (status.dataEntropy > 7.95) {
        // 极高的熵值（接近理论最大值8.0），极可能是真正的随机数据
        if (!hasValidStructure) {
            status.isOverwritten = true;
            status.overwriteReason = "Extremely high entropy (>7.95), likely random/encrypted data";
        } else {
            status.isOverwritten = false;
            status.overwriteReason = "Very high entropy but has valid structure (compressed/encrypted file)";
        }
    } else if (status.dataEntropy > 7.5) {
        // 非常高的熵值（典型的压缩/加密文件范围）
        // 在这个范围内，优先考虑为合法的压缩/加密数据
        if (hasValidStructure) {
            status.isOverwritten = false;
            status.overwriteReason = "High entropy with valid structure (compressed/encrypted file)";
        } else {
            // 即使没有识别的文件结构，也不能确定是覆盖
            // 可能是：1) 碎片化文件的中间簇 2) 未识别的压缩格式
            status.isOverwritten = false;
            status.overwriteReason = "High entropy (7.5-7.95), likely compressed/encrypted data";
        }
    } else if (status.dataEntropy < 1.0) {
        // 非常低的熵值，可能被清零或填充
        status.isOverwritten = true;
        status.overwriteReason = "Very low entropy (<1.0), likely wiped";
    } else if (status.isAllocated) {
        // 已分配给其他文件
        status.isOverwritten = true;
        status.overwriteReason = "Cluster allocated to another file";
    } else if (hasValidStructure) {
        // 包含有效的文件结构
        status.isOverwritten = false;
        status.overwriteReason = "Contains valid file structure";
    } else {
        // 中等熵值 (1.0 - 7.5)
        // 只有在熵值非常高 (>7.2) 且无任何文件特征时才判定为覆盖
        if (status.dataEntropy > 7.2) {
            status.isOverwritten = true;
            status.overwriteReason = "High entropy (>7.2) without recognizable structure";
        } else {
            status.isOverwritten = false;
            status.overwriteReason = "Medium entropy, possibly original data";
        }
    }

    return status;
}

// 批量检测多个簇
vector<ClusterStatus> OverwriteDetector::CheckClusters(const vector<ULONGLONG>& clusterNumbers) {
    vector<ClusterStatus> results;
    results.reserve(clusterNumbers.size());

    for (ULONGLONG clusterNum : clusterNumbers) {
        results.push_back(CheckCluster(clusterNum));
    }

    return results;
}

// 获取可读的覆盖检测报告
string OverwriteDetector::GetDetectionReport(const OverwriteDetectionResult& result) {
    ostringstream report;

    report << "=== Overwrite Detection Report ===" << endl;

    // 特殊情况：MFT记录无效
    if (result.totalClusters == 1 && result.overwrittenClusters == 1 && result.overwritePercentage == 100.0) {
        report << "Status: [ERROR] Invalid MFT Record" << endl;
        report << "The MFT record header is corrupted or invalid." << endl;
        report << "This record may have been overwritten or never contained valid data." << endl;
        report << "===================================" << endl;
        return report.str();
    }

    // 特殊情况：常驻文件
    if (result.totalClusters == 0 && result.isFullyAvailable) {
        report << "File Type: Resident File (data stored in MFT)" << endl;
        report << "Total Clusters: 0 (no external data clusters)" << endl;
        report << "Status: FULLY AVAILABLE - Data is in MFT record" << endl;
        report << "Detection Time: " << fixed << setprecision(2) << result.detectionTimeMs << " ms" << endl;
        report << "===================================" << endl;
        return report.str();
    }

    // 正常情况：非常驻文件
    report << "Total Clusters: " << result.totalClusters << endl;

    // 显示采样信息
    if (result.usedSampling) {
        report << "Sampled Clusters: " << result.sampledClusters
               << " (" << (result.sampledClusters * 100 / max(1ULL, result.totalClusters)) << "% sampled)" << endl;
    }

    report << "Available Clusters: " << result.availableClusters << endl;
    report << "Overwritten Clusters: " << result.overwrittenClusters << endl;
    report << "Overwrite Percentage: " << fixed << setprecision(2) << result.overwritePercentage << "%" << endl;

    // 显示存储类型
    report << "Storage Type: " << GetStorageTypeName(result.detectedStorageType) << endl;

    // 显示多线程信息
    if (result.usedMultiThreading) {
        report << "Multi-Threading: Enabled (" << result.threadCount << " threads)" << endl;
    } else {
        report << "Multi-Threading: Disabled" << endl;
    }

    // 显示检测耗时
    report << "Detection Time: " << fixed << setprecision(2) << result.detectionTimeMs << " ms" << endl;

    report << endl;

    if (result.isFullyAvailable) {
        report << "Status: FULLY AVAILABLE - All data can be recovered" << endl;
    } else if (result.isPartiallyAvailable) {
        report << "Status: PARTIALLY AVAILABLE - Some data can be recovered" << endl;
        report << "Recovery Possibility: " << fixed << setprecision(1)
               << (100.0 - result.overwritePercentage) << "%" << endl;
    } else {
        report << "Status: NOT AVAILABLE - Data has been completely overwritten" << endl;
    }

    report << endl;

    // 显示前10个簇的详细信息
    if (!result.clusterStatuses.empty()) {
        report << "Cluster Details (first 10):" << endl;
        size_t displayCount = min((size_t)10, result.clusterStatuses.size());
        for (size_t i = 0; i < displayCount; i++) {
            const ClusterStatus& status = result.clusterStatuses[i];
            report << "  Cluster " << status.clusterNumber << ": ";
            report << (status.isOverwritten ? "OVERWRITTEN" : "AVAILABLE");
            report << " (Entropy: " << fixed << setprecision(2) << status.dataEntropy << ")";
            report << " - " << status.overwriteReason << endl;
        }

        if (result.clusterStatuses.size() > 10) {
            report << "  ... and " << (result.clusterStatuses.size() - 10) << " more clusters" << endl;
        }
    } else if (result.usedSampling) {
        report << "Note: Detailed cluster information not available in sampling mode" << endl;
    }

    report << "===================================" << endl;

    return report.str();
}

// ==================== 新增优化功能实现 ====================

// 从指针计算熵值（用于批量处理）
double OverwriteDetector::CalculateEntropyFromPointer(const BYTE* data, size_t size) {
    if (size == 0) return 0.0;

    int frequency[256] = { 0 };
    for (size_t i = 0; i < size; i++) {
        frequency[data[i]]++;
    }

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            double probability = (double)frequency[i] / size;
            entropy -= probability * log2(probability);
        }
    }

    return entropy;
}

// 从指针检查是否全为0
bool OverwriteDetector::IsAllZerosFromPointer(const BYTE* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

// 从指针检查是否全为同一个值
bool OverwriteDetector::IsAllSameValueFromPointer(const BYTE* data, size_t size) {
    if (size == 0) return true;
    BYTE firstByte = data[0];
    for (size_t i = 1; i < size; i++) {
        if (data[i] != firstByte) return false;
    }
    return true;
}

// 从指针检查是否包含有效的文件结构
bool OverwriteDetector::HasValidFileStructureFromPointer(const BYTE* data, size_t size) {
    if (size < 16) return false;

    // 检查常见文件签名 (与 HasValidFileStructure 保持一致)
    if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) return true;  // JPEG
    if (size >= 4 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) return true;  // PNG
    if (size >= 4 && data[0] == 0x25 && data[1] == 0x50 && data[2] == 0x44 && data[3] == 0x46) return true;  // PDF
    if (size >= 4 && data[0] == 0x50 && data[1] == 0x4B && data[2] == 0x03 && data[3] == 0x04) return true;  // ZIP
    if (size >= 6 && data[0] == 0x37 && data[1] == 0x7A && data[2] == 0xBC &&
        data[3] == 0xAF && data[4] == 0x27 && data[5] == 0x1C) return true;  // 7z
    if (size >= 6 && data[0] == 0x52 && data[1] == 0x61 && data[2] == 0x72 &&
        data[3] == 0x21 && data[4] == 0x1A && data[5] == 0x07) return true;  // RAR
    if (size >= 2 && data[0] == 0x1F && data[1] == 0x8B) return true;  // GZ
    if (size >= 3 && data[0] == 0x42 && data[1] == 0x5A && data[2] == 0x68) return true;  // BZ2
    if (size >= 6 && data[0] == 0xFD && data[1] == 0x37 && data[2] == 0x7A &&
        data[3] == 0x58 && data[4] == 0x5A && data[5] == 0x00) return true;  // XZ
    if (size >= 2 && data[0] == 0x4D && data[1] == 0x5A) return true;  // EXE/DLL
    if (size >= 8 && data[4] == 0x66 && data[5] == 0x74 &&
        data[6] == 0x79 && data[7] == 0x70) return true;  // MP4/MOV
    if (size >= 12 && data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46 &&
        data[8] == 0x41 && data[9] == 0x56 && data[10] == 0x49 && data[11] == 0x20) return true;  // AVI
    if (size >= 3) {
        if (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) return true;  // MP3
        if (data[0] == 0x49 && data[1] == 0x44 && data[2] == 0x33) return true;  // ID3
    }

    // 检查可打印文本
    int printableCount = 0;
    size_t sampleSize = min((size_t)256, size);
    for (size_t i = 0; i < sampleSize; i++) {
        BYTE b = data[i];
        if ((b >= 0x20 && b <= 0x7E) || b == 0x09 || b == 0x0A || b == 0x0D) {
            printableCount++;
        }
    }

    return (printableCount > sampleSize * 0.7);
}

// 存储类型检测
StorageType OverwriteDetector::DetectStorageType() {
    LOG_INFO("Detecting storage type...");

    HANDLE volumeHandle = reader->GetVolumeHandle();
    if (volumeHandle == INVALID_HANDLE_VALUE) {
        LOG_ERROR("Volume handle is invalid");
        return STORAGE_SSD;  // 默认返回 SSD
    }

    // 第一步：从卷句柄获取物理驱动器号
    VOLUME_DISK_EXTENTS diskExtents;
    ZeroMemory(&diskExtents, sizeof(diskExtents));
    DWORD bytesReturned = 0;

    BOOL result = DeviceIoControl(
        volumeHandle,
        IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
        NULL,
        0,
        &diskExtents,
        sizeof(diskExtents),
        &bytesReturned,
        NULL
    );

    if (!result) {
        DWORD error = GetLastError();
        LOG_WARNING_FMT("Failed to get physical drive number (Error: %lu), defaulting to SSD", error);
        return STORAGE_SSD;
    }

    if (diskExtents.NumberOfDiskExtents == 0) {
        LOG_WARNING("No disk extents found, defaulting to SSD");
        return STORAGE_SSD;
    }

    DWORD physicalDriveNumber = diskExtents.Extents[0].DiskNumber;
    LOG_INFO_FMT("Physical drive number: %lu", physicalDriveNumber);

    // 第二步：打开物理驱动器
    char physicalDrivePath[MAX_PATH];
    sprintf_s(physicalDrivePath, "\\\\.\\PhysicalDrive%lu", physicalDriveNumber);
    LOG_DEBUG_FMT("Opening physical drive: %s", physicalDrivePath);

    HANDLE hPhysicalDrive = CreateFileA(
        physicalDrivePath,
        0,  // 不需要读写权限，只需要查询
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );

    if (hPhysicalDrive == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        LOG_WARNING_FMT("Failed to open physical drive (Error: %lu), defaulting to SSD", error);
        return STORAGE_SSD;
    }

    // 第三步：查询存储属性
    STORAGE_PROPERTY_QUERY query;
    ZeroMemory(&query, sizeof(query));
    query.PropertyId = StorageDeviceSeekPenaltyProperty;
    query.QueryType = PropertyStandardQuery;

    DEVICE_SEEK_PENALTY_DESCRIPTOR seekPenaltyDesc;
    ZeroMemory(&seekPenaltyDesc, sizeof(seekPenaltyDesc));

    bytesReturned = 0;
    result = DeviceIoControl(
        hPhysicalDrive,
        IOCTL_STORAGE_QUERY_PROPERTY,
        &query,
        sizeof(query),
        &seekPenaltyDesc,
        sizeof(seekPenaltyDesc),
        &bytesReturned,
        nullptr
    );

    StorageType type = STORAGE_SSD;  // 默认 SSD

    if (result && bytesReturned >= sizeof(DEVICE_SEEK_PENALTY_DESCRIPTOR)) {
        if (seekPenaltyDesc.IncursSeekPenalty) {
            type = STORAGE_HDD;
            LOG_INFO("Storage Type Detected: HDD (mechanical hard drive)");
        } else {
            // 进一步区分SATA SSD和NVMe SSD
            double latencyMs = MeasureRandomReadLatency();
            LOG_INFO_FMT("Measured random read latency: %.2f ms", latencyMs);
            if (latencyMs < 1.0) {
                type = STORAGE_NVME;
                LOG_INFO("Storage Type Detected: NVMe SSD");
            } else {
                type = STORAGE_SSD;
                LOG_INFO("Storage Type Detected: SATA SSD");
            }
        }
    } else {
        DWORD error = GetLastError();
        LOG_WARNING_FMT("Failed to query storage properties (Error: %lu), defaulting to SSD", error);
        type = STORAGE_SSD;
    }

    CloseHandle(hPhysicalDrive);
    return type;
}

// 测量随机读取延迟
double OverwriteDetector::MeasureRandomReadLatency() {
    const int TEST_COUNT = 10;
    vector<double> latencies;

    // 获取MFT总记录数
    ULONGLONG totalRecords = reader->GetTotalMFTRecords();
    if (totalRecords < 1000) {
        LOG_WARNING("Too few MFT records for accurate storage detection");
        return 1.0; // 默认假设为SSD
    }

    // 随机读取10个MFT记录，测量延迟
    for (int i = 0; i < TEST_COUNT; i++) {
        // 随机选择一个记录号（避开前16个系统记录）
        ULONGLONG recordNum = 16 + (rand() % (totalRecords - 16));

        vector<BYTE> record;
        auto start = high_resolution_clock::now();
        bool success = reader->ReadMFT(recordNum, record);
        auto end = high_resolution_clock::now();

        if (success) {
            double latency = duration_cast<microseconds>(end - start).count() / 1000.0;
            latencies.push_back(latency);
        }
    }

    if (latencies.empty()) {
        LOG_WARNING("Failed to measure storage latency");
        return 1.0;
    }

    // 计算平均延迟
    double sum = 0.0;
    for (double lat : latencies) {
        sum += lat;
    }

    return sum / latencies.size();
}

// 获取存储类型名称
string OverwriteDetector::GetStorageTypeName(StorageType type) {
    switch (type) {
        case STORAGE_HDD: return "HDD (Mechanical Hard Drive)";
        case STORAGE_SSD: return "SATA SSD";
        case STORAGE_NVME: return "NVMe SSD";
        default: return "Unknown";
    }
}

// 获取存储类型（带缓存）
StorageType OverwriteDetector::GetStorageType() {
    if (!storageTypeDetected) {
        cachedStorageType = DetectStorageType();
        storageTypeDetected = true;
    }
    return cachedStorageType;
}

// 从内存中检测单个簇（批量处理优化）
ClusterStatus OverwriteDetector::CheckClusterFromMemory(const BYTE* clusterData, ULONGLONG clusterNumber) {
    ClusterStatus status;
    status.clusterNumber = clusterNumber;
    status.isOverwritten = false;
    status.isAllocated = false;
    status.dataEntropy = 0.0;
    status.overwriteReason = "Unknown";

    // 检查1: 全为0
    if (IsAllZerosFromPointer(clusterData, bytesPerCluster)) {
        status.isOverwritten = true;
        status.dataEntropy = 0.0;
        status.overwriteReason = "All zeros (formatted or wiped)";
        return status;
    }

    // 检查2: 全为同一个值
    if (IsAllSameValueFromPointer(clusterData, bytesPerCluster)) {
        status.isOverwritten = true;
        status.dataEntropy = 0.0;
        status.overwriteReason = "All same value (wiped)";
        return status;
    }

    // 检查3: 计算熵值
    status.dataEntropy = CalculateEntropyFromPointer(clusterData, bytesPerCluster);

    // 检查4: 是否包含有效的文件结构
    bool hasValidStructure = HasValidFileStructureFromPointer(clusterData, bytesPerCluster);

    // ========== 改进的综合判断逻辑 (与 CheckCluster 保持一致) ==========
    if (status.dataEntropy > 7.95) {
        // 极高的熵值（接近理论最大值8.0）
        if (!hasValidStructure) {
            status.isOverwritten = true;
            status.overwriteReason = "Extremely high entropy (>7.95), likely random/encrypted data";
        } else {
            status.isOverwritten = false;
            status.overwriteReason = "Very high entropy but has valid structure (compressed/encrypted file)";
        }
    } else if (status.dataEntropy > 7.5) {
        // 非常高的熵值（典型的压缩/加密文件范围）
        if (hasValidStructure) {
            status.isOverwritten = false;
            status.overwriteReason = "High entropy with valid structure (compressed/encrypted file)";
        } else {
            // 优先假设为合法的压缩/加密数据
            status.isOverwritten = false;
            status.overwriteReason = "High entropy (7.5-7.95), likely compressed/encrypted data";
        }
    } else if (status.dataEntropy < 1.0) {
        status.isOverwritten = true;
        status.overwriteReason = "Very low entropy (<1.0), likely wiped";
    } else if (hasValidStructure) {
        status.isOverwritten = false;
        status.overwriteReason = "Contains valid file structure";
    } else {
        // 中等熵值 (1.0 - 7.5)
        if (status.dataEntropy > 7.2) {
            status.isOverwritten = true;
            status.overwriteReason = "High entropy (>7.2) without recognizable structure";
        } else {
            status.isOverwritten = false;
            status.overwriteReason = "Medium entropy, possibly original data";
        }
    }

    return status;
}

// 批量检测簇（优化版本）
vector<ClusterStatus> OverwriteDetector::BatchCheckClusters(const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns) {
    vector<ClusterStatus> results;
    ULONGLONG totalChecked = 0;

    LOG_INFO("Starting batch cluster checking...");

    for (const auto& run : dataRuns) {
        ULONGLONG startLCN = run.first;
        ULONGLONG clusterCount = run.second;

        LOG_DEBUG_FMT("Processing run: LCN=%llu, Count=%llu", startLCN, clusterCount);

        // 批量读取整个run
        vector<BYTE> batchData;
        if (!reader->ReadClusters(startLCN, clusterCount, batchData)) {
            LOG_WARNING_FMT("Failed to read clusters at LCN %llu", startLCN);
            // 标记这些簇为读取失败
            for (ULONGLONG i = 0; i < clusterCount; i++) {
                ClusterStatus status;
                status.clusterNumber = startLCN + i;
                status.isOverwritten = true;
                status.overwriteReason = "Failed to read cluster";
                results.push_back(status);
            }
            continue;
        }

        // 在内存中处理每个簇
        for (ULONGLONG i = 0; i < clusterCount; i++) {
            BYTE* clusterPtr = batchData.data() + i * bytesPerCluster;
            ClusterStatus status = CheckClusterFromMemory(clusterPtr, startLCN + i);
            results.push_back(status);

            totalChecked++;

            // 智能跳过：如果连续10个簇都被覆盖，可能整个文件都被覆盖了
            if (ShouldSkipRemaining(results)) {
                LOG_INFO_FMT("Smart skip triggered after %llu clusters", totalChecked);
                // 标记剩余簇为可能被覆盖
                for (ULONGLONG j = i + 1; j < clusterCount; j++) {
                    ClusterStatus skipStatus;
                    skipStatus.clusterNumber = startLCN + j;
                    skipStatus.isOverwritten = true;
                    skipStatus.overwriteReason = "Skipped (likely overwritten based on pattern)";
                    results.push_back(skipStatus);
                }
                return results; // 提前返回
            }
        }
    }

    LOG_INFO_FMT("Batch checking completed: %llu clusters checked", totalChecked);
    return results;
}

// 采样检测簇
vector<ClusterStatus> OverwriteDetector::SamplingCheckClusters(const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
                                                               ULONGLONG totalClusters) {
    vector<ClusterStatus> results;

    // 计算采样间隔（检测约1%的簇，最少10个，最多1000个）
    ULONGLONG sampleCount = max(10ULL, min(1000ULL, totalClusters / 100));
    ULONGLONG sampleInterval = max(1ULL, totalClusters / sampleCount);

    LOG_INFO_FMT("Sampling mode: checking %llu out of %llu clusters (interval: %llu)",
                 sampleCount, totalClusters, sampleInterval);

    ULONGLONG currentCluster = 0;
    ULONGLONG nextSampleAt = 0;

    for (const auto& run : dataRuns) {
        ULONGLONG startLCN = run.first;
        ULONGLONG clusterCount = run.second;

        for (ULONGLONG i = 0; i < clusterCount; i++) {
            if (currentCluster == nextSampleAt) {
                // 这是一个采样点，检测这个簇
                ClusterStatus status = CheckCluster(startLCN + i);
                results.push_back(status);

                nextSampleAt += sampleInterval;

                if (results.size() >= sampleCount) {
                    LOG_INFO_FMT("Sampling completed: %llu samples collected", results.size());
                    return results;
                }
            }
            currentCluster++;
        }
    }

    LOG_INFO_FMT("Sampling completed: %llu samples collected", results.size());
    return results;
}

// 智能跳过判断
bool OverwriteDetector::ShouldSkipRemaining(const vector<ClusterStatus>& results) {
    if (results.size() < 10) return false;

    // 检查最后10个簇是否都被覆盖
    int consecutiveOverwritten = 0;
    for (size_t i = results.size() - 10; i < results.size(); i++) {
        if (results[i].isOverwritten) {
            consecutiveOverwritten++;
        }
    }

    // 如果最后10个簇都被覆盖，认为剩余部分也被覆盖
    return (consecutiveOverwritten >= 10);
}

// ==================== 多线程和智能策略实现 ====================

// 判断是否应该使用多线程
bool OverwriteDetector::ShouldUseMultiThreading(ULONGLONG clusterCount, StorageType storageType) {
    // 根据多线程策略决定
    if (threadingStrategy == THREADING_DISABLED) {
        return false;
    }

    if (threadingStrategy == THREADING_ENABLED) {
        return true;
    }

    // THREADING_AUTO: 自动决定
    // 1. 簇数量太少，不值得多线程
    if (clusterCount < 100) {
        LOG_DEBUG("Cluster count too small for multi-threading");
        return false;
    }

    // 2. 根据存储类型决定
    switch (storageType) {
        case STORAGE_HDD:
            // HDD不使用多线程（会导致随机I/O）
            LOG_DEBUG("HDD detected: multi-threading disabled");
            return false;

        case STORAGE_SSD:
            // SSD：中等文件使用多线程
            if (clusterCount >= 1000) {
                LOG_DEBUG("SSD detected with sufficient clusters: multi-threading enabled");
                return true;
            }
            return false;

        case STORAGE_NVME:
            // NVMe：小文件也可以使用多线程
            if (clusterCount >= 500) {
                LOG_DEBUG("NVMe detected with sufficient clusters: multi-threading enabled");
                return true;
            }
            return false;

        default:
            // 未知存储类型，保守处理
            return (clusterCount >= 2000);
    }
}

// 获取最优线程数
int OverwriteDetector::GetOptimalThreadCount(ULONGLONG clusterCount, StorageType storageType) {
    // 获取CPU核心数
    int cpuCores = thread::hardware_concurrency();
    if (cpuCores == 0) cpuCores = 4; // 默认假设4核

    int optimalThreads = 4; // 默认值

    switch (storageType) {
        case STORAGE_HDD:
            // HDD不应该使用多线程
            optimalThreads = 1;
            break;

        case STORAGE_SSD:
            // SSD：最多4个线程
            optimalThreads = min(4, cpuCores);
            break;

        case STORAGE_NVME:
            // NVMe：可以使用更多线程
            optimalThreads = min(8, cpuCores);
            break;

        default:
            optimalThreads = min(4, cpuCores);
            break;
    }

    // 根据簇数量调整
    // 每个线程至少处理50个簇
    int maxThreadsByClusterCount = max(1, (int)(clusterCount / 50));
    optimalThreads = min(optimalThreads, maxThreadsByClusterCount);

    LOG_INFO_FMT("Optimal thread count: %d (CPU cores: %d, Storage: %s, Clusters: %llu)",
                 optimalThreads, cpuCores, GetStorageTypeName(storageType).c_str(), clusterCount);

    return optimalThreads;
}

// 多线程检测簇
vector<ClusterStatus> OverwriteDetector::MultiThreadedCheckClusters(
    const vector<ULONGLONG>& clusterNumbers, int threadCount) {

    LOG_INFO_FMT("Starting multi-threaded detection with %d threads for %zu clusters",
                 threadCount, clusterNumbers.size());

    // 使用线程池执行检测
    vector<ClusterStatus> results = threadPool->DetectClusters(clusterNumbers, threadCount);

    LOG_INFO_FMT("Multi-threaded detection completed: %zu clusters processed", results.size());

    return results;
}

// 更新主检测函数以支持多线程
OverwriteDetectionResult OverwriteDetector::DetectOverwrite(const vector<BYTE>& mftRecord) {
    auto startTime = high_resolution_clock::now();

    OverwriteDetectionResult result;
    result.totalClusters = 0;
    result.overwrittenClusters = 0;
    result.availableClusters = 0;
    result.sampledClusters = 0;
    result.overwritePercentage = 0.0;
    result.isFullyAvailable = false;
    result.isPartiallyAvailable = false;
    result.usedSampling = false;
    result.usedMultiThreading = false;
    result.threadCount = 1;
    result.detectedStorageType = STORAGE_UNKNOWN;
    result.detectionTimeMs = 0.0;

    LOG_DEBUG("Starting optimized overwrite detection with multi-threading support");

    // 首先验证MFT记录头
    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        LOG_ERROR("MFT record too small");
        result.isFullyAvailable = false;
        result.isPartiallyAvailable = false;
        result.overwritePercentage = 100.0;
        result.overwrittenClusters = 1;
        result.totalClusters = 1;
        return result;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // ========== 宽松的记录头验证策略 ==========
    // 已删除文件的MFT记录头可能被部分覆盖，但数据簇可能仍然完好
    // 采用多层次验证策略，而非一刀切

    bool headerPartiallyCorrupted = false;
    string corruptionWarning;

    // 检查1：签名验证（宽松）
    if (header->Signature != 'ELIF') {  // "FILE" in little-endian
        headerPartiallyCorrupted = true;
        corruptionWarning = "MFT record signature invalid, but attempting to continue";
        LOG_WARNING_FMT("MFT signature: 0x%08X (expected 0x454C4946)", header->Signature);
    }

    // 检查2：UsedSize 合理性验证
    if (header->UsedSize == 0) {
        // UsedSize为0，记录完全为空或被清零
        LOG_ERROR("MFT record UsedSize is 0, record appears to be wiped");
        result.isFullyAvailable = false;
        result.overwritePercentage = 100.0;
        result.overwrittenClusters = 1;
        result.totalClusters = 1;
        return result;
    }

    if (header->UsedSize > mftRecord.size()) {
        // UsedSize超出记录大小，可能是损坏
        // 但尝试使用记录的实际大小继续
        headerPartiallyCorrupted = true;
        corruptionWarning += "; UsedSize exceeds record size, using actual size";
        LOG_WARNING_FMT("UsedSize (%u) > actual size (%zu), attempting recovery",
                       header->UsedSize, mftRecord.size());
        // 不返回，继续尝试
    }

    // 检查3：FirstAttributeOffset 合理性验证
    if (header->FirstAttributeOffset < sizeof(FILE_RECORD_HEADER)) {
        LOG_ERROR_FMT("FirstAttributeOffset (%u) too small, record severely corrupted",
                     header->FirstAttributeOffset);
        result.isFullyAvailable = false;
        result.overwritePercentage = 100.0;
        result.overwrittenClusters = 1;
        result.totalClusters = 1;
        return result;
    }

    if (header->FirstAttributeOffset >= mftRecord.size()) {
        LOG_ERROR("FirstAttributeOffset out of bounds, record severely corrupted");
        result.isFullyAvailable = false;
        result.overwritePercentage = 100.0;
        result.overwrittenClusters = 1;
        result.totalClusters = 1;
        return result;
    }

    // 如果检测到部分损坏，记录警告但继续尝试
    if (headerPartiallyCorrupted) {
        LOG_WARNING("MFT record header partially corrupted, but attempting data recovery");
        LOG_WARNING(corruptionWarning);
        cout << "Warning: " << corruptionWarning << endl;
        cout << "Attempting to recover data despite header corruption..." << endl;
    }

    // 提取Data Runs
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;
    if (!ExtractDataRuns(mftRecord, dataRuns)) {
        LOG_WARNING("No data runs found - file is resident (data stored in MFT)");
        // 对于常驻文件，数据在MFT记录中，检查记录是否有效即可
        result.isFullyAvailable = true;
        result.totalClusters = 0;
        result.availableClusters = 0;
        result.overwrittenClusters = 0;
        result.overwritePercentage = 0.0;
        return result;
    }

    // 计算总簇数
    for (const auto& run : dataRuns) {
        result.totalClusters += run.second;
    }

    LOG_INFO_FMT("Total clusters to check: %llu", result.totalClusters);

    // 检测存储类型
    result.detectedStorageType = GetStorageType();
    LOG_INFO_FMT("Storage type: %s", GetStorageTypeName(result.detectedStorageType).c_str());

    // 根据检测模式和文件大小选择策略
    bool useSampling = false;
    bool useMultiThreading = false;
    bool useBatchReading = true;

    if (detectionMode == MODE_FAST) {
        useSampling = true;
        useMultiThreading = false; // 采样模式不需要多线程
    } else if (detectionMode == MODE_BALANCED) {
        // 平衡模式：大文件使用采样
        useSampling = (result.totalClusters > 10000);
        // 判断是否使用多线程
        if (!useSampling) {
            useMultiThreading = ShouldUseMultiThreading(result.totalClusters, result.detectedStorageType);
        }
    } else {
        // 完整模式：总是检测所有簇
        useSampling = false;
        // 判断是否使用多线程
        useMultiThreading = ShouldUseMultiThreading(result.totalClusters, result.detectedStorageType);
    }

    // 获取最优线程数
    if (useMultiThreading) {
        result.threadCount = GetOptimalThreadCount(result.totalClusters, result.detectedStorageType);
        result.usedMultiThreading = true;
    }

    LOG_INFO_FMT("Detection strategy: Sampling=%s, MultiThreading=%s (threads=%d), BatchReading=%s",
                 useSampling ? "Yes" : "No",
                 useMultiThreading ? "Yes" : "No",
                 result.threadCount,
                 useBatchReading ? "Yes" : "No");

    // 执行检测
    vector<ClusterStatus> clusterStatuses;

    if (useSampling) {
        result.usedSampling = true;
        clusterStatuses = SamplingCheckClusters(dataRuns, result.totalClusters);
        result.sampledClusters = clusterStatuses.size();
    } else if (useMultiThreading) {
        // 多线程模式：收集所有簇号
        vector<ULONGLONG> allClusterNumbers;
        for (const auto& run : dataRuns) {
            for (ULONGLONG i = 0; i < run.second; i++) {
                allClusterNumbers.push_back(run.first + i);
            }
        }

        clusterStatuses = MultiThreadedCheckClusters(allClusterNumbers, result.threadCount);
        result.sampledClusters = clusterStatuses.size();
    } else {
        result.usedSampling = false;
        if (useBatchReading) {
            clusterStatuses = BatchCheckClusters(dataRuns);
        } else {
            // 逐个检测（旧方法，保留用于对比）
            for (const auto& run : dataRuns) {
                for (ULONGLONG i = 0; i < run.second; i++) {
                    clusterStatuses.push_back(CheckCluster(run.first + i));
                }
            }
        }
        result.sampledClusters = clusterStatuses.size();
    }

    // 统计结果
    for (const auto& status : clusterStatuses) {
        if (status.isOverwritten) {
            result.overwrittenClusters++;
        } else {
            result.availableClusters++;
        }
    }

    // 如果使用采样，根据样本推断总体情况
    if (useSampling && result.sampledClusters > 0) {
        double sampleOverwriteRate = (double)result.overwrittenClusters / result.sampledClusters;
        result.overwrittenClusters = (ULONGLONG)(result.totalClusters * sampleOverwriteRate);
        result.availableClusters = result.totalClusters - result.overwrittenClusters;
    }

    // 计算覆盖百分比
    if (result.totalClusters > 0) {
        result.overwritePercentage = (double)result.overwrittenClusters / result.totalClusters * 100.0;
    }

    // 判断可用性
    result.isFullyAvailable = (result.overwrittenClusters == 0);
    result.isPartiallyAvailable = (result.availableClusters > 0 && result.overwrittenClusters > 0);

    // 保存簇状态（如果不是采样模式，或者采样数量不多）
    if (!useSampling || clusterStatuses.size() <= 1000) {
        result.clusterStatuses = clusterStatuses;
    }

    // 计算耗时
    auto endTime = high_resolution_clock::now();
    result.detectionTimeMs = duration_cast<milliseconds>(endTime - startTime).count();

    LOG_INFO_FMT("Detection completed in %.2f ms: Total=%llu, Available=%llu, Overwritten=%llu (%.2f%%), MultiThreading=%s",
                 result.detectionTimeMs, result.totalClusters, result.availableClusters,
                 result.overwrittenClusters, result.overwritePercentage,
                 result.usedMultiThreading ? "Yes" : "No");

    return result;
}
