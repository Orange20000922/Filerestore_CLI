#include "MFTReader.h"
#include "Logger.h"
#include <iostream>
#include <winioctl.h>

using namespace std;

MFTReader::MFTReader() :
    bytesPerSector(0), sectorsPerCluster(0),
    mftStartLCN(0), bytesPerFileRecord(0), totalClusters(0),
    mftDataRunsLoaded(false) {
}

MFTReader::~MFTReader() {
    // ScopedHandle 自动关闭 hVolume
}

bool MFTReader::OpenVolume(char driveLetter) {
    LOG_DEBUG_FMT("MFTReader::OpenVolume called for drive %c:", driveLetter);

    char volumePath[MAX_PATH];
    sprintf_s(volumePath, "\\\\.\\%c:", driveLetter);
    LOG_DEBUG_FMT("Volume path: %s", volumePath);

    LOG_DEBUG("Attempting to open volume handle...");
    hVolume.Reset(CreateFileA(volumePath, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
        OPEN_EXISTING, 0, NULL));

    if (!hVolume.IsValid()) {
        DWORD error = GetLastError();
        string msg = "打开卷失败。错误代码: " + to_string(error);
        cout << msg << endl;
        cout << "访问原始卷需要管理员权限。" << endl;
        LOG_ERROR(msg);
        return false;
    }
    LOG_DEBUG("Volume handle opened successfully");

    // 读取引导扇区
    LOG_DEBUG("Reading boot sector...");
    NTFS_BOOT_SECTOR bootSector;
    DWORD bytesRead;
    if (!ReadFile(hVolume, &bootSector, sizeof(NTFS_BOOT_SECTOR), &bytesRead, NULL)) {
        string msg = "读取引导扇区失败。";
        cout << msg << endl;
        LOG_ERROR_FMT("%s Error: %d", msg.c_str(), GetLastError());
        CloseVolume();
        return false;
    }
    LOG_DEBUG_FMT("Boot sector read successfully, %d bytes", bytesRead);

    // 验证NTFS签名
    LOG_DEBUG("Verifying NTFS signature...");
    if (memcmp(bootSector.OemId, "NTFS    ", 8) != 0) {
        string msg = "不是有效的 NTFS 卷。";
        cout << msg << endl;
        LOG_ERROR(msg);
        CloseVolume();
        return false;
    }
    LOG_DEBUG("NTFS signature verified");

    // 获取文件系统参数
    bytesPerSector = bootSector.BytesPerSector;
    sectorsPerCluster = bootSector.SectorsPerCluster;
    mftStartLCN = bootSector.MftStartLCN;
    totalClusters = bootSector.TotalSectors / sectorsPerCluster;

    LOG_DEBUG_FMT("BytesPerSector: %d", bytesPerSector);
    LOG_DEBUG_FMT("SectorsPerCluster: %d", (int)sectorsPerCluster);
    LOG_DEBUG_FMT("MFT Start LCN: %llu", mftStartLCN);
    LOG_DEBUG_FMT("TotalClusters: %llu", totalClusters);
    LOG_DEBUG_FMT("ClustersPerFileRecord (raw): %d", (int)bootSector.ClustersPerFileRecord);

    // 验证基本参数
    if (bytesPerSector == 0 || sectorsPerCluster == 0) {
        string msg = "无效的引导扇区参数：bytesPerSector 或 sectorsPerCluster 为 0";
        cout << msg << endl;
        LOG_FATAL(msg);
        CloseVolume();
        return false;
    }

    // 计算每个文件记录的大小
    if (bootSector.ClustersPerFileRecord > 0) {
        bytesPerFileRecord = bootSector.ClustersPerFileRecord * sectorsPerCluster * bytesPerSector;
        LOG_DEBUG_FMT("Calculated bytesPerFileRecord (positive): %d", bytesPerFileRecord);
    } else {
        // 负数表示以2的幂次方表示大小
        int shift = -bootSector.ClustersPerFileRecord;
        LOG_DEBUG_FMT("Using negative value, shift amount: %d", shift);

        // 安全检查：防止位移量过大
        if (shift < 0 || shift > 31) {
            string msg = "无效的 ClustersPerFileRecord 位移值: " + to_string(shift);
            cout << msg << endl;
            LOG_FATAL(msg);
            CloseVolume();
            return false;
        }

        bytesPerFileRecord = 1 << shift;
        LOG_DEBUG_FMT("Calculated bytesPerFileRecord (negative): %d", bytesPerFileRecord);
    }

    // 最终验证
    if (bytesPerFileRecord == 0) {
        string msg = "严重错误：bytesPerFileRecord 计算结果为 0！";
        cout << msg << endl;
        LOG_FATAL(msg);
        CloseVolume();
        return false;
    }

    LOG_DEBUG_FMT("Bytes per file record: %d", bytesPerFileRecord);

    cout << "卷打开成功。" << endl;
    cout << "每扇区字节数: " << bytesPerSector << endl;
    cout << "每簇扇区数: " << (int)sectorsPerCluster << endl;
    cout << "MFT 起始 LCN: " << mftStartLCN << endl;
    cout << "每文件记录字节数: " << bytesPerFileRecord << endl;

    LOG_INFO_FMT("Volume %c: opened successfully", driveLetter);
    return true;
}

void MFTReader::CloseVolume() {
    hVolume.Close();
}

bool MFTReader::ReadClusters(ULONGLONG startLCN, ULONGLONG clusterCount, vector<BYTE>& buffer) {
    if (!hVolume.IsValid()) {
        return false;
    }

    // 安全检查：防止除以零或使用无效参数
    if (sectorsPerCluster == 0 || bytesPerSector == 0) {
        LOG_ERROR_FMT("Invalid volume parameters: sectorsPerCluster=%d, bytesPerSector=%d",
                     sectorsPerCluster, bytesPerSector);
        return false;
    }

    // 安全检查：防止请求过大的数据（ReadFile 最多读取 DWORD 字节）
    ULONGLONG bytesToReadFull = clusterCount * sectorsPerCluster * bytesPerSector;
    const ULONGLONG MAX_READ_SIZE = 0xFFFFFFFF;  // DWORD 最大值（约 4GB）

    if (bytesToReadFull > MAX_READ_SIZE) {
        LOG_ERROR_FMT("Cluster read size too large: %llu bytes (max: %llu). LCN=%llu, Count=%llu",
                     bytesToReadFull, MAX_READ_SIZE, startLCN, clusterCount);
        cout << "错误：单次操作无法读取超过 4GB 的数据。" << endl;
        return false;
    }

    ULONGLONG offset = startLCN * sectorsPerCluster * bytesPerSector;
    DWORD bytesToRead = (DWORD)bytesToReadFull;  // 现在是安全的

    // 安全检查：防止 vector 分配过大内存导致崩溃
    try {
        buffer.resize(bytesToRead);
    } catch (const std::bad_alloc&) {
        LOG_ERROR_FMT("Failed to allocate memory for cluster read: %u bytes", bytesToRead);
        cout << "错误：为簇读取分配缓冲区时内存不足。" << endl;
        return false;
    }

    LARGE_INTEGER liOffset;
    liOffset.QuadPart = offset;
    if (!SetFilePointerEx(hVolume, liOffset, NULL, FILE_BEGIN)) {
        DWORD error = GetLastError();
        LOG_ERROR_FMT("Failed to seek to cluster position. LCN=%llu, Offset=%llu, Error=%d",
                     startLCN, offset, error);
        cout << "定位到簇位置失败。错误代码: " << error << endl;
        return false;
    }

    DWORD bytesRead;
    if (!ReadFile(hVolume, buffer.data(), bytesToRead, &bytesRead, NULL)) {
        DWORD error = GetLastError();
        LOG_ERROR_FMT("Failed to read clusters. LCN=%llu, BytesToRead=%u, Error=%d",
                     startLCN, bytesToRead, error);
        cout << "读取簇失败。错误代码: " << error << endl;
        return false;
    }

    if (bytesRead != bytesToRead) {
        LOG_WARNING_FMT("Partial read: requested=%u, got=%u", bytesToRead, bytesRead);
    }

    return bytesRead == bytesToRead;
}

bool MFTReader::ReadMFT(ULONGLONG fileRecordNumber, vector<BYTE>& record) {
    LOG_DEBUG_FMT("ReadMFT called for record #%llu", fileRecordNumber);

    // 安全检查：确保参数已初始化
    if (bytesPerFileRecord == 0) {
        LOG_FATAL("bytesPerFileRecord is 0! Volume may not be properly opened.");
        return false;
    }

    if (bytesPerSector == 0 || sectorsPerCluster == 0) {
        LOG_FATAL_FMT("Invalid volume parameters: bytesPerSector=%d, sectorsPerCluster=%d",
                      bytesPerSector, sectorsPerCluster);
        return false;
    }

    record.resize(bytesPerFileRecord);

    // 使用GetMFTRecordLCN来获取正确的LCN（支持碎片化MFT）
    ULONGLONG lcn;
    ULONGLONG offsetInCluster;

    if (!GetMFTRecordLCN(fileRecordNumber, lcn, offsetInCluster)) {
        LOG_ERROR_FMT("Failed to locate MFT record #%llu", fileRecordNumber);
        return false;
    }

    LOG_DEBUG_FMT("Record #%llu -> LCN=%llu, offset=%llu", fileRecordNumber, lcn, offsetInCluster);

    vector<BYTE> clusterData;
    if (!ReadClusters(lcn, 1, clusterData)) {
        LOG_ERROR_FMT("Failed to read cluster for MFT record #%llu", fileRecordNumber);
        return false;
    }
    if (offsetInCluster + bytesPerFileRecord > clusterData.size()) {
        LOG_ERROR_FMT("Offset out of bounds: offset=%llu, size=%zu", offsetInCluster, clusterData.size());
        return false;
    }
    memcpy(record.data(), clusterData.data() + offsetInCluster, bytesPerFileRecord);

    // 验证FILE签名(放宽验证 - 删除的记录可能签名不完整)
    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record.data();
    if (header->Signature != 'ELIF') { // 小端序的 'FILE'
        // 仅记录日志，不返回失败 - 允许读取签名不匹配的记录
        LOG_DEBUG_FMT("Warning: MFT record #%llu has invalid signature 0x%X (expected 0x454C4946)",
                     fileRecordNumber, header->Signature);
        // 不返回false，继续处理
    }

    return true;
}

bool MFTReader::ReadMFTBatch(ULONGLONG startRecordNumber, ULONGLONG recordCount, vector<vector<BYTE>>& records) {
    LOG_DEBUG_FMT("ReadMFTBatch called: start=%llu, count=%llu", startRecordNumber, recordCount);

    // 安全检查
    if (bytesPerFileRecord == 0 || bytesPerSector == 0 || sectorsPerCluster == 0) {
        LOG_FATAL("Invalid volume parameters in ReadMFTBatch");
        return false;
    }

    records.clear();
    records.reserve(recordCount);

    ULONGLONG bytesPerCluster = sectorsPerCluster * bytesPerSector;
    ULONGLONG recordsPerCluster = bytesPerCluster / bytesPerFileRecord;

    if (recordsPerCluster == 0) {
        LOG_FATAL("recordsPerCluster is 0 in ReadMFTBatch");
        return false;
    }

    // 计算需要读取的簇范围
    ULONGLONG startCluster = startRecordNumber / recordsPerCluster;
    ULONGLONG endRecord = startRecordNumber + recordCount - 1;
    ULONGLONG endCluster = endRecord / recordsPerCluster;
    ULONGLONG clustersToRead = endCluster - startCluster + 1;

    LOG_DEBUG_FMT("Reading %llu clusters (from %llu to %llu)", clustersToRead, startCluster, endCluster);

    // 批量读取簇数据
    vector<BYTE> clusterData;
    if (!ReadClusters(mftStartLCN + startCluster, clustersToRead, clusterData)) {
        LOG_ERROR_FMT("Failed to read clusters in batch mode");
        return false;
    }

    // 从批量数据中提取单个记录
    for (ULONGLONG i = 0; i < recordCount; i++) {
        ULONGLONG recordNumber = startRecordNumber + i;
        ULONGLONG clusterOffset = (recordNumber / recordsPerCluster) - startCluster;
        ULONGLONG recordOffsetInCluster = (recordNumber % recordsPerCluster) * bytesPerFileRecord;
        ULONGLONG totalOffset = clusterOffset * bytesPerCluster + recordOffsetInCluster;

        // 检查偏移是否有效
        if (totalOffset + bytesPerFileRecord > clusterData.size()) {
            LOG_WARNING_FMT("Record #%llu offset out of bounds, skipping", recordNumber);
            continue;
        }

        // 提取单个记录
        vector<BYTE> record(bytesPerFileRecord);
        memcpy(record.data(), clusterData.data() + totalOffset, bytesPerFileRecord);
        /**
        // 验证签名（可选，提高性能可以跳过）
        PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record.data();
        if (header->Signature == 'ELIF') { // 'FILE' in little-endian
            records.push_back(move(record));
        } else {
            // 无效记录，添加空记录作为占位符
            records.push_back(vector<BYTE>());
        }
        **/
    }

    LOG_DEBUG_FMT("ReadMFTBatch completed: extracted %zu valid records", records.size());
    return true;
}

ULONGLONG MFTReader::GetTotalMFTRecords() {
    LOG_DEBUG("GetTotalMFTRecords called");

    if (!hVolume.IsValid()) {
        LOG_ERROR("Volume not open");
        return 0;
    }

    // 读取$MFT文件本身的记录（记录号0）
    LOG_DEBUG("Reading MFT record #0...");
    vector<BYTE> mftRecord;
    if (!ReadMFT(0, mftRecord)) {
        LOG_ERROR("Failed to read MFT record #0");
        return 0;
    }
    LOG_DEBUG("MFT record #0 read successfully");

    // 从$MFT的DATA属性获取大小
    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();
    BYTE* attrPtr = mftRecord.data() + header->FirstAttributeOffset;

    LOG_DEBUG("Parsing MFT attributes...");
    while (attrPtr < mftRecord.data() + header->UsedSize) {
        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            LOG_DEBUG("Reached end of attribute list");
            break;
        }

        if (attr->Type == AttributeData) {
            LOG_DEBUG("Found DATA attribute");
            if (attr->NonResident) {
                PNONRESIDENT_ATTRIBUTE nonResAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                ULONGLONG mftSize = nonResAttr->RealSize;

                // 解析MFT的data runs以支持碎片化的MFT
                BYTE* dataRunPtr = attrPtr + nonResAttr->DataRunOffset;
                if (!ParseMFTDataRuns(dataRunPtr)) {
                    LOG_WARNING("Failed to parse MFT data runs, will assume contiguous MFT");
                }

                // 安全检查：防止除以零
                if (bytesPerFileRecord == 0) {
                    LOG_FATAL("bytesPerFileRecord is 0 in GetTotalMFTRecords!");
                    return 10000; // 返回默认值
                }

                ULONGLONG totalRecords = mftSize / bytesPerFileRecord;
                LOG_INFO_FMT("Total MFT records calculated: %llu (MFT size: %llu bytes)", totalRecords, mftSize);
                return totalRecords;
            }
        }

        // 安全检查
        if (attr->Length == 0) {
            LOG_WARNING("Attribute length is 0, breaking");
            break;
        }

        attrPtr += attr->Length;
    }

    // 默认估算值
    LOG_WARNING("Could not determine MFT size, using default estimate: 10000");
    return 10000;
}

bool MFTReader::GetMFTRecordLCN(ULONGLONG recordNumber, ULONGLONG& lcn, ULONGLONG& offsetInCluster) {
    // 如果MFT data runs还没加载，使用简单的线性计算（假设MFT连续）
    if (!mftDataRunsLoaded || mftDataRuns.empty()) {
        ULONGLONG bytesPerCluster = sectorsPerCluster * bytesPerSector;
        ULONGLONG recordsPerCluster = bytesPerCluster / bytesPerFileRecord;
        
        ULONGLONG clusterNumber = recordNumber / recordsPerCluster;
        offsetInCluster = (recordNumber % recordsPerCluster) * bytesPerFileRecord;
        
        lcn = mftStartLCN + clusterNumber;
        return true;
    }
    
    // 使用data runs查找正确的LCN
    ULONGLONG bytesPerCluster = sectorsPerCluster * bytesPerSector;
    ULONGLONG recordsPerCluster = bytesPerCluster / bytesPerFileRecord;
    
    // 计算这个记录在MFT中的cluster偏移
    ULONGLONG targetClusterOffset = recordNumber / recordsPerCluster;
    offsetInCluster = (recordNumber % recordsPerCluster) * bytesPerFileRecord;
    
    // 遍历data runs找到包含目标cluster的run
    ULONGLONG currentClusterOffset = 0;
    for (const auto& run : mftDataRuns) {
        ULONGLONG runLCN = run.first;
        ULONGLONG runLength = run.second;  // clusters
        
        if (targetClusterOffset < currentClusterOffset + runLength) {
            // 找到了！
            ULONGLONG offsetInRun = targetClusterOffset - currentClusterOffset;
            lcn = runLCN + offsetInRun;
            return true;
        }
        
        currentClusterOffset += runLength;
    }
    
    // 未找到
    LOG_ERROR_FMT("Record #%llu not found in MFT data runs", recordNumber);
    return false;
}

bool MFTReader::ParseMFTDataRuns(BYTE* dataRun) {
    mftDataRuns.clear();
    
    BYTE* current = dataRun;
    LONGLONG currentLCN = 0;
    
    LOG_DEBUG("Parsing MFT data runs...");
    
    while (*current != 0) {
        BYTE header = *current;
        current++;
        
        BYTE lengthBytes = header & 0x0F;
        BYTE offsetBytes = (header >> 4) & 0x0F;
        
        if (lengthBytes == 0 || lengthBytes > 8) {
            LOG_WARNING("Invalid length bytes in data run");
            break;
        }
        
        // 读取运行长度
        ULONGLONG length = 0;
        memcpy(&length, current, lengthBytes);
        current += lengthBytes;
        
        // 读取运行偏移（有符号）
        LONGLONG offset = 0;
        if (offsetBytes > 0) {
            memcpy(&offset, current, offsetBytes);
            
            // 符号扩展
            BYTE signByte = current[offsetBytes - 1];
            if (signByte & 0x80) {
                for (int i = offsetBytes; i < 8; i++) {
                    ((BYTE*)&offset)[i] = 0xFF;
                }
            }
            current += offsetBytes;
        }
        
        // 计算绝对LCN
        currentLCN += offset;
        
        if (offset != 0) {
            mftDataRuns.push_back(make_pair(currentLCN, length));
            LOG_DEBUG_FMT("  Data run: LCN=%lld, Length=%llu clusters", currentLCN, length);
        }
    }
    
    mftDataRunsLoaded = !mftDataRuns.empty();
    LOG_INFO_FMT("Loaded %zu MFT data runs", mftDataRuns.size());
    
    return mftDataRunsLoaded;
}

void MFTReader::DiagnoseMFTFragmentation() {
    cout << "\n========================================" << endl;
    cout << "    MFT 碎片化分析" << endl;
    cout << "========================================\n" << endl;

    if (!hVolume.IsValid()) {
        cout << "错误：卷未打开！" << endl;
        return;
    }

    // 读取MFT记录#0
    cout << "[1/4] 正在读取 MFT 记录 #0..." << endl;
    vector<BYTE> mftRecord;

    // 暂时禁用data runs以读取记录#0
    bool savedDataRunsLoaded = mftDataRunsLoaded;
    mftDataRunsLoaded = false;

    if (!ReadMFT(0, mftRecord)) {
        cout << "错误：无法读取 MFT 记录 #0" << endl;
        mftDataRunsLoaded = savedDataRunsLoaded;
        return;
    }

    mftDataRunsLoaded = savedDataRunsLoaded;
    cout << "成功：MFT 记录 #0 读取成功\n" << endl;

    // 解析DATA属性
    cout << "[2/4] 正在解析 DATA 属性..." << endl;
    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();
    BYTE* attrPtr = mftRecord.data() + header->FirstAttributeOffset;
    BYTE* recordEnd = mftRecord.data() + header->UsedSize;

    PNONRESIDENT_ATTRIBUTE dataAttr = nullptr;
    BYTE* dataRunPtr = nullptr;

    while (attrPtr < recordEnd) {
        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            break;
        }

        if (attr->Type == AttributeData && attr->NonResident) {
            dataAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
            dataRunPtr = attrPtr + dataAttr->DataRunOffset;
            break;
        }

        if (attr->Length == 0) break;
        attrPtr += attr->Length;
    }

    if (!dataAttr || !dataRunPtr) {
        cout << "错误：未找到非常驻 DATA 属性" << endl;
        return;
    }

    ULONGLONG mftSize = dataAttr->RealSize;
    cout << "MFT 大小: " << (mftSize / (1024 * 1024)) << " MB (" << mftSize << " 字节)" << endl;
    cout << "成功：找到 DATA 属性\n" << endl;

    // 解析data runs
    cout << "[3/4] 正在解析数据运行..." << endl;
    
    vector<pair<ULONGLONG, ULONGLONG>> runs;
    BYTE* current = dataRunPtr;
    LONGLONG currentLCN = 0;
    int runIndex = 0;

    while (*current != 0) {
        BYTE headerByte = *current;
        current++;

        BYTE lengthBytes = headerByte & 0x0F;
        BYTE offsetBytes = (headerByte >> 4) & 0x0F;

        if (lengthBytes == 0 || lengthBytes > 8) {
            break;
        }

        // 读取长度
        ULONGLONG length = 0;
        memcpy(&length, current, lengthBytes);
        current += lengthBytes;

        // 读取偏移
        LONGLONG offset = 0;
        if (offsetBytes > 0) {
            memcpy(&offset, current, offsetBytes);

            // 符号扩展
            BYTE signByte = current[offsetBytes - 1];
            if (signByte & 0x80) {
                for (int i = offsetBytes; i < 8; i++) {
                    ((BYTE*)&offset)[i] = 0xFF;
                }
            }
            current += offsetBytes;
        }

        currentLCN += offset;

        if (offset != 0) {
            runs.push_back(make_pair(currentLCN, length));
            runIndex++;
        }
    }

    cout << "成功：找到 " << runs.size() << " 个数据运行\n" << endl;

    // 分析结果
    cout << "[4/4] 分析结果:" << endl;
    cout << "----------------------------------------" << endl;

    if (runs.empty()) {
        cout << "状态：未知（未找到数据运行）" << endl;
    } else if (runs.size() == 1) {
        cout << "状态：MFT 是连续的（无碎片）" << endl;
        cout << "  运行 #1: LCN " << runs[0].first << ", 长度 " << runs[0].second << " 簇" << endl;
        ULONGLONG sizeInMB = (runs[0].second * sectorsPerCluster * bytesPerSector) / (1024 * 1024);
        cout << "  大小: " << sizeInMB << " MB" << endl;
    } else {
        cout << "状态：MFT 有碎片（" << runs.size() << " 个片段）" << endl;
        cout << "\n碎片详情:" << endl;

        ULONGLONG totalClusters = 0;
        for (size_t i = 0; i < runs.size() && i < 10; i++) {
            ULONGLONG lcn = runs[i].first;
            ULONGLONG length = runs[i].second;
            totalClusters += length;

            ULONGLONG sizeInMB = (length * sectorsPerCluster * bytesPerSector) / (1024 * 1024);
            cout << "  片段 #" << (i + 1) << ": LCN " << lcn
                 << ", 长度 " << length << " 簇 (" << sizeInMB << " MB)" << endl;
        }

        if (runs.size() > 10) {
            cout << "  ... 还有 " << (runs.size() - 10) << " 个片段" << endl;
        }

        ULONGLONG totalSizeInMB = (totalClusters * sectorsPerCluster * bytesPerSector) / (1024 * 1024);
        cout << "\n总计: " << totalClusters << " 簇 (" << totalSizeInMB << " MB)" << endl;

        // 碎片化程度评估
        double fragmentation = ((double)(runs.size() - 1) / runs.size()) * 100.0;
        cout << "\n碎片化程度: ";
        if (runs.size() <= 5) {
            cout << "低（" << runs.size() << " 个片段）" << endl;
        } else if (runs.size() <= 20) {
            cout << "中等（" << runs.size() << " 个片段）" << endl;
        } else {
            cout << "高（" << runs.size() << " 个片段）" << endl;
        }
    }

    cout << "========================================\n" << endl;
}

// ============================================================================
// 新的错误处理 API 实现
// ============================================================================

// 打开卷（新版本）- 返回详细错误信息
// 验证驱动器字母
Result<void> MFTReader::OpenVolumeNew(char driveLetter) {
    LOG_DEBUG_FMT("MFTReader::OpenVolumeNew called for drive %c:", driveLetter);

    // 验证驱动器字母
    if (!isalpha(driveLetter)) {
        return Result<void>::Failure(
            ErrorCode::SystemInvalidDriveLetter,
            "无效的驱动器盘符"
        );
    }

    // 构造卷路径
    char volumePath[MAX_PATH];
    sprintf_s(volumePath, "\\\\.\\%c:", driveLetter);
    LOG_DEBUG_FMT("Volume path: %s", volumePath);

    // 打开卷句柄
    LOG_DEBUG("Attempting to open volume handle...");
    hVolume.Reset(CreateFileA(
        volumePath,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    ));

    if (!hVolume.IsValid()) {
        DWORD lastError = GetLastError();

        // 根据系统错误码返回相应的错误类型
        if (lastError == ERROR_ACCESS_DENIED) {
            return Result<void>::Failure(
                MakeSystemError(
                    ErrorCode::SystemDiskAccessDenied,
                    "打开卷失败 - 访问被拒绝。需要管理员权限。",
                    std::string("驱动器: ") + driveLetter
                )
            );
        } else if (lastError == ERROR_FILE_NOT_FOUND || lastError == ERROR_PATH_NOT_FOUND) {
            return Result<void>::Failure(
                MakeSystemError(
                    ErrorCode::SystemVolumeNotFound,
                    "未找到卷",
                    std::string("驱动器: ") + driveLetter
                )
            );
        } else {
            return Result<void>::Failure(
                MakeSystemError(
                    ErrorCode::IOHandleInvalid,
                    "打开卷失败",
                    std::string("驱动器: ") + driveLetter
                )
            );
        }
    }
    LOG_DEBUG("Volume handle opened successfully");

    // 读取引导扇区
    LOG_DEBUG("Reading boot sector...");
    NTFS_BOOT_SECTOR bootSector;
    DWORD bytesRead;

    if (!ReadFile(hVolume, &bootSector, sizeof(NTFS_BOOT_SECTOR), &bytesRead, NULL)) {
        CloseVolume();
        return Result<void>::Failure(
            MakeSystemError(
                ErrorCode::IOReadFailed,
                "读取引导扇区失败",
                volumePath
            )
        );
    }
    LOG_DEBUG_FMT("Boot sector read successfully, %d bytes", bytesRead);

    // 验证 NTFS 签名
    LOG_DEBUG("Verifying NTFS signature...");
    if (memcmp(bootSector.OemId, "NTFS    ", 8) != 0) {
        CloseVolume();
        return Result<void>::Failure(
            ErrorCode::FSMFTCorrupted,
            "不是有效的 NTFS 卷"
        );
    }
    LOG_DEBUG("NTFS signature verified");

    // 获取文件系统参数
    bytesPerSector = bootSector.BytesPerSector;
    sectorsPerCluster = bootSector.SectorsPerCluster;
    mftStartLCN = bootSector.MftStartLCN;
    totalClusters = bootSector.TotalSectors / sectorsPerCluster;

    // 计算文件记录大小
    CHAR clustersPerFileRecord = bootSector.ClustersPerFileRecord;
    if (clustersPerFileRecord > 0) {
        bytesPerFileRecord = clustersPerFileRecord * sectorsPerCluster * bytesPerSector;
    } else {
        bytesPerFileRecord = 1 << (-clustersPerFileRecord);
    }

    LOG_DEBUG_FMT("File system parameters:");
    LOG_DEBUG_FMT("  Bytes per sector: %d", bytesPerSector);
    LOG_DEBUG_FMT("  Sectors per cluster: %d", sectorsPerCluster);
    LOG_DEBUG_FMT("  MFT start LCN: %llu", mftStartLCN);
    LOG_DEBUG_FMT("  Bytes per file record: %d", bytesPerFileRecord);

    mftDataRunsLoaded = false;

    return Result<void>::Success();
}

// 读取簇（新版本）- 返回数据或错误
Result<vector<BYTE>> MFTReader::ReadClustersNew(ULONGLONG startLCN, ULONGLONG clusterCount) {
    if (!hVolume.IsValid()) {
        return Result<vector<BYTE>>::Failure(
            ErrorCode::IOHandleInvalid,
            "卷未打开"
        );
    }

    if (startLCN >= totalClusters) {
        return Result<vector<BYTE>>::Failure(
            ErrorCode::FSInvalidClusterNumber,
            "起始 LCN 超出卷大小"
        );
    }

    ULONGLONG clusterSize = (ULONGLONG)bytesPerSector * sectorsPerCluster;
    ULONGLONG totalBytes = clusterCount * clusterSize;

    vector<BYTE> buffer(totalBytes);

    LARGE_INTEGER offset;
    offset.QuadPart = startLCN * clusterSize;

    if (SetFilePointerEx(hVolume, offset, NULL, FILE_BEGIN) == 0) {
        return Result<vector<BYTE>>::Failure(
            MakeSystemError(
                ErrorCode::IOSeekFailed,
                "定位到簇失败",
                "LCN: " + std::to_string(startLCN)
            )
        );
    }

    DWORD bytesRead;
    if (!ReadFile(hVolume, buffer.data(), (DWORD)totalBytes, &bytesRead, NULL)) {
        return Result<vector<BYTE>>::Failure(
            MakeSystemError(
                ErrorCode::IOReadFailed,
                "读取簇失败",
                "LCN: " + std::to_string(startLCN) + ", 数量: " + std::to_string(clusterCount)
            )
        );
    }

    if (bytesRead != totalBytes) {
        return Result<vector<BYTE>>::Failure(
            ErrorCode::IOReadFailed,
            "读取不完整 - 预期 " + std::to_string(totalBytes) +
            " 字节，实际 " + std::to_string(bytesRead) + " 字节"
        );
    }

    return Result<vector<BYTE>>::Success(std::move(buffer));
}

// 读取 MFT 记录（新版本）
Result<vector<BYTE>> MFTReader::ReadMFTNew(ULONGLONG fileRecordNumber) {
    if (!hVolume.IsValid()) {
        return Result<vector<BYTE>>::Failure(
            ErrorCode::IOHandleInvalid,
            "卷未打开"
        );
    }

    // 确保 MFT data runs 已加载
    if (!mftDataRunsLoaded) {
        vector<BYTE> mftRecord0;
        if (!ReadMFT(0, mftRecord0)) {
            return Result<vector<BYTE>>::Failure(
                ErrorCode::FSMFTCorrupted,
                "加载 MFT 数据运行失败"
            );
        }
    }

    // 查找 MFT 记录的实际 LCN
    ULONGLONG lcn, offsetInCluster;
    if (!GetMFTRecordLCN(fileRecordNumber, lcn, offsetInCluster)) {
        return Result<vector<BYTE>>::Failure(
            FR::ErrorInfo(
                ErrorCode::FSRecordNotFound,
                "未找到 MFT 记录",
                "记录号: " + std::to_string(fileRecordNumber),
                0
            )
        );
    }

    // 读取包含该记录的簇
    auto clusterResult = ReadClustersNew(lcn, 1);
    if (clusterResult.IsFailure()) {
        return Result<vector<BYTE>>::Failure(clusterResult.Error());
    }

    // 提取 MFT 记录
    vector<BYTE>& clusterData = clusterResult.Value();
    if (offsetInCluster + bytesPerFileRecord > clusterData.size()) {
        return Result<vector<BYTE>>::Failure(
            ErrorCode::LogicBufferTooSmall,
            "MFT 记录超出簇边界"
        );
    }

    vector<BYTE> record(bytesPerFileRecord);
    memcpy(record.data(), clusterData.data() + offsetInCluster, bytesPerFileRecord);

    // 验证 MFT 记录签名
    if (record.size() >= 4) {
        if (memcmp(record.data(), "FILE", 4) != 0) {
            return Result<vector<BYTE>>::Failure(
                FR::ErrorInfo(
                    ErrorCode::FSMFTCorrupted,
                    "无效的 MFT 记录签名",
                    "记录号: " + std::to_string(fileRecordNumber),
                    0
                )
            );
        }
    }

    return Result<vector<BYTE>>::Success(std::move(record));
}
