#include "MFTReader.h"
#include "Logger.h"
#include <iostream>
#include <winioctl.h>

using namespace std;

MFTReader::MFTReader() : hVolume(INVALID_HANDLE_VALUE),
    bytesPerSector(0), sectorsPerCluster(0),
    mftStartLCN(0), bytesPerFileRecord(0), totalClusters(0),
    mftDataRunsLoaded(false) {
}

MFTReader::~MFTReader() {
    CloseVolume();
}

bool MFTReader::OpenVolume(char driveLetter) {
    LOG_DEBUG_FMT("MFTReader::OpenVolume called for drive %c:", driveLetter);

    char volumePath[MAX_PATH];
    sprintf_s(volumePath, "\\\\.\\%c:", driveLetter);
    LOG_DEBUG_FMT("Volume path: %s", volumePath);

    LOG_DEBUG("Attempting to open volume handle...");
    hVolume = CreateFileA(volumePath, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
        OPEN_EXISTING, 0, NULL);

    if (hVolume == INVALID_HANDLE_VALUE) {
        DWORD error = GetLastError();
        string msg = "Failed to open volume. Error: " + to_string(error);
        cout << msg << endl;
        cout << "You need administrator privileges to access raw volume." << endl;
        LOG_ERROR(msg);
        return false;
    }
    LOG_DEBUG("Volume handle opened successfully");

    // 读取引导扇区
    LOG_DEBUG("Reading boot sector...");
    NTFS_BOOT_SECTOR bootSector;
    DWORD bytesRead;
    if (!ReadFile(hVolume, &bootSector, sizeof(NTFS_BOOT_SECTOR), &bytesRead, NULL)) {
        string msg = "Failed to read boot sector.";
        cout << msg << endl;
        LOG_ERROR_FMT("%s Error: %d", msg.c_str(), GetLastError());
        CloseVolume();
        return false;
    }
    LOG_DEBUG_FMT("Boot sector read successfully, %d bytes", bytesRead);

    // 验证NTFS签名
    LOG_DEBUG("Verifying NTFS signature...");
    if (memcmp(bootSector.OemId, "NTFS    ", 8) != 0) {
        string msg = "Not a valid NTFS volume.";
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
        string msg = "Invalid boot sector parameters: bytesPerSector or sectorsPerCluster is 0";
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
            string msg = "Invalid ClustersPerFileRecord shift value: " + to_string(shift);
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
        string msg = "CRITICAL: bytesPerFileRecord calculated as 0!";
        cout << msg << endl;
        LOG_FATAL(msg);
        CloseVolume();
        return false;
    }

    LOG_DEBUG_FMT("Bytes per file record: %d", bytesPerFileRecord);

    cout << "Volume opened successfully." << endl;
    cout << "Bytes per sector: " << bytesPerSector << endl;
    cout << "Sectors per cluster: " << (int)sectorsPerCluster << endl;
    cout << "MFT start LCN: " << mftStartLCN << endl;
    cout << "Bytes per file record: " << bytesPerFileRecord << endl;

    LOG_INFO_FMT("Volume %c: opened successfully", driveLetter);
    return true;
}

void MFTReader::CloseVolume() {
    if (hVolume != INVALID_HANDLE_VALUE) {
        CloseHandle(hVolume);
        hVolume = INVALID_HANDLE_VALUE;
    }
}

bool MFTReader::ReadClusters(ULONGLONG startLCN, ULONGLONG clusterCount, vector<BYTE>& buffer) {
    if (hVolume == INVALID_HANDLE_VALUE) {
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
        cout << "Error: Cannot read more than 4GB in a single operation." << endl;
        return false;
    }

    ULONGLONG offset = startLCN * sectorsPerCluster * bytesPerSector;
    DWORD bytesToRead = (DWORD)bytesToReadFull;  // 现在是安全的

    // 安全检查：防止 vector 分配过大内存导致崩溃
    try {
        buffer.resize(bytesToRead);
    } catch (const std::bad_alloc&) {
        LOG_ERROR_FMT("Failed to allocate memory for cluster read: %u bytes", bytesToRead);
        cout << "Error: Out of memory while allocating buffer for cluster read." << endl;
        return false;
    }

    LARGE_INTEGER liOffset;
    liOffset.QuadPart = offset;
    if (!SetFilePointerEx(hVolume, liOffset, NULL, FILE_BEGIN)) {
        DWORD error = GetLastError();
        LOG_ERROR_FMT("Failed to seek to cluster position. LCN=%llu, Offset=%llu, Error=%d",
                     startLCN, offset, error);
        cout << "Failed to seek to cluster position. Error: " << error << endl;
        return false;
    }

    DWORD bytesRead;
    if (!ReadFile(hVolume, buffer.data(), bytesToRead, &bytesRead, NULL)) {
        DWORD error = GetLastError();
        LOG_ERROR_FMT("Failed to read clusters. LCN=%llu, BytesToRead=%u, Error=%d",
                     startLCN, bytesToRead, error);
        cout << "Failed to read clusters. Error: " << error << endl;
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
    if (header->Signature != 'ELIF') { // 'FILE' in little-endian
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

    if (hVolume == INVALID_HANDLE_VALUE) {
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
    cout << "    MFT Fragmentation Analysis" << endl;
    cout << "========================================\n" << endl;

    if (hVolume == INVALID_HANDLE_VALUE) {
        cout << "ERROR: Volume not open!" << endl;
        return;
    }

    // 读取MFT记录#0
    cout << "[1/4] Reading MFT record #0..." << endl;
    vector<BYTE> mftRecord;
    
    // 暂时禁用data runs以读取记录#0
    bool savedDataRunsLoaded = mftDataRunsLoaded;
    mftDataRunsLoaded = false;
    
    if (!ReadMFT(0, mftRecord)) {
        cout << "ERROR: Failed to read MFT record #0" << endl;
        mftDataRunsLoaded = savedDataRunsLoaded;
        return;
    }
    
    mftDataRunsLoaded = savedDataRunsLoaded;
    cout << "SUCCESS: MFT record #0 read successfully\n" << endl;

    // 解析DATA属性
    cout << "[2/4] Parsing DATA attribute..." << endl;
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
        cout << "ERROR: No non-resident DATA attribute found" << endl;
        return;
    }

    ULONGLONG mftSize = dataAttr->RealSize;
    cout << "MFT Size: " << (mftSize / (1024 * 1024)) << " MB (" << mftSize << " bytes)" << endl;
    cout << "SUCCESS: DATA attribute found\n" << endl;

    // 解析data runs
    cout << "[3/4] Parsing data runs..." << endl;
    
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

    cout << "SUCCESS: Found " << runs.size() << " data run(s)\n" << endl;

    // 分析结果
    cout << "[4/4] Analysis Results:" << endl;
    cout << "----------------------------------------" << endl;

    if (runs.empty()) {
        cout << "STATUS: Unknown (no data runs found)" << endl;
    } else if (runs.size() == 1) {
        cout << "STATUS: MFT is CONTIGUOUS (not fragmented)" << endl;
        cout << "  Run #1: LCN " << runs[0].first << ", Length " << runs[0].second << " clusters" << endl;
        ULONGLONG sizeInMB = (runs[0].second * sectorsPerCluster * bytesPerSector) / (1024 * 1024);
        cout << "  Size: " << sizeInMB << " MB" << endl;
    } else {
        cout << "STATUS: MFT is FRAGMENTED (" << runs.size() << " fragments)" << endl;
        cout << "\nFragment details:" << endl;

        ULONGLONG totalClusters = 0;
        for (size_t i = 0; i < runs.size() && i < 10; i++) {
            ULONGLONG lcn = runs[i].first;
            ULONGLONG length = runs[i].second;
            totalClusters += length;

            ULONGLONG sizeInMB = (length * sectorsPerCluster * bytesPerSector) / (1024 * 1024);
            cout << "  Fragment #" << (i + 1) << ": LCN " << lcn 
                 << ", Length " << length << " clusters (" << sizeInMB << " MB)" << endl;
        }

        if (runs.size() > 10) {
            cout << "  ... and " << (runs.size() - 10) << " more fragments" << endl;
        }

        ULONGLONG totalSizeInMB = (totalClusters * sectorsPerCluster * bytesPerSector) / (1024 * 1024);
        cout << "\nTotal: " << totalClusters << " clusters (" << totalSizeInMB << " MB)" << endl;
        
        // 碎片化程度评估
        double fragmentation = ((double)(runs.size() - 1) / runs.size()) * 100.0;
        cout << "\nFragmentation severity: ";
        if (runs.size() <= 5) {
            cout << "LOW (" << runs.size() << " fragments)" << endl;
        } else if (runs.size() <= 20) {
            cout << "MODERATE (" << runs.size() << " fragments)" << endl;
        } else {
            cout << "HIGH (" << runs.size() << " fragments)" << endl;
        }
    }

    cout << "========================================\n" << endl;
}
