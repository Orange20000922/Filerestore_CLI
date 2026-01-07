#include "MFTLCNIndex.h"
#include "Logger.h"
#include <iostream>
#include <iomanip>

using namespace std;

// ============================================================================
// 构造函数和析构函数
// ============================================================================
MFTLCNIndex::MFTLCNIndex(MFTReader* mftReader)
    : reader(mftReader), indexBuilt(false) {
}

MFTLCNIndex::~MFTLCNIndex() {
    ClearIndex();
}

void MFTLCNIndex::ClearIndex() {
    lcnIndex.clear();
    indexBuilt = false;
}

// ============================================================================
// Data Runs 解析
// ============================================================================
bool MFTLCNIndex::ParseDataRuns(const BYTE* dataRun, size_t maxLen,
                                 vector<pair<ULONGLONG, ULONGLONG>>& runs) {
    runs.clear();
    if (!dataRun || maxLen == 0) return false;

    size_t offset = 0;
    LONGLONG currentLCN = 0;  // 相对偏移，需要累加

    while (offset < maxLen && dataRun[offset] != 0) {
        BYTE header = dataRun[offset];
        if (header == 0) break;

        int lengthSize = header & 0x0F;
        int offsetSize = (header >> 4) & 0x0F;

        if (lengthSize == 0 || lengthSize > 8 || offsetSize > 8) break;
        if (offset + 1 + lengthSize + offsetSize > maxLen) break;

        // 读取长度
        ULONGLONG runLength = 0;
        for (int i = 0; i < lengthSize; i++) {
            runLength |= ((ULONGLONG)dataRun[offset + 1 + i] << (i * 8));
        }

        // 读取偏移（有符号）
        LONGLONG runOffset = 0;
        if (offsetSize > 0) {
            for (int i = 0; i < offsetSize; i++) {
                runOffset |= ((LONGLONG)dataRun[offset + 1 + lengthSize + i] << (i * 8));
            }
            // 符号扩展
            if (dataRun[offset + lengthSize + offsetSize] & 0x80) {
                for (int i = offsetSize; i < 8; i++) {
                    runOffset |= (0xFFLL << (i * 8));
                }
            }
            currentLCN += runOffset;
        }

        if (runLength > 0 && currentLCN >= 0) {
            runs.push_back(make_pair((ULONGLONG)currentLCN, runLength));
        }

        offset += 1 + lengthSize + offsetSize;
    }

    return !runs.empty();
}

// ============================================================================
// 从 MFT 记录提取时间戳
// ============================================================================
bool MFTLCNIndex::ExtractTimestamps(const BYTE* record, size_t recordSize,
                                     FILETIME& creation, FILETIME& modification, FILETIME& access) {
    if (recordSize < sizeof(FILE_RECORD_HEADER)) return false;

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record;
    if (header->Signature != 'ELIF') return false;  // "FILE" 的小端序表示

    WORD attrOffset = header->FirstAttributeOffset;
    if (attrOffset >= recordSize) return false;

    // 遍历属性寻找 $STANDARD_INFORMATION 或 $FILE_NAME
    while (attrOffset + sizeof(ATTRIBUTE_HEADER) < recordSize) {
        PATTRIBUTE_HEADER attrHeader = (PATTRIBUTE_HEADER)(record + attrOffset);

        if (attrHeader->Type == AttributeEndOfList || attrHeader->Length == 0) break;
        if (attrOffset + attrHeader->Length > recordSize) break;

        // $FILE_NAME (0x30) 包含时间戳
        if (attrHeader->Type == AttributeFileName && attrHeader->NonResident == 0) {
            PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(record + attrOffset + sizeof(ATTRIBUTE_HEADER));
            if (resAttr->ValueLength >= sizeof(FILE_NAME_ATTRIBUTE) - sizeof(WCHAR)) {
                PFILE_NAME_ATTRIBUTE fnAttr = (PFILE_NAME_ATTRIBUTE)(record + attrOffset +
                    sizeof(ATTRIBUTE_HEADER) + resAttr->ValueOffset);

                // NTFS 时间戳是 100 纳秒间隔（从 1601-01-01）
                ULARGE_INTEGER li;

                li.QuadPart = fnAttr->CreationTime;
                creation.dwLowDateTime = li.LowPart;
                creation.dwHighDateTime = li.HighPart;

                li.QuadPart = fnAttr->ModificationTime;
                modification.dwLowDateTime = li.LowPart;
                modification.dwHighDateTime = li.HighPart;

                li.QuadPart = fnAttr->LastAccessTime;
                access.dwLowDateTime = li.LowPart;
                access.dwHighDateTime = li.HighPart;

                return true;
            }
        }

        attrOffset += attrHeader->Length;
    }

    return false;
}

// ============================================================================
// 从 MFT 记录提取文件名
// ============================================================================
wstring MFTLCNIndex::ExtractFileName(const BYTE* record, size_t recordSize) {
    if (recordSize < sizeof(FILE_RECORD_HEADER)) return L"";

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record;
    if (header->Signature != 'ELIF') return L"";

    WORD attrOffset = header->FirstAttributeOffset;

    while (attrOffset + sizeof(ATTRIBUTE_HEADER) < recordSize) {
        PATTRIBUTE_HEADER attrHeader = (PATTRIBUTE_HEADER)(record + attrOffset);

        if (attrHeader->Type == AttributeEndOfList || attrHeader->Length == 0) break;
        if (attrOffset + attrHeader->Length > recordSize) break;

        if (attrHeader->Type == AttributeFileName && attrHeader->NonResident == 0) {
            PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(record + attrOffset + sizeof(ATTRIBUTE_HEADER));

            PFILE_NAME_ATTRIBUTE fnAttr = (PFILE_NAME_ATTRIBUTE)(record + attrOffset +
                sizeof(ATTRIBUTE_HEADER) + resAttr->ValueOffset);

            // 优先使用长文件名 (NameType == 0 或 3)
            if (fnAttr->NameType == 0 || fnAttr->NameType == 3) {
                return wstring(fnAttr->FileName, fnAttr->FileNameLength);
            }
            // 如果只有短文件名，也使用
            else if (fnAttr->NameType == 2) {
                return wstring(fnAttr->FileName, fnAttr->FileNameLength);
            }
        }

        attrOffset += attrHeader->Length;
    }

    return L"";
}

// ============================================================================
// 构建 LCN 索引
// ============================================================================
bool MFTLCNIndex::BuildIndex(bool includeActiveFiles, bool showProgress) {
    if (!reader || !reader->IsVolumeOpen()) {
        LOG_ERROR("卷未打开，无法进行 MFT LCN 索引");
        return false;
    }

    ClearIndex();

    ULONGLONG totalRecords = reader->GetTotalMFTRecords();
    if (totalRecords == 0) {
        LOG_ERROR("无法获取 MFT 记录总数");
        return false;
    }

    if (showProgress) {
        cout << "\n--- 正在构建 MFT LCN 索引 ---" << endl;
        cout << "MFT 记录总数: " << totalRecords << endl;
        cout << "模式: " << (includeActiveFiles ? "所有文件" : "仅已删除文件") << endl;
        cout << "正在扫描..." << endl;
    }

    DWORD startTime = GetTickCount();
    ULONGLONG processedRecords = 0;
    ULONGLONG indexedFiles = 0;
    const ULONGLONG BATCH_SIZE = 1000;
    const ULONGLONG PROGRESS_INTERVAL = 100000;

    for (ULONGLONG startRecord = 0; startRecord < totalRecords; startRecord += BATCH_SIZE) {
        ULONGLONG recordCount = min(BATCH_SIZE, totalRecords - startRecord);
        vector<vector<BYTE>> records;

        if (!reader->ReadMFTBatch(startRecord, recordCount, records)) {
            continue;
        }

        for (ULONGLONG i = 0; i < records.size(); i++) {
            if (records[i].empty()) continue;

            const BYTE* record = records[i].data();
            size_t recordSize = records[i].size();

            if (recordSize < sizeof(FILE_RECORD_HEADER)) continue;

            PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record;
            if (header->Signature != 'ELIF') continue;

            bool isDeleted = !(header->Flags & 0x01);
            bool isDirectory = (header->Flags & 0x02) != 0;

            // 根据模式过滤
            if (!includeActiveFiles && !isDeleted) continue;

            // 跳过目录（目录通常没有有意义的数据）
            if (isDirectory) continue;

            // 遍历属性寻找 $DATA
            WORD attrOffset = header->FirstAttributeOffset;
            while (attrOffset + sizeof(ATTRIBUTE_HEADER) < recordSize) {
                PATTRIBUTE_HEADER attrHeader = (PATTRIBUTE_HEADER)(record + attrOffset);

                if (attrHeader->Type == AttributeEndOfList || attrHeader->Length == 0) break;
                if (attrOffset + attrHeader->Length > recordSize) break;

                // $DATA 属性 (0x80)
                if (attrHeader->Type == AttributeData && attrHeader->NonResident == 1) {
                    PNONRESIDENT_ATTRIBUTE nrAttr = (PNONRESIDENT_ATTRIBUTE)(record + attrOffset +
                        sizeof(ATTRIBUTE_HEADER));

                    // 解析 Data Runs
                    const BYTE* dataRuns = record + attrOffset + nrAttr->DataRunOffset;
                    size_t dataRunsMaxLen = attrHeader->Length - nrAttr->DataRunOffset;

                    vector<pair<ULONGLONG, ULONGLONG>> runs;
                    if (ParseDataRuns(dataRuns, dataRunsMaxLen, runs)) {
                        // 提取时间戳和文件名
                        FILETIME creation = {0}, modification = {0}, access = {0};
                        ExtractTimestamps(record, recordSize, creation, modification, access);
                        wstring fileName = ExtractFileName(record, recordSize);

                        // 为每个 Data Run 创建索引条目
                        for (const auto& run : runs) {
                            LCNMappingInfo info;
                            info.mftRecordNumber = startRecord + i;
                            info.startLCN = run.first;
                            info.clusterCount = run.second;
                            info.creationTime = creation;
                            info.modificationTime = modification;
                            info.accessTime = access;
                            info.fileName = fileName;
                            info.isDeleted = isDeleted;

                            lcnIndex[run.first] = info;
                            indexedFiles++;
                        }
                    }
                }

                attrOffset += attrHeader->Length;
            }
        }

        processedRecords += recordCount;

        // 进度更新
        if (showProgress && processedRecords % PROGRESS_INTERVAL < BATCH_SIZE) {
            double progress = (double)processedRecords / totalRecords * 100.0;
            cout << "\r进度: " << fixed << setprecision(1) << progress << "% | "
                 << "记录数: " << processedRecords << " | "
                 << "已索引: " << indexedFiles << " 个数据运行" << flush;
        }
    }

    indexBuilt = true;

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- 索引构建完成 ---" << endl;
        cout << "耗时: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " 秒" << endl;
        cout << "已处理记录数: " << processedRecords << endl;
        cout << "已索引数据运行数: " << indexedFiles << endl;
    }

    LOG_INFO_FMT("MFT LCN 索引构建完成: 已索引 %llu 个数据运行", indexedFiles);
    return true;
}

// ============================================================================
// 根据 LCN 查找匹配的 MFT 记录
// ============================================================================
vector<LCNMappingInfo> MFTLCNIndex::FindByLCN(ULONGLONG lcn) {
    vector<LCNMappingInfo> results;

    if (!indexBuilt || lcnIndex.empty()) {
        return results;
    }

    // 查找包含该 LCN 的条目
    // 使用 upper_bound 找到第一个 startLCN > lcn 的条目，然后往前检查
    auto it = lcnIndex.upper_bound(lcn);

    // 检查前一个条目
    if (it != lcnIndex.begin()) {
        --it;
        // 检查 LCN 是否在该数据运行的范围内
        if (it->second.startLCN <= lcn &&
            lcn < it->second.startLCN + it->second.clusterCount) {
            results.push_back(it->second);
        }
    }

    // 同时检查精确匹配
    auto exactIt = lcnIndex.find(lcn);
    if (exactIt != lcnIndex.end() && (results.empty() || results[0].mftRecordNumber != exactIt->second.mftRecordNumber)) {
        results.push_back(exactIt->second);
    }

    return results;
}

// ============================================================================
// 根据 LCN 范围查找
// ============================================================================
vector<LCNMappingInfo> MFTLCNIndex::FindByLCNRange(ULONGLONG startLCN, ULONGLONG endLCN) {
    vector<LCNMappingInfo> results;
    set<ULONGLONG> seenRecords;  // 用于避免重复记录

    if (!indexBuilt || lcnIndex.empty()) {
        return results;
    }

    // 找到第一个可能重叠的条目
    auto it = lcnIndex.upper_bound(startLCN);
    if (it != lcnIndex.begin()) {
        --it;
    }

    // 遍历所有可能重叠的条目
    while (it != lcnIndex.end() && it->second.startLCN < endLCN) {
        ULONGLONG runStart = it->second.startLCN;
        ULONGLONG runEnd = runStart + it->second.clusterCount;

        // 检查是否有重叠
        if (runEnd > startLCN && runStart < endLCN) {
            if (seenRecords.find(it->second.mftRecordNumber) == seenRecords.end()) {
                results.push_back(it->second);
                seenRecords.insert(it->second.mftRecordNumber);
            }
        }

        ++it;
    }

    return results;
}
