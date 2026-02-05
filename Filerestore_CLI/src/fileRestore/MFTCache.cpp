#include "MFTCache.h"
#include "MFTReader.h"
#include "Logger.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

using namespace std;

// 静态成员初始化
unique_ptr<MFTCache> MFTCacheManager::globalCache = nullptr;
char MFTCacheManager::cachedDrive = 0;

// ============================================================================
// MFTCache 构造和析构
// ============================================================================
MFTCache::MFTCache() : driveLetter(0), bytesPerCluster(4096), isValid(false) {
    ZeroMemory(&buildTime, sizeof(FILETIME));
}

MFTCache::~MFTCache() {
    Clear();
}

void MFTCache::Clear() {
    entriesByRecord.clear();
    lcnToRecord.clear();
    isValid = false;
}

string MFTCache::GenerateCachePath(char drive) {
    char tempPath[MAX_PATH];
    GetTempPathA(MAX_PATH, tempPath);
    return string(tempPath) + "mft_cache_" + drive + ".bin";
}

// ============================================================================
// 从 MFT 构建缓存
// ============================================================================
bool MFTCache::BuildFromMFT(char drive, bool includeActive, bool showProgress) {
    Clear();

    MFTReader reader;
    if (!reader.OpenVolume(drive)) {
        LOG_ERROR("无法打开卷进行 MFT 缓存构建");
        return false;
    }

    driveLetter = drive;
    bytesPerCluster = reader.GetBytesPerCluster();

    ULONGLONG totalRecords = reader.GetTotalMFTRecords();
    if (totalRecords == 0) {
        LOG_ERROR("无法获取 MFT 记录总数");
        return false;
    }

    if (showProgress) {
        cout << "\n=== Building MFT Cache ===" << endl;
        cout << "Drive: " << drive << ":" << endl;
        cout << "Total MFT records: " << totalRecords << endl;
        cout << "Mode: " << (includeActive ? "All files" : "Deleted only") << endl;
    }

    DWORD startTime = GetTickCount();
    const ULONGLONG BATCH_SIZE = 1000;
    ULONGLONG processedRecords = 0;
    ULONGLONG indexedFiles = 0;
    ULONGLONG deletedCount = 0;

    for (ULONGLONG startRecord = 0; startRecord < totalRecords; startRecord += BATCH_SIZE) {
        ULONGLONG recordCount = min(BATCH_SIZE, totalRecords - startRecord);
        vector<vector<BYTE>> records;

        if (!reader.ReadMFTBatch(startRecord, recordCount, records)) {
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
            if (!includeActive && !isDeleted) continue;

            // 跳过目录
            if (isDirectory) continue;

            MFTCacheEntry entry;
            entry.recordNumber = startRecord + i;
            entry.isDeleted = isDeleted;
            entry.isDirectory = isDirectory;
            entry.sequenceNumber = header->SequenceNumber;

            if (isDeleted) deletedCount++;

            // 遍历属性
            WORD attrOffset = header->FirstAttributeOffset;
            bool hasData = false;
            bool hasFileName = false;

            while (attrOffset + sizeof(ATTRIBUTE_HEADER) < recordSize) {
                PATTRIBUTE_HEADER attrHeader = (PATTRIBUTE_HEADER)(record + attrOffset);

                if (attrHeader->Type == AttributeEndOfList || attrHeader->Length == 0) break;
                if (attrOffset + attrHeader->Length > recordSize) break;

                // $FILE_NAME (0x30)
                if (attrHeader->Type == AttributeFileName && !hasFileName) {
                    if (attrHeader->NonResident == 0) {
                        PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(record + attrOffset + sizeof(ATTRIBUTE_HEADER));
                        if (resAttr->ValueLength >= sizeof(FILE_NAME_ATTRIBUTE) - sizeof(WCHAR)) {
                            PFILE_NAME_ATTRIBUTE fnAttr = (PFILE_NAME_ATTRIBUTE)(record + attrOffset +
                                sizeof(ATTRIBUTE_HEADER) + resAttr->ValueOffset);

                            // 优先长文件名
                            if (fnAttr->NameType == 0 || fnAttr->NameType == 3 || !hasFileName) {
                                entry.fileName = wstring(fnAttr->FileName, fnAttr->FileNameLength);
                                entry.parentRecord = fnAttr->ParentDirectory & 0x0000FFFFFFFFFFFF;

                                // 提取扩展名
                                size_t dotPos = entry.fileName.rfind(L'.');
                                if (dotPos != wstring::npos && dotPos < entry.fileName.length() - 1) {
                                    entry.extension = entry.fileName.substr(dotPos + 1);
                                    // 转小写
                                    transform(entry.extension.begin(), entry.extension.end(),
                                              entry.extension.begin(), ::towlower);
                                }

                                // 时间戳
                                ULARGE_INTEGER li;
                                li.QuadPart = fnAttr->CreationTime;
                                entry.creationTime.dwLowDateTime = li.LowPart;
                                entry.creationTime.dwHighDateTime = li.HighPart;

                                li.QuadPart = fnAttr->ModificationTime;
                                entry.modificationTime.dwLowDateTime = li.LowPart;
                                entry.modificationTime.dwHighDateTime = li.HighPart;

                                hasFileName = true;
                            }
                        }
                    }
                }
                // $DATA (0x80)
                else if (attrHeader->Type == AttributeData && !hasData) {
                    if (attrHeader->NonResident == 1) {
                        PNONRESIDENT_ATTRIBUTE nrAttr = (PNONRESIDENT_ATTRIBUTE)(record + attrOffset + sizeof(ATTRIBUTE_HEADER));
                        entry.fileSize = nrAttr->RealSize;
                        entry.isResident = false;

                        // 解析第一个数据运行获取起始 LCN
                        const BYTE* dataRuns = record + attrOffset + nrAttr->DataRunOffset;
                        size_t maxLen = attrHeader->Length - nrAttr->DataRunOffset;

                        if (maxLen > 0 && dataRuns[0] != 0) {
                            BYTE hdr = dataRuns[0];
                            int lengthSize = hdr & 0x0F;
                            int offsetSize = (hdr >> 4) & 0x0F;

                            if (lengthSize > 0 && offsetSize > 0 && 1 + lengthSize + offsetSize <= maxLen) {
                                // 读取长度
                                ULONGLONG runLength = 0;
                                for (int j = 0; j < lengthSize; j++) {
                                    runLength |= ((ULONGLONG)dataRuns[1 + j] << (j * 8));
                                }

                                // 读取偏移
                                LONGLONG runOffset = 0;
                                for (int j = 0; j < offsetSize; j++) {
                                    runOffset |= ((LONGLONG)dataRuns[1 + lengthSize + j] << (j * 8));
                                }
                                // 符号扩展
                                if (dataRuns[lengthSize + offsetSize] & 0x80) {
                                    for (int j = offsetSize; j < 8; j++) {
                                        runOffset |= (0xFFLL << (j * 8));
                                    }
                                }

                                if (runOffset > 0) {
                                    entry.startLCN = (ULONGLONG)runOffset;
                                    entry.totalClusters = runLength;
                                    hasData = true;
                                }
                            }
                        }
                    } else {
                        // 驻留数据
                        PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(record + attrOffset + sizeof(ATTRIBUTE_HEADER));
                        entry.fileSize = resAttr->ValueLength;
                        entry.isResident = true;
                        entry.startLCN = 0;
                        hasData = true;
                    }
                }

                attrOffset += attrHeader->Length;
            }

            // 只索引有有效数据的文件
            if (hasFileName && hasData && entry.fileSize > 0) {
                entriesByRecord[entry.recordNumber] = entry;

                // LCN 索引（非驻留文件）
                if (!entry.isResident && entry.startLCN > 0) {
                    lcnToRecord[entry.startLCN] = entry.recordNumber;
                }

                indexedFiles++;
            }
        }

        processedRecords += recordCount;

        // 进度更新
        if (showProgress && processedRecords % 100000 < BATCH_SIZE) {
            double progress = (double)processedRecords / totalRecords * 100.0;
            cout << "\r  Progress: " << fixed << setprecision(1) << progress << "% | "
                 << "Indexed: " << indexedFiles << " files" << flush;
        }
    }

    // 设置构建时间
    GetSystemTimeAsFileTime(&buildTime);
    isValid = true;
    cacheFilePath = GenerateCachePath(drive);

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                              " << endl;
        cout << "\n=== MFT Cache Built ===" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Total indexed: " << indexedFiles << " files" << endl;
        cout << "Deleted files: " << deletedCount << endl;
        cout << "Active files: " << (indexedFiles - deletedCount) << endl;
        cout << "LCN index entries: " << lcnToRecord.size() << endl;
    }

    LOG_INFO_FMT("MFT cache built: %llu files indexed", indexedFiles);
    return true;
}

// ============================================================================
// 序列化
// ============================================================================
void MFTCache::SerializeEntry(ofstream& out, const MFTCacheEntry& entry) {
    out.write((char*)&entry.recordNumber, sizeof(ULONGLONG));
    out.write((char*)&entry.fileSize, sizeof(ULONGLONG));
    out.write((char*)&entry.startLCN, sizeof(ULONGLONG));
    out.write((char*)&entry.totalClusters, sizeof(ULONGLONG));
    out.write((char*)&entry.creationTime, sizeof(FILETIME));
    out.write((char*)&entry.modificationTime, sizeof(FILETIME));
    out.write((char*)&entry.parentRecord, sizeof(ULONGLONG));
    out.write((char*)&entry.sequenceNumber, sizeof(WORD));

    BYTE flags = 0;
    if (entry.isDeleted) flags |= 0x01;
    if (entry.isDirectory) flags |= 0x02;
    if (entry.isResident) flags |= 0x04;
    out.write((char*)&flags, sizeof(BYTE));

    // 文件名（长度 + 数据）
    WORD nameLen = (WORD)entry.fileName.length();
    out.write((char*)&nameLen, sizeof(WORD));
    if (nameLen > 0) {
        out.write((char*)entry.fileName.data(), nameLen * sizeof(WCHAR));
    }

    // 扩展名
    WORD extLen = (WORD)entry.extension.length();
    out.write((char*)&extLen, sizeof(WORD));
    if (extLen > 0) {
        out.write((char*)entry.extension.data(), extLen * sizeof(WCHAR));
    }
}

bool MFTCache::DeserializeEntry(ifstream& in, MFTCacheEntry& entry) {
    in.read((char*)&entry.recordNumber, sizeof(ULONGLONG));
    in.read((char*)&entry.fileSize, sizeof(ULONGLONG));
    in.read((char*)&entry.startLCN, sizeof(ULONGLONG));
    in.read((char*)&entry.totalClusters, sizeof(ULONGLONG));
    in.read((char*)&entry.creationTime, sizeof(FILETIME));
    in.read((char*)&entry.modificationTime, sizeof(FILETIME));
    in.read((char*)&entry.parentRecord, sizeof(ULONGLONG));
    in.read((char*)&entry.sequenceNumber, sizeof(WORD));

    BYTE flags;
    in.read((char*)&flags, sizeof(BYTE));
    entry.isDeleted = (flags & 0x01) != 0;
    entry.isDirectory = (flags & 0x02) != 0;
    entry.isResident = (flags & 0x04) != 0;

    WORD nameLen;
    in.read((char*)&nameLen, sizeof(WORD));
    if (nameLen > 0 && nameLen < 256) {
        entry.fileName.resize(nameLen);
        in.read((char*)entry.fileName.data(), nameLen * sizeof(WCHAR));
    }

    WORD extLen;
    in.read((char*)&extLen, sizeof(WORD));
    if (extLen > 0 && extLen < 32) {
        entry.extension.resize(extLen);
        in.read((char*)entry.extension.data(), extLen * sizeof(WCHAR));
    }

    return in.good();
}

bool MFTCache::SaveToFile(const string& path) {
    string filePath = path.empty() ? GenerateCachePath(driveLetter) : path;

    ofstream out(filePath, ios::binary);
    if (!out) {
        LOG_ERROR("无法创建 MFT 缓存文件");
        return false;
    }

    // 写入头
    MFTCacheHeader header;
    header.magic = MFT_CACHE_MAGIC;
    header.version = MFT_CACHE_VERSION;
    header.totalRecords = entriesByRecord.size();
    header.deletedCount = GetDeletedCount();
    header.activeCount = GetActiveCount();
    header.driveLetter = driveLetter;
    memset(header.padding, 0, sizeof(header.padding));
    header.buildTime = buildTime;
    header.bytesPerCluster = bytesPerCluster;

    out.write((char*)&header, sizeof(header));

    // 写入条目
    for (const auto& [recordNum, entry] : entriesByRecord) {
        SerializeEntry(out, entry);
    }

    out.close();
    cacheFilePath = filePath;

    LOG_INFO_FMT("MFT cache saved: %s (%llu entries)", filePath.c_str(), header.totalRecords);
    return true;
}

bool MFTCache::LoadFromFile(char drive) {
    return LoadFromFile(GenerateCachePath(drive));
}

bool MFTCache::LoadFromFile(const string& path) {
    Clear();

    ifstream in(path, ios::binary);
    if (!in) {
        return false;
    }

    // 读取头
    MFTCacheHeader header;
    in.read((char*)&header, sizeof(header));

    if (header.magic != MFT_CACHE_MAGIC || header.version != MFT_CACHE_VERSION) {
        LOG_ERROR("MFT 缓存文件格式无效或版本不匹配");
        return false;
    }

    driveLetter = header.driveLetter;
    bytesPerCluster = header.bytesPerCluster;
    buildTime = header.buildTime;

    // 读取条目
    for (ULONGLONG i = 0; i < header.totalRecords; i++) {
        MFTCacheEntry entry;
        if (!DeserializeEntry(in, entry)) {
            LOG_ERROR("MFT 缓存文件损坏");
            Clear();
            return false;
        }

        entriesByRecord[entry.recordNumber] = entry;

        if (!entry.isResident && entry.startLCN > 0) {
            lcnToRecord[entry.startLCN] = entry.recordNumber;
        }
    }

    in.close();
    isValid = true;
    cacheFilePath = path;

    LOG_INFO_FMT("MFT cache loaded: %llu entries from %s", entriesByRecord.size(), path.c_str());
    return true;
}

// ============================================================================
// 查询接口
// ============================================================================
const MFTCacheEntry* MFTCache::GetByRecordNumber(ULONGLONG recordNum) const {
    auto it = entriesByRecord.find(recordNum);
    return (it != entriesByRecord.end()) ? &it->second : nullptr;
}

const MFTCacheEntry* MFTCache::GetByLCN(ULONGLONG lcn) const {
    // 精确匹配
    auto exactIt = lcnToRecord.find(lcn);
    if (exactIt != lcnToRecord.end()) {
        return GetByRecordNumber(exactIt->second);
    }

    // 范围匹配：找到第一个 startLCN <= lcn 的条目
    auto it = lcnToRecord.upper_bound(lcn);
    if (it != lcnToRecord.begin()) {
        --it;
        const MFTCacheEntry* entry = GetByRecordNumber(it->second);
        if (entry && entry->startLCN <= lcn &&
            lcn < entry->startLCN + entry->totalClusters) {
            return entry;
        }
    }

    return nullptr;
}

vector<const MFTCacheEntry*> MFTCache::GetByLCNRange(ULONGLONG startLCN, ULONGLONG endLCN) const {
    vector<const MFTCacheEntry*> results;

    auto it = lcnToRecord.upper_bound(startLCN);
    if (it != lcnToRecord.begin()) --it;

    while (it != lcnToRecord.end() && it->first < endLCN) {
        const MFTCacheEntry* entry = GetByRecordNumber(it->second);
        if (entry) {
            ULONGLONG entryEnd = entry->startLCN + entry->totalClusters;
            if (entryEnd > startLCN && entry->startLCN < endLCN) {
                results.push_back(entry);
            }
        }
        ++it;
    }

    return results;
}

vector<const MFTCacheEntry*> MFTCache::GetDeletedFiles() const {
    vector<const MFTCacheEntry*> results;
    for (const auto& [recordNum, entry] : entriesByRecord) {
        if (entry.isDeleted) {
            results.push_back(&entry);
        }
    }
    return results;
}

vector<const MFTCacheEntry*> MFTCache::FilterByExtension(const wstring& ext) const {
    vector<const MFTCacheEntry*> results;
    wstring lowerExt = ext;
    transform(lowerExt.begin(), lowerExt.end(), lowerExt.begin(), ::towlower);

    for (const auto& [recordNum, entry] : entriesByRecord) {
        if (entry.extension == lowerExt) {
            results.push_back(&entry);
        }
    }
    return results;
}

vector<const MFTCacheEntry*> MFTCache::SearchByName(const wstring& pattern) const {
    vector<const MFTCacheEntry*> results;
    wstring lowerPattern = pattern;
    transform(lowerPattern.begin(), lowerPattern.end(), lowerPattern.begin(), ::towlower);

    for (const auto& [recordNum, entry] : entriesByRecord) {
        wstring lowerName = entry.fileName;
        transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::towlower);

        if (lowerName.find(lowerPattern) != wstring::npos) {
            results.push_back(&entry);
        }
    }
    return results;
}

// ============================================================================
// 与签名扫描结果关联
// ============================================================================
bool MFTCache::EnrichCarvedInfo(CarvedFileInfo& carved) const {
    const MFTCacheEntry* entry = GetByLCN(carved.startLCN);
    if (!entry) return false;

    // 填充 MFT 信息
    carved.matchedMftRecord = entry->recordNumber;
    carved.creationTime = entry->creationTime;
    carved.modificationTime = entry->modificationTime;
    carved.tsSource = TS_MFT_MATCH;

    // 删除状态
    carved.deletionChecked = true;
    carved.isDeleted = entry->isDeleted;
    carved.isActiveFile = !entry->isDeleted;

    return true;
}

size_t MFTCache::EnrichCarvedInfoBatch(vector<CarvedFileInfo>& carved) const {
    size_t enrichedCount = 0;
    for (auto& info : carved) {
        if (EnrichCarvedInfo(info)) {
            enrichedCount++;
        }
    }
    return enrichedCount;
}

bool MFTCache::IsLCNActive(ULONGLONG lcn) const {
    const MFTCacheEntry* entry = GetByLCN(lcn);
    return entry && !entry->isDeleted;
}

// ============================================================================
// 状态查询
// ============================================================================
size_t MFTCache::GetDeletedCount() const {
    size_t count = 0;
    for (const auto& [recordNum, entry] : entriesByRecord) {
        if (entry.isDeleted) count++;
    }
    return count;
}

size_t MFTCache::GetActiveCount() const {
    return entriesByRecord.size() - GetDeletedCount();
}

bool MFTCache::HasValidCache(char drive, int maxAgeMinutes) {
    string path = GenerateCachePath(drive);

    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fileInfo)) {
        return false;
    }

    // 检查文件年龄
    FILETIME now;
    GetSystemTimeAsFileTime(&now);

    ULARGE_INTEGER fileTime, currentTime;
    fileTime.LowPart = fileInfo.ftLastWriteTime.dwLowDateTime;
    fileTime.HighPart = fileInfo.ftLastWriteTime.dwHighDateTime;
    currentTime.LowPart = now.dwLowDateTime;
    currentTime.HighPart = now.dwHighDateTime;

    // 100纳秒单位转分钟
    ULONGLONG diffMinutes = (currentTime.QuadPart - fileTime.QuadPart) / 600000000ULL;

    return diffMinutes <= (ULONGLONG)maxAgeMinutes;
}

ULONGLONG MFTCache::GetCacheFileSize(char drive) {
    string path = GenerateCachePath(drive);

    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fileInfo)) {
        return 0;
    }

    ULARGE_INTEGER size;
    size.LowPart = fileInfo.nFileSizeLow;
    size.HighPart = fileInfo.nFileSizeHigh;
    return size.QuadPart;
}

// ============================================================================
// MFTCacheManager 单例管理
// ============================================================================
MFTCache* MFTCacheManager::GetCache(char drive, bool forceRebuild) {
    // 检查是否已有正确驱动器的缓存
    if (!forceRebuild && globalCache && cachedDrive == drive && globalCache->IsValid()) {
        return globalCache.get();
    }

    // 尝试从磁盘加载
    if (!forceRebuild && MFTCache::HasValidCache(drive, 60)) {
        if (!globalCache) {
            globalCache = make_unique<MFTCache>();
        }
        if (globalCache->LoadFromFile(drive)) {
            cachedDrive = drive;
            cout << "MFT cache loaded from disk." << endl;
            return globalCache.get();
        }
    }

    // 需要重建
    if (!globalCache) {
        globalCache = make_unique<MFTCache>();
    }

    if (globalCache->BuildFromMFT(drive, true, true)) {
        globalCache->SaveToFile();
        cachedDrive = drive;
        return globalCache.get();
    }

    return nullptr;
}

void MFTCacheManager::ReleaseCache() {
    globalCache.reset();
    cachedDrive = 0;
}

bool MFTCacheManager::IsCacheReady(char drive) {
    return globalCache && cachedDrive == drive && globalCache->IsValid();
}
