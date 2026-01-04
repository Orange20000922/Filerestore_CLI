#include "MFTParser.h"
#include <iostream>
#include "Logger.h"
using namespace std;

MFTParser::MFTParser(MFTReader* mftReader) : reader(mftReader) {
}

MFTParser::~MFTParser() {
}

bool MFTParser::ParseDataRuns(BYTE* dataRun, vector<pair<ULONGLONG, ULONGLONG>>& runs) {
    BYTE* current = dataRun;
    LONGLONG currentLCN = 0;
    int runCount = 0;

    LOG_DEBUG("Starting data run parsing...");

    while (*current != 0) {
        BYTE header = *current;
        current++;
        runCount++;

        // 提取长度字段的字节数和偏移字段的字节数
        BYTE lengthBytes = header & 0x0F;
        BYTE offsetBytes = (header >> 4) & 0x0F;

        LOG_DEBUG_FMT("Data run #%d: header=0x%02X, lengthBytes=%d, offsetBytes=%d",
                     runCount, header, lengthBytes, offsetBytes);

        // 验证字段大小
        if (lengthBytes == 0 || lengthBytes > 8) {
            LOG_WARNING_FMT("Invalid lengthBytes: %d", lengthBytes);
            break;
        }

        if (offsetBytes > 8) {
            LOG_WARNING_FMT("Invalid offsetBytes: %d", offsetBytes);
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

        LOG_DEBUG_FMT("  Length=%llu clusters, Offset=%lld, Absolute LCN=%lld",
                     length, offset, currentLCN);

        // 验证LCN是否合理（不能为负数）
        if (currentLCN < 0) {
            LOG_ERROR_FMT("Invalid LCN calculated: %lld (negative)", currentLCN);
            cout << "Error: Data run points to invalid location (negative LCN)" << endl;
            return false;
        }

        // 验证长度是否合理
        if (length == 0) {
            LOG_WARNING("Data run has zero length, skipping");
            continue;
        }

        if (length > 0x100000) {  // 超过 1M 个集群（通常不合理）
            LOG_WARNING_FMT("Suspiciously large data run: %llu clusters", length);
        }

        if (offset != 0) { // 非稀疏运行
            runs.push_back(make_pair((ULONGLONG)currentLCN, length));
            LOG_DEBUG_FMT("  Added data run: LCN=%llu, Length=%llu", (ULONGLONG)currentLCN, length);
        } else {
            LOG_DEBUG("  Sparse run (skipped)");
        }

        // 防止无限循环
        if (runCount > 1000) {
            LOG_ERROR("Too many data runs (>1000), possible corruption");
            break;
        }
    }

    LOG_DEBUG_FMT("Data run parsing completed: %d runs found, %zu valid runs",
                 runCount, runs.size());

    if (runs.empty()) {
        LOG_WARNING("No valid data runs found");
    }

    return !runs.empty();
}

bool MFTParser::ExtractFileData(vector<BYTE>& mftRecord, vector<BYTE>& fileData) {
    // 边界检查
    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        cout << "Error: MFT record too small." << endl;
        return false;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    if (header->UsedSize > mftRecord.size() || header->FirstAttributeOffset >= header->UsedSize) {
        cout << "Error: Invalid MFT record header." << endl;
        return false;
    }

    // 检查文件是否已被删除
    if (!(header->Flags & 0x01)) {
        cout << "File record is marked as deleted." << endl;
    }

    // 遍历属性找到DATA属性
    BYTE* attrPtr = mftRecord.data() + header->FirstAttributeOffset;
    BYTE* recordEnd = mftRecord.data() + header->UsedSize;

    while (attrPtr < recordEnd) {
        // 边界检查
        if (attrPtr + sizeof(ATTRIBUTE_HEADER) > recordEnd) {
            break;
        }

        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            break;
        }

        // 安全检查
        if (attr->Length == 0 || attr->Length > (DWORD)(recordEnd - attrPtr)) {
            break;
        }

        if (attr->Type == AttributeData) {
            if (attr->NonResident == 0) {
                // 常驻属性 - 数据直接存储在MFT记录中
                PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                BYTE* data = attrPtr + resAttr->ValueOffset;
                fileData.resize(resAttr->ValueLength);
                memcpy(fileData.data(), data, resAttr->ValueLength);
                cout << "File data is resident. Size: " << resAttr->ValueLength << " bytes" << endl;
                return true;
            } else {
                // 非常驻属性 - 数据存储在集群中
                PNONRESIDENT_ATTRIBUTE nonResAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                BYTE* dataRun = attrPtr + nonResAttr->DataRunOffset;

                cout << "File data is non-resident. Real size: " << nonResAttr->RealSize << " bytes" << endl;

                // 解析数据运行
                vector<pair<ULONGLONG, ULONGLONG>> runs;
                if (!ParseDataRuns(dataRun, runs)) {
                    cout << "Error: Failed to parse data runs." << endl;
                    cout << "Possible causes:" << endl;
                    cout << "  - Data run information is corrupted" << endl;
                    cout << "  - File metadata has been overwritten" << endl;
                    return false;
                }

                cout << "Found " << runs.size() << " data run(s) to read." << endl;

                // 读取所有数据运行
                fileData.clear();
                ULONGLONG totalExpectedSize = nonResAttr->RealSize;
                ULONGLONG totalReadSize = 0;
                int successfulRuns = 0;
                int failedRuns = 0;

                for (size_t i = 0; i < runs.size(); i++) {
                    auto& run = runs[i];
                    cout << "Reading data run " << (i + 1) << "/" << runs.size()
                         << ": LCN=" << run.first << ", Clusters=" << run.second << endl;

                    vector<BYTE> clusterData;
                    if (reader->ReadClusters(run.first, run.second, clusterData)) {
                        fileData.insert(fileData.end(), clusterData.begin(), clusterData.end());
                        totalReadSize += clusterData.size();
                        successfulRuns++;
                        cout << "  Success: Read " << clusterData.size() << " bytes" << endl;
                    } else {
                        failedRuns++;
                        cout << "  ERROR: Failed to read cluster run at LCN " << run.first << endl;
                        cout << "  Possible causes:" << endl;
                        cout << "    - Clusters have been overwritten by new files" << endl;
                        cout << "    - LCN is out of volume range" << endl;
                        cout << "    - Physical disk read error" << endl;
                    }
                }

                cout << endl;
                cout << "Data run summary:" << endl;
                cout << "  Successful: " << successfulRuns << "/" << runs.size() << endl;
                cout << "  Failed: " << failedRuns << "/" << runs.size() << endl;
                cout << "  Total read: " << totalReadSize << " bytes" << endl;
                cout << "  Expected: " << totalExpectedSize << " bytes" << endl;

                // 如果有任何数据运行失败，返回失败
                if (failedRuns > 0) {
                    cout << endl;
                    cout << "Recovery failed: Unable to read all data runs." << endl;
                    cout << "The file data has likely been overwritten." << endl;
                    return false;
                }

                // 截断到实际文件大小
                if (fileData.size() > nonResAttr->RealSize) {
                    fileData.resize((size_t)nonResAttr->RealSize);
                }

                if (fileData.empty()) {
                    cout << "Error: No data was recovered." << endl;
                    return false;
                }

                return true;
            }
        }

        attrPtr += attr->Length;
    }

    cout << "Error: No DATA attribute found in MFT record." << endl;
    cout << "This file may be a directory or have no data stream." << endl;
    return false;
}

bool MFTParser::GetIndexRoot(vector<BYTE>& mftRecord, vector<BYTE>& indexData) {
    // 边界检查
    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        return false;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    if (header->UsedSize > mftRecord.size() || header->FirstAttributeOffset >= header->UsedSize) {
        return false;
    }

    BYTE* attrPtr = mftRecord.data() + header->FirstAttributeOffset;
    BYTE* recordEnd = mftRecord.data() + header->UsedSize;

    while (attrPtr < recordEnd) {
        // 边界检查
        if (attrPtr + sizeof(ATTRIBUTE_HEADER) > recordEnd) {
            break;
        }

        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            break;
        }

        // 安全检查
        if (attr->Length == 0 || attr->Length > (DWORD)(recordEnd - attrPtr)) {
            break;
        }

        if (attr->Type == AttributeIndexRoot) {
            if (attr->NonResident == 0) {
                // 边界检查
                if (attrPtr + sizeof(ATTRIBUTE_HEADER) + sizeof(RESIDENT_ATTRIBUTE) <= recordEnd) {
                    PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));

                    // 检查数据是否在有效范围内
                    if (attrPtr + resAttr->ValueOffset + resAttr->ValueLength <= recordEnd) {
                        BYTE* data = attrPtr + resAttr->ValueOffset;
                        indexData.resize(resAttr->ValueLength);
                        memcpy(indexData.data(), data, resAttr->ValueLength);
                        return true;
                    }
                }
            }
        }

        attrPtr += attr->Length;
    }

    return false;
}

wstring MFTParser::GetFileNameFromAttribute(BYTE* attr) {
    PFILE_NAME_ATTRIBUTE fileNameAttr = (PFILE_NAME_ATTRIBUTE)attr;
    return wstring(fileNameAttr->FileName, fileNameAttr->FileNameLength);
}

wstring MFTParser::GetFileNameFromRecord(vector<BYTE>& mftRecord, ULONGLONG& parentDir, bool enableDebug) {
    parentDir = 0;

    // 边界检查: 确保记录大小足够
    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        if (enableDebug) LOG_DEBUG("Record size too small");
        return L"";
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    // 验证记录头的基本有效性
    if (header->UsedSize > mftRecord.size() || header->FirstAttributeOffset >= header->UsedSize) {
        if (enableDebug) LOG_DEBUG_FMT("Invalid header: UsedSize=%u, RecordSize=%zu, FirstAttrOffset=%u",
                                       header->UsedSize, mftRecord.size(), header->FirstAttributeOffset);
        return L"";
    }

    // 放宽签名检查：对于已删除的文件，签名可能不完整
    // 只要记录头的其他字段看起来合理，就尝试提取文件名
    if (header->Signature != 'ELIF') {  // 'FILE' in little-endian
        if (enableDebug) LOG_DEBUG_FMT("Warning: Invalid signature 0x%08X, but continuing...", header->Signature);
        // 不再直接返回，而是继续尝试提取文件名
    }

    BYTE* attrPtr = mftRecord.data() + header->FirstAttributeOffset;
    BYTE* recordEnd = mftRecord.data() + header->UsedSize;

    int attrCount = 0;
    bool foundFileName = false;

    while (attrPtr < recordEnd) {
        // 边界检查: 确保有足够空间读取属性头
        if (attrPtr + sizeof(ATTRIBUTE_HEADER) > recordEnd) {
            if (enableDebug) LOG_DEBUG_FMT("Not enough space for attribute header (attr %d)", attrCount);
            break;
        }

        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            if (enableDebug) LOG_DEBUG_FMT("Reached end of attributes (total %d attributes)", attrCount);
            break;
        }

        // 安全检查：确保不会无限循环和越界
        if (attr->Length == 0 || attr->Length > (DWORD)(recordEnd - attrPtr)) {
            if (enableDebug) LOG_DEBUG_FMT("Invalid attribute length: %u at attr %d", attr->Length, attrCount);
            break;
        }

        if (enableDebug) LOG_DEBUG_FMT("Attr %d: Type=0x%02X, Length=%u, NonRes=%d",
                                       attrCount, attr->Type, attr->Length, attr->NonResident);

        if (attr->Type == AttributeFileName) {
            foundFileName = true;

            // 边界检查: 确保有足够空间读取FileName属性
            if (attrPtr + sizeof(ATTRIBUTE_HEADER) + sizeof(RESIDENT_ATTRIBUTE) > recordEnd) {
                if (enableDebug) LOG_DEBUG("Not enough space for RESIDENT_ATTRIBUTE");
                attrPtr += attr->Length;
                attrCount++;
                continue;
            }

            PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));

            if (enableDebug) LOG_DEBUG_FMT("FileName attr: ValueOffset=%u, ValueLength=%u",
                                           resAttr->ValueOffset, resAttr->ValueLength);

            // 检查ValueOffset是否在有效范围内
            if (attrPtr + resAttr->ValueOffset + sizeof(FILE_NAME_ATTRIBUTE) > recordEnd) {
                if (enableDebug) LOG_DEBUG("ValueOffset out of range");
                attrPtr += attr->Length;
                attrCount++;
                continue;
            }

            PFILE_NAME_ATTRIBUTE fileNameAttr = (PFILE_NAME_ATTRIBUTE)(attrPtr + resAttr->ValueOffset);

            if (enableDebug) LOG_DEBUG_FMT("FileNameLength=%u, NameType=%u",
                                           fileNameAttr->FileNameLength, fileNameAttr->NameType);

            // 检查FileName长度是否合理
            if (fileNameAttr->FileNameLength > 0 && fileNameAttr->FileNameLength < 256) {
                // 确保FileName数组在有效范围内
                if (attrPtr + resAttr->ValueOffset + sizeof(FILE_NAME_ATTRIBUTE) +
                    (fileNameAttr->FileNameLength * sizeof(WCHAR)) <= recordEnd) {

                    parentDir = fileNameAttr->ParentDirectory & 0xFFFFFFFFFFFF;
                    wstring fileName(fileNameAttr->FileName, fileNameAttr->FileNameLength);

                    if (enableDebug) LOG_DEBUG_FMT("Successfully extracted filename, length=%u",
                                                   fileNameAttr->FileNameLength);

                    return fileName;
                } else {
                    if (enableDebug) LOG_DEBUG("FileName array out of range");
                }
            } else {
                if (enableDebug) LOG_DEBUG_FMT("Invalid FileNameLength: %u", fileNameAttr->FileNameLength);
            }
        }

        attrPtr += attr->Length;
        attrCount++;

        // 防止无限循环
        if (attrCount > 100) {
            if (enableDebug) LOG_WARNING("Too many attributes (>100), breaking");
            break;
        }
    }

    if (enableDebug && !foundFileName) {
        LOG_DEBUG_FMT("No FileName attribute found in %d attributes", attrCount);
    }

    return L"";
}

bool MFTParser::CheckDataAvailable(vector<BYTE>& mftRecord) {
    // 边界检查
    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        return false;
    }

    PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)mftRecord.data();

    if (header->UsedSize > mftRecord.size() || header->FirstAttributeOffset >= header->UsedSize) {
        return false;
    }

    BYTE* attrPtr = mftRecord.data() + header->FirstAttributeOffset;
    BYTE* recordEnd = mftRecord.data() + header->UsedSize;

    while (attrPtr < recordEnd) {
        // 边界检查
        if (attrPtr + sizeof(ATTRIBUTE_HEADER) > recordEnd) {
            break;
        }

        PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

        if (attr->Type == AttributeEndOfList) {
            break;
        }

        // 安全检查
        if (attr->Length == 0 || attr->Length > (DWORD)(recordEnd - attrPtr)) {
            break;
        }

        if (attr->Type == AttributeData) {
            if (attr->NonResident == 0) {
                // 常驻数据，肯定可用
                return true;
            } else {
                // 非常驻数据，检查是否有有效的data runs
                if (attrPtr + sizeof(ATTRIBUTE_HEADER) + sizeof(NONRESIDENT_ATTRIBUTE) <= recordEnd) {
                    PNONRESIDENT_ATTRIBUTE nonResAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));

                    // 如果RealSize为0，说明是空文件或数据已清除
                    if (nonResAttr->RealSize == 0) {
                        return false;
                    }

                    // 检查DataRunOffset是否在有效范围内
                    if (attrPtr + nonResAttr->DataRunOffset < recordEnd) {
                        BYTE* dataRun = attrPtr + nonResAttr->DataRunOffset;

                        // 尝试解析data runs以验证其有效性
                        vector<pair<ULONGLONG, ULONGLONG>> runs;
                        if (!ParseDataRuns(dataRun, runs)) {
                            // 无法解析data runs，数据不可用
                            return false;
                        }

                        // 检查是否有至少一个有效的data run
                        if (runs.empty()) {
                            return false;
                        }

                        // 注意：即使data runs看起来有效，实际的集群可能已被覆盖
                        // 这里只能检查元数据的完整性，不能保证数据真的可以恢复
                        return true;
                    }
                }
            }
        }

        attrPtr += attr->Length;
    }

    return false;
}
