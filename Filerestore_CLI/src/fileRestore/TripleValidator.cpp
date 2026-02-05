// TripleValidator.cpp - 三角交叉验证实现
// USN Journal + MFT Record + Signature Scan 深度关联验证

#include "TripleValidator.h"
#include "MFTStructures.h"
#include "Logger.h"
#include "ProgressBar.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

// ============================================================================
// 构造和析构
// ============================================================================

TripleValidator::TripleValidator(MFTReader* mftReader, MFTParser* mftParser)
    : reader(mftReader), parser(mftParser),
      timestampToleranceSeconds(60.0),  // 默认1分钟容差
      sizeTolerance(0.05),              // 默认5%大小容差
      bytesPerCluster(0) {
    if (reader) {
        bytesPerCluster = reader->GetBytesPerCluster();
    }
}

TripleValidator::~TripleValidator() {
    // lcnIndex 由 unique_ptr 自动管理
    // reader 和 parser 不拥有，不释放
}

// ============================================================================
// 初始化方法
// ============================================================================

bool TripleValidator::BuildLcnIndex(bool deletedOnly, bool showProgress) {
    if (!reader) {
        LOG_ERROR("TripleValidator: MFTReader not initialized");
        return false;
    }

    lcnIndex = make_unique<MFTLCNIndex>(reader);

    if (showProgress) {
        cout << "Building MFT LCN index..." << endl;
    }

    bool success = lcnIndex->BuildIndex(!deletedOnly, showProgress);

    if (success && showProgress) {
        cout << "LCN index built: " << lcnIndex->GetIndexSize() << " entries" << endl;
    }

    return success;
}

bool TripleValidator::LoadUsnDeletedRecords(const vector<UsnDeletedFileInfo>& usnRecords) {
    usnRecordsByMft.clear();
    usnRecordsByLcn.clear();

    for (const auto& usn : usnRecords) {
        UsnDeletedRecord record;
        record.fileReferenceNumber = usn.FileReferenceNumber;
        record.fileName = usn.FileName;
        record.fileSize = 0;  // USN不直接包含文件大小
        record.parentDirectory = usn.ParentFileReferenceNumber;
        record.timestamp = usn.TimeStamp;

        ULONGLONG mftRecordNum = record.GetMftRecordNumber();

        // 尝试从MFT获取数据运行
        ULONGLONG fileSize = 0;
        bool isResident = false;
        if (ParseMftDataRuns(mftRecordNum, record.dataRuns, fileSize, isResident)) {
            record.fileSize = fileSize;
        }

        // 按MFT记录号索引
        usnRecordsByMft[mftRecordNum] = record;

        // 按起始LCN索引（如果有数据运行）
        if (!record.dataRuns.empty()) {
            usnRecordsByLcn.insert({ record.dataRuns[0].lcn, mftRecordNum });
        }
    }

    LOG_INFO_FMT("Loaded %zu USN deleted records", usnRecordsByMft.size());
    return true;
}

bool TripleValidator::LoadCarvedResults(vector<CarvedFileInfo>& carvedResults) {
    carvedByLcn.clear();

    for (auto& carved : carvedResults) {
        carvedByLcn.insert({ carved.startLCN, &carved });
    }

    LOG_INFO_FMT("Loaded %zu carved file results", carvedByLcn.size());
    return true;
}

// ============================================================================
// 三角验证核心方法
// ============================================================================

TripleValidationResult TripleValidator::ValidateCarvedFile(CarvedFileInfo& carved) {
    TripleValidationResult result;
    result.hasCarvedSource = true;
    result.startLCN = carved.startLCN;
    result.signatureValid = carved.confidence >= 0.5;
    result.detectedExtension = carved.extension;

    // ========== Step 1: 查找MFT匹配 ==========
    LCNMappingInfo mftInfo;
    if (lcnIndex && FindMftInfoByLcn(carved.startLCN, mftInfo)) {
        result.hasMftSource = true;
        result.mftRecordNumber = mftInfo.mftRecordNumber;
        result.lcnMatched = true;

        // 提取MFT数据
        result.fileName = mftInfo.fileName;
        result.creationTime = mftInfo.creationTime;
        result.modificationTime = mftInfo.modificationTime;

        // 获取完整的数据运行
        ULONGLONG mftFileSize = 0;
        bool isResident = false;
        vector<DataRun> mftRuns;
        if (ParseMftDataRuns(mftInfo.mftRecordNumber, mftRuns, mftFileSize, isResident)) {
            result.dataRuns = mftRuns;
            result.exactFileSize = mftFileSize;
            result.isFragmented = (mftRuns.size() > 1);

            // 大小匹配验证
            if (carved.fileSize > 0 && mftFileSize > 0) {
                double sizeDiff = abs((double)carved.fileSize - (double)mftFileSize) / (double)mftFileSize;
                result.sizeMatched = (sizeDiff <= sizeTolerance);
            }
        }

        // 类型匹配验证
        wstring mftExt = GetExtensionFromFileName(mftInfo.fileName);
        if (!mftExt.empty() && !carved.extension.empty()) {
            wstring carvedExtW(carved.extension.begin(), carved.extension.end());
            result.typeMatched = (_wcsicmp(mftExt.c_str(), carvedExtW.c_str()) == 0);
        }

        // 时间戳匹配验证（如果签名扫描也有时间戳）
        if (carved.modificationTime.dwHighDateTime != 0 || carved.modificationTime.dwLowDateTime != 0) {
            result.timestampMatched = CompareTimestamps(carved.modificationTime,
                                                        mftInfo.modificationTime,
                                                        timestampToleranceSeconds);
        }
    }

    // ========== Step 2: 查找USN匹配 ==========
    UsnDeletedRecord usnRecord;
    if (FindMatchingUsn(carved.startLCN, usnRecord)) {
        result.hasUsnSource = true;

        // 如果还没有MFT记录号，从USN获取
        if (result.mftRecordNumber == 0) {
            result.mftRecordNumber = usnRecord.GetMftRecordNumber();
        }

        // 序列号验证
        result.expectedSequence = usnRecord.GetExpectedSequence();

        // 从MFT获取实际序列号
        if (reader) {
            vector<BYTE> mftRecord;
            if (reader->ReadMFT(result.mftRecordNumber, mftRecord) && mftRecord.size() >= 48) {
                // 序列号在偏移16处
                result.actualSequence = *(WORD*)(mftRecord.data() + 16);
                result.sequenceValid = (result.expectedSequence == result.actualSequence);
            }
        }

        // 如果MFT没有文件名，使用USN的
        if (result.fileName.empty()) {
            result.fileName = usnRecord.fileName;
        }

        // 类型匹配（USN文件名 vs 签名检测）
        if (!result.typeMatched) {
            wstring usnExt = GetExtensionFromFileName(usnRecord.fileName);
            if (!usnExt.empty() && !carved.extension.empty()) {
                wstring carvedExtW(carved.extension.begin(), carved.extension.end());
                result.typeMatched = (_wcsicmp(usnExt.c_str(), carvedExtW.c_str()) == 0);
            }
        }
    }

    // ========== Step 3: 计算置信度和验证级别 ==========
    result.confidence = CalculateConfidence(result);
    result.level = DetermineLevel(result);

    // ========== Step 4: 生成诊断信息 ==========
    result.canRecover = (result.confidence >= 0.3);

    if (result.level == VAL_TRIPLE) {
        result.diagnosis = "Triple validation passed - highest confidence";
    } else if (result.level == VAL_USN_MFT) {
        result.diagnosis = "USN+MFT matched but signature weak - verify data integrity";
    } else if (result.level == VAL_MFT_SIGNATURE) {
        result.diagnosis = "MFT+Signature matched - good confidence";
    } else if (result.level == VAL_USN_SIGNATURE) {
        result.diagnosis = "USN+Signature matched - MFT may be reused";
    } else if (result.level == VAL_SIGNATURE_ONLY) {
        result.diagnosis = "Signature only - no metadata confirmation";
    } else {
        result.diagnosis = "Validation failed - low recovery probability";
        result.canRecover = false;
    }

    // 更新原始 carved 结构
    if (result.hasMftSource) {
        carved.matchedMftRecord = result.mftRecordNumber;
        carved.isDeleted = true;  // MFT匹配表示曾经存在
        carved.deletionChecked = true;
    }

    return result;
}

TripleValidationResult TripleValidator::ValidateUsnRecord(const UsnDeletedRecord& usn) {
    TripleValidationResult result;
    result.hasUsnSource = true;
    result.mftRecordNumber = usn.GetMftRecordNumber();
    result.expectedSequence = usn.GetExpectedSequence();
    result.fileName = usn.fileName;

    // ========== Step 1: 验证MFT记录 ==========
    if (reader) {
        vector<BYTE> mftRecord;
        if (reader->ReadMFT(result.mftRecordNumber, mftRecord) && mftRecord.size() >= 48) {
            result.hasMftSource = true;

            // 序列号验证
            result.actualSequence = *(WORD*)(mftRecord.data() + 16);
            result.sequenceValid = (result.expectedSequence == result.actualSequence);

            // 获取数据运行
            ULONGLONG fileSize = 0;
            bool isResident = false;
            if (ParseMftDataRuns(result.mftRecordNumber, result.dataRuns, fileSize, isResident)) {
                result.exactFileSize = fileSize;
                result.isFragmented = (result.dataRuns.size() > 1);

                if (!result.dataRuns.empty()) {
                    result.startLCN = result.dataRuns[0].lcn;
                    result.lcnMatched = true;
                }
            }
        }
    }

    // ========== Step 2: 查找匹配的签名扫描结果 ==========
    CarvedFileInfo* carved = nullptr;
    if (FindMatchingCarved(result.mftRecordNumber, carved) && carved) {
        result.hasCarvedSource = true;
        result.signatureValid = (carved->confidence >= 0.5);
        result.detectedExtension = carved->extension;

        // 类型匹配
        wstring usnExt = GetExtensionFromFileName(usn.fileName);
        if (!usnExt.empty() && !carved->extension.empty()) {
            wstring carvedExtW(carved->extension.begin(), carved->extension.end());
            result.typeMatched = (_wcsicmp(usnExt.c_str(), carvedExtW.c_str()) == 0);
        }

        // 大小匹配
        if (result.exactFileSize > 0 && carved->fileSize > 0) {
            double sizeDiff = abs((double)carved->fileSize - (double)result.exactFileSize) / (double)result.exactFileSize;
            result.sizeMatched = (sizeDiff <= sizeTolerance);
        }
    }

    // ========== Step 3: 计算置信度 ==========
    result.confidence = CalculateConfidence(result);
    result.level = DetermineLevel(result);

    // ========== Step 4: 诊断 ==========
    if (result.sequenceValid && result.signatureValid) {
        result.diagnosis = "MFT record intact, signature valid - high recovery probability";
        result.canRecover = true;
    } else if (result.sequenceValid && !result.hasCarvedSource) {
        result.diagnosis = "MFT record intact but no signature found - data may be overwritten";
        result.canRecover = (result.confidence >= 0.4);
    } else if (!result.sequenceValid && result.signatureValid) {
        result.diagnosis = "MFT reused but signature valid - data exists at different location";
        result.canRecover = true;
    } else {
        result.diagnosis = "MFT reused and no valid signature - data likely overwritten";
        result.canRecover = false;
    }

    return result;
}

vector<TripleValidationResult> TripleValidator::ValidateCarvedFiles(
    vector<CarvedFileInfo>& carvedFiles, bool showProgress) {

    vector<TripleValidationResult> results;
    results.reserve(carvedFiles.size());

    unique_ptr<ProgressBar> progress;
    if (showProgress && carvedFiles.size() > 100) {
        progress = make_unique<ProgressBar>(carvedFiles.size());
        cout << "Validating carved files..." << endl;
    }

    for (size_t i = 0; i < carvedFiles.size(); i++) {
        results.push_back(ValidateCarvedFile(carvedFiles[i]));

        if (progress) {
            progress->Update(i + 1);
        }
    }

    if (showProgress) {
        // 统计
        size_t tripleCount = 0, doubleCount = 0, singleCount = 0, noneCount = 0;
        for (const auto& r : results) {
            switch (r.level) {
            case VAL_TRIPLE: tripleCount++; break;
            case VAL_MFT_SIGNATURE:
            case VAL_USN_SIGNATURE:
            case VAL_USN_MFT: doubleCount++; break;
            case VAL_SIGNATURE_ONLY: singleCount++; break;
            default: noneCount++; break;
            }
        }

        cout << "\nValidation Summary:" << endl;
        cout << "  Triple validation:  " << tripleCount << endl;
        cout << "  Double validation:  " << doubleCount << endl;
        cout << "  Signature only:     " << singleCount << endl;
        cout << "  No validation:      " << noneCount << endl;
    }

    return results;
}

vector<TripleValidationResult> TripleValidator::ValidateUsnRecords(bool showProgress) {
    vector<TripleValidationResult> results;
    results.reserve(usnRecordsByMft.size());

    unique_ptr<ProgressBar> progress;
    if (showProgress && usnRecordsByMft.size() > 100) {
        progress = make_unique<ProgressBar>(usnRecordsByMft.size());
        cout << "Validating USN records..." << endl;
    }

    size_t i = 0;
    for (const auto& pair : usnRecordsByMft) {
        results.push_back(ValidateUsnRecord(pair.second));

        if (progress) {
            progress->Update(++i);
        }
    }

    return results;
}

// ============================================================================
// 交叉匹配方法
// ============================================================================

bool TripleValidator::FindMatchingUsn(ULONGLONG lcn, UsnDeletedRecord& outUsn) {
    // 在LCN附近查找（允许一定范围）
    auto it = usnRecordsByLcn.lower_bound(lcn > 10 ? lcn - 10 : 0);
    auto end = usnRecordsByLcn.upper_bound(lcn + 10);

    for (; it != end; ++it) {
        ULONGLONG mftRecordNum = it->second;
        auto recordIt = usnRecordsByMft.find(mftRecordNum);
        if (recordIt != usnRecordsByMft.end()) {
            const auto& usn = recordIt->second;
            // 检查是否有数据运行匹配
            for (const auto& run : usn.dataRuns) {
                if (lcn >= run.lcn && lcn < run.lcn + run.clusterCount) {
                    outUsn = usn;
                    return true;
                }
            }
        }
    }

    return false;
}

bool TripleValidator::FindMatchingCarved(ULONGLONG mftRecordNumber, CarvedFileInfo*& outCarved) {
    // 先从USN获取LCN
    auto usnIt = usnRecordsByMft.find(mftRecordNumber);
    if (usnIt == usnRecordsByMft.end() || usnIt->second.dataRuns.empty()) {
        return false;
    }

    ULONGLONG targetLcn = usnIt->second.dataRuns[0].lcn;

    // 在carved结果中查找
    auto it = carvedByLcn.lower_bound(targetLcn > 10 ? targetLcn - 10 : 0);
    auto end = carvedByLcn.upper_bound(targetLcn + 10);

    for (; it != end; ++it) {
        if (abs((LONGLONG)it->first - (LONGLONG)targetLcn) <= 1) {
            outCarved = it->second;
            return true;
        }
    }

    return false;
}

bool TripleValidator::FindMftInfoByLcn(ULONGLONG lcn, LCNMappingInfo& outInfo) {
    if (!lcnIndex) {
        return false;
    }

    auto matches = lcnIndex->FindByLCN(lcn);
    if (!matches.empty()) {
        outInfo = matches[0];
        return true;
    }

    return false;
}

// ============================================================================
// 私有辅助方法
// ============================================================================

bool TripleValidator::ParseMftDataRuns(ULONGLONG recordNumber, vector<DataRun>& outRuns,
                                        ULONGLONG& outFileSize, bool& outIsResident) {
    if (!reader) {
        return false;
    }

    vector<BYTE> mftRecord;
    if (!reader->ReadMFT(recordNumber, mftRecord)) {
        return false;
    }

    if (mftRecord.size() < sizeof(FILE_RECORD_HEADER)) {
        return false;
    }

    FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)mftRecord.data();

    // 验证签名
    if (header->Signature != 'ELIF') {  // "FILE" 反向
        return false;
    }

    outRuns.clear();
    outFileSize = 0;
    outIsResident = false;

    // 遍历属性查找 $DATA (0x80)
    WORD offset = header->FirstAttributeOffset;
    while (offset < mftRecord.size() - 4) {
        ATTRIBUTE_HEADER* attr = (ATTRIBUTE_HEADER*)(mftRecord.data() + offset);

        if (attr->Type == 0xFFFFFFFF) {
            break;  // 属性列表结束
        }

        if (attr->Length == 0 || attr->Length > mftRecord.size() - offset) {
            break;  // 无效属性
        }

        if (attr->Type == 0x80) {  // $DATA
            if (attr->NonResident) {
                // 非驻留数据
                NONRESIDENT_ATTRIBUTE* nonRes = (NONRESIDENT_ATTRIBUTE*)attr;
                outFileSize = nonRes->RealSize;

                // 解析数据运行
                BYTE* runData = (BYTE*)attr + nonRes->DataRunOffset;
                size_t maxLen = attr->Length - nonRes->DataRunOffset;

                ULONGLONG currentLCN = 0;
                size_t pos = 0;

                while (pos < maxLen && runData[pos] != 0) {
                    BYTE header = runData[pos++];
                    BYTE lengthSize = header & 0x0F;
                    BYTE offsetSize = (header >> 4) & 0x0F;

                    if (lengthSize == 0 || pos + lengthSize + offsetSize > maxLen) {
                        break;
                    }

                    // 读取簇数量
                    ULONGLONG clusterCount = 0;
                    for (BYTE i = 0; i < lengthSize; i++) {
                        clusterCount |= ((ULONGLONG)runData[pos++] << (i * 8));
                    }

                    // 读取LCN偏移（有符号）
                    LONGLONG lcnOffset = 0;
                    for (BYTE i = 0; i < offsetSize; i++) {
                        lcnOffset |= ((LONGLONG)runData[pos++] << (i * 8));
                    }
                    // 符号扩展
                    if (offsetSize > 0 && (runData[pos - 1] & 0x80)) {
                        for (BYTE i = offsetSize; i < 8; i++) {
                            lcnOffset |= (0xFFLL << (i * 8));
                        }
                    }

                    if (lcnOffset != 0) {  // 非稀疏运行
                        currentLCN += lcnOffset;
                        outRuns.push_back(DataRun(currentLCN, clusterCount));
                    }
                }

                return !outRuns.empty();
            } else {
                // 驻留数据
                RESIDENT_ATTRIBUTE* res = (RESIDENT_ATTRIBUTE*)attr;
                outFileSize = res->ValueLength;
                outIsResident = true;
                return true;
            }
        }

        offset += attr->Length;
    }

    return false;
}

double TripleValidator::CalculateConfidence(const TripleValidationResult& result) {
    double confidence = 0.0;

    // 基础权重
    const double WEIGHT_SEQUENCE = 0.30;      // MFT序列号验证
    const double WEIGHT_SIGNATURE = 0.25;     // 签名验证
    const double WEIGHT_LCN = 0.20;           // LCN位置匹配
    const double WEIGHT_TYPE = 0.10;          // 类型匹配
    const double WEIGHT_TIMESTAMP = 0.10;     // 时间戳匹配
    const double WEIGHT_SIZE = 0.05;          // 大小匹配

    if (result.sequenceValid) {
        confidence += WEIGHT_SEQUENCE;
    }

    if (result.signatureValid) {
        confidence += WEIGHT_SIGNATURE;
    }

    if (result.lcnMatched) {
        confidence += WEIGHT_LCN;
    }

    if (result.typeMatched) {
        confidence += WEIGHT_TYPE;
    }

    if (result.timestampMatched) {
        confidence += WEIGHT_TIMESTAMP;
    }

    if (result.sizeMatched) {
        confidence += WEIGHT_SIZE;
    }

    // 三角验证加成
    if (result.hasUsnSource && result.hasMftSource && result.hasCarvedSource) {
        confidence = min(1.0, confidence * 1.1);  // 10%加成
    }

    return confidence;
}

ValidationLevel TripleValidator::DetermineLevel(const TripleValidationResult& result) {
    bool hasUsn = result.hasUsnSource;
    bool hasMft = result.hasMftSource;
    bool hasCarved = result.hasCarvedSource && result.signatureValid;

    if (hasUsn && hasMft && hasCarved) {
        return VAL_TRIPLE;
    } else if (hasUsn && hasMft) {
        return VAL_USN_MFT;
    } else if (hasUsn && hasCarved) {
        return VAL_USN_SIGNATURE;
    } else if (hasMft && hasCarved) {
        return VAL_MFT_SIGNATURE;
    } else if (hasCarved) {
        return VAL_SIGNATURE_ONLY;
    }

    return VAL_NONE;
}

bool TripleValidator::CompareTimestamps(const FILETIME& ft1, const FILETIME& ft2, double toleranceSeconds) {
    ULARGE_INTEGER time1, time2;
    time1.LowPart = ft1.dwLowDateTime;
    time1.HighPart = ft1.dwHighDateTime;
    time2.LowPart = ft2.dwLowDateTime;
    time2.HighPart = ft2.dwHighDateTime;

    // FILETIME单位是100纳秒
    ULONGLONG toleranceTicks = (ULONGLONG)(toleranceSeconds * 10000000.0);

    ULONGLONG diff = (time1.QuadPart > time2.QuadPart) ?
                     (time1.QuadPart - time2.QuadPart) :
                     (time2.QuadPart - time1.QuadPart);

    return diff <= toleranceTicks;
}

wstring TripleValidator::GetExtensionFromFileName(const wstring& fileName) {
    size_t dotPos = fileName.find_last_of(L'.');
    if (dotPos != wstring::npos && dotPos < fileName.length() - 1) {
        return fileName.substr(dotPos + 1);
    }
    return L"";
}

// ============================================================================
// 工具方法
// ============================================================================

string TripleValidator::ValidationLevelToString(ValidationLevel level) {
    switch (level) {
    case VAL_TRIPLE: return "TRIPLE (USN+MFT+Signature)";
    case VAL_USN_MFT: return "DOUBLE (USN+MFT)";
    case VAL_USN_SIGNATURE: return "DOUBLE (USN+Signature)";
    case VAL_MFT_SIGNATURE: return "DOUBLE (MFT+Signature)";
    case VAL_SIGNATURE_ONLY: return "SINGLE (Signature)";
    default: return "NONE";
    }
}

string TripleValidator::FormatConfidence(double confidence) {
    char buffer[32];
    sprintf_s(buffer, "%.1f%%", confidence * 100);
    return string(buffer);
}
