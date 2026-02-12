#include "CarvedResultEnricher.h"
#include "MFTReader.h"
#include "TimestampExtractor.h"
#include "Logger.h"
#include <iostream>
#include <iomanip>

using namespace std;

// ============================================================================
// 构造函数
// ============================================================================

CarvedResultEnricher::CarvedResultEnricher(MFTReader* reader)
    : reader_(reader),
      timestampExtractionEnabled_(true),
      mftIndexBuilt_(false),
      integrityValidationEnabled_(true),
      validatedCount_(0),
      corruptedCount_(0) {
}

// ============================================================================
// 提取文件数据
// ============================================================================

bool CarvedResultEnricher::ExtractFile(ULONGLONG startLCN, ULONGLONG startOffset,
                                       ULONGLONG fileSize, vector<BYTE>& fileData) {
    ULONGLONG bytesPerCluster = reader_->GetBytesPerCluster();

    ULONGLONG totalBytes = startOffset + fileSize;
    ULONGLONG clustersNeeded = (totalBytes + bytesPerCluster - 1) / bytesPerCluster;

    const ULONGLONG MAX_READ_CLUSTERS = 100000;
    if (clustersNeeded > MAX_READ_CLUSTERS) {
        clustersNeeded = MAX_READ_CLUSTERS;
        fileSize = clustersNeeded * bytesPerCluster - startOffset;
    }

    vector<BYTE> clusterData;
    if (!reader_->ReadClusters(startLCN, clustersNeeded, clusterData)) {
        return false;
    }

    if (startOffset + fileSize > clusterData.size()) {
        fileSize = clusterData.size() - startOffset;
    }

    fileData.resize((size_t)fileSize);
    memcpy(fileData.data(), clusterData.data() + startOffset, (size_t)fileSize);

    return true;
}

// ============================================================================
// MFT 索引构建
// ============================================================================

bool CarvedResultEnricher::BuildMFTIndex(bool includeActiveFiles) {
    lcnIndex_ = make_unique<MFTLCNIndex>(reader_);
    mftIndexBuilt_ = lcnIndex_->BuildIndex(includeActiveFiles, true);

    if (!mftIndexBuilt_) {
        lcnIndex_.reset();
    }

    return mftIndexBuilt_;
}

// ============================================================================
// 为单个文件提取时间戳
// ============================================================================

void CarvedResultEnricher::ExtractTimestampForFile(CarvedFileInfo& info, const BYTE* fileData, size_t dataSize) {
    bool hasEmbedded = false;
    bool hasMftMatch = false;

    // 方案1：尝试从内嵌元数据提取
    ExtractedTimestamp embedded = TimestampExtractor::Extract(fileData, dataSize, info.extension);

    if (embedded.hasAnyTimestamp()) {
        hasEmbedded = true;

        if (embedded.hasCreation) {
            info.creationTime = embedded.creationTime;
        }
        if (embedded.hasModification) {
            info.modificationTime = embedded.modificationTime;
        }
        if (embedded.hasAccess) {
            info.accessTime = embedded.accessTime;
        }
        if (!embedded.additionalInfo.empty()) {
            info.embeddedInfo = embedded.additionalInfo;
        }
    }

    // 方案2：尝试 MFT 交叉匹配
    if (lcnIndex_ && mftIndexBuilt_) {
        vector<LCNMappingInfo> matches = lcnIndex_->FindByLCN(info.startLCN);

        if (!matches.empty()) {
            const LCNMappingInfo* bestMatch = nullptr;
            for (const auto& match : matches) {
                if (!bestMatch || (match.isDeleted && !bestMatch->isDeleted)) {
                    bestMatch = &match;
                }
            }

            if (bestMatch) {
                hasMftMatch = true;
                info.matchedMftRecord = bestMatch->mftRecordNumber;

                if (!hasEmbedded) {
                    info.creationTime = bestMatch->creationTime;
                    info.modificationTime = bestMatch->modificationTime;
                    info.accessTime = bestMatch->accessTime;
                }

                if (!bestMatch->fileName.empty()) {
                    string fileName;
                    for (wchar_t wc : bestMatch->fileName) {
                        if (wc < 128) fileName += (char)wc;
                    }
                    if (!info.embeddedInfo.empty()) {
                        info.embeddedInfo += ", ";
                    }
                    info.embeddedInfo += "MFT Name: " + fileName;
                }
            }
        }
    }

    // 设置时间戳来源
    if (hasEmbedded && hasMftMatch) {
        info.tsSource = TS_BOTH;
    } else if (hasEmbedded) {
        info.tsSource = TS_EMBEDDED;
    } else if (hasMftMatch) {
        info.tsSource = TS_MFT_MATCH;
    } else {
        info.tsSource = TS_NONE_1;
    }
}

// ============================================================================
// 批量时间戳提取
// ============================================================================

void CarvedResultEnricher::ExtractTimestampsForResults(vector<CarvedFileInfo>& results, bool showProgress) {
    if (results.empty()) {
        return;
    }

    if (!timestampExtractionEnabled_) {
        cout << "Timestamp extraction is disabled." << endl;
        return;
    }

    if (showProgress) {
        cout << "\n--- Extracting Timestamps ---" << endl;
        cout << "Files to process: " << results.size() << endl;
        cout << "MFT Index: " << (mftIndexBuilt_ ? "Available" : "Not built") << endl;
    }

    DWORD startTime = GetTickCount();
    size_t processedCount = 0;
    size_t embeddedCount = 0;
    size_t mftMatchCount = 0;
    size_t totalWithTimestamp = 0;

    const size_t HEADER_READ_SIZE = 64 * 1024;

    for (auto& info : results) {
        vector<BYTE> headerData;
        size_t readSize = min((size_t)info.fileSize, HEADER_READ_SIZE);

        if (ExtractFile(info.startLCN, info.startOffset, readSize, headerData)) {
            ExtractTimestampForFile(info, headerData.data(), headerData.size());
        } else {
            if (lcnIndex_ && mftIndexBuilt_) {
                vector<LCNMappingInfo> matches = lcnIndex_->FindByLCN(info.startLCN);
                if (!matches.empty()) {
                    const LCNMappingInfo& match = matches[0];
                    info.creationTime = match.creationTime;
                    info.modificationTime = match.modificationTime;
                    info.accessTime = match.accessTime;
                    info.matchedMftRecord = match.mftRecordNumber;
                    info.tsSource = TS_MFT_MATCH;
                }
            }
        }

        if (info.tsSource == TS_EMBEDDED || info.tsSource == TS_BOTH) {
            embeddedCount++;
        }
        if (info.tsSource == TS_MFT_MATCH || info.tsSource == TS_BOTH) {
            mftMatchCount++;
        }
        if (info.tsSource != TS_NONE_1) {
            totalWithTimestamp++;
        }

        processedCount++;

        if (showProgress && processedCount % 100 == 0) {
            double progress = (double)processedCount / results.size() * 100.0;
            cout << "\rProgress: " << fixed << setprecision(1) << progress << "% | "
                 << "Processed: " << processedCount << "/" << results.size() << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- Timestamp Extraction Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files processed: " << processedCount << endl;
        cout << "With embedded timestamp: " << embeddedCount << " ("
             << fixed << setprecision(1) << (100.0 * embeddedCount / processedCount) << "%)" << endl;
        cout << "With MFT match: " << mftMatchCount << " ("
             << fixed << setprecision(1) << (100.0 * mftMatchCount / processedCount) << "%)" << endl;
        cout << "Total with timestamp: " << totalWithTimestamp << " ("
             << fixed << setprecision(1) << (100.0 * totalWithTimestamp / processedCount) << "%)" << endl;
    }

    LOG_INFO_FMT("Timestamp extraction: %zu/%zu files have timestamps (embedded: %zu, MFT: %zu)",
                 totalWithTimestamp, processedCount, embeddedCount, mftMatchCount);
}

// ============================================================================
// 完整性验证
// ============================================================================

FileIntegrityScore CarvedResultEnricher::ValidateFileIntegrity(const CarvedFileInfo& info) {
    vector<BYTE> fileData;
    size_t readSize = (size_t)min(info.fileSize, (ULONGLONG)(2 * 1024 * 1024));

    if (!ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
        FileIntegrityScore score;
        score.diagnosis = "Failed to read file data";
        score.isLikelyCorrupted = true;
        return score;
    }

    return FileIntegrityValidator::Validate(fileData.data(), fileData.size(), info.extension);
}

void CarvedResultEnricher::ValidateIntegrityForResults(vector<CarvedFileInfo>& results, bool showProgress) {
    if (results.empty()) {
        return;
    }

    if (!integrityValidationEnabled_) {
        cout << "Integrity validation is disabled." << endl;
        return;
    }

    if (showProgress) {
        cout << "\n--- Validating File Integrity ---" << endl;
        cout << "Files to validate: " << results.size() << endl;
    }

    DWORD startTime = GetTickCount();
    validatedCount_ = 0;
    corruptedCount_ = 0;
    size_t highConfidenceCount = 0;
    size_t lowConfidenceCount = 0;

    const size_t MAX_VALIDATION_SIZE = 2 * 1024 * 1024;

    for (auto& info : results) {
        vector<BYTE> fileData;
        size_t readSize = (size_t)min(info.fileSize, (ULONGLONG)MAX_VALIDATION_SIZE);

        if (ExtractFile(info.startLCN, info.startOffset, readSize, fileData)) {
            FileIntegrityScore score = FileIntegrityValidator::Validate(
                fileData.data(), fileData.size(), info.extension);

            info.integrityScore = score.overallScore;
            info.integrityValidated = true;
            info.integrityDiagnosis = score.diagnosis;

            if (score.isLikelyCorrupted) {
                corruptedCount_++;
            }
            if (score.overallScore >= FileIntegrityValidator::HIGH_CONFIDENCE_SCORE) {
                highConfidenceCount++;
            } else if (score.overallScore < FileIntegrityValidator::MIN_INTEGRITY_SCORE) {
                lowConfidenceCount++;
            }

            if (score.overallScore < 0.5) {
                info.confidence *= 0.7;
            } else if (score.overallScore >= 0.8) {
                info.confidence = min(1.0, info.confidence * 1.1);
            }
        } else {
            info.integrityScore = 0.0;
            info.integrityValidated = false;
            info.integrityDiagnosis = "Failed to read file data";
            corruptedCount_++;
        }

        validatedCount_++;

        if (showProgress && validatedCount_ % 50 == 0) {
            double progress = (double)validatedCount_ / results.size() * 100.0;
            cout << "\rValidating: " << fixed << setprecision(1) << progress << "% | "
                 << "Processed: " << validatedCount_ << "/" << results.size() << " | "
                 << "Corrupted: " << corruptedCount_ << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- Integrity Validation Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files validated: " << validatedCount_ << endl;
        cout << "High confidence (>= 80%): " << highConfidenceCount << " ("
             << fixed << setprecision(1) << (100.0 * highConfidenceCount / validatedCount_) << "%)" << endl;
        cout << "Low confidence (< 50%): " << lowConfidenceCount << " ("
             << fixed << setprecision(1) << (100.0 * lowConfidenceCount / validatedCount_) << "%)" << endl;
        cout << "Likely corrupted: " << corruptedCount_ << " ("
             << fixed << setprecision(1) << (100.0 * corruptedCount_ / validatedCount_) << "%)" << endl;
    }

    LOG_INFO_FMT("Integrity validation: %zu/%zu files validated, %zu likely corrupted",
                 validatedCount_, results.size(), corruptedCount_);
}

vector<CarvedFileInfo> CarvedResultEnricher::FilterCorruptedFiles(const vector<CarvedFileInfo>& results,
                                                                   double minIntegrityScore) {
    vector<CarvedFileInfo> filtered;

    for (const auto& info : results) {
        if (!info.integrityValidated || info.integrityScore >= minIntegrityScore) {
            filtered.push_back(info);
        }
    }

    LOG_INFO_FMT("Filtered %zu/%zu files (min integrity score: %.2f)",
                 filtered.size(), results.size(), minIntegrityScore);

    return filtered;
}

// ============================================================================
// 删除状态检查
// ============================================================================

void CarvedResultEnricher::CheckDeletionStatus(CarvedFileInfo& info) {
    info.deletionChecked = true;
    info.isDeleted = false;
    info.isActiveFile = false;

    if (!lcnIndex_ || !mftIndexBuilt_) {
        info.isDeleted = true;
        return;
    }

    vector<LCNMappingInfo> matches = lcnIndex_->FindByLCN(info.startLCN);

    if (matches.empty()) {
        info.isDeleted = true;
        return;
    }

    bool hasActiveMatch = false;
    bool hasDeletedMatch = false;

    for (const auto& match : matches) {
        if (match.isDeleted) {
            hasDeletedMatch = true;
        } else {
            hasActiveMatch = true;
        }
    }

    if (hasActiveMatch) {
        info.isActiveFile = true;
        info.isDeleted = false;

        for (const auto& match : matches) {
            if (!match.isDeleted) {
                info.matchedMftRecord = match.mftRecordNumber;
                break;
            }
        }
    } else if (hasDeletedMatch) {
        info.isDeleted = true;
        info.isActiveFile = false;

        for (const auto& match : matches) {
            if (match.isDeleted) {
                info.matchedMftRecord = match.mftRecordNumber;
                break;
            }
        }
    } else {
        info.isDeleted = true;
    }
}

void CarvedResultEnricher::CheckDeletionStatusForResults(vector<CarvedFileInfo>& results, bool showProgress) {
    if (results.empty()) {
        return;
    }

    if (!lcnIndex_ || !mftIndexBuilt_) {
        if (showProgress) {
            cout << "Building MFT index for deletion status check..." << endl;
        }
        if (!BuildMFTIndex(true)) {
            if (showProgress) {
                cout << "[WARNING] Failed to build MFT index. Cannot verify deletion status." << endl;
            }
            for (auto& info : results) {
                info.deletionChecked = true;
                info.isDeleted = true;
                info.isActiveFile = false;
            }
            return;
        }
    }

    if (showProgress) {
        cout << "\n--- Checking Deletion Status ---" << endl;
        cout << "Files to check: " << results.size() << endl;
        cout << "MFT Index entries: " << lcnIndex_->GetIndexSize() << endl;
    }

    DWORD startTime = GetTickCount();
    size_t processedCount = 0;
    size_t deletedCount = 0;
    size_t activeCount = 0;
    size_t unknownCount = 0;

    for (auto& info : results) {
        CheckDeletionStatus(info);

        if (info.isActiveFile) {
            activeCount++;
        } else if (info.isDeleted) {
            deletedCount++;
        } else {
            unknownCount++;
        }

        processedCount++;

        if (showProgress && processedCount % 100 == 0) {
            double progress = (double)processedCount / results.size() * 100.0;
            cout << "\rChecking: " << fixed << setprecision(1) << progress << "% | "
                 << "Deleted: " << deletedCount << " | Active: " << activeCount << flush;
        }
    }

    DWORD elapsed = GetTickCount() - startTime;

    if (showProgress) {
        cout << "\r                                                                    " << endl;
        cout << "\n--- Deletion Status Check Complete ---" << endl;
        cout << "Time: " << (elapsed / 1000) << "." << ((elapsed % 1000) / 100) << " seconds" << endl;
        cout << "Files checked: " << processedCount << endl;
        cout << "Deleted files: " << deletedCount << " ("
             << fixed << setprecision(1) << (100.0 * deletedCount / processedCount) << "%)" << endl;
        cout << "Active files (not deleted): " << activeCount << " ("
             << fixed << setprecision(1) << (100.0 * activeCount / processedCount) << "%)" << endl;
        if (unknownCount > 0) {
            cout << "Unknown status: " << unknownCount << " ("
                 << fixed << setprecision(1) << (100.0 * unknownCount / processedCount) << "%)" << endl;
        }
    }

    LOG_INFO_FMT("Deletion status check: %zu deleted, %zu active, %zu unknown (total: %zu)",
                 deletedCount, activeCount, unknownCount, processedCount);
}

vector<CarvedFileInfo> CarvedResultEnricher::FilterDeletedOnly(const vector<CarvedFileInfo>& results) {
    vector<CarvedFileInfo> filtered;

    for (const auto& info : results) {
        if (!info.deletionChecked || info.isDeleted) {
            filtered.push_back(info);
        }
    }

    LOG_INFO_FMT("Filtered deleted files: %zu/%zu", filtered.size(), results.size());
    return filtered;
}

vector<CarvedFileInfo> CarvedResultEnricher::FilterActiveOnly(const vector<CarvedFileInfo>& results) {
    vector<CarvedFileInfo> filtered;

    for (const auto& info : results) {
        if (info.deletionChecked && info.isActiveFile) {
            filtered.push_back(info);
        }
    }

    LOG_INFO_FMT("Filtered active files: %zu/%zu", filtered.size(), results.size());
    return filtered;
}

size_t CarvedResultEnricher::CountDeletedFiles(const vector<CarvedFileInfo>& results) {
    size_t count = 0;
    for (const auto& info : results) {
        if (!info.deletionChecked || info.isDeleted) {
            count++;
        }
    }
    return count;
}

size_t CarvedResultEnricher::CountActiveFiles(const vector<CarvedFileInfo>& results) {
    size_t count = 0;
    for (const auto& info : results) {
        if (info.deletionChecked && info.isActiveFile) {
            count++;
        }
    }
    return count;
}
