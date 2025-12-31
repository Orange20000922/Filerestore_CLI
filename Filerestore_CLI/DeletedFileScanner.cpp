#include "DeletedFileScanner.h"
#include "Logger.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <cctype>

using namespace std;

// ==================== 路径重建过滤（性能优化）====================

// 低价值文件扩展名集合（系统文件、临时文件、配置文件等）
// 这些文件通常不是用户想要恢复的，跳过路径重建可显著提高性能
static const set<wstring> LOW_VALUE_EXTENSIONS = {
    // 系统/日志文件
    L".log", L".etl", L".evtx", L".dmp", L".mdmp",
    // 临时/缓存文件
    L".tmp", L".temp", L".cache", L".bak",
    // Windows 系统文件
    L".pf", L".mui", L".manifest", L".cat", L".mum",
    L".regtrans-ms", L".blf", L".tm.blf",
    // 配置文件（通常不是用户数据）
    L".xml", L".config", L".ini", L".inf",
    // 数据库/索引文件（系统用）
    L".edb", L".sdb", L".mdb",
    // 其他低价值文件
    L".chk", L".old", L".prx", L".lock", L".lck"
};

wstring DeletedFileScanner::GetFileExtension(const wstring& fileName) {
    size_t dotPos = fileName.rfind(L'.');
    if (dotPos == wstring::npos || dotPos == fileName.length() - 1) {
        return L"";  // 没有扩展名
    }
    wstring ext = fileName.substr(dotPos);
    // 转换为小写
    transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
    return ext;
}

bool DeletedFileScanner::ShouldSkipPathReconstruction(const wstring& fileName) {
    // 跳过以 $ 开头的系统文件（如 $MFT, $LogFile 等）
    if (!fileName.empty() && fileName[0] == L'$') {
        return true;
    }

    // 跳过以 ~ 开头的临时文件
    if (!fileName.empty() && fileName[0] == L'~') {
        return true;
    }

    // 检查扩展名是否在低价值列表中
    wstring ext = GetFileExtension(fileName);
    if (!ext.empty() && LOW_VALUE_EXTENSIONS.find(ext) != LOW_VALUE_EXTENSIONS.end()) {
        return true;
    }

    return false;
}

DeletedFileScanner::DeletedFileScanner(MFTReader* mftReader, MFTParser* mftParser, PathResolver* resolver)
    : reader(mftReader), parser(mftParser), pathResolver(resolver),
      batchReader(nullptr), useBatchReading(true), filterLevel(FILTER_SKIP_PATH) {
    // 默认启用批量读取以提高性能
    // 默认使用 FILTER_SKIP_PATH：跳过低价值文件的路径重建，平衡性能和完整性
}

DeletedFileScanner::~DeletedFileScanner() {
    if (batchReader != nullptr) {
        delete batchReader;
        batchReader = nullptr;
    }
}

vector<DeletedFileInfo> DeletedFileScanner::ScanDeletedFiles(ULONGLONG maxRecords) {
    // 自动选择：如果启用批量读取且扫描超过1000条记录，使用批量读取方式
    if (useBatchReading && (maxRecords == 0 || maxRecords > 1000)) {
        LOG_INFO("Using batch reading for better performance");
        cout << "Using batch reading (high performance mode)" << endl;
        return ScanDeletedFilesBatch(maxRecords);
    }

    // 否则使用原来的逐条读取方式（适合小规模扫描）
    LOG_INFO("Using traditional scanning method");

    vector<DeletedFileInfo> deletedFiles;

    LOG_INFO("=== Starting ScanDeletedFiles ===");

    if (!reader->IsVolumeOpen()) {
        string msg = "Volume is not open.";
        cout << msg << endl;
        LOG_ERROR(msg);
        return deletedFiles;
    }

    ULONGLONG totalRecords = reader->GetTotalMFTRecords();
    if (maxRecords == 0 || maxRecords > totalRecords) {
        maxRecords = totalRecords;
    }

    LOG_INFO_FMT("Total MFT records available: %llu", totalRecords);
    LOG_INFO_FMT("Will scan up to: %llu records", maxRecords);

    cout << "Scanning MFT for deleted files..." << endl;
    cout << "Total MFT records to scan: " << maxRecords << endl;

    ULONGLONG foundCount = 0;
    ULONGLONG scannedCount = 0;
    ULONGLONG deletedCount = 0;      // 标记为删除的记录总数
    ULONGLONG noFileNameCount = 0;   // 没有文件名的记录
    ULONGLONG filteredCount = 0;     // 被筛选条件过滤的记录
    ULONGLONG readFailCount = 0;     // 读取失败的记录
    ULONGLONG inUseCount = 0;        // 正在使用的记录

    // 创建进度条
    ProgressBar progressBar(maxRecords - 16);

    // 从记录号16开始（前16个是系统保留的）
    for (ULONGLONG i = 16; i < maxRecords; i++) {
        scannedCount++;

        // 更新进度条(每1000条记录更新一次)
        if (scannedCount % 1000 == 0) {
            progressBar.Update(scannedCount, foundCount);
        }

        vector<BYTE> record;
        if (!reader->ReadMFT(i, record)) {
            readFailCount++;
            LOG_WARNING_FMT("Failed to read MFT record #%llu", i);
            continue;
        }

        PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record.data();

        // 调试: 显示前10条记录的Flags值
        if (i < 26) {
            cout << "[DEBUG] Record #" << i << ": Signature=0x" << hex << header->Signature << dec
                 << ", Flags=0x" << hex << header->Flags << dec
                 << " (InUse=" << (header->Flags & 0x01) << ", IsDir=" << ((header->Flags & 0x02) >> 1) << ")"
                 << endl;
        }

        // 检查是否为已删除的文件（Flags=0表示未使用）
        if ((header->Flags & 0x01) == 0) {
            // 这是一个已删除的记录
            deletedCount++;

            // 在前10条删除记录上启用调试，以诊断为何没有文件名
            bool enableDebug = (deletedCount <= 10);
            if (enableDebug) {
                LOG_INFO_FMT("=== Analyzing deleted record #%llu (deleted count: %llu) ===", i, deletedCount);
            }

            ULONGLONG parentDir;
            wstring fileName = parser->GetFileNameFromRecord(record, parentDir, enableDebug);

            if (fileName.empty()) {
                noFileNameCount++;
                if (enableDebug) {
                    LOG_WARNING_FMT("Record #%llu has no filename extracted", i);
                }
                continue; // 无效记录
            }

            if (enableDebug) {
                LOG_INFO_FMT("Record #%llu: Successfully extracted filename", i);
            }

            // 根据过滤级别处理低价值文件
            bool isLowValueFile = ShouldSkipPathReconstruction(fileName);

            if (isLowValueFile && filterLevel == FILTER_EXCLUDE) {
                // 完全排除低价值文件
                filteredCount++;
                continue;
            }

            DeletedFileInfo info;
            info.recordNumber = i;
            info.fileName = fileName;
            info.parentDirectory = parentDir;
            info.isDirectory = (header->Flags & 0x02) != 0;

            // 根据过滤级别决定是否重建路径
            if (isLowValueFile && filterLevel == FILTER_SKIP_PATH) {
                // 跳过路径重建，使用占位符
                info.filePath = L"\\LowValue\\" + fileName;
            } else {
                // FILTER_NONE 或非低价值文件：重建完整路径
                try {
                    info.filePath = pathResolver->ReconstructPath(i);
                    if (info.filePath.empty()) {
                        info.filePath = L"<unknown>\\" + fileName;
                    }
                }
                catch (const exception& e) {
                    LOG_ERROR_FMT("Exception during path reconstruction for record #%llu: %s", i, e.what());
                    info.filePath = L"<unknown>\\" + fileName;
                }
                catch (...) {
                    LOG_ERROR_FMT("Unknown exception during path reconstruction for record #%llu", i);
                    info.filePath = L"<unknown>\\" + fileName;
                }
            }

            // 获取文件大小和时间
            info.fileSize = 0;
            BYTE* attrPtr = record.data() + header->FirstAttributeOffset;

            while (attrPtr < record.data() + header->UsedSize) {
                PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

                if (attr->Type == AttributeEndOfList) {
                    break;
                }

                if (attr->Type == AttributeFileName) {
                    PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                    PFILE_NAME_ATTRIBUTE fileNameAttr = (PFILE_NAME_ATTRIBUTE)(attrPtr + resAttr->ValueOffset);
                    info.fileSize = fileNameAttr->RealSize;

                    // 转换时间（FILETIME格式）
                    info.deletionTime.dwLowDateTime = (DWORD)(fileNameAttr->ModificationTime & 0xFFFFFFFF);
                    info.deletionTime.dwHighDateTime = (DWORD)(fileNameAttr->ModificationTime >> 32);
                }

                if (attr->Type == AttributeData && !info.isDirectory) {
                    if (attr->NonResident) {
                        PNONRESIDENT_ATTRIBUTE nonResAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                        if (info.fileSize == 0) {
                            info.fileSize = nonResAttr->RealSize;
                        }
                    }
                }

                // 安全检查：确保不会无限循环
                if (attr->Length == 0) {
                    LOG_WARNING_FMT("Attribute length is 0 at record #%llu, breaking attribute loop", i);
                    break;
                }

                attrPtr += attr->Length;
            }

            // 检查数据是否可用
            info.dataAvailable = parser->CheckDataAvailable(record);

            // 放宽筛选条件: 保存所有非目录的删除文件
            if (!info.isDirectory) {
                deletedFiles.push_back(info);
                foundCount++;
            } else {
                filteredCount++;
            }
        } else {
            // 正在使用的记录
            inUseCount++;
        }
    }

    // 完成进度条
    progressBar.Finish();

    cout << "Scan completed. Found " << foundCount << " deleted files." << endl;
    cout << "\n=== Detailed Scan Statistics ===" << endl;
    cout << "  Total scanned: " << scannedCount << " records" << endl;
    cout << "  Read failures: " << readFailCount << " ("
         << (scannedCount > 0 ? (readFailCount * 100 / scannedCount) : 0) << "%)" << endl;
    cout << "  In-use records: " << inUseCount << " ("
         << (scannedCount > 0 ? (inUseCount * 100 / scannedCount) : 0) << "%)" << endl;
    cout << "  Deleted records: " << deletedCount << " ("
         << (scannedCount > 0 ? (deletedCount * 100 / scannedCount) : 0) << "%)" << endl;
    cout << "    - No filename: " << noFileNameCount << endl;
    cout << "    - Directories: " << filteredCount << endl;
    cout << "    - Valid files: " << foundCount << endl;
    cout << "=================================" << endl;

    LOG_INFO_FMT("=== Scan completed. Scanned: %llu, Found: %llu deleted files ===", scannedCount, foundCount);
    LOG_INFO_FMT("Statistics: ReadFail=%llu, InUse=%llu, Deleted=%llu, NoFileName=%llu, Filtered=%llu",
                 readFailCount, inUseCount, deletedCount, noFileNameCount, filteredCount);

    // 显示缓存统计
    ULONGLONG cacheHits, cacheMisses;
    pathResolver->GetCacheStats(cacheHits, cacheMisses);
    if (cacheHits + cacheMisses > 0) {
        double hitRate = (double)cacheHits / (cacheHits + cacheMisses) * 100.0;
        LOG_INFO_FMT("Path cache stats: Hits=%llu, Misses=%llu, Hit rate=%.2f%%",
                     cacheHits, cacheMisses, hitRate);
        cout << "Path cache hit rate: " << (int)hitRate << "%" << endl;
    }

    return deletedFiles;
}

// 使用批量缓冲读取的高性能扫描实现
vector<DeletedFileInfo> DeletedFileScanner::ScanDeletedFilesBatch(ULONGLONG maxRecords) {
    vector<DeletedFileInfo> deletedFiles;

    LOG_INFO("=== Starting Batch Reading Scan ===");

    if (!reader->IsVolumeOpen()) {
        string msg = "Volume is not open.";
        cout << msg << endl;
        LOG_ERROR(msg);
        return deletedFiles;
    }

    // 初始化批量读取器
    if (batchReader == nullptr) {
        batchReader = new MFTBatchReader();

        if (!batchReader->Initialize(reader)) {
            LOG_ERROR("Failed to initialize batch reader, falling back to traditional method");
            delete batchReader;
            batchReader = nullptr;
            useBatchReading = false;
            return ScanDeletedFiles(maxRecords);  // 回退到传统方法
        }
    }

    ULONGLONG totalRecords = batchReader->GetTotalRecords();
    if (maxRecords == 0 || maxRecords > totalRecords) {
        maxRecords = totalRecords;
    }

    LOG_INFO_FMT("Batch reading scan starting: %llu records", maxRecords);

    cout << "Scanning " << maxRecords << " MFT records (batch reading mode)..." << endl;

    ULONGLONG foundCount = 0;
    ULONGLONG scannedCount = 0;
    ULONGLONG deletedCount = 0;      // 标记为删除的记录总数
    ULONGLONG noFileNameCount = 0;   // 没有文件名的记录
    ULONGLONG filteredCount = 0;     // 被筛选条件过滤的记录
    ULONGLONG emptyRecordCount = 0;  // 空记录(签名验证失败)
    ULONGLONG inUseCount = 0;        // 正在使用的记录

    // 创建进度条
    ProgressBar progressBar(maxRecords - 16);  // 从记录16开始,所以总数减16

    // 批量处理的大小(减小批量大小以降低内存压力)
    const ULONGLONG BATCH_SIZE = 256;  // 一次处理256条记录

    // 预分配结果向量空间(估计删除文件约占1%,避免频繁重新分配)
    deletedFiles.reserve(maxRecords / 100);

    // 从记录号16开始（前16个是系统保留的）
    for (ULONGLONG i = 16; i < maxRecords; i += BATCH_SIZE) {
        ULONGLONG batchCount = min(BATCH_SIZE, maxRecords - i);

        // 定期清理缓存以防止内存累积(每10万条记录清理一次)
        if (scannedCount > 0 && scannedCount % 100000 == 0) {
            pathResolver->ClearCache();
            LOG_INFO_FMT("Cache cleared at %llu records to free memory", scannedCount);
        }

        // 批量读取
        vector<vector<BYTE>> batchRecords;
        if (!batchReader->ReadMFTBatch(i, batchCount, batchRecords)) {
            LOG_WARNING_FMT("Failed to read batch starting at record %llu", i);
            continue;
        }

        // 处理批量记录
        for (ULONGLONG j = 0; j < batchRecords.size(); j++) {
            ULONGLONG recordNumber = i + j;
            scannedCount++;

            // 更新进度条(每1000条记录更新一次)
            if (scannedCount % 1000 == 0) {
                progressBar.Update(scannedCount, foundCount);
            }

            // 检查记录是否有效
            if (batchRecords[j].empty()) {
                emptyRecordCount++;
                continue;  // 无效记录(签名验证失败或读取失败)
            }

            vector<BYTE>& record = batchRecords[j];

            // 安全检查: 确保记录大小足够
            if (record.size() < sizeof(FILE_RECORD_HEADER)) {
                emptyRecordCount++;
                continue;
            }

            PFILE_RECORD_HEADER header = (PFILE_RECORD_HEADER)record.data();

            // 验证记录头的基本有效性
            if (header->UsedSize > record.size() || header->FirstAttributeOffset >= header->UsedSize) {
                emptyRecordCount++;
                continue;
            }

            // 检查是否为已删除的文件（Flags=0表示未使用）
            if ((header->Flags & 0x01) == 0) {
                // 这是一个已删除的记录
                deletedCount++;

                ULONGLONG parentDir;
                wstring fileName = parser->GetFileNameFromRecord(record, parentDir);

                if (fileName.empty()) {
                    noFileNameCount++;
                    continue; // 无效记录
                }

                // 根据过滤级别处理低价值文件
                bool isLowValueFile = ShouldSkipPathReconstruction(fileName);

                if (isLowValueFile && filterLevel == FILTER_EXCLUDE) {
                    // 完全排除低价值文件
                    filteredCount++;
                    continue;
                }

                DeletedFileInfo info;
                info.recordNumber = recordNumber;
                info.fileName = fileName;
                info.parentDirectory = parentDir;
                info.isDirectory = (header->Flags & 0x02) != 0;

                // 根据过滤级别决定是否重建路径
                if (isLowValueFile && filterLevel == FILTER_SKIP_PATH) {
                    // 跳过路径重建，使用占位符
                    info.filePath = L"\\LowValue\\" + fileName;
                } else {
                    // FILTER_NONE 或非低价值文件：重建完整路径
                    try {
                        info.filePath = pathResolver->ReconstructPath(recordNumber);
                        if (info.filePath.empty()) {
                            info.filePath = L"<unknown>\\" + fileName;
                        }
                    }
                    catch (...) {
                        info.filePath = L"<unknown>\\" + fileName;
                    }
                }

                // 获取文件大小和时间
                info.fileSize = 0;
                BYTE* attrPtr = record.data() + header->FirstAttributeOffset;
                BYTE* recordEnd = record.data() + header->UsedSize;

                while (attrPtr < recordEnd) {
                    // 边界检查: 确保有足够空间读取属性头
                    if (attrPtr + sizeof(ATTRIBUTE_HEADER) > recordEnd) {
                        break;
                    }

                    PATTRIBUTE_HEADER attr = (PATTRIBUTE_HEADER)attrPtr;

                    if (attr->Type == AttributeEndOfList) {
                        break;
                    }

                    // 安全检查：确保不会无限循环和越界
                    if (attr->Length == 0 || attr->Length > (DWORD)(recordEnd - attrPtr)) {
                        break;
                    }

                    if (attr->Type == AttributeFileName) {
                        // 边界检查: 确保有足够空间读取FileName属性
                        if (attrPtr + sizeof(ATTRIBUTE_HEADER) + sizeof(RESIDENT_ATTRIBUTE) <= recordEnd) {
                            PRESIDENT_ATTRIBUTE resAttr = (PRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                            if (attrPtr + resAttr->ValueOffset + sizeof(FILE_NAME_ATTRIBUTE) <= recordEnd) {
                                PFILE_NAME_ATTRIBUTE fileNameAttr = (PFILE_NAME_ATTRIBUTE)(attrPtr + resAttr->ValueOffset);
                                info.fileSize = fileNameAttr->RealSize;

                                // 转换时间（FILETIME格式）
                                info.deletionTime.dwLowDateTime = (DWORD)(fileNameAttr->ModificationTime & 0xFFFFFFFF);
                                info.deletionTime.dwHighDateTime = (DWORD)(fileNameAttr->ModificationTime >> 32);
                            }
                        }
                    }

                    if (attr->Type == AttributeData && !info.isDirectory) {
                        if (attr->NonResident) {
                            // 边界检查: 确保有足够空间读取NonResident属性
                            if (attrPtr + sizeof(ATTRIBUTE_HEADER) + sizeof(NONRESIDENT_ATTRIBUTE) <= recordEnd) {
                                PNONRESIDENT_ATTRIBUTE nonResAttr = (PNONRESIDENT_ATTRIBUTE)(attrPtr + sizeof(ATTRIBUTE_HEADER));
                                if (info.fileSize == 0) {
                                    info.fileSize = nonResAttr->RealSize;
                                }
                            }
                        }
                    }

                    attrPtr += attr->Length;
                }

                // 检查数据是否可用
                info.dataAvailable = parser->CheckDataAvailable(record);

                // 放宽筛选条件: 保存所有非目录的删除文件(包括fileSize=0的)
                // 或者保存所有有文件名的记录
                if (!info.isDirectory) {
                    // 非目录文件都保存
                    deletedFiles.push_back(info);
                    foundCount++;
                } else {
                    // 目录被过滤
                    filteredCount++;
                }
            } else {
                // 正在使用的记录
                inUseCount++;
            }
        }

        // 显式清理批处理数据,释放内存
        batchRecords.clear();
        batchRecords.shrink_to_fit();
    }

    // 完成进度条
    progressBar.Finish();

    cout << "Scan completed. Found " << foundCount << " deleted files." << endl;
    cout << "\n=== Detailed Scan Statistics ===" << endl;
    cout << "  Total scanned: " << scannedCount << " records" << endl;
    cout << "  Empty/Invalid: " << emptyRecordCount << " ("
         << (scannedCount > 0 ? (emptyRecordCount * 100 / scannedCount) : 0) << "%)" << endl;
    cout << "  In-use records: " << inUseCount << " ("
         << (scannedCount > 0 ? (inUseCount * 100 / scannedCount) : 0) << "%)" << endl;
    cout << "  Deleted records: " << deletedCount << " ("
         << (scannedCount > 0 ? (deletedCount * 100 / scannedCount) : 0) << "%)" << endl;
    cout << "    - No filename: " << noFileNameCount << endl;
    cout << "    - Directories: " << filteredCount << endl;
    cout << "    - Valid files: " << foundCount << endl;
    cout << "=================================" << endl;

    LOG_INFO_FMT("=== Batch reading scan completed. Scanned: %llu, Found: %llu deleted files ===",
                 scannedCount, foundCount);
    LOG_INFO_FMT("Statistics: Empty=%llu, InUse=%llu, Deleted=%llu, NoFileName=%llu, Filtered=%llu",
                 emptyRecordCount, inUseCount, deletedCount, noFileNameCount, filteredCount);

    // 显示缓存统计
    ULONGLONG cacheHits, cacheMisses;
    pathResolver->GetCacheStats(cacheHits, cacheMisses);
    if (cacheHits + cacheMisses > 0) {
        double hitRate = (double)cacheHits / (cacheHits + cacheMisses) * 100.0;
        LOG_INFO_FMT("Path cache stats: Hits=%llu, Misses=%llu, Hit rate=%.2f%%",
                     cacheHits, cacheMisses, hitRate);
        cout << "Path cache hit rate: " << (int)hitRate << "%" << endl;
    }

    return deletedFiles;
}

// 静态辅助函数：按扩展名筛选
vector<DeletedFileInfo> DeletedFileScanner::FilterByExtension(const vector<DeletedFileInfo>& files, const wstring& extension) {
    vector<DeletedFileInfo> filtered;
    wstring ext = extension;

    // 确保扩展名以点开头
    if (!ext.empty() && ext[0] != L'.') {
        ext = L"." + ext;
    }

    // 转换为小写以进行不区分大小写的比较
    transform(ext.begin(), ext.end(), ext.begin(), ::towlower);

    wcout << L"[DIAGNOSTIC FilterByExtension] Searching for extension: \"" << ext << L"\"" << endl;
    wcout << L"[DIAGNOSTIC FilterByExtension] Total files to check: " << files.size() << endl;

    int xmlCount = 0;
    for (const auto& file : files) {
        wstring fileName = file.fileName;
        transform(fileName.begin(), fileName.end(), fileName.begin(), ::towlower);

        // Count XML files for debugging
        if (fileName.length() >= 4 && fileName.substr(fileName.length() - 4) == L".xml") {
            xmlCount++;
            if (xmlCount <= 5) {  // Show first 5 XML files
                wcout << L"  [DEBUG] Found XML file: \"" << file.fileName << L"\"" << endl;
            }
        }

        // 检查文件名是否以指定扩展名结尾
        if (fileName.length() >= ext.length() &&
            fileName.substr(fileName.length() - ext.length()) == ext) {
            filtered.push_back(file);
        }
    }

    wcout << L"[DIAGNOSTIC FilterByExtension] Total XML files seen: " << xmlCount << endl;
    wcout << L"[DIAGNOSTIC FilterByExtension] Files matched: " << filtered.size() << endl;

    return filtered;
}

// 静态辅助函数：按文件大小筛选
vector<DeletedFileInfo> DeletedFileScanner::FilterBySize(const vector<DeletedFileInfo>& files,
                                                          ULONGLONG minSize, ULONGLONG maxSize) {
    vector<DeletedFileInfo> filtered;

    for (const auto& file : files) {
        if (file.fileSize >= minSize && file.fileSize <= maxSize) {
            filtered.push_back(file);
        }
    }

    return filtered;
}

// 静态辅助函数：按文件名模式筛选
vector<DeletedFileInfo> DeletedFileScanner::FilterByName(const vector<DeletedFileInfo>& files,
                                                          const wstring& namePattern) {
    vector<DeletedFileInfo> filtered;
    wstring pattern = namePattern;

    // 转换为小写以进行不区分大小写的比较
    transform(pattern.begin(), pattern.end(), pattern.begin(), ::towlower);

    for (const auto& file : files) {
        wstring fileName = file.fileName;
        transform(fileName.begin(), fileName.end(), fileName.begin(), ::towlower);

        // 检查文件名是否包含模式
        if (fileName.find(pattern) != wstring::npos) {
            filtered.push_back(file);
        }
    }

    return filtered;
}

// 静态辅助函数：过滤出用户文件（排除系统文件和临时文件）
vector<DeletedFileInfo> DeletedFileScanner::FilterUserFiles(const vector<DeletedFileInfo>& files) {
    vector<DeletedFileInfo> filtered;

    // 常见的用户文件扩展名
    vector<wstring> userExtensions = {
        L".txt", L".doc", L".docx", L".pdf", L".xls", L".xlsx", L".ppt", L".pptx",
        L".jpg", L".jpeg", L".png", L".gif", L".bmp", L".tif", L".tiff",
        L".mp3", L".mp4", L".avi", L".mkv", L".mov", L".wmv", L".flv",
        L".zip", L".rar", L".7z", L".tar", L".gz",
        L".cpp", L".h", L".c", L".java", L".py", L".js", L".html", L".css",
        L".exe", L".dll", L".msi", L".apk",
        L".db", L".sqlite", L".mdb", L".accdb"
    };

    // 需要排除的系统/临时文件模式
    vector<wstring> excludePatterns = {
        L"$", L"~", L".tmp", L".temp", L".log", L".cache",
        L".etl", L".regtrans-ms", L".blf", L".dat"
    };

    for (const auto& file : files) {
        wstring fileName = file.fileName;
        wstring lowerFileName = fileName;
        transform(lowerFileName.begin(), lowerFileName.end(), lowerFileName.begin(), ::towlower);

        // 跳过以 $ 开头的系统文件
        if (!fileName.empty() && fileName[0] == L'$') {
            continue;
        }

        // 检查是否包含排除模式
        bool shouldExclude = false;
        for (const auto& pattern : excludePatterns) {
            if (lowerFileName.find(pattern) != wstring::npos) {
                shouldExclude = true;
                break;
            }
        }

        if (shouldExclude) {
            continue;
        }

        // 检查是否是用户文件扩展名
        bool isUserFile = false;
        for (const auto& ext : userExtensions) {
            if (lowerFileName.length() >= ext.length() &&
                lowerFileName.substr(lowerFileName.length() - ext.length()) == ext) {
                isUserFile = true;
                break;
            }
        }

        // 如果是用户文件，或者文件大小 > 1KB（可能是有价值的文件）
        if (isUserFile || file.fileSize > 1024) {
            filtered.push_back(file);
        }
    }

    return filtered;
}

// ==================== 缓存管理函数 ====================

string DeletedFileScanner::GetCachePath(char driveLetter) {
    char cachePath[MAX_PATH];
    GetTempPathA(MAX_PATH, cachePath);
    string cacheFile = string(cachePath) + "deleted_files_" + driveLetter + ".cache";
    return cacheFile;
}

bool DeletedFileScanner::SaveToCache(const vector<DeletedFileInfo>& files, char driveLetter) {
    string cachePath = GetCachePath(driveLetter);
    
    LOG_INFO_FMT("Saving %zu files to cache: %s", files.size(), cachePath.c_str());
    
    ofstream ofs(cachePath, ios::binary);
    if (!ofs.is_open()) {
        LOG_ERROR_FMT("Failed to open cache file for writing: %s", cachePath.c_str());
        return false;
    }

    try {
        // 写入版本号和文件数量
        DWORD version = 1;
        ULONGLONG fileCount = files.size();
        ofs.write((char*)&version, sizeof(version));
        ofs.write((char*)&fileCount, sizeof(fileCount));

        // 写入每个文件的信息
        for (const auto& file : files) {
            // 写入记录号
            ofs.write((char*)&file.recordNumber, sizeof(file.recordNumber));
            
            // 写入文件大小
            ofs.write((char*)&file.fileSize, sizeof(file.fileSize));
            
            // 写入数据可用性标志
            ofs.write((char*)&file.dataAvailable, sizeof(file.dataAvailable));
            
            // 写入是否是目录
            ofs.write((char*)&file.isDirectory, sizeof(file.isDirectory));
            
            // 写入文件名长度和内容
            DWORD nameLen = (DWORD)file.fileName.length();
            ofs.write((char*)&nameLen, sizeof(nameLen));
            ofs.write((char*)file.fileName.c_str(), nameLen * sizeof(wchar_t));
            
            // 写入文件路径长度和内容
            DWORD pathLen = (DWORD)file.filePath.length();
            ofs.write((char*)&pathLen, sizeof(pathLen));
            ofs.write((char*)file.filePath.c_str(), pathLen * sizeof(wchar_t));
        }

        ofs.close();
        LOG_INFO_FMT("Successfully saved %zu files to cache", files.size());
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while saving cache: %s", e.what());
        ofs.close();
        return false;
    }
}

bool DeletedFileScanner::LoadFromCache(vector<DeletedFileInfo>& files, char driveLetter) {
    string cachePath = GetCachePath(driveLetter);
    
    LOG_INFO_FMT("Loading cache from: %s", cachePath.c_str());
    
    ifstream ifs(cachePath, ios::binary);
    if (!ifs.is_open()) {
        LOG_WARNING_FMT("Cache file not found: %s", cachePath.c_str());
        return false;
    }

    try {
        // 读取版本号
        DWORD version;
        ifs.read((char*)&version, sizeof(version));
        if (version != 1) {
            LOG_ERROR_FMT("Unsupported cache version: %d", version);
            ifs.close();
            return false;
        }

        // 读取文件数量
        ULONGLONG fileCount;
        ifs.read((char*)&fileCount, sizeof(fileCount));
        
        LOG_INFO_FMT("Loading %llu files from cache", fileCount);
        files.clear();
        files.reserve((size_t)fileCount);

        // 读取每个文件的信息
        for (ULONGLONG i = 0; i < fileCount; i++) {
            DeletedFileInfo info;
            
            // 读取记录号
            ifs.read((char*)&info.recordNumber, sizeof(info.recordNumber));
            
            // 读取文件大小
            ifs.read((char*)&info.fileSize, sizeof(info.fileSize));
            
            // 读取数据可用性标志
            ifs.read((char*)&info.dataAvailable, sizeof(info.dataAvailable));
            
            // 读取是否是目录
            ifs.read((char*)&info.isDirectory, sizeof(info.isDirectory));
            
            // 读取文件名
            DWORD nameLen;
            ifs.read((char*)&nameLen, sizeof(nameLen));
            info.fileName.resize(nameLen);
            ifs.read((char*)info.fileName.data(), nameLen * sizeof(wchar_t));
            
            // 读取文件路径
            DWORD pathLen;
            ifs.read((char*)&pathLen, sizeof(pathLen));
            info.filePath.resize(pathLen);
            ifs.read((char*)info.filePath.data(), pathLen * sizeof(wchar_t));
            
            files.push_back(info);
        }

        ifs.close();
        LOG_INFO_FMT("Successfully loaded %zu files from cache", files.size());
        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("Exception while loading cache: %s", e.what());
        ifs.close();
        return false;
    }
}

bool DeletedFileScanner::IsCacheValid(char driveLetter, int maxAgeMinutes) {
    string cachePath = GetCachePath(driveLetter);
    
    WIN32_FILE_ATTRIBUTE_DATA fileInfo;
    if (!GetFileAttributesExA(cachePath.c_str(), GetFileExInfoStandard, &fileInfo)) {
        LOG_DEBUG_FMT("Cache file does not exist: %s", cachePath.c_str());
        return false;
    }

    // 获取当前时间
    SYSTEMTIME currentTime;
    GetSystemTime(&currentTime);
    FILETIME currentFileTime;
    SystemTimeToFileTime(&currentTime, &currentFileTime);

    // 计算时间差（单位：100纳秒）
    ULARGE_INTEGER current, cached;
    current.LowPart = currentFileTime.dwLowDateTime;
    current.HighPart = currentFileTime.dwHighDateTime;
    cached.LowPart = fileInfo.ftLastWriteTime.dwLowDateTime;
    cached.HighPart = fileInfo.ftLastWriteTime.dwHighDateTime;

    // 转换为分钟
    ULONGLONG diffMinutes = (current.QuadPart - cached.QuadPart) / (600000000ULL);

    bool isValid = (diffMinutes <= (ULONGLONG)maxAgeMinutes);
    LOG_INFO_FMT("Cache age: %llu minutes, max age: %d minutes, valid: %s",
                 diffMinutes, maxAgeMinutes, isValid ? "yes" : "no");
    
    return isValid;
}
