#include "UsnTargetedRecovery.h"
#include "../utils/Logger.h"
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace fs = std::filesystem;

// ============================================================================
// 构造函数 / 析构函数
// ============================================================================

UsnTargetedRecovery::UsnTargetedRecovery(MFTReader* mftReader, MFTParser* mftParser)
    : reader(mftReader), parser(mftParser) {
    InitializeSignatures();
}

UsnTargetedRecovery::~UsnTargetedRecovery() {
    usnReader.Close();
}

// ============================================================================
// 签名初始化
// ============================================================================

void UsnTargetedRecovery::InitializeSignatures() {
    signatures.clear();

    // 常见文件类型签名
    signatures.push_back({"zip", {0x50, 0x4B, 0x03, 0x04}, 22});
    signatures.push_back({"docx", {0x50, 0x4B, 0x03, 0x04}, 1024});
    signatures.push_back({"xlsx", {0x50, 0x4B, 0x03, 0x04}, 1024});
    signatures.push_back({"pptx", {0x50, 0x4B, 0x03, 0x04}, 1024});
    signatures.push_back({"pdf", {0x25, 0x50, 0x44, 0x46}, 1024});
    signatures.push_back({"jpg", {0xFF, 0xD8, 0xFF}, 1024});
    signatures.push_back({"jpeg", {0xFF, 0xD8, 0xFF}, 1024});
    signatures.push_back({"png", {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}, 67});
    signatures.push_back({"gif", {0x47, 0x49, 0x46, 0x38}, 13});
    signatures.push_back({"bmp", {0x42, 0x4D}, 54});
    signatures.push_back({"mp4", {0x00, 0x00, 0x00}, 1024});  // ftyp 在偏移 4
    signatures.push_back({"mp3", {0x49, 0x44, 0x33}, 128});   // ID3 标签
    signatures.push_back({"rar", {0x52, 0x61, 0x72, 0x21}, 20});
    signatures.push_back({"7z", {0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C}, 32});
    signatures.push_back({"exe", {0x4D, 0x5A}, 1024});
    signatures.push_back({"dll", {0x4D, 0x5A}, 1024});
    signatures.push_back({"doc", {0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1}, 1024});
    signatures.push_back({"xls", {0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1}, 1024});
    signatures.push_back({"ppt", {0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1}, 1024});
}

optional<UsnTargetedRecovery::SimpleSignature>
UsnTargetedRecovery::GetSignatureForExtension(const wstring& fileName) {
    wstring ext = GetExtension(fileName);
    if (ext.empty()) return nullopt;

    // 转换为小写
    string extStr = WideToNarrow(ext);
    transform(extStr.begin(), extStr.end(), extStr.begin(), ::tolower);

    // Office 2007+ 格式共享 ZIP 签名
    if (extStr == "docx" || extStr == "xlsx" || extStr == "pptx" ||
        extStr == "odt" || extStr == "ods" || extStr == "odp") {
        extStr = "zip";
    }
    // Office 97-2003 格式共享 OLE 签名
    if (extStr == "doc" || extStr == "xls" || extStr == "ppt") {
        extStr = "doc";
    }

    for (const auto& sig : signatures) {
        if (sig.extension == extStr) {
            return sig;
        }
    }
    return nullopt;
}

// ============================================================================
// 签名验证
// ============================================================================

bool UsnTargetedRecovery::ValidateSignature(const BYTE* data, size_t dataSize,
                                            const SimpleSignature& sig,
                                            double& confidence) {
    confidence = 0.0;

    if (dataSize < sig.header.size()) {
        return false;
    }

    // 检查文件头
    bool headerMatch = memcmp(data, sig.header.data(), sig.header.size()) == 0;

    if (headerMatch) {
        confidence = 0.9;  // 签名匹配，高置信度

        // 对于特定类型，进行额外验证
        if (sig.extension == "zip" && dataSize >= 30) {
            // ZIP 本地文件头有效性检查
            if (data[4] >= 10 && data[4] <= 63) {  // 版本检查
                confidence = 0.95;
            }
        }
        else if (sig.extension == "pdf" && dataSize >= 8) {
            // PDF 版本检查 %PDF-1.x
            if (data[5] == '1' && data[6] == '.') {
                confidence = 0.95;
            }
        }
        else if (sig.extension == "jpg" && dataSize >= 4) {
            // JPEG 验证 APP0/APP1 标记
            if (data[2] == 0xFF && (data[3] == 0xE0 || data[3] == 0xE1)) {
                confidence = 0.95;
            }
        }
        return true;
    }

    return false;
}

string UsnTargetedRecovery::DetectFileType(const BYTE* data, size_t dataSize,
                                           double& confidence) {
    confidence = 0.0;

    for (const auto& sig : signatures) {
        double sigConf = 0.0;
        if (ValidateSignature(data, dataSize, sig, sigConf)) {
            if (sigConf > confidence) {
                confidence = sigConf;
                // 特殊处理：Office 文件需要进一步区分
                if (sig.extension == "zip" && dataSize >= 100) {
                    // 检查是否为 Office 文档
                    string content(reinterpret_cast<const char*>(data),
                                  min(dataSize, (size_t)1000));
                    if (content.find("word/") != string::npos) return "docx";
                    if (content.find("xl/") != string::npos) return "xlsx";
                    if (content.find("ppt/") != string::npos) return "pptx";
                }
                return sig.extension;
            }
        }
    }

    // 检查文本文件
    if (dataSize >= 10) {
        bool isText = true;
        size_t checkSize = min(dataSize, (size_t)1000);
        for (size_t i = 0; i < checkSize && isText; i++) {
            BYTE b = data[i];
            if (b < 0x09 || (b > 0x0D && b < 0x20 && b != 0x1B)) {
                if (b < 0x80) {  // ASCII 范围内的不可打印字符
                    isText = false;
                }
            }
        }
        if (isText) {
            confidence = 0.6;
            // 检查是否为 HTML/XML
            string header(reinterpret_cast<const char*>(data), min(dataSize, (size_t)100));
            if (header.find("<!DOCTYPE") != string::npos ||
                header.find("<html") != string::npos) {
                confidence = 0.8;
                return "html";
            }
            if (header.find("<?xml") != string::npos) {
                confidence = 0.8;
                return "xml";
            }
            return "txt";
        }
    }

    return "unknown";
}

// ============================================================================
// MFT 记录解析
// ============================================================================

bool UsnTargetedRecovery::ParseMFTRecordForDataRuns(
    ULONGLONG recordNumber,
    vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
    ULONGLONG& fileSize,
    bool& isResident,
    vector<BYTE>& residentData,
    WORD& sequenceNumber) {

    dataRuns.clear();
    fileSize = 0;
    isResident = false;
    residentData.clear();
    sequenceNumber = 0;

    // 读取 MFT 记录
    vector<BYTE> recordData;
    if (!reader->ReadMFT(recordNumber, recordData)) {
        LOG_ERROR_FMT("无法读取 MFT 记录 #%llu", recordNumber);
        return false;
    }

    if (recordData.size() < sizeof(FILE_RECORD_HEADER)) {
        LOG_ERROR("MFT 记录太小");
        return false;
    }

    // 解析文件记录头
    FILE_RECORD_HEADER* header = reinterpret_cast<FILE_RECORD_HEADER*>(recordData.data());

    // 验证签名 "FILE"
    if (header->Signature != 0x454C4946) {  // "FILE"
        LOG_ERROR("无效的 MFT 记录签名");
        return false;
    }

    sequenceNumber = header->SequenceNumber;

    // 遍历属性查找 $DATA
    BYTE* attrPtr = recordData.data() + header->FirstAttributeOffset;
    BYTE* endPtr = recordData.data() + recordData.size();

    while (attrPtr < endPtr) {
        ATTRIBUTE_HEADER* attrHeader = reinterpret_cast<ATTRIBUTE_HEADER*>(attrPtr);

        // 结束标记
        if (attrHeader->Type == 0xFFFFFFFF || attrHeader->Length == 0) {
            break;
        }

        // 边界检查
        if (attrPtr + attrHeader->Length > endPtr) {
            break;
        }

        // 找到 $DATA 属性 (0x80)
        if (attrHeader->Type == 0x80) {
            // 检查是否为命名流（跳过 ADS）
            if (attrHeader->NameLength == 0) {
                if (attrHeader->NonResident == 0) {
                    // 常驻数据
                    isResident = true;
                    RESIDENT_ATTRIBUTE* resAttr = reinterpret_cast<RESIDENT_ATTRIBUTE*>(
                        attrPtr + sizeof(ATTRIBUTE_HEADER));
                    fileSize = resAttr->ValueLength;

                    if (resAttr->ValueLength > 0) {
                        BYTE* dataStart = attrPtr + sizeof(ATTRIBUTE_HEADER) +
                                         sizeof(RESIDENT_ATTRIBUTE) -
                                         sizeof(BYTE) + resAttr->ValueOffset -
                                         sizeof(RESIDENT_ATTRIBUTE);
                        // 修正：使用正确的偏移
                        dataStart = attrPtr + resAttr->ValueOffset;
                        if (dataStart + resAttr->ValueLength <= endPtr) {
                            residentData.assign(dataStart, dataStart + resAttr->ValueLength);
                        }
                    }
                    return true;
                }
                else {
                    // 非常驻数据
                    isResident = false;
                    NONRESIDENT_ATTRIBUTE* nonResAttr = reinterpret_cast<NONRESIDENT_ATTRIBUTE*>(
                        attrPtr + sizeof(ATTRIBUTE_HEADER));
                    fileSize = nonResAttr->RealSize;

                    // 解析 Data Runs
                    BYTE* dataRunPtr = attrPtr + nonResAttr->DataRunOffset;
                    LONGLONG currentLCN = 0;

                    while (dataRunPtr < endPtr && *dataRunPtr != 0) {
                        BYTE header = *dataRunPtr++;
                        BYTE lengthBytes = header & 0x0F;
                        BYTE offsetBytes = (header >> 4) & 0x0F;

                        if (lengthBytes == 0) break;

                        // 读取长度
                        ULONGLONG length = 0;
                        for (int i = 0; i < lengthBytes && dataRunPtr < endPtr; i++) {
                            length |= ((ULONGLONG)*dataRunPtr++) << (i * 8);
                        }

                        // 读取偏移
                        LONGLONG offset = 0;
                        for (int i = 0; i < offsetBytes && dataRunPtr < endPtr; i++) {
                            offset |= ((LONGLONG)*dataRunPtr++) << (i * 8);
                        }

                        // 处理有符号偏移
                        if (offsetBytes > 0) {
                            int signBit = 1 << (offsetBytes * 8 - 1);
                            if (offset & signBit) {
                                offset |= ~((1LL << (offsetBytes * 8)) - 1);
                            }
                        }

                        // 计算绝对 LCN
                        currentLCN += offset;

                        if (offset != 0 && currentLCN > 0) {  // 跳过稀疏运行
                            dataRuns.push_back(make_pair((ULONGLONG)currentLCN, length));
                        }
                    }
                    return !dataRuns.empty();
                }
            }
        }

        attrPtr += attrHeader->Length;
    }

    LOG_DEBUG_FMT("MFT 记录 #%llu 没有找到 $DATA 属性", recordNumber);
    return false;
}

// ============================================================================
// 文件数据读取
// ============================================================================

bool UsnTargetedRecovery::ReadFileFromDataRuns(
    const vector<pair<ULONGLONG, ULONGLONG>>& dataRuns,
    ULONGLONG fileSize,
    vector<BYTE>& fileData) {

    fileData.clear();

    DWORD clusterSize = (DWORD)reader->GetBytesPerCluster();
    ULONGLONG bytesRead = 0;

    for (const auto& run : dataRuns) {
        ULONGLONG lcn = run.first;
        ULONGLONG clusters = run.second;

        vector<BYTE> clusterData;
        if (!reader->ReadClusters(lcn, clusters, clusterData)) {
            LOG_ERROR_FMT("无法读取簇 LCN=%llu, 数量=%llu", lcn, clusters);
            // 尝试继续读取其他片段
            ULONGLONG skipBytes = clusters * clusterSize;
            fileData.insert(fileData.end(), skipBytes, 0);
            bytesRead += skipBytes;
            continue;
        }

        // 计算本次需要的字节数
        ULONGLONG needBytes = min((ULONGLONG)clusterData.size(), fileSize - bytesRead);
        fileData.insert(fileData.end(), clusterData.begin(),
                       clusterData.begin() + needBytes);
        bytesRead += needBytes;

        if (bytesRead >= fileSize) break;
    }

    // 截断到实际文件大小
    if (fileData.size() > fileSize) {
        fileData.resize((size_t)fileSize);
    }

    return fileData.size() > 0;
}

// ============================================================================
// 文件保存
// ============================================================================

bool UsnTargetedRecovery::SaveFile(const vector<BYTE>& data, const wstring& outputPath) {
    try {
        // 确保目录存在
        fs::path path(outputPath);
        fs::create_directories(path.parent_path());

        // 写入文件
        ofstream file(outputPath, ios::binary);
        if (!file) {
            LOG_ERROR_FMT("无法创建文件: %ls", outputPath.c_str());
            return false;
        }

        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();

        return true;
    }
    catch (const exception& e) {
        LOG_ERROR_FMT("保存文件失败: %s", e.what());
        return false;
    }
}

// ============================================================================
// 核心功能：验证
// ============================================================================

UsnTargetedRecoveryResult UsnTargetedRecovery::Validate(const UsnDeletedFileInfo& usnInfo) {
    UsnTargetedRecoveryResult result;
    result.usnInfo = usnInfo;
    result.mftRecordNumber = usnInfo.GetMftRecordNumber();
    result.expectedSequence = ExtractSequenceNumber(usnInfo.FileReferenceNumber);

    // 1. 尝试解析 MFT 记录
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;
    ULONGLONG fileSize = 0;
    bool isResident = false;
    vector<BYTE> residentData;
    WORD actualSequence = 0;

    bool mftParsed = ParseMFTRecordForDataRuns(result.mftRecordNumber, dataRuns,
                                                fileSize, isResident, residentData, actualSequence);

    // 记录是否序列号匹配（用于后续判断）
    bool sequenceMatched = false;
    if (mftParsed) {
        result.dataRuns = dataRuns;
        result.fileSize = fileSize;
        result.isResident = isResident;
        result.residentData = residentData;
        result.actualSequence = actualSequence;
        sequenceMatched = (result.expectedSequence == result.actualSequence);
        result.sequenceMatched = sequenceMatched;
    }

    // 2. 计算总簇数
    result.totalClusters = 0;
    for (const auto& run : dataRuns) {
        result.totalClusters += run.second;
    }

    // 3. 读取文件头进行签名验证
    //    即使 MFT 解析失败或序列号不匹配，只要有 Data Runs 就尝试读取验证
    vector<BYTE> headerData;
    if (isResident && !residentData.empty()) {
        headerData = residentData;
    }
    else if (!dataRuns.empty()) {
        // 读取第一个簇进行签名验证
        vector<BYTE> clusterData;
        if (reader->ReadClusters(dataRuns[0].first, 1, clusterData)) {
            size_t headerSize = min(clusterData.size(), (size_t)4096);
            headerData.assign(clusterData.begin(), clusterData.begin() + headerSize);
        }
    }

    // 4. 签名验证
    result.expectedType = WideToNarrow(GetExtension(usnInfo.FileName));
    transform(result.expectedType.begin(), result.expectedType.end(),
             result.expectedType.begin(), ::tolower);

    // 检测实际类型
    double confidence = 0.0;
    if (!headerData.empty()) {
        result.detectedType = DetectFileType(headerData.data(), headerData.size(), confidence);
        result.confidence = confidence;

        // 检查期望签名
        auto expectedSig = GetSignatureForExtension(usnInfo.FileName);
        if (expectedSig.has_value()) {
            double expectedConf = 0.0;
            result.signatureMatched = ValidateSignature(headerData.data(), headerData.size(),
                                                        expectedSig.value(), expectedConf);
            if (result.signatureMatched) {
                result.confidence = max(result.confidence, expectedConf);
            }
        }
        else {
            // 无签名的文件类型（txt, csv 等），依赖检测结果
            result.signatureMatched = (result.detectedType != "unknown");
        }
    }

    // 5. 综合判断恢复状态
    //    核心改进：即使 MFT 异常，只要签名验证通过就允许恢复

    if (!mftParsed) {
        // MFT 记录无法解析
        result.status = UsnRecoveryStatus::NO_DATA_ATTRIBUTE;
        result.statusMessage = GetStatusMessage(result.status);
        result.canRecover = false;
    }
    else if (isResident && !residentData.empty()) {
        // 常驻数据，始终可恢复
        if (sequenceMatched) {
            result.status = UsnRecoveryStatus::RESIDENT_DATA;
        } else {
            // 序列号不匹配但是常驻数据
            result.status = UsnRecoveryStatus::MFT_REUSED_DATA_VALID;
        }
        result.canRecover = true;
    }
    else if (sequenceMatched) {
        // MFT 序列号匹配 - 正常流程
        if (headerData.empty()) {
            result.status = UsnRecoveryStatus::READ_ERROR;
            result.canRecover = false;
        }
        else if (result.signatureMatched || result.confidence >= 0.5) {
            result.status = UsnRecoveryStatus::SUCCESS;
            result.canRecover = true;
        }
        else {
            result.status = UsnRecoveryStatus::SIGNATURE_MISMATCH;
            result.canRecover = false;  // 可用 --force 强制恢复
        }
    }
    else {
        // 序列号不匹配 - 关键改进点
        // 尝试基于签名验证判断数据是否仍然有效
        if (headerData.empty()) {
            // 无法读取数据
            result.status = UsnRecoveryStatus::MFT_RECORD_REUSED;
            result.canRecover = false;
        }
        else if (result.signatureMatched && result.confidence >= 0.8) {
            // 签名验证通过且高置信度 - 数据很可能仍然有效
            result.status = UsnRecoveryStatus::MFT_REUSED_DATA_VALID;
            result.canRecover = true;
            LOG_INFO_FMT("MFT 序列号不匹配，但签名验证通过 (置信度: %.1f%%)，允许恢复",
                        result.confidence * 100);
        }
        else if (result.confidence >= 0.6) {
            // 中等置信度 - 可尝试恢复但需警告
            result.status = UsnRecoveryStatus::MFT_REUSED_DATA_VALID;
            result.canRecover = true;
            LOG_WARNING_FMT("MFT 序列号不匹配，签名置信度中等 (%.1f%%)，恢复可能不完整",
                           result.confidence * 100);
        }
        else {
            // 签名验证失败或低置信度 - 数据可能已被覆盖
            result.status = UsnRecoveryStatus::MFT_RECORD_REUSED;
            result.canRecover = false;
        }
    }

    result.statusMessage = GetStatusMessage(result.status);
    return result;
}

// ============================================================================
// 核心功能：恢复
// ============================================================================

UsnTargetedRecoveryResult UsnTargetedRecovery::Recover(
    const UsnDeletedFileInfo& usnInfo,
    const wstring& outputDir,
    bool forceRecover) {

    // 先验证
    UsnTargetedRecoveryResult result = Validate(usnInfo);

    // 检查是否可以恢复
    if (!result.canRecover && !forceRecover) {
        return result;
    }

    // 读取文件数据
    vector<BYTE> fileData;
    if (result.isResident) {
        fileData = result.residentData;
    }
    else {
        if (!ReadFileFromDataRuns(result.dataRuns, result.fileSize, fileData)) {
            result.status = UsnRecoveryStatus::READ_ERROR;
            result.statusMessage = GetStatusMessage(result.status);
            return result;
        }
    }

    // 构建输出路径
    wstring outputPath = outputDir;
    if (!outputPath.empty() && outputPath.back() != L'\\' && outputPath.back() != L'/') {
        outputPath += L"\\";
    }
    outputPath += usnInfo.FileName;

    // 如果文件已存在，添加序号
    if (fs::exists(outputPath)) {
        wstring baseName = fs::path(outputPath).stem();
        wstring ext = fs::path(outputPath).extension();
        int counter = 1;
        do {
            outputPath = outputDir + L"\\" + baseName + L"_" + to_wstring(counter) + ext;
            counter++;
        } while (fs::exists(outputPath));
    }

    // 保存文件
    if (!SaveFile(fileData, outputPath)) {
        result.status = UsnRecoveryStatus::WRITE_ERROR;
        result.statusMessage = GetStatusMessage(result.status);
        return result;
    }

    result.recoveredPath = outputPath;
    result.recoveredSize = fileData.size();

    if (forceRecover && !result.signatureMatched) {
        result.status = UsnRecoveryStatus::PARTIAL_RECOVERY;
    }
    else if (result.isResident) {
        result.status = UsnRecoveryStatus::RESIDENT_DATA;
    }
    else {
        result.status = UsnRecoveryStatus::SUCCESS;
    }

    result.statusMessage = GetStatusMessage(result.status);
    result.canRecover = true;

    return result;
}

// ============================================================================
// 批量操作
// ============================================================================

vector<UsnFileListItem> UsnTargetedRecovery::ValidateBatch(
    const vector<UsnDeletedFileInfo>& usnFiles,
    UsnRecoveryProgressCallback progressCallback) {

    vector<UsnFileListItem> results;
    results.reserve(usnFiles.size());

    for (size_t i = 0; i < usnFiles.size(); i++) {
        if (progressCallback) {
            progressCallback(i + 1, usnFiles.size(), usnFiles[i].FileName);
        }

        UsnFileListItem item;
        item.usnInfo = usnFiles[i];

        UsnTargetedRecoveryResult validation = Validate(usnFiles[i]);
        item.validated = true;
        item.canRecover = validation.canRecover;
        item.detectedType = validation.detectedType;
        item.confidence = validation.confidence;
        item.status = validation.status;
        item.statusMessage = validation.statusMessage;

        results.push_back(item);
    }

    return results;
}

vector<UsnTargetedRecoveryResult> UsnTargetedRecovery::RecoverBatch(
    const vector<UsnDeletedFileInfo>& usnFiles,
    const wstring& outputDir,
    bool forceRecover,
    UsnRecoveryProgressCallback progressCallback) {

    vector<UsnTargetedRecoveryResult> results;
    results.reserve(usnFiles.size());

    for (size_t i = 0; i < usnFiles.size(); i++) {
        if (progressCallback) {
            progressCallback(i + 1, usnFiles.size(), usnFiles[i].FileName);
        }

        UsnTargetedRecoveryResult result = Recover(usnFiles[i], outputDir, forceRecover);
        results.push_back(result);
    }

    return results;
}

// ============================================================================
// USN 搜索辅助
// ============================================================================

vector<UsnFileListItem> UsnTargetedRecovery::SearchAndValidate(
    char driveLetter,
    int maxTimeHours,
    const wstring& namePattern,
    size_t maxResults) {

    vector<UsnFileListItem> results;

    // 打开 USN 日志
    if (!usnReader.Open(driveLetter)) {
        LOG_ERROR_FMT("无法打开驱动器 %c: 的 USN 日志", driveLetter);
        return results;
    }

    // 搜索删除的文件
    vector<UsnDeletedFileInfo> deletedFiles;
    if (namePattern.empty()) {
        deletedFiles = usnReader.ScanRecentlyDeletedFiles(maxTimeHours * 3600, maxResults);
    }
    else {
        deletedFiles = usnReader.SearchDeletedByName(namePattern, false);
        if (deletedFiles.size() > maxResults) {
            deletedFiles.resize(maxResults);
        }
    }

    // 验证每个文件
    return ValidateBatch(deletedFiles);
}

UsnTargetedRecoveryResult UsnTargetedRecovery::SearchAndRecover(
    char driveLetter,
    const wstring& fileName,
    const wstring& outputDir,
    bool forceRecover) {

    UsnTargetedRecoveryResult result;

    // 打开 USN 日志
    if (!usnReader.Open(driveLetter)) {
        result.status = UsnRecoveryStatus::UNKNOWN_ERROR;
        result.statusMessage = "无法打开 USN 日志";
        return result;
    }

    // 搜索文件
    auto deletedFiles = usnReader.SearchDeletedByName(fileName, true);
    if (deletedFiles.empty()) {
        // 尝试模糊搜索
        deletedFiles = usnReader.SearchDeletedByName(fileName, false);
    }

    if (deletedFiles.empty()) {
        result.status = UsnRecoveryStatus::MFT_RECORD_NOT_FOUND;
        result.statusMessage = "未找到匹配的删除记录";
        return result;
    }

    // 恢复最近的一个
    return Recover(deletedFiles[0], outputDir, forceRecover);
}

// ============================================================================
// 工具方法
// ============================================================================

wstring UsnTargetedRecovery::GetExtension(const wstring& fileName) {
    size_t pos = fileName.rfind(L'.');
    if (pos != wstring::npos && pos < fileName.length() - 1) {
        return fileName.substr(pos + 1);
    }
    return L"";
}

string UsnTargetedRecovery::WideToNarrow(const wstring& wide) {
    if (wide.empty()) return "";
    int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return "";
    string narrow(size - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &narrow[0], size, nullptr, nullptr);
    return narrow;
}

wstring UsnTargetedRecovery::NarrowToWide(const string& narrow) {
    if (narrow.empty()) return L"";
    int size = MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, nullptr, 0);
    if (size <= 0) return L"";
    wstring wide(size - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, &wide[0], size);
    return wide;
}

wstring UsnTargetedRecovery::FormatFileSize(ULONGLONG size) {
    wstringstream ss;
    if (size < 1024) {
        ss << size << L" B";
    }
    else if (size < 1024 * 1024) {
        ss << fixed << setprecision(1) << (size / 1024.0) << L" KB";
    }
    else if (size < 1024 * 1024 * 1024) {
        ss << fixed << setprecision(1) << (size / (1024.0 * 1024)) << L" MB";
    }
    else {
        ss << fixed << setprecision(2) << (size / (1024.0 * 1024 * 1024)) << L" GB";
    }
    return ss.str();
}

wstring UsnTargetedRecovery::FormatTimestamp(const LARGE_INTEGER& timestamp) {
    FILETIME ft;
    ft.dwLowDateTime = timestamp.LowPart;
    ft.dwHighDateTime = timestamp.HighPart;

    SYSTEMTIME st;
    FileTimeToSystemTime(&ft, &st);

    wstringstream ss;
    ss << st.wYear << L"-"
       << setfill(L'0') << setw(2) << st.wMonth << L"-"
       << setw(2) << st.wDay << L" "
       << setw(2) << st.wHour << L":"
       << setw(2) << st.wMinute << L":"
       << setw(2) << st.wSecond;
    return ss.str();
}

string UsnTargetedRecovery::GetStatusString(UsnRecoveryStatus status) {
    switch (status) {
        case UsnRecoveryStatus::SUCCESS: return "SUCCESS";
        case UsnRecoveryStatus::MFT_RECORD_REUSED: return "MFT_REUSED";
        case UsnRecoveryStatus::MFT_REUSED_DATA_VALID: return "MFT_REUSED_OK";
        case UsnRecoveryStatus::MFT_RECORD_NOT_FOUND: return "NOT_FOUND";
        case UsnRecoveryStatus::NO_DATA_ATTRIBUTE: return "NO_DATA";
        case UsnRecoveryStatus::NO_DATA_SIGNATURE_SCAN: return "SIG_SCAN_OK";
        case UsnRecoveryStatus::DATA_OVERWRITTEN: return "OVERWRITTEN";
        case UsnRecoveryStatus::SIGNATURE_MISMATCH: return "SIG_MISMATCH";
        case UsnRecoveryStatus::PARTIAL_RECOVERY: return "PARTIAL";
        case UsnRecoveryStatus::READ_ERROR: return "READ_ERROR";
        case UsnRecoveryStatus::WRITE_ERROR: return "WRITE_ERROR";
        case UsnRecoveryStatus::RESIDENT_DATA: return "RESIDENT";
        default: return "UNKNOWN";
    }
}

string UsnTargetedRecovery::GetStatusMessage(UsnRecoveryStatus status) {
    switch (status) {
        case UsnRecoveryStatus::SUCCESS:
            return "文件可恢复";
        case UsnRecoveryStatus::MFT_RECORD_REUSED:
            return "MFT 记录已被重用，签名验证失败，数据可能已丢失";
        case UsnRecoveryStatus::MFT_REUSED_DATA_VALID:
            return "MFT 记录已被重用，但签名验证通过，数据可能有效";
        case UsnRecoveryStatus::MFT_RECORD_NOT_FOUND:
            return "未找到 MFT 记录";
        case UsnRecoveryStatus::NO_DATA_ATTRIBUTE:
            return "文件没有数据属性（可能是目录或特殊文件）";
        case UsnRecoveryStatus::NO_DATA_SIGNATURE_SCAN:
            return "无 MFT 数据，但签名扫描找到匹配数据";
        case UsnRecoveryStatus::DATA_OVERWRITTEN:
            return "文件数据已被覆盖";
        case UsnRecoveryStatus::SIGNATURE_MISMATCH:
            return "文件签名不匹配，数据可能已损坏";
        case UsnRecoveryStatus::PARTIAL_RECOVERY:
            return "部分恢复（强制模式）";
        case UsnRecoveryStatus::READ_ERROR:
            return "读取磁盘错误";
        case UsnRecoveryStatus::WRITE_ERROR:
            return "写入文件错误";
        case UsnRecoveryStatus::RESIDENT_DATA:
            return "小文件，数据完整（常驻数据）";
        default:
            return "未知错误";
    }
}

// ============================================================================
// MFT 信息增强功能（复用 ParseMFTRecordForDataRuns）
// ============================================================================

bool UsnTargetedRecovery::EnrichWithMFT(UsnDeletedFileInfo& usnInfo) {
    usnInfo.MftInfoValid = false;
    usnInfo.MftRecordReused = false;
    usnInfo.FileSize = 0;

    ULONGLONG recordNumber = usnInfo.GetMftRecordNumber();
    WORD expectedSequence = usnInfo.GetExpectedSequence();

    // 复用现有的 MFT 解析函数
    vector<pair<ULONGLONG, ULONGLONG>> dataRuns;
    ULONGLONG fileSize = 0;
    bool isResident = false;
    vector<BYTE> residentData;
    WORD actualSequence = 0;

    if (!ParseMFTRecordForDataRuns(recordNumber, dataRuns, fileSize,
                                    isResident, residentData, actualSequence)) {
        return false;
    }

    // 填充信息
    usnInfo.FileSize = fileSize;
    usnInfo.MftRecordReused = (actualSequence != expectedSequence);
    usnInfo.MftInfoValid = true;

    return true;
}

size_t UsnTargetedRecovery::EnrichWithMFTBatch(
    vector<UsnDeletedFileInfo>& usnFiles,
    UsnRecoveryProgressCallback progressCallback) {

    size_t successCount = 0;
    size_t total = usnFiles.size();

    for (size_t i = 0; i < total; i++) {
        if (EnrichWithMFT(usnFiles[i])) {
            successCount++;
        }

        if (progressCallback && (i % 100 == 0 || i == total - 1)) {
            progressCallback(i + 1, total, usnFiles[i].FileName);
        }
    }

    return successCount;
}

vector<UsnDeletedFileInfo> UsnTargetedRecovery::FilterBySize(
    const vector<UsnDeletedFileInfo>& usnFiles,
    ULONGLONG minSize,
    ULONGLONG maxSize,
    bool requireMftInfo) {

    vector<UsnDeletedFileInfo> filtered;

    for (const auto& info : usnFiles) {
        if (requireMftInfo && !info.MftInfoValid) {
            continue;
        }
        if (minSize > 0 && info.FileSize < minSize) {
            continue;
        }
        if (maxSize > 0 && info.FileSize > maxSize) {
            continue;
        }
        filtered.push_back(info);
    }

    return filtered;
}

vector<UsnDeletedFileInfo> UsnTargetedRecovery::FilterByExtension(
    const vector<UsnDeletedFileInfo>& usnFiles,
    const vector<wstring>& extensions) {

    vector<UsnDeletedFileInfo> filtered;

    for (const auto& info : usnFiles) {
        wstring ext = GetExtension(info.FileName);
        transform(ext.begin(), ext.end(), ext.begin(), ::towlower);

        for (const auto& targetExt : extensions) {
            wstring targetLower = targetExt;
            transform(targetLower.begin(), targetLower.end(), targetLower.begin(), ::towlower);

            if (ext == targetLower) {
                filtered.push_back(info);
                break;
            }
        }
    }

    return filtered;
}
