// DiagnosticCommands.cpp - 系统诊断相关命令实现
// 包含: DiagnoseMFTCommand, DetectOverwriteCommand, ScanUsnCommand, ZipInfoCommand

#include "cmd.h"
#include "CommandUtils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include "MFTReader.h"
#include "FileRestore.h"
#include "OverwriteDetector.h"
#include "UsnJournalReader.h"
#include "ProgressBar.h"
#include "ZipStructureParser.h"

using namespace std;

// USN_REASON 标志定义（用于显示删除类型）
#ifndef USN_REASON_FILE_DELETE
#define USN_REASON_FILE_DELETE 0x00000200
#endif
#ifndef USN_REASON_RENAME_OLD_NAME
#define USN_REASON_RENAME_OLD_NAME 0x00001000
#endif

// ============================================================================
// DiagnoseMFTCommand - 诊断 MFT 碎片化
// ============================================================================
DEFINE_COMMAND_BASE(DiagnoseMFTCommand, "diagnosemft |name", TRUE)
REGISTER_COMMAND(DiagnoseMFTCommand);

void DiagnoseMFTCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 1) {
		cout << "Usage: diagnosemft <drive_letter>" << endl;
		cout << "Example: diagnosemft C" << endl;
		return;
	}

	string& driveStr = GET_ARG_STRING(0);
	char driveLetter;

	if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
		cout << "Invalid drive letter." << endl;
		return;
	}

	MFTReader reader;
	if (!reader.OpenVolume(driveLetter)) {
		cout << "Failed to open volume " << driveLetter << ":/" << endl;
		return;
	}

	reader.DiagnoseMFTFragmentation();
}

// ============================================================================
// DetectOverwriteCommand - 检测文件覆盖
// ============================================================================
DEFINE_COMMAND_BASE(DetectOverwriteCommand, "detectoverwrite |name |name |name", TRUE)
REGISTER_COMMAND(DetectOverwriteCommand);

void DetectOverwriteCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Invalid Args! Usage: detectoverwrite <drive_letter> <MFT_record_number> [mode]" << endl;
		cout << "Modes: fast, balanced, thorough" << endl;
		return;
	}

	string& driveStr = GET_ARG_STRING(0);
	string& recordStr = GET_ARG_STRING(1);

	char driveLetter;
	if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
		cout << "Invalid drive letter." << endl;
		return;
	}

	ULONGLONG recordNumber;
	if (!CommandUtils::ParseRecordNumber(recordStr, recordNumber)) {
		cout << "Invalid MFT record number." << endl;
		return;
	}

	DetectionMode mode = MODE_BALANCED;
	if (GET_ARG_COUNT() >= 3) {
		string& modeStr = GET_ARG_STRING(2);
		if (modeStr == "fast") mode = MODE_FAST;
		else if (modeStr == "thorough") mode = MODE_THOROUGH;
	}

	cout << "=== Overwrite Detection ===" << endl;
	cout << "Drive: " << driveLetter << ":" << endl;
	cout << "MFT Record: " << recordNumber << endl;

	FileRestore fileRestore;
	OverwriteDetector* detector = fileRestore.GetOverwriteDetector();
	detector->SetDetectionMode(mode);

	OverwriteDetectionResult result = fileRestore.DetectFileOverwrite(driveLetter, recordNumber);

	cout << "\n=== Detection Summary ===" << endl;
	cout << "Overwrite Percentage: " << result.overwritePercentage << "%" << endl;

	if (result.isFullyAvailable) {
		cout << "Status: [EXCELLENT] Fully Recoverable" << endl;
	}
	else if (result.isPartiallyAvailable) {
		cout << "Status: [WARNING] Partially Recoverable" << endl;
	}
	else {
		cout << "Status: [FAILED] Not Recoverable" << endl;
	}
}

// ============================================================================
// ScanUsnCommand - 扫描 USN 日志
// ============================================================================
DEFINE_COMMAND_BASE(ScanUsnCommand, "scanusn |name |name", TRUE)
REGISTER_COMMAND(ScanUsnCommand);

void ScanUsnCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1 || GET_ARG_COUNT() > 2) {
		cout << "Invalid Args! Usage: scanusn <drive_letter> [max_hours]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		int maxHours = 1;

		if (HAS_ARG(1)) {
			string& hoursStr = GET_ARG_STRING(1);
			try {
				maxHours = stoi(hoursStr);
				if (maxHours <= 0) maxHours = 1;
			}
			catch (...) {}
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		cout << "\n========== USN Journal Scanner ==========" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Time range: Last " << maxHours << " hour(s)" << endl;

		UsnJournalReader usnReader;

		if (!usnReader.Open(driveLetter)) {
			cout << "\n[ERROR] Failed to open USN Journal for drive " << driveLetter << ":" << endl;
			cout << "Possible reasons:" << endl;
			cout << "  - USN Journal is disabled on this drive" << endl;
			cout << "  - Insufficient privileges (run as Administrator)" << endl;
			return;
		}

		// 显示 USN Journal 统计信息（诊断用）
		UsnJournalStats stats;
		if (usnReader.GetJournalStats(stats)) {
			cout << "\n--- USN Journal Statistics ---" << endl;
			cout << "Journal ID: " << stats.UsnJournalID << endl;
			cout << "First USN: " << stats.FirstUsn << endl;
			cout << "Next USN: " << stats.NextUsn << endl;
			cout << "Max Size: " << (stats.MaximumSize / (1024 * 1024)) << " MB" << endl;

			// 计算 Journal 使用率
			ULONGLONG usedSize = stats.NextUsn - stats.FirstUsn;
			double usagePercent = (stats.MaximumSize > 0) ?
				((double)usedSize / stats.MaximumSize * 100.0) : 0;
			cout << "Usage: ~" << fixed << setprecision(1) << usagePercent << "%" << endl;

			if (usagePercent > 90.0) {
				cout << "WARNING: Journal is nearly full! Old records may have been overwritten." << endl;
			}
			cout << "------------------------------" << endl;
		}

		int maxTimeSeconds = maxHours * 3600;
		vector<UsnDeletedFileInfo> deletedFiles = usnReader.ScanRecentlyDeletedFiles(maxTimeSeconds, 10000);

		if (deletedFiles.empty()) {
			cout << "\nNo deleted files found in the specified time range." << endl;
			cout << "\nPossible reasons:" << endl;
			cout << "  1. No files were deleted in the last " << maxHours << " hour(s)" << endl;
			cout << "  2. USN Journal may have wrapped (old records overwritten)" << endl;
			cout << "  3. Files were moved to Recycle Bin (try searching by name)" << endl;
			cout << "  4. Try increasing the time range: scanusn " << driveLetter << " 24" << endl;
			return;
		}

		cout << "\n===== Recently Deleted Files =====" << endl;
		cout << "Found: " << deletedFiles.size() << " deleted file records" << endl;
		cout << "(Sorted by time, newest first)" << endl;
		cout << endl;

		size_t displayLimit = min(deletedFiles.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = deletedFiles[i];

			SYSTEMTIME st;
			FILETIME ft;
			ZeroMemory(&ft, sizeof(FILETIME));
			ft.dwLowDateTime = info.TimeStamp.LowPart;
			ft.dwHighDateTime = info.TimeStamp.HighPart;
			FileTimeToSystemTime(&ft, &st);

			// 转换为本地时间
			SYSTEMTIME localSt;
			SystemTimeToTzSpecificLocalTime(NULL, &st, &localSt);

			cout << "[" << info.GetMftRecordNumber() << "] ";
			wcout << info.FileName << " | ";
			printf("%04d-%02d-%02d %02d:%02d:%02d",
				localSt.wYear, localSt.wMonth, localSt.wDay,
				localSt.wHour, localSt.wMinute, localSt.wSecond);

			// 显示操作类型
			if (info.Reason & USN_REASON_FILE_DELETE) {
				cout << " [DELETE]";
			}
			if (info.Reason & USN_REASON_RENAME_OLD_NAME) {
				cout << " [RENAME/MOVE]";
			}
			cout << endl;
		}

		if (deletedFiles.size() > displayLimit) {
			cout << "\n... and " << (deletedFiles.size() - displayLimit) << " more records" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// ZipInfoCommand - ZIP 结构解析和诊断
// ============================================================================
DEFINE_COMMAND_BASE(ZipInfoCommand, "zipinfo |name |name |name |name |name", TRUE)
REGISTER_COMMAND(ZipInfoCommand);

void ZipInfoCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1) {
		cout << "Usage: zipinfo <zip_file_path> [options]" << endl;
		cout << "Options:" << endl;
		cout << "  -v, --verbose    Show detailed information" << endl;
		cout << "  -l, --list       List all files in archive" << endl;
		cout << "  -c, --crc        Verify CRC32 checksums" << endl;
		cout << endl;
		cout << "Example:" << endl;
		cout << "  zipinfo D:\\test.zip" << endl;
		cout << "  zipinfo D:\\test.zip -l" << endl;
		return;
	}

	// 解析参数
	string filePath = GET_ARG_STRING(0);
	bool verbose = false;
	bool listFiles = false;
	bool verifyCRC = false;

	for (size_t i = 1; i < GET_ARG_COUNT(); i++) {
		string arg = GET_ARG_STRING(i);
		if (arg == "-v" || arg == "--verbose") {
			verbose = true;
		} else if (arg == "-l" || arg == "--list") {
			listFiles = true;
		} else if (arg == "-c" || arg == "--crc") {
			verifyCRC = true;
		}
	}

	// 转换路径
	wstring wFilePath(filePath.begin(), filePath.end());

	// 检查文件是否存在
	DWORD attrs = GetFileAttributesW(wFilePath.c_str());
	if (attrs == INVALID_FILE_ATTRIBUTES) {
		cout << "[ERROR] File not found: " << filePath << endl;
		return;
	}

	cout << "========================================" << endl;
	cout << "ZIP Structure Analysis" << endl;
	cout << "========================================" << endl;
	cout << "File: " << filePath << endl;
	cout << endl;

	// 解析ZIP结构
	ZipParser::ZipParseResult result = ZipParser::ZipStructureParser::ParseFile(wFilePath);

	// 显示基本信息
	cout << "--- Basic Info ---" << endl;
	cout << "File Size:     " << result.actualFileSize << " bytes" << endl;

	if (!result.hasValidEOCD) {
		cout << "[ERROR] " << result.errorMessage << endl;
		cout << endl;
		cout << "Recovery Suggestion:" << endl;
		cout << "  - File may be truncated or corrupted" << endl;
		cout << "  - Try searching for Local File Headers (PK\\x03\\x04)" << endl;
		return;
	}

	cout << "EOCD Offset:   " << result.eocdOffset << endl;
	cout << "CD Offset:     " << result.cdOffset << endl;
	cout << "CD Size:       " << result.cdSize << " bytes" << endl;
	cout << "Entry Count:   " << result.declaredEntryCount << " (declared)" << endl;
	cout << "Parsed:        " << result.entries.size() << " entries" << endl;
	cout << "ZIP64:         " << (result.isZip64 ? "Yes" : "No") << endl;
	cout << "Complete:      " << (result.isComplete ? "Yes" : "No") << endl;

	if (!result.hasValidCD) {
		cout << endl;
		cout << "[WARNING] " << result.errorMessage << endl;
	}

	// 显示间隙信息
	if (!result.gaps.empty()) {
		cout << endl;
		cout << "--- Detected Gaps ---" << endl;
		for (size_t i = 0; i < result.gaps.size(); i++) {
			const auto& gap = result.gaps[i];
			cout << "Gap " << (i + 1) << ": offset " << gap.start
			     << " - " << gap.end << " (" << gap.size() << " bytes)" << endl;
		}
	}

	// 显示恢复建议
	ZipParser::ZipRecoveryAdvice advice = ZipParser::ZipStructureParser::GetRecoveryAdvice(result);
	cout << endl;
	cout << "--- Recovery Status ---" << endl;
	cout << "Status: ";
	switch (advice.status) {
		case ZipParser::ZipRecoveryAdvice::Status::COMPLETE:
			cout << "COMPLETE (no repair needed)" << endl;
			break;
		case ZipParser::ZipRecoveryAdvice::Status::REPAIRABLE:
			cout << "REPAIRABLE" << endl;
			break;
		case ZipParser::ZipRecoveryAdvice::Status::PARTIAL_RECOVERY:
			cout << "PARTIAL RECOVERY possible" << endl;
			break;
		case ZipParser::ZipRecoveryAdvice::Status::UNRECOVERABLE:
			cout << "UNRECOVERABLE" << endl;
			break;
	}
	cout << "Description: " << advice.description << endl;

	if (!advice.steps.empty()) {
		cout << "Suggestions:" << endl;
		for (const auto& step : advice.steps) {
			cout << "  - " << step << endl;
		}
	}

	// 列出文件
	if (listFiles || verbose) {
		cout << endl;
		cout << "--- File List ---" << endl;
		cout << setw(8) << "Index" << " | "
		     << setw(12) << "CompSize" << " | "
		     << setw(12) << "OrigSize" << " | "
		     << setw(10) << "CRC32" << " | "
		     << setw(8) << "Method" << " | "
		     << "Name" << endl;
		cout << string(80, '-') << endl;

		for (size_t i = 0; i < result.entries.size(); i++) {
			const auto& entry = result.entries[i];

			cout << setw(8) << i << " | "
			     << setw(12) << entry.compressedSize << " | "
			     << setw(12) << entry.uncompressedSize << " | "
			     << hex << setw(10) << entry.crc32 << dec << " | "
			     << setw(8) << ZipParser::ZipStructureParser::GetCompressionMethodName(entry.compression) << " | "
			     << entry.filename;

			if (verbose) {
				cout << " [offset=" << entry.localHeaderOffset;
				if (!entry.hasValidLocalHeader) cout << " INVALID_HDR";
				if (!entry.hasValidData) cout << " NO_DATA";
				cout << "]";
			}
			cout << endl;
		}

		cout << string(80, '-') << endl;
		cout << "Total: " << result.entries.size() << " files" << endl;

		// 计算总大小
		uint64_t totalCompressed = 0, totalUncompressed = 0;
		for (const auto& entry : result.entries) {
			totalCompressed += entry.compressedSize;
			totalUncompressed += entry.uncompressedSize;
		}
		cout << "Compressed:   " << totalCompressed << " bytes" << endl;
		cout << "Uncompressed: " << totalUncompressed << " bytes" << endl;
		if (totalUncompressed > 0) {
			double ratio = (1.0 - (double)totalCompressed / totalUncompressed) * 100;
			cout << "Ratio:        " << fixed << setprecision(1) << ratio << "%" << endl;
		}
	}

	// CRC验证
	if (verifyCRC) {
		cout << endl;
		cout << "--- CRC32 Verification ---" << endl;

		// 打开文件（按需读取每个条目）
		HANDLE hFile = CreateFileW(wFilePath.c_str(), GENERIC_READ, FILE_SHARE_READ,
		                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
		if (hFile != INVALID_HANDLE_VALUE) {
			int verified = 0, failed = 0, skipped = 0;

			for (size_t i = 0; i < result.entries.size(); i++) {
				const auto& entry = result.entries[i];

				// 只验证存储(未压缩)的文件
				if (entry.compression != 0) {
					skipped++;
					continue;
				}

				// 限制验证的文件大小（避免读取超大文件）
				if (entry.compressedSize > 100 * 1024 * 1024) {  // 100MB
					skipped++;
					if (verbose) {
						cout << "  [SKIP] " << entry.filename << " (too large)" << endl;
					}
					continue;
				}

				// 读取Local Header获取数据偏移
				LARGE_INTEGER seekPos;
				seekPos.QuadPart = entry.localHeaderOffset;
				SetFilePointerEx(hFile, seekPos, NULL, FILE_BEGIN);

				ZipParser::LocalFileHeader lh;
				DWORD bytesRead;
				if (!ReadFile(hFile, &lh, sizeof(lh), &bytesRead, NULL) ||
				    bytesRead != sizeof(lh) ||
				    lh.signature != ZipParser::LocalFileHeader::SIGNATURE) {
					failed++;
					cout << "  [FAIL] " << entry.filename << " (invalid local header)" << endl;
					continue;
				}

				// 跳过文件名和extra字段
				seekPos.QuadPart = entry.localHeaderOffset + ZipParser::LocalFileHeader::MIN_SIZE +
				                   lh.filenameLength + lh.extraLength;
				SetFilePointerEx(hFile, seekPos, NULL, FILE_BEGIN);

				// 读取文件数据
				vector<BYTE> fileData((size_t)entry.compressedSize);
				if (!ReadFile(hFile, fileData.data(), (DWORD)entry.compressedSize, &bytesRead, NULL) ||
				    bytesRead != entry.compressedSize) {
					failed++;
					cout << "  [FAIL] " << entry.filename << " (read error)" << endl;
					continue;
				}

				// 计算CRC32
				uint32_t crc = ZipParser::ZipStructureParser::CalculateCRC32(
				    fileData.data(), fileData.size());

				if (crc == entry.crc32) {
					verified++;
					if (verbose) {
						cout << "  [OK] " << entry.filename << endl;
					}
				} else {
					failed++;
					cout << "  [FAIL] " << entry.filename << " (CRC mismatch: "
					     << hex << crc << " != " << entry.crc32 << dec << ")" << endl;
				}
			}

			CloseHandle(hFile);

			cout << endl;
			cout << "Verified: " << verified << ", Failed: " << failed
			     << ", Skipped (compressed/large): " << skipped << endl;
		} else {
			cout << "[ERROR] Cannot open file for CRC verification" << endl;
		}
	}

	cout << endl;
	cout << "========================================" << endl;
}
