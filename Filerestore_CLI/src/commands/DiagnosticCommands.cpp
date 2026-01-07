// DiagnosticCommands.cpp - 系统诊断相关命令实现
// 包含: DiagnoseMFTCommand, DetectOverwriteCommand, ScanUsnCommand

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

	FileRestore* fileRestore = new FileRestore();
	OverwriteDetector* detector = fileRestore->GetOverwriteDetector();
	detector->SetDetectionMode(mode);

	OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNumber);

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

	delete fileRestore;
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
