// RestoreCommands.cpp - 文件恢复相关命令实现
// 包含: ListDeletedFilesCommand, RestoreByRecordCommand, ForceRestoreCommand, BatchRestoreCommand

#include "cmd.h"
#include "CommandUtils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <map>
#include "FileRestore.h"
#include "DeletedFileScanner.h"
#include "MFTReader.h"
#include "MFTBatchReader.h"
#include "MFTParser.h"
#include "ProgressBar.h"

using namespace std;

// ============================================================================
// ListDeletedFilesCommand - 列出已删除文件
// ============================================================================
DEFINE_COMMAND_BASE(ListDeletedFilesCommand, "listdeleted |name |name", TRUE)
REGISTER_COMMAND(ListDeletedFilesCommand);

void ListDeletedFilesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1 || GET_ARG_COUNT() > 2) {
		cout << "Invalid Args! Usage: listdeleted <drive_letter> [filter_level]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		char driveLetter;

		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		FilterLevel filterLevel = FILTER_SKIP_PATH;
		if (HAS_ARG(1)) {
			string& filterStr = GET_ARG_STRING(1);
			if (filterStr == "none") filterLevel = FILTER_NONE;
			else if (filterStr == "exclude") filterLevel = FILTER_EXCLUDE;
		}

		cout << "Scanning drive " << driveLetter << ": for deleted files..." << endl;

		FileRestore* fileRestore = new FileRestore();
		fileRestore->SetFilterLevel(filterLevel);
		vector<DeletedFileInfo> deletedFiles = fileRestore->ScanDeletedFiles(driveLetter, 0);
		delete fileRestore;

		if (deletedFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		DeletedFileScanner::SaveToCache(deletedFiles, driveLetter);

		cout << "\n===== Deleted Files on " << driveLetter << ": =====" << endl;
		cout << "Found: " << deletedFiles.size() << " deleted files." << endl;

		size_t displayLimit = min(deletedFiles.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = deletedFiles[i];
			cout << "[" << info.recordNumber << "] ";
			cout << info.fileSize << " bytes | ";
			wcout << info.filePath << endl;
		}

		if (deletedFiles.size() > 100) {
			cout << "\nNote: Showing first 100 of " << deletedFiles.size() << " files." << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// RestoreByRecordCommand - 按记录号恢复文件
// ============================================================================
DEFINE_COMMAND_BASE(RestoreByRecordCommand, "restorebyrecord |name |name |file", TRUE)
REGISTER_COMMAND(RestoreByRecordCommand);

void RestoreByRecordCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 3) {
		cout << "Invalid Args! Usage: restorebyrecord <drive_letter> <MFT_record_number> <output_path>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& recordStr = GET_ARG_STRING(1);
		string& outputPath = GET_ARG_STRING(2);

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

		cout << "=== File Recovery ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "MFT Record: " << recordNumber << endl;
		cout << "Output Path: " << outputPath << endl;

		FileRestore* fileRestore = new FileRestore();
		OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNumber);

		if (!result.isFullyAvailable && !result.isPartiallyAvailable) {
			cout << "\n[FAILED] File data has been completely overwritten." << endl;
			delete fileRestore;
			return;
		}

		bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNumber, outputPath);

		if (success) {
			cout << "\n=== Recovery Successful ===" << endl;
			cout << "File has been saved to: " << outputPath << endl;
		}
		else {
			cout << "\n=== Recovery Failed ===" << endl;
		}

		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// ForceRestoreCommand - 强制恢复文件（跳过覆盖检测）
// ============================================================================
DEFINE_COMMAND_BASE(ForceRestoreCommand, "forcerestore |name |name |file", TRUE)
REGISTER_COMMAND(ForceRestoreCommand);

void ForceRestoreCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 3) {
		cout << "Invalid Args! Usage: forcerestore <drive_letter> <MFT_record_number> <output_path>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& recordStr = GET_ARG_STRING(1);
		string& outputPath = GET_ARG_STRING(2);

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

		cout << "=== FORCE File Recovery (Overwrite Detection Bypassed) ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "MFT Record: " << recordNumber << endl;
		cout << "Output Path: " << outputPath << endl;
		cout << endl;

		cout << "WARNING: This command bypasses overwrite detection!" << endl;
		cout << "         - May recover corrupted or partial data" << endl;
		cout << "         - May recover random data if clusters were reused" << endl;
		cout << "         - Useful for SSD TRIM, high-entropy files, or false positives" << endl;
		cout << endl;

		FileRestore* fileRestore = new FileRestore();

		// 直接尝试恢复，不检测覆盖
		cout << "Attempting forced recovery..." << endl;
		bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNumber, outputPath);

		if (success) {
			cout << "\n=== Recovery Completed ===" << endl;
			cout << "File has been saved to: " << outputPath << endl;
			cout << "\nIMPORTANT: Please verify the recovered file!" << endl;
			cout << "           - Check file size" << endl;
			cout << "           - Open with appropriate application" << endl;
			cout << "           - Compare with known good file if possible" << endl;
		}
		else {
			cout << "\n=== Recovery Failed ===" << endl;
			cout << "Possible reasons:" << endl;
			cout << "  - MFT record is completely invalid" << endl;
			cout << "  - No DATA attribute found" << endl;
			cout << "  - Cannot read cluster data" << endl;
		}

		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// BatchRestoreCommand - 批量恢复文件
// ============================================================================
DEFINE_COMMAND_BASE(BatchRestoreCommand, "batchrestore |name |name |file", TRUE)
REGISTER_COMMAND(BatchRestoreCommand);

void BatchRestoreCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: batchrestore <drive_letter> <record_numbers> <output_directory>" << endl;
		cout << "Record numbers format: comma-separated list" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& recordsStr = GET_ARG_STRING(1);
		string& outputDir = GET_ARG_STRING(2);

		if (driveStr.empty() || recordsStr.empty() || outputDir.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		// Parse record numbers using CommandUtils
		vector<ULONGLONG> recordNumbers;
		if (!CommandUtils::ParseRecordNumbers(recordsStr, recordNumbers)) {
			cout << "No valid record numbers provided." << endl;
			return;
		}

		// Create output directory
		if (!CommandUtils::CreateOutputDirectory(outputDir)) {
			cout << "Failed to create output directory." << endl;
			return;
		}

		cout << "=== Batch File Recovery ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Output Directory: " << outputDir << endl;
		cout << "Files to restore: " << recordNumbers.size() << endl;

		FileRestore* fileRestore = new FileRestore();

		if (!fileRestore->OpenDrive(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":/" << endl;
			delete fileRestore;
			return;
		}

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open MFT reader" << endl;
			delete fileRestore;
			return;
		}

		MFTBatchReader batchReader;
		if (!batchReader.Initialize(&reader)) {
			cout << "Failed to initialize batch reader." << endl;
			delete fileRestore;
			return;
		}

		MFTParser parser(&reader);

		cout << "Pre-loading MFT records..." << endl;
		map<ULONGLONG, vector<BYTE>> preloadedRecords;
		for (ULONGLONG recordNum : recordNumbers) {
			vector<BYTE> record;
			if (batchReader.ReadMFTRecord(recordNum, record)) {
				preloadedRecords[recordNum] = record;
			}
		}
		cout << "Pre-loaded " << preloadedRecords.size() << " records." << endl;

		size_t successCount = 0;
		size_t failCount = 0;
		size_t skipCount = 0;

		ProgressBar progress(recordNumbers.size(), 40);
		progress.Show();

		for (size_t i = 0; i < recordNumbers.size(); i++) {
			ULONGLONG recordNum = recordNumbers[i];

			progress.Update(i + 1, successCount);

			auto it = preloadedRecords.find(recordNum);
			if (it == preloadedRecords.end() || it->second.empty()) {
				failCount++;
				continue;
			}

			vector<BYTE>& record = it->second;
			ULONGLONG parentDir;
			wstring fileName = parser.GetFileNameFromRecord(record, parentDir);

			if (fileName.empty()) {
				fileName = L"file_" + to_wstring(recordNum);
			}

			string fileNameStr(fileName.begin(), fileName.end());
			string outputPath = outputDir;
			if (outputPath.back() != '\\' && outputPath.back() != '/') {
				outputPath += '\\';
			}
			outputPath += fileNameStr;

			DWORD fileAttr = GetFileAttributesA(outputPath.c_str());
			if (fileAttr != INVALID_FILE_ATTRIBUTES) {
				size_t dotPos = outputPath.find_last_of('.');
				string baseName = (dotPos != string::npos) ? outputPath.substr(0, dotPos) : outputPath;
				string extension = (dotPos != string::npos) ? outputPath.substr(dotPos) : "";

				int suffix = 1;
				do {
					outputPath = baseName + "_" + to_string(suffix) + extension;
					suffix++;
					fileAttr = GetFileAttributesA(outputPath.c_str());
				} while (fileAttr != INVALID_FILE_ATTRIBUTES && suffix < 1000);
			}

			OverwriteDetectionResult result = fileRestore->DetectFileOverwrite(driveLetter, recordNum);

			if (!result.isFullyAvailable && !result.isPartiallyAvailable) {
				skipCount++;
				continue;
			}

			bool success = fileRestore->RestoreFileByRecordNumber(driveLetter, recordNum, outputPath);

			if (success) {
				successCount++;
			}
			else {
				failCount++;
			}
		}

		progress.Finish();

		cout << "\n=== Batch Recovery Summary ===" << endl;
		cout << "Total files: " << recordNumbers.size() << endl;
		cout << "Successfully restored: " << successCount << endl;
		cout << "Failed to restore: " << failCount << endl;
		cout << "Skipped (overwritten): " << skipCount << endl;
		cout << "\nOutput directory: " << outputDir << endl;

		batchReader.ClearCache();
		delete fileRestore;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}
