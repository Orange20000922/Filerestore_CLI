// SearchCommands.cpp - 文件搜索相关命令实现
// 包含: SearchDeletedFilesCommand, DiagnoseFileCommand, SearchUsnCommand,
//       FilterSizeCommand, FindRecordCommand, FindUserFilesCommand

#include "cmd.h"
#include "CommandUtils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <map>
#include <algorithm>
#include "FileRestore.h"
#include "DeletedFileScanner.h"
#include "MFTReader.h"
#include "MFTParser.h"
#include "PathResolver.h"
#include "UsnJournalReader.h"

using namespace std;

// 辅助函数：获取或扫描已删除文件（使用缓存）
static vector<DeletedFileInfo> GetOrScanDeletedFiles(char driveLetter, FilterLevel filterLevel = FILTER_SKIP_PATH) {
	vector<DeletedFileInfo> allFiles;

	// 尝试从缓存加载
	if (DeletedFileScanner::IsCacheValid(driveLetter, 60)) {
		cout << "Loading from cache..." << endl;
		DeletedFileScanner::LoadFromCache(allFiles, driveLetter);
	}

	// 缓存无效或为空，重新扫描
	if (allFiles.empty()) {
		cout << "Scanning MFT..." << endl;
		FileRestore fileRestore;
		fileRestore.SetFilterLevel(filterLevel);
		allFiles = fileRestore.ScanDeletedFiles(driveLetter, 0);

		if (!allFiles.empty()) {
			DeletedFileScanner::SaveToCache(allFiles, driveLetter);
		}
	}

	return allFiles;
}

// ============================================================================
// SearchDeletedFilesCommand - 搜索已删除文件
// ============================================================================
DEFINE_COMMAND_BASE(SearchDeletedFilesCommand, "searchdeleted |name |name |name |name", TRUE)
REGISTER_COMMAND(SearchDeletedFilesCommand);

void SearchDeletedFilesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Invalid Args! Usage: searchdeleted <drive_letter> <filename_pattern> [extension] [filter_level]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& pattern = GET_ARG_STRING(1);
		string extension = HAS_ARG(2) ? GET_ARG_STRING(2) : "";

		FilterLevel filterLevel = FILTER_SKIP_PATH;
		if (HAS_ARG(3)) {
			string& filterStr = GET_ARG_STRING(3);
			if (filterStr == "none") filterLevel = FILTER_NONE;
			else if (filterStr == "exclude") filterLevel = FILTER_EXCLUDE;
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		cout << "Searching drive " << driveLetter << ": for deleted files..." << endl;
		cout << "Pattern: " << pattern << endl;

		vector<DeletedFileInfo> allFiles = GetOrScanDeletedFiles(driveLetter, filterLevel);

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		vector<DeletedFileInfo> filtered = allFiles;

		if (!extension.empty() && extension != "*") {
			wstring wext(extension.begin(), extension.end());
			filtered = DeletedFileScanner::FilterByExtension(filtered, wext);
		}

		if (pattern != "*") {
			wstring wpattern(pattern.begin(), pattern.end());
			filtered = DeletedFileScanner::FilterByName(filtered, wpattern);
		}

		cout << "\n===== Search Results =====" << endl;
		cout << "Found: " << filtered.size() << " matching files." << endl;

		size_t displayLimit = min(filtered.size(), (size_t)100);
		for (size_t i = 0; i < displayLimit; i++) {
			const auto& info = filtered[i];
			cout << "[" << info.recordNumber << "] ";
			cout << info.fileSize << " bytes | ";
			wcout << info.filePath << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// DiagnoseFileCommand - 文件诊断
// ============================================================================
DEFINE_COMMAND_BASE(DiagnoseFileCommand, "diagnosefile |name |name", TRUE)
REGISTER_COMMAND(DiagnoseFileCommand);

void DiagnoseFileCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() != 2) {
		cout << "Invalid Args! Usage: diagnosefile <drive_letter> <filename>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& fileNameStr = GET_ARG_STRING(1);

		if (driveStr.empty() || fileNameStr.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		wstring searchName(fileNameStr.begin(), fileNameStr.end());

		cout << "\n========== File Diagnostic Tool ==========" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		wcout << L"Searching for: " << searchName << endl;

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "ERROR: Failed to open volume " << driveLetter << ":" << endl;
			return;
		}

		MFTParser parser(&reader);
		PathResolver pathResolver(&reader, &parser);

		ULONGLONG totalRecords = reader.GetTotalMFTRecords();
		cout << "Total MFT records: " << totalRecords << endl;

		vector<BYTE> record;
		ULONGLONG foundCount = 0;
		ULONGLONG scannedCount = 0;

		wstring searchNameLower = searchName;
		transform(searchNameLower.begin(), searchNameLower.end(), searchNameLower.begin(), ::towlower);

		cout << "Attempting to load path cache..." << endl;
		if (pathResolver.LoadCache(driveLetter)) {
			cout << "Path cache loaded (" << pathResolver.GetCacheSize() << " entries)" << endl;

			auto& cache = pathResolver.GetCacheRef();

			for (const auto& entry : cache) {
				ULONGLONG recordNum = entry.first;
				const wstring& fullPath = entry.second;

				size_t lastSlash = fullPath.find_last_of(L"\\/");
				wstring fileName = (lastSlash != wstring::npos) ?
					fullPath.substr(lastSlash + 1) : fullPath;

				wstring fileNameLower = fileName;
				transform(fileNameLower.begin(), fileNameLower.end(), fileNameLower.begin(), ::towlower);

				if (fileNameLower.find(searchNameLower) != wstring::npos) {
					if (!reader.ReadMFT(recordNum, record)) continue;

					FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
					bool isDeleted = ((header->Flags & 0x01) == 0);
					bool isDirectory = ((header->Flags & 0x02) != 0);

					foundCount++;

					cout << "\n[" << foundCount << "] MFT Record #" << recordNum << endl;
					wcout << L"  Name: " << fileName << endl;
					cout << "  Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
					cout << "  Type: " << (isDirectory ? "Directory" : "File") << endl;
					wcout << L"  Full Path: " << fullPath << endl;

					if (foundCount >= 50) break;
				}
				scannedCount++;
			}
		}
		else {
			cout << "Cache not available, performing MFT scan..." << endl;
		}

		cout << "\n========== Scan Results ==========" << endl;
		cout << "Total matches found: " << foundCount << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// SearchUsnCommand - USN 搜索已删除文件
// ============================================================================
DEFINE_COMMAND_BASE(SearchUsnCommand, "searchusn |name |name |name", TRUE)
REGISTER_COMMAND(SearchUsnCommand);

void SearchUsnCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Usage: searchusn <drive_letter> <filename> [exact]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& filenameStr = GET_ARG_STRING(1);

		bool exactMatch = false;
		if (HAS_ARG(2)) {
			string& matchMode = GET_ARG_STRING(2);
			exactMatch = (matchMode == "exact" || matchMode == "e");
		}

		if (driveStr.empty() || filenameStr.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		wstring searchName(filenameStr.begin(), filenameStr.end());

		cout << "=== USN Journal Search ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		wcout << L"Searching for: " << searchName << endl;

		UsnJournalReader reader;
		if (!reader.Open(driveLetter)) {
			cout << "Failed to open USN Journal" << endl;
			return;
		}

		auto results = reader.SearchDeletedByName(searchName, exactMatch);

		if (results.empty()) {
			cout << "\nNo deleted files found matching '" << filenameStr << "'" << endl;
			return;
		}

		cout << "\n=== Found " << results.size() << " deleted file(s) ===" << endl;

		for (size_t i = 0; i < results.size(); i++) {
			const auto& info = results[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << info.GetMftRecordNumber() << endl;
			wcout << L"  Name: " << info.FileName << endl;
			cout << "  Parent Record: " << info.GetParentMftRecordNumber() << endl;

			FILETIME ft;
			ft.dwLowDateTime = info.TimeStamp.LowPart;
			ft.dwHighDateTime = info.TimeStamp.HighPart;
			SYSTEMTIME st;
			FileTimeToSystemTime(&ft, &st);
			cout << "  Deleted: " << st.wYear << "-"
				<< setfill('0') << setw(2) << st.wMonth << "-"
				<< setw(2) << st.wDay << " "
				<< setw(2) << st.wHour << ":"
				<< setw(2) << st.wMinute << ":"
				<< setw(2) << st.wSecond << endl;
		}

		cout << "\n=== Recovery Instructions ===" << endl;
		cout << "To restore: restorebyrecord " << driveLetter << " <record_number> <output_path>" << endl;
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// FilterSizeCommand - 按文件大小过滤
// ============================================================================
DEFINE_COMMAND_BASE(FilterSizeCommand, "filtersize |name |name |name |name", TRUE)
REGISTER_COMMAND(FilterSizeCommand);

void FilterSizeCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 3) {
		cout << "Usage: filtersize <drive_letter> <min_size> <max_size> [limit]" << endl;
		cout << "Size format: number + unit (B/K/M/G)" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& minSizeStr = GET_ARG_STRING(1);
		string& maxSizeStr = GET_ARG_STRING(2);

		size_t displayLimit = 100;
		if (HAS_ARG(3)) {
			string& limitStr = GET_ARG_STRING(3);
			displayLimit = stoull(limitStr);
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		// Parse file sizes using CommandUtils
		ULONGLONG minSize, maxSize;
		if (!CommandUtils::ParseFileSize(minSizeStr, minSize)) {
			cout << "Invalid minimum size format." << endl;
			return;
		}
		if (!CommandUtils::ParseFileSize(maxSizeStr, maxSize)) {
			cout << "Invalid maximum size format." << endl;
			return;
		}

		cout << "=== Filter Deleted Files by Size ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Size range: " << minSize << " - " << maxSize << " bytes" << endl;

		vector<DeletedFileInfo> allFiles = GetOrScanDeletedFiles(driveLetter);

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		auto filtered = DeletedFileScanner::FilterBySize(allFiles, minSize, maxSize);

		cout << "\n=== Found " << filtered.size() << " file(s) ===" << endl;

		size_t displayCount = min(filtered.size(), displayLimit);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& file = filtered[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << file.recordNumber << endl;
			wcout << L"  Name: " << file.fileName << endl;
			cout << "  Size: " << file.fileSize << " bytes" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// FindRecordCommand - 查找 MFT 记录号
// ============================================================================
DEFINE_COMMAND_BASE(FindRecordCommand, "findrecord |name |file", TRUE)
REGISTER_COMMAND(FindRecordCommand);

void FindRecordCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 2) {
		cout << "Usage: findrecord <drive_letter> <file_path>" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);
		string& filePath = GET_ARG_STRING(1);

		if (driveStr.empty() || filePath.empty()) {
			cout << "Invalid arguments." << endl;
			return;
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		cout << "=== Find MFT Record Number ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;
		cout << "Path: " << filePath << endl;

		MFTReader reader;
		if (!reader.OpenVolume(driveLetter)) {
			cout << "Failed to open volume " << driveLetter << ":/" << endl;
			return;
		}

		// 加载 MFT data runs（支持碎片化 MFT 的正确记录定位）
		reader.GetTotalMFTRecords();

		MFTParser parser(&reader);
		PathResolver resolver(&reader, &parser);

		cout << "Searching for file..." << endl;
		ULONGLONG recordNumber = resolver.FindFileRecordByPath(filePath);

		if (recordNumber == 0) {
			cout << "\n[NOT FOUND] File not found in MFT." << endl;
			return;
		}

		cout << "\n=== File Found ===" << endl;
		cout << "MFT Record Number: " << recordNumber << endl;

		vector<BYTE> record;
		if (reader.ReadMFT(recordNumber, record)) {
			FILE_RECORD_HEADER* header = (FILE_RECORD_HEADER*)record.data();
			bool isDeleted = ((header->Flags & 0x01) == 0);
			bool isDirectory = ((header->Flags & 0x02) != 0);

			cout << "Status: " << (isDeleted ? "DELETED" : "ACTIVE") << endl;
			cout << "Type: " << (isDirectory ? "Directory" : "File") << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}

// ============================================================================
// FindUserFilesCommand - 查找用户文件
// ============================================================================
DEFINE_COMMAND_BASE(FindUserFilesCommand, "finduserfiles |name |name", TRUE)
REGISTER_COMMAND(FindUserFilesCommand);

void FindUserFilesCommand::Execute(string command) {
	if (!CheckName(command)) {
		return;
	}

	if (GET_ARG_COUNT() < 1) {
		cout << "Usage: finduserfiles <drive_letter> [limit]" << endl;
		return;
	}

	try {
		string& driveStr = GET_ARG_STRING(0);

		size_t displayLimit = 100;
		if (HAS_ARG(1)) {
			string& limitStr = GET_ARG_STRING(1);
			displayLimit = stoull(limitStr);
		}

		char driveLetter;
		if (!CommandUtils::ValidateDriveLetter(driveStr, driveLetter)) {
			cout << "Invalid drive letter." << endl;
			return;
		}

		cout << "=== Find User Files ===" << endl;
		cout << "Drive: " << driveLetter << ":" << endl;

		vector<DeletedFileInfo> allFiles = GetOrScanDeletedFiles(driveLetter);

		if (allFiles.empty()) {
			cout << "No deleted files found." << endl;
			return;
		}

		auto userFiles = DeletedFileScanner::FilterUserFiles(allFiles);

		if (userFiles.empty()) {
			cout << "\nNo user files found." << endl;
			return;
		}

		cout << "\n=== Found " << userFiles.size() << " user file(s) ===" << endl;

		map<wstring, vector<DeletedFileInfo>> filesByType;
		for (const auto& file : userFiles) {
			wstring ext = DeletedFileScanner::GetFileExtension(file.fileName);
			transform(ext.begin(), ext.end(), ext.begin(), ::towlower);
			filesByType[ext].push_back(file);
		}

		cout << "\n=== File Type Summary ===" << endl;
		for (const auto& pair : filesByType) {
			wcout << L"  " << pair.first << L": " << pair.second.size() << L" files" << endl;
		}

		cout << "\n=== File List ===" << endl;
		size_t displayCount = min(userFiles.size(), displayLimit);
		for (size_t i = 0; i < displayCount; i++) {
			const auto& file = userFiles[i];
			cout << "\n[" << (i + 1) << "] MFT Record #" << file.recordNumber << endl;
			wcout << L"  Name: " << file.fileName << endl;
			cout << "  Size: " << file.fileSize << " bytes" << endl;
		}

		if (userFiles.size() > displayLimit) {
			cout << "\n(Showing first " << displayLimit << " of " << userFiles.size() << " files)" << endl;
		}
	}
	catch (const exception& e) {
		cout << "[ERROR] Exception: " << e.what() << endl;
	}
}
