#pragma once
#include <Windows.h>
#include <string>
#include "climodule.h"
#include <queue>
#include <vector>
#include "ImageTable.h"
using namespace std;
class  Command 
{
    protected:
		BOOL FlagHasArgs = false;
		
	public:
		virtual void AcceptArgs(vector<LPVOID> argslist)=0;
		virtual void Execute(string command) = 0;
		virtual BOOL HasArgs() = 0;
};
class PrintAllCommand : public Command
{
public:
	static string name;
public:
	PrintAllCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new PrintAllCommand();
	}
	static string GetName() {
		return name;
	}
};
class HelpCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	HelpCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new HelpCommand();
	}
	static string GetName() {
		return name;
	}
};
class QueueDLLsCommand : public Command
{
    public:
	static string name;
	static vector<LPVOID> ArgsList;
	public:
		QueueDLLsCommand();
		~QueueDLLsCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new QueueDLLsCommand();
	}
	static string GetName() {
		return name;
	}
};
class GetProcessFuncAddressCommand : public Command
{
    public:
	static string name;
	static vector<LPVOID> ArgsList;
	public:
	GetProcessFuncAddressCommand();
	~GetProcessFuncAddressCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new GetProcessFuncAddressCommand();
	}
	static string GetName() {
		return name;
	}
};
// IATHookDLLCommand removed - IAT hooking functionality not included in public version

class ExitCommand : public Command
{
   public:
	static string name;
	static vector<LPVOID> ArgsList;
   public:
		ExitCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new ExitCommand();
	}
	static string GetName() {
		return name;
	}
};
class PrintAllFunction : public Command{
    public:
	 static string name;
	 static vector<LPVOID> Arglist;
	 ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
   public:
	   PrintAllFunction();
	   ~PrintAllFunction();
	   void AcceptArgs(vector<LPVOID> argslist) override;
	   void Execute(string command) override;
	   BOOL HasArgs() override;
	   static BOOL CheckName(string input);
	   static LPVOID GetInstancePtr() {
		   return new PrintAllFunction();
	   }
	   static string GetName() {
		   return name;
	   }
};
// IATHookByNameCommand removed - IAT hooking functionality not included in public version
// IATHookByCreateProc removed - IAT hooking functionality not included in public version
// ElevateAdminPrivilegeCommand removed - privilege elevation functionality not included in public version
// ElevateSystemPrivilegeCommand removed - privilege elevation functionality not included in public version
class ListDeletedFilesCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	ListDeletedFilesCommand();
	~ListDeletedFilesCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new ListDeletedFilesCommand();
	}
	static string GetName() {
		return name;
	}
};

class RestoreByRecordCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	RestoreByRecordCommand();
	~RestoreByRecordCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new RestoreByRecordCommand();
	}
	static string GetName() {
		return name;
	}
};

class DiagnoseMFTCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	DiagnoseMFTCommand();
	~DiagnoseMFTCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new DiagnoseMFTCommand();
	}
	static string GetName() {
		return name;
	}
};

class DetectOverwriteCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	DetectOverwriteCommand();
	~DetectOverwriteCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new DetectOverwriteCommand();
	}
	static string GetName() {
		return name;
	}
};

class SearchDeletedFilesCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	SearchDeletedFilesCommand();
	~SearchDeletedFilesCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new SearchDeletedFilesCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== USN Journal 扫描命令 ====================
class ScanUsnCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	ScanUsnCommand();
	~ScanUsnCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new ScanUsnCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== 文件诊断命令 ====================
class DiagnoseFileCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	DiagnoseFileCommand();
	~DiagnoseFileCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new DiagnoseFileCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== USN搜索已删除文件命令 ====================
class SearchUsnCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	SearchUsnCommand();
	~SearchUsnCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new SearchUsnCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== 按文件大小过滤命令 ====================
class FilterSizeCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	FilterSizeCommand();
	~FilterSizeCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new FilterSizeCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== 查找文件记录号命令 ====================
class FindRecordCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	FindRecordCommand();
	~FindRecordCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new FindRecordCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== 查找用户文件命令 ====================
class FindUserFilesCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	FindUserFilesCommand();
	~FindUserFilesCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new FindUserFilesCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== 批量恢复文件命令 ====================
class BatchRestoreCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	BatchRestoreCommand();
	~BatchRestoreCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new BatchRestoreCommand();
	}
	static string GetName() {
		return name;
	}
};

// ==================== 设置语言命令 ====================
class SetLanguageCommand : public Command
{
public:
	static string name;
	static vector<LPVOID> ArgsList;
public:
	SetLanguageCommand();
	~SetLanguageCommand();
	void AcceptArgs(vector<LPVOID> argslist) override;
	void Execute(string command) override;
	BOOL HasArgs() override;
	static BOOL CheckName(string input);
	static LPVOID GetInstancePtr() {
		return new SetLanguageCommand();
	}
	static string GetName() {
		return name;
	}
};