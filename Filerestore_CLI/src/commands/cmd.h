#pragma once
#include <Windows.h>
#include <string>
#include <memory>
#include "climodule.h"
#include <queue>
#include <vector>
#include "ImageTable.h"
#include "CommandMacros.h"
using namespace std;

// ============================================================================
// 命令基类
// ============================================================================
class Command
{
protected:
	BOOL FlagHasArgs = false;

public:
	virtual void AcceptArgs(vector<LPVOID> argslist) = 0;
	virtual void Execute(string command) = 0;
	virtual BOOL HasArgs() = 0;
};

// ============================================================================
// 系统命令
// ============================================================================

// 打印所有命令 (无参数)
DECLARE_COMMAND_NOARGS(PrintAllCommand);

// 帮助命令
DECLARE_COMMAND(HelpCommand);

// 退出命令 (无参数)
DECLARE_COMMAND_NOARGS(ExitCommand);

// 设置语言命令
DECLARE_COMMAND(SetLanguageCommand);

// ============================================================================
// PE 分析命令
// ============================================================================

// 列出 DLL 依赖
DECLARE_COMMAND(QueueDLLsCommand);

// 获取函数名及函数名的地址
DECLARE_COMMAND(GetProcessFuncAddressCommand);

// 打印所有导入函数 - 特殊：包含额外成员变量
class PrintAllFunction : public Command
{
public:
	static string name;
	static vector<LPVOID> Arglist;
	unique_ptr<ImageTableAnalyzer> analyzer = make_unique<ImageTableAnalyzer>();
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

// ============================================================================
// 文件恢复命令
// ============================================================================

// 列出已删除文件
DECLARE_COMMAND(ListDeletedFilesCommand);

// 按记录号恢复文件
DECLARE_COMMAND(RestoreByRecordCommand);

// 强制恢复文件（跳过覆盖检测）
DECLARE_COMMAND(ForceRestoreCommand);

// 批量恢复文件
DECLARE_COMMAND(BatchRestoreCommand);

// ============================================================================
// 文件搜索命令
// ============================================================================

// 搜索已删除文件
DECLARE_COMMAND(SearchDeletedFilesCommand);

// 按文件大小过滤
DECLARE_COMMAND(FilterSizeCommand);

// 查找 MFT 记录号
DECLARE_COMMAND(FindRecordCommand);

// 查找用户文件
DECLARE_COMMAND(FindUserFilesCommand);

// 文件诊断
DECLARE_COMMAND(DiagnoseFileCommand);

// ZIP 结构解析
DECLARE_COMMAND(ZipInfoCommand);

// ============================================================================
// MFT/USN 诊断命令
// ============================================================================

// 诊断 MFT 碎片化
DECLARE_COMMAND(DiagnoseMFTCommand);

// 检测文件覆盖
DECLARE_COMMAND(DetectOverwriteCommand);

// 扫描 USN 日志
DECLARE_COMMAND(ScanUsnCommand);

// 搜索 USN 日志
DECLARE_COMMAND(SearchUsnCommand);

// ============================================================================
// 文件签名搜索命令（文件雕刻）
// ============================================================================

// 签名搜索扫描
DECLARE_COMMAND(CarveCommand);

// 列出支持的文件类型
DECLARE_COMMAND(CarveTypesCommand);

// 恢复雕刻文件
DECLARE_COMMAND(CarveRecoverCommand);

DECLARE_COMMAND(CarveCommandAsync);
DECLARE_COMMAND(CarveCommandThreadPool);
DECLARE_COMMAND(CarveTimestampCommand);
DECLARE_COMMAND(CarveListCommand);
DECLARE_COMMAND(CarveValidateCommand);
DECLARE_COMMAND(CarveIntegrityCommand);

// 分页交互恢复命令
DECLARE_COMMAND(CarveRecoverPageCommand);

// ============================================================================
// USN 定点恢复命令
// ============================================================================

// 列出 USN 删除记录（带验证）
DECLARE_COMMAND(UsnListCommand);

// USN 定点恢复
DECLARE_COMMAND(UsnRecoverCommand);

// 智能恢复向导（USN + 签名扫描联合）
DECLARE_COMMAND(RecoverCommand);

// ============================================================================
// 文件修复命令
// ============================================================================

// 修复单个损坏的文件
DECLARE_COMMAND(RepairCommand);

// 批量修复目录中的损坏文件
DECLARE_COMMAND(RepairBatchCommand);

// ============================================================================
// 机器学习数据集生成命令
// ============================================================================

// 扫描目录/卷生成机器学习训练数据集
DECLARE_COMMAND(MLScanCommand);
