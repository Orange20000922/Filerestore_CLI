#include "CommandHelper.h"
#include "CommandMacros.h"
#include <algorithm>

std::map<std::string, CommandHelper::CommandMetadata> CommandHelper::commandMetadata_;
bool CommandHelper::initialized_ = false;

void CommandHelper::InitializeMetadata() {
    if (initialized_) return;

    // ======== 系统命令 ========
    commandMetadata_["help"] = {"help", "Show help information", "help [command]", {
        {"command", "Command", false, "", {}},
    }};
    commandMetadata_["exit"] = {"exit", "Exit program", "exit", {}};
    commandMetadata_["setlang"] = {"setlang", "Set language", "setlang <zh|en>", {
        {"lang", "Language", true, "en", {"zh", "en"}},
    }};

    // ======== 文件恢复命令 ========
    commandMetadata_["restorebyrecord"] = {"restorebyrecord", "Restore file by MFT record number",
        "restorebyrecord <drive> <record_number> [output_dir]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"record", "Record Number", true, "", {}},
        {"output", "Output Directory", false, "", {}},
    }};
    commandMetadata_["forcerestore"] = {"forcerestore", "Force restore (skip overwrite check)",
        "forcerestore <drive> <record_number> [output_dir]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"record", "Record Number", true, "", {}},
        {"output", "Output Directory", false, "", {}},
    }};
    commandMetadata_["batchrestore"] = {"batchrestore", "Batch restore deleted files",
        "batchrestore <drive> <record_list_file> [output_dir]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"list_file", "Record List File", true, "", {}},
        {"output", "Output Directory", false, "", {}},
    }};

    // ======== 文件搜索命令 ========
    commandMetadata_["listdeleted"] = {"listdeleted", "List deleted files on drive",
        "listdeleted <drive> [option]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"option", "Filter Option", false, "", {"all", "recent", "large"}},
    }};
    commandMetadata_["searchdeleted"] = {"searchdeleted", "Search deleted files by name",
        "searchdeleted <drive> <pattern>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"pattern", "Search Pattern", true, "", {}},
    }};
    commandMetadata_["searchusn"] = {"searchusn", "Search recently deleted via USN journal",
        "searchusn <drive> [filename]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"filename", "File Name", false, "", {}},
    }};
    commandMetadata_["filtersize"] = {"filtersize", "Filter by file size range",
        "filtersize <min_size> <max_size>", {
        {"min_size", "Min Size (bytes)", true, "", {}},
        {"max_size", "Max Size (bytes)", true, "", {}},
    }};
    commandMetadata_["finduserfiles"] = {"finduserfiles", "Find user files (docs/images/videos)",
        "finduserfiles <drive>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
    }};
    commandMetadata_["findrecord"] = {"findrecord", "Find MFT record by file path",
        "findrecord <drive> <path>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"path", "File Path", true, "", {}},
    }};
    commandMetadata_["diagnosefile"] = {"diagnosefile", "Diagnose and search for a file",
        "diagnosefile <drive> <filename>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"filename", "File Name", true, "", {}},
    }};

    // ======== 系统诊断命令 ========
    commandMetadata_["checkdrive"] = {"checkdrive", "Check drive status",
        "checkdrive <drive>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
    }};
    commandMetadata_["detectoverwrite"] = {"detectoverwrite", "Detect MFT record overwrite",
        "detectoverwrite <drive> <record_number>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"record", "Record Number", true, "", {}},
    }};
    commandMetadata_["analyzefragmentation"] = {"analyzefragmentation", "Analyze file fragmentation",
        "analyzefragmentation <drive> <record_number>", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"record", "Record Number", true, "", {}},
    }};
    commandMetadata_["dumpmft"] = {"dumpmft", "Dump MFT record",
        "dumpmft <drive> <record_number> [output_file]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"record", "Record Number", true, "", {}},
        {"output", "Output File", false, "", {}},
    }};

    // ======== 深度扫描命令 ========
    commandMetadata_["carve"] = {"carve", "Deep scan for specific file type",
        "carve <drive> <type> [output_dir]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"type", "File Type", true, "", {"jpg", "png", "pdf", "zip", "gif", "bmp", "mp3", "mp4", "avi", "7z", "rar", "exe", "all"}},
        {"output", "Output Directory", false, "", {}},
    }};
    commandMetadata_["carvepool"] = {"carvepool", "Multi-threaded deep scan",
        "carvepool <drive> <type> <output_dir> [threads]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"type", "File Type", true, "", {"jpg", "png", "pdf", "zip", "gif", "bmp", "mp3", "mp4", "avi", "7z", "rar", "exe", "all"}},
        {"output", "Output Directory", true, "", {}},
        {"threads", "Threads", false, "8", {}},
    }};
    commandMetadata_["carvelist"] = {"carvelist", "List carved results", "carvelist", {}};
    commandMetadata_["carverestore"] = {"carverestore", "Restore carved file by index",
        "carverestore <index> [output_path]", {
        {"index", "Result Index", true, "", {}},
        {"output", "Output Path", false, "", {}},
    }};
    commandMetadata_["carvevalidate"] = {"carvevalidate", "Validate carved result integrity",
        "carvevalidate <index>", {
        {"index", "Result Index", true, "", {}},
    }};
    commandMetadata_["carvefilter"] = {"carvefilter", "Filter carved results",
        "carvefilter <type|size> <value>", {
        {"filter_type", "Filter Type", true, "", {"type", "size"}},
        {"value", "Filter Value", true, "", {}},
    }};

    // ======== USN 恢复命令 ========
    commandMetadata_["usnrecover"] = {"usnrecover", "USN journal targeted recovery",
        "usnrecover <drive> <index|filename|record> <output_dir> [--force]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"target", "Index/Filename/Record", true, "", {}},
        {"output", "Output Directory", true, "", {}},
    }};

    // ======== 智能恢复命令（USN + 签名联合）========
    commandMetadata_["recover"] = {"recover", "Smart recovery (USN + Signature scan)",
        "recover <drive> [filename] [output_dir]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"filename", "Target Filename", false, "", {}},
        {"output", "Output Directory", false, "", {}},
    }};

    // ======== 机器学习命令 ========
    commandMetadata_["mlpredict"] = {"mlpredict", "ML-predict file integrity",
        "mlpredict <file_path>", {
        {"file", "File Path", true, "", {}},
    }};
    commandMetadata_["mlgendata"] = {"mlgendata", "Generate ML training dataset",
        "mlgendata <drive> [output_dir]", {
        {"drive", "Drive", true, "", {"C:", "D:", "E:", "F:"}},
        {"output", "Output Directory", false, "", {}},
    }};

    // ======== 文件修复命令 ========
    commandMetadata_["repair"] = {"repair", "Repair corrupted file",
        "repair <input_file> [output_file]", {
        {"input", "Input File", true, "", {}},
        {"output", "Output File", false, "", {}},
    }};

    initialized_ = true;
}

std::vector<std::string> CommandHelper::GetAllCommandNames() {
    InitializeMetadata();
    std::vector<std::string> names;
    for (const auto& pair : commandMetadata_) {
        names.push_back(pair.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::string CommandHelper::GetDescription(const std::string& commandName) {
    InitializeMetadata();
    auto it = commandMetadata_.find(commandName);
    if (it != commandMetadata_.end()) return it->second.description;
    return "";
}

std::string CommandHelper::GetUsage(const std::string& commandName) {
    InitializeMetadata();
    auto it = commandMetadata_.find(commandName);
    if (it != commandMetadata_.end()) return it->second.usage;
    return commandName;
}

std::vector<CommandHelper::ParamInfo> CommandHelper::GetParams(const std::string& commandName) {
    InitializeMetadata();
    auto it = commandMetadata_.find(commandName);
    if (it != commandMetadata_.end()) return it->second.params;
    return {};
}

std::vector<std::string> CommandHelper::MatchCommands(const std::string& prefix) {
    InitializeMetadata();
    if (prefix.empty()) return GetAllCommandNames();

    std::vector<std::string> matches;
    std::string lowerPrefix = prefix;
    std::transform(lowerPrefix.begin(), lowerPrefix.end(), lowerPrefix.begin(), ::tolower);

    for (const auto& pair : commandMetadata_) {
        std::string lowerName = pair.first;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
        if (lowerName.find(lowerPrefix) == 0) {
            matches.push_back(pair.first);
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

std::string CommandHelper::AssembleCommand(const std::string& cmdName,
                                            const std::string paramValues[],
                                            size_t paramCount) {
    std::string result = cmdName;
    for (size_t i = 0; i < paramCount; i++) {
        if (!paramValues[i].empty()) {
            result += " " + paramValues[i];
        }
    }
    return result;
}
