#include "LocalizationManager.h"
#include <sstream>
#include <algorithm>
#include <vector>
#include <nlohmann/json.hpp>
#include "Logger.h"
using namespace std;

// 支持的语言列表
const wstring LocalizationManager::SUPPORTED_LANGUAGES[] = { L"en", L"zh" };
const int LocalizationManager::LANGUAGE_COUNT = 2;
static string WideToUtf8(const wstring& wide) {
    if (wide.empty()) return "";

    int len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, NULL, 0, NULL, NULL);
    if (len <= 0) return "";

    string utf8(len - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &utf8[0], len, NULL, NULL);
    return utf8;
}

static wstring Utf8ToWide(const string& utf8) {
    if (utf8.empty()) return L"";

    int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, NULL, 0);
    if (len <= 0) return L"";

    wstring wide(len - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], len);
    return wide;
}
LocalizationManager::LocalizationManager() : currentLanguage(L"en") {
    // 默认加载英文
    SetLanguage(L"en");
}

LocalizationManager& LocalizationManager::Instance() {
    static LocalizationManager instance;
    return instance;
}

bool LocalizationManager::SetLanguage(const wstring& languageCode) {
    if (!IsLanguageSupported(languageCode)) {
        wcout << L"Unsupported language: " << languageCode << endl;
        return false;
    }

    currentLanguage = languageCode;
    return ParseLanguageFile(languageCode);
}

wstring LocalizationManager::Get(const wstring& key) const {
    auto it = translations.find(key);
    if (it != translations.end()) {
        return it->second;
    }
    // 如果找不到翻译，返回键本身（开发时方便调试）
    string logmessage = "[" + WideToUtf8(key) + "]";
    LOG_WARNING_FMT("Missing translation for key: %ls", logmessage);
    return L"[" + key + L"]";
}

wstring LocalizationManager::Get(const wstring& key, const wstring& defaultValue) const {
    auto it = translations.find(key);
    if (it != translations.end()) {
        return it->second;
    }
    return defaultValue;
}

// wstring 转 UTF-8 string 辅助函数


string LocalizationManager::GetUtf8(const wstring& key) const {
    auto it = translations.find(key);
    if (it != translations.end()) {
        return WideToUtf8(it->second);
    }
    // 如果找不到翻译，返回键本身（方便调试）
    string logmessage = "[" + WideToUtf8(key) + "]";
	LOG_WARNING_FMT("Missing translation for key: %ls", logmessage);
    return "[" + WideToUtf8(key) + "]";
}

string LocalizationManager::GetUtf8(const wstring& key, const string& defaultValue) const {
    auto it = translations.find(key);
    if (it != translations.end()) {
        return WideToUtf8(it->second);
    }
    return defaultValue;
}

bool LocalizationManager::IsLanguageSupported(const wstring& languageCode) const {
    for (int i = 0; i < LANGUAGE_COUNT; i++) {
        if (SUPPORTED_LANGUAGES[i] == languageCode) {
            return true;
        }
    }
    return false;
}

void LocalizationManager::GetSupportedLanguages(vector<wstring>& languages) const {
    languages.clear();
    for (int i = 0; i < LANGUAGE_COUNT; i++) {
        languages.push_back(SUPPORTED_LANGUAGES[i]);
    }
}

bool LocalizationManager::Reload() {
    return ParseLanguageFile(currentLanguage);
}

bool LocalizationManager::ParseLanguageFile(const wstring& languageCode) {
    // 构建语言文件路径
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(NULL, exePath, MAX_PATH);
    wstring exeDir = exePath;
    size_t lastSlash = exeDir.find_last_of(L"\\/");
    if (lastSlash != wstring::npos) {
        exeDir = exeDir.substr(0, lastSlash);
    }

    wstring langFilePath = exeDir + L"\\langs\\" + languageCode + L".json";

    // 以 UTF-8 读取文件
    ifstream file(langFilePath, ios::binary);
    if (!file.is_open()) {
        LOG_ERROR_FMT("Cannot open language file: %ls", langFilePath.c_str());
        wcout << L"Failed to load language file: " << langFilePath << endl;
        return false;
    }

    string utf8Content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    // 跳过 UTF-8 BOM (EF BB BF)
    if (utf8Content.size() >= 3 &&
        (unsigned char)utf8Content[0] == 0xEF &&
        (unsigned char)utf8Content[1] == 0xBB &&
        (unsigned char)utf8Content[2] == 0xBF) {
        utf8Content = utf8Content.substr(3);
    }

    // 使用 nlohmann/json 解析
    try {
        auto j = nlohmann::json::parse(utf8Content);
        translations.clear();

        for (auto& [key, value] : j.items()) {
            if (value.is_string()) {
                translations[Utf8ToWide(key)] = Utf8ToWide(value.get<string>());
            }
        }
    } catch (const nlohmann::json::parse_error& e) {
        LOG_ERROR_FMT("JSON parse error in %ls: %s", langFilePath.c_str(), e.what());
        wcout << L"Failed to parse language file: " << langFilePath << endl;
        return false;
    }

    if (translations.empty()) {
        LOG_WARNING("No translations loaded from language file!");
    }

    wcout << L"Language loaded: " << languageCode << L" (" << translations.size() << L" translations)" << endl;
    return true;
}
