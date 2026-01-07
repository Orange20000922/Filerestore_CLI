#include "LocalizationManager.h"
#include <sstream>
#include <algorithm>
#include <vector>

using namespace std;

// 支持的语言列表
const wstring LocalizationManager::SUPPORTED_LANGUAGES[] = { L"en", L"zh" };
const int LocalizationManager::LANGUAGE_COUNT = 2;

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
static string WideToUtf8(const wstring& wide) {
    if (wide.empty()) return "";

    int len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, NULL, 0, NULL, NULL);
    if (len <= 0) return "";

    string utf8(len - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &utf8[0], len, NULL, NULL);
    return utf8;
}

string LocalizationManager::GetUtf8(const wstring& key) const {
    auto it = translations.find(key);
    if (it != translations.end()) {
        return WideToUtf8(it->second);
    }
    // 如果找不到翻译，返回键本身（方便调试）
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

wstring LocalizationManager::ReadFileContent(const wstring& filePath) {
    // 打开文件（UTF-8编码）
    ifstream file(filePath, ios::binary);
    if (!file.is_open()) {
        return L"";
    }

    // 读取所有内容
    stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    string content = buffer.str();

    // 转换为宽字符串（UTF-8 to wstring）
    int len = MultiByteToWideChar(CP_UTF8, 0, content.c_str(), -1, NULL, 0);
    if (len <= 0) {
        return L"";
    }

    wstring wideContent(len - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, content.c_str(), -1, &wideContent[0], len);

    return wideContent;
}

void LocalizationManager::ParseSimpleJson(const wstring& jsonContent) {
    // 简单的JSON解析：查找 "key": "value" 模式
    // 这是一个简化版本，足够处理简单的键值对JSON

    translations.clear();

    size_t pos = 0;
    while (pos < jsonContent.length()) {
        // 查找键的开始引号
        size_t keyStart = jsonContent.find(L'"', pos);
        if (keyStart == wstring::npos) break;
        keyStart++;

        // 查找键的结束引号
        size_t keyEnd = jsonContent.find(L'"', keyStart);
        if (keyEnd == wstring::npos) break;

        wstring key = jsonContent.substr(keyStart, keyEnd - keyStart);

        // 查找冒号
        size_t colon = jsonContent.find(L':', keyEnd);
        if (colon == wstring::npos) break;

        // 查找值的开始引号
        size_t valueStart = jsonContent.find(L'"', colon);
        if (valueStart == wstring::npos) break;
        valueStart++;

        // 查找值的结束引号（处理转义字符）
        size_t valueEnd = valueStart;
        while (valueEnd < jsonContent.length()) {
            valueEnd = jsonContent.find(L'"', valueEnd);
            if (valueEnd == wstring::npos) break;

            // 检查是否被转义
            if (valueEnd > 0 && jsonContent[valueEnd - 1] != L'\\') {
                break;
            }
            valueEnd++;
        }

        if (valueEnd == wstring::npos) break;

        wstring value = jsonContent.substr(valueStart, valueEnd - valueStart);

        // 处理转义字符
        size_t escapePos = 0;
        while ((escapePos = value.find(L"\\n", escapePos)) != wstring::npos) {
            value.replace(escapePos, 2, L"\n");
            escapePos++;
        }
        while ((escapePos = value.find(L"\\t", escapePos)) != wstring::npos) {
            value.replace(escapePos, 2, L"\t");
            escapePos++;
        }
        while ((escapePos = value.find(L"\\\"", escapePos)) != wstring::npos) {
            value.replace(escapePos, 2, L"\"");
            escapePos++;
        }

        // 存储键值对
        translations[key] = value;

        pos = valueEnd + 1;
    }
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

    // 读取文件内容
    wstring jsonContent = ReadFileContent(langFilePath);
    if (jsonContent.empty()) {
        wcout << L"Failed to load language file: " << langFilePath << endl;
        return false;
    }

    // 解析JSON
    ParseSimpleJson(jsonContent);

    wcout << L"Language loaded: " << languageCode << L" (" << translations.size() << L" translations)" << endl;
    return true;
}
