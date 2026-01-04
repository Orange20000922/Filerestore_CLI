#pragma once
#include <Windows.h>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

// 多语言管理器 - 单例模式
class LocalizationManager
{
private:
    // 当前语言
    wstring currentLanguage;

    // 翻译字典：键 -> 翻译文本
    map<wstring, wstring> translations;

    // 支持的语言列表
    static const wstring SUPPORTED_LANGUAGES[];
    static const int LANGUAGE_COUNT;

    // 私有构造函数（单例模式）
    LocalizationManager();

    // 禁用拷贝构造和赋值
    LocalizationManager(const LocalizationManager&) = delete;
    LocalizationManager& operator=(const LocalizationManager&) = delete;

    // 解析JSON文件（简单实现，不依赖外部库）
    bool ParseLanguageFile(const wstring& languageCode);

    // 从文件读取所有内容
    wstring ReadFileContent(const wstring& filePath);

    // 简单的JSON解析（键值对提取）
    void ParseSimpleJson(const wstring& jsonContent);

public:
    // 获取单例实例
    static LocalizationManager& Instance();

    // 设置语言
    bool SetLanguage(const wstring& languageCode);

    // 获取当前语言
    wstring GetCurrentLanguage() const { return currentLanguage; }

    // 获取翻译文本
    wstring Get(const wstring& key) const;

    // 获取翻译文本（带默认值）
    wstring Get(const wstring& key, const wstring& defaultValue) const;

    // 检查语言是否支持
    bool IsLanguageSupported(const wstring& languageCode) const;

    // 获取支持的语言列表
    void GetSupportedLanguages(vector<wstring>& languages) const;

    // 重新加载当前语言文件
    bool Reload();
};

// 便捷宏：获取翻译文本
#define LOC(key) LocalizationManager::Instance().Get(L##key)
#define LOC_DEF(key, def) LocalizationManager::Instance().Get(L##key, L##def)
