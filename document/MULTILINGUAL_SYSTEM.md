# 多语言支持系统文档

## 系统概述

本项目现已支持完整的多语言界面系统，使用UTF-8编码和宽字符（wstring）来防止中文等Unicode字符的乱码问题。

## 架构设计

### 1. 核心组件

- **LocalizationManager** - 多语言管理器（单例模式）
  - 负责加载、解析和管理翻译文本
  - 提供语言切换功能
  - 使用简单的JSON解析器（无外部依赖）

### 2. 支持的语言

- `en` - English (英文)
- `zh` - 中文

### 3. 文件结构

```
ConsoleApplication5/
├── LocalizationManager.h         # 多语言管理器头文件
├── LocalizationManager.cpp       # 多语言管理器实现
├── langs/                        # 语言文件目录
│   ├── en.json                   # 英文翻译
│   └── zh.json                   # 中文翻译
└── ...
```

## 使用方法

### 1. 查看当前语言和支持的语言

```bash
setlang
```

输出示例：
```
Change the interface language
Usage: setlang <language_code>

Current language: en

Supported languages:
  - en (English)
  - zh (中文)
```

### 2. 切换语言

```bash
# 切换到中文
setlang zh

# 切换到英文
setlang en
```

### 3. 在代码中使用翻译

```cpp
#include "LocalizationManager.h"

// 使用宏获取翻译文本
wcout << LOC("help.title") << endl;

// 使用带默认值的宏
wcout << LOC_DEF("unknown.key", "Default Text") << endl;

// 直接调用方法
wstring text = LocalizationManager::Instance().Get(L"help.title");
```

## JSON语言文件格式

语言文件使用UTF-8编码保存，格式如下：

```json
{
  "key1": "Translation text 1",
  "key2": "Translation text 2",
  "help.title": "File Recovery Tool - Help",
  "help.cmd.listdeleted": "List all deleted files"
}
```

### 命名规范

- 使用点号分隔层级：`section.subsection.key`
- 通用键使用 `common.` 前缀
- 命令相关使用命令名作为前缀：`listdeleted.`, `help.`
- 错误消息使用 `error.` 前缀

## 现有翻译键

### 通用键（Common）

- `common.parameters` - 参数：
- `common.examples` - 示例：
- `common.notes` - 注意：
- `common.description` - 描述：
- `common.syntax` - 语法：
- `common.command` - 命令：
- `common.required` - 必需
- `common.optional` - 可选

### 帮助系统（Help）

- `help.title` - 帮助标题
- `help.parameter_types` - 参数类型说明
- `help.section.recovery` - 文件恢复命令分类
- `help.section.system` - 系统命令分类
- `help.tip1` ~ `help.tip5` - 使用提示

### 扫描相关（Scan）

- `scan.starting` - 正在扫描驱动器
- `scan.completed` - 扫描完成
- `scan.filter_level` - 过滤级别
- `scan.saving_cache` - 保存缓存中
- `scan.cache_saved` - 缓存保存成功

### 错误消息（Error）

- `error.invalid_args` - 无效参数
- `error.invalid_drive` - 无效驱动器
- `error.no_files` - 未找到文件
- `error.admin_required` - 需要管理员权限

## 技术细节

### 1. UTF-8编码处理

```cpp
// 设置控制台输出为UTF-8
SetConsoleOutputCP(CP_UTF8);

// 使用wcout输出宽字符
wcout << L"中文测试" << endl;
```

### 2. 字符串转换

```cpp
// string 转 wstring（UTF-8 -> Unicode）
int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
wstring wideStr(len - 1, 0);
MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wideStr[0], len);
```

### 3. JSON解析

系统使用自定义的简单JSON解析器，支持：
- 键值对提取
- 转义字符处理（`\n`, `\t`, `\"`）
- 嵌套引号处理

**不支持**：
- 复杂嵌套对象
- 数组
- 注释

### 4. 缓存机制

- 翻译文本在加载后缓存在内存中
- 使用 `map<wstring, wstring>` 存储
- 支持运行时重新加载

## 添加新语言

1. 创建新的JSON文件：`langs/<language_code>.json`
2. 复制现有语言文件作为模板
3. 翻译所有键值对
4. 在 `LocalizationManager.h` 中添加语言代码到 `SUPPORTED_LANGUAGES` 数组
5. 更新 `LANGUAGE_COUNT` 常量

示例（添加日语）：

```cpp
// LocalizationManager.h
const wstring LocalizationManager::SUPPORTED_LANGUAGES[] = {
    L"en", L"zh", L"ja"  // 添加日语
};
const int LocalizationManager::LANGUAGE_COUNT = 3;  // 更新数量
```

## 最佳实践

1. **使用wstring和wcout**
   ```cpp
   // 正确
   wcout << LOC("help.title") << endl;

   // 错误（会乱码）
   cout << LOC("help.title") << endl;
   ```

2. **翻译键命名**
   - 使用有意义的键名
   - 保持层级清晰
   - 避免过长的键名

3. **默认值**
   - 为关键文本提供默认值
   - 使用 `LOC_DEF` 宏处理可能缺失的翻译

4. **一致性**
   - 所有语言文件保持相同的键集合
   - 使用相同的格式和术语

## 性能考虑

- JSON解析只在语言加载时执行一次
- 翻译查找使用 `map`，时间复杂度 O(log n)
- 内存占用：约 50KB（2000条翻译 × 50字符平均长度）

## 故障排除

### 1. 中文显示乱码

**原因**：未设置UTF-8控制台输出

**解决**：
```cpp
SetConsoleOutputCP(CP_UTF8);
```

### 2. 翻译不生效

**原因**：
- 语言文件路径错误
- JSON格式错误
- 键名拼写错误

**排查**：
1. 检查 `langs/` 目录是否在可执行文件同级目录
2. 验证JSON文件格式
3. 检查日志输出

### 3. 无法切换语言

**原因**：语言代码不在支持列表中

**解决**：
- 使用 `setlang` 不带参数查看支持的语言
- 确保语言代码拼写正确

## 未来扩展

可能的改进方向：
1. 支持动态语言插件
2. 使用第三方JSON库（如 nlohmann/json）
3. 添加语言文件验证工具
4. 支持区域设置（en_US, en_GB等）
5. 添加日期、时间、数字格式本地化
6. 实现翻译文本的热重载

## 示例代码

### 完整使用示例

```cpp
#include "LocalizationManager.h"

void ShowWelcomeMessage() {
    // 设置UTF-8输出
    SetConsoleOutputCP(CP_UTF8);

    // 获取翻译文本
    wcout << LOC("welcome.title") << endl;
    wcout << LOC("welcome.message") << endl;

    // 使用默认值
    wcout << LOC_DEF("optional.text", "Default message") << endl;

    // 直接调用
    LocalizationManager& loc = LocalizationManager::Instance();
    wstring currentLang = loc.GetCurrentLanguage();
    wcout << L"Current language: " << currentLang << endl;
}
```

## 贡献指南

添加新翻译时：
1. 在 `en.json` 中添加新键和英文翻译
2. 在所有其他语言文件中添加相同的键
3. 测试所有语言的显示效果
4. 更新本文档的翻译键列表
