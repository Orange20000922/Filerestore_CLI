# Security Policy

**简体中文** | [English](#english)

## 安全声明

Filerestore_CLI 是一个 NTFS 文件恢复工具，需要**管理员权限**进行原始磁盘读取。本工具的设计目标是**只读恢复**，不会修改磁盘上的任何数据。

## 支持的版本

| 版本 | 支持状态 |
|------|---------|
| v1.0.x | 当前支持 |
| < v1.0.0 | 不再支持 |

## 报告安全漏洞

如果你发现了安全漏洞，**请不要通过公开 Issue 报告**。

请通过以下方式联系：

1. 发送邮件至项目维护者（通过 GitHub 个人资料页获取联系方式）
2. 或通过 [GitHub Security Advisories](https://github.com/Orange20000922/Filerestore_CLI/security/advisories) 提交私密报告

请在报告中包含：
- 漏洞描述
- 复现步骤
- 潜在影响评估
- 修复建议（如果有）

## 安全设计原则

### 磁盘访问

- 所有磁盘操作均为**只读**（`GENERIC_READ` + `FILE_SHARE_READ | FILE_SHARE_WRITE`）
- 不调用任何写入磁盘扇区/簇的 API
- 恢复文件仅写入用户指定的输出路径

### 权限模型

- 需要管理员权限是因为 Windows 限制了对卷（`\\.\C:`）的直接访问
- 程序不会提升自身权限或修改系统安全设置
- 内核驱动桥接（实验性）需要单独加载签名驱动，默认禁用

### 数据处理

- 不联网、不上传任何用户数据
- 所有处理在本地完成
- ML 模型推理使用本地 ONNX Runtime，不依赖云端服务
- 缓存文件（MFT 快照、扫描结果）存储在本地磁盘，不加密

### 已知安全边界

- 本工具读取原始磁盘数据，恢复的文件可能包含恶意内容（如恢复的 .exe 可能是恶意软件）
- 用户应对恢复的文件进行安全扫描后再打开
- 扫描结果缓存文件包含 MFT 元数据（文件名、大小、时间戳），不包含文件内容

---

<a name="english"></a>

[简体中文](#security-policy) | **English**

# Security Policy

## Security Statement

Filerestore_CLI is an NTFS file recovery tool that requires **administrator privileges** for raw disk access. The tool is designed for **read-only recovery** and does not modify any data on disk.

## Supported Versions

| Version | Status |
|---------|--------|
| v1.0.x | Currently supported |
| < v1.0.0 | No longer supported |

## Reporting Vulnerabilities

If you discover a security vulnerability, **do not report it via a public Issue**.

Please use one of the following channels:

1. Email the project maintainer (contact info available on the GitHub profile page)
2. Or submit a private report via [GitHub Security Advisories](https://github.com/Orange20000922/Filerestore_CLI/security/advisories)

Please include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

## Security Design Principles

### Disk Access

- All disk operations are **read-only** (`GENERIC_READ` + `FILE_SHARE_READ | FILE_SHARE_WRITE`)
- No APIs that write to disk sectors/clusters are called
- Recovered files are only written to user-specified output paths

### Permission Model

- Administrator privileges are required because Windows restricts direct volume access (`\\.\C:`)
- The program does not escalate its own privileges or modify system security settings
- Kernel driver bridge (experimental) requires a separately loaded signed driver and is disabled by default

### Data Handling

- No network connections; no user data is uploaded
- All processing is done locally
- ML model inference uses local ONNX Runtime with no cloud dependency
- Cache files (MFT snapshots, scan results) are stored on local disk, unencrypted

### Known Security Boundaries

- This tool reads raw disk data; recovered files may contain malicious content (e.g., a recovered .exe could be malware)
- Users should scan recovered files for security before opening them
- Scan result caches contain MFT metadata (filenames, sizes, timestamps) but not file content
