# Filerestore_CLI 项目配置

## 项目目录结构

- **主项目**：`D:\Users\21405\source\repos\Filerestore_CLI`
  - 整个项目的主体，包含 CLI/TUI 应用层代码（Filerestore_CLI）
- **内核驱动项目**：`D:\Users\21405\source\repos\Filerestore_sys`
  - 内核层代码的主项目，涉及驱动相关的修改应先在此处进行（用户在此编译测试）
  - 测试通过后再同步到主项目

**注意：仅内核驱动相关改动走 Filerestore_sys → 主项目 的同步流程，应用层代码直接在主项目修改。**

## 构建环境

MSBuild 路径：
```
C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe
```

**重要：必须使用 PowerShell 执行构建命令，CMD 可能会报错。**

## 构建命令

```powershell
# Release x64
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI\Filerestore_CLI.vcxproj' /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal

# Debug x64
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI\Filerestore_CLI.vcxproj' /p:Configuration=Debug /p:Platform=x64 /t:Build /v:minimal
```

如果在 Bash 中执行，需要包装为 PowerShell 命令：
```bash
powershell -Command "& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' ..."
```

## 输出路径

- Release: `Filerestore_CLI\x64\Release\Filerestore_CLI.exe`
- Debug: `Filerestore_CLI\x64\Debug\Filerestore_CLI.exe`
