# Filerestore_CLI 项目配置

## 构建环境

MSBuild 路径：
```
C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe
```

## 构建命令

```powershell
# Release x64
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI\Filerestore_CLI.vcxproj' /p:Configuration=Release /p:Platform=x64 /t:Build /v:minimal

# Debug x64
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 'D:\Users\21405\source\repos\Filerestore_CLI\Filerestore_CLI\Filerestore_CLI.vcxproj' /p:Configuration=Debug /p:Platform=x64 /t:Build /v:minimal
```

## 输出路径

- Release: `Filerestore_CLI\x64\Release\Filerestore_CLI.exe`
- Debug: `Filerestore_CLI\x64\Debug\Filerestore_CLI.exe`
