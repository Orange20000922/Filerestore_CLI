# Filerestore_CLI Release 打包脚本
# 用法: .\package_release.ps1 [-Version "1.0.0"] [-IncludeCUDA]

param(
    [string]$Version = "0.3.1",
    [switch]$IncludeCUDA = $false
)

$ErrorActionPreference = "Stop"

# 路径配置
$ProjectRoot = $PSScriptRoot
$ReleaseDir = "$ProjectRoot\x64\Release"
$OutputDir = "$ProjectRoot\release_packages"
$PackageName = "Filerestore_CLI_v${Version}_x64"

if ($IncludeCUDA) {
    $PackageName += "_cuda"
}

$PackageDir = "$OutputDir\$PackageName"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Filerestore_CLI Release Packager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Version: $Version"
Write-Host "Include CUDA: $IncludeCUDA"
Write-Host "Output: $PackageDir"
Write-Host ""

# 检查 Release 是否存在
if (-not (Test-Path "$ReleaseDir\Filerestore_CLI.exe")) {
    Write-Host "Error: Filerestore_CLI.exe not found in $ReleaseDir" -ForegroundColor Red
    Write-Host "Please build Release configuration first." -ForegroundColor Red
    exit 1
}

# 创建输出目录
if (Test-Path $PackageDir) {
    Write-Host "Removing existing package directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $PackageDir
}

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path "$PackageDir\models" | Out-Null
New-Item -ItemType Directory -Force -Path "$PackageDir\langs" | Out-Null

Write-Host ""
Write-Host "Copying files..." -ForegroundColor Green

# 1. 复制主程序
Write-Host "  - Filerestore_CLI.exe"
Copy-Item "$ReleaseDir\Filerestore_CLI.exe" "$PackageDir\"

# 2. 复制 ONNX Runtime DLL (CPU)
Write-Host "  - onnxruntime.dll (CPU inference)"
Copy-Item "$ReleaseDir\onnxruntime.dll" "$PackageDir\"

# 3. 复制 CUDA DLLs (可选)
if ($IncludeCUDA) {
    Write-Host "  - onnxruntime_providers_cuda.dll (CUDA support)" -ForegroundColor Yellow
    Copy-Item "$ReleaseDir\onnxruntime_providers_cuda.dll" "$PackageDir\"
    Copy-Item "$ReleaseDir\onnxruntime_providers_shared.dll" "$PackageDir\"
    Copy-Item "$ReleaseDir\onnxruntime_providers_tensorrt.dll" "$PackageDir\"
}

# 4. 复制 ML 模型
Write-Host "  - models/file_classifier_deep.onnx"
Write-Host "  - models/file_classifier_deep.json"
Copy-Item "$ReleaseDir\models\*" "$PackageDir\models\" -Recurse

# 5. 复制语言文件
Write-Host "  - langs/en.json"
Write-Host "  - langs/zh.json"
Copy-Item "$ReleaseDir\langs\*" "$PackageDir\langs\" -Recurse

# 6. 创建简要说明文件
$ReadmeContent = @"
Filerestore_CLI v$Version
========================

NTFS File Recovery Tool with ML-based file type classification.

USAGE:
------
Run as Administrator (required for raw disk access):

  Filerestore_CLI.exe

Common Commands:
  help              - Show all commands
  scan <drive>      - Scan deleted files (e.g., scan C)
  carvepool <drive> <types> <output> - Fast file carving

Examples:
  scan C                           - Scan deleted files on C:
  carvepool C jpg,png D:\recovered - Recover images from C:
  carvepool C all D:\recovered     - Recover all supported types

REQUIREMENTS:
-------------
- Windows 10/11 x64
- Administrator privileges
- NTFS file system

ML MODEL:
---------
The ML model (models/file_classifier_deep.onnx) provides intelligent
file type classification for enhanced recovery accuracy.

Supported file types: jpg, png, gif, bmp, pdf, doc, xls, ppt,
                      zip, exe, mp4, mp3, txt, html, xml, etc.

LICENSE:
--------
MIT License

GitHub: https://github.com/yourusername/Filerestore_CLI
"@

$ReadmeContent | Out-File -FilePath "$PackageDir\README.txt" -Encoding UTF8

Write-Host ""
Write-Host "Creating ZIP archive..." -ForegroundColor Green

# 创建 ZIP
$ZipPath = "$OutputDir\$PackageName.zip"
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath
}

Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipPath -CompressionLevel Optimal

# 计算文件大小
$ZipSize = (Get-Item $ZipPath).Length
$ZipSizeMB = [math]::Round($ZipSize / 1MB, 2)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Package created successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files:"
Write-Host "  Directory: $PackageDir"
Write-Host "  ZIP file:  $ZipPath ($ZipSizeMB MB)"
Write-Host ""

# 显示包内容
Write-Host "Package contents:" -ForegroundColor Yellow
Get-ChildItem $PackageDir -Recurse | ForEach-Object {
    $relativePath = $_.FullName.Replace($PackageDir, "").TrimStart("\")
    $size = if ($_.PSIsContainer) { "[DIR]" } else { "$([math]::Round($_.Length / 1KB, 1)) KB" }
    Write-Host "  $relativePath - $size"
}

Write-Host ""
Write-Host "Ready to upload to GitHub Releases!" -ForegroundColor Green
