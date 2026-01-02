# ========================================
# 文件恢复工具测试脚本
# File Recovery Tool Test Script
# ========================================

param(
    [string]$TestFileName = "test_recovery_file.txt",
    [string]$TestFilePath = "C:\Temp",
    [int]$FileSizeKB = 10,
    [string]$ProgramPath = ".\x64\Debug\ConsoleApplication5.exe",
    [switch]$MultipleFiles,
    [switch]$SkipPrompt
)

Write-Host @"

========================================
  文件恢复工具测试脚本
  File Recovery Tool Test Script
========================================

"@ -ForegroundColor Cyan

# 创建测试目录（如果不存在）
if (-not (Test-Path $TestFilePath)) {
    New-Item -ItemType Directory -Path $TestFilePath -Force | Out-Null
    Write-Host "[+] Created test directory: $TestFilePath" -ForegroundColor Green
}

# 存储创建的文件信息
$createdFiles = @()

# ========== Step 1: 创建测试文件 ==========
Write-Host "`n========== Step 1: Creating Test Files ==========" -ForegroundColor Cyan

if ($MultipleFiles) {
    # 创建多个不同类型的测试文件
    $testFiles = @(
        @{Name="test_document.txt"; Size=5; Content="Text document"},
        @{Name="test_data.xml"; Size=3; Content="XML data"},
        @{Name="test_image.png"; Size=20; Content="Binary data"},
        @{Name="test_config.json"; Size=2; Content="JSON config"}
    )

    foreach ($file in $testFiles) {
        $filePath = Join-Path $TestFilePath $file.Name
        $content = "$($file.Content) - Created at $(Get-Date)`n"
        $content += "=" * 80 + "`n"

        # 填充到指定大小
        while ($content.Length -lt ($file.Size * 1024)) {
            $content += "Random data: $(Get-Random -Maximum 999999)`n"
        }

        $content | Out-File -FilePath $filePath -Encoding UTF8
        $fileInfo = Get-Item $filePath
        $createdFiles += $fileInfo

        Write-Host "[+] Created: $($file.Name) ($($fileInfo.Length) bytes)" -ForegroundColor Green
    }
} else {
    # 创建单个测试文件
    $FullPath = Join-Path $TestFilePath $TestFileName

    $content = @"
======================================
Test File for Recovery Tool
Created at: $(Get-Date)
======================================

This is a test file to verify deleted file scanning and recovery functionality.

File Information:
- Name: $TestFileName
- Path: $TestFilePath
- Size: Approximately $FileSizeKB KB

Random Test Data:
"@

    # 填充到指定大小
    while ($content.Length -lt ($FileSizeKB * 1024)) {
        $content += "Line $(Get-Random): $(Get-Random -Maximum 999999) - Random test data`n"
    }

    $content | Out-File -FilePath $FullPath -Encoding UTF8
    $fileInfo = Get-Item $FullPath
    $createdFiles += $fileInfo

    Write-Host "[+] Created test file: $FullPath" -ForegroundColor Green
    Write-Host "    Size: $($fileInfo.Length) bytes ($([math]::Round($fileInfo.Length/1KB, 2)) KB)" -ForegroundColor Yellow
}

# 显示所有创建的文件信息
Write-Host "`n[*] File Information:" -ForegroundColor Cyan
foreach ($file in $createdFiles) {
    Write-Host "    Name:     $($file.Name)"
    Write-Host "    Path:     $($file.FullName)"
    Write-Host "    Size:     $($file.Length) bytes"
    Write-Host "    Created:  $($file.CreationTime)"
    Write-Host "    Modified: $($file.LastWriteTime)"
    Write-Host "    ---"
}

# 等待用户确认
if (-not $SkipPrompt) {
    Write-Host "`n[!] Press ANY KEY to DELETE the test file(s)..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# ========== Step 2: 删除文件 ==========
Write-Host "`n========== Step 2: Deleting Test Files ==========" -ForegroundColor Cyan
Write-Host "[*] Deleting files (bypassing Recycle Bin - permanent deletion)..." -ForegroundColor Yellow

$deletedCount = 0
foreach ($file in $createdFiles) {
    try {
        # 使用 .NET 方法永久删除（绕过回收站）
        [System.IO.File]::Delete($file.FullName)

        if (-not (Test-Path $file.FullName)) {
            Write-Host "[+] Deleted: $($file.Name)" -ForegroundColor Green
            $deletedCount++
        } else {
            Write-Host "[-] ERROR: Failed to delete $($file.Name)" -ForegroundColor Red
        }
    } catch {
        Write-Host "[-] ERROR deleting $($file.Name): $_" -ForegroundColor Red
    }
}

Write-Host "`n[*] Deletion Summary:" -ForegroundColor Cyan
Write-Host "    Files created: $($createdFiles.Count)"
Write-Host "    Files deleted: $deletedCount"
Write-Host "    Deleted at: $(Get-Date)" -ForegroundColor Yellow

if ($deletedCount -eq 0) {
    Write-Host "`n[-] ERROR: No files were deleted!" -ForegroundColor Red
    exit 1
}

# 等待文件系统刷新
Write-Host "`n[*] Waiting for filesystem to flush metadata..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# ========== Step 3: 启动恢复程序 ==========
Write-Host "`n========== Step 3: Launching Recovery Program ==========" -ForegroundColor Cyan

# 查找程序 - 自动选择最新编译的版本
$possiblePaths = @(
    $ProgramPath,
    ".\x64\Debug\ImportTableAnalyzer.exe",
    ".\x64\Release\ImportTableAnalyzer.exe",
    ".\Debug\ImportTableAnalyzer.exe",
    ".\Release\ImportTableAnalyzer.exe",
    ".\ImportTableAnalyzer.exe"
)

# 收集所有存在的程序文件及其修改时间
$foundFiles = @()
foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $fileInfo = Get-Item $path
        $foundFiles += [PSCustomObject]@{
            Path = $fileInfo.FullName
            LastWriteTime = $fileInfo.LastWriteTime
            SizeKB = [math]::Round($fileInfo.Length / 1KB, 2)
            Config = if ($path -match "Release") { "Release" } elseif ($path -match "Debug") { "Debug" } else { "Unknown" }
        }
    }
}

$foundPath = $null
if ($foundFiles.Count -gt 0) {
    # 按最后修改时间排序，选择最新的版本
    $latestFile = $foundFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $foundPath = $latestFile.Path

    # 显示找到的所有版本
    if ($foundFiles.Count -gt 1) {
        Write-Host "[*] Found multiple versions:" -ForegroundColor Yellow
        foreach ($file in ($foundFiles | Sort-Object LastWriteTime -Descending)) {
            $isLatest = if ($file.Path -eq $foundPath) { " [LATEST]" } else { "" }
            Write-Host ("    - {0} ({1}, {2} KB, Modified: {3}){4}" -f `
                (Split-Path $file.Path -Leaf), `
                $file.Config, `
                $file.SizeKB, `
                $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"), `
                $isLatest) -ForegroundColor $(if ($isLatest) { "Green" } else { "Gray" })
        }
        Write-Host ""
    }
}

if ($foundPath) {
    Write-Host "[+] Found program: $foundPath" -ForegroundColor Green

    # 显示建议的测试命令
    Write-Host "`n[*] Suggested test commands:" -ForegroundColor Cyan
    Write-Host "    listdeleted C none" -ForegroundColor White
    Write-Host "    searchdeleted C test .txt" -ForegroundColor White

    if ($MultipleFiles) {
        Write-Host "    searchdeleted C test .xml" -ForegroundColor White
        Write-Host "    searchdeleted C test .png" -ForegroundColor White
        Write-Host "    searchdeleted C test .json" -ForegroundColor White
    } else {
        Write-Host "    searchdeleted C $([System.IO.Path]::GetFileNameWithoutExtension($TestFileName))" -ForegroundColor White
    }

    Write-Host "`n[*] Test file details:" -ForegroundColor Yellow
    foreach ($file in $createdFiles) {
        Write-Host "    - $($file.Name) (deleted from $TestFilePath)"
    }

    Write-Host "`n[*] Launching program in 3 seconds..." -ForegroundColor Green
    Start-Sleep -Seconds 3

    # 启动程序
    Start-Process -FilePath $foundPath -WorkingDirectory (Split-Path $foundPath -Parent)

    Write-Host "[+] Program launched!" -ForegroundColor Green
} else {
    Write-Host "[-] ERROR: Program not found!" -ForegroundColor Red
    Write-Host "    Searched paths:" -ForegroundColor Yellow
    foreach ($path in $possiblePaths) {
        Write-Host "    - $path"
    }
    Write-Host "`n    Please build the project first or specify correct path using -ProgramPath parameter" -ForegroundColor Yellow
}

Write-Host "`n========== Test Setup Complete ==========" -ForegroundColor Cyan
Write-Host "`nTest completed at: $(Get-Date)`n" -ForegroundColor Gray
