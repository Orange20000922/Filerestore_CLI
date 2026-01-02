# PowerShell 脚本：读取 Windows 错误报告
# 需要以管理员权限运行

Write-Host "=== Windows Error Report Reader ===" -ForegroundColor Green
Write-Host ""

# 查找最近的崩溃报告
$reportPath = "C:\ProgramData\Microsoft\Windows\WER\ReportQueue"
Write-Host "Searching for crash reports in: $reportPath" -ForegroundColor Yellow

# 查找包含 ImportTable 的目录
$crashDirs = Get-ChildItem $reportPath -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*ImportTable*" } |
    Sort-Object LastWriteTime -Descending

if ($crashDirs.Count -eq 0) {
    Write-Host "No ImportTableAnalyzer crash reports found." -ForegroundColor Red
    Write-Host "Showing all recent crash reports:" -ForegroundColor Yellow
    $crashDirs = Get-ChildItem $reportPath -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 10
}

foreach ($dir in $crashDirs) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Report: $($dir.Name)" -ForegroundColor Cyan
    Write-Host "Time: $($dir.LastWriteTime)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # 列出目录中的所有文件
    $files = Get-ChildItem $dir.FullName -ErrorAction SilentlyContinue

    foreach ($file in $files) {
        Write-Host "`nFile: $($file.Name) ($($file.Length) bytes)" -ForegroundColor Yellow

        # 读取 Report.wer 文件（包含崩溃摘要）
        if ($file.Name -eq "Report.wer") {
            Write-Host "--- Report.wer Content ---" -ForegroundColor Green
            Get-Content $file.FullName -ErrorAction SilentlyContinue | Write-Host
        }

        # 如果是 .dmp 文件，显示路径
        if ($file.Extension -eq ".dmp") {
            Write-Host "*** Memory Dump File Found ***" -ForegroundColor Magenta
            Write-Host "Path: $($file.FullName)" -ForegroundColor Magenta
            Write-Host "Use WinDbg to analyze this file with: !analyze -v" -ForegroundColor Magenta
        }

        # 如果是 .txt 文件，显示内容
        if ($file.Extension -eq ".txt") {
            Write-Host "--- $($file.Name) Content ---" -ForegroundColor Green
            Get-Content $file.FullName -ErrorAction SilentlyContinue | Write-Host
        }
    }
}

Write-Host "`n=== Analysis Complete ===" -ForegroundColor Green
