#!/usr/bin/env pwsh
# MFT 存活窗口测试 - 创建/删除/验证
param(
    [string]$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe",
    [string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log",
    [string]$TestDrive = "D"
)

Write-Host "`n=== MFT Survival Test (Create/Delete/Verify) ===" -ForegroundColor Cyan

# 0. 清除旧日志
Remove-Item $LogPath -Force -ErrorAction SilentlyContinue
Write-Host "[1/4] Log cleared" -ForegroundColor Yellow

# 1. 创建测试文件
$testDir = "${TestDrive}:\__mft_test_tmp__"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

$rng = New-Object System.Random

# 1KB (可能常驻)
$buf1k = New-Object byte[] 1024
$rng.NextBytes($buf1k)
[System.IO.File]::WriteAllBytes("$testDir\test_1kb.txt", $buf1k)

# 100KB (非常驻, 有 data runs)
$buf100k = New-Object byte[] 102400
$rng.NextBytes($buf100k)
[System.IO.File]::WriteAllBytes("$testDir\test_100kb.bin", $buf100k)

# 1MB
$buf1m = New-Object byte[] 1048576
$rng.NextBytes($buf1m)
[System.IO.File]::WriteAllBytes("$testDir\test_1mb.dat", $buf1m)

# 500KB (带 JPG 签名头)
$bufJpg = New-Object byte[] 512000
$rng.NextBytes($bufJpg)
$bufJpg[0] = 0xFF; $bufJpg[1] = 0xD8; $bufJpg[2] = 0xFF; $bufJpg[3] = 0xE0
[System.IO.File]::WriteAllBytes("$testDir\test_photo.jpg", $bufJpg)

# 200KB (带 PDF 签名头)
$bufPdf = New-Object byte[] 204800
$rng.NextBytes($bufPdf)
$pdfHeader = [System.Text.Encoding]::ASCII.GetBytes("%PDF-1.4")
[Array]::Copy($pdfHeader, $bufPdf, $pdfHeader.Length)
[System.IO.File]::WriteAllBytes("$testDir\test_doc.pdf", $bufPdf)

Write-Host "[2/4] Created 5 test files:" -ForegroundColor Yellow
Get-ChildItem $testDir | ForEach-Object {
    Write-Host ("       {0,-25} {1,10:N0} bytes" -f $_.Name, $_.Length)
}

# 2. 删除文件
Start-Sleep -Milliseconds 200
Remove-Item "$testDir\*" -Force
Remove-Item $testDir -Force
$deleteTime = Get-Date
Write-Host "[3/4] Files deleted at $($deleteTime.ToString('HH:mm:ss'))" -ForegroundColor Yellow

# 3. 立即用 usnlist --validate 验证
Start-Sleep -Milliseconds 500
Write-Host "[4/4] Running usnlist --validate --test ..." -ForegroundColor Yellow

$startTime = Get-Date
& $ExePath --cmd "usnlist $TestDrive 1 --validate --pattern=test_" --test 2>$null
$duration = ((Get-Date) - $startTime).TotalSeconds

Write-Host "       Scan completed in $([math]::Round($duration, 1))s" -ForegroundColor Yellow

# 4. 解析日志结果
Write-Host "`n=== Results ===" -ForegroundColor Cyan

$logLines = Get-Content $LogPath -Encoding UTF8
$outputLines = $logLines | Where-Object { $_ -match '\[OUTPUT\]' } | ForEach-Object {
    if ($_ -match '\[OUTPUT\]\s*(.*)$') { $Matches[1] }
}

Write-Host "Total [OUTPUT] lines: $($outputLines.Count)"

# 找我们的测试文件
$testResults = $outputLines | Where-Object { $_ -match 'test_' }

if ($testResults.Count -eq 0) {
    Write-Host "`nNo test files found in results." -ForegroundColor Red
    Write-Host "Showing all data lines:" -ForegroundColor Yellow
    $outputLines | Where-Object { $_ -match '^\s*\d+\s+' } | ForEach-Object {
        Write-Host "  $_" -ForegroundColor Gray
    }
} else {
    Write-Host "`nTest file results:" -ForegroundColor Green
    Write-Host ("{0,-30} {1,-15} {2}" -f "Filename", "Status", "Confidence")
    Write-Host ("{0,-30} {1,-15} {2}" -f "--------", "------", "----------")

    foreach ($line in $testResults) {
        if ($line -match '^\s*\d+\s+(\S+)\s.*\[(\w+)\]\s*(\S*)') {
            $name = $Matches[1]
            $status = $Matches[2]
            $conf = $Matches[3]

            $color = switch ($status) {
                "OK"           { "Green" }
                "SUCCESS"      { "Green" }
                "RESIDENT"     { "Green" }
                "MFT_REUSED"   { "Red" }
                "NO_DATA"      { "Yellow" }
                default        { "Gray" }
            }

            Write-Host ("{0,-30} " -f $name) -NoNewline
            Write-Host ("{0,-15} " -f "[$status]") -NoNewline -ForegroundColor $color
            Write-Host $conf
        }
    }
}

# 统计
$okCount = ($testResults | Where-Object { $_ -match '\[OK\]|\[SUCCESS\]|\[RESIDENT\]' }).Count
$totalCount = $testResults.Count
Write-Host "`n--- Summary ---" -ForegroundColor White
Write-Host "  Test files found:   $totalCount / 5"
Write-Host "  Data runs valid:    $okCount / $totalCount"

if ($okCount -gt 0) {
    Write-Host "`n  CONCLUSION: MFT data runs survive immediately after deletion!" -ForegroundColor Green
    Write-Host "  MFT-guided scan IS viable for recently deleted files." -ForegroundColor Green
} elseif ($totalCount -gt 0) {
    Write-Host "`n  CONCLUSION: MFT data runs NOT available even immediately after deletion." -ForegroundColor Red
    Write-Host "  MFT-guided scan has very limited value on this system." -ForegroundColor Red
} else {
    Write-Host "`n  CONCLUSION: Test files not found in USN. Check permissions or drive." -ForegroundColor Yellow
}

Write-Host ""
