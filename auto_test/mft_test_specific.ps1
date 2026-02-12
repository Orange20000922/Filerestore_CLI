#!/usr/bin/env pwsh
# 精确测试：创建大文件、删除、用特定文件名查找
param(
    [string]$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe",
    [string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log",
    [string]$TestDrive = "D"
)

Write-Host "`n=== MFT Data Run Survival Test ===" -ForegroundColor Cyan

# 0. 清除旧日志
Remove-Item $LogPath -Force -ErrorAction SilentlyContinue
Write-Host "[1/5] Log cleared" -ForegroundColor Yellow

# 1. 创建测试文件（使用唯一前缀避免匹配其他文件）
$testDir = "${TestDrive}:\__mft_surv_test__"
New-Item -ItemType Directory -Path $testDir -Force | Out-Null

$rng = New-Object System.Random
$prefix = "mftsurv"

# 100KB (非常驻，有 data runs)
$buf100k = New-Object byte[] 102400
$rng.NextBytes($buf100k)
[System.IO.File]::WriteAllBytes("$testDir\${prefix}_100kb.bin", $buf100k)

# 1MB
$buf1m = New-Object byte[] 1048576
$rng.NextBytes($buf1m)
[System.IO.File]::WriteAllBytes("$testDir\${prefix}_1mb.dat", $buf1m)

# 500KB (带 JPG 签名头)
$bufJpg = New-Object byte[] 512000
$rng.NextBytes($bufJpg)
$bufJpg[0] = 0xFF; $bufJpg[1] = 0xD8; $bufJpg[2] = 0xFF; $bufJpg[3] = 0xE0
[System.IO.File]::WriteAllBytes("$testDir\${prefix}_photo.jpg", $bufJpg)

# 200KB (带 PDF 签名头)
$bufPdf = New-Object byte[] 204800
$rng.NextBytes($bufPdf)
$pdfHeader = [System.Text.Encoding]::ASCII.GetBytes("%PDF-1.4")
[Array]::Copy($pdfHeader, $bufPdf, $pdfHeader.Length)
[System.IO.File]::WriteAllBytes("$testDir\${prefix}_doc.pdf", $bufPdf)

Write-Host "[2/5] Created 4 test files:" -ForegroundColor Yellow
Get-ChildItem $testDir | ForEach-Object {
    Write-Host ("       {0,-25} {1,10:N0} bytes" -f $_.Name, $_.Length)
}

# 2. 删除文件
Start-Sleep -Milliseconds 300
Remove-Item "$testDir\*" -Force
Remove-Item $testDir -Force
$deleteTime = Get-Date
Write-Host "[3/5] Files deleted at $($deleteTime.ToString('HH:mm:ss.fff'))" -ForegroundColor Yellow

# 3. 立即用 usnlist --validate 验证（用唯一前缀）
Start-Sleep -Milliseconds 500
Write-Host "[4/5] Running usnlist --validate --pattern=$prefix ..." -ForegroundColor Yellow

$startTime = Get-Date
& $ExePath --cmd "usnlist $TestDrive 1 --validate --pattern=$prefix" --test 2>$null
$duration = ((Get-Date) - $startTime).TotalSeconds

Write-Host "       Scan completed in $([math]::Round($duration, 1))s" -ForegroundColor Yellow

# 4. 等待日志写入完成
Start-Sleep -Milliseconds 1000

# 5. 解析结果
Write-Host "[5/5] Parsing results..." -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $LogPath)) {
    Write-Host "ERROR: Log file not found at $LogPath" -ForegroundColor Red
    Write-Host "Program may need admin rights or wrote log elsewhere." -ForegroundColor Red

    # 尝试搜索其他可能的位置
    $altPaths = @(
        "$env:TEMP\debug.log",
        "C:\debug.log",
        "$env:USERPROFILE\debug.log"
    )
    foreach ($alt in $altPaths) {
        if (Test-Path $alt) {
            Write-Host "Found log at: $alt" -ForegroundColor Green
        }
    }
    exit 1
}

$logLines = Get-Content $LogPath -Encoding UTF8
$outputLines = $logLines | Where-Object { $_ -match '\[OUTPUT\]' } | ForEach-Object {
    if ($_ -match '\[OUTPUT\]\s*(.*)$') { $Matches[1] }
}

Write-Host "=== Results ===" -ForegroundColor Cyan
Write-Host "Total [OUTPUT] lines: $($outputLines.Count)"

# 找我们的测试文件
$testResults = $outputLines | Where-Object { $_ -match $prefix }

if ($testResults.Count -eq 0) {
    Write-Host "`nNo test files ($prefix*) found in USN results." -ForegroundColor Red
    Write-Host ""

    # 显示所有数据行（如果有的话）
    $dataLines = $outputLines | Where-Object { $_ -match '^\s*\d+\s+' }
    if ($dataLines.Count -gt 0) {
        Write-Host "Files found in USN (first 10):" -ForegroundColor Yellow
        $dataLines | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Gray
        }
    }

    # 显示汇总行
    $summaryLines = $outputLines | Where-Object { $_ -match '\[usnlist' }
    foreach ($sl in $summaryLines) {
        Write-Host "  $sl" -ForegroundColor Gray
    }

    Write-Host "`nPossible reasons:" -ForegroundColor Yellow
    Write-Host "  1. USN journal doesn't capture these files (too fast create/delete?)" -ForegroundColor Gray
    Write-Host "  2. Time filter excluded them" -ForegroundColor Gray
    Write-Host "  3. The files were on a different volume" -ForegroundColor Gray
} else {
    Write-Host "`nTest file results:" -ForegroundColor Green
    Write-Host ("{0,-30} {1,-15} {2}" -f "Filename", "Status", "Confidence")
    Write-Host ("{0,-30} {1,-15} {2}" -f "--------", "------", "----------")

    foreach ($line in $testResults) {
        Write-Host "  $line" -ForegroundColor White

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

            Write-Host ("  -> {0,-25} " -f $name) -NoNewline
            Write-Host ("[{0}]" -f $status) -ForegroundColor $color -NoNewline
            Write-Host " $conf"
        }
    }

    $okCount = ($testResults | Where-Object { $_ -match '\[OK\]|\[SUCCESS\]|\[RESIDENT\]' }).Count
    $totalCount = $testResults.Count
    Write-Host "`n--- Summary ---" -ForegroundColor White
    Write-Host "  Test files found:   $totalCount / 4"
    Write-Host "  Data runs valid:    $okCount / $totalCount"

    if ($okCount -gt 0) {
        Write-Host "`n  CONCLUSION: MFT data runs survive immediately after deletion!" -ForegroundColor Green
    } else {
        Write-Host "`n  CONCLUSION: MFT data runs NOT found even immediately after deletion." -ForegroundColor Red
    }
}

Write-Host ""

# 额外：显示日志中的关键信息
Write-Host "=== Debug Info ===" -ForegroundColor DarkGray
$mftLines = $logLines | Where-Object { $_ -match 'MFT|data.run|sequence' } | Select-Object -First 20
foreach ($ml in $mftLines) {
    Write-Host "  $ml" -ForegroundColor DarkGray
}
Write-Host ""
