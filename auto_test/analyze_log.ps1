#!/usr/bin/env pwsh
# 分析 debug.log 中的状态分布
$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log"
$log = Get-Content $LogPath -Encoding UTF8

# 提取状态行
$statusCounts = @{}
$total = 0

foreach ($line in $log) {
    if ($line -match '\[OUTPUT\].*\[(OK|MFT_REUSED|NO_DATA|RESIDENT|SUCCESS|OVERWRITTEN|MFT_REUSED_OK)\]') {
        $status = $Matches[1]
        if (-not $statusCounts.ContainsKey($status)) { $statusCounts[$status] = 0 }
        $statusCounts[$status]++
        $total++
    }
}

Write-Host "=== Status Distribution ===" -ForegroundColor Cyan
foreach ($key in $statusCounts.Keys | Sort-Object) {
    $count = $statusCounts[$key]
    $pct = [math]::Round($count / $total * 100, 1)
    Write-Host ("  {0,-20} {1,5} ({2,5}%)" -f $key, $count, $pct)
}
Write-Host ""
Write-Host "Total files with status: $total"

# 摘要行
Write-Host ""
$log | Where-Object { $_ -match 'Loaded.*MFT data runs|usnlist\.total|usnlist\.recoverable' } | ForEach-Object {
    Write-Host "  $_" -ForegroundColor Gray
}

# 显示一些 OK 的文件
Write-Host ""
Write-Host "=== Sample OK files ===" -ForegroundColor Green
$okFiles = $log | Where-Object { $_ -match '\[OUTPUT\].*\[OK\]' } | Select-Object -First 10
foreach ($f in $okFiles) {
    Write-Host "  $f" -ForegroundColor Green
}

# 显示一些 RESIDENT 的文件
Write-Host ""
Write-Host "=== Sample RESIDENT files ===" -ForegroundColor Green
$resFiles = $log | Where-Object { $_ -match '\[OUTPUT\].*\[RESIDENT\]' } | Select-Object -First 5
foreach ($f in $resFiles) {
    Write-Host "  $f" -ForegroundColor Green
}
