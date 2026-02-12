#!/usr/bin/env pwsh
# ============================================================================
# MFT 记录存活时间窗口测试
# ============================================================================
# 使用 --test 模式将命令输出重定向到 debug.log，
# 然后解析日志中的 [OUTPUT] 行来分析 MFT 存活率。
#
# 用法: .\mft_survival_test.ps1
#       .\mft_survival_test.ps1 -TestDrive C -MaxHours 336
# ============================================================================

param(
    [string]$ExePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe",
    [string]$LogPath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\debug.log",
    [string]$TestDrive = "D",
    [int]$MaxHours = 168
)

$ErrorActionPreference = "Continue"

function Write-Header($msg) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Yellow }
function Write-OK($msg)    { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err($msg)   { Write-Host "[ERR]  $msg" -ForegroundColor Red }

# ============================================================================
# 前置检查
# ============================================================================
Write-Header "MFT Record Survival Time Window Test"
Write-Info "Executable: $ExePath"
Write-Info "Log file:   $LogPath"
Write-Info "Test Drive: $TestDrive"
Write-Info "Time Range: Last $MaxHours hours"

if (-not (Test-Path $ExePath)) {
    Write-Err "Executable not found. Build Release x64 first."
    exit 1
}

# ============================================================================
# Phase 1: 运行 usnlist --validate --test（输出写入日志）
# ============================================================================
Write-Header "Phase 1: Running usnlist with --test mode"

# 记录日志起始位置（通过在日志中插入标记）
$marker = "===MFT_SURVIVAL_TEST_START_$(Get-Date -Format 'yyyyMMddHHmmss')==="
Write-Info "Marker: $marker"

$cmd = "usnlist $TestDrive $MaxHours --validate"
Write-Info "Command: $cmd"
Write-Info "Running with --test flag (output -> debug.log)..."

$startTime = Get-Date
& $ExePath --cmd $cmd --test 2>$null
$exitCode = $LASTEXITCODE
$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Info "Exit code: $exitCode, Duration: $([math]::Round($duration, 1))s"

if (-not (Test-Path $LogPath)) {
    Write-Err "Log file not found: $LogPath"
    exit 1
}

# ============================================================================
# Phase 2: 从日志中提取 [OUTPUT] 行
# ============================================================================
Write-Header "Phase 2: Parsing log output"

# 读取日志最近的内容（取最后 5000 行应该够了）
$logLines = Get-Content $LogPath -Tail 5000 -Encoding UTF8

# 提取 [OUTPUT] 行
$outputLines = $logLines | Where-Object { $_ -match '\[OUTPUT\]' } | ForEach-Object {
    # 格式: [timestamp] [INFO] [OUTPUT] 实际内容
    if ($_ -match '\[OUTPUT\]\s*(.*)$') {
        $Matches[1]
    }
}

Write-Info "Total [OUTPUT] lines found: $($outputLines.Count)"

if ($outputLines.Count -eq 0) {
    Write-Err "No [OUTPUT] lines found in log. Check if --test mode worked."
    Write-Info "Last 10 log lines:"
    $logLines | Select-Object -Last 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    exit 1
}

# 显示前几行看格式
Write-Info "Sample output lines:"
$outputLines | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

# ============================================================================
# Phase 3: 解析结果并统计
# ============================================================================
Write-Header "Phase 3: Analyzing MFT survival"

# 时间分桶
$timeBuckets = @(
    @{ Name = "0-10min";   MinH = 0;     MaxH = 0.167;  Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "10-30min";  MinH = 0.167; MaxH = 0.5;    Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "30min-1h";  MinH = 0.5;   MaxH = 1;      Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "1-3h";      MinH = 1;     MaxH = 3;      Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "3-6h";      MinH = 3;     MaxH = 6;      Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "6-12h";     MinH = 6;     MaxH = 12;     Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "12-24h";    MinH = 12;    MaxH = 24;     Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "1-3d";      MinH = 24;    MaxH = 72;     Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "3-7d";      MinH = 72;    MaxH = 168;    Total = 0; OK = 0; Reused = 0; Other = 0 }
    @{ Name = "7d+";       MinH = 168;   MaxH = 99999;  Total = 0; OK = 0; Reused = 0; Other = 0 }
)

$now = Get-Date
$totalParsed = 0
$statusCounts = @{}

foreach ($line in $outputLines) {
    # 匹配数据行格式:
    # 0     filename.txt                            -           2026-02-11 14:30:00 [OK]        85%
    # 也匹配没有百分比的情况:
    # 1     another.docx                            -           2026-02-10 09:15:23 [MFT_REUSED]-
    if ($line -match '^\s*(\d+)\s+(.+?)\s{2,}-\s{2,}(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\[(\w+)\]') {
        $fileName = $Matches[2].Trim()
        $timeStr = $Matches[3]
        $status = $Matches[4]

        $totalParsed++

        # 统计状态分布
        if (-not $statusCounts.ContainsKey($status)) { $statusCounts[$status] = 0 }
        $statusCounts[$status]++

        # 计算删除距今多少小时
        try {
            $deleteTime = [DateTime]::ParseExact($timeStr, "yyyy-MM-dd HH:mm:ss", $null)
            $ageHours = ($now - $deleteTime).TotalHours
            if ($ageHours -lt 0) { $ageHours = 0 }

            # 判断是否可恢复
            $isOK = ($status -eq "OK" -or $status -eq "SUCCESS" -or $status -eq "RESIDENT" -or $status -eq "MFT_REUSED_OK")
            $isReused = ($status -eq "MFT_REUSED")

            # 分桶
            foreach ($bucket in $timeBuckets) {
                if ($ageHours -ge $bucket.MinH -and $ageHours -lt $bucket.MaxH) {
                    $bucket.Total++
                    if ($isOK) { $bucket.OK++ }
                    elseif ($isReused) { $bucket.Reused++ }
                    else { $bucket.Other++ }
                    break
                }
            }
        }
        catch { }
    }
}

Write-Info "Files parsed: $totalParsed"

if ($totalParsed -eq 0) {
    Write-Err "No data rows parsed. Showing raw output sample:"
    $outputLines | Select-Object -First 30 | ForEach-Object { Write-Host "  |$_|" -ForegroundColor Gray }
    exit 1
}

# ============================================================================
# Phase 4: 输出结果
# ============================================================================
Write-Header "Results: MFT Record Survival Curve"

# 状态分布
Write-Host "`n--- Status Distribution ---" -ForegroundColor White
foreach ($key in $statusCounts.Keys | Sort-Object) {
    $count = $statusCounts[$key]
    $pct = [math]::Round($count / $totalParsed * 100, 1)
    Write-Host ("  {0,-20} {1,5} ({2,5}%)" -f $key, $count, $pct)
}

# 生存曲线表格
Write-Host "`n--- MFT Survival by Deletion Age ---`n" -ForegroundColor White
Write-Host ("{0,-12} {1,6} {2,6} {3,8} {4,9} {5}" -f "Age", "Total", "OK", "Rate", "Reused", "Survival")
Write-Host ("{0,-12} {1,6} {2,6} {3,8} {4,9} {5}" -f "---", "-----", "----", "------", "-------", "--------")

foreach ($bucket in $timeBuckets) {
    if ($bucket.Total -gt 0) {
        $rate = [math]::Round($bucket.OK / $bucket.Total * 100, 1)
        $barLen = [math]::Round($rate / 2.5)
        $bar = [string]::new('#', [math]::Max($barLen, 0))

        $color = if ($rate -ge 70) { "Green" } elseif ($rate -ge 30) { "Yellow" } else { "Red" }

        Write-Host ("{0,-12} {1,6} {2,6} " -f $bucket.Name, $bucket.Total, $bucket.OK) -NoNewline
        Write-Host ("{0,6}% " -f $rate) -NoNewline -ForegroundColor $color
        Write-Host ("{0,9} " -f $bucket.Reused) -NoNewline
        Write-Host $bar -ForegroundColor $color
    }
    else {
        Write-Host ("{0,-12} {1,6}    -       -         -" -f $bucket.Name, 0) -ForegroundColor DarkGray
    }
}

# ============================================================================
# Phase 5: 关键指标和结论
# ============================================================================
Write-Header "Key Metrics & Conclusion"

$totalOK = ($timeBuckets | Measure-Object -Property OK -Sum).Sum
$totalAll = ($timeBuckets | Measure-Object -Property Total -Sum).Sum
$overallRate = if ($totalAll -gt 0) { [math]::Round($totalOK / $totalAll * 100, 1) } else { 0 }

# 分时间段统计
$calc = @(
    @{ Label = "Last 1 hour";  Buckets = $timeBuckets | Where-Object { $_.MaxH -le 1 } }
    @{ Label = "Last 6 hours"; Buckets = $timeBuckets | Where-Object { $_.MaxH -le 6 } }
    @{ Label = "Last 24 hours";Buckets = $timeBuckets | Where-Object { $_.MaxH -le 24 } }
)

Write-Host ""
foreach ($c in $calc) {
    $ok = ($c.Buckets | Measure-Object -Property OK -Sum).Sum
    $tot = ($c.Buckets | Measure-Object -Property Total -Sum).Sum
    $r = if ($tot -gt 0) { [math]::Round($ok / $tot * 100, 1) } else { "N/A" }
    Write-Host ("  {0,-20} {1}% ({2}/{3})" -f $c.Label, $r, $ok, $tot)
}
Write-Host ("  {0,-20} {1}% ({2}/{3})" -f "Overall", $overallRate, $totalOK, $totalAll)

# 半衰期
$halfLife = "N/A"
foreach ($bucket in $timeBuckets) {
    if ($bucket.Total -ge 3) {
        $r = $bucket.OK / $bucket.Total * 100
        if ($r -lt 50) { $halfLife = $bucket.Name; break }
    }
}
Write-Host "  Half-life (< 50%):   $halfLife"

# 结论
Write-Host "`n--- Conclusion ---" -ForegroundColor White
$recent1h = $timeBuckets | Where-Object { $_.MaxH -le 1 }
$r1hOK = ($recent1h | Measure-Object -Property OK -Sum).Sum
$r1hTot = ($recent1h | Measure-Object -Property Total -Sum).Sum
$r1hRate = if ($r1hTot -gt 0) { [math]::Round($r1hOK / $r1hTot * 100, 1) } else { 0 }

if ($r1hRate -ge 70) {
    Write-OK "MFT-guided scan: HIGH value (recent files $r1hRate% recoverable)"
} elseif ($r1hRate -ge 40) {
    Write-Info "MFT-guided scan: MODERATE value (recent files $r1hRate% recoverable)"
} else {
    Write-Err "MFT-guided scan: LOW value (recent files $r1hRate% recoverable)"
}

Write-Host ""
Write-Host "  This data shows the time window where MFT data runs" -ForegroundColor White
Write-Host "  can be used to skip full-disk signature scanning." -ForegroundColor White
Write-Host ""

# ============================================================================
# Phase 6: 保存 JSON 报告
# ============================================================================
$reportFile = "mft_survival_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$reportPath = "D:\Users\21405\source\repos\Filerestore_CLI\document\$reportFile"

$report = @{
    timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    drive = $TestDrive
    maxHours = $MaxHours
    durationSeconds = [math]::Round($duration, 1)
    totalFiles = $totalParsed
    overallSurvivalRate = $overallRate
    halfLife = $halfLife
    statusDistribution = $statusCounts
    buckets = $timeBuckets | ForEach-Object {
        @{
            name = $_.Name
            total = $_.Total
            ok = $_.OK
            reused = $_.Reused
            rate = if ($_.Total -gt 0) { [math]::Round($_.OK / $_.Total * 100, 1) } else { 0 }
        }
    }
}

$report | ConvertTo-Json -Depth 5 | Out-File $reportPath -Encoding UTF8
Write-Info "Report saved: $reportFile"
Write-Host ""
