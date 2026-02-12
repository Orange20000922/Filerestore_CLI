# 收集大量数据并按时间+大小分析 MFT 复用率
# 使用 --limit=10000 获取更多数据

param(
    [string]$Drive = "D",
    [int]$Hours = 720,
    [int]$Limit = 10000
)

$exePath = "D:\Users\21405\source\repos\Filerestore_CLI\x64\Release\Filerestore_CLI.exe"
$outputFile = "D:\usnlist_analysis_$Drive.txt"

Write-Host "=== MFT Reuse Rate Analysis ==="
Write-Host "Drive: $Drive, Hours: $Hours, Limit: $Limit"
Write-Host ""

# 运行 usnlist 命令（需要管理员权限）
Write-Host "Running usnlist (requires admin)..."
$cmd = "$exePath --cmd `"usnlist $Drive $Hours --validate --limit=$Limit`" > $outputFile 2>&1"
Start-Process -FilePath 'cmd' -ArgumentList '/c', $cmd -Verb RunAs -Wait

Write-Host "Output saved to: $outputFile"
Write-Host ""

# 读取输出
$content = Get-Content $outputFile -Encoding UTF8

# 解析函数
function Parse-FileSize {
    param([string]$sizeStr)
    if ($sizeStr -eq '-' -or [string]::IsNullOrWhiteSpace($sizeStr)) { return -1 }
    $sizeStr = $sizeStr.Trim()
    if ($sizeStr -match '^([\d.]+)\s*(B|KB|MB|GB)$') {
        $num = [double]$Matches[1]
        $unit = $Matches[2]
        switch ($unit) {
            'B'  { return $num }
            'KB' { return $num * 1024 }
            'MB' { return $num * 1024 * 1024 }
            'GB' { return $num * 1024 * 1024 * 1024 }
        }
    }
    return -1
}

function Get-SizeBucket {
    param([double]$size)
    if ($size -lt 0) { return "unknown" }
    if ($size -lt 1024) { return "0-1KB" }
    if ($size -lt 10*1024) { return "1-10KB" }
    if ($size -lt 100*1024) { return "10-100KB" }
    if ($size -lt 1024*1024) { return "100KB-1MB" }
    if ($size -lt 10*1024*1024) { return "1-10MB" }
    return "10MB+"
}

function Get-AgeBucket {
    param([datetime]$deleteTime)
    $now = Get-Date
    $age = $now - $deleteTime
    if ($age.TotalHours -lt 24) { return "0-24h" }
    if ($age.TotalDays -lt 3) { return "1-3d" }
    if ($age.TotalDays -lt 7) { return "3-7d" }
    if ($age.TotalDays -lt 14) { return "7-14d" }
    if ($age.TotalDays -lt 30) { return "14-30d" }
    return "30d+"
}

# 初始化统计表
$sizeBuckets = @("0-1KB", "1-10KB", "10-100KB", "100KB-1MB", "1-10MB", "10MB+")
$ageBuckets = @("0-24h", "1-3d", "3-7d", "7-14d", "14-30d", "30d+")

$stats = @{}
foreach ($age in $ageBuckets) {
    $stats[$age] = @{}
    foreach ($size in $sizeBuckets) {
        $stats[$age][$size] = @{ OK = 0; REUSED = 0 }
    }
}

# 解析每一行
$okLines = @($content | Select-String '\[OK\]')
$reusedLines = @($content | Select-String '\[MFT_REUSED\]')

Write-Host "Parsing $($okLines.Count) OK files and $($reusedLines.Count) REUSED files..."

foreach ($line in $okLines) {
    $text = $line.Line
    # 格式: idx   filename   size   timestamp   [status]
    if ($text -match '^\d+\s+\S+.*?\s+([\d.]+\s*(?:B|KB|MB|GB)|-)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[OK\]') {
        $sizeStr = $Matches[1]
        $timeStr = $Matches[2]

        $size = Parse-FileSize $sizeStr
        $sizeBucket = Get-SizeBucket $size

        try {
            $deleteTime = [datetime]::ParseExact($timeStr, "yyyy-MM-dd HH:mm:ss", $null)
            $ageBucket = Get-AgeBucket $deleteTime

            if ($sizeBucket -ne "unknown" -and $stats.ContainsKey($ageBucket)) {
                $stats[$ageBucket][$sizeBucket].OK++
            }
        } catch {}
    }
}

foreach ($line in $reusedLines) {
    $text = $line.Line
    if ($text -match '^\d+\s+\S+.*?\s+([\d.]+\s*(?:B|KB|MB|GB)|-)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[MFT_REUSED\]') {
        $sizeStr = $Matches[1]
        $timeStr = $Matches[2]

        $size = Parse-FileSize $sizeStr
        $sizeBucket = Get-SizeBucket $size

        try {
            $deleteTime = [datetime]::ParseExact($timeStr, "yyyy-MM-dd HH:mm:ss", $null)
            $ageBucket = Get-AgeBucket $deleteTime

            if ($sizeBucket -ne "unknown" -and $stats.ContainsKey($ageBucket)) {
                $stats[$ageBucket][$sizeBucket].REUSED++
            }
        } catch {}
    }
}

# 输出结果
Write-Host ""
Write-Host "=== MFT Reuse Rate by Time and Size ==="
Write-Host ""

# 表头
$header = "Age".PadRight(10)
foreach ($size in $sizeBuckets) {
    $header += $size.PadRight(12)
}
$header += "Total"
Write-Host $header
Write-Host ("-" * 90)

# 每个时间段
foreach ($age in $ageBuckets) {
    $row = $age.PadRight(10)
    $ageTotal = 0
    $ageReused = 0

    foreach ($size in $sizeBuckets) {
        $ok = $stats[$age][$size].OK
        $reused = $stats[$age][$size].REUSED
        $total = $ok + $reused
        $ageTotal += $total
        $ageReused += $reused

        if ($total -gt 0) {
            $rate = [math]::Round($reused / $total * 100, 0)
            $cell = "$rate%($total)"
        } else {
            $cell = "-"
        }
        $row += $cell.PadRight(12)
    }

    # 该时间段总计
    if ($ageTotal -gt 0) {
        $ageRate = [math]::Round($ageReused / $ageTotal * 100, 0)
        $row += "$ageRate%($ageTotal)"
    } else {
        $row += "-"
    }

    Write-Host $row
}

Write-Host ("-" * 90)

# 每个大小段总计
$totalRow = "Total".PadRight(10)
$grandTotal = 0
$grandReused = 0

foreach ($size in $sizeBuckets) {
    $sizeOK = 0
    $sizeReused = 0
    foreach ($age in $ageBuckets) {
        $sizeOK += $stats[$age][$size].OK
        $sizeReused += $stats[$age][$size].REUSED
    }
    $sizeTotal = $sizeOK + $sizeReused
    $grandTotal += $sizeTotal
    $grandReused += $sizeReused

    if ($sizeTotal -gt 0) {
        $rate = [math]::Round($sizeReused / $sizeTotal * 100, 0)
        $cell = "$rate%($sizeTotal)"
    } else {
        $cell = "-"
    }
    $totalRow += $cell.PadRight(12)
}

if ($grandTotal -gt 0) {
    $grandRate = [math]::Round($grandReused / $grandTotal * 100, 0)
    $totalRow += "$grandRate%($grandTotal)"
}
Write-Host $totalRow

Write-Host ""
Write-Host "Legend: Rate%(Count) - MFT reuse rate and sample count"

# 保存 JSON 结果
$jsonResult = @{
    drive = $Drive
    hours = $Hours
    timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    totalFiles = $grandTotal
    overallReuseRate = $grandRate
    byTimeAndSize = $stats
}

$jsonPath = "D:\mft_reuse_analysis_$Drive.json"
$jsonResult | ConvertTo-Json -Depth 4 | Out-File $jsonPath -Encoding UTF8
Write-Host ""
Write-Host "JSON saved to: $jsonPath"
