# 分析 usnlist 输出
param([string]$InputFile = 'D:\usnlist_output.txt')
$content = Get-Content $InputFile -Encoding UTF8

# 统计各状态
$okLines = @($content | Select-String '\[OK\]')
$reusedLines = @($content | Select-String '\[MFT_REUSED\]')
$noDataLines = @($content | Select-String '\[NO_DATA\]')
$sigMismatchLines = @($content | Select-String '\[SIG_MISMATCH\]')

Write-Host "=== USN 文件状态统计 ==="
Write-Host "OK (MFT 未复用): $($okLines.Count)"
Write-Host "MFT_REUSED (MFT 已复用): $($reusedLines.Count)"
Write-Host "NO_DATA (无数据属性): $($noDataLines.Count)"
Write-Host "SIG_MISMATCH (签名不匹配): $($sigMismatchLines.Count)"
Write-Host ""

# 分析 OK 状态文件的大小分布
Write-Host "=== OK 状态文件大小分布 ==="

# 解析文件大小
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

# 分桶统计
$buckets = @{
    'resident_0-1KB' = 0
    'small_1-10KB' = 0
    'medium_10-100KB' = 0
    'large_100KB-1MB' = 0
    'xlarge_1-10MB' = 0
    'huge_10MB+' = 0
    'unknown' = 0
}

$okSizes = @()
foreach ($line in $okLines) {
    # 提取大小列 (第3列，在文件名之后)
    $text = $line.Line
    # 格式: idx   filename                        size        timestamp   [status]
    if ($text -match '^\d+\s+\S+.*?\s+([\d.]+\s*(?:B|KB|MB|GB)|-)\s+\d{4}-\d{2}-\d{2}') {
        $sizeStr = $Matches[1]
        $size = Parse-FileSize $sizeStr
        if ($size -ge 0) {
            $okSizes += $size
            if ($size -lt 1024) { $buckets['resident_0-1KB']++ }
            elseif ($size -lt 10*1024) { $buckets['small_1-10KB']++ }
            elseif ($size -lt 100*1024) { $buckets['medium_10-100KB']++ }
            elseif ($size -lt 1024*1024) { $buckets['large_100KB-1MB']++ }
            elseif ($size -lt 10*1024*1024) { $buckets['xlarge_1-10MB']++ }
            else { $buckets['huge_10MB+']++ }
        } else {
            $buckets['unknown']++
        }
    }
}

Write-Host "文件大小分布 (OK 状态):"
Write-Host "  0-1KB (常驻候选): $($buckets['resident_0-1KB'])"
Write-Host "  1-10KB (小文件): $($buckets['small_1-10KB'])"
Write-Host "  10-100KB (中文件): $($buckets['medium_10-100KB'])"
Write-Host "  100KB-1MB (大文件): $($buckets['large_100KB-1MB'])"
Write-Host "  1-10MB (超大文件): $($buckets['xlarge_1-10MB'])"
Write-Host "  10MB+ (巨型文件): $($buckets['huge_10MB+'])"
Write-Host "  未知大小: $($buckets['unknown'])"

# 同样分析 MFT_REUSED 状态
Write-Host ""
Write-Host "=== MFT_REUSED 状态文件大小分布 ==="

$reusedBuckets = @{
    'resident_0-1KB' = 0
    'small_1-10KB' = 0
    'medium_10-100KB' = 0
    'large_100KB-1MB' = 0
    'xlarge_1-10MB' = 0
    'huge_10MB+' = 0
    'unknown' = 0
}

foreach ($line in $reusedLines) {
    $text = $line.Line
    if ($text -match '^\d+\s+\S+.*?\s+([\d.]+\s*(?:B|KB|MB|GB)|-)\s+\d{4}-\d{2}-\d{2}') {
        $sizeStr = $Matches[1]
        $size = Parse-FileSize $sizeStr
        if ($size -ge 0) {
            if ($size -lt 1024) { $reusedBuckets['resident_0-1KB']++ }
            elseif ($size -lt 10*1024) { $reusedBuckets['small_1-10KB']++ }
            elseif ($size -lt 100*1024) { $reusedBuckets['medium_10-100KB']++ }
            elseif ($size -lt 1024*1024) { $reusedBuckets['large_100KB-1MB']++ }
            elseif ($size -lt 10*1024*1024) { $reusedBuckets['xlarge_1-10MB']++ }
            else { $reusedBuckets['huge_10MB+']++ }
        } else {
            $reusedBuckets['unknown']++
        }
    }
}

Write-Host "文件大小分布 (MFT_REUSED 状态):"
Write-Host "  0-1KB (常驻候选): $($reusedBuckets['resident_0-1KB'])"
Write-Host "  1-10KB (小文件): $($reusedBuckets['small_1-10KB'])"
Write-Host "  10-100KB (中文件): $($reusedBuckets['medium_10-100KB'])"
Write-Host "  100KB-1MB (大文件): $($reusedBuckets['large_100KB-1MB'])"
Write-Host "  1-10MB (超大文件): $($reusedBuckets['xlarge_1-10MB'])"
Write-Host "  10MB+ (巨型文件): $($reusedBuckets['huge_10MB+'])"
Write-Host "  未知大小: $($reusedBuckets['unknown'])"

# 计算各大小段的复用率
Write-Host ""
Write-Host "=== 各大小段 MFT 复用率 ==="
$sizes = @('resident_0-1KB', 'small_1-10KB', 'medium_10-100KB', 'large_100KB-1MB', 'xlarge_1-10MB', 'huge_10MB+')
foreach ($s in $sizes) {
    $ok = $buckets[$s]
    $reused = $reusedBuckets[$s]
    $total = $ok + $reused
    if ($total -gt 0) {
        $rate = [math]::Round($reused / $total * 100, 1)
        Write-Host "  $s : OK=$ok, REUSED=$reused, 复用率=$rate%"
    }
}
