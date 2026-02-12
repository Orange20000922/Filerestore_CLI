# 合并 C 盘和 D 盘的分析结果
$cJson = Get-Content "D:\mft_reuse_analysis_C.json" -Encoding UTF8 | ConvertFrom-Json
$dJson = Get-Content "D:\mft_reuse_analysis_D.json" -Encoding UTF8 | ConvertFrom-Json

$sizeBuckets = @("0-1KB", "1-10KB", "10-100KB", "100KB-1MB", "1-10MB", "10MB+")
$ageBuckets = @("0-24h", "1-3d", "3-7d", "7-14d", "14-30d", "30d+")

# 合并统计
$combined = @{}
foreach ($age in $ageBuckets) {
    $combined[$age] = @{}
    foreach ($size in $sizeBuckets) {
        $cOK = 0; $cReused = 0; $dOK = 0; $dReused = 0

        if ($cJson.byTimeAndSize.$age.$size) {
            $cOK = $cJson.byTimeAndSize.$age.$size.OK
            $cReused = $cJson.byTimeAndSize.$age.$size.REUSED
        }
        if ($dJson.byTimeAndSize.$age.$size) {
            $dOK = $dJson.byTimeAndSize.$age.$size.OK
            $dReused = $dJson.byTimeAndSize.$age.$size.REUSED
        }

        $combined[$age][$size] = @{
            OK = $cOK + $dOK
            REUSED = $cReused + $dReused
        }
    }
}

Write-Host "=============================================="
Write-Host "  MFT Reuse Rate Analysis (C + D Combined)"
Write-Host "  Total Files: $($cJson.totalFiles + $dJson.totalFiles)"
Write-Host "=============================================="
Write-Host ""

# 表头
$header = "Age".PadRight(10)
foreach ($size in $sizeBuckets) {
    $header += $size.PadRight(14)
}
$header += "Row Total"
Write-Host $header
Write-Host ("=" * 110)

# 每个时间段
foreach ($age in $ageBuckets) {
    $row = $age.PadRight(10)
    $ageTotal = 0
    $ageReused = 0

    foreach ($size in $sizeBuckets) {
        $ok = $combined[$age][$size].OK
        $reused = $combined[$age][$size].REUSED
        $total = $ok + $reused
        $ageTotal += $total
        $ageReused += $reused

        if ($total -ge 10) {  # 至少10个样本才显示
            $rate = [math]::Round($reused / $total * 100, 0)
            $cell = "$rate% (n=$total)"
        } elseif ($total -gt 0) {
            $cell = "n=$total"
        } else {
            $cell = "-"
        }
        $row += $cell.PadRight(14)
    }

    if ($ageTotal -ge 10) {
        $ageRate = [math]::Round($ageReused / $ageTotal * 100, 0)
        $row += "$ageRate% (n=$ageTotal)"
    } elseif ($ageTotal -gt 0) {
        $row += "n=$ageTotal"
    } else {
        $row += "-"
    }

    Write-Host $row
}

Write-Host ("=" * 110)

# 列总计
$totalRow = "Col Total".PadRight(10)
$grandTotal = 0
$grandReused = 0

foreach ($size in $sizeBuckets) {
    $sizeOK = 0
    $sizeReused = 0
    foreach ($age in $ageBuckets) {
        $sizeOK += $combined[$age][$size].OK
        $sizeReused += $combined[$age][$size].REUSED
    }
    $sizeTotal = $sizeOK + $sizeReused
    $grandTotal += $sizeTotal
    $grandReused += $sizeReused

    if ($sizeTotal -ge 10) {
        $rate = [math]::Round($sizeReused / $sizeTotal * 100, 0)
        $cell = "$rate% (n=$sizeTotal)"
    } else {
        $cell = "n=$sizeTotal"
    }
    $totalRow += $cell.PadRight(14)
}

$grandRate = [math]::Round($grandReused / $grandTotal * 100, 0)
$totalRow += "$grandRate% (n=$grandTotal)"
Write-Host $totalRow

Write-Host ""
Write-Host "=============================================="
Write-Host "  Key Findings"
Write-Host "=============================================="
Write-Host ""

# 分析趋势
Write-Host "1. Size Impact on MFT Reuse Rate:"
foreach ($size in $sizeBuckets) {
    $sizeOK = 0; $sizeReused = 0
    foreach ($age in $ageBuckets) {
        $sizeOK += $combined[$age][$size].OK
        $sizeReused += $combined[$age][$size].REUSED
    }
    $sizeTotal = $sizeOK + $sizeReused
    if ($sizeTotal -ge 50) {
        $rate = [math]::Round($sizeReused / $sizeTotal * 100, 1)
        Write-Host "   $size : $rate% reuse rate (n=$sizeTotal)"
    }
}

Write-Host ""
Write-Host "2. Time Impact on MFT Reuse Rate:"
foreach ($age in $ageBuckets) {
    $ageOK = 0; $ageReused = 0
    foreach ($size in $sizeBuckets) {
        $ageOK += $combined[$age][$size].OK
        $ageReused += $combined[$age][$size].REUSED
    }
    $ageTotal = $ageOK + $ageReused
    if ($ageTotal -ge 50) {
        $rate = [math]::Round($ageReused / $ageTotal * 100, 1)
        Write-Host "   $age : $rate% reuse rate (n=$ageTotal)"
    }
}

Write-Host ""
Write-Host "3. Recommendations:"
Write-Host "   - Small files (0-1KB): Low reuse rate, good recovery chance"
Write-Host "   - Medium files (10-100KB): Moderate reuse, act quickly"
Write-Host "   - Large files (1MB+): High reuse rate, prioritize immediate recovery"
Write-Host "   - Time is critical: Reuse rate increases significantly after 24h"
