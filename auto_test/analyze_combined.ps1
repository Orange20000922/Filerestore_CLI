$files = @('D:\usnlist_output.txt', 'D:\usnlist_c_output.txt')

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

$totalOK = @{}
$totalReused = @{}
$sizes = @('resident_0-1KB', 'small_1-10KB', 'medium_10-100KB', 'large_100KB-1MB', 'xlarge_1-10MB', 'huge_10MB+')
foreach ($s in $sizes) { $totalOK[$s] = 0; $totalReused[$s] = 0 }

foreach ($file in $files) {
    $content = Get-Content $file -Encoding UTF8
    $okLines = @($content | Select-String '\[OK\]')
    $reusedLines = @($content | Select-String '\[MFT_REUSED\]')

    foreach ($line in $okLines) {
        $text = $line.Line
        if ($text -match '^\d+\s+\S+.*?\s+([\d.]+\s*(?:B|KB|MB|GB)|-)\s+\d{4}-\d{2}-\d{2}') {
            $size = Parse-FileSize $Matches[1]
            if ($size -ge 0) {
                if ($size -lt 1024) { $totalOK['resident_0-1KB']++ }
                elseif ($size -lt 10*1024) { $totalOK['small_1-10KB']++ }
                elseif ($size -lt 100*1024) { $totalOK['medium_10-100KB']++ }
                elseif ($size -lt 1024*1024) { $totalOK['large_100KB-1MB']++ }
                elseif ($size -lt 10*1024*1024) { $totalOK['xlarge_1-10MB']++ }
                else { $totalOK['huge_10MB+']++ }
            }
        }
    }

    foreach ($line in $reusedLines) {
        $text = $line.Line
        if ($text -match '^\d+\s+\S+.*?\s+([\d.]+\s*(?:B|KB|MB|GB)|-)\s+\d{4}-\d{2}-\d{2}') {
            $size = Parse-FileSize $Matches[1]
            if ($size -ge 0) {
                if ($size -lt 1024) { $totalReused['resident_0-1KB']++ }
                elseif ($size -lt 10*1024) { $totalReused['small_1-10KB']++ }
                elseif ($size -lt 100*1024) { $totalReused['medium_10-100KB']++ }
                elseif ($size -lt 1024*1024) { $totalReused['large_100KB-1MB']++ }
                elseif ($size -lt 10*1024*1024) { $totalReused['xlarge_1-10MB']++ }
                else { $totalReused['huge_10MB+']++ }
            }
        }
    }
}

Write-Host "=== Combined Stats (C + D drives) ==="
Write-Host ""
Write-Host "Size Range           OK    REUSED  Total   Rate"
Write-Host "-------------------------------------------------------"

$grandOK = 0
$grandReused = 0

foreach ($s in $sizes) {
    $ok = $totalOK[$s]
    $reused = $totalReused[$s]
    $total = $ok + $reused
    $grandOK += $ok
    $grandReused += $reused
    if ($total -gt 0) {
        $rate = [math]::Round($reused / $total * 100, 1)
        $sName = $s.PadRight(20)
        Write-Host "$sName $ok`t$reused`t$total`t$rate"
    }
}

Write-Host "-------------------------------------------------------"
$grandTotal = $grandOK + $grandReused
$grandRate = [math]::Round($grandReused / $grandTotal * 100, 1)
Write-Host "TOTAL                $grandOK`t$grandReused`t$grandTotal`t$grandRate"
