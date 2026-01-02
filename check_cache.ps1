# 缓存内容检查脚本
# Check Cache Content Script

$cachePath = Join-Path $env:TEMP "deleted_files_C.cache"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Cache File Inspector" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

if (Test-Path $cachePath) {
    $cacheInfo = Get-Item $cachePath
    Write-Host "[+] Cache file found: $cachePath" -ForegroundColor Green
    Write-Host "    Size: $($cacheInfo.Length) bytes ($([math]::Round($cacheInfo.Length/1MB, 2)) MB)" -ForegroundColor Yellow
    Write-Host "    Created: $($cacheInfo.CreationTime)" -ForegroundColor Yellow
    Write-Host "    Modified: $($cacheInfo.LastWriteTime)" -ForegroundColor Yellow

    # 计算缓存年龄
    $age = (Get-Date) - $cacheInfo.LastWriteTime
    Write-Host "    Age: $($age.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Yellow

    if ($age.TotalMinutes -gt 60) {
        Write-Host "    [!] Cache is older than 60 minutes - may be expired" -ForegroundColor Red
    } else {
        Write-Host "    [✓] Cache is valid (< 60 minutes)" -ForegroundColor Green
    }

    Write-Host "`n[*] Reading cache content..." -ForegroundColor Cyan

    # 尝试读取缓存（简单统计）
    try {
        $bytes = [System.IO.File]::ReadAllBytes($cachePath)
        Write-Host "    Total bytes: $($bytes.Length)" -ForegroundColor White

        # 搜索文件名中的 .txt 字符串（UTF-16LE编码）
        $txtPattern = [System.Text.Encoding]::Unicode.GetBytes(".txt")
        $xmlPattern = [System.Text.Encoding]::Unicode.GetBytes(".xml")
        $testPattern = [System.Text.Encoding]::Unicode.GetBytes("test")

        $txtCount = 0
        $xmlCount = 0
        $testCount = 0

        for ($i = 0; $i -lt ($bytes.Length - 8); $i++) {
            # 检查 .txt
            if ($bytes[$i] -eq $txtPattern[0] -and
                $bytes[$i+1] -eq $txtPattern[1] -and
                $bytes[$i+2] -eq $txtPattern[2] -and
                $bytes[$i+3] -eq $txtPattern[3] -and
                $bytes[$i+4] -eq $txtPattern[4] -and
                $bytes[$i+5] -eq $txtPattern[5] -and
                $bytes[$i+6] -eq $txtPattern[6] -and
                $bytes[$i+7] -eq $txtPattern[7]) {
                $txtCount++
            }

            # 检查 .xml
            if ($bytes[$i] -eq $xmlPattern[0] -and
                $bytes[$i+1] -eq $xmlPattern[1] -and
                $bytes[$i+2] -eq $xmlPattern[2] -and
                $bytes[$i+3] -eq $xmlPattern[3] -and
                $bytes[$i+4] -eq $xmlPattern[4] -and
                $bytes[$i+5] -eq $xmlPattern[5] -and
                $bytes[$i+6] -eq $xmlPattern[6] -and
                $bytes[$i+7] -eq $xmlPattern[7]) {
                $xmlCount++
            }

            # 检查 test
            if ($bytes[$i] -eq $testPattern[0] -and
                $bytes[$i+1] -eq $testPattern[1] -and
                $bytes[$i+2] -eq $testPattern[2] -and
                $bytes[$i+3] -eq $testPattern[3] -and
                $bytes[$i+4] -eq $testPattern[4] -and
                $bytes[$i+5] -eq $testPattern[5] -and
                $bytes[$i+6] -eq $testPattern[6] -and
                $bytes[$i+7] -eq $testPattern[7]) {
                $testCount++
            }
        }

        Write-Host "`n[*] Pattern search results:" -ForegroundColor Cyan
        Write-Host "    Files with '.txt' extension: $txtCount" -ForegroundColor $(if ($txtCount -gt 0) { "Green" } else { "Red" })
        Write-Host "    Files with '.xml' extension: $xmlCount" -ForegroundColor $(if ($xmlCount -gt 0) { "Green" } else { "Yellow" })
        Write-Host "    Files containing 'test': $testCount" -ForegroundColor $(if ($testCount -gt 0) { "Green" } else { "Yellow" })

        if ($txtCount -eq 0) {
            Write-Host "`n[!] WARNING: No .txt files found in cache!" -ForegroundColor Red
            Write-Host "    This explains why searching for .txt returns 0 results." -ForegroundColor Yellow
        }

        if ($testCount -eq 0) {
            Write-Host "`n[!] WARNING: No files containing 'test' found in cache!" -ForegroundColor Red
            Write-Host "    Your test file may not have been scanned/cached." -ForegroundColor Yellow
        }

    } catch {
        Write-Host "[-] Error reading cache: $_" -ForegroundColor Red
    }

} else {
    Write-Host "[-] Cache file not found: $cachePath" -ForegroundColor Red
    Write-Host "    Run 'listdeleted C none' to generate cache." -ForegroundColor Yellow
}

Write-Host "`n========================================`n" -ForegroundColor Cyan
