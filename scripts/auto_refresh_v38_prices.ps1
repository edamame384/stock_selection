$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$dataDir = Join-Path $repoRoot "data"
$logPath = Join-Path $dataDir "auto_refresh_v38_prices.log"
$statePath = Join-Path $dataDir "last_v38_refresh_date.txt"

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

$now = Get-Date
$today = $now.ToString("yyyy-MM-dd")
$cutoff = [TimeSpan]::Parse("22:30:00")

$alreadyRanToday = $false
if (Test-Path $statePath) {
    $last = (Get-Content -Path $statePath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($last -eq $today) {
        $alreadyRanToday = $true
    }
}

if ($alreadyRanToday) {
    exit 0
}

if ($now.TimeOfDay -lt $cutoff) {
    exit 0
}

Push-Location $repoRoot
try {
    "[$($now.ToString('yyyy-MM-dd HH:mm:ss'))] V38 refresh start" | Out-File -FilePath $logPath -Append -Encoding utf8
    & python scripts/update_nikkei_phase.py 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
    $nikkeiExit = $LASTEXITCODE
    if ($nikkeiExit -ne 0) {
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] V38 nikkei refresh failed (exit=$nikkeiExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
        exit $nikkeiExit
    }
    & python scripts/run_v38_signal.py --dataset 2026-2Q --refresh-only --max-stale-count 200 --max-selected-stale-count 0 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
    $runExit = $LASTEXITCODE
    if ($runExit -eq 0) {
        Set-Content -Path $statePath -Value $today -Encoding utf8
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] V38 refresh success" | Out-File -FilePath $logPath -Append -Encoding utf8
    } else {
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] V38 refresh failed (exit=$runExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
    }
} finally {
    Pop-Location
}
