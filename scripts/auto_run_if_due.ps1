$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$dataDir = Join-Path $repoRoot "data"
$logPath = Join-Path $dataDir "auto_run.log"
$signalLogPath = Join-Path $dataDir "last_run_v38_local.log"
$signalStatePath = Join-Path $dataDir "last_signal_state_v38_local.json"
$statePath = Join-Path $dataDir "last_v38_signal_run_date.txt"
$refreshStatePath = Join-Path $dataDir "last_v38_refresh_date.txt"
$webhookUrl = "https://discord.com/api/webhooks/1479858388090355762/WfKk_sdIufDciR-g-LksZCjlzdSrLHZQ__558rfwGrv-wH9E9nQzdeiSRE9gmfHbrjN8"

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

$now = Get-Date
$today = $now.ToString("yyyy-MM-dd")
$cutoff = [TimeSpan]::Parse("23:00:00")

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
    "[$($now.ToString('yyyy-MM-dd HH:mm:ss'))] Auto run start" | Out-File -FilePath $logPath -Append -Encoding utf8
    $refreshRanToday = $false
    if (Test-Path $refreshStatePath) {
        $lastRefresh = (Get-Content -Path $refreshStatePath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if ($lastRefresh -eq $today) {
            $refreshRanToday = $true
        }
    }
    if (-not $refreshRanToday) {
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] V38 refresh missing for today; running fallback refresh" | Out-File -FilePath $logPath -Append -Encoding utf8
        & python scripts/update_nikkei_phase.py 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
        $nikkeiExit = $LASTEXITCODE
        if ($nikkeiExit -ne 0) {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Fallback nikkei refresh failed (exit=$nikkeiExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
            exit $nikkeiExit
        }
        & python scripts/run_v38_signal.py --dataset 2026-2Q --refresh-only --max-stale-count 200 --max-selected-stale-count 0 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
        $refreshExit = $LASTEXITCODE
        if ($refreshExit -ne 0) {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Fallback V38 refresh failed (exit=$refreshExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
            exit $refreshExit
        }
        Set-Content -Path $refreshStatePath -Value $today -Encoding utf8
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Fallback V38 refresh success" | Out-File -FilePath $logPath -Append -Encoding utf8
    }
    if (Test-Path $signalLogPath) {
        Remove-Item -LiteralPath $signalLogPath -Force
    }
    & python scripts/run_v38_signal.py --dataset 2026-2Q --skip-refresh --max-stale-count 25 --max-selected-stale-count 0 2>&1 | Out-File -FilePath $signalLogPath -Encoding utf8
    $runExit = $LASTEXITCODE
    if (Test-Path $signalLogPath) {
        Get-Content -Path $signalLogPath | Out-File -FilePath $logPath -Append -Encoding utf8
    }
    if ($runExit -eq 0) {
        & python scripts/workflow_notify.py --mode signal-log --log-path $signalLogPath --state-path $signalStatePath --take-profit-ratio 0.06 --webhook-url $webhookUrl 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
        $notifyExit = $LASTEXITCODE
        if ($notifyExit -eq 0) {
            Set-Content -Path $statePath -Value $today -Encoding utf8
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Auto run success" | Out-File -FilePath $logPath -Append -Encoding utf8
        } else {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Auto run notify failed (exit=$notifyExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
            exit $notifyExit
        }
    } else {
        & python scripts/workflow_notify.py --mode failure --webhook-url $webhookUrl 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
        $notifyExit = $LASTEXITCODE
        if ($notifyExit -ne 0) {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Failure notify failed (exit=$notifyExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
        }
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Auto run failed (exit=$runExit)" | Out-File -FilePath $logPath -Append -Encoding utf8
        exit $runExit
    }
} finally {
    Pop-Location
}
