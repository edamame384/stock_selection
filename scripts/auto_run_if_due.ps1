$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$dataDir = Join-Path $repoRoot "data"
$logPath = Join-Path $dataDir "auto_run.log"
$statePath = Join-Path $dataDir "last_auto_run_date.txt"

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

$now = Get-Date
$today = $now.ToString("yyyy-MM-dd")
$cutoff = [TimeSpan]::Parse("23:30:00")

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
    & python test.py 2>&1 | Out-File -FilePath $logPath -Append -Encoding utf8
    if ($LASTEXITCODE -eq 0) {
        Set-Content -Path $statePath -Value $today -Encoding utf8
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Auto run success" | Out-File -FilePath $logPath -Append -Encoding utf8
    } else {
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Auto run failed (exit=$LASTEXITCODE)" | Out-File -FilePath $logPath -Append -Encoding utf8
    }
} finally {
    Pop-Location
}
