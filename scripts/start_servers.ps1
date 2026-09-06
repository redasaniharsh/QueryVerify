# Start the QueryVerify demo fresh: kill anything on ports 8000/8501, wait,
# then start backend + frontend with py -3.11 and print their PIDs.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\start_servers.ps1

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Stop-Port([int]$Port) {
    $owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        Write-Host "Killing PID $owner on port $Port"
        Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
    }
}

Stop-Port 8000
Stop-Port 8501

Start-Sleep -Seconds 2

$backend = Start-Process -FilePath "py" `
    -ArgumentList "-3.11", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru

$frontend = Start-Process -FilePath "py" `
    -ArgumentList "-3.11", "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true" `
    -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru

Write-Host ""
Write-Host "Started (py launcher PIDs):"
Write-Host "  Backend  -> $($backend.Id)"
Write-Host "  Frontend -> $($frontend.Id)"

# Wait up to ~30s for each to actually start listening, then report.
$beUp = $false; $feUp = $false
foreach ($i in 1..60) {
    if (-not $beUp -and (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) { $beUp = $true }
    if (-not $feUp -and (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)) { $feUp = $true }
    if ($beUp -and $feUp) { break }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host ("Backend  :8000  up=" + $beUp + "   PID=" + ((Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)))
Write-Host ("Frontend :8501  up=" + $feUp + "   PID=" + ((Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)))