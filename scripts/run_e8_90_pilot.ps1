$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_e8.ps1') -Pilot -RunName 'm13_e8b_pilot_watchdog90'
exit $LASTEXITCODE
