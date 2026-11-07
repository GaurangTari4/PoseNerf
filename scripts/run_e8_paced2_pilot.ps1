$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_e8.ps1') -Pilot -RunName 'm13_e8b_pilot_10x512_pause_2s' -OptimizerPauseSeconds 2.0
exit $LASTEXITCODE
