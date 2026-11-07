$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_e8.ps1') -Pilot -RunName 'm13_e8b_pilot_10x512_pause_0p5s' -OptimizerPauseSeconds 0.5
exit $LASTEXITCODE
