$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_e8.ps1') -RunName 'm13_e8b_extended_10x512_1500' -BoundedSteps 1500
exit $LASTEXITCODE
