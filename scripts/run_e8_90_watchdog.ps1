$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
& (Join-Path $PSScriptRoot 'run_e8_watchdog.ps1') -RunDirectory (Join-Path $root 'workspace\outputs\m13_e8b_pilot_watchdog90')
exit $LASTEXITCODE
