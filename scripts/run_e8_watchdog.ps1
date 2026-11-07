[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunDirectory, [Parameter(ValueFromRemainingArguments=$true)][string[]]$WatchdogArgs)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$python = if ($env:POSENERF_E8_PYTHON) { $env:POSENERF_E8_PYTHON } else { Join-Path $root '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw "Repository-local Python is unavailable: $python" }
& $python (Join-Path $PSScriptRoot 'e8_thermal_watchdog.py') --run-dir $RunDirectory --trainer-lock (Join-Path $RunDirectory 'e8_training.lock') @WatchdogArgs
exit $LASTEXITCODE
