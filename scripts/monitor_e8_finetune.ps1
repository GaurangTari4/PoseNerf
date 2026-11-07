param([Parameter(Mandatory=$true)][string]$RunDirectory)
$metrics=Join-Path $RunDirectory 'e8_logs\training_metrics.csv'
if(-not(Test-Path -LiteralPath $metrics)){Write-Output "Metrics not initialized; training has not started or has not reached its first telemetry event.";exit 0}
Get-Content -LiteralPath $metrics -Wait
