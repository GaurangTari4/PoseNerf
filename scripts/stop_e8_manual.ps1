$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$run = Join-Path $root 'workspace\outputs\m13_e8_manual_10x512_48k'
$request = Join-Path $run 'graceful_stop.request'
@{reason='manual_stop_request'; requested_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content $request
Write-Host "Stop request written: $request"
