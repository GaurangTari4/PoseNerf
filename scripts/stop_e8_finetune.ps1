param([Parameter(Mandatory=$true)][string]$RunDirectory) New-Item -ItemType File -Force -Path (Join-Path $RunDirectory 'graceful_stop.request') | Out-Null
