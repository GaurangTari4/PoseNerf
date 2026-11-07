param([switch]$Watch)
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path; $run = Join-Path $root 'workspace\outputs\m13_e8_manual_10x512_48k'
do { if(Test-Path "$run\live\status.json"){Get-Content "$run\live\status.json"}; nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total --format=csv,noheader; if(Test-Path "$run\thermal_watchdog.jsonl"){Get-Content "$run\thermal_watchdog.jsonl" -Tail 3}; if($Watch){Start-Sleep 5} } while($Watch)
