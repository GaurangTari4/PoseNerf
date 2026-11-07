[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$run = Join-Path $root 'workspace\outputs\m13_e8_manual_10x512_48k'
$logs = Join-Path $run 'e8_logs'
if (-not (Test-Path $python)) { throw "Missing interpreter: $python" }
if (Test-Path (Join-Path $run 'e8_training.lock')) { throw 'E8 trainer lock exists; refusing duplicate start.' }
if (Test-Path (Join-Path $run 'manifest.json')) { throw 'Manual E8 run already exists; use resume_e8_manual.ps1.' }
$free = (Get-PSDrive -Name ((Split-Path -Qualifier $root).TrimEnd(':'))).Free
if ($free -lt 2GB) { throw 'Less than 2 GiB free disk space; refusing start.' }
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
New-Item -ItemType Directory -Force $logs | Out-Null
Set-Content (Join-Path $logs 'errors.log') ''
$watch = Start-Process -FilePath powershell.exe -WindowStyle Normal -PassThru -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $PSScriptRoot 'run_e8_watchdog.ps1'),'-RunDirectory',$run)
Write-Host "Watchdog PID: $($watch.Id)"; Write-Host "Logs: $logs"
& $python -m posenerf.stage_c --project-dir workspace --run-name m13_e8_manual_10x512_48k --steps 48000 --train-images 192 --batch-rays 256 --downsample 4 --hidden 512 --layers 10 --coarse-samples 48 --fine-samples 48 --depth-loss-weight 0.03 --accumulation-steps 4 --snapshot-steps 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000,41000,42000,43000,44000,45000,46000,47000,48000
if ($LASTEXITCODE -ne 0) { Add-Content (Join-Path $logs 'errors.log') "$(Get-Date -Format o) trainer exited $LASTEXITCODE"; exit $LASTEXITCODE }
