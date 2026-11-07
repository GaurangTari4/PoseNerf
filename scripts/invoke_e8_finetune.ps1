param([Parameter(Mandatory=$true)][string]$RunName,[Parameter(Mandatory=$true)][double]$DepthWeight,[switch]$Resume)
$ErrorActionPreference='Stop'
$root=(Resolve-Path "$PSScriptRoot\..").Path; $python=Join-Path $root '.venv\Scripts\python.exe'
$source=Join-Path $root 'workspace\outputs\m13_e8_manual_10x512_48k\checkpoints\hierarchical_nerf_resume_latest.pth'
$expected='66523e38054adc67c3e6c1e4650e5ac92aacddc8979ab6577313f667deaab972'
$run=Join-Path $root (Join-Path 'workspace\outputs' $RunName); $lock=Join-Path $run 'e8_training.lock'
if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Immutable E8 source SHA-256 mismatch.' }
New-Item -ItemType Directory -Force -Path $run | Out-Null
if(Test-Path $lock){$old=Get-Content $lock -Raw; $m=[regex]::Match($old,'trainer_pid=(\d+)');if($m.Success -and (Get-Process -Id $m.Groups[1].Value -ErrorAction SilentlyContinue)){throw "Active trainer owns $lock"};Move-Item $lock "$lock.stale.$([DateTime]::UtcNow.Ticks)"}
$stream=[IO.File]::Open($lock,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);$writer=[IO.StreamWriter]::new($stream)
$watchdog=$null
try{
 $writer.WriteLine("run=$RunName`nlauncher_pid=$PID`nsource_sha256=$expected");$writer.Flush()
 $watchdog=Start-Process -FilePath $python -ArgumentList @((Join-Path $PSScriptRoot 'e8_thermal_watchdog.py'),'--run-dir',$run,'--trainer-lock',$lock) -PassThru -WindowStyle Hidden
 $deadline=(Get-Date).AddSeconds(35);$heart=Join-Path $run 'thermal_watchdog.jsonl';while((Get-Date) -lt $deadline -and -not(Test-Path $heart)){Start-Sleep -Seconds 1};if(-not(Test-Path $heart)){throw 'Watchdog heartbeat was not created; trainer will not start.'}
 $args=@('-m','posenerf.stage_c','--project-dir',(Join-Path $root 'workspace'),'--run-name',$RunName,'--e8-continuation','--source-sha256',$expected,'--depth-loss-weight',$DepthWeight,'--batch-rays','256','--accumulation-steps','4','--coarse-samples','48','--fine-samples','48','--hidden','512','--layers','10')
 if($Resume){$args+='--resume-existing-continuation'}else{$args+=@('--resume-from',$source)}
 $child=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -PassThru -NoNewWindow
 $writer.WriteLine("trainer_pid=$($child.Id)");$writer.Flush();$child.WaitForExit();exit $child.ExitCode
}finally{if($writer){$writer.Dispose()};if($watchdog -and -not $watchdog.HasExited){Stop-Process -Id $watchdog.Id -ErrorAction SilentlyContinue};if(Test-Path $lock){Remove-Item $lock -Force}}
