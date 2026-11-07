[CmdletBinding()]
param([switch]$DryRun, [switch]$Pilot, [string]$RunName = '', [double]$OptimizerPauseSeconds = 0, [int]$BoundedSteps = 0)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$root = (Resolve-Path -LiteralPath $root).Path
$python = if ($env:POSENERF_E8_PYTHON) { $env:POSENERF_E8_PYTHON } else { Join-Path $root '.venv\Scripts\python.exe' }
$runName = if ($RunName) { $RunName } elseif ($Pilot) { 'm13_e8b_pilot_10x512' } else { 'm13_e8b_10x512_12k' }
$destination = Join-Path $root (Join-Path 'workspace\outputs' $runName)
$lockPath = Join-Path $destination 'e8_training.lock'
$steps = if ($BoundedSteps -gt 0) { $BoundedSteps } elseif ($Pilot) { 500 } else { 12000 }

if (-not (Test-Path -LiteralPath $python)) { throw "Repository-local Python is unavailable: $python" }
if (Test-Path -LiteralPath $lockPath) { throw "E8 launcher lock exists: $lockPath. Refusing a duplicate launch." }
if ((Test-Path -LiteralPath (Join-Path $destination 'manifest.json')) -and -not $Pilot) { throw "E8 destination already exists: $destination. Refusing to overwrite a run." }

$arguments = @('-m','posenerf.stage_c','--project-dir','workspace','--run-name',$runName,'--steps',$steps,'--train-images','192','--batch-rays','256','--downsample','4','--hidden','512','--layers','10','--coarse-samples','48','--fine-samples','48','--depth-loss-weight','0.03','--accumulation-steps','4','--optimizer-pause-seconds',$OptimizerPauseSeconds,'--snapshot-steps',$(if ($Pilot) {'500'} elseif ($BoundedSteps -gt 0) {'500,1000,1500'} else {'2000,4000,6000,8000,10000,12000'}))
if ($DryRun) { Write-Output ($python + ' ' + ($arguments -join ' ')); exit 0 }

New-Item -ItemType Directory -Force -Path $destination | Out-Null
$lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
$writer = [System.IO.StreamWriter]::new($lock)
try {
    $writer.WriteLine("pid=$PID started_utc=$([DateTime]::UtcNow.ToString('o')) run=$runName microbatch_rays=256 accumulation_steps=4 effective_rays=1024")
    $writer.Flush()
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $python; $info.WorkingDirectory = $root; $info.UseShellExecute = $false; $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true; $info.CreateNoWindow = $true
    $info.Arguments = (($arguments | ForEach-Object { if ($_ -match '[\s\"]') { '"' + ($_ -replace '(\\*)"', '$1$1\\"') + '"' } else { $_ } }) -join ' ')
    $child = [System.Diagnostics.Process]::new(); $child.StartInfo = $info
    if (-not $child.Start()) { throw 'Unable to start the E8 Python child process.' }
    $writer.WriteLine("trainer_pid=$($child.Id) trainer_exe=$python")
    $writer.Flush()
    $stdoutTask = $child.StandardOutput.ReadToEndAsync(); $stderrTask = $child.StandardError.ReadToEndAsync(); $child.WaitForExit()
    $logPath = Join-Path $destination 'e8.log'; $stdout = $stdoutTask.GetAwaiter().GetResult(); $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($stdout) { Add-Content -LiteralPath $logPath -Value $stdout; Write-Output $stdout }
    if ($stderr) { Add-Content -LiteralPath $logPath -Value $stderr; [Console]::Error.WriteLine($stderr) }
    if ($child.ExitCode -ne 0) { exit $child.ExitCode }
} finally {
    if ($writer) { $writer.Dispose() }
    if ($lock) { $lock.Dispose() }
    Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
