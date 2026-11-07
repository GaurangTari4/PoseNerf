[CmdletBinding()]
param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = if ($env:POSENERF_E7_PYTHON) { $env:POSENERF_E7_PYTHON } else { Join-Path $root '.venv\Scripts\python.exe' }
$source = if ($env:POSENERF_E7_SOURCE) { $env:POSENERF_E7_SOURCE } else { Join-Path $root 'workspace\outputs\m5b_e6_capacity_8x384_12k\checkpoints\hierarchical_nerf_resume_latest.pth' }
$destination = if ($env:POSENERF_E7_DESTINATION) { $env:POSENERF_E7_DESTINATION } else { Join-Path $root 'workspace\outputs\m12_e7_extended_8x384_48k' }
$lockPath = Join-Path $destination 'e7_training.lock'
$expectedHash = if ($env:POSENERF_E7_EXPECTED_HASH) { $env:POSENERF_E7_EXPECTED_HASH } else { 'E0255AC301E0A173F1256C2024B3F262BEBEB9FB9EA195A12ED3CC6767EB1669' }
$snapshots = '14000,16000,18000,20000,24000,28000,32000,36000,40000,44000,48000'

if (-not (Test-Path -LiteralPath $python)) { throw 'Repository-local Python is unavailable.' }
if (-not (Test-Path -LiteralPath $source)) { throw "E6 source checkpoint is unavailable: $source" }
if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $expectedHash) { throw 'E6 source checkpoint SHA-256 mismatch.' }

$resume = Join-Path $destination 'checkpoints\hierarchical_nerf_resume_latest.pth'
$resumeFrom = if (Test-Path -LiteralPath $resume) { $resume } else { $source }
$arguments = @(
    '-m', 'posenerf.stage_c', '--project-dir', 'workspace', '--run-name', 'm12_e7_extended_8x384_48k',
    '--steps', '48000', '--resume', '--resume-from', $resumeFrom,
    '--scheduler-policy', 'controlled_extension_scheduler', '--snapshot-steps', $snapshots
)
if ($env:POSENERF_E7_TEST_PYTHON_CODE) {
    # Test-only hook: exercises native stderr/exit handling without invoking
    # Stage C or creating a training checkpoint.
    $arguments = @('-c', $env:POSENERF_E7_TEST_PYTHON_CODE)
}

if ($DryRun) {
    Write-Output ('Source SHA-256 verified: ' + $expectedHash)
    Write-Output ('Resume source: ' + $resumeFrom)
    Write-Output ($python + ' ' + ($arguments -join ' '))
    exit 0
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
$lock = $null
$writer = $null
$ownsLock = $false
try {
    # CreateNew makes duplicate runs fail closed. A stale lock requires an
    # explicit operator decision rather than an unsafe automatic override.
    $lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    $ownsLock = $true
} catch {
    throw "An E7 launcher lock exists: $lockPath. Refusing a duplicate or ambiguous restart."
}

try {
    $writer = [System.IO.StreamWriter]::new($lock)
    $writer.WriteLine("pid=$PID started_utc=$([DateTime]::UtcNow.ToString('o')) resume_from=$resumeFrom")
    $writer.Flush()
    if ($env:POSENERF_E7_TEST_EXIT_CODE) {
        $exitCode = [int]$env:POSENERF_E7_TEST_EXIT_CODE
        "Test-only simulated child exit: $exitCode" | Tee-Object -FilePath (Join-Path $destination 'e7.log') -Append
    } else {
        # Do not use `2>&1 | Tee-Object`: Windows PowerShell 5.1 can convert
        # harmless native stderr into a terminating NativeCommandError when
        # ErrorActionPreference is Stop.  Capture streams independently and
        # return only the child's actual exit code.
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $python
        $info.UseShellExecute = $false
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $info.CreateNoWindow = $true
        $info.Arguments = (($arguments | ForEach-Object {
            if ($_ -match '[\s\"]') { '"' + ($_ -replace '(\\*)"', '$1$1\\"') + '"' } else { $_ }
        }) -join ' ')
        $child = New-Object System.Diagnostics.Process
        $child.StartInfo = $info
        if (-not $child.Start()) { throw 'Unable to start the Python child process.' }
        $stdoutTask = $child.StandardOutput.ReadToEndAsync()
        $stderrTask = $child.StandardError.ReadToEndAsync()
        $child.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $logPath = Join-Path $destination 'e7.log'
        if ($stdout) { Add-Content -LiteralPath $logPath -Value $stdout; Write-Output $stdout }
        if ($stderr) { Add-Content -LiteralPath $logPath -Value $stderr; [Console]::Error.WriteLine($stderr) }
        $exitCode = $child.ExitCode
        $child.Dispose()
    }
    if ($exitCode -ne 0) { exit $exitCode }
} finally {
    if ($writer) { $writer.Dispose() } elseif ($lock) { $lock.Dispose() }
    if ($ownsLock) { Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue }
}
