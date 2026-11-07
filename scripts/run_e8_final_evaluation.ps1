param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [ValidateSet("E3_final", "E6_final", "E7_final", "E8_final")]
    [string]$Label,
    [int[]]$FrameIndices
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$evaluator = Join-Path $PSScriptRoot "evaluate_e8_final.py"
$output = [IO.Path]::GetFullPath($OutputDir)
if (-not (Test-Path -LiteralPath $output)) { New-Item -ItemType Directory -Path $output -Force | Out-Null }
$stdout = Join-Path $output "evaluation_stdout.log"
$stderr = Join-Path $output "evaluation_stderr.log"
$exitArtifact = Join-Path $output "evaluation_exit_code.txt"

try {
    $arguments = @($evaluator, "--workspace", (Join-Path $repo "workspace"), "--output", $output)
    if ($Label) { $arguments += @("--label", $Label) }
    if ($FrameIndices) { $arguments += "--frame-indices"; $arguments += $FrameIndices }
    & $python @arguments 1>$stdout 2>$stderr
    $code = $LASTEXITCODE
} catch {
    $_ | Out-File -LiteralPath $stderr -Append -Encoding utf8
    $code = 1
}
Set-Content -LiteralPath $exitArtifact -Value $code -Encoding ascii
exit $code
