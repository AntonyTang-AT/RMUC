param(
    [int]$TrainingPid = 20728,
    [string]$ProjectRoot = '',
    [string]$Python = 'D:\Pyhon3\python.exe'
)

$ErrorActionPreference = 'Stop'

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$runRoot = Join-Path $ProjectRoot 'out_m2\finetuned_temporal_visible_context_v4_qv_run2'
$adapter = Join-Path $runRoot 'adapter'
$meta = Join-Path $runRoot 'meta.json'
$logs = Join-Path $runRoot 'logs'
$smokeOutput = Join-Path $ProjectRoot 'out_m2\llm_v4_qv_run2_smoke'
$reliabilityOutput = Join-Path $ProjectRoot 'out_m2\llm_v4_qv_run2_reliability'
$reliabilityManifest = Join-Path $ProjectRoot 'out_m2\llm_reliability_holdout.json'
$waitLog = Join-Path $logs 'v4_smoke_after_training.log'
$smokeLog = Join-Path $logs 'v4_smoke.stdout.log'
$reliabilityLog = Join-Path $logs 'v4_reliability.stdout.log'

function Write-Status([string]$Message) {
    $line = "$(Get-Date -Format s) $Message"
    $line | Tee-Object -FilePath $waitLog -Append
}

Write-Status "waiting for training PID $TrainingPid before smoke evaluation"
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

if (-not (Test-Path -LiteralPath $adapter) -or -not (Test-Path -LiteralPath $meta)) {
    Write-Status 'training exited without final adapter/meta artifacts; smoke evaluation not started'
    exit 2
}
if (Test-Path -LiteralPath $smokeOutput) {
    Write-Status "smoke output already exists at $smokeOutput; refusing to overwrite"
    exit 3
}
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Status "python executable not found at $Python"
    exit 4
}

Write-Status 'final adapter detected; starting four-scenario visible-input smoke evaluation'
Set-Location -LiteralPath $ProjectRoot
$arguments = @(
    '-m', 'eval.evaluate_llm_candidate_matrix',
    '--adapter', $adapter,
    '--games', '1778661259223', '1778661970735', '1778663517697', '1778664469370',
    '--output-dir', $smokeOutput,
    '--device', 'cuda', '--limit', '10', '--sampling', 'uniform', '--batch-size', '4', '--min-games', '4'
)
& $Python @arguments 2>&1 | Tee-Object -FilePath $smokeLog -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Status "smoke evaluation failed with exit code $exitCode"
    exit $exitCode
}
Write-Status "smoke evaluation completed successfully at $smokeOutput"

$smokeAcceptance = Join-Path $smokeOutput 'acceptance.json'
if (-not (Test-Path -LiteralPath $smokeAcceptance)) {
    Write-Status 'smoke completed without acceptance.json; reliability evaluation not started'
    exit 5
}
$smokeAudit = Get-Content -LiteralPath $smokeAcceptance -Raw | ConvertFrom-Json
if ($smokeAudit.gates.structural_reliability -ne 'pass') {
    Write-Status 'smoke structural gate failed; reliability evaluation not started'
    exit 0
}
if (-not (Test-Path -LiteralPath $reliabilityManifest)) {
    Write-Status "reliability manifest missing at $reliabilityManifest"
    exit 6
}
if (Test-Path -LiteralPath $reliabilityOutput) {
    Write-Status "reliability output already exists at $reliabilityOutput; refusing to overwrite"
    exit 7
}

Write-Status 'smoke structural gate passed; starting 20-game reliability evaluation'
$reliabilityArguments = @(
    '-m', 'eval.evaluate_llm_candidate_matrix',
    '--adapter', $adapter,
    '--games-manifest', $reliabilityManifest,
    '--output-dir', $reliabilityOutput,
    '--device', 'cuda', '--limit', '12', '--sampling', 'uniform', '--batch-size', '4', '--min-games', '12'
)
& $Python @reliabilityArguments 2>&1 | Tee-Object -FilePath $reliabilityLog -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Status "reliability evaluation failed with exit code $exitCode"
    exit $exitCode
}
Write-Status "reliability evaluation completed successfully at $reliabilityOutput"
