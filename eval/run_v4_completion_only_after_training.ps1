param(
    [int]$TrainingPid,
    [string]$ProjectRoot = '',
    [string]$Python = 'D:\Pyhon3\python.exe'
)

$ErrorActionPreference = 'Stop'

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$runRoot = Join-Path $ProjectRoot 'out_m2\finetuned_temporal_visible_context_v4_completion_only_run1'
$adapter = Join-Path $runRoot 'adapter'
$meta = Join-Path $runRoot 'meta.json'
$logs = Join-Path $runRoot 'logs'
$smokeOutput = Join-Path $ProjectRoot 'out_m2\llm_v4_completion_only_run1_smoke'
$reliabilityOutput = Join-Path $ProjectRoot 'out_m2\llm_v4_completion_only_run1_reliability'
$reliabilityManifest = Join-Path $ProjectRoot 'out_m2\llm_reliability_holdout.json'
$waitLog = Join-Path $logs 'completion_only_after_training.log'
$smokeLog = Join-Path $logs 'completion_only_smoke.stdout.log'
$reliabilityLog = Join-Path $logs 'completion_only_reliability.stdout.log'
$deploymentLog = Join-Path $logs 'completion_only_deployment_audit.stdout.log'
$packageLog = Join-Path $logs 'completion_only_package.stdout.log'
$packageOutput = Join-Path $ProjectRoot 'out_m2\acceptance_package_final'

function Write-Status([string]$Message) {
    $line = "$(Get-Date -Format s) $Message"
    $line | Tee-Object -FilePath $waitLog -Append
}

Write-Status "waiting for training PID $TrainingPid before candidate evaluation"
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

if (-not (Test-Path -LiteralPath $adapter) -or -not (Test-Path -LiteralPath $meta)) {
    Write-Status 'training exited without final adapter/meta artifacts; evaluation not started'
    exit 2
}
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Status "python executable not found at $Python"
    exit 3
}
if (Test-Path -LiteralPath $smokeOutput) {
    Write-Status "smoke output already exists at $smokeOutput; refusing to overwrite"
    exit 4
}

Set-Location -LiteralPath $ProjectRoot
Write-Status 'final adapter detected; starting four-scenario visible-input smoke evaluation'
$smokeArguments = @(
    '-m', 'eval.evaluate_llm_candidate_matrix',
    '--adapter', $adapter,
    '--games', '1778661259223', '1778661970735', '1778663517697', '1778664469370',
    '--output-dir', $smokeOutput,
    '--device', 'cuda', '--limit', '10', '--sampling', 'uniform', '--batch-size', '4', '--min-games', '4',
    '--min-mode-rows', '12', '--min-mode-games', '4'
)
& $Python @smokeArguments 2>&1 | Tee-Object -FilePath $smokeLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Status "smoke evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Status "smoke evaluation completed successfully at $smokeOutput"

$smokeAcceptance = Join-Path $smokeOutput 'acceptance.json'
if (-not (Test-Path -LiteralPath $smokeAcceptance)) {
    Write-Status 'smoke completed without acceptance.json; reliability evaluation not started'
    exit 5
}
$smokeAudit = Get-Content -LiteralPath $smokeAcceptance -Raw | ConvertFrom-Json
if ($smokeAudit.gates.structural_reliability -ne 'pass') {
    Write-Status 'smoke runtime-structural gate failed; reliability evaluation not started'
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

Write-Status 'smoke runtime-structural gate passed; starting 20-game reliability evaluation'
$reliabilityArguments = @(
    '-m', 'eval.evaluate_llm_candidate_matrix',
    '--adapter', $adapter,
    '--games-manifest', $reliabilityManifest,
    '--output-dir', $reliabilityOutput,
    '--device', 'cuda', '--limit', '12', '--sampling', 'uniform', '--batch-size', '4', '--min-games', '12',
    '--min-mode-rows', '12', '--min-mode-games', '12'
)
& $Python @reliabilityArguments 2>&1 | Tee-Object -FilePath $reliabilityLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Status "reliability evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Status "reliability evaluation completed successfully at $reliabilityOutput"

$reliabilityAcceptance = Join-Path $reliabilityOutput 'acceptance.json'
if (-not (Test-Path -LiteralPath $reliabilityAcceptance)) {
    Write-Status 'reliability evaluation completed without acceptance.json; deployment audit and package not started'
    exit 8
}
$reliabilityAudit = Get-Content -LiteralPath $reliabilityAcceptance -Raw | ConvertFrom-Json
if ($reliabilityAudit.gates.structural_reliability -ne 'pass' -or
    $reliabilityAudit.gates.semantic_mode_coverage -ne 'pass' -or
    $reliabilityAudit.gates.semantic_hdq_by_global_mode_vs_legacy_rule -ne 'pass' -or
    $reliabilityAudit.gates.semantic_hdq_vs_legacy_rule -ne 'pass') {
    Write-Status ('reliability promotion gates failed: structural={0}; mode_coverage={1}; per_mode_hdq={2}; semantic_hdq={3}; ' +
                  'deployment audit and package not started' -f
                  $reliabilityAudit.gates.structural_reliability,
                  $reliabilityAudit.gates.semantic_mode_coverage,
                  $reliabilityAudit.gates.semantic_hdq_by_global_mode_vs_legacy_rule,
                  $reliabilityAudit.gates.semantic_hdq_vs_legacy_rule)
    exit 0
}

Write-Status 'reliability promotion gates passed; refreshing deployment readiness audit'
$deploymentArguments = @(
    '-m', 'eval.audit_deployment_readiness',
    '--llm-acceptance', $reliabilityAcceptance
)
& $Python @deploymentArguments 2>&1 | Tee-Object -FilePath $deploymentLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Status "deployment readiness audit failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if (Test-Path -LiteralPath $packageOutput) {
    Write-Status "acceptance package output already exists at $packageOutput; refusing to overwrite"
    exit 9
}
Write-Status 'deployment readiness audit completed; building local acceptance package without model weights'
$packageArguments = @(
    'scripts/build_acceptance_package.py',
    '--output', $packageOutput,
    '--llm-acceptance', $reliabilityAcceptance
)
& $Python @packageArguments 2>&1 | Tee-Object -FilePath $packageLog -Append
if ($LASTEXITCODE -ne 0) {
    Write-Status "acceptance package build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Status "acceptance package completed successfully at $packageOutput"
