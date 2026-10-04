param(
    [Parameter(Mandatory = $true)][int]$TrainingPid,
    [Parameter(Mandatory = $true)][string]$CandidateRoot,
    [Parameter(Mandatory = $true)][string]$SmokeOutput,
    [Parameter(Mandatory = $true)][string]$ReliabilityOutput,
    [string]$ReliabilityManifest = 'out_m2/llm_reliability_holdout.json',
    [string]$Python = 'D:\Pyhon3\python.exe',
    [int]$Limit = 10
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$candidate = Join-Path $project $CandidateRoot
$adapter = Join-Path $candidate 'adapter'
$meta = Join-Path $candidate 'meta.json'
$logDir = Join-Path $candidate 'logs'
$log = Join-Path $logDir 'candidate_after_training.log'
$smokeLog = Join-Path $logDir 'candidate_smoke.stdout.log'
$reliabilityLog = Join-Path $logDir 'candidate_reliability.stdout.log'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
function Status([string]$message) {
    "$(Get-Date -Format s) $message" | Tee-Object -FilePath $log -Append
}

Status "waiting for training PID $TrainingPid"
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
if (-not (Test-Path $adapter) -or -not (Test-Path $meta)) {
    Status 'training exited without adapter/meta; evaluation stopped'
    exit 2
}
if (-not (Test-Path $Python)) { throw "python executable not found: $Python" }
$smoke = Join-Path $project $SmokeOutput
$reliability = Join-Path $project $ReliabilityOutput
if (Test-Path $smoke) { throw "refusing to overwrite smoke output: $smoke" }

Set-Location $project
Status 'starting four-scenario smoke evaluation'
& $Python -m eval.evaluate_llm_candidate_matrix `
    --adapter $adapter `
    --games 1778661259223 1778661970735 1778663517697 1778664469370 `
    --output-dir $smoke `
    --device cuda --limit $Limit --sampling uniform --batch-size 4 --min-games 4 `
    --min-mode-rows 6 --min-mode-games 4 2>&1 | Tee-Object -FilePath $smokeLog -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$smokeAcceptance = Join-Path $smoke 'acceptance.json'
if (-not (Test-Path $smokeAcceptance)) { throw 'smoke did not write acceptance.json' }
$smokeAudit = Get-Content $smokeAcceptance -Raw | ConvertFrom-Json
if ($smokeAudit.gates.structural_reliability -ne 'pass') {
    Status ('smoke structural gate failed; stopping before reliability: ' +
        ($smokeAudit.gates | Out-String).Trim())
    exit 0
}
if (Test-Path $reliability) { throw "refusing to overwrite reliability output: $reliability" }
Status 'smoke structural gate passed; starting reliability evaluation'
& $Python -m eval.evaluate_llm_candidate_matrix `
    --adapter $adapter `
    --games-manifest (Join-Path $project $ReliabilityManifest) `
    --output-dir $reliability `
    --device cuda --limit 12 --sampling uniform --batch-size 4 --min-games 12 `
    --min-mode-rows 12 --min-mode-games 12 2>&1 | Tee-Object -FilePath $reliabilityLog -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Status 'candidate evaluation completed'
