$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

if (-not (Test-Path "artifacts/labels_v1.csv")) {
    Write-Host "labels_v1.csv not found; running build first..."
    & "$PSScriptRoot/run_build_labels.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path "D:\[260626] ViT_Infectivity")) {
    Write-Host "DICOM work dir missing; preparing from RAW..."
    & "$PSScriptRoot/prepare_dicom_gangnam.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Training ViT v1 (5-fold CV)..."
& $Python src/train.py --config configs/default.yaml --tag v1_default

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. See artifacts/v1_v1_default/metrics.json"
