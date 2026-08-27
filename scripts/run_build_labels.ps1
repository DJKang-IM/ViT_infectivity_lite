$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

Write-Host "Building graded labels (D1-D5, D7)..."
& $Python src/labels/build_graded_labels.py `
  --afb-scheme afb_grade_v1 `
  --pcr-scheme pcr_soft_v1 `
  --solid-transform loginv `
  --liquid-transform twostep `
  --registry-mode union `
  --out artifacts/labels_v1.csv `
  --audit-out artifacts/labels_v1_audit.csv

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done."
