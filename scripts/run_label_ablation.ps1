$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\_env.ps1"

Write-Host "Running label ablation (AFB x culture, fold 0 only)..."
Write-Host "This runs 9 label builds + 9 short training runs."

& $Python src/run_ablation.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. See artifacts/ablation_summary.md"
