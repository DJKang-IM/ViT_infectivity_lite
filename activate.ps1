# Activate project venv (CUDA PyTorch). Run: . .\activate.ps1
$Root = $PSScriptRoot
$VenvActivate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $VenvActivate)) {
    Write-Host "Venv missing; running setup..."
    & (Join-Path $Root "scripts\setup_venv.ps1")
}
. $VenvActivate
Write-Host "Using: $(python -c 'import torch; print(torch.__version__, \"cuda=\"+str(torch.cuda.is_available()))')"
