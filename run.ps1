# =============================================================================
#  run.ps1 - Start the Video Summarizer web server
# -----------------------------------------------------------------------------
#  Activates backend\venv and launches uvicorn on http://localhost:8000
#
#  Usage (from the vsum-web folder):
#      .\run.ps1
#
#  Optional: pass a different checkpoint path
#      .\run.ps1 -Checkpoint ".\models\my_model.pt"
#
#  Optional: use a different port
#      .\run.ps1 -Port 8001
# =============================================================================

param(
    [string]$Checkpoint = "",
    [int]$Port          = 8000
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$backendDir  = Join-Path $projectRoot "backend"
$venvPy      = Join-Path $backendDir "venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[ERR]  backend\venv not found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

Set-Location $backendDir

# -----------------------------------------------------------------------------
# HuggingFace cache → project drive (not C:\)
# -----------------------------------------------------------------------------
# BLIP-2 is ~12 GB; CLIP is ~600 MB. Without this, they land in
#   %USERPROFILE%\.cache\huggingface  (i.e. C:\Users\...)
# which fills the system drive. Point every HF lookup at backend\hf_cache
# instead. Using the same drive as the project means downloads go to D:, H:,
# or wherever this project lives.
# -----------------------------------------------------------------------------
$hfCache = Join-Path $backendDir "hf_cache"
if (-not (Test-Path $hfCache)) {
    New-Item -ItemType Directory -Path $hfCache -Force | Out-Null
}
$env:HF_HOME                = $hfCache
$env:HUGGINGFACE_HUB_CACHE  = Join-Path $hfCache "hub"
$env:TRANSFORMERS_CACHE     = Join-Path $hfCache "hub"   # legacy var, some code still reads it
Write-Host "[INFO] HF_HOME = $hfCache" -ForegroundColor Yellow

if ($Checkpoint -ne "") {
    $env:CHECKPOINT_PATH = $Checkpoint
    Write-Host "[INFO] Using checkpoint: $Checkpoint" -ForegroundColor Yellow
}

Write-Host "[INFO] Starting uvicorn on http://localhost:$Port ..." -ForegroundColor Yellow
Write-Host "[INFO] Press Ctrl+C to stop."                          -ForegroundColor Yellow
Write-Host ""

& $venvPy -m uvicorn main:app --host 0.0.0.0 --port $Port
