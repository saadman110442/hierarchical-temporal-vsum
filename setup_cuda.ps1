# =============================================================================
#  setup_cuda.ps1 - Swap the venv from CPU-only PyTorch to a CUDA-enabled build
# -----------------------------------------------------------------------------
#  Run after .\setup.ps1 has succeeded.
#  From the vsum-web folder:
#      .\setup_cuda.ps1
#
#  What it does
#  ------------
#  1. Verifies backend\venv exists (setup.ps1 must have run first).
#  2. Uninstalls the CPU-only torch/torchvision that setup.ps1 installed.
#  3. Installs CUDA 12.1 wheels of torch 2.4.1 + torchvision 0.19.1.
#     (These wheels include sm_61 binaries that work on Pascal cards like
#      the GTX 1050 Ti.)
#  4. Runs a quick "torch.cuda.is_available()" probe and prints the result.
#
#  After this, run .\run.ps1 as usual - the server will autodetect CUDA and
#  the "GPU" button in the UI's compute picker will light up.
#
#  Optional args
#  -------------
#  -CudaVersion 121   default; matches CUDA 12.1 wheels
#  -CudaVersion 118   for CUDA 11.8 builds (older driver setups)
# =============================================================================

param(
    [string]$CudaVersion = "121"
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$venvPy  = Join-Path $projectRoot "backend\venv\Scripts\python.exe"
$venvPip = Join-Path $projectRoot "backend\venv\Scripts\pip.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[ERR]  backend\venv not found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " setup_cuda.ps1 - install CUDA $CudaVersion build of torch 2.4.1"      -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. nvidia-smi sanity check (informational only)
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    Write-Host "[INFO] nvidia-smi found - probing the driver..." -ForegroundColor Yellow
    & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Host "[WARN] nvidia-smi not on PATH. CUDA install may still work if drivers are present elsewhere." -ForegroundColor Yellow
}
Write-Host ""

# 2. Uninstall current torch/torchvision (CPU-only)
# -----------------------------------------------------------------------------
# PowerShell quirk: with $ErrorActionPreference = "Stop", any text pip prints to
# stderr (e.g. "Skipping torchaudio as it is not installed") is treated as a
# terminating error when merged via 2>&1. We temporarily relax that and discard
# stderr, since this step is informational anyway - pip uninstall is idempotent.
Write-Host "[INFO] Removing CPU-only torch/torchvision from the venv..." -ForegroundColor Yellow
$prevPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $venvPy -m pip uninstall -y torch torchvision 2>$null | Out-Null
& $venvPy -m pip uninstall -y torchaudio          2>$null | Out-Null
$ErrorActionPreference = $prevPref
Write-Host "[OK]   Old torch removed (if any was present)." -ForegroundColor Green
Write-Host ""

# 3. Install CUDA build
$cudaIndex = "https://download.pytorch.org/whl/cu$CudaVersion"
Write-Host "[INFO] Installing torch==2.4.1 and torchvision==0.19.1 from $cudaIndex" -ForegroundColor Yellow
Write-Host "       This download is ~2.5 GB. Be patient." -ForegroundColor Yellow
& $venvPy -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url $cudaIndex
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERR]  CUDA torch install failed." -ForegroundColor Red
    Write-Host "       Common causes:"                -ForegroundColor Red
    Write-Host "         - No matching wheel for your Python version (need 3.11)" -ForegroundColor Red
    Write-Host "         - Pip cache corruption: try --no-cache-dir"               -ForegroundColor Red
    Write-Host "         - Network blocks pytorch.org"                              -ForegroundColor Red
    Write-Host ""
    Write-Host "[INFO] Falling back: reinstalling CPU-only torch so the server still runs." -ForegroundColor Yellow
    & $venvPy -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
    exit 1
}
Write-Host ""

# 4. Probe
Write-Host "[INFO] Probing the new torch install for CUDA..." -ForegroundColor Yellow
$probe = @'
import torch, sys
print(f"torch        = {torch.__version__}")
print(f"is_available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  device {i}: {p.name}  (compute {p.major}.{p.minor})  "
              f"{free // (1024*1024)} / {total // (1024*1024)} MB free")
else:
    print("  no CUDA device detected - is the NVIDIA driver installed?")
    sys.exit(2)
'@
$probeFile = Join-Path $projectRoot "_cuda_probe.py"
$probe | Set-Content -Path $probeFile -Encoding UTF8
& $venvPy $probeFile
$probeCode = $LASTEXITCODE
Remove-Item $probeFile -Force -ErrorAction SilentlyContinue

Write-Host ""
if ($probeCode -eq 0) {
    Write-Host "[OK]   CUDA build ready. Start the server with: .\run.ps1" -ForegroundColor Green
    Write-Host "       In the UI, the GPU button in the compute picker will be enabled." -ForegroundColor Green
} else {
    Write-Host "[WARN] torch installed but CUDA is not visible to it." -ForegroundColor Yellow
    Write-Host "       Possible causes:"                                  -ForegroundColor Yellow
    Write-Host "         - NVIDIA driver missing or out of date"           -ForegroundColor Yellow
    Write-Host "         - No NVIDIA GPU on this machine"                  -ForegroundColor Yellow
    Write-Host "         - GPU disabled in Device Manager / BIOS"          -ForegroundColor Yellow
    Write-Host "       Update the driver: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    exit 1
}
Write-Host ""
