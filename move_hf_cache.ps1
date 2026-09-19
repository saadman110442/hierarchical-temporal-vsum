# =============================================================================
#  move_hf_cache.ps1 - Move HuggingFace cache from C:\ into the project folder
# -----------------------------------------------------------------------------
#  Run once, from the vsum-web folder:
#      .\move_hf_cache.ps1
#
#  What it does
#  ------------
#  1. Computes the current HF cache location (default:
#       %USERPROFILE%\.cache\huggingface\hub)
#     and the project-local destination:
#       <project>\backend\hf_cache\hub
#  2. Moves every model snapshot already there into the project folder.
#  3. Scans the moved snapshots for `*.incomplete` / `tmp_*` files, which HF
#     uses for partial downloads. Any model with an incomplete file gets its
#     entire snapshot folder DELETED so it will re-download from scratch the
#     next time the server runs.
#  4. Leaves HF_HOME wiring to run.ps1 for future runs.
#
#  After this script finishes, use .\run.ps1 as usual - HF_HOME is already
#  wired in run.ps1 to point at backend\hf_cache.
# =============================================================================

param(
    [switch]$DryRun,
    [string]$Source = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$backendDir  = Join-Path $projectRoot "backend"
$destHome    = Join-Path $backendDir "hf_cache"
$destHub     = Join-Path $destHome "hub"

if ($Source -eq "") {
    # Where HF has been caching so far (the default if HF_HOME was never set)
    $sourceHome = Join-Path $env:USERPROFILE ".cache\huggingface"
} else {
    $sourceHome = $Source
}
$sourceHub = Join-Path $sourceHome "hub"

Write-Host ""
Write-Host "[plan] source: $sourceHome" -ForegroundColor Cyan
Write-Host "[plan] dest:   $destHome"   -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $sourceHub)) {
    Write-Host "[info] No existing HF cache at $sourceHub - nothing to move." -ForegroundColor Yellow
    Write-Host "[info] Future downloads will land in $destHub (run.ps1 sets HF_HOME)." -ForegroundColor Yellow
    exit 0
}

# Make sure destination exists
if (-not (Test-Path $destHub)) {
    New-Item -ItemType Directory -Path $destHub -Force | Out-Null
}

# Each model lives in a directory like:
#   hub\models--openai--clip-vit-base-patch32\
#   hub\models--Salesforce--blip2-opt-2.7b\
$modelDirs = Get-ChildItem -Path $sourceHub -Directory -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -like "models--*" }

if ($modelDirs.Count -eq 0) {
    Write-Host "[info] No model folders in $sourceHub." -ForegroundColor Yellow
    exit 0
}

foreach ($m in $modelDirs) {
    $name        = $m.Name
    $destPath    = Join-Path $destHub $name
    $isIncomplete = $false

    # Look for partial-download markers. HF uses `.incomplete` for interrupted
    # blob downloads and `tmp*` for in-flight writes.
    $incomplete = Get-ChildItem -Path $m.FullName -Recurse -Force -ErrorAction SilentlyContinue |
                  Where-Object {
                      $_.Name -like "*.incomplete" -or
                      $_.Name -like "tmp_*"        -or
                      $_.Name -like "*.part"
                  }
    if ($incomplete) {
        $isIncomplete = $true
    }

    $sizeMB = [math]::Round(
        (Get-ChildItem $m.FullName -Recurse -Force -ErrorAction SilentlyContinue |
         Measure-Object -Property Length -Sum).Sum / 1MB, 1
    )

    if ($isIncomplete) {
        $msg = "[skip] $name  (incomplete, " + $sizeMB + " MB) -> deleting, will re-download"
        Write-Host $msg -ForegroundColor Red
        if (-not $DryRun) {
            # Delete both source (partial) and any pre-existing dest so HF
            # starts clean. HF is happy to re-download straight into dest.
            Remove-Item $m.FullName -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path $destPath) {
                Remove-Item $destPath -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        continue
    }

    if (Test-Path $destPath) {
        Write-Host "[keep] $name  already at dest - safe to delete source later" -ForegroundColor DarkGray
        continue
    }

    $msg2 = "[move] $name  (" + $sizeMB + " MB) -> $destPath"
    Write-Host $msg2 -ForegroundColor Green
    if (-not $DryRun) {
        try {
            Move-Item -Path $m.FullName -Destination $destPath -Force
        } catch {
            Write-Host "       Move failed (cross-drive?), falling back to robocopy..." -ForegroundColor Yellow
            $null = robocopy $m.FullName $destPath /E /NFL /NDL /NJH /NJS /NP /MT:8
            Remove-Item $m.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Also move any refs / version.txt metadata siblings
Get-ChildItem $sourceHub -Force -ErrorAction SilentlyContinue |
  Where-Object { -not $_.PSIsContainer -or ($_.Name -eq "refs") } |
  ForEach-Object {
      $destPathSibling = Join-Path $destHub $_.Name
      if (-not (Test-Path $destPathSibling)) {
          if (-not $DryRun) { Move-Item $_.FullName $destPathSibling -Force -ErrorAction SilentlyContinue }
          Write-Host ("[move] " + $_.Name) -ForegroundColor DarkGreen
      }
  }

Write-Host ""
Write-Host "[done] Source folder remnants: $sourceHome" -ForegroundColor Cyan
Write-Host "       You can delete it manually once you have confirmed things load." -ForegroundColor Cyan
Write-Host "[done] Now run:  .\run.ps1"                  -ForegroundColor Green
Write-Host ""
