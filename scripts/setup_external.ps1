[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath "sam2\.git")) {
    git clone https://github.com/facebookresearch/sam2.git sam2
}

if (-not (Test-Path -LiteralPath "MASt3R-SLAM-main\.git")) {
    git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git MASt3R-SLAM-main
}

$Patch = Join-Path $RepoRoot "patches\mast3r-slam-robospatial.patch"
git -C MASt3R-SLAM-main apply --reverse --check $Patch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C MASt3R-SLAM-main apply --check $Patch
    if ($LASTEXITCODE -ne 0) { throw "MASt3R-SLAM patch no longer applies cleanly." }
    git -C MASt3R-SLAM-main apply $Patch
}

python -m pip install -r requirements.txt
python -m pip install -e .\sam2
python -m pip install -e .\MASt3R-SLAM-main\thirdparty\mast3r
python -m pip install -e .\MASt3R-SLAM-main\thirdparty\in3d
python -m pip install --no-build-isolation -e .\MASt3R-SLAM-main

Write-Host "External source dependencies are installed. Run scripts\download_models.ps1 next."
