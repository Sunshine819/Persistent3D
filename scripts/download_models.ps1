[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$SamDir = Join-Path $RepoRoot "checkpoints"
$Mast3rDir = Join-Path $RepoRoot "MASt3R-SLAM-main\checkpoints"
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "MASt3R-SLAM-main\.git"))) {
    throw "MASt3R-SLAM-main is missing; run scripts\setup_external.ps1 first."
}
New-Item -ItemType Directory -Force -Path $SamDir, $Mast3rDir | Out-Null

Invoke-WebRequest `
    -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" `
    -OutFile (Join-Path $SamDir "sam2.1_hiera_large.pt")

$Mast3rBase = "https://download.europe.naverlabs.com/ComputerVision/MASt3R"
$Mast3rFiles = @(
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
)
foreach ($File in $Mast3rFiles) {
    Invoke-WebRequest -Uri "$Mast3rBase/$File" -OutFile (Join-Path $Mast3rDir $File)
}

Write-Host "Model weights downloaded to ignored checkpoint directories."
