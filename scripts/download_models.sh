#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$repo_root/MASt3R-SLAM-main/.git" ]]; then
  echo "MASt3R-SLAM-main is missing; run scripts/setup_external.sh first." >&2
  exit 1
fi
mkdir -p "$repo_root/checkpoints" "$repo_root/MASt3R-SLAM-main/checkpoints"

curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o "$repo_root/checkpoints/sam2.1_hiera_large.pt"

mast3r_base=https://download.europe.naverlabs.com/ComputerVision/MASt3R
curl -fL --retry 3 "$mast3r_base/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
  -o "$repo_root/MASt3R-SLAM-main/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
curl -fL --retry 3 "$mast3r_base/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" \
  -o "$repo_root/MASt3R-SLAM-main/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
curl -fL --retry 3 "$mast3r_base/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" \
  -o "$repo_root/MASt3R-SLAM-main/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"

echo "Model weights downloaded to ignored checkpoint directories."
