#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -d sam2/.git ]]; then
  git clone https://github.com/facebookresearch/sam2.git sam2
fi

if [[ ! -d MASt3R-SLAM-main/.git ]]; then
  git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git MASt3R-SLAM-main
fi

if ! git -C MASt3R-SLAM-main apply --reverse --check "$repo_root/patches/mast3r-slam-robospatial.patch" >/dev/null 2>&1; then
  git -C MASt3R-SLAM-main apply --check "$repo_root/patches/mast3r-slam-robospatial.patch"
  git -C MASt3R-SLAM-main apply "$repo_root/patches/mast3r-slam-robospatial.patch"
fi

python -m pip install -r requirements.txt
python -m pip install -e ./sam2
python -m pip install -e ./MASt3R-SLAM-main/thirdparty/mast3r
python -m pip install -e ./MASt3R-SLAM-main/thirdparty/in3d
python -m pip install --no-build-isolation -e ./MASt3R-SLAM-main

echo "External source dependencies are installed. Run scripts/download_models.sh next."
