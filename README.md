# Persistent3D

Persistent3D is a 3D spatial-understanding pipeline for videos and multi-frame RGB scenes. It combines MASt3R-SLAM reconstruction, vision-language-model prompting, SAM 2 segmentation and propagation, cross-frame instance association, 3D point-cloud cleaning, and semantic oriented bounding boxes (OBBs).

The repository contains two project-owned pipeline entry points:

- `tools/Inference_video.py` processes one video file or one directory of RGB frames.
- `tools/inference_scan.py` processes one scene selected from a ScanNet-style multi-scene dataset.
- `data/scannet.py` optionally converts official ScanNet RGB and instance data into the `2D_segment/` format used by Persistent3D.

Datasets, generated outputs, third-party source trees, and model weights are intentionally excluded from the repository.

## Repository layout

```text
Persistent3D/
|-- tools/
|   |-- Inference_video.py       # Single-video or RGB-directory pipeline
|   `-- inference_scan.py        # Single-scene pipeline for a dataset root
|-- configs/
|   |-- video.json               # Example configuration for Inference_video.py
|   `-- scan.json                # Example configuration for inference_scan.py
|-- data/
|   `-- scannet.py               # Optional ScanNet instance-mask converter
|-- scripts/
|   |-- setup_external.sh/.ps1   # Clone, patch, and install external projects
|   |-- download_models.sh/.ps1  # Download model checkpoints
|   |-- run_pipeline.py          # Validate a JSON config and launch a pipeline
|   |-- run_video.sh/.ps1        # Run configs/video.json
|   `-- run_scan.sh/.ps1         # Run configs/scan.json
|-- patches/
|   `-- mast3r-slam-robospatial.patch
|-- .gitattributes
|-- .gitignore
|-- environment.yml
|-- README.md
`-- requirements.txt
```

The setup scripts create `sam2/` and `MASt3R-SLAM-main/` at the repository root. The download scripts create `checkpoints/` and populate `MASt3R-SLAM-main/checkpoints/`. These generated directories are ignored by Git.

## Required models and official sources

| Component | Purpose | Official source |
| --- | --- | --- |
| SAM 2.1 Hiera Large | Image segmentation and video-mask propagation | [Meta SAM 2 repository](https://github.com/facebookresearch/sam2) and [official Hiera Large checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| MASt3R-SLAM | Camera poses, depth, and dense 3D reconstruction | [MASt3R-SLAM repository](https://github.com/rmurai0610/MASt3R-SLAM) |
| MASt3R | 3D prior and retrieval models used by MASt3R-SLAM | [MASt3R repository](https://github.com/naver/mast3r); see its `Checkpoints` section |

No model weights are distributed in this repository. MASt3R-SLAM and some MASt3R checkpoints have non-commercial restrictions. Read the upstream licenses and checkpoint notices before use.

## 1. Create the environment

The recommended environment is Ubuntu 22.04 or WSL2 with Python 3.11 and an NVIDIA GPU. The default dependency set uses PyTorch 2.5.1.

With Conda:

```bash
conda env create -f environment.yml
conda activate persistent3d
```

With `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the default PyTorch wheel does not support your CUDA version, use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to install compatible builds of `torch==2.5.1` and `torchvision==0.20.1`, then install the remaining dependencies.

The optional `data/scannet.py` converter also requires `scikit-image`. Install it if it is not already available in the environment:

```bash
python -m pip install scikit-image
```

## 2. Install external source dependencies

On Linux or WSL:

```bash
bash scripts/setup_external.sh
```

In PowerShell:

```powershell
.\scripts\setup_external.ps1
```

The script clones `sam2/` and `MASt3R-SLAM-main/` from their official repositories, installs their Python packages, and applies `patches/mast3r-slam-robospatial.patch` to MASt3R-SLAM. The patch exports the RGB frames, depth maps, poses, keyframe list, scene point cloud, and point-cloud provenance required by Persistent3D.

MASt3R-SLAM officially targets Ubuntu. WSL users who switch to the upstream `windows` branch may need to adapt the patch to that branch.

## 3. Download model weights

On Linux or WSL:

```bash
bash scripts/download_models.sh
```

In PowerShell:

```powershell
.\scripts\download_models.ps1
```

The scripts use official download URLs and produce:

```text
checkpoints/sam2.1_hiera_large.pt
MASt3R-SLAM-main/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
MASt3R-SLAM-main/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
MASt3R-SLAM-main/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
```

All of these files are excluded by `.gitignore`.

## 4. Configure the vision-language model

Both pipelines require a local multimodal model service. Supply its endpoint, access token, and model identifier only through the private runtime environment or an external secret manager. Do not store these values in this repository, JSON configuration files, command-line arguments, shell scripts, logs, or documentation.

## 5. Prepare input data

`tools/Inference_video.py` accepts either a video file or a directory of RGB frames. The example `configs/video.json` uses the following directory:

```text
data/video_1/
|-- 000000.png
|-- 000001.png
`-- ...
```

`tools/inference_scan.py` accepts a ScanNet-style root with one directory per scene and RGB frames in each scene's `color/` directory. The example `configs/scan.json` selects scene `00035`:

```text
data/scannet/
|-- 00035/
|   `-- color/
|       |-- 000000.jpg
|       `-- ...
`-- another_scene/
    `-- color/
```

Input data is excluded from version control.

### Optional ScanNet instance-mask conversion

`data/scannet.py` converts one official ScanNet scene into Persistent3D-compatible 2D instance artifacts. The input root, or exactly one descendant directory under it, must contain:

```text
SCENE_ROOT/
|-- color_100/
|   `-- <frame-id>.jpg
|-- instance-filt_100/
|   `-- <frame-id>.png
`-- <scene-id>.aggregation.json
```

Run the converter with:

```bash
python data/scannet.py \
  --input-root data/raw_scannet/<scene-id> \
  --output-root outputs/<scene-id>
```

Use `--edge-radius N` to change the RGB-edge refinement radius from its default value of `3`. Add `--overwrite` to replace an existing `2D_segment/` directory:

```bash
python data/scannet.py \
  --input-root data/raw_scannet/<scene-id> \
  --output-root outputs/<scene-id> \
  --edge-radius 3 \
  --overwrite
```

The converter excludes structural background categories, preserves ScanNet aggregation identities across frames, refines instance boundaries with an edge-aware watershed, and writes:

```text
outputs/<scene-id>/
`-- 2D_segment/
    |-- instances_2d.json
    |-- merged_masks/
    |   `-- <frame-name>/mask_<index>.png
    `-- overlay/<frame-name>.overlay.jpg
```

## 6. Edit the run configurations

Update paths and runtime options in:

- `configs/video.json` for `tools/Inference_video.py`
- `configs/scan.json` for `tools/inference_scan.py`

The `script` field must match the current entry-point path exactly:

| Configuration | Required `script` value |
| --- | --- |
| `configs/video.json` | `tools/Inference_video.py` |
| `configs/scan.json` | `tools/inference_scan.py` |

The launcher passes `arguments` to the selected Python script in list order. Runtime service settings are intentionally rejected if placed in a JSON configuration.

## 7. Run the pipelines

Run the single-video pipeline:

```bash
bash scripts/run_video.sh
```

Run one scene from a multi-scene dataset:

```bash
bash scripts/run_scan.sh
```

PowerShell equivalents:

```powershell
.\scripts\run_video.ps1
.\scripts\run_scan.ps1
```

Additional command-line arguments are appended after configured arguments, so later duplicate options take precedence. For example:

```bash
bash scripts/run_scan.sh --scene 00061 --output_dir outputs/scan_00061
```

Validate a configuration and print the final command without starting the pipeline:

```bash
python scripts/run_pipeline.py configs/scan.json --print-command
```

Results are written under `outputs/`. A complete run can include the MASt3R reconstruction, 2D instance masks, cross-frame associations, object point clouds, semantic OBBs, and a timing report.

## Pipeline stages and output layout

Each entry point runs the same nine high-level stages:

1. Reconstruct the scene with MASt3R-SLAM and export RGB, depth, poses, keyframes, and scene points.
2. Generate and load object labels with the VLM.
3. Generate prompts, segment instances with SAM 2, and render 2D overlays.
4. Propagate masks and associate instances across frames.
5. Project the scene point cloud into each frame and extract object points.
6. Clean object point clouds and refresh related visualizations.
7. Infer semantic object orientation from multiple views.
8. Fit OBBs with category-aware geometric constraints.
9. Record scheduling and timing information not assigned to a preceding stage.

The default output structure is:

```text
OUTPUT_DIR/
|-- pipeline_timing.txt
|-- sequence/
|   |-- color/
|   |-- depth/
|   |-- camera_poses/
|   |-- keyframes.txt
|   |-- scene.ply
|   |-- scene.provenance.npz
|   `-- labels.json
|-- 2D_segment/
|-- instance_association/
|-- 3D_point/
|-- 3D_point_clean/
`-- VLM_OBB/
```

## Version-control scope

`.gitignore` excludes:

- `data/`, `datasets/`, `outputs/`, and logs
- `sam2/`, `MASt3R-SLAM-main/`, and other third-party source trees
- model weights such as `.pt`, `.pth`, `.gguf`, and `.safetensors`
- IDE settings, caches, generated build artifacts, and local runtime-secret files

Before publishing, inspect the repository with:

```bash
git status --short
git ls-files
```

The tracked project should contain the two pipeline entry points, their configurations, setup and run scripts, the MASt3R-SLAM patch, environment files, and this README.

The current `data/` ignore rule also excludes `data/scannet.py`. To publish this utility, either move it to `tools/` or add explicit `.gitignore` exceptions for the directory and file before staging it.
