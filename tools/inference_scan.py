"""
Pipeline stages:
1. mast3r_slam_and_export: reconstruct with MASt3R-SLAM, locate the result,
   and copy RGB, depth, poses, and the scene point cloud into sequence/.
2. label_generation: collect input frames, generate labels.json with the VLM,
   and load the category list.
3. vlm_sam2_segmentation: initialize the VLM and SAM 2, generate prompts,
   segment instances, and save masks and 2D visualizations.
4. instance_association: propagate SAM 2 masks across frames, associate
   instances, and visualize the associations.
5. 3d_points_project: project scene.ply into each frame and extract 3D points
   through object masks.
6. 3d_points_clean: clean object point clouds and refresh the cleaned 2D and
   association visualizations.
7. vlm_object_orientation: infer semantic object orientation from multiple
   views; this stage excludes subsequent OBB refitting time.
8. obb_generation: prepare OBB data, apply category-aware geometric
   constraints, refit boxes, and write OBB JSON files.
9. time_tracking: record scheduling, timing, and miscellaneous overhead not
   covered by the preceding stages.

OUTPUT_DIR/
|-- pipeline_timing.txt
|-- sequence/
|   |-- color/<frame files>
|   |-- depth/<frame files>
|   |-- camera_poses/_info.txt
|   |-- camera_poses/<per-frame pose files>
|   |-- keyframes.txt
|   |-- scene.ply
|   |-- scene.provenance.npz
|   `-- labels.json
|-- 2D_segment/
|   |-- instances_2d.json
|   |-- merged_masks/<frame_name>/mask_XXX.png
|   `-- overlay/<frame_name>.overlay.jpg
|-- instance_association/
|   |-- 3D_segment.json
|   |-- instances_2d_with_association.json
|   `-- association_overlay/<frame_name>.association_overlay.jpg
|-- 3D_point/
|   |-- object_3d_points.json
|   `-- object_3d_points/<object_id>.xyz and <object_id>.ply
|-- 3D_point_clean/
|   |-- object_3d_points_clean.json
|   `-- object_3d_points_clean/<object_id>.xyz and <object_id>.ply
`-- VLM_OBB/
    |-- object_3d_obbs.json
    `-- <object_id>_3D_OBB.json
"""
from __future__ import annotations
import os
from pathlib import Path
_THIS_SCRIPT_DIR = Path(__file__).resolve().parent
def _find_this_repo_root(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "tools").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return script_dir.parent.resolve()
_THIS_REPO_ROOT = _find_this_repo_root(_THIS_SCRIPT_DIR)

# Global execution controls
MAX_FRAMES = None
DEVICE = "cuda:0"

# Pipeline input.  A ScanNet root is expected to contain one directory per
# scene, with RGB frames under <scene>/color.  Only those color directories are
# passed to MASt3R; sibling folders such as depth, pose, or previous outputs are
# deliberately ignored.
INPUT_RGB_DATASET = _THIS_REPO_ROOT / "data" / "scannet"
# Process exactly one scene per invocation.  Change this scene directory name
# before each run; the pipeline will never continue to the other scenes.
# Available local scenes: 00003, 00035, 00036, 00061, 00701.
INPUT_SCENE = "00035"

# Pipeline output
OUTPUT_DIR = _THIS_REPO_ROOT / "outputs" / "scan_00035"

# Step 1: MASt3R-SLAM reconstruction
MAST3R_SLAM_ROOT = _THIS_REPO_ROOT / "MASt3R-SLAM-main"
MAST3R_CONFIG = MAST3R_SLAM_ROOT / "config" / "base.yaml"
MAST3R_CALIB = Path("")  
MAST3R_SAVE_AS = "mast3r_2D_3_association_3Dpoint_tmp"
MAST3R_SCAN_ROOT = Path("")
MAST3R_SHOW_VISUALIZATION = False
POSE_TYPE = "camera_to_world"
DEPTH_SHIFT = 1.0

# Step 2: model service and label generation. Runtime connection details are
# read only from the process environment and are never stored in this project.
API_BASE_URL = os.environ.get("PERSISTENT3D_SERVICE_URL")
API_KEY = os.environ.get("PERSISTENT3D_SERVICE_TOKEN")
SERVICE_MODEL = os.environ.get("PERSISTENT3D_SERVICE_MODEL")
VLM_CONCURRENCY = 4
DEFAULT_VLM_MAX_RETRIES = 3
DEFAULT_VLM_LABEL_REQUEST_TIMEOUT = 120
DEFAULT_VLM_RETRY_DELAY = 1.0
DEFAULT_VLM_TEMPERATURE = 0.0
VLM_LABEL_STAGE1_PROMPT = "List the name of all visible objects in the image."
VLM_LABEL_STAGE2_PROMPT = """
Based on the objects you listed above, filter down to only those that are visually distinct and occupy a noticeable spatial area in the image.
Omit objects that are too thin.
If objects of the same type appear multiple times, replace those objects by the object's plural form.
Organise the remaining objects into a JSON array of strings.
""".strip()

# Step 3: VLM-guided SAM2 instance segmentation and 2D visualization
SAM2_CHECKPOINT = _THIS_REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
COORD_RANGE = 1000
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_OVERLAY_ALPHA = 0.45
DEFAULT_OVERLAY_DRAW_OUTLINE = False
DEFAULT_OVERLAY_MIN_AREA = 1
DEFAULT_SAM2_MASK_THRESHOLD = 0.0
DEFAULT_PIPELINE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_PIPELINE_OUTPUT_FORMAT = "parquet"

# Step 4: cross-frame instance association and scene-color merging
ASSOCIATION_WINDOW_FRAMES = 20
ASSOCIATION_IOU = 0.30
ASSOCIATION_SAME_LABEL = False
ASSOCIATION_SAME_FRAME_EXCLUSIVITY = True
SCENE_COLOR_MERGE_COLOR_SIMILARITY_THRESHOLD = 0.5
SCENE_COLOR_MERGE_INDEX_OVERLAP_THRESHOLD = 0.50

# Step 5: 2D-mask to 3D-point projection
DEFAULT_ENABLE_MASK_EROSION = True
DEFAULT_MASK_ERODE_PIXELS = 3
DEFAULT_DEPTH_SHIFT = 1000.0
POINT3D_SEED_EXPANSION_RADIUS_M = 0.05
DEFAULT_MIN_DEPTH_M = 1e-6
DEFAULT_MAX_DEPTH_M = 20.0
DEFAULT_SAMPLE_STRIDE = 1
DEFAULT_MAX_POINTS_PER_OBJECT = 0

# Step 6: 3D-point cleaning
DEFAULT_AREA_RATIO_THRESHOLD = 3.0
DEFAULT_CENTROID_DISTANCE_M = 1.0
DEFAULT_CENTROID_MAD_MULTIPLIER = 3.5
DEFAULT_EXTENT_RATIO_THRESHOLD = 3.0
DEFAULT_MIN_INSTANCE_POINTS = 20
DEFAULT_MIN_KEEP_INSTANCES = 1
DEFAULT_CLEAN_MODE = "preserve"
DEFAULT_SPATIAL_CLUSTER_INSTANCES = False
DEFAULT_CLUSTER_DISTANCE_M = 1.5
DEFAULT_CLUSTER_MIN_INSTANCES = 1
DEFAULT_SPLIT_SPATIAL_CLUSTERS = False
DEFAULT_VOXEL_SIZE = 0.02
DEFAULT_NB_NEIGHBORS = 16
DEFAULT_STD_RATIO = 1.5
DEFAULT_RADIUS = 0.05
DEFAULT_MIN_POINTS_IN_RADIUS = 4
DEFAULT_MIN_POINTS_AFTER_FILTER = 20
DEFAULT_MAX_POINTS_FOR_KNN = 6000

# Step 7: multi-frame VLM semantic-front orientation
DEFAULT_OBB_VLM_MAX_RETRIES = 3
DEFAULT_OBB_VLM_REQUEST_TIMEOUT = 120.0
DEFAULT_OBB_VLM_RETRY_SLEEP = 5.0
DEFAULT_OBB_VLM_CROP_MARGIN_RATIO = 0.25
DEFAULT_OBB_VLM_ORIENTATION_MIN_CONFIDENCE = 0.35
DEFAULT_OBB_VLM_MAX_REPRESENTATIVE_FRAMES = 10
VALID_FRONT_DIRECTIONS = {"left", "right", "up", "down", "toward_camera", "away_camera", "unknown"}
VLM_ORIENTATION_GENERAL_PROMPT = """
You are determining the semantic front direction of the target object.
The semantic front is the side used for human interaction, viewing, opening, sitting, typing, operation, reading, or approach.
Do not choose a direction only because one side is visible or close to the camera.
Do not infer the front direction only from the camera viewpoint.
Choose exactly one front_direction_2d from:
left, right, up, down, toward_camera, away_camera, unknown.
Use toward_camera only when the semantic front faces the viewer.
Use away_camera only when the semantic front faces away from the viewer.
For left/right/up/down, choose the direction that the semantic front points toward in the image.
If the object is too blurry, heavily occluded, has no natural front direction, or confidence is below 0.70, answer unknown.
Return only JSON with keys: front_direction_2d, confidence, reason.
""".strip()
VLM_ORIENTATION_LABEL_PROMPT: dict[str, str] = {
    "air purifier": "The front of the air purifier is the main display, control, or air-intake side. Use the display screen, buttons, indicator lights, air inlet, or brand logo as evidence.",
    "keyboard": "The front of the keyboard is the direction perpendicular to and upward from the keyboard surface.",
    "monitor": "The front of the monitor is the screen/display side. Ignore visibility alone; use the visible screen plane, bezel, stand orientation, or back panel cues.",
    "mouse": "The front of the mouse is the forward end indicated by the direction in which the mouse buttons extend. Use the button layout, scroll-wheel position, and the narrower-front, wider-rear shape as evidence.",
    "printer": "The front of the printer is the paper output/input, control-panel, cartridge-access, or user-service side.",
    "plate": "The front of the plate is the upper surface used to hold and display food, opposite the bottom side that contacts the table.",
    "television": "The front of the television is the screen/display side.",
    "pillow": "The front of the pillow is the side facing the user, opposite the side against the backrest.",
    "table": "The front of the table is the upward direction perpendicular to the tabletop surface, opposite the underside and floor-facing direction.",
    "cabinet": "The front of the cabinet is the door/drawer/handle/access side, with the front direction perpendicular to the cabinet door. Use handles, drawer gaps, shelves, or door panels as evidence.",
    "curtain": "The front of the curtain is the interior-facing direction perpendicular to the overall curtain plane, intended for viewing and operation.",
    "picture": "The front of the picture is the image/display direction perpendicular to the overall picture plane.",
    "picture frame": "The front of the picture frame is the image/glass/display direction perpendicular to the overall picture plane.",
    "shelf": "The front of the shelf is the open access side used for viewing or retrieving items. Use the shelf openings, exposed compartments, or cabinet doors as evidence.",
    "door": "The front of the door is the functional access/handle-facing side only when handle, panels, swing side, or visible room context makes it clear; otherwise answer unknown.",

    "chair": "The front of the chair is the side perpendicular to the chair backrest and facing the direction from which the user sits down.",
    "desk": "The front direction of the desk lies parallel to the tabletop plane and points toward the chair or the user interaction side.",
    "sofa": "The front of the sofa is the side perpendicular to the sofa backrest and facing the direction from which the user sits down.",

    "teapot": "The front of the teapot is the spout-facing side, with the handle typically on the opposite side. Use the spout, handle, lid, or decorative face as evidence.",
    "computer": "The front of the computer is the screen side for an all-in-one computer, or the front panel, ports, power button, or drive bay side for a desktop tower.",
    "computer tower": "The front of the computer tower is the panel with power button, ports, vents, or drive bays; not merely the side closest to the camera.",
}

# Step 8: semantic OBB generation
DEFAULT_OBB_EIGENVALUE_EPS = 1e-12
DEFAULT_OBB_MIN_POINTS_FOR_PCA = 4

# ---------------------------------------------------------------------------
# MASt3R orchestration helpers
# ---------------------------------------------------------------------------
import argparse as _argparse_for_mast3r
import logging as _logging_for_mast3r
import shutil as _shutil_for_mast3r
import subprocess as _subprocess_for_mast3r
import sys as _sys_for_mast3r
import time as _time_for_mast3r

SCRIPT_START_PERF = _time_for_mast3r.perf_counter()
SCRIPT_STARTED_AT = _time_for_mast3r.strftime("%Y-%m-%d %H:%M:%S")

_mast3r_logger = _logging_for_mast3r.getLogger("integrated_mast3r_pipeline")


def _resolve_user_path(path_value, base=_THIS_REPO_ROOT):
    path = Path(path_value)
    if str(path) == ".":
        return base.resolve()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _nonempty_path(path_value) -> bool:
    return bool(str(path_value).strip()) and str(path_value) != "."


def _require_local_vlm_endpoint(value: str) -> str:
    """Reject endpoints that could transmit source images off this machine."""

    endpoint = str(value).strip()
    parsed = urlparse(endpoint)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            "The model-service endpoint must use a loopback host because image payloads may "
            "contain private visual information."
        )
    return endpoint


def _top_level_image_files(directory: Path) -> list[Path]:
    """Return supported image files directly inside ``directory`` only."""

    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def _discover_rgb_datasets(dataset_path: Path) -> tuple[list[tuple[str, Path]], bool]:
    """Discover one RGB input or all ``<scene>/color`` inputs under a root.

    The boolean return value indicates whether ``dataset_path`` is a multi-scene
    root.  Discovery is intentionally one level deep so images from folders
    such as ``2D_segment`` can never leak into the reconstruction input.
    """

    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"RGB dataset not found: {dataset_path}")
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            raise ValueError(f"Unsupported RGB input file: {dataset_path}")
        return [(dataset_path.stem, dataset_path)], False

    if _top_level_image_files(dataset_path):
        scene_name = dataset_path.parent.name if dataset_path.name.lower() == "color" else dataset_path.name
        return [(scene_name, dataset_path)], False

    scenes: list[tuple[str, Path]] = []
    for scene_dir in sorted(
        (path for path in dataset_path.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        color_dir = scene_dir / "color"
        if _top_level_image_files(color_dir):
            scenes.append((scene_dir.name, color_dir.resolve()))

    if not scenes:
        raise FileNotFoundError(
            f"No ScanNet scene color images found under {dataset_path}; expected "
            "<dataset>/<scene>/color/*.{jpg,jpeg,png,bmp,webp}."
        )
    return scenes, True


def _prepare_mast3r_dataset(dataset_path: Path, staging_parent: Path, scene_name: str) -> Path:
    """Provide MASt3R with a top-level PNG directory for one scene.

    The MASt3R RGBFiles loader used by this pipeline scans only ``*.png``.
    ScanNet color folders commonly contain JPEG files, so non-PNG inputs are
    converted in a temporary per-scene directory.  No depth, pose, annotation,
    or generated image directories are read.
    """

    if dataset_path.is_file():
        return dataset_path

    image_files = _top_level_image_files(dataset_path)
    if not image_files:
        raise FileNotFoundError(f"No supported top-level RGB images found: {dataset_path}")
    # A ScanNet folder is literally named "color", which would make every
    # MASt3R export collide under robospatial/color.  Stage it under the scene
    # name even when its source frames are already PNG files.
    if (
        dataset_path.name.lower() != "color"
        and all(path.suffix.lower() == ".png" for path in image_files)
    ):
        return dataset_path

    staging_dir = staging_parent / scene_name
    staging_dir.mkdir(parents=True, exist_ok=False)
    used_names: set[str] = set()
    for source_path in image_files:
        target_name = f"{source_path.stem}.png"
        normalized_name = target_name.lower()
        if normalized_name in used_names:
            raise ValueError(
                f"Duplicate frame stem in {dataset_path}: {source_path.stem!r}"
            )
        used_names.add(normalized_name)
        target_path = staging_dir / target_name
        if source_path.suffix.lower() == ".png":
            _shutil_for_mast3r.copy2(source_path, target_path)
        else:
            with Image.open(source_path) as source_image:
                source_image.convert("RGB").save(target_path, format="PNG")
    return staging_dir.resolve()


def _is_robospatial_scan_root(path: Path) -> bool:
    return (
        (path / "sequence" / "color").is_dir()
        and (path / "sequence" / "depth").is_dir()
        and (path / "sequence" / "camera_poses" / "_info.txt").is_file()
    )


def _run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    _mast3r_logger.info("Running MASt3R command with sensitive paths omitted.")
    if dry_run:
        return
    _subprocess_for_mast3r.run(cmd, cwd=str(cwd), check=True)


def _validate_mast3r_dataset(dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"MASt3R input dataset not found: {dataset_path}")
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            raise ValueError(f"MASt3R input file must be a video file: {dataset_path}")
        return
    png_files = sorted(dataset_path.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(
            f"MASt3R input directory contains no top-level .png frames: {dataset_path}. "
            "ScanNet JPEG inputs must first be normalized by _prepare_mast3r_dataset."
        )


def _run_mast3r_slam(args) -> None:
    _validate_mast3r_dataset(_resolve_user_path(args.dataset))
    cmd = [
        _sys_for_mast3r.executable,
        "main.py",
        "--dataset",
        str(_resolve_user_path(args.dataset)),
        "--config",
        str(_resolve_user_path(args.mast3r_config, _resolve_user_path(args.mast3r_root))),
        "--save-as",
        args.mast3r_save_as,
        "--export-robospatial",
    ]
    if not args.viz:
        cmd.append("--no-viz")
    if args.mast3r_calib:
        cmd.extend(["--calib", str(_resolve_user_path(args.mast3r_calib, _resolve_user_path(args.mast3r_root)))])
    _run_command(cmd, cwd=_resolve_user_path(args.mast3r_root), dry_run=args.dry_run)


def _discover_mast3r_scan_root(args) -> Path:
    if args.mast3r_scan_root:
        scan_root = _resolve_user_path(args.mast3r_scan_root)
        if not _is_robospatial_scan_root(scan_root):
            raise FileNotFoundError(f"Not a MASt3R RoboSpatial scan root: {scan_root}")
        return scan_root

    mast3r_root = _resolve_user_path(args.mast3r_root)
    robospatial_root = mast3r_root / "logs" / args.mast3r_save_as / "robospatial"
    if not robospatial_root.exists():
        raise FileNotFoundError(f"MASt3R RoboSpatial export not found: {robospatial_root}")

    # MASt3R exports each dataset under robospatial/<dataset-name>.  A shared
    # --save-as directory can therefore contain exports from several runs.
    # Select the export belonging to this invocation explicitly; choosing the
    # directory with the newest mtime can incorrectly pick an older dataset
    # whose directory metadata happens to be newer.
    dataset_path = _resolve_user_path(args.dataset)
    dataset_name = dataset_path.stem if dataset_path.is_file() else dataset_path.name
    expected_scan_root = robospatial_root / dataset_name
    if _is_robospatial_scan_root(expected_scan_root):
        return expected_scan_root.resolve()

    candidates = [path for path in robospatial_root.iterdir() if path.is_dir() and _is_robospatial_scan_root(path)]
    if not candidates:
        raise FileNotFoundError(
            f"No valid scan root under {robospatial_root}; expected {expected_scan_root}"
        )
    available = ", ".join(sorted(path.name for path in candidates))
    raise FileNotFoundError(
        f"MASt3R export for dataset '{dataset_name}' not found: {expected_scan_root}. "
        f"Available exports: {available}"
    )


def _find_mast3r_scene_ply(scan_root: Path, mast3r_root: Path, save_as: str) -> str:
    seq_name = scan_root.name
    candidates = [
        mast3r_root / "logs" / save_as / f"{seq_name}.ply",
        scan_root.parent.parent / f"{seq_name}.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _copytree_merge(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.mkdir(parents=True, exist_ok=True)
    _shutil_for_mast3r.copytree(src, dst, dirs_exist_ok=True)


def _materialize_mast3r_outputs(source_scan_root: Path, scene_ply: str, output_dir: Path) -> tuple[Path, str]:
    """Place MASt3R outputs in OUTPUT_DIR/sequence and return OUTPUT_DIR as scan root."""
    target_sequence_dir = output_dir / "sequence"
    target_scan_root = output_dir
    source_sequence_dir = source_scan_root / "sequence"
    if not source_sequence_dir.exists():
        raise FileNotFoundError(f"MASt3R source sequence directory not found: {source_sequence_dir}")

    # This directory is a materialized copy, not a cache.  Remove files from a
    # previous dataset before copying so stale RGB/depth/pose files cannot be
    # mixed into the current run.
    if target_sequence_dir.exists():
        _shutil_for_mast3r.rmtree(target_sequence_dir)
    target_sequence_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("color", "depth", "camera_poses"):
        src = source_sequence_dir / subdir
        dst = target_sequence_dir / subdir
        if not src.exists():
            raise FileNotFoundError(f"MASt3R source {subdir} directory not found: {src}")
        _copytree_merge(src, dst)

    info_src = source_sequence_dir / "camera_poses" / "_info.txt"
    info_dst = target_sequence_dir / "camera_poses" / "_info.txt"
    if not info_src.exists():
        raise FileNotFoundError(f"MASt3R source _info.txt not found: {info_src}")
    if info_src.resolve() != info_dst.resolve():
        _shutil_for_mast3r.copy2(info_src, info_dst)

    keyframes_src = source_sequence_dir / "keyframes.txt"
    keyframes_dst = target_sequence_dir / "keyframes.txt"
    if not keyframes_src.exists():
        raise FileNotFoundError(
            f"MASt3R final-keyframe list not found: {keyframes_src}. "
            "Rerun MASt3R-SLAM with the updated exporter instead of reusing an older export."
        )
    if keyframes_src.resolve() != keyframes_dst.resolve():
        _shutil_for_mast3r.copy2(keyframes_src, keyframes_dst)

    target_scene_ply = target_sequence_dir / "scene.ply"
    if scene_ply:
        scene_src = Path(scene_ply)
        if scene_src.exists() and scene_src.resolve() != target_scene_ply.resolve():
            _shutil_for_mast3r.copy2(scene_src, target_scene_ply)
        provenance_src = scene_src.with_name(f"{scene_src.stem}.provenance.npz")
        provenance_dst = target_sequence_dir / "scene.provenance.npz"
        if not provenance_src.exists():
            raise FileNotFoundError(
                f"MASt3R scene provenance not found: {provenance_src}. "
                "Rerun MASt3R-SLAM with the updated reconstruction exporter."
            )
        if provenance_src.resolve() != provenance_dst.resolve():
            _shutil_for_mast3r.copy2(provenance_src, provenance_dst)
    return target_scan_root.resolve(), str(target_scene_ply.resolve()) if target_scene_ply.exists() else ""


def parse_integrated_args():
    parser = _argparse_for_mast3r.ArgumentParser(
        description=(
            "Integrated video-frame to VLM/SAM2 object reconstruction and semantic OBB pipeline."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(INPUT_RGB_DATASET),
        help=(
            "RGB input: a ScanNet root containing <scene>/color image folders, "
            "one color directory, or one video file."
        ),
    )
    parser.add_argument(
        "--scene",
        default=INPUT_SCENE,
        help=(
            "Process exactly one scene from a multi-scene ScanNet root. "
            "Accepts the scene directory name or a 1-based scene index. "
            "Required when --dataset contains more than one scene."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
        help="Output root for reconstruction, masks, object geometry, and semantic OBBs.",
    )
    parser.add_argument("--mast3r_root", default=str(MAST3R_SLAM_ROOT), help="MASt3R-SLAM repository root.")
    parser.add_argument("--mast3r_config", default=str(MAST3R_CONFIG), help="MASt3R config YAML.")
    parser.add_argument("--mast3r_calib", default=str(MAST3R_CALIB) if _nonempty_path(MAST3R_CALIB) else "", help="Optional MASt3R intrinsics YAML.")
    parser.add_argument("--mast3r_save_as", default=MAST3R_SAVE_AS, help="MASt3R logs/<save-as> name.")
    parser.add_argument("--mast3r_scan_root", default=str(MAST3R_SCAN_ROOT) if _nonempty_path(MAST3R_SCAN_ROOT) else "", help="Existing MASt3R RoboSpatial export root.")
    parser.add_argument("--skip_mast3r", action="store_true", help="Reuse --mast3r_scan_root instead of running MASt3R-SLAM.")
    parser.add_argument("--skip_object_pipeline", action="store_true", help="Only run/discover MASt3R export; skip VLM/SAM2/object point extraction.")
    parser.add_argument("--viz", action="store_true", default=MAST3R_SHOW_VISUALIZATION, help="Show MASt3R visualization window.")
    parser.add_argument(
        "--label_file",
        default="",
        help=(
            "Generated VLM label JSON path for single-scene input. "
            "For a multi-scene ScanNet root, each scene always uses "
            "<output_dir>/<scene>/sequence/labels.json."
        ),
    )
    parser.add_argument("--vlm_concurrency", type=int, default=VLM_CONCURRENCY)
    parser.add_argument("--sam2_checkpoint", default=str(SAM2_CHECKPOINT))
    parser.add_argument("--sam2_config", default=str(SAM2_CONFIG))
    parser.add_argument("--max_frames", type=int, default=0 if MAX_FRAMES is None else int(MAX_FRAMES))
    parser.add_argument("--device", default="" if DEVICE is None else str(DEVICE))
    parser.add_argument("--association_iou_threshold", type=float, default=None)
    parser.add_argument("--association_window_frames", type=int, default=None)
    parser.add_argument("--association_same_label", action="store_true", default=ASSOCIATION_SAME_LABEL)
    parser.add_argument("--pose_type", choices=("camera_to_world", "world_to_camera"), default=POSE_TYPE)
    parser.add_argument("--depth_shift", type=float, default=DEPTH_SHIFT)
    parser.add_argument("--clean_mode", choices=("preserve", "strict"), default="")
    parser.add_argument("--spatial_cluster_instances", action="store_true", default=DEFAULT_SPATIAL_CLUSTER_INSTANCES)
    parser.add_argument("--split_spatial_clusters", action="store_true", default=DEFAULT_SPLIT_SPATIAL_CLUSTERS)
    parser.add_argument("--obb_eigenvalue_eps", type=float, default=DEFAULT_OBB_EIGENVALUE_EPS)
    parser.add_argument("--obb_min_points_for_pca", type=int, default=DEFAULT_OBB_MIN_POINTS_FOR_PCA)
    parser.add_argument("--obb_vlm_max_retries", type=int, default=DEFAULT_OBB_VLM_MAX_RETRIES)
    parser.add_argument("--obb_vlm_request_timeout", type=float, default=DEFAULT_OBB_VLM_REQUEST_TIMEOUT)
    parser.add_argument("--obb_vlm_retry_sleep", type=float, default=DEFAULT_OBB_VLM_RETRY_SLEEP)
    parser.add_argument("--obb_vlm_crop_margin_ratio", type=float, default=DEFAULT_OBB_VLM_CROP_MARGIN_RATIO)
    parser.add_argument("--obb_vlm_orientation_min_confidence", type=float, default=DEFAULT_OBB_VLM_ORIENTATION_MIN_CONFIDENCE)
    parser.add_argument("--dry_run", action="store_true", help="Validate command construction without executing the pipeline.")
    parser.add_argument("--log_level", default=DEFAULT_LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _build_embedded_object_argv(args, scan_root: Path, output_dir: Path) -> list[str]:
    argv = [
        "embedded_object_pipeline",
        "--scan_root", str(scan_root),
        "--image_dir", str(scan_root / "sequence" / "color"),
        "--camera_info_file", str(scan_root / "sequence" / "camera_poses" / "_info.txt"),
        "--camera_pose_dir", str(scan_root / "sequence" / "camera_poses"),
        "--depth_dir", str(scan_root / "sequence" / "depth"),
        "--output_dir", str(output_dir),
        "--label_file", str(_resolve_user_path(args.label_file)),
        "--vlm_concurrency", str(args.vlm_concurrency),
        "--sam2_checkpoint", str(_resolve_user_path(args.sam2_checkpoint)),
        "--sam2_config", str(args.sam2_config),
        "--pose_type", str(args.pose_type),
        "--depth_shift", str(args.depth_shift),
    ]
    if args.max_frames > 0:
        argv.extend(["--max_frames", str(args.max_frames)])
    if args.device:
        argv.extend(["--device", str(args.device)])
    if args.association_iou_threshold is not None:
        argv.extend(["--association_iou_threshold", str(args.association_iou_threshold)])
    if args.association_window_frames is not None:
        argv.extend(["--association_window_frames", str(args.association_window_frames)])
    if args.association_same_label:
        argv.append("--association_same_label")
    if args.clean_mode:
        argv.extend(["--clean_mode", args.clean_mode])
    if args.spatial_cluster_instances:
        argv.append("--spatial_cluster_instances")
    if args.split_spatial_clusters:
        argv.append("--split_spatial_clusters")
    argv.extend(["--obb_eigenvalue_eps", str(args.obb_eigenvalue_eps)])
    argv.extend(["--obb_min_points_for_pca", str(args.obb_min_points_for_pca)])
    argv.extend(["--obb_vlm_max_retries", str(args.obb_vlm_max_retries)])
    argv.extend(["--obb_vlm_request_timeout", str(args.obb_vlm_request_timeout)])
    argv.extend(["--obb_vlm_retry_sleep", str(args.obb_vlm_retry_sleep)])
    argv.extend(["--obb_vlm_crop_margin_ratio", str(args.obb_vlm_crop_margin_ratio)])
    argv.extend(["--obb_vlm_orientation_min_confidence", str(args.obb_vlm_orientation_min_confidence)])
    return argv

"""
Single RGB image -> 2D instance masks -> single-frame object point clouds.

Main flow:
1. Read one RGB image and a fixed label list.
2. Use the VLM to identify visible label instances and produce point/box prompts.
3. Use the SAM2 image predictor to produce masks for those prompts.
4. Save per-instance 2D mask artifacts and a 2D overlay.
5. Treat every 2D instance in this single frame as an independent object.
6. Back-project each mask with aligned depth and camera pose into 3D points.
7. Clean and export per-object 3D point clouds.

Outputs:
- 2D_segment/merged_masks/<frame>/mask_XXX.png
- 2D_segment/overlay/<frame>.overlay.jpg
- 2D_segment/instances_2d.json
- 3D_point/object_3d_points.json and object_3d_points/*.xyz/*.ply
- 3D_point_clean/object_3d_points_clean.json and object_3d_points_clean/*.xyz/*.ply
- VLM_OBB/object_3d_obbs.json and per-object *_3D_OBB.json
- pipeline_timing.txt
"""
import time

import argparse
import asyncio
import base64
import copy
import io
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import zlib
from contextlib import nullcontext
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(script_dir: Path) -> Path:
    env_root = os.environ.get("SAM2_MAIN_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "sam2").exists() and (candidate / "tools").exists():
            return candidate.resolve()
    return script_dir.parents[1].resolve()


REPO_ROOT = find_repo_root(SCRIPT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("seg_instances")

# ---------------------------------------------------------------------------
# Derived paths and aliases; edit the source values at the top of this file.
# ---------------------------------------------------------------------------
DEFAULT_SCAN_ROOT = OUTPUT_DIR.resolve()
DEFAULT_IMAGE_DIR = (DEFAULT_SCAN_ROOT / "sequence" / "color").resolve()
DEFAULT_DEPTH_DIR = (DEFAULT_SCAN_ROOT / "sequence" / "depth").resolve()
DEFAULT_CAMERA_INFO_FILE = (DEFAULT_SCAN_ROOT / "sequence" / "camera_poses" / "_info.txt").resolve()
DEFAULT_CAMERA_POSE_DIR = (DEFAULT_SCAN_ROOT / "sequence" / "camera_poses").resolve()
DEFAULT_LABEL_FILE = (DEFAULT_SCAN_ROOT / "label.json").resolve()
DEFAULT_OUTPUT_DIR = OUTPUT_DIR.resolve()
DEFAULT_SAM2_CHECKPOINT = str(SAM2_CHECKPOINT.resolve())
DEFAULT_SAM2_CONFIG = SAM2_CONFIG
DEFAULT_API_BASE_URL = API_BASE_URL
DEFAULT_API_KEY = API_KEY
DEFAULT_SERVICE_MODEL = SERVICE_MODEL


# Output directory and file names used by the embedded object pipeline
DEFAULT_2D_SEGMENT_SUBDIR = "2D_segment"
DEFAULT_FRAME_ASSOCIATION_SUBDIR = "instance_association"
DEFAULT_3D_POINT_SUBDIR = "3D_point"
DEFAULT_3D_POINT_CLEAN_SUBDIR = "3D_point_clean"
DEFAULT_3D_OBB_SUBDIR = "VLM_OBB"
DEFAULT_ASSOCIATION_OUTPUT_FILENAME = "3D_segment.json"
DEFAULT_ASSOCIATION_OVERLAY_SUBDIR = "association_overlay"
DEFAULT_POINT3D_OUTPUT_FILENAME = "object_3d_points.json"
DEFAULT_POINT3D_OUTPUT_SUBDIR = "object_3d_points"
DEFAULT_POINT3D_CLEAN_OUTPUT_FILENAME = "object_3d_points_clean.json"
DEFAULT_POINT3D_CLEAN_OUTPUT_SUBDIR = "object_3d_points_clean"
DEFAULT_LOG_LEVEL = "INFO"

# Category-specific geometry constraints applied after the VLM vote
# These rules are applied after the multi-frame VLM vote. A category rule may
# constrain only the front axis (cabinet and planar display/interior-facing
# objects), or both the axis and its semantic sign when object-only geometry
# provides a reliable cue (sofa backrest).
CATEGORY_GEOMETRY_RULES: dict[str, dict[str, str]] = {
    "cabinet": {
        "policy": "smallest_variance_principal_axis",
        "description": "Force the cabinet front axis to the normal of its thinnest principal 3D plane (the cabinet-door/front-plane proxy).",
    },
    "curtain": {
        "policy": "smallest_variance_principal_axis",
        "description": "Force the front axis to the normal of the curtain's overall best-fit 3D plane; orient it toward the indoor observer-camera side, with multi-frame VLM used only as a fallback sign cue.",
    },
    "desk": {
        "policy": "tabletop_short_axis_horizontal_front",
        "description": "Point horizontally toward the co-visible associated chair when available; otherwise use the tabletop footprint's short axis with the multi-frame VLM sign.",
    },
    "sofa": {
        "policy": "backrest_away_horizontal_axis",
        "description": "Fit the top backrest band and orient its horizontal normal from the seat/support region toward the backrest side.",
    },
    "picture": {
        "policy": "smallest_variance_principal_axis",
        "description": "Force the front axis to the normal of the picture's overall best-fit 3D plane; use the multi-frame VLM vote only to choose the image-facing sign.",
    },
    "picture frame": {
        "policy": "smallest_variance_principal_axis",
        "description": "Force the front axis to the normal of the picture frame's overall best-fit 3D plane; use the multi-frame VLM vote only to choose the image/glass-facing sign.",
    },
}






class VLMServiceUnavailableError(RuntimeError):
    """Raised when the VLM endpoint cannot be reached or serve requests."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image(image_input: str | Image.Image) -> str:
    """Base64-encode an image file path or PIL.Image for VLM API calls."""
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    buffer = io.BytesIO()
    image_input.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def extract_json(text: str) -> Optional[Any]:
    """Robust JSON extraction from VLM text output."""
    if not text:
        return None
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

    return None


def mask_to_base64(mask: np.ndarray) -> str:
    """Encode a (H, W) bool mask as a base64 PNG string."""
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_mask(b64: str) -> np.ndarray:
    """Decode a base64 PNG mask back to (H, W) bool numpy array."""
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return (np.array(img) > 127).astype(np.uint8)


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def sam2_config_for_image_predictor(config_path: str | Path) -> str:
    config_text = str(config_path).replace("\\", "/")
    if config_text.startswith("../sam2/configs/"):
        return config_text

    path = resolve_path(config_path)
    configs_root = (REPO_ROOT / "sam2" / "configs").resolve()
    try:
        rel = path.relative_to(configs_root).as_posix()
    except ValueError:
        return str(config_path)
    return f"../sam2/configs/{rel}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_instances(instances_json: Path) -> list[dict]:
    data = json.loads(instances_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("instances", [])
    if not isinstance(data, list):
        raise ValueError(f"Invalid instances_2d.json format: {instances_json}")
    return data


def frame_sort_key(frame_name: str):
    numbers = re.findall(r"\d+", frame_name)
    return tuple(int(x) for x in numbers) if numbers else (frame_name,)


def label_from_instance_name(instance_name: str) -> str:
    return re.sub(r"_\d+$", "", instance_name)


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).casefold()


def obb_normalize_category_label(label: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(label).strip().casefold())


def matched_orientation_category(label: str) -> str | None:
    normalized = obb_normalize_category_label(label)
    if not normalized:
        return None
    if normalized in VLM_ORIENTATION_LABEL_PROMPT:
        return normalized

    # Prefer the most specific (longest) category when labels contain multiple
    # category names, e.g. "desktop computer tower".
    for key in sorted(VLM_ORIENTATION_LABEL_PROMPT, key=lambda item: len(obb_normalize_category_label(item)), reverse=True):
        key_norm = obb_normalize_category_label(key)
        if key_norm and (key_norm in normalized or normalized in key_norm):
            return key
    return None


def category_orientation_hint(label: str) -> str:
    category = matched_orientation_category(label)
    if category is not None:
        return VLM_ORIENTATION_LABEL_PROMPT[category]
    return (
        "Use object-specific functional cues only. If this category does not have a stable semantic front, "
        "or the highlighted view does not provide enough evidence, answer unknown."
    )


def has_category_orientation_hint(label: str) -> bool:
    return matched_orientation_category(label) is not None


def natural_path_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_json_array(text: str) -> list[str] | None:
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                data = None
        else:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    if isinstance(data, dict):
        data = data.get("labels") or data.get("categories") or data.get("objects")
    if not isinstance(data, list):
        return None
    return [str(item).strip() for item in data if str(item).strip()]


def clean_generated_labels(text: str) -> list[str]:
    labels = extract_json_array(text)
    if labels is None:
        labels = re.split(r"[,\uFF0C\n]+", text)

    cleaned_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        label = str(label).strip()
        label = re.sub(r"^[`'\"\-\d\.\s]+|[`'\"\s]+$", "", label)
        label = re.sub(r"^(label|category|object category)\s*:\s*", "", label, flags=re.IGNORECASE)
        if not label:
            continue
        key = normalize_label(label)
        if key not in seen:
            cleaned_labels.append(label)
            seen.add(key)
    return cleaned_labels


def build_label_payload(image_path: Path, model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Identify all salient foreground object categories in this image. "
                            "Exclude background objects, scene stuff, walls, floor, ceiling, sky, road, "
                            "and other non-target background. If a small object is inside, attached to, "
                            "or clearly used as part of another foreground object, do not output the "
                            "small contained object separately; label the combined whole instead. "
                            "Return only a JSON array of concise category names, for example: "
                            "[\"chair\", \"table\"]. Do not explain and do not include any text outside "
                            "the JSON array."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract_response_content(result: dict) -> str:
    choices = result.get("choices")
    if not choices:
        raise ValueError(f"empty choices in API response: {json.dumps(result, ensure_ascii=False)[:500]}")

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = choice.get("text")
    if content is None:
        raise ValueError(f"missing content in API response: {json.dumps(result, ensure_ascii=False)[:500]}")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    return str(content)


def request_generated_labels(
    image_path: Path,
    api_base_url: str,
    api_key: str,
    model: str,
    max_retries: int,
    timeout_seconds: int = DEFAULT_VLM_LABEL_REQUEST_TIMEOUT,
) -> list[str]:
    api_base_url = _require_local_vlm_endpoint(api_base_url)
    url = api_base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(build_label_payload(image_path, model)).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error: Exception | None = None
    for attempt in range(max(1, max_retries)):
        request = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            return clean_generated_labels(extract_response_content(result))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max(1, max_retries) - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to label {image_path.name}: {last_error}") from last_error


def generate_label_json(
    image_paths: list[Path],
    output_file: Path,
    api_base_url: str,
    api_key: str,
    model: str,
    max_retries: int,
) -> dict[str, list[str]]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    label_results: dict[str, list[str]] = {}
    for index, image_path in enumerate(image_paths, start=1):
        try:
            labels = request_generated_labels(
                image_path=image_path,
                api_base_url=api_base_url,
                api_key=api_key,
                model=model,
                max_retries=max_retries,
            )
        except RuntimeError as exc:
            logger.error("%s", exc)
            labels = []
        relative_image_path = image_path.resolve().relative_to(REPO_ROOT).as_posix()
        label_results[relative_image_path] = labels
        logger.info("[%d/%d] Generated labels for %s -> %s", index, len(image_paths), relative_image_path, labels)

    write_json(output_file, label_results)
    return label_results


def load_label_list(label_file: Path) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for line in label_file.read_text(encoding="utf-8-sig").splitlines():
        label = line.strip()
        if not label or label.startswith("#"):
            continue
        key = normalize_label(label)
        if key in seen:
            continue
        labels.append(label)
        seen.add(key)
    if not labels:
        raise ValueError(f"No labels found in label_file: {label_file}")
    return labels


def normalize_label_items(items: Any) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    if not isinstance(items, list):
        return ()
    for item in items:
        label = str(item).strip()
        if not label:
            continue
        key = normalize_label(label)
        if key in seen:
            continue
        labels.append(label)
        seen.add(key)
    return tuple(labels)


def image_label_keys(image_path: Path) -> list[str]:
    resolved = image_path.resolve()
    keys = [str(resolved), resolved.as_posix(), image_path.name, image_path.stem]
    try:
        rel = resolved.relative_to(REPO_ROOT).as_posix()
        keys.insert(0, rel)
    except ValueError:
        pass
    return keys


def load_label_mapping(label_file: Path, image_paths: list[Path]) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    if label_file.suffix.lower() != ".json":
        return {}, tuple(load_label_list(label_file))

    data = json.loads(label_file.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return {}, normalize_label_items(data)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON label file format: {label_file}")

    raw_mapping: dict[str, tuple[str, ...]] = {
        str(key): normalize_label_items(value) for key, value in data.items()
    }
    label_map: dict[str, tuple[str, ...]] = {}
    for image_path in image_paths:
        labels = ()
        for key in image_label_keys(image_path):
            labels = raw_mapping.get(key, ())
            if labels:
                break
        for key in image_label_keys(image_path):
            label_map[key] = labels

    all_labels: list[str] = []
    seen: set[str] = set()
    for labels in raw_mapping.values():
        for label in labels:
            key = normalize_label(label)
            if key not in seen:
                all_labels.append(label)
                seen.add(key)
    return label_map, tuple(all_labels)


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def load_font(size: int = 18):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def color_for_key(key: str) -> tuple[int, int, int]:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(key))
    rng = np.random.default_rng(seed)
    color = rng.integers(40, 235, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    bg = (max(color[0] - 35, 0), max(color[1] - 35, 0), max(color[2] - 35, 0))
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def export_instance_artifacts(rows: list[dict], export_dir: Path) -> Path:
    masks_root = export_dir / "merged_masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    instances_2d: list[dict] = []

    for row in rows:
        image_path = Path(row["image_path"])
        frame_name = image_path.stem
        frame_mask_dir = masks_root / frame_name
        frame_mask_dir.mkdir(parents=True, exist_ok=True)
        seg_list = json.loads(row["segmentation_results"])
        mask_idx = 1

        for item in seg_list:
            masks_b64 = item.get("mask_base64")
            selected_idx = item.get("selected_mask_idx")
            if not masks_b64 or selected_idx is None:
                continue

            mask = base64_to_mask(masks_b64[int(selected_idx)]).astype(bool)
            if not mask.any():
                continue

            instance_name = str(item.get("instance", f"instance_{mask_idx}"))
            label = label_from_instance_name(instance_name)
            mask_name = f"mask_{mask_idx:03d}"
            object2d_id = f"{frame_name}__{mask_name}"
            mask_rel = Path("merged_masks") / frame_name / f"{mask_name}.png"
            mask_path = export_dir / mask_rel
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)

            instances_2d.append(
                {
                    "object2d_id": object2d_id,
                    "frame_name": frame_name,
                    "label": label,
                    "mask_name": mask_name,
                    "mask_path": str(mask_rel).replace("\\", "/"),
                    "coord": item.get("coord"),
                    "box_2d": item.get("box_2d"),
                    "bbox_xywh": mask_bbox_xywh(mask),
                    "area": int(mask.sum()),
                }
            )
            mask_idx += 1

    instances_path = export_dir / "instances_2d.json"
    instances_2d.sort(key=lambda item: (frame_sort_key(str(item.get("frame_name", ""))), str(item.get("object2d_id", ""))))
    write_json(instances_path, {"instances": instances_2d})
    logger.info("Mask PNGs saved to %s", masks_root)
    logger.info("instances_2d.json saved to %s", instances_path)
    return instances_path


def render_2d_overlays(
    instances_json: Path,
    masks_root: Path,
    image_dir: Path,
    output_dir: Path,
    alpha: float,
    draw_outline: bool,
    min_area: int,
) -> int:
    instances = read_instances(instances_json)
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        frame_name = inst.get("frame_name") or Path(inst.get("image_name", inst.get("image_path", ""))).stem
        if frame_name:
            by_frame[str(frame_name)].append(inst)

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_overlay in output_dir.glob("*.overlay.jpg"):
        frame_name = existing_overlay.name.removesuffix(".overlay.jpg")
        if frame_name not in by_frame:
            try:
                existing_overlay.unlink()
                logger.info("Deleted stale overlay without current instances: %s", existing_overlay)
            except OSError as exc:
                logger.warning("Failed to delete stale overlay %s: %s", existing_overlay, exc)

    font = load_font(18)
    written = 0
    image_lookup = {p.stem: p for p in discover_images([image_dir])}
    for frame_name, rows in sorted(by_frame.items(), key=lambda kv: frame_sort_key(kv[0])):
        image_path = image_lookup.get(frame_name)
        if image_path is None:
            logger.warning("Missing image for overlay frame: %s", frame_name)
            continue

        base = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        labels_to_draw: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []

        for inst in rows:
            mask_rel = inst.get("mask_path")
            if mask_rel:
                mask_rel_path = Path(str(mask_rel))
                mask_path = masks_root.parent / mask_rel_path if mask_rel_path.parts and mask_rel_path.parts[0] == masks_root.name else masks_root / mask_rel_path
            else:
                object2d_id = str(inst.get("object2d_id", ""))
                mask_name = object2d_id.split("__")[-1] if "__" in object2d_id else ""
                mask_path = masks_root / frame_name / f"{mask_name}.png"
            if not mask_path.exists():
                logger.warning("Missing mask for overlay: %s", mask_path)
                continue

            mask = np.array(Image.open(mask_path).convert("L")) > 0
            area = int(mask.sum())
            if area < min_area:
                continue

            label = str(inst.get("label") or "object")
            suffix = str(inst.get("mask_name") or "").replace("mask_", "")
            text = f"{label}_{suffix}" if suffix else label
            color = color_for_key(text)
            rgba = color + (int(max(0.0, min(1.0, alpha)) * 255),)
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            fill = Image.new("RGBA", base.size, rgba)
            overlay.paste(fill, (0, 0), mask_img)

            ys, xs = np.where(mask)
            if xs.size:
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                if draw_outline:
                    draw_overlay.rectangle((x0, y0, x1, y1), outline=color + (255,), width=2)
                labels_to_draw.append(((x0, max(0, y0 - 22)), text, color))

        composed = Image.alpha_composite(base, overlay)
        draw_final = ImageDraw.Draw(composed)
        for xy, text, color in labels_to_draw:
            draw_label(draw_final, xy, text, color, font)
        out_path = output_dir / f"{frame_name}.overlay.jpg"
        composed.convert("RGB").save(out_path, quality=92)
        written += 1

    logger.info("Wrote %d overlay image(s) to %s", written, output_dir)
    return written


def report_instance_coverage(instances_json: Path, image_paths: list[Path]) -> None:
    instances = read_instances(instances_json)
    frames_with_instances = {str(item.get("frame_name", "")) for item in instances}
    expected_frames = [path.stem for path in image_paths]
    missing_frames = [frame for frame in expected_frames if frame not in frames_with_instances]
    logger.info(
        "2D instance coverage: %d/%d image(s) have at least one valid instance.",
        len(expected_frames) - len(missing_frames),
        len(expected_frames),
    )
    if missing_frames:
        logger.warning("No valid 2D instances for frame(s): %s", ", ".join(missing_frames))


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item_a, item_b):
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def load_mask(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path).convert("L")) > 0


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union)


def prepare_association_instances(instances_2d_path: Path, merged_masks_dir: Path):
    instances = read_instances(instances_2d_path)
    instances_by_frame: dict[str, list[dict]] = defaultdict(list)
    instance_by_id: dict[str, dict] = {}

    for instance in instances:
        frame_name = str(instance["frame_name"])
        instance_id = str(instance["object2d_id"])
        mask_rel = instance.get("mask_path")
        if mask_rel:
            mask_rel_path = Path(str(mask_rel))
            mask_path = (
                merged_masks_dir.parent / mask_rel_path
                if mask_rel_path.parts and mask_rel_path.parts[0] == merged_masks_dir.name
                else merged_masks_dir / mask_rel_path
            )
        else:
            mask_path = merged_masks_dir / frame_name / f"{instance['mask_name']}.png"
        prepared = {
            **instance,
            "object2d_id": instance_id,
            "frame_name": frame_name,
            "mask_abs_path": mask_path,
        }
        instances_by_frame[frame_name].append(prepared)
        instance_by_id[instance_id] = prepared

    for frame_name in instances_by_frame:
        instances_by_frame[frame_name].sort(key=lambda item: str(item.get("mask_name", "")))
    return instances_by_frame, instance_by_id


def build_association_frame_index(image_paths: list[Path]):
    ordered = sorted(image_paths, key=lambda p: frame_sort_key(p.name))
    frame_names = [path.stem for path in ordered]
    frame_to_idx = {frame_name: idx for idx, frame_name in enumerate(frame_names)}
    return ordered, frame_names, frame_to_idx


def build_numeric_frame_dir(image_paths: list[Path], temp_root: Path, repeat_count: int = 1) -> Path:
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="sam2_numeric_frames_", dir=str(temp_root)))
    output_idx = 0
    for _ in range(max(1, repeat_count)):
        for src in image_paths:
            dst = temp_dir / f"{output_idx:06d}.jpg"
            output_idx += 1
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                Image.open(src).convert("RGB").save(dst, quality=95)
    return temp_dir


def greedy_association_match(
    predictions,
    targets,
    instance_by_id,
    mask_cache,
    threshold: float,
    association_same_label: bool,
):
    candidates = []
    for source_id, pred_mask in predictions.items():
        source_label = normalize_label(str(instance_by_id[source_id].get("label", "")))
        for target in targets:
            if association_same_label and normalize_label(str(target.get("label", ""))) != source_label:
                continue
            target_id = target["object2d_id"]
            iou = mask_iou(pred_mask, mask_cache[target_id])
            if iou >= threshold:
                candidates.append((iou, source_id, target_id))

    candidates.sort(reverse=True)
    used_sources = set()
    used_targets = set()
    matches = []
    for iou, source_id, target_id in candidates:
        if source_id in used_sources or target_id in used_targets:
            continue
        used_sources.add(source_id)
        used_targets.add(target_id)
        matches.append((source_id, target_id, iou))
    return matches


def association_edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    return tuple(sorted((str(source_id), str(target_id))))


def build_union_find_from_edges(instance_ids: list[str], edges: list[dict]) -> UnionFind:
    unions = UnionFind(instance_ids)
    for edge in edges:
        unions.union(edge["source_id"], edge["target_id"])
    return unions


def find_same_frame_conflict_group(unions: UnionFind, instance_by_id: dict[str, dict]) -> set[str] | None:
    groups: dict[str, list[str]] = defaultdict(list)
    for instance_id in instance_by_id:
        groups[unions.find(instance_id)].append(instance_id)
    for group_ids in groups.values():
        frame_seen: set[str] = set()
        for instance_id in group_ids:
            frame_name = str(instance_by_id[instance_id].get("frame_name", ""))
            if frame_name in frame_seen:
                return set(group_ids)
            frame_seen.add(frame_name)
    return None


def resolve_same_frame_conflicts(instance_ids: list[str], candidate_edges: list[dict], instance_by_id: dict[str, dict]):
    best_edges: dict[tuple[str, str], dict] = {}
    for edge in candidate_edges:
        key = edge["edge_key"]
        old = best_edges.get(key)
        if old is None or edge["score"] > old["score"]:
            best_edges[key] = edge

    active_edges = list(best_edges.values())
    removed_edges: list[dict] = []
    while active_edges:
        unions = build_union_find_from_edges(instance_ids, active_edges)
        conflict_group = find_same_frame_conflict_group(unions, instance_by_id)
        if conflict_group is None:
            return unions, active_edges, removed_edges
        group_edges = [
            edge for edge in active_edges
            if edge["source_id"] in conflict_group and edge["target_id"] in conflict_group
        ]
        if not group_edges:
            break
        weakest = min(
            group_edges,
            key=lambda edge: (
                edge["score"],
                -int(edge.get("offset", 0)),
                str(edge.get("source_id", "")),
                str(edge.get("target_id", "")),
            ),
        )
        active_edges.remove(weakest)
        removed_edges.append(weakest)
    return build_union_find_from_edges(instance_ids, active_edges), active_edges, removed_edges


def build_video_predictor(config_path: str | Path, checkpoint_path: Path, device: torch.device):
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(
        config_file=sam2_config_for_image_predictor(config_path),
        ckpt_path=str(checkpoint_path),
        device=device,
        apply_postprocessing=True,
    )


def track_associated_instances(
    predictor,
    inference_state,
    frame_names: list[str],
    frame_to_idx: dict[str, int],
    instances_by_frame,
    instance_by_id,
    mask_cache,
    match_iou_threshold: float,
    association_window_frames: int,
    association_same_label: bool,
    association_same_frame_exclusivity: bool,
):
    all_instance_ids = sorted(instance_by_id.keys())
    unions = UnionFind(all_instance_ids)
    pair_summaries = []
    candidate_edges = []
    frame_count = len(frame_names)
    window = max(0, min(int(association_window_frames), max(0, frame_count - 1)))
    if frame_count <= 1 or window == 0:
        return unions, pair_summaries

    def propagate_from(start_idx: int, reverse: bool) -> dict[int, dict]:
        frame_predictions = {}
        for out_frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_idx,
            max_frame_num_to_track=window + 1,
            reverse=reverse,
        ):
            if out_frame_idx == start_idx:
                continue
            distance = start_idx - out_frame_idx if reverse else out_frame_idx - start_idx
            if distance < 1 or distance > window:
                continue
            predicted_masks = {}
            for obj_id, mask_logits in zip(obj_ids, video_res_masks):
                predicted_masks[obj_id] = mask_logits[0].detach().cpu().numpy() > 0.0
            frame_predictions[out_frame_idx] = predicted_masks
        return frame_predictions

    for start_pos, current_frame in enumerate(frame_names):
        current_instances = instances_by_frame.get(current_frame, [])
        if not current_instances:
            continue
        current_idx = frame_to_idx[current_frame]
        direction_predictions: list[tuple[str, dict[int, dict]]] = []
        for direction, start_idx, reverse in (
            ("forward", current_idx, False),
            ("backward", current_idx, True),
        ):
            predictor.reset_state(inference_state)
            for instance in current_instances:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=start_idx,
                    obj_id=instance["object2d_id"],
                    mask=mask_cache[instance["object2d_id"]],
                )
            direction_predictions.append((direction, propagate_from(start_idx, reverse=reverse)))

        target_offsets = []
        for offset in range(1, window + 1):
            if start_pos + offset < frame_count:
                target_offsets.append(("forward", offset, start_pos + offset))
            if start_pos - offset >= 0:
                target_offsets.append(("backward", offset, start_pos - offset))

        predictions_by_direction = dict(direction_predictions)
        for direction, offset, next_pos in target_offsets:
            next_frame = frame_names[next_pos]
            next_instances = instances_by_frame.get(next_frame, [])
            predictions = predictions_by_direction.get(direction, {}).get(next_pos, {})
            if not next_instances:
                pair_summaries.append({
                    "from_frame": current_frame,
                    "to_frame": next_frame,
                    "direction": direction,
                    "offset": offset,
                    "matches": [],
                })
                continue
            matches = greedy_association_match(
                predictions=predictions,
                targets=next_instances,
                instance_by_id=instance_by_id,
                mask_cache=mask_cache,
                threshold=match_iou_threshold,
                association_same_label=association_same_label,
            )
            match_rows = []
            for source_id, target_id, iou in matches:
                edge_key = association_edge_key(source_id, target_id)
                candidate_edges.append({
                    "edge_key": edge_key,
                    "source_id": source_id,
                    "target_id": target_id,
                    "score": float(iou),
                    "iou": float(iou),
                    "from_frame": current_frame,
                    "to_frame": next_frame,
                    "direction": direction,
                    "offset": int(offset),
                })
                match_rows.append({
                    "source_object2d_id": source_id,
                    "target_object2d_id": target_id,
                    "iou": round(iou, 4),
                    "label": instance_by_id[source_id]["label"],
                    "accepted": False,
                })
            pair_summaries.append({
                "from_frame": current_frame,
                "to_frame": next_frame,
                "direction": direction,
                "offset": offset,
                "matches": match_rows,
            })

    if not association_same_frame_exclusivity:
        logger.warning(
            "association_same_frame_exclusivity=False is ignored because same-frame exclusivity is a hard invariant."
        )
    unions, accepted_edges, removed_edges = resolve_same_frame_conflicts(
        all_instance_ids, candidate_edges, instance_by_id
    )

    accepted_edge_keys = {edge["edge_key"] for edge in accepted_edges}
    removed_edge_keys = {edge["edge_key"] for edge in removed_edges}
    for pair_summary in pair_summaries:
        for match in pair_summary.get("matches", []):
            edge_key = association_edge_key(
                match.get("source_object2d_id", ""),
                match.get("target_object2d_id", ""),
            )
            match["accepted"] = edge_key in accepted_edge_keys
            if edge_key in removed_edge_keys:
                match["rejected_reason"] = "same_frame_conflict"
    return unions, pair_summaries


def histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    if hist_a.size == 0 or hist_b.size == 0:
        return 0.0
    return float(np.minimum(hist_a, hist_b).sum())


def build_3d_segments(unions: UnionFind, instance_by_id: dict[str, dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for instance_id in sorted(instance_by_id.keys()):
        groups[unions.find(instance_id)].append(instance_by_id[instance_id])

    def instance_sort_key(item):
        return frame_sort_key(str(item["frame_name"])), str(item.get("mask_name", ""))

    sorted_groups = sorted(
        groups.values(),
        key=lambda items: instance_sort_key(sorted(items, key=instance_sort_key)[0]),
    )
    label_totals = Counter()
    for items in sorted_groups:
        label = Counter(item["label"] for item in items).most_common(1)[0][0]
        label_totals[label] += 1
    label_seen = Counter()
    objects = []
    for items in sorted_groups:
        items = sorted(items, key=instance_sort_key)
        label = Counter(item["label"] for item in items).most_common(1)[0][0]
        label_seen[label] += 1
        object_name = f"{label}_{label_seen[label]}" if label_totals[label] > 1 else label
        objects.append({
            "objectId": object_name,
            "label": label,
            "instance_count": len(items),
            "frames": [item["frame_name"] for item in items],
            "object2d_ids": [item["object2d_id"] for item in items],
        })
    return objects


def write_association_instances(instances_2d_path: Path, objects: list[dict], output_path: Path) -> Path:
    instances = read_instances(instances_2d_path)
    association_by_2d_id = {}
    for obj in objects:
        for object2d_id in obj["object2d_ids"]:
            association_by_2d_id[str(object2d_id)] = {
                "objectId": obj["objectId"],
                "label": obj["label"],
            }
    enriched = []
    for instance in instances:
        association = association_by_2d_id.get(str(instance.get("object2d_id", "")))
        if association is None:
            enriched.append({**instance, "objectId": None})
        else:
            enriched.append({**instance, **association})
    write_json(output_path, {"instances": enriched})
    return output_path


def run_instance_association(
    instances_2d_path: Path,
    merged_masks_dir: Path,
    image_paths: list[Path],
    output_path: Path,
    sam2_config: str | Path,
    sam2_checkpoint: Path,
    device_arg: str | None,
    match_iou_threshold: float,
    association_window_frames: int,
    association_same_label: bool,
    association_same_frame_exclusivity: bool,
) -> tuple[Path, list[dict], list[dict]]:
    if not instances_2d_path.exists():
        raise FileNotFoundError(f"Missing instances_2d.json: {instances_2d_path}")
    if not merged_masks_dir.exists():
        raise FileNotFoundError(f"Missing merged_masks directory: {merged_masks_dir}")
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {sam2_checkpoint}")

    ordered_images, frame_names, frame_to_idx = build_association_frame_index(image_paths)
    instances_by_frame, instance_by_id = prepare_association_instances(instances_2d_path, merged_masks_dir)
    if not instance_by_id:
        output = {"objects": [], "pairwise_matches": []}
        write_json(output_path, output)
        return output_path, [], []
    mask_cache = {
        instance_id: load_mask(instance["mask_abs_path"])
        for instance_id, instance in instance_by_id.items()
    }
    device = torch.device(device_arg) if device_arg is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    predictor = build_video_predictor(sam2_config, sam2_checkpoint, device)
    numeric_frame_dir = build_numeric_frame_dir(ordered_images, output_path.parent, repeat_count=1)
    try:
        inference_state = predictor.init_state(
            video_path=str(numeric_frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=(device.type == "cpu"),
            async_loading_frames=False,
        )
        unions, pair_summaries = track_associated_instances(
            predictor=predictor,
            inference_state=inference_state,
            frame_names=frame_names,
            frame_to_idx=frame_to_idx,
            instances_by_frame=instances_by_frame,
            instance_by_id=instance_by_id,
            mask_cache=mask_cache,
            match_iou_threshold=match_iou_threshold,
            association_window_frames=association_window_frames,
            association_same_label=association_same_label,
            association_same_frame_exclusivity=association_same_frame_exclusivity,
        )
    finally:
        shutil.rmtree(numeric_frame_dir, ignore_errors=True)
    objects = build_3d_segments(unions=unions, instance_by_id=instance_by_id)
    output = {
        "objects": objects,
        "pairwise_matches": pair_summaries,
    }
    write_json(output_path, output)
    logger.info("Cross-frame association saved to %s", output_path)
    return output_path, objects, pair_summaries


def render_association_overlays(
    instances_json: Path,
    association_json: Path,
    masks_root: Path,
    image_dir: Path,
    output_dir: Path,
    alpha: float,
    draw_outline: bool,
    min_area: int,
) -> int:
    instances = read_instances(instances_json)
    association = json.loads(association_json.read_text(encoding="utf-8"))
    object_by_2d_id = {}
    label_by_object = {}
    for obj in association.get("objects", []):
        object_id = obj.get("objectId")
        if not object_id:
            continue
        label_by_object[object_id] = obj.get("label", object_id)
        for object2d_id in obj.get("object2d_ids", []):
            object_by_2d_id[object2d_id] = object_id
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        frame_name = inst.get("frame_name") or Path(inst.get("image_name", inst.get("image_path", ""))).stem
        if frame_name:
            by_frame[str(frame_name)].append(inst)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    written = 0
    image_lookup = {p.stem: p for p in discover_images([image_dir])}
    for frame_name, rows in sorted(by_frame.items(), key=lambda kv: frame_sort_key(kv[0])):
        image_path = image_lookup.get(frame_name)
        if image_path is None:
            logger.warning("Missing image for association overlay frame: %s", frame_name)
            continue
        base = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        labels_to_draw = []
        for inst in rows:
            object2d_id = str(inst.get("object2d_id", ""))
            object_id = object_by_2d_id.get(object2d_id, object2d_id)
            mask_rel = inst.get("mask_path")
            if mask_rel:
                mask_rel_path = Path(str(mask_rel))
                mask_path = masks_root.parent / mask_rel_path if mask_rel_path.parts and mask_rel_path.parts[0] == masks_root.name else masks_root / mask_rel_path
            else:
                mask_name = object2d_id.split("__")[-1] if "__" in object2d_id else ""
                mask_path = masks_root / frame_name / f"{mask_name}.png"
            if not mask_path.exists():
                logger.warning("Missing mask for association overlay: %s", mask_path)
                continue
            mask = load_mask(mask_path)
            if int(mask.sum()) < min_area:
                continue
            label = str(label_by_object.get(object_id, inst.get("label") or "object"))
            text = f"{object_id}"
            if label and label not in text:
                text = f"{label}:{object_id}"
            color = color_for_key(str(object_id))
            rgba = color + (int(max(0.0, min(1.0, alpha)) * 255),)
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            overlay.paste(Image.new("RGBA", base.size, rgba), (0, 0), mask_img)
            ys, xs = np.where(mask)
            if xs.size:
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                if draw_outline:
                    draw_overlay.rectangle((x0, y0, x1, y1), outline=color + (255,), width=2)
                labels_to_draw.append(((x0, max(0, y0 - 22)), text, color))
        composed = Image.alpha_composite(base, overlay)
        draw_final = ImageDraw.Draw(composed)
        for xy, text, color in labels_to_draw:
            draw_label(draw_final, xy, text, color, font)
        composed.convert("RGB").save(output_dir / f"{frame_name}.association_overlay.jpg", quality=92)
        written += 1
    logger.info("Wrote %d association overlay image(s) to %s", written, output_dir)
    return written


def build_single_frame_objects(instances_2d_path: Path) -> list[dict]:
    instances = read_instances(instances_2d_path)
    objects: list[dict] = []
    label_seen: dict[tuple[str, str], int] = defaultdict(int)

    ordered_instances = sorted(
        instances,
        key=lambda item: (frame_sort_key(str(item.get("frame_name", ""))), str(item.get("mask_name", ""))),
    )
    for instance in ordered_instances:
        label = str(instance.get("label") or "object")
        frame_name = str(instance.get("frame_name", ""))
        label_key = (frame_name, label)
        label_seen[label_key] += 1
        object_id = f"{frame_name}__{label}_{label_seen[label_key]:03d}" if frame_name else f"{label}_{label_seen[label_key]:03d}"
        objects.append(
            {
                "objectId": object_id,
                "label": label,
                "instance_count": 1,
                "frames": [frame_name],
                "object2d_ids": [str(instance["object2d_id"])],
            }
        )
    return objects


def point3d_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def point3d_parse_info_file(info_path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    with info_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    return info


def point3d_parse_matrix_4x4(value: str) -> np.ndarray:
    numbers = [float(item) for item in value.split()]
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers for 4x4 matrix, got {len(numbers)}.")
    return np.asarray(numbers, dtype=np.float64).reshape(4, 4)


def point3d_intrinsic_params_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])


def point3d_discover_color_lookup(color_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in sorted(color_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            lookup[path.stem] = path
    return lookup


def point3d_erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    pixels = int(pixels)
    if pixels <= 0 or not np.any(mask):
        return mask

    eroded = mask.astype(bool)
    for _ in range(pixels):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, :-2]
            & padded[:-2, 2:]
            & padded[2:, :-2]
            & padded[2:, 2:]
        )
        if not np.any(eroded):
            break
    return eroded


def point3d_load_color_shape(color_path: Path) -> tuple[int, int]:
    with Image.open(color_path) as image:
        width, height = image.size
    return height, width


def point3d_load_pose(pose_path: Path, pose_type: str) -> np.ndarray:
    matrix = np.loadtxt(pose_path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Pose file {pose_path} does not contain a 4x4 matrix.")
    if pose_type == "world_to_camera":
        return np.linalg.inv(matrix)
    if pose_type == "camera_to_world":
        return matrix
    raise ValueError(f"Unsupported pose_type: {pose_type}")


def point3d_frame_stem_to_pose_name(frame_name: str) -> str:
    return frame_name.replace(".color", ".pose.txt")


def point3d_resolve_color_frame_path(color_dir: Path, frame_name: str, color_lookup: dict[str, Path]) -> Path:
    candidates = [
        color_lookup.get(frame_name),
        color_dir / f"{frame_name}.png",
        color_dir / f"{frame_name}.jpg",
        color_dir / f"{frame_name}.jpeg",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return color_dir / f"{frame_name}.png"


def point3d_resolve_pose_frame_path(pose_dir: Path, frame_name: str) -> Path:
    candidates = [
        pose_dir / point3d_frame_stem_to_pose_name(frame_name),
        pose_dir / f"{frame_name}.pose.txt",
        pose_dir / f"{frame_name}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def point3d_load_final_keyframe_list(scan_root: Path) -> list[str]:
    keyframes_path = scan_root / "sequence" / "keyframes.txt"
    if not keyframes_path.is_file():
        raise FileNotFoundError(
            f"Final-keyframe list not found: {keyframes_path}. "
            "The 3D extraction requires a fresh MASt3R export containing sequence/keyframes.txt."
        )
    names = [
        line.strip()
        for line in keyframes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"Final-keyframe list is empty: {keyframes_path}")
    return names


def point3d_load_final_keyframe_names(scan_root: Path) -> set[str]:
    return set(point3d_load_final_keyframe_list(scan_root))


def point3d_load_scene_provenance(
    scan_root: Path,
    scene_point_count: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    keyframe_list = point3d_load_final_keyframe_list(scan_root)
    provenance_path = scan_root / "sequence" / "scene.provenance.npz"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Scene provenance not found: {provenance_path}")
    with np.load(provenance_path, allow_pickle=False) as data:
        source_keyframe_index = np.asarray(data["source_keyframe_index"], dtype=np.int64)
        source_pixel_index = np.asarray(data["source_pixel_index"], dtype=np.int64)
    if source_keyframe_index.shape != (scene_point_count,) or source_pixel_index.shape != (scene_point_count,):
        raise ValueError(
            "scene.ply/provenance point-count mismatch: "
            f"scene={scene_point_count}, keyframe_index={source_keyframe_index.size}, "
            f"pixel_index={source_pixel_index.size}"
        )
    if source_keyframe_index.size and (
        int(source_keyframe_index.min()) < 0
        or int(source_keyframe_index.max()) >= len(keyframe_list)
    ):
        raise ValueError("Scene provenance contains an invalid keyframe index.")
    return keyframe_list, source_keyframe_index, source_pixel_index


def point3d_build_provenance_lookup(
    keyframe_list: list[str],
    source_keyframe_index: np.ndarray,
    source_pixel_index: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for keyframe_index, frame_name in enumerate(keyframe_list):
        scene_indices = np.flatnonzero(source_keyframe_index == keyframe_index).astype(np.int64)
        lookup[frame_name] = (scene_indices, source_pixel_index[scene_indices])
    return lookup


def point3d_points_to_lists(points: np.ndarray) -> list:
    return points.astype(np.float32).tolist() if points.size else []


def point3d_sanitize_name(name: str) -> str:
    safe_chars = []
    for char in str(name):
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars) or "object"


def point3d_write_xyz(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def point3d_write_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


POINT3D_PLY_PROPERTY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def point3d_read_scene_ply_xyz_rgb(scene_ply: Path) -> tuple[np.ndarray, np.ndarray]:
    with scene_ply.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {scene_ply}")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        data = f.read()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {scene_ply}")

    ply_format = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            ply_format = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) >= 3 and parts[1] != "list":
                vertex_properties.append((parts[2], parts[1]))

    property_names = [name for name, _ in vertex_properties]
    required_properties = {"x", "y", "z", "red", "green", "blue"}
    if vertex_count <= 0 or not required_properties.issubset(property_names):
        raise ValueError(f"PLY file has no usable vertex x/y/z/red/green/blue scannet: {scene_ply}")

    if ply_format == "ascii":
        values = np.loadtxt(
            data.decode("ascii", errors="ignore").splitlines()[:vertex_count],
            dtype=np.float64,
        )
        if values.ndim == 1:
            values = values[None, :]
        points = values[:, [property_names.index(name) for name in ("x", "y", "z")]].astype(np.float32)
        colors = values[:, [property_names.index(name) for name in ("red", "green", "blue")]].astype(np.float32)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        return points[valid], np.clip(colors[valid], 0, 255).astype(np.uint8)

    if ply_format != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format {ply_format!r}: {scene_ply}")

    dtype_fields = []
    for name, property_type in vertex_properties:
        dtype_code = POINT3D_PLY_PROPERTY_DTYPES.get(property_type)
        if dtype_code is None:
            raise ValueError(f"Unsupported PLY vertex property type {property_type!r}: {scene_ply}")
        dtype_fields.append((name, "<" + dtype_code))

    vertex_dtype = np.dtype(dtype_fields)
    required_bytes = vertex_dtype.itemsize * vertex_count
    if len(data) < required_bytes:
        raise ValueError(f"PLY vertex scannet is truncated: {scene_ply}")
    vertices = np.frombuffer(data[:required_bytes], dtype=vertex_dtype, count=vertex_count)
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.float32)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    return points[valid], np.clip(colors[valid], 0, 255).astype(np.uint8)


def point3d_read_scene_ply_xyz(scene_ply: Path) -> np.ndarray:
    points, _ = point3d_read_scene_ply_xyz_rgb(scene_ply)
    return points


def point3d_project_scene_points_to_frame(
    scene_points: np.ndarray,
    color_intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    pose_type: str,
    color_shape: tuple[int, int],
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, np.ndarray]:
    if pose_type == "camera_to_world":
        world_to_camera = np.linalg.inv(camera_to_world)
    elif pose_type == "world_to_camera":
        world_to_camera = camera_to_world
    else:
        raise ValueError(f"Unsupported pose_type: {pose_type}")

    points_h = np.concatenate(
        [scene_points.astype(np.float64), np.ones((scene_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_camera = (world_to_camera @ points_h.T).T[:, :3]
    z = points_camera[:, 2]
    valid_depth = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid_depth):
        return {
            "indices": np.empty((0,), dtype=np.int64),
            "u": np.empty((0,), dtype=np.int64),
            "v": np.empty((0,), dtype=np.int64),
            "camera_z": np.empty((0,), dtype=np.float32),
        }

    valid_indices = np.nonzero(valid_depth)[0]
    points_camera = points_camera[valid_depth]
    z = z[valid_depth]
    fx, fy, cx, cy = point3d_intrinsic_params_from_matrix(color_intrinsic)
    u = np.rint((points_camera[:, 0] * fx / z) + cx).astype(np.int64)
    v = np.rint((points_camera[:, 1] * fy / z) + cy).astype(np.int64)
    height, width = color_shape
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return {
        "indices": valid_indices[in_frame].astype(np.int64),
        "u": u[in_frame],
        "v": v[in_frame],
        "camera_z": z[in_frame].astype(np.float32),
    }


def point3d_keep_zbuffer_nearest(
    projection: dict[str, np.ndarray],
    color_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Keep only the nearest projected scene point at each integer image pixel."""
    indices = projection["indices"]
    if indices.size == 0:
        return projection
    _, width = color_shape
    pixel_ids = projection["v"] * int(width) + projection["u"]
    order = np.lexsort((projection["camera_z"], pixel_ids))
    ordered_pixels = pixel_ids[order]
    keep_ordered = np.empty(order.size, dtype=bool)
    keep_ordered[0] = True
    keep_ordered[1:] = ordered_pixels[1:] != ordered_pixels[:-1]
    keep = order[keep_ordered]
    return {name: values[keep] for name, values in projection.items()}


def point3d_filter_indices_near_seeds(
    scene_points: np.ndarray,
    candidate_indices: np.ndarray,
    seed_indices: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Return candidate indices no farther than radius_m from any reliable seed."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    seed_indices = np.asarray(seed_indices, dtype=np.int64)
    if candidate_indices.size == 0 or seed_indices.size == 0 or radius_m <= 0:
        return np.empty((0,), dtype=np.int64)
    candidates = scene_points[candidate_indices].astype(np.float64, copy=False)
    seeds = scene_points[seed_indices].astype(np.float64, copy=False)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(seeds).query(candidates, k=1, distance_upper_bound=float(radius_m))
        return candidate_indices[np.isfinite(distances)]
    except ImportError:
        # Exact bounded-memory fallback for environments without scipy.
        radius_sq = float(radius_m) ** 2
        accepted = np.zeros(candidate_indices.size, dtype=bool)
        for candidate_start in range(0, candidates.shape[0], 256):
            candidate_chunk = candidates[candidate_start:candidate_start + 256]
            best_sq = np.full(candidate_chunk.shape[0], np.inf, dtype=np.float64)
            for seed_start in range(0, seeds.shape[0], 2048):
                seed_chunk = seeds[seed_start:seed_start + 2048]
                delta = candidate_chunk[:, None, :] - seed_chunk[None, :, :]
                best_sq = np.minimum(best_sq, np.min(np.sum(delta * delta, axis=2), axis=1))
            accepted[candidate_start:candidate_start + candidate_chunk.shape[0]] = best_sq <= radius_sq
        return candidate_indices[accepted]


def point3d_scene_color_histogram(colors: np.ndarray, bins: int = 8) -> np.ndarray:
    """Return a normalized concatenated RGB histogram for scene points."""
    if colors.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    histogram = np.concatenate([
        np.histogram(colors[:, channel], bins=bins, range=(0, 256))[0].astype(np.float32)
        for channel in range(3)
    ])
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def renumber_association_objects_by_label(objects: list[dict]) -> list[dict]:
    """Return objects with gap-free objectIds within each semantic label.

    Association components receive sequential IDs before the scene-color merge.
    That merge can remove intermediate representatives, so keeping their old IDs
    would leave gaps such as cabinet_1, cabinet_3, cabinet_6. Renumber only after
    all merges are complete; object2d_ids remain unchanged and continue to carry
    the actual association identity.
    """
    label_totals = Counter(
        normalize_label(str(obj.get("label") or "object"))
        for obj in objects
    )
    label_seen = Counter()
    renumbered = []
    for obj in objects:
        updated = dict(obj)
        label = str(updated.get("label") or "object")
        label_key = normalize_label(label)
        label_seen[label_key] += 1
        updated["objectId"] = (
            f"{label}_{label_seen[label_key]}"
            if label_totals[label_key] > 1
            else label
        )
        renumbered.append(updated)
    return renumbered


def merge_association_objects_by_scene_color(
    association_json: Path,
    association_objects: list[dict],
    instances_2d_json: Path,
    scan_root: Path,
    camera_info_file: Path,
    camera_pose_dir: Path,
    pose_type: str,
    min_depth_m: float,
    max_depth_m: float,
) -> list[dict]:
    """Second-pass merge using label, scene RGB, and shared indices.

    Same-frame exclusivity is a hard invariant: two association components may
    be merged only when their complete frame sets are disjoint.  The check is
    repeated against the current union-find roots so transitive merges cannot
    introduce a same-frame conflict.
    """
    scene_ply = scan_root / "sequence" / "scene.ply"
    scene_points, scene_colors = point3d_read_scene_ply_xyz_rgb(scene_ply)
    info = point3d_parse_info_file(camera_info_file)
    intrinsic = point3d_parse_matrix_4x4(info["m_calibrationColorIntrinsic"])
    color_dir = scan_root / "sequence" / "color"
    color_lookup = point3d_discover_color_lookup(color_dir)
    instances = point3d_build_instance_lookup(instances_2d_json)
    base_dir = instances_2d_json.parent
    projection_cache: dict[str, dict[str, np.ndarray]] = {}
    mask_cache: dict[Path, np.ndarray] = {}
    object_stats = []

    for obj in association_objects:
        selected_parts = []
        for object2d_id in obj.get("object2d_ids", []):
            instance = instances.get(str(object2d_id))
            if instance is None:
                continue
            frame_name = str(instance["frame_name"])
            mask_path = point3d_resolve_mask_path(base_dir, instance)
            if mask_path not in mask_cache:
                mask_cache[mask_path] = load_mask(mask_path)
            mask = mask_cache[mask_path]
            if frame_name not in projection_cache:
                color_path = point3d_resolve_color_frame_path(color_dir, frame_name, color_lookup)
                color_shape = point3d_load_color_shape(color_path)
                pose_path = point3d_resolve_pose_frame_path(camera_pose_dir, frame_name)
                pose = point3d_load_pose(pose_path, pose_type)
                projection_cache[frame_name] = point3d_project_scene_points_to_frame(
                    scene_points, intrinsic, pose, pose_type, color_shape, min_depth_m, max_depth_m
                )
            projection = projection_cache[frame_name]
            indices, u, v = projection["indices"], projection["u"], projection["v"]
            inside_mask = mask[v, u] > 0 if indices.size else np.zeros((0,), dtype=bool)
            if np.any(inside_mask):
                selected_parts.append(indices[inside_mask])
        selected = np.unique(np.concatenate(selected_parts)) if selected_parts else np.empty((0,), dtype=np.int64)
        object_stats.append({
            "indices": selected,
            "hist": point3d_scene_color_histogram(scene_colors[selected]),
        })

    parent = list(range(len(association_objects)))
    component_frames: list[set[str]] = []
    for obj in association_objects:
        frames = {
            str(instances[str(object2d_id)]["frame_name"])
            for object2d_id in obj.get("object2d_ids", [])
            if str(object2d_id) in instances
        }
        component_frames.append(frames)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    candidates = []
    for i in range(len(association_objects)):
        for j in range(i + 1, len(association_objects)):
            label_i = normalize_label(str(association_objects[i].get("label", "")))
            label_j = normalize_label(str(association_objects[j].get("label", "")))
            if label_i != label_j:
                continue
            indices_i, indices_j = object_stats[i]["indices"], object_stats[j]["indices"]
            denominator = min(indices_i.size, indices_j.size)
            if denominator == 0:
                continue
            overlap = np.intersect1d(indices_i, indices_j, assume_unique=True).size / float(denominator)
            color_similarity = histogram_similarity(object_stats[i]["hist"], object_stats[j]["hist"])
            if (color_similarity > SCENE_COLOR_MERGE_COLOR_SIMILARITY_THRESHOLD
                    and overlap > SCENE_COLOR_MERGE_INDEX_OVERLAP_THRESHOLD):
                candidates.append((overlap, color_similarity, i, j))

    for overlap, color_similarity, i, j in sorted(candidates, reverse=True):
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            continue
        conflicting_frames = component_frames[root_i] & component_frames[root_j]
        if conflicting_frames:
            logger.info(
                "Scene-color merge rejected by same-frame exclusivity: %s x %s shared_frames=%s",
                association_objects[root_i]["objectId"],
                association_objects[root_j]["objectId"],
                ",".join(sorted(conflicting_frames, key=frame_sort_key)),
            )
            continue
        parent[root_j] = root_i
        component_frames[root_i].update(component_frames[root_j])
        component_frames[root_j].clear()
        logger.info(
            "Scene-color merge: %s <- %s label=%s color_similarity=%.4f scene_index_overlap=%.4f",
            association_objects[root_i]["objectId"], association_objects[root_j]["objectId"],
            association_objects[root_i].get("label", ""), color_similarity, overlap,
        )

    groups: dict[int, list[int]] = {}
    for index in range(len(association_objects)):
        groups.setdefault(find(index), []).append(index)
    merged_objects = []
    for member_indices in groups.values():
        representative = dict(association_objects[member_indices[0]])
        merged_object2d_ids = sorted({
            str(object2d_id)
            for index in member_indices
            for object2d_id in association_objects[index].get("object2d_ids", [])
        })
        merged_frames = sorted(
            {
                str(instances[object2d_id]["frame_name"])
                for object2d_id in merged_object2d_ids
                if object2d_id in instances
            },
            key=frame_sort_key,
        )
        representative["object2d_ids"] = merged_object2d_ids
        representative["frames"] = merged_frames
        representative["instance_count"] = len(merged_object2d_ids)
        merged_objects.append(representative)

    merged_objects = renumber_association_objects_by_label(merged_objects)

    association_data = point3d_read_json(association_json)
    association_data["objects"] = merged_objects
    association_data["scene_color_merge"] = {
        "label_same": True,
        "color_similarity_threshold": SCENE_COLOR_MERGE_COLOR_SIMILARITY_THRESHOLD,
        "scene_index_overlap_threshold": SCENE_COLOR_MERGE_INDEX_OVERLAP_THRESHOLD,
        "same_frame_instances_forbid_merge": True,
    }
    write_json(association_json, association_data)
    logger.info("Scene-color association merge reduced objects from %d to %d.", len(association_objects), len(merged_objects))
    return merged_objects


def point3d_build_instance_lookup(instances_2d_json: Path) -> dict[str, dict]:
    return {str(item["object2d_id"]): item for item in read_instances(instances_2d_json)}


def point3d_resolve_mask_path(base_dir: Path, instance: dict) -> Path:
    mask_rel = instance.get("mask_path")
    if mask_rel:
        return (base_dir / mask_rel).resolve()
    frame_name = str(instance["frame_name"])
    mask_name = str(instance["mask_name"])
    return (base_dir / "merged_masks" / frame_name / f"{mask_name}.png").resolve()


def prune_zero_point_2d_artifacts(
    instances_2d_json: Path,
    removed_object2d_ids: set[str],
    removed_mask_paths: list[Path],
) -> None:
    if not removed_object2d_ids:
        return

    data = point3d_read_json(instances_2d_json)
    if isinstance(data, dict):
        instances = data.get("instances", [])
    elif isinstance(data, list):
        instances = data
    else:
        raise ValueError(f"Invalid instances_2d.json format: {instances_2d_json}")

    kept_instances = [
        inst for inst in instances
        if str(inst.get("object2d_id", "")) not in removed_object2d_ids
    ]
    if isinstance(data, dict):
        data["instances"] = kept_instances
        write_json(instances_2d_json, data)
    else:
        write_json(instances_2d_json, kept_instances)

    for mask_path in removed_mask_paths:
        try:
            if mask_path.exists():
                mask_path.unlink()
                logger.info("Deleted zero-point 2D mask: %s", mask_path)
            parent = mask_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                logger.info("Deleted empty mask directory: %s", parent)
        except OSError as exc:
            logger.warning("Failed to delete zero-point mask artifact %s: %s", mask_path, exc)

    logger.info(
        "Pruned %d zero-point 2D instance(s) from %s.",
        len(removed_object2d_ids),
        instances_2d_json,
    )


def prune_zero_point_3d_raw_artifacts(input_json: Path, removed_object_ids: set[str]) -> None:
    if not removed_object_ids or not input_json.exists():
        return

    data = point3d_read_json(input_json)
    objects = data.get("objects", []) if isinstance(data, dict) else []
    kept_objects = [
        obj for obj in objects
        if str(obj.get("objectId", "")) not in removed_object_ids
    ]
    data["objects"] = kept_objects
    write_json(input_json, data)
    logger.info(
        "Pruned %d zero-point raw 3D object(s) from %s.",
        len(objects) - len(kept_objects),
        input_json,
    )


def run_point3d_projection(
    scan_root: Path,
    instances_2d_json: Path,
    segment_3d_json: Path,
    camera_info_file: Path,
    camera_pose_dir: Path,
    depth_dir: Path,
    output_json: Path,
    output_dir: Path,
    pose_type: str,
    enable_mask_erosion: bool,
    mask_erode_pixels: int,
    seed_expansion_radius_m: float,
    min_depth_m: float,
    max_depth_m: float,
    sample_stride: int,
    max_points_per_object: int,
    depth_shift_arg: float,
) -> Path:
    color_dir = scan_root / "sequence" / "color"
    scene_ply = scan_root / "sequence" / "scene.ply"
    for path, description in (
        (instances_2d_json, "instances_2d.json"),
        (segment_3d_json, "3D_segment.json"),
        (camera_info_file, "_info.txt"),
        (camera_pose_dir, "camera pose directory"),
        (color_dir, "color directory"),
        (scene_ply, "MASt3R scene.ply"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    info = point3d_parse_info_file(camera_info_file)
    color_intrinsic = point3d_parse_matrix_4x4(info["m_calibrationColorIntrinsic"])
    depth_shift = float(depth_shift_arg) if float(depth_shift_arg) > 0 else float(info["m_depthShift"])
    scene_points = point3d_read_scene_ply_xyz(scene_ply)
    keyframe_list, source_keyframe_index, source_pixel_index = point3d_load_scene_provenance(
        scan_root, scene_points.shape[0]
    )
    provenance_lookup = point3d_build_provenance_lookup(
        keyframe_list, source_keyframe_index, source_pixel_index
    )
    final_keyframe_names = set(keyframe_list)
    logger.info("Loaded MASt3R scene PLY: %s points=%d", scene_ply, scene_points.shape[0])
    logger.info("Restricting 3D extraction to %d final keyframes.", len(final_keyframe_names))

    base_dir = instances_2d_json.parent
    instances_by_id = point3d_build_instance_lookup(instances_2d_json)
    segment_3d = point3d_read_json(segment_3d_json)
    color_lookup = point3d_discover_color_lookup(color_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_cache: dict[Path, np.ndarray] = {}
    color_shape_cache: dict[Path, tuple[int, int]] = {}
    pose_cache: dict[Path, np.ndarray] = {}
    visible_projection_cache: dict[str, dict[str, np.ndarray]] = {}
    objects = []

    for obj in segment_3d.get("objects", []):
        object_id = str(obj["objectId"])
        label = str(obj.get("label", object_id))
        object_point_indices = []
        instance_summaries = []
        keyframe_records = []
        skipped_non_keyframe_count = 0

        for object2d_id in obj.get("object2d_ids", []):
            instance = instances_by_id.get(str(object2d_id))
            if instance is None:
                logger.warning("Missing instance metadata for %s", object2d_id)
                continue

            frame_name = str(instance["frame_name"])
            if frame_name not in final_keyframe_names:
                skipped_non_keyframe_count += 1
                continue
            mask_path = point3d_resolve_mask_path(base_dir, instance)
            color_path = point3d_resolve_color_frame_path(color_dir, frame_name, color_lookup)
            pose_path = point3d_resolve_pose_frame_path(camera_pose_dir, frame_name)
            missing = [path for path in (mask_path, color_path, pose_path) if not path.exists()]
            if missing:
                logger.warning("Skipping %s because required file(s) are missing: %s", object2d_id, missing)
                continue

            mask = mask_cache.setdefault(mask_path, load_mask(mask_path))
            color_shape = color_shape_cache.setdefault(color_path, point3d_load_color_shape(color_path))
            camera_to_world = pose_cache.setdefault(pose_path, point3d_load_pose(pose_path, pose_type))
            if mask.shape[:2] != color_shape:
                raise ValueError(f"Mask shape {mask.shape[:2]} does not match color shape {color_shape} for {frame_name}.")
            erode_pixels = mask_erode_pixels if enable_mask_erosion else 0
            eroded_mask = point3d_erode_mask(mask, erode_pixels)

            source_scene_indices, source_pixels = provenance_lookup[frame_name]
            if source_pixels.size and int(source_pixels.max()) >= eroded_mask.size:
                raise ValueError(f"Provenance pixel index exceeds mask size for {frame_name}.")
            inside_mask = (
                eroded_mask.reshape(-1)[source_pixels] > 0
                if source_scene_indices.size
                else np.zeros((0,), dtype=bool)
            )
            selected_indices = source_scene_indices[inside_mask]
            if selected_indices.size:
                object_point_indices.append(selected_indices)

            keyframe_records.append(
                {
                    "frame_name": frame_name,
                    "eroded_mask": eroded_mask,
                    "color_shape": color_shape,
                    "camera_to_world": camera_to_world,
                    "seed_indices": selected_indices,
                }
            )

            instance_summaries.append(
                {
                    "object2d_id": str(object2d_id),
                    "frame_name": frame_name,
                    "mask_path": str(mask_path),
                    "color_path": str(color_path),
                    "pose_path": str(pose_path),
                    "raw_mask_pixel_count": int(mask.sum()),
                    "eroded_mask_pixel_count": int(eroded_mask.sum()),
                    "source_keyframe_scene_point_count": int(source_scene_indices.shape[0]),
                    "provenance_seed_point_count": int(selected_indices.shape[0]),
                }
            )

        object_seed_indices = (
            np.unique(np.concatenate(object_point_indices))
            if object_point_indices
            else np.empty((0,), dtype=np.int64)
        )
        supplemental_parts = []
        for record, summary in zip(keyframe_records, instance_summaries):
            frame_name = record["frame_name"]
            if frame_name not in visible_projection_cache:
                projection = point3d_project_scene_points_to_frame(
                    scene_points=scene_points,
                    color_intrinsic=color_intrinsic,
                    camera_to_world=record["camera_to_world"],
                    pose_type=pose_type,
                    color_shape=record["color_shape"],
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
                visible_projection_cache[frame_name] = point3d_keep_zbuffer_nearest(
                    projection, record["color_shape"]
                )
            visible = visible_projection_cache[frame_name]
            inside_mask = (
                record["eroded_mask"][visible["v"], visible["u"]] > 0
                if visible["indices"].size
                else np.zeros((0,), dtype=bool)
            )
            mask_candidates = visible["indices"][inside_mask]
            supplemental_indices = point3d_filter_indices_near_seeds(
                scene_points,
                mask_candidates,
                object_seed_indices,
                seed_expansion_radius_m,
            )
            if supplemental_indices.size:
                supplemental_parts.append(supplemental_indices)
            summary["zbuffer_mask_candidate_count"] = int(mask_candidates.size)
            summary["supplemental_point_count"] = int(supplemental_indices.size)
            summary["selected_scene_point_count"] = int(
                np.union1d(record["seed_indices"], supplemental_indices).size
            )
            summary["point_count"] = summary["selected_scene_point_count"]

        all_index_parts = object_point_indices + supplemental_parts

        if all_index_parts:
            merged_indices = np.unique(np.concatenate(all_index_parts, axis=0))
            if sample_stride > 1:
                merged_indices = merged_indices[::sample_stride]
            merged_points = scene_points[merged_indices]
        else:
            merged_points = np.empty((0, 3), dtype=np.float32)
        if max_points_per_object > 0 and merged_points.shape[0] > max_points_per_object:
            keep_indices = np.linspace(0, merged_points.shape[0] - 1, num=max_points_per_object, dtype=np.int64)
            merged_points = merged_points[keep_indices]

        safe_name = point3d_sanitize_name(object_id)
        xyz_path = output_dir / f"{safe_name}.xyz"
        ply_path = output_dir / f"{safe_name}.ply"
        point3d_write_xyz(xyz_path, merged_points)
        point3d_write_ply(ply_path, merged_points)

        objects.append(
            {
                "objectId": object_id,
                "label": label,
                "instance_count": len(obj.get("object2d_ids", [])),
                "projected_keyframe_instance_count": len(instance_summaries),
                "skipped_non_keyframe_instance_count": skipped_non_keyframe_count,
                "point_count": int(merged_points.shape[0]),
                "xyz_path": str(xyz_path),
                "ply_path": str(ply_path),
                "points": point3d_points_to_lists(merged_points),
                "instances": instance_summaries,
            }
        )
        logger.info("Projected objectId=%s label=%s points=%d", object_id, label, merged_points.shape[0])

    write_json(
        output_json,
        {
            "scan_root": str(scan_root),
            "scene_ply": str(scene_ply),
            "instances_2d_json": str(instances_2d_json),
            "segment_3d_json": str(segment_3d_json),
            "camera_info_file": str(camera_info_file),
            "camera_pose_dir": str(camera_pose_dir),
            "depth_dir": str(depth_dir),
            "pose_type": pose_type,
            "point_source": "provenance_seeds_plus_mask_zbuffer_seed_radius_projection",
            "projection_frames": "final_keyframes_only",
            "final_keyframes_file": str(scan_root / "sequence" / "keyframes.txt"),
            "scene_provenance_file": str(scan_root / "sequence" / "scene.provenance.npz"),
            "final_keyframe_count": len(final_keyframe_names),
            "pose_transform_used": "inverse(pose) @ scene_point" if pose_type == "camera_to_world" else "pose @ scene_point",
            "enable_mask_erosion": bool(enable_mask_erosion),
            "mask_erode_pixels": int(mask_erode_pixels),
            "seed_expansion_radius_m": float(seed_expansion_radius_m),
            "depth_shift": depth_shift,
            "min_depth_m": float(min_depth_m),
            "max_depth_m": float(max_depth_m),
            "sample_stride": int(sample_stride),
            "objects": objects,
        },
    )
    return output_json



def clean3d_ensure_points(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    array = array.reshape(-1, 3)
    valid = np.isfinite(array).all(axis=1)
    return array[valid]


def clean3d_mask_area(mask_path: str | None) -> int:
    if not mask_path:
        return 0
    path = Path(mask_path)
    if not path.exists():
        return 0
    try:
        return int((np.array(Image.open(path).convert("L")) > 0).sum())
    except Exception:
        return 0


def clean3d_split_object_points_by_instance(obj: dict) -> list[dict]:
    points = clean3d_ensure_points(obj.get("points", []))
    instances = obj.get("instances", [])
    expected = sum(max(0, int(inst.get("point_count", 0))) for inst in instances)
    rows = []

    if points.shape[0] == expected and instances:
        cursor = 0
        for inst in instances:
            count = max(0, int(inst.get("point_count", 0)))
            inst_points = points[cursor:cursor + count]
            cursor += count
            rows.append({"instance": inst, "points": clean3d_ensure_points(inst_points)})
        return rows

    rows.append(
        {
            "instance": {
                "object2d_id": "__aggregate__",
                "frame_name": "__aggregate__",
                "point_count": int(points.shape[0]),
            },
            "points": points,
        }
    )
    return rows


def clean3d_instance_stats(rows: list[dict], min_instance_points: int) -> list[dict]:
    stats = []
    for row in rows:
        pts = clean3d_ensure_points(row["points"])
        inst = row["instance"]
        area = clean3d_mask_area(inst.get("mask_path"))
        if area <= 0:
            area = int(inst.get("point_count", pts.shape[0]) or pts.shape[0])

        if pts.shape[0] > 0:
            centroid = np.mean(pts, axis=0)
            extent = np.ptp(pts, axis=0)
            max_extent = float(np.max(extent))
        else:
            centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            extent = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            max_extent = 0.0

        stats.append(
            {
                "row": row,
                "object2d_id": inst.get("object2d_id"),
                "frame_name": inst.get("frame_name"),
                "mask_path": inst.get("mask_path"),
                "point_count": int(pts.shape[0]),
                "mask_area": int(area),
                "centroid": centroid,
                "extent": extent,
                "max_extent": max_extent,
                "valid": pts.shape[0] >= min_instance_points and np.isfinite(centroid).all(),
                "reject_reasons": [],
            }
        )
    return stats


def clean3d_robust_centroid_threshold(distances: np.ndarray, hard_threshold: float, mad_multiplier: float) -> float:
    if distances.size == 0:
        return hard_threshold
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    robust = median + mad_multiplier * max(mad, 1e-6)
    return min(float(hard_threshold), robust)


def clean3d_filter_object_instances(
    obj: dict,
    area_ratio_threshold: float,
    centroid_distance_m: float,
    centroid_mad_multiplier: float,
    extent_ratio_threshold: float,
    min_instance_points: int,
    min_keep_instances: int,
    clean_mode: str,
) -> tuple[np.ndarray, dict, list[dict]]:
    rows = clean3d_split_object_points_by_instance(obj)
    stats = clean3d_instance_stats(rows, min_instance_points=min_instance_points)
    valid_stats = [item for item in stats if item["valid"]]

    if not valid_stats:
        merged = clean3d_ensure_points(obj.get("points", []))
        return merged, {"mode": "fallback_no_valid_instances", "kept_instances": 0, "rejected_instances": 0}, []

    areas = np.array([item["mask_area"] for item in valid_stats if item["mask_area"] > 0], dtype=np.float32)
    median_area = float(np.median(areas)) if areas.size else 0.0
    centroids = np.stack([item["centroid"] for item in valid_stats], axis=0)
    median_centroid = np.median(centroids, axis=0)
    distances = np.linalg.norm(centroids - median_centroid[None, :], axis=1)
    centroid_threshold = clean3d_robust_centroid_threshold(
        distances,
        hard_threshold=centroid_distance_m,
        mad_multiplier=centroid_mad_multiplier,
    )
    extents = np.array([item["max_extent"] for item in valid_stats if item["max_extent"] > 0], dtype=np.float32)
    median_extent = float(np.median(extents)) if extents.size else 0.0

    distance_by_id = {id(item): float(distance) for item, distance in zip(valid_stats, distances)}
    kept = []
    rejected = []
    for item in stats:
        if not item["valid"]:
            item["reject_reasons"].append("too_few_points_or_invalid_centroid")
        elif clean_mode == "strict":
            distance = distance_by_id[id(item)]
            item["centroid_distance_m"] = distance
            if median_area > 0 and item["mask_area"] > median_area * area_ratio_threshold:
                item["reject_reasons"].append("mask_area_too_large")
            if median_area > 0 and item["mask_area"] < median_area / area_ratio_threshold:
                item["reject_reasons"].append("mask_area_too_small")
            if distance > centroid_threshold:
                item["reject_reasons"].append("centroid_too_far")
            if median_extent > 0 and item["max_extent"] > median_extent * extent_ratio_threshold:
                item["reject_reasons"].append("extent_too_large")
        else:
            item["centroid_distance_m"] = distance_by_id.get(id(item), 0.0)

        if item["reject_reasons"]:
            rejected.append(item)
        else:
            kept.append(item)

    if len(kept) < min_keep_instances:
        valid_sorted = sorted(valid_stats, key=lambda item: distance_by_id.get(id(item), float("inf")))
        kept = valid_sorted[:max(1, min_keep_instances)]
        rejected = [item for item in stats if item not in kept]
        for item in kept:
            item["reject_reasons"] = []

    kept_points = [clean3d_ensure_points(item["row"]["points"]) for item in kept]
    merged_points = np.concatenate(kept_points, axis=0) if kept_points else np.empty((0, 3), dtype=np.float32)

    summary = {
        "mode": "per_instance_consistency_filter" if clean_mode == "strict" else "preserve_valid_instances",
        "clean_mode": clean_mode,
        "input_instances": len(stats),
        "kept_instances": len(kept),
        "rejected_instances": len(rejected),
        "median_mask_area": round(median_area, 4),
        "median_centroid": np.asarray(median_centroid, dtype=np.float32).round(6).tolist(),
        "centroid_distance_threshold_m": round(float(centroid_threshold), 4),
        "median_instance_max_extent_m": round(median_extent, 4),
    }
    rejected_rows = [
        {
            "object2d_id": item["object2d_id"],
            "frame_name": item["frame_name"],
            "mask_path": item["mask_path"],
            "point_count": item["point_count"],
            "mask_area": item["mask_area"],
            "centroid": np.asarray(item["centroid"], dtype=np.float32).round(6).tolist(),
            "extent": np.asarray(item["extent"], dtype=np.float32).round(6).tolist(),
            "centroid_distance_m": round(float(item.get("centroid_distance_m", -1.0)), 4),
            "reject_reasons": item["reject_reasons"],
        }
        for item in rejected
    ]
    return merged_points.astype(np.float32), summary, rejected_rows


def clean3d_format_rejected_instance(item: dict, reason: str) -> dict:
    reasons = list(item.get("reject_reasons", []))
    if reason not in reasons:
        reasons.append(reason)
    return {
        "object2d_id": item["object2d_id"],
        "frame_name": item["frame_name"],
        "mask_path": item["mask_path"],
        "point_count": item["point_count"],
        "mask_area": item["mask_area"],
        "centroid": np.asarray(item["centroid"], dtype=np.float32).round(6).tolist(),
        "extent": np.asarray(item["extent"], dtype=np.float32).round(6).tolist(),
        "centroid_distance_m": round(float(item.get("centroid_distance_m", -1.0)), 4),
        "reject_reasons": reasons,
    }


def clean3d_connected_components_by_distance(centroids: np.ndarray, distance_m: float) -> list[list[int]]:
    count = int(centroids.shape[0])
    if count == 0:
        return []

    threshold = max(float(distance_m), 1e-6)
    visited = np.zeros(count, dtype=bool)
    components = []
    for start in range(count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            distances = np.linalg.norm(centroids - centroids[idx][None, :], axis=1)
            neighbors = np.where((distances <= threshold) & (~visited))[0]
            for neighbor in neighbors.tolist():
                visited[neighbor] = True
                stack.append(int(neighbor))
        components.append(sorted(component))
    return components


def clean3d_make_cluster_object(obj: dict, cluster_items: list[dict], object_id: str) -> dict:
    cluster_points = [clean3d_ensure_points(item["row"]["points"]) for item in cluster_items]
    points = np.concatenate(cluster_points, axis=0) if cluster_points else np.empty((0, 3), dtype=np.float32)
    instances = []
    for item in cluster_items:
        inst = dict(item["row"]["instance"])
        inst["point_count"] = int(clean3d_ensure_points(item["row"]["points"]).shape[0])
        instances.append(inst)

    return {
        "objectId": object_id,
        "label": obj.get("label", obj.get("objectId", object_id)),
        "instance_count": len(instances),
        "point_count": int(points.shape[0]),
        "points": point3d_points_to_lists(points),
        "instances": instances,
    }


def clean3d_spatial_cluster_object_instances(
    obj: dict,
    cluster_distance_m: float,
    cluster_min_instances: int,
    split_spatial_clusters: bool,
) -> list[tuple[dict, dict, list[dict]]]:
    rows = clean3d_split_object_points_by_instance(obj)
    stats = clean3d_instance_stats(rows, min_instance_points=1)
    valid_stats = [item for item in stats if item["valid"]]
    base_object_id = str(obj["objectId"])

    if len(valid_stats) <= 1:
        summary = {
            "enabled": True,
            "mode": "single_or_no_valid_instance",
            "input_instances": len(stats),
            "valid_instances": len(valid_stats),
            "cluster_count": len(valid_stats),
            "accepted_cluster_count": len(valid_stats),
            "cluster_id": 1 if valid_stats else 0,
            "cluster_instances": len(valid_stats),
        }
        return [(obj, summary, [])]

    centroids = np.stack([item["centroid"] for item in valid_stats], axis=0)
    components = clean3d_connected_components_by_distance(centroids, distance_m=cluster_distance_m)
    components = sorted(
        components,
        key=lambda comp: (-len(comp), min(str(valid_stats[idx]["frame_name"]) for idx in comp)),
    )

    min_instances = max(1, int(cluster_min_instances))
    accepted = [comp for comp in components if len(comp) >= min_instances]
    if not accepted and components:
        accepted = [components[0]]
    if not split_spatial_clusters and accepted:
        accepted = [accepted[0]]

    accepted_ids = {id(valid_stats[idx]) for comp in accepted for idx in comp}
    rejected_instances = [
        clean3d_format_rejected_instance(item, reason="different_spatial_cluster")
        for item in stats
        if id(item) not in accepted_ids
    ]

    results = []
    for cluster_number, comp in enumerate(accepted, start=1):
        cluster_items = [valid_stats[idx] for idx in comp]
        cluster_centroids = np.stack([item["centroid"] for item in cluster_items], axis=0)
        object_id = base_object_id
        if split_spatial_clusters and len(accepted) > 1:
            object_id = f"{base_object_id}__cluster_{cluster_number:02d}"

        cluster_obj = clean3d_make_cluster_object(obj, cluster_items, object_id=object_id)
        summary = {
            "enabled": True,
            "mode": "instance_centroid_connected_components",
            "input_instances": len(stats),
            "valid_instances": len(valid_stats),
            "cluster_count": len(components),
            "accepted_cluster_count": len(accepted),
            "cluster_id": cluster_number,
            "cluster_instances": len(cluster_items),
            "cluster_distance_m": float(cluster_distance_m),
            "cluster_min_instances": min_instances,
            "split_spatial_clusters": bool(split_spatial_clusters),
            "cluster_centroid": np.median(cluster_centroids, axis=0).astype(np.float32).round(6).tolist(),
            "cluster_object2d_ids": [str(item["object2d_id"]) for item in cluster_items],
        }
        cluster_rejected = [] if split_spatial_clusters else rejected_instances
        results.append((cluster_obj, summary, cluster_rejected))

    return results


def clean3d_deduplicate_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    return np.unique(points, axis=0).astype(np.float32)


def clean3d_uniform_subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices]


def clean3d_voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    voxel_map = {}
    for point, voxel_index in zip(points, voxel_indices):
        voxel_map.setdefault(tuple(voxel_index.tolist()), []).append(point)
    return np.asarray([np.mean(voxel_points, axis=0) for voxel_points in voxel_map.values()], dtype=np.float32)


def clean3d_pairwise_distances(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    diffs = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2, dtype=np.float32)).astype(np.float32)


def clean3d_statistical_outlier_removal(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    num_points = points.shape[0]
    if num_points == 0 or num_points <= max(3, nb_neighbors):
        return points
    distances = clean3d_pairwise_distances(points)
    sorted_distances = np.sort(distances, axis=1)
    neighbor_count = min(nb_neighbors + 1, num_points)
    mean_knn = np.mean(sorted_distances[:, 1:neighbor_count], axis=1)
    threshold = float(np.mean(mean_knn)) + std_ratio * float(np.std(mean_knn))
    filtered = points[mean_knn <= threshold]
    return points if filtered.shape[0] == 0 else filtered


def clean3d_radius_outlier_removal(points: np.ndarray, radius: float, min_points_in_radius: int) -> np.ndarray:
    num_points = points.shape[0]
    if num_points == 0 or radius <= 0 or num_points <= min_points_in_radius:
        return points
    distances = clean3d_pairwise_distances(points)
    neighbor_counts = np.sum(distances <= radius, axis=1) - 1
    filtered = points[neighbor_counts >= min_points_in_radius]
    return points if filtered.shape[0] == 0 else filtered


def clean3d_clean_points(
    points: np.ndarray,
    voxel_size: float,
    nb_neighbors: int,
    std_ratio: float,
    radius: float,
    min_points_in_radius: int,
    min_points_after_filter: int,
    max_points_for_knn: int,
    clean_mode: str,
) -> tuple[np.ndarray, dict]:
    stage_raw = clean3d_ensure_points(points)
    stage_dedup = clean3d_deduplicate_points(stage_raw)
    stage_voxel = clean3d_voxel_downsample(stage_dedup, voxel_size)
    if clean_mode == "preserve":
        final_points = stage_voxel if stage_voxel.shape[0] >= min_points_after_filter else stage_dedup
        final_stage = "voxel_downsample" if final_points is stage_voxel else "deduplicate_points"
        stats = {
            "clean_mode": clean_mode,
            "raw_count_after_instance_filter": int(stage_raw.shape[0]),
            "deduplicated_count": int(stage_dedup.shape[0]),
            "voxel_downsampled_count": int(stage_voxel.shape[0]),
            "knn_input_count": 0,
            "statistical_filtered_count": 0,
            "radius_filtered_count": 0,
            "final_count": int(final_points.shape[0]),
            "final_stage": final_stage,
            "skipped_strict_outlier_filters": True,
        }
        return final_points.astype(np.float32), stats

    stage_for_knn = clean3d_uniform_subsample(stage_voxel, max_points_for_knn)
    stage_stat = clean3d_statistical_outlier_removal(stage_for_knn, nb_neighbors, std_ratio)
    stage_radius = clean3d_radius_outlier_removal(stage_stat, radius, min_points_in_radius)

    final_points = stage_radius
    final_stage = "radius_outlier_removal"
    if final_points.shape[0] < min_points_after_filter:
        if stage_stat.shape[0] >= min_points_after_filter:
            final_points = stage_stat
            final_stage = "statistical_outlier_removal"
        elif stage_voxel.shape[0] >= min_points_after_filter:
            final_points = stage_voxel
            final_stage = "voxel_downsample"
        else:
            final_points = stage_dedup
            final_stage = "deduplicate_points"

    stats = {
        "clean_mode": clean_mode,
        "raw_count_after_instance_filter": int(stage_raw.shape[0]),
        "deduplicated_count": int(stage_dedup.shape[0]),
        "voxel_downsampled_count": int(stage_voxel.shape[0]),
        "knn_input_count": int(stage_for_knn.shape[0]),
        "statistical_filtered_count": int(stage_stat.shape[0]),
        "radius_filtered_count": int(stage_radius.shape[0]),
        "final_count": int(final_points.shape[0]),
        "final_stage": final_stage,
    }
    return final_points.astype(np.float32), stats


def run_point3d_cleaning(
    input_json: Path,
    output_json: Path,
    output_dir: Path,
    area_ratio_threshold: float,
    centroid_distance_m: float,
    centroid_mad_multiplier: float,
    extent_ratio_threshold: float,
    min_instance_points: int,
    min_keep_instances: int,
    clean_mode: str,
    spatial_cluster_instances: bool,
    cluster_distance_m: float,
    cluster_min_instances: int,
    split_spatial_clusters: bool,
    voxel_size: float,
    nb_neighbors: int,
    std_ratio: float,
    radius: float,
    min_points_in_radius: int,
    min_points_after_filter: int,
    max_points_for_knn: int,
) -> Path:
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    data = point3d_read_json(input_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_objects = []
    removed_clean_object_ids: set[str] = set()
    removed_clean_object2d_ids: set[str] = set()
    removed_clean_mask_paths: list[Path] = []
    for obj in data.get("objects", []):
        cluster_inputs = [(obj, {"enabled": False}, [])]
        if spatial_cluster_instances:
            cluster_inputs = clean3d_spatial_cluster_object_instances(
                obj=obj,
                cluster_distance_m=cluster_distance_m,
                cluster_min_instances=cluster_min_instances,
                split_spatial_clusters=split_spatial_clusters,
            )

        for cluster_obj, spatial_cluster_stats, spatial_rejected_instances in cluster_inputs:
            object_id = cluster_obj["objectId"]
            label = cluster_obj.get("label", object_id)
            instance_filtered_points, instance_filter_stats, rejected_instances = clean3d_filter_object_instances(
                obj=cluster_obj,
                area_ratio_threshold=area_ratio_threshold,
                centroid_distance_m=centroid_distance_m,
                centroid_mad_multiplier=centroid_mad_multiplier,
                extent_ratio_threshold=extent_ratio_threshold,
                min_instance_points=min_instance_points,
                min_keep_instances=min_keep_instances,
                clean_mode=clean_mode,
            )
            cleaned_points, point_filter_stats = clean3d_clean_points(
                points=instance_filtered_points,
                voxel_size=voxel_size,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
                radius=radius,
                min_points_in_radius=min_points_in_radius,
                min_points_after_filter=min_points_after_filter,
                max_points_for_knn=max_points_for_knn,
                clean_mode=clean_mode,
            )

            safe_name = point3d_sanitize_name(object_id)
            xyz_path = output_dir / f"{safe_name}.xyz"
            ply_path = output_dir / f"{safe_name}.ply"

            if cleaned_points.shape[0] == 0:
                removed_clean_object_ids.add(str(obj.get("objectId", object_id)))
                for instance in cluster_obj.get("instances", []):
                    object2d_id = str(instance.get("object2d_id", ""))
                    if object2d_id:
                        removed_clean_object2d_ids.add(object2d_id)
                    mask_path_value = instance.get("mask_path")
                    if mask_path_value:
                        removed_clean_mask_paths.append(Path(str(mask_path_value)))
                stale_paths = [xyz_path, ply_path]
                for path_key in ("xyz_path", "ply_path"):
                    path_value = cluster_obj.get(path_key)
                    if path_value:
                        stale_paths.append(Path(str(path_value)))
                for stale_path in stale_paths:
                    try:
                        if stale_path.exists() and stale_path.is_file():
                            stale_path.unlink()
                            logger.info("Deleted stale zero-point 3D file: %s", stale_path)
                    except OSError as exc:
                        logger.warning("Failed to delete stale zero-point 3D file %s: %s", stale_path, exc)
                logger.warning(
                    "Skipping cleaned objectId=%s label=%s because clean point count is 0; related artifacts will be pruned.",
                    object_id,
                    label,
                )
                continue

            point3d_write_xyz(xyz_path, cleaned_points)
            point3d_write_ply(ply_path, cleaned_points)

            cleaned_objects.append(
                {
                    "objectId": object_id,
                    "source_objectId": obj.get("objectId"),
                    "label": label,
                    "instance_count": cluster_obj.get("instance_count"),
                    "raw_point_count": int(cluster_obj.get("point_count", len(cluster_obj.get("points", [])))),
                    "instance_filtered_point_count": int(instance_filtered_points.shape[0]),
                    "clean_point_count": int(cleaned_points.shape[0]),
                    "spatial_cluster_stats": spatial_cluster_stats,
                    "spatial_rejected_instances": spatial_rejected_instances,
                    "instance_filter_stats": instance_filter_stats,
                    "rejected_instances": rejected_instances,
                    "point_filter_stats": point_filter_stats,
                    "xyz_path": str(xyz_path),
                    "ply_path": str(ply_path),
                    "instances": cluster_obj.get("instances", []),
                    "points": point3d_points_to_lists(cleaned_points),
                }
            )
            logger.info(
                "Cleaned %s: raw=%s clean=%d",
                object_id,
                cluster_obj.get("point_count"),
                cleaned_points.shape[0],
            )

    if removed_clean_object_ids:
        prune_zero_point_3d_raw_artifacts(input_json, removed_clean_object_ids)
        instances_2d_json = data.get("instances_2d_json")
        if instances_2d_json:
            prune_zero_point_2d_artifacts(Path(str(instances_2d_json)), removed_clean_object2d_ids, removed_clean_mask_paths)

    output = {
        "input_json": str(input_json),
        "clean_mode": clean_mode,
        "area_ratio_threshold": area_ratio_threshold,
        "centroid_distance_m": centroid_distance_m,
        "centroid_mad_multiplier": centroid_mad_multiplier,
        "extent_ratio_threshold": extent_ratio_threshold,
        "min_instance_points": min_instance_points,
        "spatial_cluster_instances": spatial_cluster_instances,
        "cluster_distance_m": cluster_distance_m,
        "cluster_min_instances": cluster_min_instances,
        "split_spatial_clusters": split_spatial_clusters,
        "voxel_size": voxel_size,
        "nb_neighbors": nb_neighbors,
        "std_ratio": std_ratio,
        "radius": radius,
        "min_points_in_radius": min_points_in_radius,
        "min_points_after_filter": min_points_after_filter,
        "max_points_for_knn": max_points_for_knn,
        "objects": cleaned_objects,
    }
    write_json(output_json, output)
    return output_json


def obb_frame_base_from_name(name: str) -> str:
    stem = Path(str(name)).stem
    if stem.endswith(".color"):
        stem = stem[: -len(".color")]
    return stem


def obb_object2d_to_clean_object_map(point3d_clean_json: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for obj in point3d_read_json(point3d_clean_json).get("objects", []):
        object_id = str(obj.get("objectId", ""))
        for inst in obj.get("instances", []):
            object2d_id = str(inst.get("object2d_id", ""))
            if object2d_id:
                mapping[object2d_id] = object_id
    return mapping


def obb_ensure_points(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    array = array.reshape(-1, 3)
    valid = np.isfinite(array).all(axis=1)
    return array[valid]


def obb_normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return fallback.copy()
    return vector / norm


def obb_orthonormalize_axes(axes: np.ndarray) -> np.ndarray:
    axis0 = obb_normalize_vector(axes[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    axis1 = axes[:, 1] - np.dot(axes[:, 1], axis0) * axis0
    axis1 = obb_normalize_vector(axis1, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    axis2 = obb_normalize_vector(np.cross(axis0, axis1), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if np.linalg.det(np.column_stack([axis0, axis1, axis2])) < 0:
        axis2 = -axis2
    return np.column_stack([axis0, axis1, axis2])


def obb_fallback(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return (mins + maxs) / 2.0, np.maximum(maxs - mins, 0.0), np.eye(3, dtype=np.float64)


def obb_compute(
    points: np.ndarray,
    eigenvalue_eps: float,
    min_points_for_pca: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], str]:
    if points.shape[0] < min_points_for_pca:
        return obb_fallback(points), "fallback_insufficient_points"
    center = np.mean(points, axis=0)
    centered = points - center
    if np.allclose(centered, 0.0):
        return obb_fallback(points), "fallback_zero_variance"
    covariance = np.cov(centered, rowvar=False, bias=True)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return obb_fallback(points), "fallback_invalid_covariance"
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, eigenvalue_eps)
    rotation = obb_orthonormalize_axes(eigenvectors[:, np.argsort(eigenvalues)[::-1]])
    projected = centered @ rotation
    mins = np.min(projected, axis=0)
    maxs = np.max(projected, axis=0)
    axes_lengths = np.maximum(maxs - mins, 0.0)
    centroid = center + rotation @ ((mins + maxs) / 2.0)
    return (centroid, axes_lengths, rotation), "pca_obb"


def obb_to_float_list(array: np.ndarray) -> list[float]:
    return [float(x) for x in array.tolist()]


def obb_axes_matrix_to_list(rotation: np.ndarray) -> list[float]:
    if rotation.shape != (3, 3):
        rotation = np.eye(3, dtype=np.float64)
    return obb_to_float_list(rotation.reshape(-1))


def obb_has_valid_floor_footprint(
    centroid: np.ndarray,
    axes_lengths: np.ndarray,
    rotation: np.ndarray,
    eps: float = 1e-12,
) -> bool:
    if centroid.shape != (3,) or axes_lengths.shape != (3,) or rotation.shape != (3, 3):
        return False
    if not (np.isfinite(centroid).all() and np.isfinite(axes_lengths).all() and np.isfinite(rotation).all()):
        return False
    if np.any(axes_lengths <= eps):
        return False
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    corners = centroid + signs @ np.diag(axes_lengths * 0.5) @ rotation.T
    footprint = corners[:, :2]
    if np.unique(np.round(footprint, decimals=18), axis=0).shape[0] < 3:
        return False
    return np.linalg.matrix_rank(footprint - footprint.mean(axis=0), tol=eps) >= 2


def obb_compute_data(
    input_json: Path,
    objects: list[dict[str, Any]],
    eigenvalue_eps: float,
    min_points_for_pca: int,
) -> dict[str, Any]:
    obb_objects = []
    for obj in objects:
        points = obb_ensure_points(obj.get("points", []))
        if points.shape[0] < min_points_for_pca:
            logger.info(
                "[OBB_SKIP] objectId=%s label=%s points=%d below min_points_for_pca=%d",
                obj.get("objectId"),
                obj.get("label"),
                points.shape[0],
                min_points_for_pca,
            )
            continue
        (centroid, axes_lengths, rotation), method = obb_compute(points, eigenvalue_eps, min_points_for_pca)
        if not obb_has_valid_floor_footprint(centroid, axes_lengths, rotation):
            logger.info(
                "[OBB_SKIP] objectId=%s label=%s points=%d has degenerate floor footprint",
                obj.get("objectId"),
                obj.get("label"),
                points.shape[0],
            )
            continue
        obb_objects.append(
            {
                "objectId": str(obj.get("objectId")),
                "source_objectId": obj.get("source_objectId"),
                "label": str(obj.get("label", "object")),
                "point_count": int(points.shape[0]),
                "clean_point_count": int(obj.get("clean_point_count", points.shape[0])),
                "obb_method": method,
                "obb_source": "3D_point_clean",
                "instances": obj.get("instances", []),
                "obb": {
                    "centroid": obb_to_float_list(centroid),
                    "axesLengths": obb_to_float_list(axes_lengths),
                    "normalizedAxes": obb_axes_matrix_to_list(rotation),
                },
            }
        )
        logger.info("[OBB] objectId=%s label=%s points=%d method=%s", obj.get("objectId"), obj.get("label"), points.shape[0], method)
    return {
        "input_json": str(input_json),
        "source_stage": "8_clean_3d_points",
        "eigenvalue_eps": eigenvalue_eps,
        "min_points_for_pca": min_points_for_pca,
        "objects": obb_objects,
    }


def obb_prepare_clean_objects_for_semantic_obb(
    input_json: Path,
    objects: list[dict[str, Any]],
    min_points_for_obb: int,
) -> dict[str, Any]:
    obb_objects = []
    for obj in objects:
        points = obb_ensure_points(obj.get("points", []))
        if points.shape[0] < min_points_for_obb:
            logger.info(
                "[OBB_SKIP] objectId=%s label=%s points=%d below min_points_for_obb=%d",
                obj.get("objectId"),
                obj.get("label"),
                points.shape[0],
                min_points_for_obb,
            )
            continue
        centroid, axes_lengths, rotation = obb_fallback(points)
        obb_objects.append(
            {
                "objectId": str(obj.get("objectId")),
                "source_objectId": obj.get("source_objectId"),
                "label": str(obj.get("label", "object")),
                "point_count": int(points.shape[0]),
                "clean_point_count": int(obj.get("clean_point_count", points.shape[0])),
                "obb_method": "pending_vlm_semantic_front_refit",
                "obb_source": "3D_point_clean",
                "instances": obj.get("instances", []),
                "obb": {
                    "centroid": obb_to_float_list(centroid),
                    "axesLengths": obb_to_float_list(axes_lengths),
                    "normalizedAxes": obb_axes_matrix_to_list(rotation),
                },
            }
        )
        logger.info(
            "[OBB_CANDIDATE] objectId=%s label=%s points=%d method=vlm_semantic_front_then_plane_pca",
            obj.get("objectId"),
            obj.get("label"),
            points.shape[0],
        )
    return {
        "input_json": str(input_json),
        "source_stage": "8_clean_3d_points",
        "min_points_for_obb": min_points_for_obb,
        "initial_obb": "axis_aligned_fallback_until_vlm_front",
        "objects": obb_objects,
    }


def obb_split_data_by_object(data: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    objects = sorted(data.get("objects", []), key=lambda obj: str(obj.get("objectId", "")))
    for obj in objects:
        object_id = str(obj.get("objectId", "")).strip()
        if not object_id:
            logger.warning("Skipping per-object OBB JSON because objectId is empty.")
            continue
        object_data = {key: value for key, value in data.items() if key != "objects"}
        if "vlm_orientation_summary" in object_data:
            object_data["vlm_orientation_summary"] = [
                item for item in object_data["vlm_orientation_summary"] if str(item.get("objectId", "")) == object_id
            ]
        object_data["objectId"] = object_id
        object_data["objects"] = [obj]
        output_path = output_dir / f"{point3d_sanitize_name(object_id)}_3D_OBB.json"
        write_json(output_path, object_data)
        outputs[object_id] = output_path
    return outputs


def obb_clear_managed_json_outputs(output_dir: Path, aggregate_filename: str) -> int:
    removed = 0
    managed_paths = list(output_dir.glob("*_3D_OBB.json"))
    managed_paths.append(output_dir / aggregate_filename)
    for path in managed_paths:
        if path.exists() and path.is_file():
            path.unlink()
            removed += 1
    return removed


def obb_resolve_color_frame_path(scan_root: Path, frame_name: str) -> Path | None:
    color_dir = scan_root / "sequence" / "color"
    frame_id = obb_frame_base_from_name(frame_name)
    for name in (
        f"{frame_name}.png",
        f"{frame_name}.jpg",
        f"{frame_name}.jpeg",
        f"{frame_id}.color.jpg",
        f"{frame_id}.jpg",
        f"{frame_id}.jpeg",
        f"{frame_id}.png",
    ):
        candidate = color_dir / name
        if candidate.exists():
            return candidate
    return None


def obb_resolve_pose_frame_path(scan_root: Path, frame_name: str) -> Path | None:
    pose_dir = scan_root / "sequence" / "camera_poses"
    frame_id = obb_frame_base_from_name(frame_name)
    for name in (f"{frame_name}.pose.txt", f"{frame_name}.txt", f"{frame_id}.pose.txt", f"{frame_id}.txt"):
        candidate = pose_dir / name
        if candidate.exists():
            return candidate
    return None


def obb_resolve_info_frame_path(scan_root: Path, frame_name: str) -> Path | None:
    pose_dir = scan_root / "sequence" / "camera_poses"
    frame_id = obb_frame_base_from_name(frame_name)
    for name in (f"{frame_name}.info.txt", f"{frame_id}.info.txt", "_info.txt"):
        candidate = pose_dir / name
        if candidate.exists():
            return candidate
    return None


def obb_parse_info_intrinsic(scan_root: Path, frame_name: str | None = None) -> np.ndarray:
    info_path = obb_resolve_info_frame_path(scan_root, frame_name or "") if frame_name else scan_root / "sequence" / "camera_poses" / "_info.txt"
    if info_path is None or not info_path.exists():
        raise FileNotFoundError(f"Could not find camera info file for frame '{frame_name or ''}' in {scan_root}")
    info = point3d_parse_info_file(info_path)
    return point3d_parse_matrix_4x4(info["m_calibrationColorIntrinsic"])


def obb_project_world_point(point: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, float] | None:
    point_h = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float64)
    cam = world_to_camera @ point_h
    z = float(cam[2])
    if not np.isfinite(z) or abs(z) < 1e-8:
        return None
    fx, fy, cx, cy = point3d_intrinsic_params_from_matrix(intrinsic)
    pixel = np.array([fx * cam[0] / z + cx, fy * cam[1] / z + cy], dtype=np.float64)
    if not np.isfinite(pixel).all():
        return None
    return pixel, z


def obb_choose_front_axis_from_vlm(
    *,
    obb: dict[str, Any],
    orientation: dict[str, Any],
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    direction = str(orientation.get("front_direction_2d", "unknown"))
    if direction == "unknown":
        return None, {"reason": "unknown_vlm_direction"}
    center = np.asarray(obb.get("centroid", []), dtype=np.float64)
    axes_lengths = np.asarray(obb.get("axesLengths", []), dtype=np.float64)
    rotation = np.asarray(obb.get("normalizedAxes", []), dtype=np.float64)
    if center.shape != (3,) or axes_lengths.shape != (3,) or rotation.size != 9:
        return None, {"reason": "invalid_obb"}
    rotation = rotation.reshape(3, 3)
    world_to_camera = np.linalg.inv(camera_to_world)
    center_proj = obb_project_world_point(center, world_to_camera, intrinsic)
    if center_proj is None:
        return None, {"reason": "center_not_projectable"}
    center_pixel, center_depth = center_proj
    target_2d = {
        "left": np.array([-1.0, 0.0]),
        "right": np.array([1.0, 0.0]),
        "up": np.array([0.0, -1.0]),
        "down": np.array([0.0, 1.0]),
    }.get(direction)

    best_score = -1.0
    best_axis = None
    best_meta: dict[str, Any] = {}
    for axis_idx in range(3):
        axis = rotation[:, axis_idx]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            continue
        axis = axis / axis_norm
        step = max(float(axes_lengths[axis_idx]) * 0.5, 0.1)
        for sign in (1.0, -1.0):
            signed_axis = axis * sign
            projected = obb_project_world_point(center + signed_axis * step, world_to_camera, intrinsic)
            if projected is None:
                continue
            pixel, depth = projected
            if target_2d is not None:
                delta = pixel - center_pixel
                delta_norm = np.linalg.norm(delta)
                if delta_norm < 1e-6:
                    continue
                score = float(np.dot(delta / delta_norm, target_2d))
            elif direction == "toward_camera":
                score = float(center_depth - depth)
            elif direction == "away_camera":
                score = float(depth - center_depth)
            else:
                continue
            if score > best_score:
                best_score = score
                best_axis = signed_axis
                best_meta = {"axis_idx": axis_idx, "sign": sign, "score": score}
    if best_axis is None or best_score <= 0:
        return None, {"reason": "no_axis_matches_vlm_direction", "best_score": best_score}
    return best_axis / np.linalg.norm(best_axis), best_meta


def obb_reorient_from_front(
    front: np.ndarray,
    rotation_flat: list[float],
    axes_lengths: list[float],
    front_axis_idx: int,
) -> tuple[list[float], list[float]]:
    rotation = np.asarray(rotation_flat, dtype=np.float64).reshape(3, 3)
    lengths = np.asarray(axes_lengths, dtype=np.float64)
    if lengths.shape != (3,):
        raise ValueError("OBB axesLengths must contain exactly three values.")
    if front_axis_idx not in (0, 1, 2):
        raise ValueError(f"Invalid OBB front axis index: {front_axis_idx}")

    front = obb_normalize_vector(np.asarray(front, dtype=np.float64), rotation[:, front_axis_idx])
    remaining = [idx for idx in range(3) if idx != front_axis_idx]
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_axis_idx = max(remaining, key=lambda idx: abs(float(np.dot(rotation[:, idx], world_up))))
    right_axis_idx = next(idx for idx in remaining if idx != up_axis_idx)

    up = obb_normalize_vector(rotation[:, up_axis_idx], world_up)
    if float(np.dot(up, world_up)) < 0.0:
        up = -up
    right = obb_normalize_vector(np.cross(up, front), rotation[:, right_axis_idx])
    up = obb_normalize_vector(np.cross(front, right), up)

    corrected_rotation = np.column_stack([front, right, up])
    corrected_lengths = lengths[[front_axis_idx, right_axis_idx, up_axis_idx]]
    return obb_axes_matrix_to_list(corrected_rotation), obb_to_float_list(corrected_lengths)


def obb_choose_representative_instances(instances_2d_json: Path, point3d_clean_json: Path) -> dict[str, dict[str, Any]]:
    mapping = obb_object2d_to_clean_object_map(point3d_clean_json)
    best: dict[str, dict[str, Any]] = {}
    for inst in point3d_read_json(instances_2d_json).get("instances", []):
        object2d_id = str(inst.get("object2d_id", ""))
        object_id = mapping.get(object2d_id)
        if object_id is None:
            continue
        old = best.get(object_id)
        if old is None or int(inst.get("area", 0)) > int(old.get("area", 0)):
            best[object_id] = inst
    return best


def obb_crop_instance_image(image_path: Path, instance: dict[str, Any], margin_ratio: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    bbox = instance.get("bbox_xywh") or [0, 0, width, height]
    x, y, w, h = [float(v) for v in bbox[:4]]
    margin = max(w, h) * float(margin_ratio)
    x0 = max(0, int(round(x - margin)))
    y0 = max(0, int(round(y - margin)))
    x1 = min(width, int(round(x + w + margin)))
    y1 = min(height, int(round(y + h + margin)))
    crop = image.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (
            max(0, int(round(x - x0))),
            max(0, int(round(y - y0))),
            min(crop.width - 1, int(round(x + w - x0))),
            min(crop.height - 1, int(round(y + h - y0))),
        ),
        outline=(255, 0, 0),
        width=4,
    )
    return crop


def obb_resolve_instance_mask_path(instance: dict[str, Any], scan_root: Path) -> Path | None:
    mask_path = instance.get("mask_path")
    if mask_path:
        candidate = Path(str(mask_path))
        if candidate.exists():
            return candidate
    frame_name = str(instance.get("frame_name", "")).strip()
    object2d_id = str(instance.get("object2d_id", "")).strip()
    if frame_name and "__" in object2d_id:
        mask_name = object2d_id.split("__", 1)[1]
        output_root = scan_root
        candidate = output_root / DEFAULT_2D_SEGMENT_SUBDIR / "merged_masks" / frame_name / f"{mask_name}.png"
        if candidate.exists():
            return candidate
    return None


def obb_instance_score(instance: dict[str, Any]) -> float:
    for key in ("point_count", "selected_scene_point_count", "eroded_mask_pixel_count", "raw_mask_pixel_count", "area"):
        value = instance.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 0.0


def obb_choose_representative_instance_list(instances: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    valid = [inst for inst in instances if str(inst.get("frame_name", "")).strip()]
    valid.sort(key=lambda inst: (-obb_instance_score(inst), frame_sort_key(str(inst.get("frame_name", "")))))
    selected = valid[: max(1, int(max_frames))]
    return sorted(selected, key=lambda inst: frame_sort_key(str(inst.get("frame_name", ""))))


def obb_make_highlight_image(
    *,
    image_path: Path,
    mask_path: Path | None,
    instance: dict[str, Any],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    mask_bbox = None
    if mask_path is not None and mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize(image.size)
        mask_bbox = mask.getbbox()
        tint = Image.new("RGBA", image.size, (255, 40, 40, 90))
        transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay.alpha_composite(Image.composite(tint, transparent, mask))

    bbox = instance.get("bbox_xywh")
    if bbox and len(bbox) >= 4:
        x, y, w, h = [float(v) for v in bbox[:4]]
    elif mask_bbox is not None:
        x0, y0, x1, y1 = mask_bbox
        x, y, w, h = float(x0), float(y0), float(x1 - x0), float(y1 - y0)
    else:
        x, y, w, h = 0.0, 0.0, float(image.width), float(image.height)

    draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0, 255), width=5)
    draw.text((max(0, x), max(0, y - 20)), "TARGET", fill=(255, 0, 0, 255))

    return overlay.convert("RGB")


def obb_parse_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        confidence = float(value)
    else:
        text = str(value).strip().lower()
        named = {
            "very high": 0.95,
            "high": 0.85,
            "medium": 0.55,
            "moderate": 0.55,
            "low": 0.25,
            "very low": 0.10,
            "unknown": 0.0,
        }
        if text in named:
            return named[text]
        if text.endswith("%"):
            try:
                confidence = float(text[:-1].strip()) / 100.0
            except ValueError:
                return 0.0
        else:
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if not match:
                return 0.0
            confidence = float(match.group(0))
    if confidence > 1.0:
        confidence /= 100.0
    return float(max(0.0, min(1.0, confidence)))


def obb_build_orientation_prompt(label: str) -> str:
    label_text = str(label or "object").strip()
    hint = category_orientation_hint(label_text)
    seating_front_examples = "- sitting-facing/open side for chairs and sofas, opposite the backrest"
    if matched_orientation_category(label_text) == "sofa":
        seating_front_examples = (
            "- sitting-facing/open side for chairs, opposite the backrest\n"
            "- backrest-located side for sofas, from the seat/support region toward the backrest"
        )
    hint_priority = (
        "The target category matches the category-specific orientation hint below. "
        "Use that hint as the primary rule for deciding the semantic front. "
        "Use the general semantic-front definition only as supporting context, and do not override the category-specific hint unless the image lacks enough visual evidence."
        if has_category_orientation_hint(label_text)
        else
        "The target category does not match a category-specific hint. Use the general semantic-front definition and object-specific functional cues."
    )
    return f"""
You are determining the semantic front direction of one highlighted target object in a full video frame.
The target object is marked by a red transparent mask and a red bounding box labeled TARGET.

Priority rule:
{hint_priority}

Semantic front means the side a human would treat as the object's functional front:
- screen/viewing side for monitors and laptops
- door, drawer, handle, or access side for cabinets and printers
{seating_front_examples}
- typing/working side for desks only when visual evidence supports it

Category-specific orientation hint:
{hint}

Do not choose a direction merely because that side is visible, close to the camera, or centered in the image.
If the object has no stable semantic front, or the visual evidence is weak, answer has_semantic_front=false and front_direction_2d=unknown.

Choose front_direction_2d from exactly:
left, right, up, down, toward_camera, away_camera, unknown.

Return only JSON:
{{
  "has_semantic_front": true,
  "front_direction_2d": "left",
  "confidence": 0.0,
  "reason": "short evidence-based reason"
}}

Target category: {label_text}.
""".strip()


def obb_query_vlm_orientation(
    *,
    image: Image.Image,
    label: str,
    api_base_url: str,
    api_key: str,
    vlm_model: str,
    max_retries: int,
    request_timeout: float,
    retry_sleep: float,
) -> dict[str, Any]:
    from openai import APIConnectionError, APIError, APITimeoutError, OpenAI  # noqa: PLC0415

    prompt = obb_build_orientation_prompt(label)
    api_base_url = _require_local_vlm_endpoint(api_base_url)
    client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=max(1.0, float(request_timeout)))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}},
            ],
        }
    ]

    last_text = ""
    last_error = ""
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=vlm_model,
                messages=messages,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[VLM_ORIENT] %s request failed (%s); attempt %d/%d", label, last_error, attempt, attempts)
            if attempt < attempts and retry_sleep > 0:
                time.sleep(float(retry_sleep))
            continue
        last_text = response.choices[0].message.content or ""
        data = extract_json(last_text)
        if isinstance(data, dict):
            direction = str(data.get("front_direction_2d", "unknown")).strip().lower()
            has_front = bool(data.get("has_semantic_front", direction != "unknown"))
            if direction in {"left", "right", "up", "down", "toward_camera", "away_camera", "unknown"}:
                return {
                    "has_semantic_front": has_front,
                    "front_direction_2d": direction if has_front else "unknown",
                    "confidence": obb_parse_confidence(data.get("confidence")),
                    "reason": str(data.get("reason", "")),
                    "raw_response": last_text,
                }
    if last_error:
        raise VLMServiceUnavailableError(
            f"VLM service request failed after {attempts} attempt(s) for label '{label}': {last_error}"
        )
    return {
        "has_semantic_front": False,
        "front_direction_2d": "unknown",
        "confidence": 0.0,
        "reason": last_error or "VLM response could not be parsed.",
        "raw_response": last_text,
    }


def obb_direction_to_world_vector(direction: str, camera_to_world: np.ndarray) -> np.ndarray | None:
    camera_vectors = {
        "left": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "up": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "down": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "toward_camera": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "away_camera": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    vector = camera_vectors.get(str(direction))
    if vector is None:
        return None
    rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
    world = rotation @ vector
    norm = float(np.linalg.norm(world))
    if norm < 1e-8 or not np.isfinite(world).all():
        return None
    return world / norm


def obb_weighted_world_front_vote(votes: list[dict[str, Any]]) -> tuple[np.ndarray | None, dict[str, Any]]:
    vectors = []
    weights = []
    for vote in votes:
        vector = vote.get("front_world_vector")
        confidence = float(vote.get("confidence", 0.0))
        if vector is None or confidence <= 0.0:
            continue
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != 3 or not np.isfinite(vector).all():
            continue
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            continue
        vectors.append(vector / norm)
        weights.append(confidence)

    if not vectors:
        return None, {"reason": "no_valid_world_front_votes"}

    weighted = np.sum([vector * weight for vector, weight in zip(vectors, weights)], axis=0)
    norm = float(np.linalg.norm(weighted))
    if norm < 1e-8:
        return None, {"reason": "cancelled_world_front_votes", "valid_front_vote_count": len(vectors)}

    front = weighted / norm
    total_weight = float(sum(weights))
    agreement = float(
        sum(max(0.0, float(np.dot(front, vector))) * weight for vector, weight in zip(vectors, weights))
        / max(total_weight, 1e-8)
    )
    return front, {
        "vote_mode": "continuous_world_front_weighted_vote",
        "valid_front_vote_count": len(vectors),
        "weight_sum": total_weight,
        "front_agreement": agreement,
        "weighted_front_axis": [float(v) for v in front.tolist()],
    }


def obb_enforce_category_geometry_rule(
    *,
    label: str,
    points: Any,
    vlm_front: np.ndarray | None,
    context_objects: list[dict[str, Any]] | None = None,
    object_instances: list[dict[str, Any]] | None = None,
    observer_camera_centers: list[list[float]] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return a category-constrained front axis and auditable rule metadata.

    A rule is a hard axis constraint: the VLM front is not allowed to tilt the
    result away from the category's geometric candidate. Cabinet, curtain,
    picture, and picture-frame plane normals have a sign ambiguity and use an
    applicable semantic sign cue, while sofa geometry points from the
    seat/support region toward the backrest.
    """
    category = matched_orientation_category(label)
    rule = CATEGORY_GEOMETRY_RULES.get(category or "")
    base_meta: dict[str, Any] = {
        "matched_category": category or "",
        "rule_available": rule is not None,
        "enforced": False,
        "input_vlm_front_axis_world": (
            obb_to_float_list(np.asarray(vlm_front, dtype=np.float64).reshape(3))
            if vlm_front is not None
            else []
        ),
    }
    if rule is None:
        return vlm_front, {
            **base_meta,
            "reason": "no_category_geometry_rule" if vlm_front is not None else "no_valid_vlm_front_and_no_category_geometry_rule",
        }

    policy = str(rule.get("policy", ""))
    base_meta.update({"policy": policy, "description": str(rule.get("description", ""))})
    points_array = obb_ensure_points(points)
    if points_array.shape[0] < 3:
        return vlm_front, {**base_meta, "reason": "insufficient_points_for_category_geometry"}

    if policy == "tabletop_short_axis_horizontal_front":
        if points_array.shape[0] < 12:
            return vlm_front, {**base_meta, "reason": "insufficient_points_for_tabletop_geometry"}

        # A desk's functional front is defined by the chair/user side. Associate
        # chairs by shared observation frames first, then by 3D distance. This is
        # a category relationship, not a scene-specific object-id rule.
        desk_center_world = np.median(points_array, axis=0)
        desk_frame_names = {
            str(instance.get("frame_name", "")).strip()
            for instance in (object_instances or [])
            if str(instance.get("frame_name", "")).strip()
        }
        chair_candidates: list[dict[str, Any]] = []
        for context_obj in context_objects or []:
            if matched_orientation_category(str(context_obj.get("label", ""))) != "chair":
                continue
            center_world = np.asarray(context_obj.get("center_world", []), dtype=np.float64).reshape(-1)
            if center_world.size < 3 or not np.isfinite(center_world[:3]).all():
                continue
            chair_frame_names = {
                str(frame_name).strip()
                for frame_name in context_obj.get("frame_names", [])
                if str(frame_name).strip()
            }
            shared_frames = sorted(desk_frame_names & chair_frame_names, key=frame_sort_key)
            if not shared_frames:
                continue
            desk_to_chair_world = center_world[:3] - desk_center_world
            distance = float(np.linalg.norm(desk_to_chair_world))
            if distance <= 1e-8:
                continue
            chair_candidates.append(
                {
                    "objectId": str(context_obj.get("objectId", "")),
                    "shared_frames": shared_frames,
                    "shared_frame_count": len(shared_frames),
                    "distance_world": distance,
                    "desk_to_chair_vector_world": obb_to_float_list(desk_to_chair_world),
                }
            )

        z_values = points_array[:, 2]
        z_low, z_high = np.quantile(z_values, [0.02, 0.98])
        height = float(z_high - z_low)
        if not np.isfinite(height) or height <= 1e-8:
            return vlm_front, {**base_meta, "reason": "degenerate_object_height_for_tabletop_geometry"}

        normalized_height = (z_values - z_low) / height
        tabletop_height_threshold = 0.55
        tabletop_points = points_array[normalized_height >= tabletop_height_threshold]
        minimum_tabletop_points = max(6, int(np.ceil(points_array.shape[0] * 0.03)))
        if tabletop_points.shape[0] < minimum_tabletop_points:
            return vlm_front, {
                **base_meta,
                "reason": "insufficient_upper_points_for_tabletop_geometry",
                "tabletop_point_count": int(tabletop_points.shape[0]),
                "required_tabletop_point_count": minimum_tabletop_points,
            }

        # Fit the actual 3D tabletop plane from the higher surface band. Z=0 is
        # merely the global horizontal plane and is not assumed to equal the
        # observed tabletop plane. All desk front candidates are projected into
        # this fitted plane before their axis or sign is used.
        tabletop_surface_height_threshold = 0.75
        tabletop_surface_points = points_array[normalized_height >= tabletop_surface_height_threshold]
        minimum_surface_points = max(6, int(np.ceil(points_array.shape[0] * 0.02)))
        if tabletop_surface_points.shape[0] < minimum_surface_points:
            tabletop_surface_points = tabletop_points

        surface_center = np.mean(tabletop_surface_points, axis=0)
        centered_surface = tabletop_surface_points - surface_center[None, :]
        surface_covariance = centered_surface.T @ centered_surface / max(int(tabletop_surface_points.shape[0]), 1)
        if surface_covariance.shape != (3, 3) or not np.isfinite(surface_covariance).all():
            return vlm_front, {**base_meta, "reason": "invalid_tabletop_surface_covariance"}
        surface_eigenvalues, surface_eigenvectors = np.linalg.eigh(surface_covariance)
        surface_order = np.argsort(surface_eigenvalues)
        tabletop_normal = np.asarray(surface_eigenvectors[:, int(surface_order[0])], dtype=np.float64)
        tabletop_normal /= max(float(np.linalg.norm(tabletop_normal)), 1e-12)
        if tabletop_normal[2] < 0.0:
            tabletop_normal = -tabletop_normal

        tabletop_center = np.mean(tabletop_points, axis=0)
        centered_tabletop = tabletop_points - tabletop_center[None, :]
        centered_tabletop_in_plane = centered_tabletop - (
            (centered_tabletop @ tabletop_normal)[:, None] * tabletop_normal[None, :]
        )
        in_plane_covariance = (
            centered_tabletop_in_plane.T @ centered_tabletop_in_plane
            / max(int(centered_tabletop_in_plane.shape[0]), 1)
        )
        in_plane_eigenvalues, in_plane_eigenvectors = np.linalg.eigh(in_plane_covariance)
        in_plane_order = np.argsort(in_plane_eigenvalues)
        in_plane_eigenvalues = np.maximum(in_plane_eigenvalues[in_plane_order], 0.0)
        tabletop_short_axis = np.asarray(in_plane_eigenvectors[:, int(in_plane_order[1])], dtype=np.float64)
        tabletop_short_axis -= float(np.dot(tabletop_short_axis, tabletop_normal)) * tabletop_normal
        tabletop_short_axis /= max(float(np.linalg.norm(tabletop_short_axis)), 1e-12)
        tabletop_axis_separation_ratio = float(
            in_plane_eigenvalues[2] / max(in_plane_eigenvalues[1], 1e-12)
        )
        minimum_tabletop_axis_separation_ratio = 2.5

        projected_chair_candidates: list[dict[str, Any]] = []
        for candidate in chair_candidates:
            desk_to_chair_world = np.asarray(candidate["desk_to_chair_vector_world"], dtype=np.float64)
            desk_to_chair_in_plane = desk_to_chair_world - (
                float(np.dot(desk_to_chair_world, tabletop_normal)) * tabletop_normal
            )
            in_plane_distance = float(np.linalg.norm(desk_to_chair_in_plane))
            if in_plane_distance <= 1e-8:
                continue
            projected_chair_candidates.append(
                {
                    **candidate,
                    "distance_in_tabletop_plane": in_plane_distance,
                    "desk_to_chair_axis_world": obb_to_float_list(desk_to_chair_in_plane / in_plane_distance),
                }
            )
        chair_candidates = projected_chair_candidates

        if chair_candidates:
            if tabletop_axis_separation_ratio >= minimum_tabletop_axis_separation_ratio:
                # For a clearly rectangular desk, geometry fixes the front/back
                # axis to the tabletop short axis. Chairs are semantic sign cues
                # only; side chairs must not rotate the desk front toward a long
                # edge. Require a candidate to lie measurably on a short-edge side.
                short_coordinates = centered_tabletop_in_plane @ tabletop_short_axis
                short_low, short_high = np.quantile(short_coordinates, [0.05, 0.95])
                tabletop_robust_short_span = float(short_high - short_low)
                minimum_chair_short_axis_offset = max(0.05 * tabletop_robust_short_span, 1e-4)
                short_edge_chair_candidates: list[dict[str, Any]] = []
                for candidate in chair_candidates:
                    chair_axis_world = np.asarray(candidate["desk_to_chair_axis_world"], dtype=np.float64)
                    short_axis_alignment = float(np.dot(tabletop_short_axis, chair_axis_world))
                    chair_short_axis_offset = (
                        float(candidate["distance_in_tabletop_plane"]) * short_axis_alignment
                    )
                    if (
                        abs(short_axis_alignment) >= 0.35
                        and abs(chair_short_axis_offset) >= minimum_chair_short_axis_offset
                    ):
                        candidate = {
                            **candidate,
                            "tabletop_short_axis_alignment": short_axis_alignment,
                            "chair_short_axis_offset": chair_short_axis_offset,
                        }
                        short_edge_chair_candidates.append(candidate)

                if short_edge_chair_candidates:
                    associated_chair = min(
                        short_edge_chair_candidates,
                        key=lambda item: (
                            -int(item["shared_frame_count"]),
                            float(item["distance_in_tabletop_plane"]),
                        ),
                    )
                    associated_chair_axis_world = np.asarray(
                        associated_chair["desk_to_chair_axis_world"], dtype=np.float64
                    )
                    if float(np.dot(tabletop_short_axis, associated_chair_axis_world)) < 0.0:
                        tabletop_short_axis = -tabletop_short_axis
                    geometry_front = tabletop_short_axis.copy()
                    return geometry_front, {
                        **base_meta,
                        "enforced": True,
                        "reason": "desk_tabletop_short_axis_points_to_coview_chair_side",
                        "sign_source": "tabletop_short_axis_with_short_edge_chair_sign",
                        "world_up_axis": [0.0, 0.0, 1.0],
                        "output_front_axis_world": obb_to_float_list(geometry_front),
                    }
            else:
                # A near-square tabletop has no stable short axis. In that case
                # the co-visible interaction chair directly defines the in-plane
                # front direction, as in VLM+OBB_5 plus the desk-only context cue.
                associated_chair = min(
                    chair_candidates,
                    key=lambda item: (
                        -int(item["shared_frame_count"]),
                        float(item["distance_in_tabletop_plane"]),
                    ),
                )
                geometry_front = np.asarray(associated_chair["desk_to_chair_axis_world"], dtype=np.float64)
                return geometry_front, {
                    **base_meta,
                    "enforced": True,
                    "reason": "ambiguous_tabletop_axis_points_to_coview_associated_chair",
                    "sign_source": "shared_frames_then_nearest_chair_center",
                    "world_up_axis": [0.0, 0.0, 1.0],
                    "output_front_axis_world": obb_to_float_list(geometry_front),
                }

        if vlm_front is None:
            return None, {
                **base_meta,
                "reason": "no_valid_vlm_front_for_tabletop_front_sign",
                "tabletop_in_plane_variances_ascending": obb_to_float_list(in_plane_eigenvalues),
                "tabletop_axis_separation_ratio": tabletop_axis_separation_ratio,
            }
        normalized_vlm_front = np.asarray(vlm_front, dtype=np.float64).reshape(3)
        vlm_in_tabletop_plane = normalized_vlm_front - (
            float(np.dot(normalized_vlm_front, tabletop_normal)) * tabletop_normal
        )
        vlm_in_plane_norm = float(np.linalg.norm(vlm_in_tabletop_plane))
        if vlm_in_plane_norm <= 1e-8:
            return vlm_front, {
                **base_meta,
                "reason": "vlm_front_normal_to_tabletop_cannot_choose_in_plane_front",
                "tabletop_in_plane_variances_ascending": obb_to_float_list(in_plane_eigenvalues),
                "tabletop_axis_separation_ratio": tabletop_axis_separation_ratio,
            }
        vlm_in_tabletop_plane /= vlm_in_plane_norm

        if tabletop_axis_separation_ratio >= minimum_tabletop_axis_separation_ratio:
            if float(np.dot(tabletop_short_axis, vlm_in_tabletop_plane)) < 0.0:
                tabletop_short_axis = -tabletop_short_axis
            geometry_front = tabletop_short_axis.copy()
            axis_source = "tabletop_short_axis_with_projected_multi_frame_vlm_sign"
            reason = "tabletop_geometry_forced_in_plane_working_side_axis"
        else:
            geometry_front = vlm_in_tabletop_plane
            axis_source = "multi_frame_vlm_front_projected_into_tabletop_plane"
            reason = "ambiguous_tabletop_axis_used_in_plane_vlm_front"

        return geometry_front, {
            **base_meta,
            "enforced": True,
            "reason": reason,
            "sign_source": axis_source,
            "world_up_axis": [0.0, 0.0, 1.0],
            "height_quantiles_02_98": [float(z_low), float(z_high)],
            "tabletop_height_threshold_normalized": tabletop_height_threshold,
            "tabletop_surface_height_threshold_normalized": tabletop_surface_height_threshold,
            "tabletop_point_count": int(tabletop_points.shape[0]),
            "tabletop_surface_point_count": int(tabletop_surface_points.shape[0]),
            "tabletop_normal_world": obb_to_float_list(tabletop_normal),
            "tabletop_in_plane_variances_ascending": obb_to_float_list(in_plane_eigenvalues),
            "tabletop_axis_separation_ratio": tabletop_axis_separation_ratio,
            "minimum_tabletop_axis_separation_ratio": minimum_tabletop_axis_separation_ratio,
            "tabletop_short_axis_world_unsigned": obb_to_float_list(tabletop_short_axis),
            "output_front_axis_world": obb_to_float_list(geometry_front),
        }

    if policy == "backrest_away_horizontal_axis":
        # This pipeline uses Z+ as world up. The upper portion of a chair or
        # sofa is normally dominated by its backrest. In the XY plane, the
        # smallest-variance axis of those points is the backrest-plane normal.
        if points_array.shape[0] < 12:
            return vlm_front, {**base_meta, "reason": "insufficient_points_for_backrest_geometry"}

        z_values = points_array[:, 2]
        z_low, z_high = np.quantile(z_values, [0.02, 0.98])
        height = float(z_high - z_low)
        if not np.isfinite(height) or height <= 1e-8:
            return vlm_front, {**base_meta, "reason": "degenerate_object_height_for_backrest_geometry"}

        normalized_height = (z_values - z_low) / height
        upper_height_threshold = 0.75
        upper_mask = normalized_height >= upper_height_threshold
        # Ignore the extreme floor band while retaining the seat and supporting
        # structure. Their robust XY center is a proxy for the open/sitting side.
        seat_support_lower_threshold = 0.0
        seat_support_mask = (
            (normalized_height >= seat_support_lower_threshold)
            & (normalized_height <= 0.58)
        )
        upper_points = points_array[upper_mask]
        seat_support_points = points_array[seat_support_mask]
        minimum_region_points = max(6, int(np.ceil(points_array.shape[0] * 0.03)))
        if upper_points.shape[0] < minimum_region_points:
            return vlm_front, {
                **base_meta,
                "reason": "insufficient_upper_points_for_backrest_geometry",
                "upper_point_count": int(upper_points.shape[0]),
                "required_region_point_count": minimum_region_points,
            }
        if seat_support_points.shape[0] < minimum_region_points:
            return vlm_front, {
                **base_meta,
                "reason": "insufficient_seat_support_points_for_backrest_sign",
                "seat_support_point_count": int(seat_support_points.shape[0]),
                "required_region_point_count": minimum_region_points,
            }

        upper_xy = upper_points[:, :2]
        upper_xy_mean = np.mean(upper_xy, axis=0)
        centered_upper_xy = upper_xy - upper_xy_mean[None, :]
        covariance_xy = centered_upper_xy.T @ centered_upper_xy / max(int(upper_xy.shape[0]), 1)
        if covariance_xy.shape != (2, 2) or not np.isfinite(covariance_xy).all():
            return vlm_front, {**base_meta, "reason": "invalid_backrest_xy_covariance"}

        xy_eigenvalues, xy_eigenvectors = np.linalg.eigh(covariance_xy)
        xy_order = np.argsort(xy_eigenvalues)
        xy_eigenvalues = np.maximum(xy_eigenvalues[xy_order], 0.0)
        backrest_normal_xy = np.asarray(xy_eigenvectors[:, int(xy_order[0])], dtype=np.float64)
        backrest_normal_xy /= max(float(np.linalg.norm(backrest_normal_xy)), 1e-12)
        backrest_axis_separation_ratio = float(xy_eigenvalues[1] / max(xy_eigenvalues[0], 1e-12))
        if backrest_axis_separation_ratio < 1.25:
            if vlm_front is None:
                return None, {
                    **base_meta,
                    "reason": "backrest_horizontal_axis_not_distinctive_and_no_valid_vlm_front",
                    "backrest_xy_variances_ascending": obb_to_float_list(xy_eigenvalues),
                    "backrest_axis_separation_ratio": backrest_axis_separation_ratio,
                }
            vlm_horizontal = np.asarray(vlm_front, dtype=np.float64).reshape(3)[:2]
            vlm_horizontal_norm = float(np.linalg.norm(vlm_horizontal))
            if vlm_horizontal_norm <= 1e-8:
                return vlm_front, {
                    **base_meta,
                    "reason": "backrest_horizontal_axis_not_distinctive_and_vertical_vlm_front",
                    "backrest_xy_variances_ascending": obb_to_float_list(xy_eigenvalues),
                    "backrest_axis_separation_ratio": backrest_axis_separation_ratio,
                }
            vlm_horizontal /= vlm_horizontal_norm
            geometry_front = np.array([vlm_horizontal[0], vlm_horizontal[1], 0.0], dtype=np.float64)
            return geometry_front, {
                **base_meta,
                "enforced": True,
                "reason": "backrest_horizontal_axis_not_distinctive_used_horizontal_vlm_front",
                "sign_source": "multi_frame_vlm_front_horizontal_axis_and_sign",
                "world_up_axis": [0.0, 0.0, 1.0],
                "backrest_xy_variances_ascending": obb_to_float_list(xy_eigenvalues),
                "backrest_axis_separation_ratio": backrest_axis_separation_ratio,
                "output_front_axis_world": obb_to_float_list(geometry_front),
            }

        # The vector from the robust backrest center toward the lower seat and
        # support center chooses the sign that points away from the backrest.
        backrest_center_xy = np.median(upper_xy, axis=0)
        seat_support_center_xy = np.median(seat_support_points[:, :2], axis=0)
        backrest_to_seat_xy = seat_support_center_xy - backrest_center_xy
        all_normal_coordinates = points_array[:, :2] @ backrest_normal_xy
        normal_depth = float(np.quantile(all_normal_coordinates, 0.95) - np.quantile(all_normal_coordinates, 0.05))
        signed_backrest_to_seat = float(np.dot(backrest_normal_xy, backrest_to_seat_xy))
        minimum_away_offset = max(0.03 * normal_depth, 1e-4)

        sign_source = ""
        backrest_slab_meta: dict[str, Any] = {}
        if abs(signed_backrest_to_seat) >= minimum_away_offset:
            # Sofa semantic front points toward the side occupied by the backrest.
            # Therefore dot(front, backrest_to_seat) must be negative.
            if signed_backrest_to_seat > 0.0:
                backrest_normal_xy = -backrest_normal_xy
                signed_backrest_to_seat = -signed_backrest_to_seat
            sign_source = "seat_support_center_to_backrest_center"

        if not sign_source:
            # Partial point clouds can make the two robust centers coincide.
            # Keep the geometric backrest normal and let VLM choose only its
            # sign instead of trusting a weak/noisy geometric displacement.
            if vlm_front is None:
                return None, {
                    **base_meta,
                    "reason": "ambiguous_backrest_away_sign_and_no_valid_vlm_front",
                    "backrest_axis_separation_ratio": backrest_axis_separation_ratio,
                    "signed_backrest_to_seat_offset": signed_backrest_to_seat,
                    "minimum_away_offset": minimum_away_offset,
                }
            normalized_vlm_front = np.asarray(vlm_front, dtype=np.float64).reshape(3)
            vlm_horizontal = normalized_vlm_front[:2]
            vlm_horizontal_norm = float(np.linalg.norm(vlm_horizontal))
            if vlm_horizontal_norm <= 1e-8:
                return vlm_front, {
                    **base_meta,
                    "reason": "ambiguous_backrest_away_sign_and_vertical_vlm_front",
                    "backrest_axis_separation_ratio": backrest_axis_separation_ratio,
                    "signed_backrest_to_seat_offset": signed_backrest_to_seat,
                    "minimum_away_offset": minimum_away_offset,
                }
            vlm_horizontal /= vlm_horizontal_norm
            if float(np.dot(backrest_normal_xy, vlm_horizontal)) < 0.0:
                backrest_normal_xy = -backrest_normal_xy
            sign_source = "multi_frame_vlm_front_horizontal_fallback"

        geometry_front = np.array([backrest_normal_xy[0], backrest_normal_xy[1], 0.0], dtype=np.float64)
        absolute_vlm_alignment = None
        if vlm_front is not None:
            normalized_vlm_front = np.asarray(vlm_front, dtype=np.float64).reshape(3)
            normalized_vlm_front /= max(float(np.linalg.norm(normalized_vlm_front)), 1e-12)
            absolute_vlm_alignment = abs(float(np.dot(geometry_front, normalized_vlm_front)))
        return geometry_front, {
            **base_meta,
            "enforced": True,
            "reason": "backrest_geometry_forced_front_toward_backrest_side",
            "sign_source": sign_source,
            "world_up_axis": [0.0, 0.0, 1.0],
            "height_quantiles_02_98": [float(z_low), float(z_high)],
            "upper_height_threshold_normalized": upper_height_threshold,
            "upper_point_count": int(upper_points.shape[0]),
            "seat_support_point_count": int(seat_support_points.shape[0]),
            "backrest_xy_variances_ascending": obb_to_float_list(xy_eigenvalues),
            "backrest_axis_separation_ratio": backrest_axis_separation_ratio,
            "backrest_center_xy": obb_to_float_list(backrest_center_xy),
            "seat_support_center_xy": obb_to_float_list(seat_support_center_xy),
            "backrest_to_seat_support_xy": obb_to_float_list(backrest_to_seat_xy),
            "signed_backrest_to_seat_offset": signed_backrest_to_seat,
            "minimum_away_offset": minimum_away_offset,
            "normal_depth_05_95": normal_depth,
            "absolute_vlm_alignment": absolute_vlm_alignment,
            "output_front_axis_world": obb_to_float_list(geometry_front),
            **backrest_slab_meta,
        }

    if policy != "smallest_variance_principal_axis":
        return vlm_front, {**base_meta, "reason": "unsupported_category_geometry_policy"}

    curtain_observers_available = category == "curtain" and bool(observer_camera_centers)
    if vlm_front is None and not curtain_observers_available:
        return None, {**base_meta, "reason": "no_valid_vlm_front_for_geometry_sign"}

    centered = points_array - np.mean(points_array, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(int(centered.shape[0]), 1)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return vlm_front, {**base_meta, "reason": "invalid_category_geometry_covariance"}

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    geometry_front = np.asarray(eigenvectors[:, int(order[0])], dtype=np.float64)
    geometry_front /= max(float(np.linalg.norm(geometry_front)), 1e-12)

    sign_source = "multi_frame_vlm_front_dot_product"
    absolute_vlm_alignment = None
    median_observer_normal_offset = None
    observer_count = 0
    if curtain_observers_available:
        camera_centers = np.asarray(observer_camera_centers, dtype=np.float64).reshape(-1, 3)
        camera_centers = camera_centers[np.isfinite(camera_centers).all(axis=1)]
        observer_count = int(camera_centers.shape[0])
        if observer_count:
            object_center = np.mean(points_array, axis=0)
            observer_normal_offsets = (camera_centers - object_center[None, :]) @ geometry_front
            median_observer_offset = float(np.median(observer_normal_offsets))
            if median_observer_offset < 0.0:
                geometry_front = -geometry_front
                observer_normal_offsets = -observer_normal_offsets
                median_observer_offset = -median_observer_offset
            median_observer_normal_offset = median_observer_offset
            sign_source = "interior_observer_camera_side"

    if sign_source == "multi_frame_vlm_front_dot_product":
        if vlm_front is None:
            return None, {**base_meta, "reason": "no_valid_curtain_interior_sign_source"}
        normalized_vlm_front = np.asarray(vlm_front, dtype=np.float64).reshape(3)
        normalized_vlm_front /= max(float(np.linalg.norm(normalized_vlm_front)), 1e-12)
        signed_alignment = float(np.dot(geometry_front, normalized_vlm_front))
        if signed_alignment < 0.0:
            geometry_front = -geometry_front
            signed_alignment = -signed_alignment
        absolute_vlm_alignment = signed_alignment
    elif vlm_front is not None:
        normalized_vlm_front = np.asarray(vlm_front, dtype=np.float64).reshape(3)
        normalized_vlm_front /= max(float(np.linalg.norm(normalized_vlm_front)), 1e-12)
        absolute_vlm_alignment = abs(float(np.dot(geometry_front, normalized_vlm_front)))

    # A large ratio means the thinnest direction is geometrically distinctive;
    # the rule remains hard even for a weak ratio, but the metadata exposes that
    # uncertainty instead of silently pretending the point cloud was decisive.
    separation_ratio = float(eigenvalues[1] / max(eigenvalues[0], 1e-12))
    geometry_meta = {
        **base_meta,
        "enforced": True,
        "reason": "category_geometry_axis_replaced_vlm_axis",
        "sign_source": sign_source,
        "absolute_vlm_alignment": absolute_vlm_alignment,
        "principal_variances_ascending": obb_to_float_list(eigenvalues),
        "smallest_axis_separation_ratio": separation_ratio,
        "output_front_axis_world": obb_to_float_list(geometry_front),
    }
    if category == "curtain":
        geometry_meta.update(
            {
                "interior_observer_camera_count": observer_count,
                "median_observer_normal_offset": median_observer_normal_offset,
            }
        )
    return geometry_front, geometry_meta


def obb_semantic_obb_from_front(
    points: Any,
    front: np.ndarray,
    fallback_obb: dict[str, Any],
    world_up: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    points_array = obb_ensure_points(points)
    if points_array.shape[0] == 0:
        raise ValueError("Cannot fit semantic OBB without object points.")

    front = np.asarray(front, dtype=np.float64).reshape(3)
    front = front / max(float(np.linalg.norm(front)), 1e-12)
    world_up = obb_normalize_vector(
        np.asarray(
            [0.0, 0.0, 1.0] if world_up is None else world_up,
            dtype=np.float64,
        ).reshape(3),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    projected_world_up = world_up - front * float(np.dot(world_up, front))
    projected_world_up_norm = float(np.linalg.norm(projected_world_up))
    if projected_world_up_norm > 1e-8:
        projected_world_up /= projected_world_up_norm

    mean = np.mean(points_array, axis=0)
    centered = points_array - mean[None, :]
    plane_points = centered - np.outer(centered @ front, front)
    covariance = plane_points.T @ plane_points / max(int(plane_points.shape[0]), 1)

    plane_axes: list[np.ndarray] = []
    fit_policy = "front_fixed_plane_pca_world_up_disambiguated"
    if covariance.shape == (3, 3) and np.isfinite(covariance).all():
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        for idx in np.argsort(eigenvalues)[::-1]:
            candidate = eigenvectors[:, int(idx)]
            candidate = candidate - front * float(np.dot(candidate, front))
            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-8:
                continue
            candidate = candidate / norm
            if plane_axes and abs(float(np.dot(candidate, plane_axes[0]))) > 1.0 - 1e-6:
                continue
            plane_axes.append(candidate)
            if len(plane_axes) == 2:
                break

    if len(plane_axes) < 2:
        fit_policy = "front_fixed_plane_pca_degenerate_fallback_world_up_disambiguated"
        fallback_axes = np.asarray(fallback_obb.get("normalizedAxes", []), dtype=np.float64)
        if fallback_axes.size == 9:
            fallback_axes = fallback_axes.reshape(3, 3)
            for idx in range(3):
                candidate = fallback_axes[:, idx] - front * float(np.dot(fallback_axes[:, idx], front))
                norm = float(np.linalg.norm(candidate))
                if norm <= 1e-8:
                    continue
                candidate = candidate / norm
                if any(abs(float(np.dot(candidate, axis))) > 1.0 - 1e-6 for axis in plane_axes):
                    continue
                plane_axes.append(candidate)
                if len(plane_axes) == 2:
                    break

    if not plane_axes:
        if projected_world_up_norm > 1e-8:
            plane_axes.append(projected_world_up.copy())
        else:
            candidate = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), front)
            if float(np.linalg.norm(candidate)) < 1e-8:
                candidate = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), front)
            plane_axes.append(candidate / max(float(np.linalg.norm(candidate)), 1e-12))

    # PCA eigenvectors have arbitrary signs, and their order says nothing about
    # which of the two in-plane axes is semantically "up".  Select the PCA axis
    # closest to the projection of gravity/world-up onto the plane normal to the
    # fixed front axis, then orient its sign toward world-up.
    up = max(plane_axes, key=lambda axis: abs(float(np.dot(axis, world_up)))).copy()
    if float(np.dot(up, world_up)) < 0.0:
        up = -up
    right = obb_normalize_vector(np.cross(up, front), np.cross(world_up, front))
    up = obb_normalize_vector(np.cross(front, right), up)
    if float(np.dot(up, world_up)) < 0.0:
        right = -right
        up = -up
    rotation = np.column_stack([front, right, up])

    projected = points_array @ rotation
    mins = np.min(projected, axis=0)
    maxs = np.max(projected, axis=0)
    center_local = (mins + maxs) * 0.5
    centroid = rotation @ center_local
    lengths = np.maximum(maxs - mins, 0.0)
    return {
        "centroid": obb_to_float_list(centroid),
        "axesLengths": obb_to_float_list(lengths),
        "normalizedAxes": obb_axes_matrix_to_list(rotation),
    }, {
        "fit_mode": "refit_points_with_front_fixed_plane_pca",
        "point_count": int(points_array.shape[0]),
        "remaining_axes_policy": fit_policy,
        "world_up_axis_reconstruction": obb_to_float_list(world_up),
        "projected_world_up_norm": projected_world_up_norm,
        "up_axis_world_alignment_cosine": float(np.dot(up, world_up)),
        "front_up_orthogonality_error": abs(float(np.dot(front, up))),
    }


def obb_load_camera_to_world_pose(path: Path | None, pose_type: str) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    return point3d_load_pose(path, pose_type)


def obb_apply_vlm_object_orientations(
    *,
    obb_data: dict[str, Any],
    scan_root: Path,
    instances_2d_json: Path,
    point3d_clean_json: Path,
    pose_type: str,
    api_base_url: str,
    api_key: str,
    vlm_model: str,
    max_retries: int,
    request_timeout: float,
    retry_sleep: float,
    crop_margin_ratio: float,
    min_confidence: float,
    world_up: np.ndarray | None = None,
    stage_timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    semantic_obb_refit_seconds = 0.0
    objects_by_id = {str(obj.get("objectId")): obj for obj in obb_data.get("objects", [])}
    clean_objects_by_id = {
        str(obj.get("objectId")): obj
        for obj in point3d_read_json(point3d_clean_json).get("objects", [])
        if isinstance(obj, dict)
    }
    context_objects: list[dict[str, Any]] = []
    for context_object_id, context_clean_obj in clean_objects_by_id.items():
        context_points = obb_ensure_points(context_clean_obj.get("points", []))
        if context_points.shape[0] == 0:
            continue
        context_obb_obj = objects_by_id.get(context_object_id, {})
        context_instances = context_clean_obj.get("instances", context_obb_obj.get("instances", []))
        context_objects.append(
            {
                "objectId": context_object_id,
                "label": str(context_obb_obj.get("label", context_clean_obj.get("label", ""))),
                "center_world": obb_to_float_list(np.median(context_points, axis=0)),
                "frame_names": sorted(
                    {
                        str(instance.get("frame_name", "")).strip()
                        for instance in context_instances
                        if isinstance(instance, dict) and str(instance.get("frame_name", "")).strip()
                    },
                    key=frame_sort_key,
                ),
            }
        )
    summaries = []

    for object_id, obj in objects_by_id.items():
        clean_obj = clean_objects_by_id.get(object_id, {})
        instances = obb_choose_representative_instance_list(
            clean_obj.get("instances", obj.get("instances", [])),
            DEFAULT_OBB_VLM_MAX_REPRESENTATIVE_FRAMES,
        )
        if not instances:
            summaries.append({"objectId": object_id, "direction": "unknown", "applied": False, "reason": "no_representative_2d_instance"})
            continue

        frame_votes = []
        direction_scores: dict[str, float] = {}
        for inst in instances:
            frame_name = str(inst.get("frame_name", ""))
            image_path = None
            color_path = inst.get("color_path")
            if color_path and Path(str(color_path)).exists():
                image_path = Path(str(color_path))
            if image_path is None:
                image_path = obb_resolve_color_frame_path(scan_root, frame_name)
            if image_path is None:
                frame_votes.append({"frame_name": frame_name, "applied": False, "reason": "missing_color_frame"})
                continue

            mask_path = obb_resolve_instance_mask_path(inst, scan_root)
            orientation = obb_query_vlm_orientation(
                image=obb_make_highlight_image(
                    image_path=image_path,
                    mask_path=mask_path,
                    instance=inst,
                ),
                label=str(obj.get("label", inst.get("label", "object"))),
                api_base_url=api_base_url,
                api_key=api_key,
                vlm_model=vlm_model,
                max_retries=max_retries,
                request_timeout=request_timeout,
                retry_sleep=retry_sleep,
            )
            confidence = obb_parse_confidence(orientation.get("confidence"))
            direction = str(orientation.get("front_direction_2d", "unknown"))
            direction_scores[direction] = direction_scores.get(direction, 0.0) + confidence

            vote = {
                "frame_name": frame_name,
                "object2d_id": inst.get("object2d_id"),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "front_direction_2d": direction,
                "has_semantic_front": bool(orientation.get("has_semantic_front", direction != "unknown")),
                "confidence": confidence,
                "reason": orientation.get("reason", ""),
                "raw_response": orientation.get("raw_response", ""),
            }
            if direction != "unknown" and confidence >= min_confidence:
                camera_to_world = obb_load_camera_to_world_pose(obb_resolve_pose_frame_path(scan_root, frame_name), pose_type)
                world_vector = obb_direction_to_world_vector(direction, camera_to_world)
                if world_vector is not None:
                    vote["front_world_vector"] = [float(v) for v in world_vector.tolist()]
            frame_votes.append(vote)

        front_axis, vote_meta = obb_weighted_world_front_vote(frame_votes)
        observer_camera_centers: list[list[float]] = []
        if matched_orientation_category(str(obj.get("label", "object"))) == "curtain":
            for inst in instances:
                frame_name = str(inst.get("frame_name", ""))
                pose_path = obb_resolve_pose_frame_path(scan_root, frame_name)
                if pose_path is None:
                    continue
                camera_to_world = obb_load_camera_to_world_pose(pose_path, pose_type)
                camera_center = np.asarray(camera_to_world[:3, 3], dtype=np.float64)
                if np.isfinite(camera_center).all():
                    observer_camera_centers.append(obb_to_float_list(camera_center))
        front_axis, geometry_rule_meta = obb_enforce_category_geometry_rule(
            label=str(obj.get("label", "object")),
            points=clean_obj.get("points", []),
            vlm_front=front_axis,
            context_objects=context_objects,
            object_instances=clean_obj.get("instances", obj.get("instances", [])),
            observer_camera_centers=observer_camera_centers,
        )
        applied = False
        semantic_fit_meta = {}
        if front_axis is not None:
            semantic_obb_started = time.perf_counter()
            try:
                semantic_obb, semantic_fit_meta = obb_semantic_obb_from_front(
                    clean_obj.get("points", []),
                    front_axis,
                    obj.get("obb", {}),
                    world_up=world_up,
                )
            except ValueError as exc:
                vote_meta = {**vote_meta, "reason": str(exc)}
            else:
                obj["obb"] = semantic_obb
                obj["obb_method"] = (
                    "category_geometry_constrained_semantic_obb"
                    if geometry_rule_meta.get("enforced", False)
                    else "vlm_semantic_front_refit_obb"
                )
                vote_meta["front_axis_idx"] = 0
                vote_meta["semantic_front_axis_world"] = [float(v) for v in front_axis.tolist()]
                applied = True
            finally:
                semantic_obb_refit_seconds += time.perf_counter() - semantic_obb_started

        axis_source = (
            str(geometry_rule_meta.get("policy", "category_geometry_rule"))
            if geometry_rule_meta.get("enforced", False)
            else "continuous_world_front_vote"
        )
        axis_meta = {"axis_idx": 0, "sign": 1.0, "source": axis_source} if applied else {}
        orientation_method = (
            "category_geometry_constrained_semantic_front_refit_obb"
            if geometry_rule_meta.get("enforced", False)
            else "multi_frame_continuous_world_front_refit_obb"
        )
        geometry_rule_output_meta = geometry_rule_meta
        if matched_orientation_category(str(obj.get("label", "object"))) == "desk":
            # Keep the public JSON schema used by VLM+OBB_5. The richer desk
            # association scannet is only needed while choosing the front axis and
            # must not leak into category_geometry_rule in the output files.
            compatibility_keys = (
                "matched_category",
                "rule_available",
                "enforced",
                "input_vlm_front_axis_world",
                "reason",
            )
            geometry_rule_output_meta = {
                key: geometry_rule_meta[key]
                for key in compatibility_keys
                if key in geometry_rule_meta
            }
        obj["vlm_orientation"] = {
            "method": orientation_method,
            "applied_to_obb_axes": applied,
            "applied_semantic_front_axis": applied,
            "kept_pca_obb_geometry": False,
            "axis_selection": axis_meta,
            "semantic_front_axis_world": [float(v) for v in front_axis.tolist()] if front_axis is not None else [],
            "vote_meta": vote_meta,
            "category_geometry_rule": geometry_rule_output_meta,
            "semantic_obb_fit": semantic_fit_meta,
            "direction_weight_scores": direction_scores,
            "frame_votes": frame_votes,
        }
        summaries.append(
            {
                "objectId": object_id,
                "direction": max(direction_scores, key=direction_scores.get) if direction_scores else "unknown",
                "applied": applied,
                "selected_frame_count": len(instances),
                "category_geometry_rule_enforced": bool(geometry_rule_meta.get("enforced", False)),
                "category_geometry_policy": str(geometry_rule_meta.get("policy", "")),
                **vote_meta,
            }
        )
        logger.info("[VLM_ORIENT] objectId=%s applied=%s frames=%d", object_id, applied, len(instances))

    obb_data["vlm_orientation_summary"] = summaries
    if stage_timing is not None:
        stage_timing["semantic_obb_refit_seconds"] = semantic_obb_refit_seconds
    return obb_data


def run_cleaned_point3d_obb_generation(
    *,
    point3d_clean_json: Path,
    instances_2d_json: Path,
    scan_root: Path,
    output_dir: Path,
    pose_type: str,
    eigenvalue_eps: float,
    min_points_for_pca: int,
    api_base_url: str,
    api_key: str,
    vlm_model: str,
    vlm_max_retries: int,
    vlm_request_timeout: float,
    vlm_retry_sleep: float,
    vlm_crop_margin_ratio: float,
    vlm_orientation_min_confidence: float,
    world_up: np.ndarray | None = None,
    stage_timer: Any | None = None,
) -> Path:
    obb_generation_seconds = 0.0
    obb_prepare_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_objects = point3d_read_json(point3d_clean_json).get("objects", [])
    if not all_objects:
        raise ValueError(f"No cleaned objects found for OBB generation in {point3d_clean_json}")

    removed = obb_clear_managed_json_outputs(output_dir, "object_3d_obbs.json")
    logger.info("Cleared managed OBB JSON files before generation: output=%d", removed)

    obb_data = obb_prepare_clean_objects_for_semantic_obb(point3d_clean_json, all_objects, min_points_for_pca)
    obb_data["orientation_stage"] = "vlm_multi_frame_continuous_front_refit_obb"
    obb_data["semantic_obb_pipeline"] = {
        "vlm_front_first": True,
        "category_geometry_rules_are_hard_constraints": True,
        "category_geometry_rule_categories": sorted(CATEGORY_GEOMETRY_RULES),
        "category_geometry_can_replace_missing_vlm_front": ["sofa", "desk"],
        "category_geometry_rule_order": "vlm_vote_if_available_then_category_geometry_rule_then_obb_refit",
        "initial_pca_obb": False,
        "remaining_axes": "pca_in_plane_perpendicular_to_semantic_front",
        "remaining_axes_up_sign": "closest_plane_pca_axis_oriented_toward_gravity_aligned_world_up",
        "world_up_axis_reconstruction": obb_to_float_list(
            obb_normalize_vector(
                np.asarray(
                    [0.0, 0.0, 1.0] if world_up is None else world_up,
                    dtype=np.float64,
                ).reshape(3),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )
        ),
        "fallback_before_vlm": "axis_aligned_clean_point_bounds",
    }
    obb_generation_seconds += time.perf_counter() - obb_prepare_started
    orientation_started = time.perf_counter()
    stage_timing: dict[str, float] = {}
    try:
        obb_data = obb_apply_vlm_object_orientations(
            obb_data=obb_data,
            scan_root=scan_root,
            instances_2d_json=instances_2d_json,
            point3d_clean_json=point3d_clean_json,
            pose_type=pose_type,
            api_base_url=api_base_url,
            api_key=api_key,
            vlm_model=vlm_model,
            max_retries=vlm_max_retries,
            request_timeout=vlm_request_timeout,
            retry_sleep=vlm_retry_sleep,
            crop_margin_ratio=vlm_crop_margin_ratio,
            min_confidence=vlm_orientation_min_confidence,
            world_up=world_up,
            stage_timing=stage_timing,
        )
    except Exception as exc:
        if stage_timer is not None:
            stage_timer.add_record(
                "7_vlm_object_orientation",
                time.perf_counter() - orientation_started,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
            stage_timer.add_record(
                "8_obb_generation",
                obb_generation_seconds,
                "failed",
                "OBB generation did not complete because the orientation stage failed.",
            )
        raise
    orientation_total_seconds = time.perf_counter() - orientation_started
    semantic_obb_refit_seconds = float(stage_timing.get("semantic_obb_refit_seconds", 0.0))
    vlm_orientation_seconds = max(0.0, orientation_total_seconds - semantic_obb_refit_seconds)
    obb_generation_seconds += semantic_obb_refit_seconds
    obb_output_started = time.perf_counter()
    try:
        all_output_path = output_dir / "object_3d_obbs.json"
        write_json(all_output_path, obb_data)
        object_paths = obb_split_data_by_object(obb_data, output_dir)
    except Exception as exc:
        obb_generation_seconds += time.perf_counter() - obb_output_started
        if stage_timer is not None:
            stage_timer.add_record("7_vlm_object_orientation", vlm_orientation_seconds, "ok")
            stage_timer.add_record(
                "8_obb_generation",
                obb_generation_seconds,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        raise
    else:
        obb_generation_seconds += time.perf_counter() - obb_output_started
        if stage_timer is not None:
            stage_timer.add_record("7_vlm_object_orientation", vlm_orientation_seconds, "ok")
            stage_timer.add_record("8_obb_generation", obb_generation_seconds, "ok")
    logger.info("[DONE] OBB JSON files: output=%s objects=%d", all_output_path, len(object_paths))
    return all_output_path


class PipelineTimer:
    def __init__(
        self,
        output_dir: Path,
        total_start: float | None = None,
        started_at: str | None = None,
    ):
        self.output_dir = output_dir
        self.records: list[dict[str, Any]] = []
        self.total_start = SCRIPT_START_PERF if total_start is None else total_start
        self.started_at = SCRIPT_STARTED_AT if started_at is None else started_at
        self.finished_at = ""
        self.status = "running"

    def step(self, name: str):
        return _PipelineTimerStep(self, name)

    def add_record(self, name: str, elapsed_seconds: float, status: str, error: str = "") -> None:
        if status == "ok" and not error:
            for record in self.records:
                if record["name"] == name and record["status"] == "ok" and not record["error"]:
                    record["elapsed_seconds"] += elapsed_seconds
                    record["calls"] += 1
                    return
        self.records.append(
            {
                "name": name,
                "elapsed_seconds": elapsed_seconds,
                "status": status,
                "error": error,
                "calls": 1,
            }
        )

    @staticmethod
    def format_seconds(seconds: float) -> str:
        return f"{seconds:.3f}s ({seconds / 60.0:.2f} min)"

    @staticmethod
    def _natural_name_key(name: str) -> tuple:
        parts = re.split(r"(\d+)", name)
        key = []
        for part in parts:
            if part.isdigit():
                key.append((1, int(part)))
            else:
                key.append((0, part))
        return tuple(key)

    @classmethod
    def record_sort_key(cls, record: dict[str, Any]) -> tuple[int, tuple]:
        name = str(record["name"])
        fixed_order = {
            "0_prepare_pipeline": 0,
            "1_mast3r_slam_and_export": 1,
            "2_label_generation": 2,
            "3_vlm_sam2_segmentation": 3,
            "4_instance_association": 4,
            "5_3d_points_project": 5,
            "6_3d_points_clean": 6,
            "7_vlm_object_orientation": 7,
            "8_obb_generation": 8,
            "9_time_tracking": 9,
            "2_object_pipeline_skipped": 2,
            "5_object_parse_args": 7,
            "5_object_setup_paths": 8,
            "5_object_validate_inputs_collect_images": 9,
            "5_object_generate_label_json": 16,
            "5_object_load_label_list": 17,
            "5_object_init_vlm_sam2_pipeline": 18,
            "5_object_run_vlm_sam2_wall_time": 19,
            "5_object_vlm_sam2_2D_total": 20,
            "5_object_finalize_results": 21,
            "5_object_2d_overlay": 22,
            "6_instance_association": 23,
            "6_instance_association_overlay": 24,
            "7_mask_to_3d_points": 25,
            "8_refresh_overlays_after_cleaning": 26,
        }
        if name in fixed_order:
            return (fixed_order[name], ())
        match = re.match(r"^(\d+)_", name)
        if match:
            return (int(match.group(1)), cls._natural_name_key(name))
        return (99, cls._natural_name_key(name))

    def records_for_report(self) -> list[dict[str, Any]]:
        records = [
            dict(record)
            for record in self.records
            if record["name"] != "5_object_vlm_sam2_2D_total"
        ]
        per_image_step_names = {
            "5_object_load_image_encode",
            "5_object_vlm_identify_label_instances",
            "5_object_vlm_point_box",
            "5_object_sam2_segmentation",
            "5_object_sam2_select_best_mask",
            "5_object_assemble_segmentation_result",
        }
        aggregate_groups = [
            (
                "2_label_generation",
                {
                    "5_object_parse_args",
                    "5_object_setup_paths",
                    "5_object_validate_inputs_collect_images",
                    "5_object_generate_label_json",
                    "5_object_load_label_list",
                },
            ),
            (
                "3_vlm_sam2_segmentation",
                {
                    "5_object_init_vlm_sam2_pipeline",
                    "5_object_run_vlm_sam2_wall_time",
                    "5_object_finalize_results",
                    "5_object_2d_overlay",
                },
            ),
            (
                "4_instance_association",
                {
                    "6_instance_association",
                    "6_instance_association_overlay",
                },
            ),
            (
                "5_3d_points_project",
                {
                    "7_mask_to_3d_points",
                },
            ),
            (
                "6_3d_points_clean",
                {
                    "8_clean_3d_points",
                    "8_refresh_overlays_after_cleaning",
                },
            ),
        ]
        group_by_step = {
            step_name: group_name
            for group_name, step_names in aggregate_groups
            for step_name in step_names
        }
        grouped: dict[str, dict[str, Any]] = {}
        report_records: list[dict[str, Any]] = []
        hidden_step_names = per_image_step_names | {"5_object_vlm_sam2_2D_total"}
        for record in records:
            name = str(record["name"])
            if name in hidden_step_names:
                continue
            group_name = group_by_step.get(name)
            if group_name is None:
                report_records.append(record)
                continue
            aggregate = grouped.setdefault(
                group_name,
                {
                    "name": group_name,
                    "elapsed_seconds": 0.0,
                    "status": "ok",
                    "error": "",
                    "calls": 1,
                    "_errors": [],
                },
            )
            aggregate["elapsed_seconds"] += float(record["elapsed_seconds"])
            if record["status"] != "ok":
                aggregate["status"] = "failed"
            if record["error"]:
                aggregate["_errors"].append(f"{name}: {record['error']}")

        for group_name, _ in aggregate_groups:
            aggregate = grouped.get(group_name)
            if aggregate is None:
                continue
            aggregate["error"] = "; ".join(aggregate.pop("_errors"))
            report_records.append(aggregate)
        return report_records

    def write(self) -> Path:
        total_seconds = time.perf_counter() - self.total_start
        self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        timing_path = self.output_dir / "pipeline_timing.txt"
        report_records = self.records_for_report()
        accounted_seconds = sum(float(record["elapsed_seconds"]) for record in report_records)
        report_records.append(
            {
                "name": "10_time_tracking",
                "elapsed_seconds": max(0.0, total_seconds - accounted_seconds),
                "status": "ok",
                "error": "",
                "calls": 1,
            }
        )
        lines = [
            f"started_at: {self.started_at}",
            f"finished_at: {self.finished_at}",
            f"status: {self.status}",
            f"total_time: {self.format_seconds(total_seconds)}",
            "",
            "steps:",
        ]
        for record in sorted(report_records, key=self.record_sort_key):
            line = (
                f"{record['name']}: "
                f"{self.format_seconds(record['elapsed_seconds'])} "
                f"[{record['status']}]"
            )
            if record.get("calls", 1) > 1:
                line += f" calls={record['calls']}"
            if record["error"]:
                line += f" error={record['error']}"
            lines.append(line)

        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
        return timing_path


class _PipelineTimerStep:
    def __init__(self, timer: PipelineTimer, name: str):
        self.timer = timer
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        elapsed = time.perf_counter() - self.start
        if exc_type is None:
            self.timer.add_record(self.name, elapsed, "ok")
        else:
            self.timer.add_record(self.name, elapsed, "failed", f"{exc_type.__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the instance segmentation pipeline."""

    # --- VLM / API ---
    api_base_url: str | None = DEFAULT_API_BASE_URL

    api_key: str | None = DEFAULT_API_KEY
    vlm_model: str | None = DEFAULT_SERVICE_MODEL
    vlm_max_concurrency: int = VLM_CONCURRENCY
    max_retries: int = DEFAULT_VLM_MAX_RETRIES
    retry_delay: float = DEFAULT_VLM_RETRY_DELAY

    # --- SAM2 ---
    sam2_checkpoint: str = DEFAULT_SAM2_CHECKPOINT
    sam2_config: str = DEFAULT_SAM2_CONFIG
    sam2_mask_threshold: float = DEFAULT_SAM2_MASK_THRESHOLD


    # --- I/O ---
    image_extensions: tuple[str, ...] = DEFAULT_PIPELINE_IMAGE_EXTENSIONS
    output_format: str = DEFAULT_PIPELINE_OUTPUT_FORMAT
    timer: Any = None
    fixed_labels: tuple[str, ...] = ()
    fixed_label_map: dict[str, tuple[str, ...]] | None = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STAGE1_PROMPT = VLM_LABEL_STAGE1_PROMPT

STAGE2_PROMPT = VLM_LABEL_STAGE2_PROMPT


def format_label_list(labels: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def build_label_guided_stage1_prompt(labels: tuple[str, ...]) -> str:
    return (
        "Inspect the image and list only visible object categories that exactly match "
        "one of the allowed labels below. Do not mention objects outside this label list.\n\n"
        "Allowed labels:\n"
        f"{format_label_list(labels)}"
    )


def build_label_guided_stage2_prompt(labels: tuple[str, ...]) -> str:
    return (
        "Convert the visible allowed labels into a JSON array of strings. "
        "Use only exact strings from the allowed label list. Do not invent synonyms, "
        "singular/plural variants, or labels outside the list. If a label is not clearly "
        "visible, omit it. If no allowed labels are visible, return [].\n\n"
        "Allowed labels:\n"
        f"{format_label_list(labels)}"
    )


def _is_plural_name(name: str) -> bool:
    """Heuristic: VLM Stage 2 uses English plural form for multi-instance categories."""
    return name.endswith("s") and not name.endswith("ss")


def _plural_to_singular(name: str) -> str:
    """Convert a simple plural to singular by stripping a trailing 's'."""
    return name.rstrip("s") if _is_plural_name(name) else name


def build_point_prompt(instance_name: str) -> str:
    return (
        f'Generate 2D coordinate points for all visible instances of "{instance_name}" in the image. '
        "Return one point for every physically distinct instance, including an identifiable instance "
        "that is only partially visible at an image boundary. Do not merge touching or overlapping "
        "instances of the same category into one point. "
        "Each point must directly land on the visible object surface and avoid occluders. "
        "Return only a JSON array like: "
        f'[{{"point_2d": [x, y]}}, ...], where x and y are normalized integers in [0, {COORD_RANGE}].'
    )


def build_bbox_prompt(instance_name: str) -> str:
    return (
        f'Generate 2D bounding boxes for all visible instances of "{instance_name}" in the image. '
        "Return one box for every physically distinct instance, including an identifiable instance "
        "that is only partially visible at an image boundary. Do not merge touching or overlapping "
        "instances of the same category into one box. "
        "Each bounding box must tightly enclose the visible extent of one object instance. "
        "Return only a JSON array like: "
        f'[{{"bbox_2d": [x1, y1, x2, y2]}}, ...], where all values are normalized integers in [0, {COORD_RANGE}].'
    )


def build_bbox_repair_prompt(instance_name: str, point_coords: list[np.ndarray], bbox_raw: str) -> str:
    points_json = [
        {"point_2d": [int(round(float(point[0]))), int(round(float(point[1])))]}
        for point in point_coords
    ]
    return (
        f'The previous bounding-box answer for "{instance_name}" was invalid or missing.\n'
        f"Reference points for the visible instances are:\n"
        f"{json.dumps(points_json, ensure_ascii=False)}\n\n"
        "For each reference point, estimate one tight visible bounding box around the same object instance. "
        "Return the same number of boxes and keep the same order as the points. "
        "Return only a JSON array like "
        f'[{{"bbox_2d": [x1, y1, x2, y2]}}, ...], with all values normalized integers in [0, {COORD_RANGE}].\n\n'
        f"Invalid previous bbox answer:\n{bbox_raw[:1000]}"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class InstanceSegmentationPipeline:
    """End-to-end instance segmentation combining VLM + SAM2 with parallelism.

    Parameters
    ----------
    config : PipelineConfig
        All pipeline settings. Sensible defaults are provided.
    device : str or torch.device, optional
        Device for SAM2 inference. Auto-detected if not given.
    """

    def __init__(self, config: PipelineConfig | None = None, device=None):
        self.config = config or PipelineConfig()
        self.timer = self.config.timer
        self.fixed_labels = tuple(self.config.fixed_labels)
        self.fixed_label_map = dict(self.config.fixed_label_map or {})
        self.fixed_label_lookup = {
            normalize_label(label): label for label in self.fixed_labels
        }

        self._workdir = Path(__file__).resolve().parent

        # --- Device ---
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        logger.info("Using device: %s", self.device)

        # --- Async VLM client ---
        from openai import AsyncOpenAI
        self.async_client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=_require_local_vlm_endpoint(self.config.api_base_url),
        )

        # --- SAM2 ---
        logger.info("Loading SAM2 model...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        hydra_config = self.config.sam2_config
        ckpt = Path(self.config.sam2_checkpoint)
        if not ckpt.is_absolute():
            ckpt = self._workdir / ckpt

        sam2_model = build_sam2(hydra_config, str(ckpt), device=self.device)
        self.predictor = SAM2ImagePredictor(
            sam2_model,
            mask_threshold=self.config.sam2_mask_threshold,
        )
        logger.info("SAM2 model loaded.")

    def _timed_step(self, name: str):
        if self.timer is None:
            return nullcontext()
        return self.timer.step(name)

    # ------------------------------------------------------------------
    # Unified async VLM request
    # ------------------------------------------------------------------

    async def _request_vlm(
        self,
        messages: list[dict],
        temperature: float = DEFAULT_VLM_TEMPERATURE,
        max_completion_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> str:
        """Unified async VLM request. Supports single-turn and multi-turn.

        Args:
            messages: Full conversation history in OpenAI chat format.
            temperature: Sampling temperature.
            max_completion_tokens: Max tokens. None means model default.
            extra_body: Extra kwargs for the API. Defaults to
                        ``{"chat_template_kwargs": {"enable_thinking": False}}``.

        Returns:
            VLM response text.
        """
        if extra_body is None:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        kwargs = dict(
            model=self.config.vlm_model,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
        )
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        resp = await self.async_client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def _request_vlm_with_retry(
        self,
        messages: list[dict],
        parser: callable | None = None,
        max_retries: int | None = None,
        step_name: str = "vlm",
        temperature: float = DEFAULT_VLM_TEMPERATURE,
        max_completion_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> tuple[Any, str]:
        """Async VLM request with retry and optional JSON parsing.

        Args:
            messages: Full conversation history.
            parser: Optional callable that returns a parsed result or None.
                    If None, the raw text is returned as the parsed result.
            max_retries: Max attempts. Defaults to config.max_retries.
            step_name: Label for logging.
            temperature, max_completion_tokens, extra_body: forwarded.

        Returns:
            (parsed_result, raw_text) tuple.
        """
        if max_retries is None:
            max_retries = self.config.max_retries

        last_raw = ""
        for attempt in range(1, max_retries + 1):
            try:
                raw = await self._request_vlm(
                    messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    extra_body=extra_body,
                )
                last_raw = raw

                if parser is None:
                    return raw, raw

                parsed = parser(raw)
                if parsed is not None:
                    return parsed, raw

                logger.warning(
                    "[%s] Parse attempt %d/%d returned None. Raw: %s",
                    step_name, attempt, max_retries, raw[:300],
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Attempt %d/%d raised %s",
                    step_name, attempt, max_retries, exc,
                )

            if attempt < max_retries:
                await asyncio.sleep(self.config.retry_delay * attempt)

        raise RuntimeError(
            f"[{step_name}] Failed to get valid output after "
            f"{max_retries} attempts. Last raw: {last_raw[:500]}"
        )

    # ------------------------------------------------------------------
    # Helpers: build OpenAI-format messages
    # ------------------------------------------------------------------

    @staticmethod
    def _make_image_message(prompt: str, image_b64: str, mime: str = "jpeg") -> list[dict]:
        """Build a single-turn user message with image + text."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{mime};base64,{image_b64}"
                        },
                    },
                ],
            }
        ]

    # ------------------------------------------------------------------
    # Step 1 - Instance identification (two-stage, asynchronous)
    # ------------------------------------------------------------------

    def _labels_for_image(self, image_path: Path) -> tuple[str, ...]:
        for key in image_label_keys(image_path):
            if key in self.fixed_label_map:
                return tuple(self.fixed_label_map[key])
        return self.fixed_labels

    def _has_image_label_entry(self, image_path: Path) -> bool:
        return any(key in self.fixed_label_map for key in image_label_keys(image_path))

    @staticmethod
    def _label_lookup(labels: tuple[str, ...]) -> dict[str, str]:
        return {normalize_label(label): label for label in labels}

    def _filter_to_fixed_labels(self, names: list[str], labels: tuple[str, ...]) -> list[str]:
        fixed_label_lookup = self._label_lookup(labels)
        if not fixed_label_lookup:
            return names

        filtered: list[str] = []
        seen: set[str] = set()
        for name in names:
            label = fixed_label_lookup.get(normalize_label(name))
            if label is None:
                continue
            key = normalize_label(label)
            if key in seen:
                continue
            filtered.append(label)
            seen.add(key)
        return filtered

    async def identify_instances(self, image_b64: str, fixed_labels: tuple[str, ...]) -> tuple[list[str], dict]:
        """Two-stage VLM identification (async).

        Returns (instance_names, answers_dict).
        """
        if fixed_labels:
            stage1_prompt = build_label_guided_stage1_prompt(fixed_labels)
            stage2_prompt = build_label_guided_stage2_prompt(fixed_labels)
        else:
            stage1_prompt = STAGE1_PROMPT
            stage2_prompt = STAGE2_PROMPT

        # Stage 1: label-guided listing
        stage1_messages = self._make_image_message(stage1_prompt, image_b64, "jpeg")
        stage1_raw = await self._request_vlm(stage1_messages)

        # Stage 2: follow-up with retry
        stage2_messages = stage1_messages + [
            {"role": "assistant", "content": stage1_raw},
            {
                "role": "user",
                "content": [{"type": "text", "text": stage2_prompt}],
            },
        ]

        def _parse(raw: str):
            data = extract_json(raw)
            if not isinstance(data, list):
                return None
            if not all(isinstance(x, str) for x in data):
                return None
            if fixed_labels:
                return self._filter_to_fixed_labels(data, fixed_labels)
            if len(data) == 0:
                return None
            return data

        instance_names, stage2_raw = await self._request_vlm_with_retry(
            stage2_messages, parser=_parse, step_name="identify_stage2", max_completion_tokens=2048
        )

        logger.info(
            "Identified %d instances: %s", len(instance_names), instance_names
        )
        return instance_names, {
            "allowed_labels": list(fixed_labels),
            "Stage 1": stage1_raw,
            "Stage 2": stage2_raw,
        }

    # ------------------------------------------------------------------
    # Step 2 - Parallel grounding (point prediction per instance)
    # ------------------------------------------------------------------

    async def _ground_one_instance(
        self, instance_name: str, image_b64: str
    ) -> tuple[list[np.ndarray], list[np.ndarray], str, bool]:
        """Run separate VLM point and bbox calls, then align the outputs."""
        point_messages = self._make_image_message(build_point_prompt(instance_name), image_b64, "jpeg")
        bbox_messages = self._make_image_message(build_bbox_prompt(instance_name), image_b64, "jpeg")

        def _parse_point(raw: str):
            data = extract_json(raw)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list) or len(data) == 0:
                return None
            coords = []
            for item in data:
                if not isinstance(item, dict) or "point_2d" not in item:
                    return None
                pt = item["point_2d"]
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    return None
                x, y = float(pt[0]), float(pt[1])
                if not (0 <= x <= COORD_RANGE and 0 <= y <= COORD_RANGE):
                    logger.warning("Grounding point out of range: (%s, %s)", x, y)
                coords.append(
                    np.array(
                        [
                            float(np.clip(x, 0, COORD_RANGE)),
                            float(np.clip(y, 0, COORD_RANGE)),
                        ],
                        dtype=np.float32,
                    )
                )
            return coords if coords else None

        def _parse_bbox(raw: str):
            data = extract_json(raw)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list) or len(data) == 0:
                return None
            bboxes = []
            for item in data:
                if not isinstance(item, dict):
                    return None
                bbox_raw = None
                for key in ("bbox_2d", "bbox", "box_2d", "box"):
                    candidate = item.get(key)
                    if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
                        bbox_raw = candidate
                        break
                if bbox_raw is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in bbox_raw]
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                x1 = float(np.clip(x1, 0, COORD_RANGE))
                y1 = float(np.clip(y1, 0, COORD_RANGE))
                x2 = float(np.clip(x2, 0, COORD_RANGE))
                y2 = float(np.clip(y2, 0, COORD_RANGE))
                if x2 <= x1 or y2 <= y1:
                    continue
                bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            return bboxes if bboxes else None

        point_coro = self._request_vlm_with_retry(
            point_messages, parser=_parse_point, step_name=f"point:{instance_name}", max_completion_tokens=1024
        )
        bbox_coro = self._request_vlm_with_retry(
            bbox_messages, parser=_parse_bbox, step_name=f"bbox:{instance_name}", max_completion_tokens=1024
        )
        point_result, bbox_result = await asyncio.gather(point_coro, bbox_coro, return_exceptions=True)

        if isinstance(point_result, Exception):
            raise point_result
        point_coords, point_raw = point_result

        if isinstance(bbox_result, Exception):
            logger.warning("Bbox grounding failed for '%s': %s. Trying bbox repair.", instance_name, bbox_result)
            bbox_coords, bbox_raw = [], str(bbox_result)
        else:
            bbox_coords, bbox_raw = bbox_result

        if not bbox_coords and point_coords:
            repair_messages = self._make_image_message(
                build_bbox_repair_prompt(instance_name, point_coords, bbox_raw),
                image_b64,
                "jpeg",
            )
            try:
                bbox_coords, repair_raw = await self._request_vlm_with_retry(
                    repair_messages,
                    parser=_parse_bbox,
                    step_name=f"bbox_repair:{instance_name}",
                    max_retries=1,
                    max_completion_tokens=1024,
                )
                bbox_raw = json.dumps(
                    {"bbox_raw": bbox_raw, "bbox_repair_raw": repair_raw},
                    ensure_ascii=False,
                )
                logger.info("Bbox repair succeeded for '%s' with %d box(es).", instance_name, len(bbox_coords))
            except Exception as exc:
                logger.warning(
                    "Bbox repair failed for '%s': %s. Proceeding with point-only SAM2 prompt(s).",
                    instance_name,
                    exc,
                )
                bbox_coords = []
                bbox_raw = json.dumps(
                    {"bbox_raw": bbox_raw, "bbox_repair_error": str(exc)},
                    ensure_ascii=False,
                )

        is_ambiguous = len(bbox_coords) > len(point_coords)
        if is_ambiguous:
            logger.warning(
                "Ambiguous grounding for '%s': %d point(s) but %d box(es).",
                instance_name,
                len(point_coords),
                len(bbox_coords),
            )

        max_groundings = 5
        if len(point_coords) > max_groundings:
            logger.warning("Too many points for '%s' (%d > %d); keeping first %d.", instance_name, len(point_coords), max_groundings, max_groundings)
            point_coords = point_coords[:max_groundings]
        if len(bbox_coords) > max_groundings:
            logger.warning("Too many boxes for '%s' (%d > %d); keeping first %d.", instance_name, len(bbox_coords), max_groundings, max_groundings)
            bbox_coords = bbox_coords[:max_groundings]

        aligned_coords, aligned_bboxes = self._align_points_and_bboxes(point_coords, bbox_coords)
        combined_raw = json.dumps({"point_raw": point_raw, "bbox_raw": bbox_raw}, ensure_ascii=False)
        logger.info(
            "Grounded '%s' -> %d point(s) + %d box(es) -> %d aligned pair(s)%s",
            instance_name,
            len(point_coords),
            len(bbox_coords),
            len(aligned_coords),
            " [AMBIGUOUS]" if is_ambiguous else "",
        )
        return aligned_coords, aligned_bboxes, combined_raw, is_ambiguous

    def _align_points_and_bboxes(
        self,
        points: list[np.ndarray],
        bboxes: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        if not points:
            return [], []
        if not bboxes:
            logger.warning("No valid bboxes available; keeping %d point-only prompt(s).", len(points))
            return points, [None for _ in points]

        n_pts = len(points)
        n_box = len(bboxes)
        cost = np.full((n_pts, n_box), np.inf, dtype=np.float64)
        for i, point in enumerate(points):
            px, py = float(point[0]), float(point[1])
            for j, box in enumerate(bboxes):
                if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                    cost[i, j] = 0.0
                else:
                    cx = float((box[0] + box[2]) * 0.5)
                    cy = float((box[1] + box[3]) * 0.5)
                    cost[i, j] = float(np.hypot(px - cx, py - cy))

        n = max(n_pts, n_box)
        dummy_cost = float(COORD_RANGE * np.sqrt(2))
        padded = np.full((n, n), dummy_cost, dtype=np.float64)
        padded[:n_pts, :n_box] = cost
        try:
            from scipy.optimize import linear_sum_assignment

            row_indices, col_indices = linear_sum_assignment(padded)
        except ImportError:
            logger.warning("scipy is not installed; using greedy point-box alignment fallback.")
            pairs = sorted(
                (float(cost[row, col]), row, col)
                for row in range(n_pts)
                for col in range(n_box)
            )
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            greedy_rows = []
            greedy_cols = []
            for value, row, col in pairs:
                if value >= dummy_cost or row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                greedy_rows.append(row)
                greedy_cols.append(col)
            row_indices = np.asarray(greedy_rows, dtype=np.int64)
            col_indices = np.asarray(greedy_cols, dtype=np.int64)

        aligned_points: list[np.ndarray] = []
        aligned_bboxes: list[np.ndarray | None] = []
        for row, col in zip(row_indices, col_indices):
            if row < n_pts and col < n_box and padded[row, col] < dummy_cost:
                aligned_points.append(points[row])
                aligned_bboxes.append(bboxes[col])

        discarded_points = n_pts - len(aligned_points)
        discarded_bboxes = n_box - len(aligned_bboxes)
        if discarded_points or discarded_bboxes:
            logger.info(
                "Alignment kept %d pair(s), discarded %d point(s) and %d box(es).",
                len(aligned_points),
                discarded_points,
                discarded_bboxes,
            )
        return aligned_points, aligned_bboxes

    async def _ground_all_instances(
        self, instance_names: list[str], image_b64: str
    ) -> list[tuple[str, list[np.ndarray] | None, list[np.ndarray] | None, str, Exception | None, bool]]:
        """Ground all instance names in parallel."""
        sem = asyncio.Semaphore(self.config.vlm_max_concurrency)

        async def _ground_one(name: str):
            async with sem:
                try:
                    coords, bboxes, raw, is_ambiguous = await self._ground_one_instance(name, image_b64)
                    return (name, coords, bboxes, raw, None, is_ambiguous)
                except Exception as exc:
                    logger.error("Grounding failed for '%s': %s", name, exc)
                    return (name, None, None, str(exc), exc, False)

        tasks = [_ground_one(name) for name in instance_names]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Step 3 - Batched SAM 2 segmentation
    # ------------------------------------------------------------------

    def _segment_all_with_sam2(
        self, image_np: np.ndarray, prompts_list: list[dict[str, np.ndarray]]
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Segment all instances using SAM2 with point + box prompts.

        Each prompt contains normalized ``point`` and ``box`` values in
        [0, COORD_RANGE]. They are converted to pixel coordinates before SAM2
        prediction.
        """
        if not prompts_list:
            return [], []

        h, w = image_np.shape[:2]

        # Scale normalized [0, COORD_RANGE] prompts to pixel coordinates.
        pixel_points = []
        pixel_boxes = []
        for prompt in prompts_list:
            coord = prompt["point"]
            box = prompt.get("box")
            px = coord[0] * w / COORD_RANGE
            py = coord[1] * h / COORD_RANGE
            pixel_points.append([px, py])
            if box is None:
                pixel_boxes.append(None)
            else:
                x1 = float(np.clip(box[0] * w / COORD_RANGE, 0, max(w - 1, 0)))
                y1 = float(np.clip(box[1] * h / COORD_RANGE, 0, max(h - 1, 0)))
                x2 = float(np.clip(box[2] * w / COORD_RANGE, 0, max(w - 1, 0)))
                y2 = float(np.clip(box[3] * h / COORD_RANGE, 0, max(h - 1, 0)))
                if x2 <= x1 and w > 1:
                    x1 = max(0.0, min(x1, float(w - 2)))
                    x2 = min(float(w - 1), x1 + 1.0)
                if y2 <= y1 and h > 1:
                    y1 = max(0.0, min(y1, float(h - 2)))
                    y2 = min(float(h - 1), y1 + 1.0)
                pixel_boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

        all_points = np.array(pixel_points, dtype=np.float32)
        all_labels = np.ones(len(prompts_list), dtype=np.int32)

        # Set image once
        self.predictor.set_image(image_np)

        all_masks: list[list[np.ndarray]] = []
        all_scores: list[np.ndarray] = []

        for i in range(len(prompts_list)):
            box = pixel_boxes[i]
            masks, scores, _ = self.predictor.predict(
                point_coords=all_points[i:i+1],
                point_labels=all_labels[i:i+1],
                box=box,
                multimask_output=True,
            )

            order = np.argsort(scores)[::-1]
            instance_masks = [masks[idx].astype(np.uint8) for idx in order]
            instance_scores = scores[order]
            all_masks.append(instance_masks)
            all_scores.append(instance_scores)

            if (i + 1) % 10 == 0 or (i + 1) == len(prompts_list):
                logger.info(
                    "SAM2: processed %d/%d instances",
                    i + 1, len(prompts_list),
                )

        return all_masks, all_scores

    # ------------------------------------------------------------------
    # Step 4 - Mask deduplication (one instance per pixel)
    # ------------------------------------------------------------------

    def _deduplicate_masks(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        """Deduplicate masks so each pixel belongs to at most one instance.

        For every pair of overlapping masks, the overlapping region is assigned
        to the mask with the larger overlap-to-self-area ratio.  Pairs are
        processed from smallest to largest overlap area so that minor boundary
        disputes are resolved first.

        Args:
            masks: list of (H, W) uint8/bool masks (the *selected* mask per instance).

        Returns:
            list of (H, W) uint8 masks with no pixel belonging to more than one mask.
        """
        n = len(masks)
        if n <= 1:
            return [m.astype(np.uint8) for m in masks]

        areas = np.array([m.sum() for m in masks], dtype=np.float64)

        # 1. Find all overlapping pairs and compute original ratios
        pairs: list[tuple[int, int, int, float, float]] = []  # (overlap, i, j, ratio_i, ratio_j)
        for i in range(n):
            if areas[i] == 0:
                continue
            for j in range(i + 1, n):
                if areas[j] == 0:
                    continue
                overlap = np.logical_and(masks[i], masks[j]).sum()
                if overlap > 0:
                    ratio_i = overlap / areas[i]
                    ratio_j = overlap / areas[j]
                    pairs.append((int(overlap), i, j, ratio_i, ratio_j))

        if not pairs:
            return [m.astype(np.uint8) for m in masks]

        # 2. Sort by overlap area ascending
        pairs.sort(key=lambda x: x[0])

        # 3. Resolve conflicts pairwise
        result = [m.astype(np.uint8).copy() for m in masks]

        for _overlap, i, j, ratio_i, ratio_j in pairs:
            current_overlap = np.logical_and(result[i], result[j])
            if current_overlap.sum() == 0:
                continue  # already resolved by earlier pairs

            if ratio_i >= ratio_j:
                result[j] = np.logical_and(result[j], ~result[i].astype(bool)).astype(np.uint8)
            else:
                result[i] = np.logical_and(result[i], ~result[j].astype(bool)).astype(np.uint8)

        return result

    # ------------------------------------------------------------------
    # Per-image pipeline (async)
    # ------------------------------------------------------------------

    async def process_image(self, image_path: str | Path) -> dict:
        """Run the full pipeline on one image (async).

        Returns dict with keys:
          image_path, instance_identification_answer, segmentation_results
        """
        image_path = Path(image_path)
        logger.info("=" * 60)
        logger.info("Processing: %s", image_path)

        # Load image
        with self._timed_step("5_object_load_image_encode"):
            pil_img = Image.open(image_path).convert("RGB")
            np_img = np.array(pil_img)
            image_b64 = encode_image(pil_img)
            fixed_labels = self._labels_for_image(image_path)
            logger.info("Using %d label(s) for %s: %s", len(fixed_labels), image_path.name, list(fixed_labels))

        if self._has_image_label_entry(image_path) and not fixed_labels:
            logger.warning("No generated labels for %s; skipping segmentation.", image_path)
            return {
                "image_path": str(image_path),
                "instance_identification_answer": json.dumps(
                    {"allowed_labels": [], "Stage 1": "", "Stage 2": "[]"},
                    ensure_ascii=False,
                ),
                "segmentation_results": json.dumps([], ensure_ascii=False),
            }

        # Step 1: Identify instances (two-stage VLM)
        with self._timed_step("5_object_vlm_identify_label_instances"):
            instance_names, id_answer = await self.identify_instances(image_b64, fixed_labels)

        if not instance_names:
            logger.warning("No instances identified in %s", image_path)
            return {
                "image_path": str(image_path),
                "instance_identification_answer": json.dumps(
                    id_answer, ensure_ascii=False
                ),
                "segmentation_results": json.dumps([], ensure_ascii=False),
            }

        # Step 2: Ground all instances in parallel
        with self._timed_step("5_object_vlm_point_box"):
            grounding_results = await self._ground_all_instances(
                instance_names, image_b64
            )

        # Expand plural names to per-prompt instances.
        # Each entry: (inst_name, {"point": point_norm, "box": box_norm}, source_category, is_ambiguous)
        per_instance: list[tuple[str, dict[str, Any], str, bool]] = []
        grounding_answers: dict[str, str] = {}
        failed_instances: list[dict] = []

        for name, coords, bboxes, raw, exc, is_ambiguous in grounding_results:
            if exc is not None or coords is None or bboxes is None or len(coords) == 0:
                failed_instances.append({
                    "instance": name,
                    "coord": None,
                    "box_2d": None,
                    "mask_base64": None,
                    "mask_scores": None,
                    "selected_mask_idx": None,
                    "mask_validation_answer": str(exc) if exc else raw,
                    "grounding_answer": raw,
                })
                continue

            if _is_plural_name(name) or len(coords) > 1:
                base = name if fixed_labels else _plural_to_singular(name)
                for i, (coord, bbox) in enumerate(zip(coords, bboxes), start=1):
                    inst_name = f"{base}_{i}"
                    per_instance.append((inst_name, {"point": coord, "box": bbox}, name, is_ambiguous or bbox is None))
                    grounding_answers[inst_name] = raw
            else:
                per_instance.append((name, {"point": coords[0], "box": bboxes[0]}, name, is_ambiguous or bboxes[0] is None))
                grounding_answers[name] = raw

        if not per_instance:
            return {
                "image_path": str(image_path),
                "instance_identification_answer": json.dumps(
                    id_answer, ensure_ascii=False
                ),
                "segmentation_results": json.dumps(
                    failed_instances, ensure_ascii=False
                ),
            }

        filtered_per_instance: list[tuple[str, dict[str, Any], str, bool]] = []
        category_removed: dict[str, bool] = {}
        for inst_name, prompt, source_category, is_ambiguous in per_instance:
            coord = prompt["point"]
            bbox = prompt["box"]
            if bbox is None:
                logger.warning("No bbox for '%s'; keeping point-only SAM2 prompt.", inst_name)
                filtered_per_instance.append((inst_name, prompt, source_category, True))
                continue
            if coord[0] < bbox[0] or coord[0] > bbox[2] or coord[1] < bbox[1] or coord[1] > bbox[3]:
                logger.warning(
                    "Point outside bbox for '%s': pt=[%.0f, %.0f] bbox=[%.0f, %.0f, %.0f, %.0f]; discarding.",
                    inst_name,
                    coord[0],
                    coord[1],
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                )
                category_removed[source_category] = True
                continue
            filtered_per_instance.append((inst_name, prompt, source_category, is_ambiguous))
        per_instance = [
            (inst_name, prompt, source_category, True if category_removed.get(source_category) else is_ambiguous)
            for inst_name, prompt, source_category, is_ambiguous in filtered_per_instance
        ]

        if not per_instance:
            return {
                "image_path": str(image_path),
                "instance_identification_answer": json.dumps(
                    id_answer, ensure_ascii=False
                ),
                "segmentation_results": json.dumps(
                    failed_instances, ensure_ascii=False
                ),
            }

        # Step 3: Batch SAM2 segmentation
        with self._timed_step("5_object_sam2_segmentation"):
            all_prompts = [prompt for _, prompt, _, _ in per_instance]
            all_masks, all_scores = self._segment_all_with_sam2(np_img, all_prompts)

        # Step 3b: Select the highest-score SAM2 mask directly.
        # _segment_all_with_sam2 sorts each instance's masks by score descending.
        with self._timed_step("5_object_sam2_select_best_mask"):
            validation_results = [(0, "SAM2_HIGHEST_SCORE") for _ in per_instance]

            # Step 4: Deduplicate selected masks (one instance per pixel)
            selected_masks = [all_masks[i][0] for i in range(len(per_instance))]
            deduped_masks = self._deduplicate_masks(selected_masks)

        # Assemble results
        with self._timed_step("5_object_assemble_segmentation_result"):
            seg_results: list[dict] = []
            for idx, (inst_name, prompt, _source_category, is_ambiguous) in enumerate(per_instance):
                coord = prompt["point"]
                box = prompt.get("box")
                masks = all_masks[idx]
                scores = all_scores[idx]
                sel_idx, val_answer = validation_results[idx]

                # Replace the selected mask with its deduplicated version
                final_masks = list(masks)
                final_masks[sel_idx] = deduped_masks[idx]

                seg_results.append({
                    "instance": inst_name,
                    "coord": coord.tolist(),
                    "box_2d": box.tolist() if box is not None else None,
                    "mask_base64": [mask_to_base64(m) for m in final_masks],
                    "mask_scores": [float(s) for s in scores],
                    "selected_mask_idx": sel_idx,
                    "mask_validation_answer": val_answer,
                    "grounding_answer": grounding_answers[inst_name],
                    "ambiguous": bool(is_ambiguous),
                })

            seg_results.extend(failed_instances)

        logger.info(
            "Finished %s: %d/%d instances segmented.",
            image_path,
            sum(1 for r in seg_results if r["mask_base64"] is not None),
            len(seg_results),
        )

        return {
            "image_path": str(image_path),
            "instance_identification_answer": json.dumps(id_answer, ensure_ascii=False),
            "segmentation_results": json.dumps(seg_results, ensure_ascii=False),
        }

    # ------------------------------------------------------------------
    # Single-frame processing
    # ------------------------------------------------------------------

    async def _run_async(self, image_paths: str | Path | list[str | Path]) -> list[dict]:
        """Async implementation for one or more input images."""
        if isinstance(image_paths, (str, Path)):
            paths = [image_paths]
        else:
            paths = image_paths
        rows = []
        for image_path in paths:
            rows.append(await self.process_image(image_path))
        return rows

    def run(self, image_paths: str | Path | list[str | Path]) -> list[dict]:
        """Process one or more images and return segmentation rows.

        Parameters
        ----------
        image_paths : str, Path, or list
            Path(s) to input image(s).

        Returns
        -------
        list[dict]
            Segmentation rows, one per image.
        """
        return asyncio.run(self._run_async(image_paths))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def discover_images(
    sources: list[str | Path],
    extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
) -> list[Path]:
    """Given file/directory paths, return sorted list of all matching image files."""
    result: list[Path] = []
    for src in sources:
        src = Path(src)
        if src.is_file():
            if src.suffix.lower() in extensions:
                result.append(src)
        elif src.is_dir():
            for ext in extensions:
                result.extend(src.rglob(f"*{ext}"))
                result.extend(src.rglob(f"*{ext.upper()}"))
    return sorted(set(result), key=lambda p: str(p))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def object_pipeline_parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Video-frame VLM label generation + SAM2 instance segmentation + cross-frame association "
            "3D point cleaning, VLM semantic-front orientation, and semantic OBB pipeline."
        )
    )
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_IMAGE_DIR), help="RGB image file or directory to process in batch.")
    parser.add_argument("--scan_root", type=str, default=str(DEFAULT_SCAN_ROOT), help="Scene root containing sequence/color.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output root.")
    parser.add_argument("--label_file", type=str, default=str(DEFAULT_LABEL_FILE), help="Generated per-image label.json output path. This file is always regenerated.")
    parser.set_defaults(
        api_base_url=DEFAULT_API_BASE_URL,
        api_key=DEFAULT_API_KEY,
        vlm_model=DEFAULT_SERVICE_MODEL,
    )
    parser.add_argument("--vlm_concurrency", "--vlm-concurrency", dest="vlm_concurrency", type=int, default=VLM_CONCURRENCY, help="Max concurrent VLM requests.")
    parser.add_argument("--sam2_checkpoint", "--sam2-checkpoint", dest="sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT, help="SAM2 checkpoint path.")
    parser.add_argument("--sam2_config", "--sam2-config", dest="sam2_config", default=DEFAULT_SAM2_CONFIG, help="SAM2 config YAML.")
    parser.add_argument("--max_retries", "--max-retries", dest="max_retries", type=int, default=DEFAULT_VLM_MAX_RETRIES, help="Max VLM retries.")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, help="Limit the number of RGB frames processed.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_OVERLAY_ALPHA, help="2D RGB overlay mask opacity.")
    parser.add_argument("--draw_outline", action="store_true", default=DEFAULT_OVERLAY_DRAW_OUTLINE, help="Draw bounding boxes on 2D RGB overlays.")
    parser.add_argument("--min_area", type=int, default=DEFAULT_OVERLAY_MIN_AREA, help="Minimum mask area to render in 2D RGB overlays.")
    parser.set_defaults(association_same_label=ASSOCIATION_SAME_LABEL)
    parser.add_argument("--association_same_label", dest="association_same_label", action="store_true", help="Only match propagated instances to target instances with the same label.")
    parser.add_argument("--no-association_same_label", "--no_association_same_label", dest="association_same_label", action="store_false", help="Allow cross-frame association across different labels.")
    parser.set_defaults(association_same_frame_exclusivity=ASSOCIATION_SAME_FRAME_EXCLUSIVITY)
    parser.add_argument("--association_same_frame_exclusivity", dest="association_same_frame_exclusivity", action="store_true", help="Enforce the invariant that one object cannot contain multiple instances from the same frame.")
    parser.add_argument("--association_iou_threshold", type=float, default=ASSOCIATION_IOU, help="IoU threshold for matching propagated masks to target-frame 2D masks.")
    parser.add_argument("--association_window_frames", type=int, default=ASSOCIATION_WINDOW_FRAMES, help="Number of previous and next frames to match for each frame in linear cross-frame association (without first/last-frame wraparound).")
    parser.add_argument("--association_output", type=str, default=DEFAULT_ASSOCIATION_OUTPUT_FILENAME, help="Cross-frame association JSON path, relative to instance_association unless absolute.")
    parser.add_argument("--association_overlay_dir", type=str, default=DEFAULT_ASSOCIATION_OVERLAY_SUBDIR, help="Directory for cross-frame association visualizations, relative to instance_association unless absolute.")
    parser.add_argument("--camera_info_file", type=str, default=str(DEFAULT_CAMERA_INFO_FILE), help="Global camera _info.txt path.")
    parser.add_argument("--camera_pose_dir", type=str, default=str(DEFAULT_CAMERA_POSE_DIR), help="Camera pose directory.")
    parser.add_argument("--depth_dir", type=str, default=str(DEFAULT_DEPTH_DIR), help="Depth frame directory.")
    parser.add_argument("--point3d_output", type=str, default=DEFAULT_POINT3D_OUTPUT_FILENAME, help="Raw 3D point JSON path, relative to 3D_point unless absolute.")
    parser.add_argument("--point3d_output_dir", type=str, default=DEFAULT_POINT3D_OUTPUT_SUBDIR, help="Raw 3D point cloud directory, relative to 3D_point unless absolute.")
    parser.add_argument("--point3d_clean_output", type=str, default=DEFAULT_POINT3D_CLEAN_OUTPUT_FILENAME, help="Cleaned 3D point JSON path, relative to 3D_point_clean unless absolute.")
    parser.add_argument("--point3d_clean_output_dir", type=str, default=DEFAULT_POINT3D_CLEAN_OUTPUT_SUBDIR, help="Cleaned 3D point cloud directory, relative to 3D_point_clean unless absolute.")
    parser.add_argument("--pose_type", choices=("world_to_camera", "camera_to_world"), default=POSE_TYPE, help="Convention used by pose files. world_to_camera is inverted before projection.")
    parser.set_defaults(enable_mask_erosion=DEFAULT_ENABLE_MASK_EROSION)
    parser.add_argument("--enable_mask_erosion", action="store_true", help="Enable mask erosion before selecting scene PLY points.")
    parser.add_argument("--no-enable_mask_erosion", "--no_enable_mask_erosion", dest="enable_mask_erosion", action="store_false", help="Disable mask erosion before selecting scene PLY points.")
    parser.add_argument("--mask_erode_pixels", type=int, default=DEFAULT_MASK_ERODE_PIXELS, help="Erode each 2D mask by this many pixels before selecting scene PLY points. 0 disables erosion.")
    parser.add_argument(
        "--point3d_seed_expansion_radius_m",
        type=float,
        default=POINT3D_SEED_EXPANSION_RADIUS_M,
        help="Maximum world-space distance from a provenance seed for Z-buffer-visible projected supplement points.",
    )
    parser.add_argument("--min_depth_m", type=float, default=DEFAULT_MIN_DEPTH_M, help="Minimum camera-space depth in meters for projected scene PLY points.")
    parser.add_argument("--max_depth_m", type=float, default=DEFAULT_MAX_DEPTH_M, help="Maximum camera-space depth in meters for projected scene PLY points.")
    parser.add_argument("--sample_stride", type=int, default=DEFAULT_SAMPLE_STRIDE, help="Keep every Nth selected scene PLY point per 2D instance.")
    parser.add_argument("--max_points_per_object", type=int, default=DEFAULT_MAX_POINTS_PER_OBJECT, help="0 keeps all projected 3D points.")
    parser.add_argument("--depth_shift", type=float, default=DEFAULT_DEPTH_SHIFT, help="Depth scale divisor. 0 uses m_depthShift from _info.txt.")
    parser.add_argument("--area_ratio_threshold", type=float, default=DEFAULT_AREA_RATIO_THRESHOLD)
    parser.add_argument("--centroid_distance_m", type=float, default=DEFAULT_CENTROID_DISTANCE_M)
    parser.add_argument("--centroid_mad_multiplier", type=float, default=DEFAULT_CENTROID_MAD_MULTIPLIER)
    parser.add_argument("--extent_ratio_threshold", type=float, default=DEFAULT_EXTENT_RATIO_THRESHOLD)
    parser.add_argument("--min_instance_points", type=int, default=DEFAULT_MIN_INSTANCE_POINTS)
    parser.add_argument("--min_keep_instances", type=int, default=DEFAULT_MIN_KEEP_INSTANCES)
    parser.add_argument("--clean_mode", choices=("preserve", "strict"), default=DEFAULT_CLEAN_MODE)
    parser.set_defaults(spatial_cluster_instances=DEFAULT_SPATIAL_CLUSTER_INSTANCES)
    parser.add_argument("--spatial_cluster_instances", dest="spatial_cluster_instances", action="store_true")
    parser.add_argument("--no_spatial_cluster_instances", "--no-spatial_cluster_instances", dest="spatial_cluster_instances", action="store_false")
    parser.add_argument("--cluster_distance_m", type=float, default=DEFAULT_CLUSTER_DISTANCE_M)
    parser.add_argument("--cluster_min_instances", type=int, default=DEFAULT_CLUSTER_MIN_INSTANCES)
    parser.set_defaults(split_spatial_clusters=DEFAULT_SPLIT_SPATIAL_CLUSTERS)
    parser.add_argument("--split_spatial_clusters", dest="split_spatial_clusters", action="store_true")
    parser.add_argument("--no_split_spatial_clusters", "--no-split_spatial_clusters", dest="split_spatial_clusters", action="store_false")
    parser.add_argument("--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE)
    parser.add_argument("--nb_neighbors", type=int, default=DEFAULT_NB_NEIGHBORS)
    parser.add_argument("--std_ratio", type=float, default=DEFAULT_STD_RATIO)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--min_points_in_radius", type=int, default=DEFAULT_MIN_POINTS_IN_RADIUS)
    parser.add_argument("--min_points_after_filter", type=int, default=DEFAULT_MIN_POINTS_AFTER_FILTER)
    parser.add_argument("--max_points_for_knn", type=int, default=DEFAULT_MAX_POINTS_FOR_KNN)
    parser.add_argument("--obb_eigenvalue_eps", type=float, default=DEFAULT_OBB_EIGENVALUE_EPS)
    parser.add_argument("--obb_min_points_for_pca", type=int, default=DEFAULT_OBB_MIN_POINTS_FOR_PCA)
    parser.add_argument("--obb_vlm_max_retries", type=int, default=DEFAULT_OBB_VLM_MAX_RETRIES)
    parser.add_argument("--obb_vlm_request_timeout", type=float, default=DEFAULT_OBB_VLM_REQUEST_TIMEOUT)
    parser.add_argument("--obb_vlm_retry_sleep", type=float, default=DEFAULT_OBB_VLM_RETRY_SLEEP)
    parser.add_argument("--obb_vlm_crop_margin_ratio", type=float, default=DEFAULT_OBB_VLM_CROP_MARGIN_RATIO)
    parser.add_argument("--obb_vlm_orientation_min_confidence", type=float, default=DEFAULT_OBB_VLM_ORIENTATION_MIN_CONFIDENCE)
    parser.add_argument("--device", default=DEVICE, help="Torch device for SAM2.")
    parser.add_argument("--log_level", default=DEFAULT_LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if not all((args.api_base_url, args.api_key, args.vlm_model)):
        parser.error(
            "Required model-service settings are missing from the private runtime environment."
        )
    return args


def collect_images(image_source: Path, max_frames: int | None = None) -> list[Path]:
    image_paths = sorted(discover_images([image_source]), key=lambda p: frame_sort_key(p.name))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise FileNotFoundError(f"No image found in {image_source}")
    return image_paths


def run_embedded_object_pipeline(timer: PipelineTimer | None = None):
    object_pipeline_start = time.perf_counter()
    parse_start = time.perf_counter()
    args = object_pipeline_parse_args()
    parse_elapsed = time.perf_counter() - parse_start

    setup_start = time.perf_counter()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    output_dir = resolve_path(args.output_dir)
    image_dir = resolve_path(args.image_dir)
    label_file = resolve_path(args.label_file)
    export_dir = output_dir / DEFAULT_2D_SEGMENT_SUBDIR
    association_dir = output_dir / DEFAULT_FRAME_ASSOCIATION_SUBDIR
    point3d_dir = output_dir / DEFAULT_3D_POINT_SUBDIR
    point3d_clean_dir = output_dir / DEFAULT_3D_POINT_CLEAN_SUBDIR
    obb_dir = output_dir / DEFAULT_3D_OBB_SUBDIR
    instances_2d_path = export_dir / "instances_2d.json"
    masks_root = export_dir / "merged_masks"
    overlay_dir = export_dir / "overlay"
    association_output_path = resolve_path(args.association_output) if Path(args.association_output).is_absolute() else association_dir / args.association_output
    association_overlay_dir = resolve_path(args.association_overlay_dir) if Path(args.association_overlay_dir).is_absolute() else association_dir / args.association_overlay_dir
    association_instances_path = association_dir / "instances_2d_with_association.json"
    point3d_output_path = resolve_path(args.point3d_output) if Path(args.point3d_output).is_absolute() else point3d_dir / args.point3d_output
    point3d_output_dir = resolve_path(args.point3d_output_dir) if Path(args.point3d_output_dir).is_absolute() else point3d_dir / args.point3d_output_dir
    point3d_clean_output_path = resolve_path(args.point3d_clean_output) if Path(args.point3d_clean_output).is_absolute() else point3d_clean_dir / args.point3d_clean_output
    point3d_clean_output_dir = resolve_path(args.point3d_clean_output_dir) if Path(args.point3d_clean_output_dir).is_absolute() else point3d_clean_dir / args.point3d_clean_output_dir
    setup_elapsed = time.perf_counter() - setup_start

    owns_timer = timer is None
    if timer is None:
        timer = PipelineTimer(
            output_dir,
            total_start=object_pipeline_start,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    timer.add_record("5_object_parse_args", parse_elapsed, "ok")
    timer.add_record("5_object_setup_paths", setup_elapsed, "ok")
    try:
        with timer.step("5_object_validate_inputs_collect_images"):
            scan_root = resolve_path(args.scan_root)
            checkpoint = resolve_path(args.sam2_checkpoint)
            camera_info_file = resolve_path(args.camera_info_file)
            camera_pose_dir = resolve_path(args.camera_pose_dir)
            depth_dir = resolve_path(args.depth_dir)
            if not scan_root.exists():
                raise FileNotFoundError(f"scan_root not found: {scan_root}")
            if not image_dir.exists():
                raise FileNotFoundError(f"image_dir not found: {image_dir}")
            if label_file.suffix.lower() != ".json":
                raise ValueError(f"label_file must be a generated per-image .json file: {label_file}")
            if not checkpoint.exists():
                raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
            if not camera_info_file.exists():
                raise FileNotFoundError(f"camera_info_file not found: {camera_info_file}")

            image_paths = collect_images(image_dir, args.max_frames)
            logger.info("Found %d image(s) to process under %s", len(image_paths), image_dir)

        with timer.step("5_object_generate_label_json"):
            label_results = generate_label_json(
                image_paths=image_paths,
                output_file=label_file,
                api_base_url=args.api_base_url,
                api_key=args.api_key,
                model=args.vlm_model,
                max_retries=args.max_retries,
            )
            if not any(label_results.values()):
                raise RuntimeError(
                    "VLM label generation returned no labels for every input image. "
                    "Check the preceding 'failed to label' messages, the VLM endpoint, "
                    "and whether the configured model supports image input."
                )

        with timer.step("5_object_load_label_list"):
            fixed_label_map, fixed_labels = load_label_mapping(label_file, image_paths)
            logger.info(
                "Loaded labels from %s: %d image mapping key(s), %d unique fallback label(s)",
                label_file,
                len(fixed_label_map),
                len(fixed_labels),
            )

        with timer.step("5_object_init_vlm_sam2_pipeline"):
            cfg = PipelineConfig(
                api_base_url=args.api_base_url,
                api_key=args.api_key,
                vlm_model=args.vlm_model,
                vlm_max_concurrency=args.vlm_concurrency,
                max_retries=args.max_retries,
                sam2_checkpoint=str(resolve_path(args.sam2_checkpoint)),
                sam2_config=sam2_config_for_image_predictor(args.sam2_config),
                timer=timer,
                fixed_labels=fixed_labels,
                fixed_label_map=fixed_label_map,
            )
            pipeline = InstanceSegmentationPipeline(cfg, device=args.device)
        with timer.step("5_object_run_vlm_sam2_wall_time"):
            rows = pipeline.run(image_paths)

        with timer.step("5_object_finalize_results"):
            export_instance_artifacts(rows, export_dir)
            if not read_instances(instances_2d_path):
                raise RuntimeError(
                    "2D segmentation produced no valid instances. Check the VLM "
                    "identification/grounding warnings and SAM2 results above."
                )
            report_instance_coverage(instances_2d_path, image_paths)

        with timer.step("5_object_2d_overlay"):
            render_2d_overlays(
                instances_2d_path,
                masks_root,
                image_dir,
                overlay_dir,
                args.alpha,
                args.draw_outline,
                args.min_area,
            )

        with timer.step("6_instance_association"):
            association_json_path, association_objects, _ = run_instance_association(
                instances_2d_path=instances_2d_path,
                merged_masks_dir=masks_root,
                image_paths=image_paths,
                output_path=association_output_path,
                sam2_config=args.sam2_config,
                sam2_checkpoint=resolve_path(args.sam2_checkpoint),
                device_arg=args.device,
                match_iou_threshold=args.association_iou_threshold,
                association_window_frames=args.association_window_frames,
                association_same_label=args.association_same_label,
                association_same_frame_exclusivity=args.association_same_frame_exclusivity,
            )
            association_objects = merge_association_objects_by_scene_color(
                association_json=association_json_path,
                association_objects=association_objects,
                instances_2d_json=instances_2d_path,
                scan_root=scan_root,
                camera_info_file=camera_info_file,
                camera_pose_dir=camera_pose_dir,
                pose_type=args.pose_type,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
            )
            write_association_instances(
                instances_2d_path,
                association_objects,
                association_instances_path,
            )

        with timer.step("6_instance_association_overlay"):
            render_association_overlays(
                instances_2d_path,
                association_json_path,
                masks_root,
                image_dir,
                association_overlay_dir,
                args.alpha,
                args.draw_outline,
                args.min_area,
            )

        with timer.step("7_mask_to_3d_points"):
            point3d_json_path = run_point3d_projection(
                scan_root=scan_root,
                instances_2d_json=instances_2d_path,
                segment_3d_json=association_json_path,
                camera_info_file=camera_info_file,
                camera_pose_dir=camera_pose_dir,
                depth_dir=depth_dir,
                output_json=point3d_output_path,
                output_dir=point3d_output_dir,
                pose_type=args.pose_type,
                enable_mask_erosion=args.enable_mask_erosion,
                mask_erode_pixels=args.mask_erode_pixels,
                seed_expansion_radius_m=args.point3d_seed_expansion_radius_m,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                sample_stride=args.sample_stride,
                max_points_per_object=args.max_points_per_object,
                depth_shift_arg=args.depth_shift,
            )

        with timer.step("8_clean_3d_points"):
            point3d_clean_json_path = run_point3d_cleaning(
                input_json=point3d_json_path,
                output_json=point3d_clean_output_path,
                output_dir=point3d_clean_output_dir,
                area_ratio_threshold=args.area_ratio_threshold,
                centroid_distance_m=args.centroid_distance_m,
                centroid_mad_multiplier=args.centroid_mad_multiplier,
                extent_ratio_threshold=args.extent_ratio_threshold,
                min_instance_points=args.min_instance_points,
                min_keep_instances=args.min_keep_instances,
                clean_mode=args.clean_mode,
                spatial_cluster_instances=args.spatial_cluster_instances,
                cluster_distance_m=args.cluster_distance_m,
                cluster_min_instances=args.cluster_min_instances,
                split_spatial_clusters=args.split_spatial_clusters,
                voxel_size=args.voxel_size,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio,
                radius=args.radius,
                min_points_in_radius=args.min_points_in_radius,
                min_points_after_filter=args.min_points_after_filter,
                max_points_for_knn=args.max_points_for_knn,
            )

        with timer.step("8_refresh_overlays_after_cleaning"):
            report_instance_coverage(instances_2d_path, image_paths)
            write_association_instances(
                instances_2d_path,
                association_objects,
                association_instances_path,
            )
            render_2d_overlays(
                instances_2d_path,
                masks_root,
                image_dir,
                overlay_dir,
                args.alpha,
                args.draw_outline,
                args.min_area,
            )
            render_association_overlays(
                instances_2d_path,
                association_json_path,
                masks_root,
                image_dir,
                association_overlay_dir,
                args.alpha,
                args.draw_outline,
                args.min_area,
            )

        with timer.step("8_estimate_world_alignment_for_obb"):
            scene_ply_path = scan_root / "sequence" / "scene.ply"
            raw_scene_points = load_scene_points(scene_ply_path)
            first_camera_pose = discover_first_camera_pose(scene_ply_path)
            world_alignment = estimate_world_alignment(raw_scene_points, first_camera_pose)

        obb_json_path = run_cleaned_point3d_obb_generation(
            point3d_clean_json=point3d_clean_json_path,
            instances_2d_json=instances_2d_path,
            scan_root=scan_root,
            output_dir=obb_dir,
            pose_type=args.pose_type,
            eigenvalue_eps=args.obb_eigenvalue_eps,
            min_points_for_pca=args.obb_min_points_for_pca,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            vlm_model=args.vlm_model,
            vlm_max_retries=args.obb_vlm_max_retries,
            vlm_request_timeout=args.obb_vlm_request_timeout,
            vlm_retry_sleep=args.obb_vlm_retry_sleep,
            vlm_crop_margin_ratio=args.obb_vlm_crop_margin_ratio,
            vlm_orientation_min_confidence=args.obb_vlm_orientation_min_confidence,
            world_up=world_alignment.up_raw,
            stage_timer=timer,
        )

        timer.status = "completed"
        print(f"[DONE] processed images: {len(image_paths)}")
        print(f"[DONE] labels: {label_file}")
        print(f"[DONE] instances_2d: {instances_2d_path}")
        print(f"[DONE] merged masks: {masks_root}")
        print(f"[DONE] overlays: {overlay_dir}")
        print(f"[DONE] cross-frame association: {association_output_path}")
        print(f"[DONE] association instances: {association_instances_path}")
        print(f"[DONE] association overlays: {association_overlay_dir}")
        print(f"[DONE] raw 3D points: {point3d_json_path}")
        print(f"[DONE] raw 3D point clouds: {point3d_output_dir}")
        print(f"[DONE] cleaned 3D points: {point3d_clean_json_path}")
        print(f"[DONE] cleaned 3D point clouds: {point3d_clean_output_dir}")
        print(f"[DONE] VLM semantic OBBs: {obb_json_path}")
        print(f"[DONE] Per-object semantic OBBs: {obb_dir}")
    except Exception as exc:
        timer.status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        if owns_timer:
            timing_path = timer.write()
            print(f"[DONE] Timing log: {timing_path}")




# ---------------------------------------------------------------------------
# World-alignment helpers required by semantic OBB generation
# ---------------------------------------------------------------------------

EPSILON = 1e-9


@dataclass(frozen=True)
class GroundPlaneFit:
    """Ground plane represented as ``normal_raw dot point = offset_raw``."""

    normal_raw: np.ndarray
    offset_raw: float
    inlier_count: int
    sample_count: int
    distance_threshold: float
    camera_up_alignment: float


@dataclass(frozen=True)
class WorldAlignment:
    """Rigid transform from reconstruction coordinates to aligned world."""

    raw_to_aligned: np.ndarray
    aligned_origin_raw: np.ndarray
    front_raw: np.ndarray
    left_raw: np.ndarray
    up_raw: np.ndarray
    camera_origin_raw: np.ndarray
    camera_optical_axis_raw: np.ndarray
    first_camera_pose_path: Path
    ground_plane: GroundPlaneFit


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        if fallback is None:
            raise ValueError("Cannot normalize a zero-length vector")
        return fallback.copy()
    return vector / norm


def discover_first_camera_pose(scene_ply: Path) -> Path:
    """Locate the first SLAM keyframe pose next to ``scene.ply``."""

    sequence_dir = scene_ply.parent
    pose_dir = sequence_dir / "camera_poses"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing camera pose directory: {pose_dir}")

    keyframes_path = sequence_dir / "keyframes.txt"
    if keyframes_path.is_file():
        with keyframes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                frame_stem = line.strip()
                if not frame_stem:
                    continue
                candidate = pose_dir / f"{frame_stem}.txt"
                if candidate.is_file():
                    return candidate

    pose_paths = [
        path
        for path in pose_dir.glob("*.txt")
        if path.name != "_info.txt"
    ]
    if not pose_paths:
        raise FileNotFoundError(f"No camera pose files found under: {pose_dir}")

    def pose_sort_key(path: Path) -> tuple[int, int | str]:
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(pose_paths, key=pose_sort_key)[0]


def load_camera_to_world(path: Path) -> np.ndarray:
    matrix = np.asarray(np.loadtxt(str(path)), dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"Camera pose must contain 16 values: {path}")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Camera pose contains non-finite values: {path}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError(f"Camera pose has an invalid homogeneous row: {path}")

    rotation = matrix[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-3 or determinant <= 0.0:
        raise ValueError(
            f"Camera pose rotation is invalid: {path}; "
            f"orthogonality_error={orthogonality_error:.6g}, "
            f"determinant={determinant:.6g}"
        )
    return matrix


def load_scene_points(scene_ply: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError:
        o3d = None

    if o3d is not None:
        point_cloud = o3d.io.read_point_cloud(str(scene_ply))
        points = np.asarray(point_cloud.points, dtype=np.float64)
    else:
        # Keep the pipeline runnable in lightweight environments where Open3D is
        # unavailable. The fallback handles the vertex-first ASCII/binary PLY
        # layouts emitted by this repository's reconstruction exporter.
        ply_types = {
            "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
            "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
        }
        with scene_ply.open("rb") as handle:
            if handle.readline().strip() != b"ply":
                raise ValueError(f"Not a PLY file: {scene_ply}")
            file_format = ""
            vertex_count = 0
            vertex_properties: list[tuple[str, str]] = []
            current_element = ""
            while True:
                raw_line = handle.readline()
                if not raw_line:
                    raise ValueError(f"PLY header has no end_header: {scene_ply}")
                line = raw_line.decode("ascii", errors="strict").strip()
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "format":
                    file_format = parts[1]
                elif parts[0] == "element":
                    current_element = parts[1]
                    if current_element == "vertex":
                        vertex_count = int(parts[2])
                elif parts[0] == "property" and current_element == "vertex":
                    if parts[1] == "list":
                        raise ValueError("List properties are unsupported inside PLY vertex elements")
                    vertex_properties.append((parts[2], parts[1]))
                elif parts[0] == "end_header":
                    break
            if vertex_count <= 0 or not {"x", "y", "z"}.issubset(name for name, _ in vertex_properties):
                raise ValueError(f"PLY has no usable vertex positions: {scene_ply}")
            if file_format == "ascii":
                data = np.loadtxt(handle, max_rows=vertex_count)
                names = [name for name, _ in vertex_properties]
                points = np.column_stack([data[:, names.index(axis)] for axis in ("x", "y", "z")])
            elif file_format in {"binary_little_endian", "binary_big_endian"}:
                endian = "<" if file_format == "binary_little_endian" else ">"
                dtype = np.dtype([(name, endian + ply_types[type_name]) for name, type_name in vertex_properties])
                data = np.fromfile(handle, dtype=dtype, count=vertex_count)
                points = np.column_stack((data["x"], data["y"], data["z"]))
            else:
                raise ValueError(f"Unsupported PLY format '{file_format}': {scene_ply}")
        points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Scene point cloud contains no points: {scene_ply}")
    points = points.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 100:
        raise ValueError(
            f"At least 100 finite scene points are required; found {points.shape[0]}"
        )
    return points


def refine_plane(points: np.ndarray, camera_up: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane to inliers with PCA and orient its normal upward."""

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / float(points.shape[0])
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = normalize(eigenvectors[:, 0])
    if float(np.dot(normal, camera_up)) < 0.0:
        normal = -normal
    return normal, float(np.dot(normal, centroid))


def fit_ground_plane(
    points: np.ndarray,
    camera_origin: np.ndarray,
    camera_up: np.ndarray,
) -> GroundPlaneFit:
    """Find the lowest well-supported plane approximately parallel to ground."""

    camera_up = normalize(camera_up)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    scene_diagonal = float(np.linalg.norm(max_bound - min_bound))
    distance_threshold = min(0.05, max(0.01, scene_diagonal * 0.003))

    random_state = np.random.RandomState(20260717)
    sample_count = min(60000, points.shape[0])
    if points.shape[0] > sample_count:
        sample_indices = random_state.choice(
            points.shape[0], sample_count, replace=False
        )
        sample = points[sample_indices]
    else:
        sample = points.copy()

    working = sample
    candidates: list[dict[str, Any]] = []
    minimum_up_alignment = 0.60
    minimum_plane_points = max(100, int(sample_count * 0.002))

    for _ in range(8):
        if working.shape[0] < minimum_plane_points:
            break
        best_count = 0
        best_mask: np.ndarray | None = None
        for _ in range(700):
            indices = random_state.choice(working.shape[0], 3, replace=False)
            point_a, point_b, point_c = working[indices]
            normal = np.cross(point_b - point_a, point_c - point_a)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= EPSILON:
                continue
            normal = normal / normal_norm
            if abs(float(np.dot(normal, camera_up))) < minimum_up_alignment:
                continue
            offset = float(np.dot(normal, point_a))
            inlier_mask = (
                np.abs(working @ normal - offset) <= distance_threshold
            )
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count > best_count:
                best_count = inlier_count
                best_mask = inlier_mask

        if best_mask is None or best_count < minimum_plane_points:
            break

        normal, offset = refine_plane(working[best_mask], camera_up)
        inlier_mask = np.abs(working @ normal - offset) <= distance_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))
        camera_up_alignment = float(np.dot(normal, camera_up))
        relative_height = offset - float(np.dot(normal, camera_origin))
        candidates.append(
            {
                "normal": normal,
                "offset": offset,
                "inlier_count": inlier_count,
                "camera_up_alignment": camera_up_alignment,
                "relative_height": relative_height,
            }
        )
        working = working[~inlier_mask]

    below_camera = [
        candidate
        for candidate in candidates
        if candidate["relative_height"] < -distance_threshold
    ]
    if not below_camera:
        raise RuntimeError(
            "Ground-plane fitting failed: no sufficiently horizontal plane "
            "was found below the first camera."
        )

    maximum_support = max(
        int(candidate["inlier_count"]) for candidate in below_camera
    )
    supported_candidates = [
        candidate
        for candidate in below_camera
        if int(candidate["inlier_count"])
        >= max(minimum_plane_points, int(maximum_support * 0.20))
    ]
    selected = min(
        supported_candidates,
        key=lambda candidate: float(candidate["relative_height"]),
    )

    initial_normal = np.asarray(selected["normal"], dtype=np.float64)
    initial_offset = float(selected["offset"])
    full_inlier_mask = (
        np.abs(points @ initial_normal - initial_offset) <= distance_threshold
    )
    full_inliers = points[full_inlier_mask]
    if full_inliers.shape[0] < minimum_plane_points:
        raise RuntimeError("Ground-plane refinement produced too few inliers")
    normal, offset = refine_plane(full_inliers, camera_up)
    final_inlier_mask = np.abs(points @ normal - offset) <= distance_threshold
    inlier_count = int(np.count_nonzero(final_inlier_mask))
    camera_up_alignment = float(np.dot(normal, camera_up))
    if camera_up_alignment < minimum_up_alignment:
        raise RuntimeError(
            "Ground-plane normal is inconsistent with the first camera up axis: "
            f"alignment={camera_up_alignment:.3f}"
        )

    return GroundPlaneFit(
        normal_raw=normal,
        offset_raw=offset,
        inlier_count=inlier_count,
        sample_count=sample_count,
        distance_threshold=distance_threshold,
        camera_up_alignment=camera_up_alignment,
    )


def estimate_world_alignment(
    points: np.ndarray,
    first_camera_pose_path: Path,
) -> WorldAlignment:
    camera_to_world = load_camera_to_world(first_camera_pose_path)
    camera_rotation = camera_to_world[:3, :3]
    camera_origin = camera_to_world[:3, 3]
    camera_up = normalize(-camera_rotation[:, 1])
    camera_optical_axis = normalize(camera_rotation[:, 2])
    ground_plane = fit_ground_plane(points, camera_origin, camera_up)
    up = ground_plane.normal_raw

    projected_front = camera_optical_axis - float(
        np.dot(camera_optical_axis, up)
    ) * up
    if float(np.linalg.norm(projected_front)) <= 1e-4:
        raise RuntimeError(
            "The first camera optical axis is nearly perpendicular to the ground; "
            "a stable horizontal world-front direction cannot be constructed."
        )
    front = normalize(projected_front)
    left = normalize(np.cross(up, front))
    front = normalize(np.cross(left, up))
    right = -left
    basis_raw = np.column_stack((right, front, up))
    raw_to_aligned = basis_raw.T
    if float(np.linalg.det(raw_to_aligned)) <= 0.0:
        raise RuntimeError("Constructed world alignment is not right-handed")

    camera_height = float(np.dot(up, camera_origin))
    aligned_origin_raw = camera_origin + (
        ground_plane.offset_raw - camera_height
    ) * up
    return WorldAlignment(
        raw_to_aligned=raw_to_aligned,
        aligned_origin_raw=aligned_origin_raw,
        front_raw=front,
        left_raw=left,
        up_raw=up,
        camera_origin_raw=camera_origin,
        camera_optical_axis_raw=camera_optical_axis,
        first_camera_pose_path=first_camera_pose_path,
        ground_plane=ground_plane,
    )


# ---------------------------------------------------------------------------
# Integrated top-level entrypoint
# ---------------------------------------------------------------------------

def _run_integrated_scene(args) -> None:
    """Run the complete pipeline for one already-resolved RGB dataset."""

    started = _time_for_mast3r.perf_counter()
    started_at = _time_for_mast3r.strftime("%Y-%m-%d %H:%M:%S")
    prepare_start = _time_for_mast3r.perf_counter()
    output_dir = _resolve_user_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mast3r_root = _resolve_user_path(args.mast3r_root)
    if not mast3r_root.exists():
        raise FileNotFoundError(f"MASt3R-SLAM root not found: {mast3r_root}")
    prepare_elapsed = _time_for_mast3r.perf_counter() - prepare_start

    timer = PipelineTimer(
        output_dir,
        total_start=started,
        started_at=started_at,
    )
    timer.add_record("0_prepare_pipeline", prepare_elapsed, "ok")

    try:
        with timer.step("1_mast3r_slam_and_export"):
            if not args.skip_mast3r:
                _run_mast3r_slam(args)

            discovered_scan_root = _discover_mast3r_scan_root(args)
            discovered_scene_ply = _find_mast3r_scene_ply(discovered_scan_root, mast3r_root, args.mast3r_save_as)
            scan_root, scene_ply = _materialize_mast3r_outputs(discovered_scan_root, discovered_scene_ply, output_dir)
            _mast3r_logger.info("Discovered MASt3R scan root: %s", discovered_scan_root)
            _mast3r_logger.info("Materialized MASt3R scan root: %s", scan_root)
            if scene_ply:
                _mast3r_logger.info("Materialized MASt3R scene PLY: %s", scene_ply)
            if not args.skip_object_pipeline:
                embedded_argv = _build_embedded_object_argv(args, scan_root, output_dir)
                _mast3r_logger.info("Embedded object pipeline configured; arguments omitted for privacy.")
            else:
                embedded_argv = []

        if not args.skip_object_pipeline:
            if not args.dry_run:
                old_argv = _sys_for_mast3r.argv[:]
                try:
                    _sys_for_mast3r.argv = embedded_argv
                    run_embedded_object_pipeline(timer=timer)
                finally:
                    _sys_for_mast3r.argv = old_argv
            else:
                timer.add_record("2_object_pipeline_skipped", 0.0, "ok", "dry_run")
        else:
            timer.add_record("2_object_pipeline_skipped", 0.0, "ok")

        timer.status = "completed"
    except Exception as exc:
        timer.status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        timing_path = timer.write()
        print(f"[DONE] Timing log: {timing_path}")


def _select_single_rgb_dataset(
    datasets: list[tuple[str, Path]],
    scene_selector: str,
) -> tuple[int, tuple[str, Path]]:
    """Select exactly one discovered scene by name or 1-based index."""

    if not datasets:
        raise ValueError("No RGB scene is available for selection.")

    selector = str(scene_selector).strip()
    if len(datasets) == 1:
        scene_name, _ = datasets[0]
        if selector and selector not in {scene_name, "1"}:
            raise ValueError(
                f"Selected scene {selector!r} does not match the only discovered "
                f"scene {scene_name!r}."
            )
        return 1, datasets[0]

    available = ", ".join(
        f"{index}:{scene_name}"
        for index, (scene_name, _) in enumerate(datasets, start=1)
    )
    if not selector:
        raise ValueError(
            "Multiple ScanNet scenes were discovered. Process them one at a time "
            f"by passing --scene <name-or-index>. Available scenes: {available}"
        )

    # Prefer an exact scene-name match so numeric names such as "00003" are not
    # accidentally interpreted as the third discovered scene.
    for index, dataset in enumerate(datasets, start=1):
        if dataset[0] == selector:
            return index, dataset

    try:
        selected_index = int(selector)
    except ValueError:
        selected_index = 0
    if 1 <= selected_index <= len(datasets):
        return selected_index, datasets[selected_index - 1]

    raise ValueError(
        f"Unknown scene selection {selector!r}. Available scenes: {available}"
    )


def main():
    args = parse_integrated_args()
    _logging_for_mast3r.getLogger().setLevel(getattr(_logging_for_mast3r, args.log_level))

    source_root = _resolve_user_path(args.dataset)
    datasets, is_multi_scene = _discover_rgb_datasets(source_root)
    scene_index, (scene_name, color_input) = _select_single_rgb_dataset(
        datasets,
        args.scene,
    )
    output_root = _resolve_user_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mast3r_root = _resolve_user_path(args.mast3r_root)
    if not mast3r_root.exists():
        raise FileNotFoundError(f"MASt3R-SLAM root not found: {mast3r_root}")

    configured_scan_root = (
        _resolve_user_path(args.mast3r_scan_root)
        if args.mast3r_scan_root
        else None
    )
    print(
        f"[INPUT] Discovered {len(datasets)} scene(s) under {source_root}; "
        f"selected {scene_name!r} ({scene_index}/{len(datasets)})."
    )

    scene_args = _argparse_for_mast3r.Namespace(**vars(args))
    scene_output_dir = output_root / scene_name if is_multi_scene else output_root
    scene_args.output_dir = str(scene_output_dir)
    if is_multi_scene or not args.label_file:
        scene_args.label_file = str(scene_output_dir / "sequence" / "labels.json")

    if configured_scan_root is not None and is_multi_scene:
        # In reuse mode, --mast3r_scan_root is a parent containing one MASt3R
        # RoboSpatial export directory per scene.
        scene_args.mast3r_scan_root = str(configured_scan_root / scene_name)

    print(
        f"[SCENE {scene_index}/{len(datasets)}] {scene_name}: "
        f"{color_input} -> {scene_output_dir}"
    )
    with tempfile.TemporaryDirectory(prefix=f"persistent3d_scan_{scene_name}_") as temporary_dir:
        prepared_dataset = _prepare_mast3r_dataset(
            color_input,
            Path(temporary_dir),
            scene_name,
        )
        scene_args.dataset = str(prepared_dataset)
        _run_integrated_scene(scene_args)


if __name__ == "__main__":
    main()
