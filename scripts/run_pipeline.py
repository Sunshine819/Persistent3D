#!/usr/bin/env python3
"""Run a Persistent3D pipeline from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SCRIPTS = {
    (REPO_ROOT / "tools/Inference_video.py").resolve(),
    (REPO_ROOT / "tools/inference_scan.py").resolve(),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to a JSON pipeline configuration")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="validate the config without running it",
    )
    args, overrides = parser.parse_known_args()

    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    script = (REPO_ROOT / str(config["script"])).resolve()
    if script not in ALLOWED_SCRIPTS or not script.is_file():
        raise ValueError(f"Unsupported or missing pipeline script: {script}")

    configured_args = config.get("arguments", [])
    if not isinstance(configured_args, list) or not all(
        isinstance(value, str) for value in configured_args
    ):
        raise TypeError("'arguments' must be an array of strings")

    if "environment" in config:
        raise ValueError(
            "Runtime service settings must not be stored in project configuration files."
        )

    command = [sys.executable, str(script), *configured_args, *overrides]
    print("Pipeline configuration validated; command details omitted for privacy.", flush=True)
    if args.print_command:
        return 0
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
