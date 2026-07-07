#!/usr/bin/env python
"""Preflight checks for reproducing MindNav/Co-NavGPT experiments."""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("numpy", "numpy"),
    ("networkx", "networkx"),
    ("yaml", "pyyaml"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
    ("skimage", "scikit-image"),
    ("habitat", "habitat-lab"),
    ("habitat_sim", "habitat-sim"),
    ("detectron2", "detectron2"),
]


def status(ok: bool, label: str, detail: str = "") -> bool:
    prefix = "OK" if ok else "MISSING"
    if detail:
        print(f"[{prefix}] {label}: {detail}")
    else:
        print(f"[{prefix}] {label}")
    return ok


def warn(label: str, detail: str) -> None:
    print(f"[WARN] {label}: {detail}")


def check_import(module_name: str, package_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return status(False, package_name, str(exc))

    version = getattr(module, "__version__", "installed")
    return status(True, package_name, str(version))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_conavgpt = repo_root / "code" / "vendor" / "conavgpt"

    parser = argparse.ArgumentParser(description="Check MindNav reproduction prerequisites.")
    parser.add_argument("--conavgpt-repo", default=os.environ.get("REPO", str(default_conavgpt)))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "val_mini"))
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            os.environ.get(
                "LLM_PATH", "/home/huaziheng/models/Qwen2.5-7B-Instruct"
            ),
        ),
    )
    parser.add_argument(
        "--vlm-path",
        default=os.environ.get(
            "VLM_PATH",
            "/tf/notebooks/models/Qwen2.5-VL-7B-Instruct",
        ),
    )
    parser.add_argument("--check-vlm", action="store_true", help="also require VLM baseline weights")
    parser.add_argument(
        "--brain-backend",
        choices=("local", "siliconflow"),
        default=os.environ.get("BRAIN_BACKEND", "local"),
    )
    args = parser.parse_args()

    ok = True
    conavgpt = Path(args.conavgpt_repo).expanduser().resolve()
    model_path = Path(args.model_path).expanduser()
    vlm_path = Path(args.vlm_path).expanduser()

    print(f"Repo root: {repo_root}")
    print(f"Co-NavGPT repo: {conavgpt}")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    py_ok = sys.version_info[:2] in {(3, 8), (3, 9), (3, 10)}
    ok &= status(py_ok, "Python version", "expected 3.8, 3.9, or 3.10 for Habitat 0.2.1")

    for module_name, package_name in REQUIRED_IMPORTS:
        ok &= check_import(module_name, package_name)

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"{torch.cuda.device_count()} CUDA device(s)" if cuda_ok else "torch.cuda.is_available() is false"
        ok &= status(cuda_ok, "CUDA", detail)
    except Exception as exc:
        ok &= status(False, "CUDA", str(exc))

    ok &= status(conavgpt.exists(), "Co-NavGPT directory", str(conavgpt))
    ok &= status((conavgpt / "exp_main_brain.py").exists(), "MindNav entrypoint", "exp_main_brain.py")
    ok &= status((conavgpt / "exp_main_original.py").exists(), "Co-NavGPT baseline entrypoint", "exp_main_original.py")

    rednet = conavgpt / "RedNet" / "model" / "rednet_semmap_mp3d_40.pth"
    ok &= status(rednet.exists(), "RedNet checkpoint", str(rednet))

    task_cfg = conavgpt / "envs" / "habitat" / "configs" / "tasks" / "multi_objectnav_hm3d.yaml"
    ok &= status(task_cfg.exists(), "Habitat task config", str(task_cfg))

    episode_file = conavgpt / "data" / "datasets" / "objectnav_hm3d_v2" / args.split / f"{args.split}.json.gz"
    scene_dir = conavgpt / "data" / "scene_datasets" / "hm3d_v0.2"
    ok &= status(episode_file.exists(), f"HM3D ObjectNav episodes ({args.split})", str(episode_file))
    ok &= status(scene_dir.exists(), "HM3D v0.2 scenes", str(scene_dir))

    if args.brain_backend == "local":
        ok &= status(model_path.exists(), "Text LLM weights", str(model_path))
    else:
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        ok &= status(bool(api_key), "SILICONFLOW_API_KEY", "required for BRAIN_BACKEND=siliconflow")
        if not model_path.exists():
            warn("Text LLM weights", f"{model_path} not found; remote brain mode will not need it")

    if args.check_vlm:
        ok &= status(vlm_path.exists(), "VLM baseline weights", str(vlm_path))

    if ok:
        print("\nPreflight passed. You can run code/scripts/run_mindnav.sh.")
        return 0

    print("\nPreflight failed. Fix the missing items above before running full Habitat experiments.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
