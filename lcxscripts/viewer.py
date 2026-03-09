import argparse
import os
import sys
from pathlib import Path

import yaml

# Ensure the examples directory is on sys.path (required for datasets.colmap import)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from examples.simple_trainer import Config, main  # type: ignore
from gsplat.distributed import cli  # type: ignore


def load_config(config_path: Path) -> tuple[Config, dict]:
    """Load YAML config and construct a Config object, also returning raw dict."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Config file must contain a mapping, got: {type(raw_cfg)}")

    # Filter to known fields of Config to avoid unexpected-key errors
    cfg_fields = {k for k in Config.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw_cfg.items() if k in cfg_fields}

    cfg = Config(**filtered)  # type: ignore[arg-type]
    return cfg, raw_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LCX viewer/eval wrapper for gsplat.examples.simple_trainer"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config. If omitted, use default viewer_config.yaml next to this script.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=False,
        help=(
            "One or more checkpoint .pt files to load for evaluation/rendering. "
            "If omitted, viewer_ckpt from the YAML config will be used."
        ),
    )
    parser.add_argument(
        "--enable_viewer",
        action="store_true",
        help="Enable interactive viewer (default: disabled for offline evaluation).",
    )
    return parser.parse_args()


def main_entry() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    # When no --config is provided, fall back to viewer_config.yaml
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = script_dir / "viewer_config.yaml"

    cfg, raw_cfg = load_config(config_path)

    # Decide which checkpoints to use:
    # 1) CLI --ckpt has highest priority
    # 2) Fallback to viewer_ckpt field in YAML config (string or list)
    ckpt_list: list[str] | None = None

    if args.ckpt:
        ckpt_list = [str(Path(p).expanduser().resolve()) for p in args.ckpt]
    else:
        viewer_ckpt = raw_cfg.get("viewer_ckpt", None)
        if isinstance(viewer_ckpt, str) and viewer_ckpt:
            ckpt_list = [viewer_ckpt]
        elif isinstance(viewer_ckpt, (list, tuple)) and viewer_ckpt:
            ckpt_list = [str(Path(p).expanduser().resolve()) for p in viewer_ckpt]

    if not ckpt_list:
        raise ValueError(
            "No checkpoint specified. Please either:\n"
            "  - pass --ckpt /path/to/ckpt_xxx.pt on the command line, or\n"
            "  - set viewer_ckpt in your YAML config."
        )

    # Use provided checkpoints for eval-only mode (simple_trainer checks this)
    cfg.ckpt = ckpt_list

    # For a "viewer" script we usually don't want training; keep steps as-is but
    # make sure we don't accidentally start a long-running interactive server.
    if not args.enable_viewer:
        cfg.disable_viewer = True

    # Apply render trajectory settings from config
    render_trajectory = raw_cfg.get("render_trajectory", True)
    if not render_trajectory:
        # Skip trajectory rendering by setting path to "none"
        cfg.render_traj_path = "none"
        print("[LCX Viewer] Trajectory rendering disabled (render_trajectory: false)")
    else:
        # Use configured trajectory path
        cfg.render_traj_path = raw_cfg.get("render_traj_path", "interp")
        print(f"[LCX Viewer] Trajectory mode: {cfg.render_traj_path}")

    # Apply metrics computation setting (new run_eval config)
    compute_metrics = raw_cfg.get("compute_metrics", True)
    cfg.run_eval = compute_metrics
    if not compute_metrics:
        print("[LCX Viewer] Metrics computation disabled (compute_metrics: false)")
    else:
        print("[LCX Viewer] Metrics computation enabled (PSNR, SSIM, LPIPS)")

    # Apply video generation setting
    disable_video = raw_cfg.get("disable_video", False)
    cfg.disable_video = disable_video
    if disable_video:
        print("[LCX Viewer] Video generation disabled (saving frames only)")
    else:
        print("[LCX Viewer] Video generation enabled (MP4)")

    # Ensure we run from the repo root so relative paths in examples work as expected
    repo_root = script_dir.parent
    os.chdir(repo_root)

    # Launch evaluation / rendering via gsplat.distributed.cli
    cli(main, cfg, verbose=True)


if __name__ == "__main__":
    main_entry()


