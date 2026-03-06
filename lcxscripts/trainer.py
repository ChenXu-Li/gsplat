import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

# Ensure the examples directory (which contains the `datasets` package) is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

# Import both 3DGS and 2DGS trainers; we'll choose at runtime based on config.
import examples.simple_trainer as simple_trainer_3d  # type: ignore
import examples.simple_trainer_2dgs as simple_trainer_2d  # type: ignore
from gsplat.distributed import cli  # type: ignore


def load_config(config_path: Path) -> Tuple[object, str]:
    """Load YAML config and construct either a 3DGS or 2DGS Config object.

    Returns:
        cfg: Instantiated config dataclass (from the chosen trainer module).
        model_type: Either \"3dgs\" or \"2dgs\".
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Config file must contain a mapping, got: {type(raw_cfg)}")

    # New optional field to choose between 3DGS and 2DGS trainers.
    model_type = str(raw_cfg.get("model_type", "3dgs")).lower()
    if model_type not in {"3dgs", "2dgs"}:
        raise ValueError(
            f"Unsupported model_type '{model_type}' in config. "
            "Expected '3dgs' or '2dgs'."
        )

    # Replace [model_type] placeholder in result_dir if present
    if "result_dir" in raw_cfg and isinstance(raw_cfg["result_dir"], str):
        raw_cfg["result_dir"] = raw_cfg["result_dir"].replace("[model_type]", model_type)

    if model_type == "3dgs":
        ConfigCls = simple_trainer_3d.Config
    else:
        ConfigCls = simple_trainer_2d.Config

    # Filter to known fields of the chosen Config to avoid unexpected-key errors.
    cfg_fields = {k for k in ConfigCls.__dataclass_fields__.keys()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw_cfg.items() if k in cfg_fields}

    return ConfigCls(**filtered), model_type  # type: ignore[call-arg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LCX trainer wrapper for gsplat.examples.simple_trainer"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML config. If omitted, use default lcx_trainer_config.yaml next to this script.",
    )
    parser.add_argument(
        "--steps_scaler",
        type=float,
        required=False,
        help="Optional override for Config.steps_scaler (scales training length).",
    )
    return parser.parse_args()


def main_entry() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    # When no --config is provided, fall back to default config file
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = script_dir / "trainer_config.yaml"

    cfg, model_type = load_config(config_path)

    # Optional CLI override for steps_scaler
    if args.steps_scaler is not None:
        cfg.steps_scaler = args.steps_scaler  # type: ignore[attr-defined]

    # Adjust training schedule based on steps_scaler, consistently with examples.*
    cfg.adjust_steps(cfg.steps_scaler)  # type: ignore[attr-defined]

    # Ensure we run from the repo root so relative paths in examples work as expected
    repo_root = script_dir.parent
    os.chdir(repo_root)

    # Choose the correct trainer entry point based on model_type.
    # gsplat.distributed.cli expects a function with the signature:
    #   fn(local_rank, world_rank, world_size, cfg)
    # examples.simple_trainer.main already matches this, but
    # examples.simple_trainer_2dgs.main takes only (cfg), so we adapt it.
    if model_type == "3dgs":
        main_fn = simple_trainer_3d.main

        def wrapped_main(local_rank: int, world_rank: int, world_size: int, cfg_obj: object):
            return main_fn(local_rank, world_rank, world_size, cfg_obj)  # type: ignore[arg-type]

    else:
        def wrapped_main(local_rank: int, world_rank: int, world_size: int, cfg_obj: object):
            # 2DGS trainer ignores distributed ranks internally and only uses cfg.
            return simple_trainer_2d.main(cfg_obj)  # type: ignore[arg-type]

    # Launch distributed / single-GPU training via gsplat.distributed.cli
    cli(wrapped_main, cfg, verbose=True)


if __name__ == "__main__":
    main_entry()

