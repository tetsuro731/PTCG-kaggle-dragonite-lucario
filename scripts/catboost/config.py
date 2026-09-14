from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {"name", "training_module", "feature_module", "rule_policy_module"}
    missing = sorted(required - config.keys())
    if missing:
        raise SystemExit(f"missing config keys: {', '.join(missing)}")
    config["_path"] = str(config_path)
    return config


def import_module(config: dict, key: str) -> ModuleType:
    return importlib.import_module(config[key])