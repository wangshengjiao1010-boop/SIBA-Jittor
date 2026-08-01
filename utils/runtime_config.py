import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "siba.json"


def load_config(path=None):
    config_path = Path(path or DEFAULT_CONFIG)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["_path"] = config_path.resolve()
    return config


def resolve_path(value):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
