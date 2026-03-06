from typing import Any, Dict

from benchmark.utils.io import read_json, read_yaml


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    if path.endswith(".json"):
        return read_json(path)
    return read_yaml(path)


def merge_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
