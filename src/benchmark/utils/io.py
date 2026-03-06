import json
import os
import tempfile
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    tmp_dir = os.path.dirname(path) or "."
    ensure_dir(tmp_dir)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_yaml(path: str) -> Any:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read YAML configs")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
