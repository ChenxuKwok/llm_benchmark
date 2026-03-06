import os


def sanitize_id(value: str) -> str:
    cleaned = value.replace(os.sep, "_").replace(" ", "_")
    cleaned = cleaned.replace("..", "_")
    return cleaned


def _normalize_audio_relpath(audio_path: str) -> str:
    path = audio_path.strip()
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("data/"):
        path = path[5:]
    marker = "L2-Arctic-plus" + os.sep
    if "L2-Arctic-plus/" in path or "L2-Arctic-plus\\" in path:
        idx = path.replace("\\", "/").find("L2-Arctic-plus/")
        if idx >= 0:
            return path.replace("\\", "/")[idx:]
    if marker in path:
        idx = path.find(marker)
        return path[idx:]
    return path


def resolve_audio_path(audio_path: str, dataset_root: str) -> str:
    if os.path.isabs(audio_path):
        return audio_path
    normalized = _normalize_audio_relpath(audio_path)
    candidates = [
        os.path.join(dataset_root, normalized),
        os.path.join(dataset_root, audio_path),
        os.path.join(dataset_root, "l2arctic_release_v5.0", normalized),
        os.path.join(dataset_root, "l2arctic_release_v5.0", audio_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else audio_path
