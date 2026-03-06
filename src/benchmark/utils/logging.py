from datetime import datetime


def log(message: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {message}")
