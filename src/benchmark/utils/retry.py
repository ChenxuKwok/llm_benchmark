import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _default_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return True
    if "rate limit" in message or "429" in message:
        return True
    if "temporarily" in message or "unavailable" in message:
        return True
    status = getattr(exc, "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    return False


def run_with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    retryable: Callable[[Exception], bool] | None = None,
) -> T:
    attempt = 0
    retryable = retryable or _default_retryable
    while True:
        try:
            return func()
        except Exception as exc:
            if attempt >= max_retries or not retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay = delay * (0.8 + 0.4 * random.random())
            time.sleep(delay)
            attempt += 1
