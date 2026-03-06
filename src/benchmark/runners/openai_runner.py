import base64
import os
import time
from typing import Any, Dict, Tuple

from benchmark.utils.retry import run_with_retry


def _audio_format_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext or "wav"


def _serialize_response(resp: Any) -> Dict[str, Any]:
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    return {"response": str(resp)}


def _is_audio_unsupported_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = ["audio", "input_audio", "not supported", "unsupported", "does not support"]
    return any(k in msg for k in keywords)


class OpenAIRunner:
    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for OpenAI runner") from exc
        self.client = OpenAI(api_key=api_key)

    def run(
        self,
        audio_path: str,
        prompt: str,
        model: str,
        fallback_model: str | None = None,
        max_retries: int = 3,
    ) -> Tuple[str, Dict[str, Any], str, float]:
        audio_bytes = open(audio_path, "rb").read()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        fmt = _audio_format_from_path(audio_path)

        def _call(requested_model: str) -> Tuple[str, Dict[str, Any], str, float]:
            started = time.perf_counter()

            def _do_call() -> Any:
                return self.client.responses.create(
                    model=requested_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": audio_b64, "format": fmt},
                                },
                            ],
                        }
                    ],
                )

            resp = run_with_retry(_do_call, max_retries=max_retries)
            elapsed = time.perf_counter() - started
            text = getattr(resp, "output_text", None) or ""
            return text, _serialize_response(resp), requested_model, elapsed

        try:
            return _call(model)
        except Exception as exc:
            if fallback_model and _is_audio_unsupported_error(exc):
                return _call(fallback_model)
            raise
