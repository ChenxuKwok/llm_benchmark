import os
import time
from typing import Any, Dict, Tuple

from benchmark.utils.retry import run_with_retry


def _mime_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".m4a":
        return "audio/mp4"
    return "audio/wav"


class GeminiRunner:
    def __init__(self, project: str, location: str) -> None:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Part
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("google-cloud-aiplatform is required for Gemini runner") from exc
        self.vertexai = vertexai
        self.Part = Part
        self.GenerativeModel = GenerativeModel
        self.project = project
        self.location = location

    def run(
        self,
        audio_path: str,
        prompt: str,
        model: str,
        max_retries: int = 3,
    ) -> Tuple[str, Dict[str, Any], str, float]:
        self.vertexai.init(project=self.project, location=self.location)
        audio_bytes = open(audio_path, "rb").read()
        mime_type = _mime_from_path(audio_path)
        model_instance = self.GenerativeModel(model)
        started = time.perf_counter()

        def _do_call() -> Any:
            return model_instance.generate_content(
                [prompt, self.Part.from_data(audio_bytes, mime_type=mime_type)],
                generation_config={"temperature": 0.2},
            )

        resp = run_with_retry(_do_call, max_retries=max_retries)
        elapsed = time.perf_counter() - started
        text = getattr(resp, "text", None) or ""
        raw = {"response": str(resp)}
        return text, raw, model, elapsed
