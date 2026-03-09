import base64
import json
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

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
    return "audio/mpeg"


def _build_audio_content(audio_path: str, audio_field: str) -> Dict[str, Any]:
    audio_bytes = open(audio_path, "rb").read()
    mime = _mime_from_path(audio_path)
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    if audio_field == "input_audio":
        fmt = os.path.splitext(audio_path)[1].lower().lstrip(".") or "wav"
        return {
            "type": "input_audio",
            "input_audio": {"data": data_url, "format": fmt},
        }
    # default: data URL
    return {"type": "audio_url", "audio_url": {"url": data_url}}


def _build_gemini_audio_part(audio_path: str) -> Dict[str, Any]:
    audio_bytes = open(audio_path, "rb").read()
    mime = _mime_from_path(audio_path)
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return {"inline_data": {"mime_type": mime, "data": b64}}


def _is_gemini_model(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return "gemini" in normalized or normalized.startswith("vertex_ai.")


def _build_payload(
    requested_model: str,
    prompt: str,
    audio_path: str,
    audio_field: str,
    audio_modalities: Optional[List[str]],
    audio_voice: Optional[str],
) -> Dict[str, Any]:
    if _is_gemini_model(requested_model):
        return {
            "model": requested_model,
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        _build_gemini_audio_part(audio_path),
                    ],
                }
            ],
        }

    audio_content = _build_audio_content(audio_path, audio_field)
    fmt = os.path.splitext(audio_path)[1].lower().lstrip(".") or "wav"
    modalities = audio_modalities or ["text"]
    payload: Dict[str, Any] = {
        "model": requested_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    audio_content,
                ],
            }
        ],
        "modalities": modalities,
        "response_format": {"type": "json_object"},
    }
    if audio_field == "input_audio" and "audio" in modalities:
        payload["audio"] = {"format": fmt}
        if audio_voice:
            payload["audio"]["voice"] = audio_voice
    return payload


def _extract_text(resp_json: Dict[str, Any]) -> str:
    choices = resp_json.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        audio = message.get("audio") or {}
        transcript = audio.get("transcript")
        if isinstance(transcript, str):
            return transcript

    candidates = resp_json.get("candidates") or []
    if candidates:
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for item in parts:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
        return "".join(texts)

    return ""


class CompatibleChatRunner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        audio_field: str = "input_audio",
        audio_voice: Optional[str] = None,
        audio_modalities: Optional[List[str]] = None,
        timeout_sec: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY or OPENAI_API_KEY.")
        self.endpoint_url = endpoint_url
        self.audio_field = audio_field
        self.audio_voice = audio_voice
        self.audio_modalities = audio_modalities
        self.timeout_sec = timeout_sec

    def run(
        self,
        audio_path: str,
        prompt: str,
        model: str,
        fallback_model: Optional[str] = None,
        max_retries: int = 3,
    ) -> Tuple[str, Dict[str, Any], str, float]:
        def _call(requested_model: str) -> Tuple[str, Dict[str, Any], str, float]:
            payload = _build_payload(
                requested_model=requested_model,
                prompt=prompt,
                audio_path=audio_path,
                audio_field=self.audio_field,
                audio_modalities=self.audio_modalities,
                audio_voice=self.audio_voice,
            )

            started = time.perf_counter()

            def _do_call() -> Dict[str, Any]:
                payload_str = json.dumps(payload, ensure_ascii=False)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                    tmp.write(payload_str)
                    tmp_path = tmp.name
                try:
                    cmd = [
                        "curl",
                        "--silent",
                        "--show-error",
                        "--location",
                        "--max-time",
                        str(self.timeout_sec),
                        "--header",
                        "Content-Type: application/json",
                        "--header",
                        f"Authorization: Bearer {self.api_key}",
                        "--data",
                        f"@{tmp_path}",
                        "--write-out",
                        "\\n__STATUS__:%{http_code}",
                        self.endpoint_url,
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                finally:
                    os.unlink(tmp_path)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or "curl_error")
                out = proc.stdout
                if "__STATUS__:" not in out:
                    raise RuntimeError("curl_missing_status")
                body, status_str = out.rsplit("__STATUS__:", 1)
                status = int(status_str.strip() or "0")
                body = body.strip()
                if status < 200 or status >= 300:
                    raise RuntimeError(f"http_{status}:{body[:4000]}")
                return json.loads(body) if body else {}

            resp_json = run_with_retry(_do_call, max_retries=max_retries)
            elapsed = time.perf_counter() - started
            text = _extract_text(resp_json)
            return text, resp_json, requested_model, elapsed

        try:
            return _call(model)
        except Exception:
            if fallback_model:
                return _call(fallback_model)
            raise
