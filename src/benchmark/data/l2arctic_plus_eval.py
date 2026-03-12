import json
import os
from typing import Any, Dict, List, Optional, Tuple


_PHONE_ERROR_TYPE_MAP = {
    "s": "substitution",
    "a": "addition",
    "d": "deletion",
}


class L2ArcticPlusWordDataset:
    """
    Parse L2-Arctic-plus train/test JSON files into structured word-level entries.

    Input:
    - dataset_root: path to the `data/` directory
    - split: `train` or `test`

    Output for each sample:
    - audio_path
    - reference
    - words: list of word-level dictionaries with
      - word
      - reference_phones
      - is_correct
      - errors
    """

    def __init__(self, dataset_root: str, split: str = "train") -> None:
        self.dataset_root = dataset_root
        self.split = split.lower()
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        json_path = self._resolve_json_path()
        with open(json_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        if not isinstance(raw_samples, list):
            raise ValueError(f"{json_path} must contain a JSON list")

        self.samples = [self._parse_sample(sample) for sample in raw_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self.samples)

    def _resolve_json_path(self) -> str:
        candidates = [
            os.path.join(self.dataset_root, "L2-Arctic-plus", f"{self.split}_data.json"),
            os.path.join(self.dataset_root, f"{self.split}_data.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Could not find dataset JSON. Expected one of: "
            + ", ".join(candidates)
        )

    def _parse_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raw_audio_path = sample.get("audio_path", "")
        annotation_info = sample.get("annotation_info", {})

        words = []
        if isinstance(annotation_info, dict):
            for word, phone_annotations in annotation_info.items():
                words.append(_parse_word(word, phone_annotations))

        return {
            "audio_path": self._resolve_audio_path(raw_audio_path),
            "reference": sample.get("text", ""),
            "words": words,
        }

    def _resolve_audio_path(self, audio_path: str) -> str:
        if not isinstance(audio_path, str):
            return ""
        if os.path.isabs(audio_path):
            return audio_path

        normalized = audio_path
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith("data/"):
            normalized = normalized[5:]

        candidates = [
            os.path.join(self.dataset_root, normalized),
            os.path.join(self.dataset_root, "L2-Arctic-plus", normalized),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]


def _parse_word(word: str, phone_annotations: Any) -> Dict[str, Any]:
    reference_phones: List[str] = []
    errors: List[Dict[str, str]] = []

    if isinstance(phone_annotations, list):
        for phone_annotation in phone_annotations:
            parsed = _parse_phone_annotation(phone_annotation)
            if parsed is None:
                if isinstance(phone_annotation, str):
                    reference_phones.append(phone_annotation.strip())
                continue

            should_be, surface, error_type = parsed
            reference_phones.append(should_be)
            errors.append(
                {
                    "should_be": should_be,
                    "surface": surface,
                    "error_type": error_type,
                }
            )

    return {
        "word": word,
        "reference_phones": reference_phones,
        "is_correct": int(len(errors) == 0),
        "errors": errors,
    }


def _parse_phone_annotation(phone_annotation: Any) -> Optional[Tuple[str, str, str]]:
    if not isinstance(phone_annotation, str):
        return None

    parts = [part.strip() for part in phone_annotation.split(",")]
    if len(parts) != 3:
        return None

    should_be, surface, raw_error_type = parts
    error_type = _PHONE_ERROR_TYPE_MAP.get(raw_error_type.lower())
    if error_type is None:
        return None

    return should_be, surface, error_type
