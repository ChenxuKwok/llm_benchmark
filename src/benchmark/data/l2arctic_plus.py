import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from benchmark.utils.io import read_json
from benchmark.utils.paths import resolve_audio_path, sanitize_id
from benchmark.utils.text import normalize_words


@dataclass
class DatasetSchema:
    audio_key: Optional[str]
    reference_key: Optional[str]
    word_error_key: Optional[str]
    phoneme_error_key: Optional[str]
    sample_id_key: Optional[str]


@dataclass
class DatasetSample:
    sample_id: str
    audio_path: str
    reference_text: Optional[str]
    raw: Dict[str, Any]
    word_errors_raw: Any
    phoneme_errors_raw: Any
    audio_exists: bool


_AUDIO_KEYS = [
    "audio_path",
    "audio",
    "wav",
    "wav_path",
    "path",
    "audio_filepath",
    "file",
]
_REFERENCE_KEYS = [
    "reference",
    "text",
    "transcript",
    "sentence",
    "prompt",
    "utterance",
]
_WORD_ERROR_KEYS = [
    "word_errors",
    "word_error",
    "word_mispronunciations",
    "word_mispronunciation",
    "word_labels",
    "word_level",
]
_PHONEME_ERROR_KEYS = [
    "phoneme_errors",
    "phoneme_error",
    "phone_errors",
    "phoneme_labels",
    "phoneme_level",
]
_ID_KEYS = ["id", "utt_id", "utterance_id", "sample_id"]
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def _find_first_key(entry: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        if key in entry:
            return key
    return None


def _find_audio_value(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    key = _find_first_key(entry, _AUDIO_KEYS)
    if key:
        value = entry.get(key)
        if isinstance(value, str):
            return key, value
        if isinstance(value, dict):
            for sub_key in ["path", "file", "audio_path"]:
                sub_val = value.get(sub_key)
                if isinstance(sub_val, str):
                    return key, sub_val
    for key, value in entry.items():
        if isinstance(value, str) and value.lower().endswith(_AUDIO_EXTS):
            return key, value
    return None, None


def inspect_schema(entries: List[Dict[str, Any]]) -> DatasetSchema:
    counts = {
        "audio": {},
        "reference": {},
        "word": {},
        "phoneme": {},
        "id": {},
    }
    for entry in entries:
        audio_key, _ = _find_audio_value(entry)
        if audio_key:
            counts["audio"][audio_key] = counts["audio"].get(audio_key, 0) + 1
        ref_key = _find_first_key(entry, _REFERENCE_KEYS)
        if ref_key:
            counts["reference"][ref_key] = counts["reference"].get(ref_key, 0) + 1
        word_key = _find_first_key(entry, _WORD_ERROR_KEYS)
        if word_key:
            counts["word"][word_key] = counts["word"].get(word_key, 0) + 1
        phoneme_key = _find_first_key(entry, _PHONEME_ERROR_KEYS)
        if phoneme_key:
            counts["phoneme"][phoneme_key] = counts["phoneme"].get(phoneme_key, 0) + 1
        id_key = _find_first_key(entry, _ID_KEYS)
        if id_key:
            counts["id"][id_key] = counts["id"].get(id_key, 0) + 1

    def pick_best(bucket: Dict[str, int]) -> Optional[str]:
        if not bucket:
            return None
        return sorted(bucket.items(), key=lambda x: x[1], reverse=True)[0][0]

    return DatasetSchema(
        audio_key=pick_best(counts["audio"]),
        reference_key=pick_best(counts["reference"]),
        word_error_key=pick_best(counts["word"]),
        phoneme_error_key=pick_best(counts["phoneme"]),
        sample_id_key=pick_best(counts["id"]),
    )


def _extract_by_key(entry: Dict[str, Any], key: Optional[str]) -> Any:
    if not key:
        return None
    return entry.get(key)


def load_l2arctic_plus(dataset_root: str, split: str) -> Tuple[List[DatasetSample], DatasetSchema]:
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    dataset_root = _resolve_dataset_root(dataset_root)
    json_path = os.path.join(dataset_root, "L2-Arctic-plus", f"{split}_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dataset file not found: {json_path}. Expected <root>/L2-Arctic-plus/{split}_data.json"
        )
    data = read_json(json_path)
    if isinstance(data, dict):
        entries = data.get("data") or data.get("samples") or []
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError("Dataset JSON must be a list or contain a 'data' list")
    schema = inspect_schema(entries[:200])
    samples: List[DatasetSample] = []
    for idx, entry in enumerate(entries):
        audio_key, audio_value = _find_audio_value(entry)
        audio_path = audio_value or ""
        audio_path = resolve_audio_path(audio_path, dataset_root)
        reference_text = _extract_by_key(entry, schema.reference_key)
        if isinstance(reference_text, dict):
            reference_text = reference_text.get("text") if isinstance(reference_text, dict) else None
        if reference_text is not None and not isinstance(reference_text, str):
            reference_text = str(reference_text)
        raw_id = _extract_by_key(entry, schema.sample_id_key)
        if not raw_id:
            base = os.path.splitext(os.path.basename(audio_path))[0]
            raw_id = base or f"sample_{idx}"
        sample_id = sanitize_id(str(raw_id))
        word_errors_raw = _extract_by_key(entry, schema.word_error_key)
        phoneme_errors_raw = _extract_by_key(entry, schema.phoneme_error_key)
        # L2-Arctic-plus (ALMs4Learning) provides mis_exp_sug keyed by word.
        if word_errors_raw is None and phoneme_errors_raw is None and "mis_exp_sug" in entry:
            word_errors_raw, phoneme_errors_raw = _derive_errors_from_mis_exp_sug(
                entry.get("mis_exp_sug"), reference_text or ""
            )
        audio_exists = bool(audio_path) and os.path.exists(audio_path)
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                audio_path=audio_path,
                reference_text=reference_text,
                raw=entry,
                word_errors_raw=word_errors_raw,
                phoneme_errors_raw=phoneme_errors_raw,
                audio_exists=audio_exists,
            )
        )
    return samples, schema


def _resolve_dataset_root(dataset_root: str) -> str:
    if os.path.isdir(os.path.join(dataset_root, "L2-Arctic-plus")):
        return dataset_root
    # fallback to ALMs4Learning/data if present
    candidate = os.path.join("ALMs4Learning", "data")
    if os.path.isdir(os.path.join(candidate, "L2-Arctic-plus")):
        return candidate
    return dataset_root


def _derive_errors_from_mis_exp_sug(mis_exp_sug: Any, text: str) -> Tuple[List[int], List[Dict[str, Any]]]:
    if not isinstance(mis_exp_sug, dict):
        return [], []
    words = normalize_words(text)
    positions: Dict[str, List[int]] = {}
    for idx, word in enumerate(words):
        positions.setdefault(word, []).append(idx)

    used_positions: Dict[str, int] = {}
    word_locations: List[int] = []
    phoneme_errors: List[Dict[str, Any]] = []
    for raw_word, issues in mis_exp_sug.items():
        tokens = normalize_words(raw_word)
        if not tokens:
            continue
        word = tokens[0]
        loc_list = positions.get(word, [])
        if not loc_list:
            continue
        used = used_positions.get(word, 0)
        if used >= len(loc_list):
            continue
        word_location = loc_list[used]
        used_positions[word] = used + 1
        word_locations.append(word_location)
        if isinstance(issues, list):
            for issue_obj in issues:
                issue = issue_obj.get("issue") if isinstance(issue_obj, dict) else None
                for error_type, expected, actual in _parse_issue_string(issue):
                    phoneme_errors.append(
                        {
                            "error_type": error_type,
                            "word_location": word_location,
                            "phoneme_expected": expected,
                            "phoneme_actual": actual,
                        }
                    )
    return word_locations, phoneme_errors


_ISSUE_QUOTE_RE = re.compile(r'"([^"]+)"')


def _parse_issue_string(issue: Any) -> List[Tuple[str, Optional[str], Optional[str]]]:
    if not isinstance(issue, str):
        return []
    lower = issue.lower()
    quotes = _ISSUE_QUOTE_RE.findall(issue)
    results: List[Tuple[str, Optional[str], Optional[str]]] = []
    if "replaced with" in lower and len(quotes) >= 2:
        results.append(("substitution", quotes[0], quotes[1]))
        return results
    if "added" in lower or "addition" in lower:
        if quotes:
            results.append(("insertion", None, quotes[0]))
        return results
    if "deleted" in lower or "omitted" in lower or "missing" in lower or "removed" in lower:
        if quotes:
            results.append(("deletion", quotes[0], None))
        return results
    return results
