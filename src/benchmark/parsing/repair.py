import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class ParseResult:
    data: Optional[Dict[str, Any]]
    repaired_text: Optional[str]
    errors: List[str]
    warnings: List[str]


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_outermost_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _repair_common(text: str) -> str:
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    text = text.replace("\t", " ")
    return text


def _coerce_loc(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_errors(items: Any, warnings: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        warnings.append("errors_not_list")
        return []
    cleaned: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            warnings.append("error_item_not_dict")
            continue
        error_type = item.get("error_type")
        if isinstance(error_type, str):
            error_type = error_type.strip().lower()
        if error_type not in {"substitution", "deletion", "insertion"}:
            warnings.append("error_type_invalid")
            continue
        word_location = _coerce_loc(item.get("word_location"))
        if word_location is None:
            warnings.append("word_location_invalid")
            continue
        phoneme_expected = item.get("phoneme_expected")
        if phoneme_expected is not None and not isinstance(phoneme_expected, str):
            phoneme_expected = str(phoneme_expected)
        phoneme_actual = item.get("phoneme_actual")
        if phoneme_actual is not None and not isinstance(phoneme_actual, str):
            phoneme_actual = str(phoneme_actual)
        confidence = item.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
            warnings.append("confidence_invalid")
        cleaned.append(
            {
                "error_type": error_type,
                "word_location": word_location,
                "phoneme_expected": phoneme_expected,
                "phoneme_actual": phoneme_actual,
                "confidence": confidence,
            }
        )
    return cleaned


def _coerce_loc_content(items: Any, warnings: List[str], field: str) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        warnings.append(f"{field}_not_list")
        return []
    cleaned: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            warnings.append(f"{field}_item_not_dict")
            continue
        location = _coerce_loc(item.get("location"))
        if location is None:
            warnings.append(f"{field}_location_invalid")
            continue
        content = item.get("content")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        cleaned.append({"location": location, "content": content})
    return cleaned


def _validate(obj: Any) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(obj, dict):
        errors.append("root_not_object")
        return None, errors, warnings
    reference = obj.get("reference")
    if reference is None:
        reference = ""
    if not isinstance(reference, str):
        reference = str(reference)
        warnings.append("reference_coerced")
    errors_list = _coerce_errors(obj.get("errors", []), warnings)
    explanation_list = _coerce_loc_content(obj.get("explanation", []), warnings, "explanation")
    suggestion_list = _coerce_loc_content(obj.get("suggestion", []), warnings, "suggestion")
    cleaned = {
        "reference": reference,
        "errors": errors_list,
        "explanation": explanation_list,
        "suggestion": suggestion_list,
    }
    return cleaned, errors, warnings


def parse_and_validate(text: str) -> ParseResult:
    errors: List[str] = []
    warnings: List[str] = []
    if not text:
        return ParseResult(None, None, ["empty_response"], warnings)
    cleaned = _strip_fences(text)
    extracted = _extract_outermost_json(cleaned)
    if extracted is None:
        return ParseResult(None, None, ["no_json_object_found"], warnings)
    repaired = _repair_common(extracted)
    try:
        obj = json.loads(repaired)
    except Exception as exc:
        errors.append(f"json_parse_error:{exc}")
        return ParseResult(None, repaired, errors, warnings)
    data, val_errors, val_warnings = _validate(obj)
    errors.extend(val_errors)
    warnings.extend(val_warnings)
    if data is None:
        return ParseResult(None, repaired, errors, warnings)
    return ParseResult(data, repaired, errors, warnings)
