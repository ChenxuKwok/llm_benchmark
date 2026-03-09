from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmark.utils.text import wer as compute_wer


def _extract_int(item: Any) -> Optional[int]:
    if isinstance(item, int):
        return item
    if isinstance(item, str) and item.strip().isdigit():
        return int(item.strip())
    return None


def _normalize_word_truth(raw: Any) -> Tuple[Optional[set[int]], Optional[str]]:
    if raw is None:
        return None, "missing_word_labels"
    if isinstance(raw, dict):
        for key in ["errors", "items", "word_errors", "labels"]:
            if key in raw:
                return _normalize_word_truth(raw[key])
    if isinstance(raw, list):
        locs: List[int] = []
        for item in raw:
            if isinstance(item, dict):
                for key in ["word_location", "word_index", "word_idx", "index", "idx"]:
                    loc = _extract_int(item.get(key))
                    if loc is not None:
                        locs.append(loc)
                        break
            else:
                loc = _extract_int(item)
                if loc is not None:
                    locs.append(loc)
        return set(locs), None
    return None, "unrecognized_word_labels"


def _normalize_phoneme_truth(raw: Any) -> Tuple[Optional[set[Tuple]], Optional[str]]:
    if raw is None:
        return None, "missing_phoneme_labels"
    if isinstance(raw, dict):
        for key in ["errors", "items", "phoneme_errors", "labels"]:
            if key in raw:
                return _normalize_phoneme_truth(raw[key])
    if isinstance(raw, list):
        tuples = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            error_type = item.get("error_type") or item.get("type")
            if isinstance(error_type, str):
                error_type = error_type.strip().lower()
            if error_type not in {"substitution", "deletion", "insertion"}:
                continue
            loc = None
            for key in ["word_location", "word_index", "word_idx", "index", "idx"]:
                loc = _extract_int(item.get(key))
                if loc is not None:
                    break
            if loc is None:
                continue
            expected = item.get("phoneme_expected") or item.get("expected")
            actual = item.get("phoneme_actual") or item.get("actual")
            if expected is not None and not isinstance(expected, str):
                expected = str(expected)
            if actual is not None and not isinstance(actual, str):
                actual = str(actual)
            tuples.append((error_type, loc, expected, actual))
        return set(tuples), None
    return None, "unrecognized_phoneme_labels"


def _pred_word_locations(parsed: Dict[str, Any]) -> set[int]:
    errors = parsed.get("errors", [])
    locs = set()
    for item in errors:
        if isinstance(item, dict):
            loc = _extract_int(item.get("word_location"))
            if loc is not None:
                locs.add(loc)
    return locs


def _pred_phoneme_tuples(parsed: Dict[str, Any]) -> set[Tuple]:
    errors = parsed.get("errors", [])
    tuples = set()
    for item in errors:
        if not isinstance(item, dict):
            continue
        error_type = item.get("error_type")
        if isinstance(error_type, str):
            error_type = error_type.strip().lower()
        if error_type not in {"substitution", "deletion", "insertion"}:
            continue
        loc = _extract_int(item.get("word_location"))
        if loc is None:
            continue
        expected = item.get("phoneme_expected")
        actual = item.get("phoneme_actual")
        tuples.add((error_type, loc, expected, actual))
    return tuples


@dataclass
class MetricAvailability:
    available: bool
    reason: Optional[str] = None


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_reference_wer(samples: Iterable, parsed_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    wers: List[float] = []
    for sample in samples:
        parsed = parsed_map.get(sample.sample_id)
        if not parsed:
            continue
        if not sample.reference_text:
            continue
        pred_ref = parsed.get("reference", "")
        wers.append(compute_wer(sample.reference_text, pred_ref))
    if not wers:
        return {"available": False, "reason": "no_reference_text_available"}
    return {"available": True, "average_wer": sum(wers) / len(wers), "count": len(wers)}


def compute_word_metrics(samples: Iterable, parsed_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = 0
    any_labels = False
    missing_reason = None
    for sample in samples:
        parsed = parsed_map.get(sample.sample_id)
        if not parsed:
            continue
        truth, reason = _normalize_word_truth(sample.word_errors_raw)
        if truth is None:
            missing_reason = reason
            continue
        any_labels = True
        pred = _pred_word_locations(parsed)
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    if not any_labels:
        return {"available": False, "reason": missing_reason or "no_word_labels"}
    metrics = _precision_recall_f1(tp, fp, fn)
    metrics.update({"available": True, "tp": tp, "fp": fp, "fn": fn})
    return metrics


def compute_phoneme_metrics(samples: Iterable, parsed_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = 0
    any_labels = False
    missing_reason = None
    for sample in samples:
        parsed = parsed_map.get(sample.sample_id)
        if not parsed:
            continue
        truth, reason = _normalize_phoneme_truth(sample.phoneme_errors_raw)
        if truth is None:
            missing_reason = reason
            continue
        any_labels = True
        pred = _pred_phoneme_tuples(parsed)
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
    if not any_labels:
        return {"available": False, "reason": missing_reason or "no_phoneme_labels"}
    metrics = _precision_recall_f1(tp, fp, fn)
    metrics.update({"available": True, "tp": tp, "fp": fp, "fn": fn})
    return metrics
