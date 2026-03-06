from typing import Any, Dict

from benchmark.utils.io import write_text


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary(path: str, metrics: Dict[str, Any]) -> None:
    lines = []
    lines.append("# L2-Arctic-plus CAPT Benchmark Summary")
    lines.append("")
    lines.append("## Run Stats")
    stats = metrics.get("stats", {})
    for key in [
        "attempted_samples",
        "successful_samples",
        "api_failures",
        "parse_failures",
        "average_latency_sec",
    ]:
        lines.append(f"- {key}: {_fmt_metric(stats.get(key))}")
    lines.append("")
    lines.append("## JSON Validity")
    validity = metrics.get("json_validity", {})
    lines.append(f"- valid_rate: {_fmt_metric(validity.get('valid_rate'))}")
    lines.append(f"- valid_count: {_fmt_metric(validity.get('valid_count'))}")
    lines.append(f"- total_with_raw: {_fmt_metric(validity.get('total_with_raw'))}")
    lines.append("")
    lines.append("## Reference Inference (WER)")
    wer = metrics.get("reference_wer", {})
    if not wer.get("available", False):
        lines.append(f"- unavailable: {wer.get('reason')}")
    else:
        lines.append(f"- average_wer: {_fmt_metric(wer.get('average_wer'))}")
        lines.append(f"- count: {_fmt_metric(wer.get('count'))}")
    lines.append("")
    lines.append("## Word-level Mispronunciation Detection")
    word = metrics.get("word_metrics", {})
    if not word.get("available", False):
        lines.append(f"- unavailable: {word.get('reason')}")
    else:
        lines.append(f"- precision: {_fmt_metric(word.get('precision'))}")
        lines.append(f"- recall: {_fmt_metric(word.get('recall'))}")
        lines.append(f"- f1: {_fmt_metric(word.get('f1'))}")
        lines.append(f"- tp/fp/fn: {word.get('tp')}/{word.get('fp')}/{word.get('fn')}")
    lines.append("")
    lines.append("## Phoneme-level Diagnosis")
    phone = metrics.get("phoneme_metrics", {})
    if not phone.get("available", False):
        lines.append(f"- unavailable: {phone.get('reason')}")
    else:
        lines.append(f"- precision: {_fmt_metric(phone.get('precision'))}")
        lines.append(f"- recall: {_fmt_metric(phone.get('recall'))}")
        lines.append(f"- f1: {_fmt_metric(phone.get('f1'))}")
        lines.append(f"- tp/fp/fn: {phone.get('tp')}/{phone.get('fp')}/{phone.get('fn')}")

    write_text(path, "\n".join(lines) + "\n")
