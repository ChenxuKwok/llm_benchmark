import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Tuple

from benchmark.data.l2arctic_plus import DatasetSample, load_l2arctic_plus
from benchmark.eval.metrics import (
    compute_phoneme_metrics,
    compute_reference_wer,
    compute_word_metrics,
)
from benchmark.eval.report import write_summary
from benchmark.parsing.repair import parse_and_validate
from benchmark.prompts.capt import PROMPT_VERSION, build_prompt
from benchmark.runners.compatible_runner import CompatibleChatRunner
from benchmark.utils.io import append_jsonl, ensure_dir, read_json, write_json
from benchmark.utils.logging import log


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_run_name(prefix: str) -> str:
    return f"{prefix}_{_timestamp()}"


def _summarize_label_sample(value: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, list) and value:
        first = value[0]
        summary["first_type"] = type(first).__name__
        if isinstance(first, dict):
            summary["first_keys"] = sorted(first.keys())
    elif isinstance(value, dict):
        summary["keys"] = sorted(value.keys())
    return summary


def _write_schema_report(
    path: str, schema: Any, samples: List[DatasetSample]
) -> None:
    sample_keys = []
    label_samples = []
    for sample in samples[:3]:
        sample_keys.append(sorted(sample.raw.keys()))
        label_samples.append(
            {
                "sample_id": sample.sample_id,
                "word_errors": _summarize_label_sample(sample.word_errors_raw),
                "phoneme_errors": _summarize_label_sample(sample.phoneme_errors_raw),
            }
        )
    report = {
        "schema": {
            "audio_key": schema.audio_key,
            "reference_key": schema.reference_key,
            "word_error_key": schema.word_error_key,
            "phoneme_error_key": schema.phoneme_error_key,
            "sample_id_key": schema.sample_id_key,
        },
        "sample_keys": sample_keys,
        "label_samples": label_samples,
    }
    write_json(path, report)


def _output_paths(run_dir: str, sample_id: str) -> Dict[str, str]:
    raw_dir = os.path.join(run_dir, "raw")
    parsed_dir = os.path.join(run_dir, "parsed")
    meta_dir = os.path.join(run_dir, "meta")
    ensure_dir(raw_dir)
    ensure_dir(parsed_dir)
    ensure_dir(meta_dir)
    return {
        "raw": os.path.join(raw_dir, f"{sample_id}.json"),
        "parsed": os.path.join(parsed_dir, f"{sample_id}.json"),
        "meta": os.path.join(meta_dir, f"{sample_id}.json"),
    }


def _is_completed(paths: Dict[str, str]) -> bool:
    return os.path.exists(paths["raw"]) and os.path.exists(paths["parsed"])


def _load_parsed_outputs(run_dir: str, sample_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    parsed_dir = os.path.join(run_dir, "parsed")
    parsed_map: Dict[str, Dict[str, Any]] = {}
    for sample_id in sample_ids:
        path = os.path.join(parsed_dir, f"{sample_id}.json")
        if os.path.exists(path):
            try:
                parsed_map[sample_id] = read_json(path)
            except Exception:
                continue
    return parsed_map


def _load_meta_outputs(run_dir: str, sample_ids: List[str]) -> List[Dict[str, Any]]:
    meta_dir = os.path.join(run_dir, "meta")
    meta_list: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        path = os.path.join(meta_dir, f"{sample_id}.json")
        if os.path.exists(path):
            try:
                meta_list.append(read_json(path))
            except Exception:
                continue
    return meta_list


def _run_parser_smoke() -> None:
    sample = (
        '{"reference":"test","errors":[{"error_type":"substitution","word_location":0,'
        '"phoneme_expected":"AH","phoneme_actual":"AA","confidence":0.9}],'
        '"explanation":[{"location":0,"content":"example"}],'
        '"suggestion":[{"location":0,"content":"example"}]}'
    )
    result = parse_and_validate(sample)
    if result.data is None:
        raise RuntimeError("Parser smoke test failed")


def run_benchmark(
    dataset_root: str,
    split: str,
    backend: str,
    model: str,
    mode: str,
    run_name: str | None = None,
    fallback_model: str | None = None,
    limit: int | None = None,
    workers: int = 1,
    force: bool = False,
    endpoint_url: str | None = None,
    audio_field: str | None = None,
    audio_voice: str | None = None,
    audio_modalities: List[str] | None = None,
    max_retries: int = 3,
) -> str:
    _run_parser_smoke()
    samples, schema = load_l2arctic_plus(dataset_root, split)
    if limit:
        samples = samples[:limit]

    run_name = run_name or _default_run_name(f"{backend}_{mode}")
    run_dir = os.path.join("results", run_name)
    ensure_dir(run_dir)
    write_json(os.path.join(run_dir, "config.json"), {
        "dataset_root": dataset_root,
        "split": split,
        "backend": backend,
        "model": model,
        "fallback_model": fallback_model,
        "mode": mode,
        "limit": limit,
        "workers": workers,
        "endpoint_url": endpoint_url,
        "audio_field": audio_field,
        "audio_voice": audio_voice,
        "audio_modalities": audio_modalities,
    })
    _write_schema_report(os.path.join(run_dir, "schema_report.json"), schema, samples)
    jsonl_path = os.path.join(run_dir, "records.jsonl")
    jsonl_lock = Lock()

    if backend not in {"openai", "gemini"}:
        raise ValueError("backend must be openai or gemini")
    runner = CompatibleChatRunner(
        endpoint_url=endpoint_url
        or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        audio_field=audio_field or "input_audio",
        audio_voice=audio_voice,
        audio_modalities=audio_modalities,
    )

    log(f"Starting run {run_name} with {len(samples)} samples")

    def _process(sample: DatasetSample) -> Tuple[str, Dict[str, Any]]:
        paths = _output_paths(run_dir, sample.sample_id)
        if not force and _is_completed(paths):
            if os.path.exists(paths["meta"]):
                meta = read_json(paths["meta"])
                record = {
                    "sample_id": sample.sample_id,
                    "audio_path": sample.audio_path,
                    "mode": mode,
                    "reference_text": sample.reference_text,
                    "requested_model": model,
                    "actual_model_used": meta.get("actual_model_used"),
                    "prompt_version": PROMPT_VERSION,
                    "prompt": None,
                    "status": "skipped_resume",
                    "raw_response": None,
                    "response_text": None,
                    "parsed": None,
                    "parse_errors": meta.get("parse_errors"),
                }
                with jsonl_lock:
                    append_jsonl(jsonl_path, record)
                return sample.sample_id, meta
            meta = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "latency_sec": None,
                "raw_response_path": paths["raw"],
                "parsed_response_path": paths["parsed"],
                "status": "skipped_resume",
            }
            write_json(paths["meta"], meta)
            record = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "reference_text": sample.reference_text,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "prompt": None,
                "status": "skipped_resume",
                "raw_response": None,
                "response_text": None,
                "parsed": None,
                "parse_errors": None,
            }
            with jsonl_lock:
                append_jsonl(jsonl_path, record)
            return sample.sample_id, meta
        if not sample.audio_exists:
            meta = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "latency_sec": None,
                "raw_response_path": None,
                "parsed_response_path": None,
                "status": "skipped_missing_audio",
                "error": "audio_not_found",
            }
            write_json(paths["meta"], meta)
            record = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "reference_text": sample.reference_text,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "prompt": None,
                "status": "skipped_missing_audio",
                "raw_response": None,
                "response_text": None,
                "parsed": None,
                "parse_errors": None,
            }
            with jsonl_lock:
                append_jsonl(jsonl_path, record)
            return sample.sample_id, meta
        if mode == "reference_given" and not sample.reference_text:
            meta = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "latency_sec": None,
                "raw_response_path": None,
                "parsed_response_path": None,
                "status": "skipped_no_reference",
                "error": "reference_text_missing",
            }
            write_json(paths["meta"], meta)
            record = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "reference_text": sample.reference_text,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "prompt": None,
                "status": "skipped_no_reference",
                "raw_response": None,
                "response_text": None,
                "parsed": None,
                "parse_errors": None,
            }
            with jsonl_lock:
                append_jsonl(jsonl_path, record)
            return sample.sample_id, meta

        prompt = build_prompt(mode, sample.reference_text)
        try:
            text, raw, actual_model, latency = runner.run(
                sample.audio_path,
                prompt,
                model=model,
                fallback_model=fallback_model,
                max_retries=max_retries,
            )
            raw_payload = {
                "requested_model": model,
                "actual_model_used": actual_model,
                "prompt_version": PROMPT_VERSION,
                "response_text": text,
                "raw_response": raw,
            }
            write_json(paths["raw"], raw_payload)
        except Exception as exc:
            meta = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "latency_sec": None,
                "raw_response_path": None,
                "parsed_response_path": None,
                "status": "api_error",
                "error": str(exc),
            }
            write_json(paths["meta"], meta)
            record = {
                "sample_id": sample.sample_id,
                "audio_path": sample.audio_path,
                "mode": mode,
                "reference_text": sample.reference_text,
                "requested_model": model,
                "actual_model_used": None,
                "prompt_version": PROMPT_VERSION,
                "prompt": prompt,
                "status": "api_error",
                "raw_response": None,
                "response_text": None,
                "parsed": None,
                "parse_errors": [str(exc)],
            }
            with jsonl_lock:
                append_jsonl(jsonl_path, record)
            return sample.sample_id, meta

        parse_result = parse_and_validate(text)
        if parse_result.data is not None:
            if mode == "reference_given" and sample.reference_text:
                if parse_result.data.get("reference") != sample.reference_text:
                    parse_result.warnings.append("reference_mismatch")
            write_json(paths["parsed"], parse_result.data)
            status = "success"
        else:
            status = "parse_error"
        meta = {
            "sample_id": sample.sample_id,
            "audio_path": sample.audio_path,
            "mode": mode,
            "requested_model": model,
            "actual_model_used": actual_model,
            "prompt_version": PROMPT_VERSION,
            "latency_sec": latency,
            "raw_response_path": paths["raw"],
            "parsed_response_path": paths["parsed"] if status == "success" else None,
            "status": status,
            "parse_errors": parse_result.errors,
            "parse_warnings": parse_result.warnings,
        }
        write_json(paths["meta"], meta)
        record = {
            "sample_id": sample.sample_id,
            "audio_path": sample.audio_path,
            "mode": mode,
            "reference_text": sample.reference_text,
            "requested_model": model,
            "actual_model_used": actual_model,
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
            "status": status,
            "raw_response": raw,
            "response_text": text,
            "parsed": parse_result.data if status == "success" else None,
            "parse_errors": parse_result.errors,
            "parse_warnings": parse_result.warnings,
        }
        with jsonl_lock:
            append_jsonl(jsonl_path, record)
        return sample.sample_id, meta

    results: Dict[str, Dict[str, Any]] = {}
    if workers <= 1:
        for sample in samples:
            sample_id, meta = _process(sample)
            results[sample_id] = meta
            log(f"{sample_id}: {meta.get('status')}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, sample): sample for sample in samples}
            for future in as_completed(futures):
                sample = futures[future]
                sample_id, meta = future.result()
                results[sample_id] = meta
                log(f"{sample_id}: {meta.get('status')}")

    sample_ids = [s.sample_id for s in samples]
    parsed_map = _load_parsed_outputs(run_dir, sample_ids)
    meta_list = _load_meta_outputs(run_dir, sample_ids)

    attempted_samples = sum(1 for m in meta_list if m.get("status") not in {"skipped_missing_audio", "skipped_no_reference"})
    successful_samples = sum(1 for m in meta_list if m.get("status") == "success")
    api_failures = sum(1 for m in meta_list if m.get("status") == "api_error")
    parse_failures = sum(1 for m in meta_list if m.get("status") == "parse_error")
    latency_values = [
        m.get("latency_sec") for m in meta_list if m.get("latency_sec") is not None
    ]
    avg_latency = sum(latency_values) / len(latency_values) if latency_values else None

    total_with_raw = sum(
        1 for m in meta_list if m.get("status") in {"success", "parse_error"}
    )
    json_validity = {
        "valid_rate": (successful_samples / total_with_raw) if total_with_raw else 0.0,
        "valid_count": successful_samples,
        "total_with_raw": total_with_raw,
    }

    metrics = {
        "stats": {
            "attempted_samples": attempted_samples,
            "successful_samples": successful_samples,
            "api_failures": api_failures,
            "parse_failures": parse_failures,
            "average_latency_sec": avg_latency,
        },
        "json_validity": json_validity,
        "reference_wer": compute_reference_wer(samples, parsed_map),
        "word_metrics": compute_word_metrics(samples, parsed_map),
        "phoneme_metrics": compute_phoneme_metrics(samples, parsed_map),
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    write_summary(os.path.join(run_dir, "summary.md"), metrics)
    return run_dir
