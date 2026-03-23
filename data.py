
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from multiprocessing import get_context, cpu_count
from functools import partial

'''
python build_multitask_chatml_dataset_parallel.py \
  --input filtered.jsonl \
  --output_dir chatml_dataset \
  --train_ratio 0.98 \
  --train_mode sample \
  --dev_mode all \
  --train_samples_per_audio 4 \
  --w_main 0.5 \
  --w_canonical 0.25 \
  --w_edit 0.25 \
  --w_realized_from_transcript 0.0 \
  --save_split_jsonl \
  --num_workers 32 \
  --chunksize 32
'''


# -----------------------------
# Timestamp parsing / alignment
# -----------------------------
def parse_timestamped_segments(s: str) -> List[Dict[str, Any]]:
    if s is None:
        return []

    text = str(s)
    parts = []
    i = 0
    n = len(text)

    while i < n:
        if not text.startswith("<|", i):
            i += 1
            continue

        j = text.find("|>", i)
        if j == -1:
            break

        start_str = text[i + 2:j].strip()
        try:
            start = float(start_str)
        except ValueError:
            i = j + 2
            continue

        k = text.find("<|", j + 2)
        if k == -1:
            break

        l = text.find("|>", k)
        if l == -1:
            break

        end_str = text[k + 2:l].strip()
        try:
            end = float(end_str)
        except ValueError:
            i = l + 2
            continue

        content = text[j + 2:k].strip()
        if content:
            parts.append(
                {
                    "start": start,
                    "end": end,
                    "content": " ".join(content.split()),
                }
            )

        i = l + 2

    return parts


def align_edit_script(src: List[str], tgt: List[str]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    n, m = len(src), len(tgt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        src_i = src[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        bt_i = bt[i]
        for j in range(1, m + 1):
            sub_cost = dp_prev[j - 1] + (0 if src_i == tgt[j - 1] else 1)
            del_cost = dp_prev[j] + 1
            ins_cost = dp_i[j - 1] + 1

            best = min(sub_cost, del_cost, ins_cost)
            dp_i[j] = best

            if best == sub_cost:
                bt_i[j] = "=" if src_i == tgt[j - 1] else "S"
            elif best == del_cost:
                bt_i[j] = "D"
            else:
                bt_i[j] = "I"

    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "=":
            ops.append(("=", src[i - 1], tgt[j - 1]))
            i -= 1
            j -= 1
        elif op == "S":
            ops.append(("S", src[i - 1], tgt[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append(("D", src[i - 1], "∅"))
            i -= 1
        elif op == "I":
            ops.append(("I", "∅", tgt[j - 1]))
            j -= 1
        else:
            break

    ops.reverse()

    lines = []
    for op, a, b in ops:
        if op == "=":
            continue
        if op == "S":
            lines.append(f"SUB {a} -> {b}")
        elif op == "D":
            lines.append(f"DEL {a}")
        elif op == "I":
            lines.append(f"INS {b}")

    return ops, lines


# -----------------------------
# Task builders
# -----------------------------
def build_system_prompt() -> str:
    return (
        "You are a speech perception assistant. "
        "Follow the instruction exactly and output only the requested phone-level representation or transformation. "
        "Use IPA phones and preserve the provided timestamp format."
    )


def base_meta(sample: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    return {
        "task_name": task_name,
        "original_audio": sample.get("original_audio"),
        "start_time": sample.get("start_time"),
        "end_time": sample.get("end_time"),
    }


def build_main_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": [{"text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"audio": sample["original_audio"]},
                    {
                        "text": (
                            "Recognize the realized phone sequence from this audio. "
                            "Output only the timestamped realized phone sequence in IPA."
                        )
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": sample["huper"]}]},
        ],
        "source": "constructed_multitask_chatml",
        "metadata": base_meta(sample, "main_audio_to_realized"),
    }


def build_aux_canonical_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": [{"text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"audio": sample["original_audio"]},
                    {
                        "text": (
                            f"Transcript:\n{sample['text']}\n\n"
                            "Generate the canonical phone sequence for this transcript. "
                            "Output only the timestamped canonical phone sequence in IPA."
                        )
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": sample["g2p"]}]},
        ],
        "source": "constructed_multitask_chatml",
        "metadata": base_meta(sample, "aux_audio_text_to_canonical"),
    }


def build_aux_edit_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    text_segments = parse_timestamped_segments(sample["text"])
    g2p_segments = parse_timestamped_segments(sample["g2p"])
    huper_segments = parse_timestamped_segments(sample["huper"])

    n = min(len(text_segments), len(g2p_segments), len(huper_segments))
    out_lines = []

    for i in range(n):
        ts_start = text_segments[i]["start"]
        ts_end = text_segments[i]["end"]
        transcript = text_segments[i]["content"]
        canonical = g2p_segments[i]["content"]
        realized = huper_segments[i]["content"]

        _, edit_lines = align_edit_script(canonical.split(), realized.split())

        out_lines.append(f"<|{ts_start:.2f}|>{transcript}<|{ts_end:.2f}|>")
        out_lines.append(f"CANONICAL: {canonical}")
        out_lines.append(f"REALIZED: {realized}")
        out_lines.append("EDITS:")
        if edit_lines:
            out_lines.extend(edit_lines)
        else:
            out_lines.append("NO_CHANGE")
        out_lines.append("")

    assistant_text = "\n".join(out_lines).strip()

    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": [{"text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"audio": sample["original_audio"]},
                    {
                        "text": (
                            f"Transcript:\n{sample['text']}\n\n"
                            "Describe how the canonical phone sequence is transformed into the realized phone sequence. "
                            "For each timestamped segment, output the canonical sequence, the realized sequence, "
                            "and the edit-style transformations."
                        )
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": assistant_text}]},
        ],
        "source": "constructed_multitask_chatml",
        "metadata": base_meta(sample, "aux_audio_text_to_edit"),
    }


def build_aux_realized_from_transcript_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    return {
        "type": "chatml",
        "messages": [
            {"role": "system", "content": [{"text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"audio": sample["original_audio"]},
                    {
                        "text": (
                            f"Transcript:\n{sample['text']}\n\n"
                            "Given the audio and transcript, output the timestamped realized phone sequence in IPA."
                        )
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": sample["huper"]}]},
        ],
        "source": "constructed_multitask_chatml",
        "metadata": base_meta(sample, "aux_audio_text_to_realized"),
    }


TASK_BUILDERS = {
    "main": build_main_task,
    "canonical": build_aux_canonical_task,
    "edit": build_aux_edit_task,
    "realized_from_transcript": build_aux_realized_from_transcript_task,
}


# -----------------------------
# I/O and split helpers
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    idxs = list(range(len(rows)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)

    train_n = int(round(len(rows) * train_ratio))
    train_ids = set(idxs[:train_n])

    train_rows, dev_rows = [], []
    for i, row in enumerate(rows):
        if i in train_ids:
            train_rows.append(row)
        else:
            dev_rows.append(row)
    return train_rows, dev_rows


def normalize_weights(task_weights: Dict[str, float]) -> Dict[str, float]:
    positive = {k: float(v) for k, v in task_weights.items() if float(v) > 0}
    s = sum(positive.values())
    if s <= 0:
        raise ValueError("At least one task weight must be > 0.")
    return {k: v / s for k, v in positive.items()}


def integer_allocation(weights: Dict[str, float], total_slots: int) -> Dict[str, int]:
    if total_slots <= 0:
        return {k: 0 for k in weights.keys()}

    normalized = normalize_weights(weights)
    raw = {k: normalized[k] * total_slots for k in normalized}
    base = {k: int(math.floor(v)) for k, v in raw.items()}
    used = sum(base.values())
    remain = total_slots - used

    remainders = sorted(
        [(raw[k] - base[k], k) for k in normalized],
        key=lambda x: (-x[0], x[1]),
    )

    for i in range(remain):
        _, k = remainders[i % len(remainders)]
        base[k] += 1

    return base


def parse_task_weights(args) -> Dict[str, float]:
    return {
        "main": args.w_main,
        "canonical": args.w_canonical,
        "edit": args.w_edit,
        "realized_from_transcript": args.w_realized_from_transcript,
    }


# -----------------------------
# Parallel builders
# -----------------------------
def build_tasks_for_one_sample(payload: Tuple[Dict[str, Any], List[str], Dict[str, int], str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    sample, enabled_tasks, alloc, system_prompt = payload

    local_tasks = []
    local_counter = {k: 0 for k in TASK_BUILDERS.keys()}

    for task_name in enabled_tasks:
        count = alloc.get(task_name, 0)
        if count <= 0:
            continue
        builder = TASK_BUILDERS[task_name]
        for _ in range(count):
            local_tasks.append(builder(sample, system_prompt))
            local_counter[task_name] += 1

    return local_tasks, local_counter


def merge_counters(counter_list: List[Dict[str, int]]) -> Dict[str, int]:
    out = {k: 0 for k in TASK_BUILDERS.keys()}
    for c in counter_list:
        for k, v in c.items():
            out[k] += v
    return out


def build_chatml_rows_for_split_parallel(
    rows: List[Dict[str, Any]],
    task_weights: Dict[str, float],
    samples_per_audio: int,
    mode: str,
    seed: int,
    num_workers: int,
    chunksize: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    system_prompt = build_system_prompt()
    enabled_tasks = [k for k, v in task_weights.items() if v > 0]
    if len(enabled_tasks) == 0:
        raise ValueError("No enabled tasks.")

    if mode == "sample":
        alloc = integer_allocation({k: task_weights[k] for k in enabled_tasks}, samples_per_audio)
    elif mode == "all":
        alloc = {k: 1 for k in enabled_tasks}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    payloads = [(sample, enabled_tasks, alloc, system_prompt) for sample in rows]

    if num_workers <= 1:
        results = [build_tasks_for_one_sample(p) for p in payloads]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = list(pool.imap(build_tasks_for_one_sample, payloads, chunksize=chunksize))

    all_rows = []
    counters = []
    for local_tasks, local_counter in results:
        all_rows.extend(local_tasks)
        counters.append(local_counter)

    rnd = random.Random(seed)
    rnd.shuffle(all_rows)

    return all_rows, merge_counters(counters)


def save_stats(
    output_dir: Path,
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    train_chatml: List[Dict[str, Any]],
    dev_chatml: List[Dict[str, Any]],
    train_counter: Dict[str, int],
    dev_counter: Dict[str, int],
    config: Dict[str, Any],
) -> None:
    stats = {
        "config": config,
        "num_input_rows": len(train_rows) + len(dev_rows),
        "num_train_rows": len(train_rows),
        "num_dev_rows": len(dev_rows),
        "num_train_chatml": len(train_chatml),
        "num_dev_chatml": len(dev_chatml),
        "train_task_counts": train_counter,
        "dev_task_counts": dev_counter,
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Build train/dev split + multitask ChatML + sampling ratio control from filtered JSONL, with CPU parallelism."
    )
    parser.add_argument("--input", type=str, required=True, help="Input filtered unified-vocab JSONL")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--train_ratio", type=float, default=0.98, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--train_mode", type=str, default="sample", choices=["sample", "all"])
    parser.add_argument("--dev_mode", type=str, default="all", choices=["sample", "all"])

    parser.add_argument("--train_samples_per_audio", type=int, default=4)
    parser.add_argument("--dev_samples_per_audio", type=int, default=3)

    parser.add_argument("--w_main", type=float, default=0.5)
    parser.add_argument("--w_canonical", type=float, default=0.25)
    parser.add_argument("--w_edit", type=float, default=0.25)
    parser.add_argument("--w_realized_from_transcript", type=float, default=0.0)

    parser.add_argument("--save_split_jsonl", action="store_true")

    parser.add_argument("--num_workers", type=int, default=cpu_count(), help="CPU workers for parallel task building")
    parser.add_argument("--chunksize", type=int, default=32, help="Multiprocessing chunksize")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input)
    train_rows, dev_rows = split_rows(rows, train_ratio=args.train_ratio, seed=args.seed)

    task_weights = parse_task_weights(args)

    train_chatml, train_counter = build_chatml_rows_for_split_parallel(
        rows=train_rows,
        task_weights=task_weights,
        samples_per_audio=args.train_samples_per_audio,
        mode=args.train_mode,
        seed=args.seed,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )

    dev_chatml, dev_counter = build_chatml_rows_for_split_parallel(
        rows=dev_rows,
        task_weights=task_weights,
        samples_per_audio=args.dev_samples_per_audio,
        mode=args.dev_mode,
        seed=args.seed + 1,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )

    save_jsonl(train_chatml, str(output_dir / "train.chatml.jsonl"))
    save_jsonl(dev_chatml, str(output_dir / "dev.chatml.jsonl"))

    if args.save_split_jsonl:
        save_jsonl(train_rows, str(output_dir / "train.split.jsonl"))
        save_jsonl(dev_rows, str(output_dir / "dev.split.jsonl"))

    save_stats(
        output_dir=output_dir,
        train_rows=train_rows,
        dev_rows=dev_rows,
        train_chatml=train_chatml,
        dev_chatml=dev_chatml,
        train_counter=train_counter,
        dev_counter=dev_counter,
        config={
            "input": args.input,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "train_mode": args.train_mode,
            "dev_mode": args.dev_mode,
            "train_samples_per_audio": args.train_samples_per_audio,
            "dev_samples_per_audio": args.dev_samples_per_audio,
            "task_weights": task_weights,
            "save_split_jsonl": args.save_split_jsonl,
            "num_workers": args.num_workers,
            "chunksize": args.chunksize,
        },
    )

    print(f"Input rows: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Train ChatML: {len(train_chatml)}")
    print(f"Dev ChatML: {len(dev_chatml)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
