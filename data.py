import json
import argparse
from typing import List, Tuple, Dict, Any


def parse_timestamped_segments(s: str) -> List[Dict[str, Any]]:
    """
    Parse strings like:
    <|5.76|>...<|9.12|><|9.84|>...<|14.24|>
    into:
    [
      {"start": 5.76, "end": 9.12, "content": "..."},
      {"start": 9.84, "end": 14.24, "content": "..."},
      ...
    ]
    """
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
    """
    Align canonical src -> realized tgt using edit distance.
    Return:
      ops: [(op, src_token, tgt_token), ...]
      lines: human-readable edit lines
    op in {"=", "S", "D", "I"}
    """
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
        for j in range(1, m + 1):
            sub_cost = dp[i - 1][j - 1] + (0 if src[i - 1] == tgt[j - 1] else 1)
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1

            best = min(sub_cost, del_cost, ins_cost)
            dp[i][j] = best

            if best == sub_cost:
                bt[i][j] = "=" if src[i - 1] == tgt[j - 1] else "S"
            elif best == del_cost:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"

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


def build_system_prompt() -> str:
    return (
        "You are a speech perception assistant. "
        "Follow the instruction exactly and output only the requested phone-level representation or transformation. "
        "Use IPA phones and preserve the provided timestamp format."
    )


def build_main_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """
    Main task:
      Audio -> realized phone sequence
    Target = HuPER
    """
    return {
        "type": "chatml",
        "messages": [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
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
            {
                "role": "assistant",
                "content": [{"text": sample["huper"]}],
            },
        ],
        "source": "jsonl_conversion_main_task",
    }


def build_aux_canonical_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """
    Auxiliary task:
      Audio + transcript -> canonical phone sequence
    Target = g2p
    """
    return {
        "type": "chatml",
        "messages": [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
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
            {
                "role": "assistant",
                "content": [{"text": sample["g2p"]}],
            },
        ],
        "source": "jsonl_conversion_aux_canonical_task",
    }


def build_aux_edit_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """
    Auxiliary task:
      Audio + transcript -> canonical-to-realized transformation / edit-style supervision
    We align g2p vs huper segment by segment and output edit lines.
    """
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
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
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
            {
                "role": "assistant",
                "content": [{"text": assistant_text}],
            },
        ],
        "source": "jsonl_conversion_aux_edit_task",
    }


def build_aux_realized_from_transcript_task(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """
    Optional auxiliary task:
      Audio + transcript -> realized phone sequence
    Target = HuPER
    """
    return {
        "type": "chatml",
        "messages": [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
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
            {
                "role": "assistant",
                "content": [{"text": sample["huper"]}],
            },
        ],
        "source": "jsonl_conversion_aux_realized_from_transcript_task",
    }


def convert_one_sample(
    sample: Dict[str, Any],
    include_main: bool,
    include_aux_canonical: bool,
    include_aux_edit: bool,
    include_aux_realized_from_transcript: bool,
) -> List[Dict[str, Any]]:
    system_prompt = build_system_prompt()
    tasks = []

    if include_main:
        tasks.append(build_main_task(sample, system_prompt))
    if include_aux_canonical:
        tasks.append(build_aux_canonical_task(sample, system_prompt))
    if include_aux_edit:
        tasks.append(build_aux_edit_task(sample, system_prompt))
    if include_aux_realized_from_transcript:
        tasks.append(build_aux_realized_from_transcript_task(sample, system_prompt))

    return tasks


def convert_file(
    input_jsonl: str,
    output_jsonl: str,
    include_main: bool = True,
    include_aux_canonical: bool = True,
    include_aux_edit: bool = True,
    include_aux_realized_from_transcript: bool = False,
) -> int:
    total_written = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            tasks = convert_one_sample(
                sample=sample,
                include_main=include_main,
                include_aux_canonical=include_aux_canonical,
                include_aux_edit=include_aux_edit,
                include_aux_realized_from_transcript=include_aux_realized_from_transcript,
            )

            for t in tasks:
                fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                total_written += 1

    return total_written


def main():
    parser = argparse.ArgumentParser(description="Convert unified-vocab JSONL into task-specific ChatML JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Input unified-vocab JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output ChatML JSONL")
    parser.add_argument("--no_main", action="store_true", help="Disable main task: audio -> realized phone sequence")
    parser.add_argument(
        "--no_aux_canonical",
        action="store_true",
        help="Disable auxiliary task: audio + transcript -> canonical phone sequence",
    )
    parser.add_argument(
        "--no_aux_edit",
        action="store_true",
        help="Disable auxiliary task: audio + transcript -> canonical-to-realized edit supervision",
    )
    parser.add_argument(
        "--include_aux_realized_from_transcript",
        action="store_true",
        help="Enable optional task: audio + transcript -> realized phone sequence",
    )
    args = parser.parse_args()

    total = convert_file(
        input_jsonl=args.input,
        output_jsonl=args.output,
        include_main=not args.no_main,
        include_aux_canonical=not args.no_aux_canonical,
        include_aux_edit=not args.no_aux_edit,
        include_aux_realized_from_transcript=args.include_aux_realized_from_transcript,
    )

    print(f"Wrote {total} ChatML samples to {args.output}")


if __name__ == "__main__":
    main()
