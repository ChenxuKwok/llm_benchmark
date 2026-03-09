import os
from typing import Optional

PROMPT_VERSION = "capt-data-prompts-v1"


def _read_prompt(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(
    mode: str, reference_text: Optional[str], prompts_dir: str = "data/prompts"
) -> str:
    if mode not in {"reference_free", "reference_given"}:
        raise ValueError("mode must be reference_free or reference_given")

    system_path = os.path.join(prompts_dir, "system.txt")
    task_path = os.path.join(
        prompts_dir, "task_1.txt" if mode == "reference_free" else "task_2.txt"
    )
    system_prompt = _read_prompt(system_path)
    task_prompt = _read_prompt(task_path)
    if not system_prompt:
        raise FileNotFoundError(f"Missing prompt file: {system_path}")
    if not task_prompt:
        raise FileNotFoundError(f"Missing prompt file: {task_path}")

    if mode == "reference_given":
        if not reference_text:
            raise ValueError("reference_text is required for reference_given mode")
        task_prompt = task_prompt.replace(
            "[Insert Ground Truth Text Here]", reference_text
        )

    # Enforce English target language for this benchmark.
    task_prompt = task_prompt.replace("[English / Mandarin Chinese]", "English")

    return "\n\n".join([system_prompt, task_prompt]).strip()
