PROMPT_VERSION = "capt-v1"


def build_prompt(mode: str, reference_text: str | None) -> str:
    if mode not in {"reference_free", "reference_given"}:
        raise ValueError("mode must be reference_free or reference_given")

    schema = """
{
  "reference": "inferred intended text",
  "errors": [
    {
      "error_type": "substitution | deletion | insertion",
      "word_location": 0,
      "phoneme_expected": "ARPAbet",
      "phoneme_actual": "ARPAbet or null",
      "confidence": 0.0
    }
  ],
  "explanation": [
    {
      "location": 0,
      "content": "brief phonetic explanation"
    }
  ],
  "suggestion": [
    {
      "location": 0,
      "content": "actionable corrective advice"
    }
  ]
}
""".strip()

    header = (
        "You are a CAPT pronunciation analyst. "
        "Return ONLY valid JSON with no markdown fences or extra text. "
        "Use the CMU ARPAbet 39 phonemes for English. "
        "Do NOT mark natural connected-speech variation as errors unless intelligibility is clearly affected."
    )
    if mode == "reference_given":
        if not reference_text:
            raise ValueError("reference_text is required for reference_given mode")
        task = (
            "Reference-given mode: the reference field MUST equal the provided reference text exactly."
        )
        reference_block = f"Reference text: {reference_text}"
    else:
        task = "Reference-free mode: infer the intended reference text from the audio."
        reference_block = ""

    instructions = "\n".join(
        [
            header,
            task,
            reference_block,
            "Output schema:",
            schema,
        ]
    ).strip()
    return instructions
