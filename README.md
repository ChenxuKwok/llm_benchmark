# Speech Pronunciation Benchmark (CAPT, L2-Arctic-plus)

Minimal, end-to-end benchmark pipeline for evaluating audio-capable LLMs on CAPT-style pronunciation analysis using L2-Arctic-plus. Supports two backends: OpenAI and Vertex AI Gemini.

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure dataset exists at one of these roots:
   - `data/L2-Arctic-plus/...`
   - `ALMs4Learning/data/L2-Arctic-plus/...`
3. Ensure prompt templates exist at `data/prompts/`:
   - `system.txt`
   - `task_1.txt` (reference-free)
   - `task_2.txt` (reference-given)
4. Run a smoke test:
   - OpenAI:
     - `python scripts/run_smoke.py --dataset l2arctic_plus --backend openai --model chatgpt-4o-latest --mode reference_free --limit 2`
   - Gemini:
     - `python scripts/run_smoke.py --dataset l2arctic_plus --backend gemini --model gemini-3.1-pro-preview --mode reference_free --limit 20`

## Full Evaluation

- `python scripts/run_eval.py --dataset l2arctic_plus --backend openai --model chatgpt-4o-latest --mode reference_given`
- By default, full eval randomly selects 200 samples (`--limit 200 --sample-seed 42`).

## Results

Outputs are saved under `results/<run_name>/`:

- `config.json` (resolved run config)
- `schema_report.json` (dataset key inspection + label shape hints)
- `raw/` (raw model outputs)
- `parsed/` (validated JSON outputs)
- `meta/` (per-sample metadata and status)
- `metrics.json` (machine-readable metrics)
- `summary.md` (human-readable report)
- `records.jsonl` (per-sample input/output records)

Runs are resumable: completed samples are skipped unless `--force` is set.

## Dataset Expectations (L2-Arctic-plus)

Expected JSON keys in `train_data.json` and `test_data.json`:

- `audio_path` (relative to dataset root)
- `text` (reference text)
- `annotation_info` (phoneme sequences per word)
- `mis_exp_sug` (word-level issues + suggestions)
- `regeneration` (boolean flag)

The loader will:

- Resolve `audio_path` entries like `./data/L2-Arctic-plus/...`
- Infer word-level error locations from `mis_exp_sug`
- Parse phoneme errors from issue strings (substitution / insertion / deletion)

If the dataset is elsewhere, set `--dataset-root` to its parent directory.

## Configuration

Default configs live in `configs/`:

- `configs/openai.yaml`
- `configs/gemini.yaml`

CLI args override config values.

## Auth and Run Sizing

DashScope compatible-mode (used for both OpenAI and Gemini models):

- Set the API key in your shell: `export DASHSCOPE_API_KEY="sk-..."`
- Both backends call the same endpoint and only differ by `model` name.
- If you prefer, you can set `OPENAI_API_KEY` instead (the runner will use it as a fallback).
- Default endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- Audio is sent as base64 with a `data:<mime>;base64,` prefix, using `input_audio`.
- For evaluation, default output is text-only JSON (audio output is disabled unless you explicitly set `--modalities text,audio`).
- `response_format` is OFF by default. Enable it with `--use-response-format` only for models that explicitly support it.
- Request style is explicit by backend: `openai -> messages`, `gemini -> contents.parts`.

Request count and concurrency:

- `--limit N` controls how many samples (requests) to send.
- `--workers N` controls concurrency.
- `--max-retries N` controls retry attempts per request.
- `--fallback-model null` and `--fallback-model none` are treated as no fallback.

Gemini troubleshooting:

- If GPT works but Gemini returns `api_error`, first run with `configs/gemini.yaml` defaults (no `audio_modalities` and no `audio_voice`).
- `gemini` backend is sent as Gemini-native `contents.parts` payload.

## Notes and Limitations

- Word-level metrics are derived from `mis_exp_sug` when explicit labels are absent.
- Phoneme-level tuples are parsed from issue strings; ambiguous issues (e.g., “unclear pronunciation”) will not yield tuples.
- OpenAI audio support varies by model; a fallback model can be configured for automatic retry.

## Repo Layout

- `src/benchmark/` — core pipeline
- `scripts/` — CLI entrypoints
- `configs/` — backend configs
- `results/` — outputs
