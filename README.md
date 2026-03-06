# Speech Pronunciation Benchmark (CAPT, L2-Arctic-plus)

Minimal, end-to-end benchmark pipeline for evaluating audio-capable LLMs on CAPT-style pronunciation analysis using L2-Arctic-plus. Supports two backends: OpenAI and Vertex AI Gemini.

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure dataset exists at one of these roots:
   - `data/L2-Arctic-plus/...`
   - `ALMs4Learning/data/L2-Arctic-plus/...`
3. Run a smoke test:
   - OpenAI:
     - `python scripts/run_smoke.py --dataset l2arctic_plus --backend openai --model chatgpt-4o-latest --fallback-model gpt-audio-1.5 --mode reference_free --limit 20`
   - Gemini:
     - `python scripts/run_smoke.py --dataset l2arctic_plus --backend gemini --model gemini-3.1-pro-preview --mode reference_free --limit 20 --project <gcp-project> --location us-central1`

## Full Evaluation

- `python scripts/run_eval.py --dataset l2arctic_plus --backend openai --model chatgpt-4o-latest --mode reference_given`

## Results

Outputs are saved under `results/<run_name>/`:

- `config.json` (resolved run config)
- `schema_report.json` (dataset key inspection + label shape hints)
- `raw/` (raw model outputs)
- `parsed/` (validated JSON outputs)
- `meta/` (per-sample metadata and status)
- `metrics.json` (machine-readable metrics)
- `summary.md` (human-readable report)

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

## Notes and Limitations

- Word-level metrics are derived from `mis_exp_sug` when explicit labels are absent.
- Phoneme-level tuples are parsed from issue strings; ambiguous issues (e.g., “unclear pronunciation”) will not yield tuples.
- OpenAI audio support varies by model; a fallback model can be configured for automatic retry.

## Repo Layout

- `src/benchmark/` — core pipeline
- `scripts/` — CLI entrypoints
- `configs/` — backend configs
- `results/` — outputs
