import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.run import run_benchmark
from benchmark.utils.config import load_config, merge_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full CAPT benchmark evaluation.")
    parser.add_argument("--dataset", default="l2arctic_plus")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--backend", required=True, choices=["openai", "gemini"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--mode", required=True, choices=["reference_free", "reference_given"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.dataset != "l2arctic_plus":
        raise ValueError("Only l2arctic_plus is supported")
    config_path = args.config
    if config_path is None:
        config_path = os.path.join("configs", f"{args.backend}.yaml")
    config = load_config(config_path) if os.path.exists(config_path) else {}
    overrides = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "backend": args.backend,
        "model": args.model,
        "fallback_model": args.fallback_model,
        "mode": args.mode,
        "run_name": args.run_name,
        "workers": args.workers,
        "force": args.force,
        "project": args.project,
        "location": args.location,
        "max_retries": args.max_retries,
    }
    cfg = merge_config(config, overrides)
    run_benchmark(
        dataset_root=cfg.get("dataset_root", "data"),
        split=cfg.get("split", "test"),
        backend=cfg.get("backend"),
        model=cfg.get("model"),
        mode=cfg.get("mode"),
        run_name=cfg.get("run_name"),
        fallback_model=cfg.get("fallback_model"),
        limit=None,
        workers=cfg.get("workers", 1),
        force=cfg.get("force", False),
        project=cfg.get("project"),
        location=cfg.get("location"),
        max_retries=cfg.get("max_retries", 3),
    )


if __name__ == "__main__":
    main()
