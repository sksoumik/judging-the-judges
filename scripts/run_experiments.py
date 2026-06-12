#!/usr/bin/env python3
"""CLI to run experiments from the experiment matrix."""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from src.runner import ExperimentRunner


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run LLM Judge Bias experiments")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names (default: all from config)"
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated strategy names (default: all from config)"
    )
    parser.add_argument(
        "--benchmarks", type=str, default=None,
        help="Comma-separated benchmark names (default: all from config)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Override sample size for all benchmarks"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Max concurrent API calls (default: 10)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate cost without running"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiments.yaml",
        help="Path to experiments config"
    )
    parser.add_argument(
        "--project-id", type=str, default=None,
        help="GCP project ID (default: from GOOGLE_PROJECT_ID env var)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use Vertex AI Batch Prediction API (Gemini models, single-call strategies only). "
             "Much faster and cheaper for large runs."
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between batch job polls (default: 30)"
    )

    args = parser.parse_args()

    project_id = args.project_id or os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id and not args.dry_run:
        parser.error("--project-id or GOOGLE_PROJECT_ID env var required")

    models = args.models.split(",") if args.models else None
    strategies = args.strategies.split(",") if args.strategies else None
    benchmarks = args.benchmarks.split(",") if args.benchmarks else None

    runner = ExperimentRunner(
        config_path=args.config,
        project_id=project_id,
        max_concurrent=args.max_concurrent,
    )

    if args.batch:
        results = runner.run_batch(
            models=models,
            strategies=strategies,
            benchmarks=benchmarks,
            sample_size=args.sample_size,
            poll_interval=args.poll_interval,
        )
    else:
        results = asyncio.run(
            runner.run(
                models=models,
                strategies=strategies,
                benchmarks=benchmarks,
                sample_size=args.sample_size,
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run:
        print(f"\nCompleted {len(results)} configurations")
        total_cost = sum(
            sum(r.total_cost_usd for r in res)
            for res in results.values()
            if isinstance(res, list)
        )
        print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
