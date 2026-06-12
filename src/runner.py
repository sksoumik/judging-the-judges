"""Experiment runner: orchestrates evaluations across the experiment matrix."""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.data_loader import DataLoader, EvalInstance
from src.judges.batch import GeminiBatchProcessor
from src.judges.factory import JudgeFactory
from src.strategies.base import StrategyResult
from src.strategies.factory import StrategyFactory

console = Console()


class ExperimentRunner:
    """Runs experiments across model x strategy x benchmark matrix."""

    def __init__(
        self,
        config_path: str = "configs/experiments.yaml",
        models_config: str = "configs/models.yaml",
        data_dir: str = "data",
        results_dir: str = "results",
        project_id: str | None = None,
        max_concurrent: int = 10,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.judge_factory = JudgeFactory(models_config, project_id)
        self.strategy_factory = StrategyFactory(self.judge_factory)
        self.data_loader = DataLoader(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "raw").mkdir(exist_ok=True)
        (self.results_dir / "aggregated").mkdir(exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _get_cache_path(
        self, model: str, strategy: str, benchmark: str, instance_id: str
    ) -> Path:
        return self.results_dir / "raw" / f"{model}_{strategy}_{benchmark}_{instance_id}.json"

    def _is_cached(
        self, model: str, strategy: str, benchmark: str, instance_id: str
    ) -> bool:
        return self._get_cache_path(model, strategy, benchmark, instance_id).exists()

    def _save_result(
        self,
        model: str,
        strategy: str,
        benchmark: str,
        instance_id: str,
        result: StrategyResult,
    ) -> None:
        path = self._get_cache_path(model, strategy, benchmark, instance_id)
        data = {
            "model": model,
            "strategy": strategy,
            "benchmark": benchmark,
            "instance_id": instance_id,
            "result": _strategy_result_to_dict(result),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cached_result(
        self, model: str, strategy: str, benchmark: str, instance_id: str
    ) -> StrategyResult | None:
        path = self._get_cache_path(model, strategy, benchmark, instance_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        r = data["result"]
        return StrategyResult(
            strategy_name=r["strategy_name"],
            verdict=r["verdict"],
            score_a=r["score_a"],
            score_b=r["score_b"],
            total_cost_usd=r.get("total_cost_usd", 0),
            total_latency_ms=r.get("total_latency_ms", 0),
            total_input_tokens=r.get("total_input_tokens", 0),
            total_output_tokens=r.get("total_output_tokens", 0),
            metadata=r.get("metadata", {}),
        )

    async def _evaluate_single(
        self,
        strategy_obj,
        instance: EvalInstance,
        model: str,
        strategy: str,
        benchmark: str,
    ) -> StrategyResult:
        """Evaluate a single instance with concurrency control."""
        # Check cache
        cached = self._load_cached_result(model, strategy, benchmark, instance.id)
        if cached:
            return cached

        async with self.semaphore:
            result = await strategy_obj.evaluate(
                question=instance.question,
                response_a=instance.response_a,
                response_b=instance.response_b,
                reference=instance.reference,
            )
            self._save_result(model, strategy, benchmark, instance.id, result)
            return result

    async def run(
        self,
        models: list[str] | None = None,
        strategies: list[str] | None = None,
        benchmarks: list[str] | None = None,
        sample_size: int | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Run the experiment matrix.

        Args:
            models: Models to test (default: all from config)
            strategies: Strategies to test (default: all from config)
            benchmarks: Benchmarks to run (default: all from config)
            sample_size: Override sample size for all benchmarks
            dry_run: If True, only estimate cost without running

        Returns:
            Dict of aggregated results keyed by (model, strategy, benchmark)
        """
        matrix = self.config["experiment_matrix"]
        model_list = models or matrix["models"]
        strategy_list = strategies or matrix["strategies"]
        benchmark_list = benchmarks or matrix["benchmarks"]
        sampling = self.config.get("sampling", {})

        if dry_run:
            return self._estimate_cost(model_list, strategy_list, benchmark_list, sampling)

        all_results = {}

        total_configs = len(model_list) * len(strategy_list) * len(benchmark_list)
        console.print(f"\n[bold]Running {total_configs} configurations[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("Overall", total=total_configs)

            for benchmark in benchmark_list:
                bench_sample = sample_size or sampling.get(benchmark)
                try:
                    instances = self.data_loader.load(benchmark, bench_sample)
                except FileNotFoundError as e:
                    console.print(f"[yellow]Skipping {benchmark}: {e}[/yellow]")
                    progress.advance(main_task, len(model_list) * len(strategy_list))
                    continue

                for model in model_list:
                    for strategy in strategy_list:
                        desc = f"{model} / {strategy} / {benchmark}"
                        progress.update(main_task, description=desc)

                        try:
                            strategy_obj = self.strategy_factory.create(strategy, model)
                        except (ValueError, Exception) as e:
                            console.print(f"[red]Skip {desc}: {e}[/red]")
                            progress.advance(main_task)
                            continue

                        tasks = [
                            self._evaluate_single(
                                strategy_obj, inst, model, strategy, benchmark
                            )
                            for inst in instances
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Filter out exceptions
                        valid_results = [r for r in results if isinstance(r, StrategyResult)]
                        errors = [r for r in results if isinstance(r, Exception)]
                        if errors:
                            console.print(
                                f"[yellow]{len(errors)} errors in {desc}[/yellow]"
                            )

                        key = f"{model}__{strategy}__{benchmark}"
                        all_results[key] = valid_results

                        # Save aggregated
                        self._save_aggregated(model, strategy, benchmark, valid_results, instances)

                        total_cost = sum(r.total_cost_usd for r in valid_results)
                        console.print(
                            f"  [green]Done[/green] {desc}: "
                            f"{len(valid_results)}/{len(instances)} instances, "
                            f"${total_cost:.4f}"
                        )
                        progress.advance(main_task)

        return all_results

    def _save_aggregated(
        self,
        model: str,
        strategy: str,
        benchmark: str,
        results: list[StrategyResult],
        instances: list[EvalInstance],
    ) -> None:
        """Save aggregated metrics for a configuration."""
        if not results:
            return

        verdicts = [r.verdict for r in results]
        gold_labels = [inst.human_preference for inst in instances[:len(results)]]

        # Agreement (only where gold labels exist)
        pairs = [(v, g) for v, g in zip(verdicts, gold_labels) if g is not None]
        agreement = sum(v == g for v, g in pairs) / len(pairs) if pairs else None

        agg = {
            "model": model,
            "strategy": strategy,
            "benchmark": benchmark,
            "n_instances": len(results),
            "verdict_distribution": {
                "A": verdicts.count("A"),
                "B": verdicts.count("B"),
                "tie": verdicts.count("tie"),
            },
            "agreement_rate": agreement,
            "total_cost_usd": sum(r.total_cost_usd for r in results),
            "avg_latency_ms": sum(r.total_latency_ms for r in results) / len(results),
            "total_input_tokens": sum(r.total_input_tokens for r in results),
            "total_output_tokens": sum(r.total_output_tokens for r in results),
        }

        path = self.results_dir / "aggregated" / f"{model}_{strategy}_{benchmark}_metrics.json"
        with open(path, "w") as f:
            json.dump(agg, f, indent=2)

    def _estimate_cost(
        self,
        models: list[str],
        strategies: list[str],
        benchmarks: list[str],
        sampling: dict,
    ) -> dict:
        """Estimate total cost without running."""
        # Load model configs for cost info
        with open("configs/models.yaml") as f:
            model_configs = yaml.safe_load(f)["models"]

        with open("configs/strategies.yaml") as f:
            strategy_configs = yaml.safe_load(f)["strategies"]

        console.print("\n[bold]Cost Estimate (Dry Run)[/bold]\n")

        total_cost = 0.0
        total_calls = 0
        avg_tokens_per_call = 2000  # rough estimate

        for benchmark in benchmarks:
            bench_size = sampling.get(benchmark) or {"mt_bench": 80, "alpaca_eval": 200, "llmbar": 200, "faireval": 200, "custom": 200}.get(benchmark, 100)
            for model in models:
                mcfg = model_configs.get(model, {})
                cost_in = mcfg.get("cost_per_1k_input_tokens", 0)
                cost_out = mcfg.get("cost_per_1k_output_tokens", 0)

                for strategy in strategies:
                    scfg = strategy_configs.get(strategy, {})
                    multiplier = scfg.get("cost_multiplier", 1)
                    calls = bench_size * multiplier
                    cost = calls * (avg_tokens_per_call / 1000 * cost_in + 500 / 1000 * cost_out)
                    total_cost += cost
                    total_calls += calls

        console.print(f"  Total API calls: {total_calls:,}")
        console.print(f"  Estimated cost: ${total_cost:.2f}")
        console.print(f"  Models: {', '.join(models)}")
        console.print(f"  Strategies: {', '.join(strategies)}")
        console.print(f"  Benchmarks: {', '.join(benchmarks)}")

        return {"estimated_cost_usd": total_cost, "estimated_calls": total_calls}

    def run_batch(
        self,
        models: list[str] | None = None,
        strategies: list[str] | None = None,
        benchmarks: list[str] | None = None,
        sample_size: int | None = None,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> dict:
        """Run experiments using Vertex AI Batch Prediction API (Gemini models only).

        Submits all requests for each (model, strategy, benchmark) as a single batch
        job, then polls for completion. Much more efficient for large runs.

        Note: Only works with Gemini models and B0_naive/S4/S5/S6 strategies
        (single-call strategies). Multi-call strategies (position swap, ensembles)
        are not supported in batch mode.
        """
        import yaml as _yaml

        matrix = self.config["experiment_matrix"]
        model_list = models or matrix["models"]
        strategy_list = strategies or matrix["strategies"]
        benchmark_list = benchmarks or matrix["benchmarks"]
        sampling = self.config.get("sampling", {})

        with open("configs/models.yaml") as f:
            model_configs = _yaml.safe_load(f)["models"]

        # Strategies that use a single judge call (batch-compatible)
        BATCH_STRATEGIES = {"B0_naive", "S4_calibrated_rubric", "S5_cot_forcing", "S6_reference_guided"}
        PROMPT_MAP = {}
        from src.judges.prompts import (
            NAIVE_JUDGE_PROMPT,
            CALIBRATED_RUBRIC_PROMPT,
            COT_FORCING_PROMPT,
            REFERENCE_GUIDED_PROMPT,
        )
        PROMPT_MAP = {
            "B0_naive": NAIVE_JUDGE_PROMPT,
            "S4_calibrated_rubric": CALIBRATED_RUBRIC_PROMPT,
            "S5_cot_forcing": COT_FORCING_PROMPT,
            "S6_reference_guided": REFERENCE_GUIDED_PROMPT,
        }

        all_results = {}
        batch_jobs = []

        for benchmark in benchmark_list:
            bench_sample = sample_size or sampling.get(benchmark)
            try:
                instances = self.data_loader.load(benchmark, bench_sample)
            except FileNotFoundError as e:
                console.print(f"[yellow]Skipping {benchmark}: {e}[/yellow]")
                continue

            for model in model_list:
                mcfg = model_configs.get(model, {})
                if mcfg.get("provider") != "gemini":
                    console.print(
                        f"[yellow]Skipping {model} in batch mode "
                        f"(only Gemini models supported)[/yellow]"
                    )
                    continue

                for strategy in strategy_list:
                    if strategy not in BATCH_STRATEGIES:
                        console.print(
                            f"[yellow]Skipping {strategy} in batch mode "
                            f"(multi-call strategies not supported)[/yellow]"
                        )
                        continue

                    # Filter out already-cached instances
                    uncached = [
                        inst for inst in instances
                        if not self._is_cached(model, strategy, benchmark, inst.id)
                    ]

                    if not uncached:
                        console.print(
                            f"  [dim]All cached: {model} / {strategy} / {benchmark}[/dim]"
                        )
                        # Load from cache
                        cached_results = [
                            self._load_cached_result(model, strategy, benchmark, inst.id)
                            for inst in instances
                        ]
                        key = f"{model}__{strategy}__{benchmark}"
                        all_results[key] = [r for r in cached_results if r is not None]
                        continue

                    # Build batch requests
                    requests = [
                        {
                            "id": inst.id,
                            "question": inst.question,
                            "response_a": inst.response_a,
                            "response_b": inst.response_b,
                            "reference": inst.reference,
                        }
                        for inst in uncached
                    ]

                    processor = GeminiBatchProcessor(
                        project_id=self.judge_factory.project_id,
                        model_id=mcfg["model_id"],
                        location=mcfg.get("location", "global"),
                        temperature=mcfg.get("temperature", 0.1),
                        max_output_tokens=mcfg.get("max_output_tokens", 8192),
                        cost_per_1k_input=mcfg.get("cost_per_1k_input_tokens", 0.0),
                        cost_per_1k_output=mcfg.get("cost_per_1k_output_tokens", 0.0),
                    )

                    display_name = f"{model}_{strategy}_{benchmark}"
                    batch_name = processor.submit_batch(
                        requests,
                        system_prompt=PROMPT_MAP.get(strategy),
                        display_name=display_name,
                    )
                    batch_jobs.append({
                        "batch_name": batch_name,
                        "processor": processor,
                        "model": model,
                        "strategy": strategy,
                        "benchmark": benchmark,
                        "instances": instances,
                        "uncached": uncached,
                    })

        if not batch_jobs:
            console.print("[yellow]No batch jobs to submit.[/yellow]")
            return all_results

        console.print(f"\n[bold]Submitted {len(batch_jobs)} batch jobs. Polling...[/bold]")

        # Poll all jobs
        for job_info in batch_jobs:
            try:
                raw_results = job_info["processor"].poll_batch(
                    job_info["batch_name"],
                    poll_interval=poll_interval,
                    timeout=timeout,
                )
                # Process and cache results
                model = job_info["model"]
                strategy = job_info["strategy"]
                benchmark = job_info["benchmark"]

                for raw in raw_results:
                    req_id = raw.get("id", "")
                    result = StrategyResult(
                        strategy_name=strategy,
                        verdict=raw.get("verdict", "tie"),
                        score_a=raw.get("score_a", 5.0),
                        score_b=raw.get("score_b", 5.0),
                    )
                    self._save_result(model, strategy, benchmark, req_id, result)

                key = f"{model}__{strategy}__{benchmark}"
                # Reload all (cached + new)
                all_cached = [
                    self._load_cached_result(model, strategy, benchmark, inst.id)
                    for inst in job_info["instances"]
                ]
                all_results[key] = [r for r in all_cached if r is not None]
                self._save_aggregated(
                    model, strategy, benchmark,
                    all_results[key], job_info["instances"],
                )
                console.print(
                    f"  [green]Done[/green] {model} / {strategy} / {benchmark}: "
                    f"{len(all_results[key])} results"
                )

            except (RuntimeError, TimeoutError) as e:
                console.print(f"[red]Batch failed: {e}[/red]")

        return all_results


def _strategy_result_to_dict(result: StrategyResult) -> dict:
    """Convert StrategyResult to serializable dict (excluding individual_results raw responses)."""
    return {
        "strategy_name": result.strategy_name,
        "verdict": result.verdict,
        "score_a": result.score_a,
        "score_b": result.score_b,
        "total_cost_usd": result.total_cost_usd,
        "total_latency_ms": result.total_latency_ms,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "num_calls": result.num_calls,
        "metadata": result.metadata,
        "individual_verdicts": [r.verdict for r in result.individual_results],
        "individual_models": [r.model_id for r in result.individual_results],
    }
