#!/usr/bin/env python3
"""Run a single (model, strategy, benchmark) configuration for quick testing."""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.data_loader import DataLoader
from src.judges.factory import JudgeFactory
from src.strategies.factory import StrategyFactory

console = Console()


async def run_single(
    model: str,
    strategy: str,
    benchmark: str,
    sample_size: int = 5,
    project_id: str | None = None,
):
    project_id = project_id or os.environ.get("GOOGLE_PROJECT_ID", "")

    judge_factory = JudgeFactory(project_id=project_id)
    strategy_factory = StrategyFactory(judge_factory)
    data_loader = DataLoader()

    console.print(f"\n[bold]Running: {model} / {strategy} / {benchmark}[/bold]")
    console.print(f"Sample size: {sample_size}\n")

    instances = data_loader.load(benchmark, sample_size=sample_size)
    strategy_obj = strategy_factory.create(strategy, model)

    table = Table(title="Results")
    table.add_column("Instance", style="cyan")
    table.add_column("Verdict", style="bold")
    table.add_column("Score A", justify="right")
    table.add_column("Score B", justify="right")
    table.add_column("Gold", style="yellow")
    table.add_column("Cost", justify="right")
    table.add_column("Latency", justify="right")

    total_cost = 0.0
    for inst in instances:
        try:
            result = await strategy_obj.evaluate(
                question=inst.question,
                response_a=inst.response_a,
                response_b=inst.response_b,
                reference=inst.reference,
            )
            total_cost += result.total_cost_usd
            table.add_row(
                inst.id[:30],
                result.verdict,
                f"{result.score_a:.1f}",
                f"{result.score_b:.1f}",
                inst.human_preference or "?",
                f"${result.total_cost_usd:.4f}",
                f"{result.total_latency_ms:.0f}ms",
            )
        except Exception as e:
            table.add_row(inst.id[:30], "ERROR", "", "", "", "", str(e)[:40])

    console.print(table)
    console.print(f"\n[bold]Total cost: ${total_cost:.4f}[/bold]")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a single experiment configuration")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of instances")
    parser.add_argument("--project-id", type=str, default=None)

    args = parser.parse_args()
    asyncio.run(run_single(args.model, args.strategy, args.benchmark, args.sample_size, args.project_id))


if __name__ == "__main__":
    main()
