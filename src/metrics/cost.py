"""Cost and latency tracking."""

from dataclasses import dataclass

from src.strategies.base import StrategyResult


@dataclass
class CostSummary:
    total_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    avg_latency_ms: float
    avg_cost_per_call: float


def aggregate_cost(results: list[StrategyResult]) -> CostSummary:
    """Aggregate cost and latency across multiple strategy results."""
    if not results:
        return CostSummary(0.0, 0, 0, 0, 0.0, 0.0)

    total_usd = sum(r.total_cost_usd for r in results)
    total_input = sum(r.total_input_tokens for r in results)
    total_output = sum(r.total_output_tokens for r in results)
    total_calls = sum(r.num_calls for r in results)
    total_latency = sum(r.total_latency_ms for r in results)

    return CostSummary(
        total_usd=total_usd,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_calls=total_calls,
        avg_latency_ms=total_latency / len(results) if results else 0.0,
        avg_cost_per_call=total_usd / total_calls if total_calls else 0.0,
    )
