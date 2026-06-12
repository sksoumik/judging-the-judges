from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from src.judges.base import JudgeResult


@dataclass
class StrategyResult:
    strategy_name: str
    verdict: Literal["A", "B", "tie"]
    score_a: float
    score_b: float
    individual_results: list[JudgeResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def num_calls(self) -> int:
        return len(self.individual_results)


class StrategyBase(ABC):
    """Abstract base class for debiasing strategies."""

    name: str = "base"

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        ...

    def _aggregate_results(
        self, results: list[JudgeResult], strategy_name: str
    ) -> StrategyResult:
        """Aggregate multiple JudgeResults into a StrategyResult via majority vote."""
        votes = {"A": 0, "B": 0, "tie": 0}
        for r in results:
            votes[r.verdict] += 1

        verdict = max(votes, key=votes.get)  # type: ignore[arg-type]
        avg_score_a = sum(r.score_a for r in results) / len(results)
        avg_score_b = sum(r.score_b for r in results) / len(results)

        return StrategyResult(
            strategy_name=strategy_name,
            verdict=verdict,
            score_a=avg_score_a,
            score_b=avg_score_b,
            individual_results=results,
            total_cost_usd=sum(r.cost_usd for r in results),
            total_latency_ms=sum(r.latency_ms for r in results),
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
        )
