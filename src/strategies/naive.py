"""B0: Baseline naive strategy. Single judge call, fixed position."""

from src.judges.base import JudgeBase
from src.strategies.base import StrategyBase, StrategyResult


class NaiveStrategy(StrategyBase):
    name = "B0_naive"

    def __init__(self, judge: JudgeBase):
        self.judge = judge

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        result = await self.judge.evaluate_pair(question, response_a, response_b)
        return StrategyResult(
            strategy_name=self.name,
            verdict=result.verdict,
            score_a=result.score_a,
            score_b=result.score_b,
            individual_results=[result],
            total_cost_usd=result.cost_usd,
            total_latency_ms=result.latency_ms,
            total_input_tokens=result.input_tokens,
            total_output_tokens=result.output_tokens,
        )
