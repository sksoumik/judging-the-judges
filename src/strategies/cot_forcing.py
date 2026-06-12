"""S5: Chain-of-thought forcing. Step-by-step analysis before verdict."""

from src.judges.base import JudgeBase
from src.judges.prompts import COT_FORCING_PROMPT
from src.strategies.base import StrategyBase, StrategyResult


class CoTForcingStrategy(StrategyBase):
    name = "S5_cot_forcing"

    def __init__(self, judge: JudgeBase):
        self.judge = judge

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        result = await self.judge.evaluate_pair(
            question, response_a, response_b,
            system_prompt=COT_FORCING_PROMPT,
            reference=reference,
        )
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
