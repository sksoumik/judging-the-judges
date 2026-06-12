"""S6: Reference-guided strategy. Gold-standard reference answer for anchoring."""

from src.judges.base import JudgeBase
from src.judges.prompts import REFERENCE_GUIDED_PROMPT
from src.strategies.base import StrategyBase, StrategyResult


class ReferenceGuidedStrategy(StrategyBase):
    name = "S6_reference_guided"

    def __init__(self, judge: JudgeBase):
        self.judge = judge

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        if not reference:
            raise ValueError(
                "Reference-guided strategy requires a reference answer. "
                "Pass reference= to evaluate()."
            )
        result = await self.judge.evaluate_pair(
            question, response_a, response_b,
            system_prompt=REFERENCE_GUIDED_PROMPT,
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
