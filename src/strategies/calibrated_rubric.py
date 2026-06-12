"""S4: Calibrated rubric strategy. Detailed scoring rubric with 5 criteria."""

from src.judges.base import JudgeBase
from src.judges.prompts import CALIBRATED_RUBRIC_PROMPT
from src.strategies.base import StrategyBase, StrategyResult


class CalibratedRubricStrategy(StrategyBase):
    name = "S4_calibrated_rubric"

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
            system_prompt=CALIBRATED_RUBRIC_PROMPT,
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
