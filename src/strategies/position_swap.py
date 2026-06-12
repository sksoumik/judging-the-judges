"""S1: Position swap strategy. Run twice with swapped positions."""

import asyncio

from src.judges.base import JudgeBase
from src.strategies.base import StrategyBase, StrategyResult


class PositionSwapStrategy(StrategyBase):
    name = "S1_position_swap"

    def __init__(self, judge: JudgeBase, system_prompt: str | None = None):
        self.judge = judge
        self.system_prompt = system_prompt

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        # Run both orderings concurrently
        result_ab, result_ba = await asyncio.gather(
            self.judge.evaluate_pair(
                question, response_a, response_b,
                system_prompt=self.system_prompt, reference=reference,
            ),
            self.judge.evaluate_pair(
                question, response_b, response_a,
                system_prompt=self.system_prompt, reference=reference,
            ),
        )

        # Flip the BA result back to AB frame
        result_ba_flipped_verdict = (
            "B" if result_ba.verdict == "A"
            else "A" if result_ba.verdict == "B"
            else "tie"
        )

        # Check consistency
        if result_ab.verdict == result_ba_flipped_verdict:
            verdict = result_ab.verdict
        else:
            verdict = "tie"

        avg_score_a = (result_ab.score_a + result_ba.score_b) / 2
        avg_score_b = (result_ab.score_b + result_ba.score_a) / 2

        return StrategyResult(
            strategy_name=self.name,
            verdict=verdict,
            score_a=avg_score_a,
            score_b=avg_score_b,
            individual_results=[result_ab, result_ba],
            total_cost_usd=result_ab.cost_usd + result_ba.cost_usd,
            total_latency_ms=result_ab.latency_ms + result_ba.latency_ms,
            total_input_tokens=result_ab.input_tokens + result_ba.input_tokens,
            total_output_tokens=result_ab.output_tokens + result_ba.output_tokens,
            metadata={
                "consistent": result_ab.verdict == result_ba_flipped_verdict,
                "ab_verdict": result_ab.verdict,
                "ba_verdict_flipped": result_ba_flipped_verdict,
            },
        )
