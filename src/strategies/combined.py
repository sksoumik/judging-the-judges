"""S7/S8: Combined strategies that chain multiple strategies together.

S7 = Position Swap + Cross-Family Ensemble + Calibrated Rubric (6x cost)
S8 = Position Swap + CoT Forcing + Calibrated Rubric (2x cost, budget-friendly)
"""

import asyncio
from typing import Literal

from src.judges.base import JudgeBase
from src.judges.prompts import CALIBRATED_RUBRIC_PROMPT, COT_FORCING_PROMPT
from src.strategies.base import StrategyBase, StrategyResult
from src.strategies.position_swap import PositionSwapStrategy


class CombinedFullStrategy(StrategyBase):
    """S7: Position Swap + Cross-Family Ensemble + Calibrated Rubric."""

    name = "S7_combined_full"

    def __init__(self, judges: list[JudgeBase]):
        if len(judges) < 2:
            raise ValueError("S7 requires at least 2 judges for cross-family ensemble")
        self.judges = judges

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        # Run position-swapped calibrated rubric evaluation for each judge
        swap_strategies = [
            PositionSwapStrategy(judge, system_prompt=CALIBRATED_RUBRIC_PROMPT)
            for judge in self.judges
        ]
        tasks = [
            strategy.evaluate(question, response_a, response_b, reference)
            for strategy in swap_strategies
        ]
        sub_results = await asyncio.gather(*tasks)

        # Majority vote across all sub-strategy verdicts
        votes: dict[Literal["A", "B", "tie"], int] = {"A": 0, "B": 0, "tie": 0}
        all_individual = []
        for sr in sub_results:
            votes[sr.verdict] += 1
            all_individual.extend(sr.individual_results)

        verdict = max(votes, key=votes.get)  # type: ignore[arg-type]
        avg_a = sum(r.score_a for r in all_individual) / len(all_individual)
        avg_b = sum(r.score_b for r in all_individual) / len(all_individual)

        return StrategyResult(
            strategy_name=self.name,
            verdict=verdict,
            score_a=avg_a,
            score_b=avg_b,
            individual_results=all_individual,
            total_cost_usd=sum(sr.total_cost_usd for sr in sub_results),
            total_latency_ms=sum(sr.total_latency_ms for sr in sub_results),
            total_input_tokens=sum(sr.total_input_tokens for sr in sub_results),
            total_output_tokens=sum(sr.total_output_tokens for sr in sub_results),
            metadata={"sub_verdicts": {j.model_id: sr.verdict for j, sr in zip(self.judges, sub_results)}},
        )


class CombinedBudgetStrategy(StrategyBase):
    """S8: Position Swap + CoT Forcing + Calibrated Rubric (budget-friendly)."""

    name = "S8_combined_budget"

    def __init__(self, judge: JudgeBase):
        self.judge = judge

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        # Position swap with CoT+Rubric combined prompt
        combined_prompt = _merge_cot_and_rubric()
        swap = PositionSwapStrategy(self.judge, system_prompt=combined_prompt)
        return await swap.evaluate(question, response_a, response_b, reference)


def _merge_cot_and_rubric() -> str:
    """Create a combined CoT + Calibrated Rubric prompt."""
    return """\
You are an expert evaluator. Analyze two responses step by step using a detailed rubric.

**Rubric (score each criterion 1-5):**
1. **Accuracy**: Is the information factually correct?
   - 1: Major errors  2: Some errors  3: Mostly correct  4: Minor issues  5: Fully accurate
2. **Relevance**: Does it address the question directly?
   - 1: Off-topic  2: Partially relevant  3: Mostly relevant  4: Directly relevant  5: Perfectly targeted
3. **Completeness**: Does it cover all aspects?
   - 1: Very incomplete  2: Missing major parts  3: Covers basics  4: Thorough  5: Comprehensive
4. **Clarity**: Is it well-organized and easy to understand?
   - 1: Confusing  2: Poorly organized  3: Adequate  4: Clear  5: Exceptionally clear
5. **Reasoning**: Does it show sound logical reasoning?
   - 1: Illogical  2: Weak  3: Adequate  4: Sound  5: Exceptional

**IMPORTANT:** Do NOT consider response length or formatting style.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Follow these steps:
Step 1: Identify the key requirements of the question.
Step 2: Score Response A on all 5 criteria with brief justification.
Step 3: Score Response B on all 5 criteria with brief justification.
Step 4: Compare totals and determine the winner.

Output JSON:
{{
  "step1_requirements": "<key requirements>",
  "scores_a": {{"accuracy": X, "relevance": X, "completeness": X, "clarity": X, "reasoning": X}},
  "scores_b": {{"accuracy": X, "relevance": X, "completeness": X, "clarity": X, "reasoning": X}},
  "total_a": X,
  "total_b": X,
  "verdict": "A" or "B" or "tie",
  "reasoning": "<brief explanation>"
}}"""
