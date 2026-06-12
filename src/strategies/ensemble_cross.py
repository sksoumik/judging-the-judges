"""S3: Cross-family ensemble. 3 judges from different model families."""

import asyncio

from src.judges.base import JudgeBase
from src.strategies.base import StrategyBase, StrategyResult


class EnsembleCrossStrategy(StrategyBase):
    name = "S3_ensemble_cross"

    def __init__(
        self,
        judges: list[JudgeBase],
        system_prompt: str | None = None,
    ):
        if len(judges) < 2:
            raise ValueError("Cross-family ensemble requires at least 2 judges")
        self.judges = judges
        self.system_prompt = system_prompt

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        tasks = [
            judge.evaluate_pair(
                question, response_a, response_b,
                system_prompt=self.system_prompt,
                reference=reference,
            )
            for judge in self.judges
        ]
        results = await asyncio.gather(*tasks)
        return self._aggregate_results(list(results), self.name)
