"""S2: Same-family ensemble. 3 calls with same model at different temperatures."""

import asyncio

from src.judges.base import JudgeBase
from src.strategies.base import StrategyBase, StrategyResult


class EnsembleSameStrategy(StrategyBase):
    name = "S2_ensemble_same"

    def __init__(
        self,
        judge: JudgeBase,
        temperatures: list[float] | None = None,
        system_prompt: str | None = None,
    ):
        self.judge = judge
        self.temperatures = temperatures or [0.0, 0.3, 0.7]
        self.system_prompt = system_prompt

    async def evaluate(
        self,
        question: str,
        response_a: str,
        response_b: str,
        reference: str | None = None,
    ) -> StrategyResult:
        tasks = [
            self.judge.evaluate_pair_with_temperature(
                question, response_a, response_b,
                temperature=temp,
                system_prompt=self.system_prompt,
                reference=reference,
            )
            for temp in self.temperatures
        ]
        results = await asyncio.gather(*tasks)
        return self._aggregate_results(list(results), self.name)
