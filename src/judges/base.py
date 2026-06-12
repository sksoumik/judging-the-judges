from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class JudgeResult:
    verdict: Literal["A", "B", "tie"]
    score_a: float
    score_b: float
    reasoning: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model_id: str = ""
    raw_response: str = ""
    metadata: dict = field(default_factory=dict)


class JudgeBase(ABC):
    """Abstract base class for all LLM judges."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    @abstractmethod
    async def evaluate_pair(
        self,
        question: str,
        response_a: str,
        response_b: str,
        system_prompt: str | None = None,
        rubric: str | None = None,
        reference: str | None = None,
    ) -> JudgeResult:
        """Evaluate a pair of responses and return a verdict."""
        ...

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self.cost_per_1k_input
            + output_tokens / 1000 * self.cost_per_1k_output
        )

    async def evaluate_pair_with_temperature(
        self,
        question: str,
        response_a: str,
        response_b: str,
        temperature: float,
        system_prompt: str | None = None,
        rubric: str | None = None,
        reference: str | None = None,
    ) -> JudgeResult:
        """Evaluate with a specific temperature override."""
        original_temp = self.temperature
        self.temperature = temperature
        try:
            return await self.evaluate_pair(
                question, response_a, response_b, system_prompt, rubric, reference
            )
        finally:
            self.temperature = original_temp
