"""OpenAI GPT judge using the official OpenAI Python SDK."""

import asyncio
import json
import logging
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.judges.base import JudgeBase, JudgeResult
from src.judges.prompts import NAIVE_JUDGE_PROMPT, build_prompt

logger = logging.getLogger(__name__)


class OpenAIJudge(JudgeBase):
    """Judge using OpenAI GPT models via the OpenAI API."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ):
        """Initialize OpenAI judge.

        Args:
            model_id: OpenAI model identifier (e.g., "gpt-4o").
            temperature: Sampling temperature.
            max_output_tokens: Maximum output tokens.
            cost_per_1k_input: Cost per 1K input tokens in USD.
            cost_per_1k_output: Cost per 1K output tokens in USD.
        """
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        # Reads OPENAI_API_KEY from environment automatically
        self._client = OpenAI()

    def _call_model(self, prompt: str) -> tuple[str, int, int]:
        """Synchronous model call via OpenAI API.

        Args:
            prompt: The full evaluation prompt.

        Returns:
            Tuple of (raw text response, input tokens, output tokens).
        """
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert evaluator. Always respond with valid JSON only. "
                        "No markdown fences, no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
        output_tokens = response.usage.completion_tokens if response.usage else len(text) // 4
        return text, input_tokens, output_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
    async def evaluate_pair(
        self,
        question: str,
        response_a: str,
        response_b: str,
        system_prompt: str | None = None,
        rubric: str | None = None,
        reference: str | None = None,
    ) -> JudgeResult:
        """Evaluate a pair of responses using GPT-4o via the OpenAI API.

        Args:
            question: The evaluation question/prompt.
            response_a: First candidate response.
            response_b: Second candidate response.
            system_prompt: Optional custom prompt template.
            rubric: Optional rubric (unused, for interface compatibility).
            reference: Optional reference answer.

        Returns:
            JudgeResult with verdict, scores, reasoning, and cost metadata.
        """
        template = system_prompt or NAIVE_JUDGE_PROMPT
        prompt = build_prompt(template, question, response_a, response_b, reference)

        start = time.monotonic()
        raw_text, input_tokens, output_tokens = await asyncio.to_thread(
            self._call_model, prompt
        )
        latency_ms = (time.monotonic() - start) * 1000

        cost = self._calculate_cost(input_tokens, output_tokens)
        parsed = _parse_judge_response(raw_text)

        return JudgeResult(
            verdict=parsed["verdict"],
            score_a=parsed["score_a"],
            score_b=parsed["score_b"],
            reasoning=parsed["reasoning"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_id=self.model_id,
            raw_response=raw_text,
        )


def _parse_judge_response(text: str) -> dict:
    """Parse JSON response from judge, handling markdown code fences.

    Args:
        text: Raw text response from the model.

    Returns:
        Dict with verdict, score_a, score_b, and reasoning.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Try to extract JSON from surrounding text
    if not cleaned.startswith("{"):
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse GPT response as JSON: %s", text[:200])
        return {
            "verdict": "tie",
            "score_a": 5.0,
            "score_b": 5.0,
            "reasoning": f"Failed to parse response: {text[:200]}",
        }

    verdict = str(data.get("verdict", "tie")).strip().upper()
    if verdict not in ("A", "B", "TIE"):
        verdict = "tie"
    verdict = verdict.lower() if verdict == "TIE" else verdict

    score_a = float(data.get("score_a", data.get("total_a", 5.0)))
    score_b = float(data.get("score_b", data.get("total_b", 5.0)))
    reasoning = data.get("reasoning", data.get("step4_comparison", ""))

    return {
        "verdict": verdict,
        "score_a": score_a,
        "score_b": score_b,
        "reasoning": reasoning,
    }
