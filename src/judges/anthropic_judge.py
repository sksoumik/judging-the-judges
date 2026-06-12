"""Claude judge using Anthropic Vertex AI."""

import asyncio
import json
import time

from anthropic import AnthropicVertex
from tenacity import retry, stop_after_attempt, wait_exponential

from src.judges.base import JudgeBase, JudgeResult
from src.judges.prompts import NAIVE_JUDGE_PROMPT, build_prompt


class AnthropicJudge(JudgeBase):
    """Judge using Claude models via Vertex AI."""

    def __init__(
        self,
        project_id: str,
        model_id: str = "claude-sonnet-4@20250514",
        region: str = "global",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )
        self.client = AnthropicVertex(
            project_id=project_id,
            region=region,
        )

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
        template = system_prompt or NAIVE_JUDGE_PROMPT
        prompt = build_prompt(template, question, response_a, response_b, reference)

        start = time.monotonic()
        # AnthropicVertex.messages.create is synchronous, run in thread pool
        # with a per-call timeout so hung calls don't block asyncio.gather.
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.messages.create,
                model=self.model_id,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=120.0,
        )
        latency_ms = (time.monotonic() - start) * 1000

        raw_text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
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
    """Parse JSON response from judge, handling markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
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
