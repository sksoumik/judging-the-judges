"""Batch processing for Gemini judges via Vertex AI Batch Prediction API.

Submits all evaluation requests as a single batch job, which is:
- Cheaper (batch pricing is typically 50% off)
- Higher throughput (no rate limiting concerns)
- Asynchronous (submit and poll for results)
"""

import asyncio
import json
import time

from google import genai
from google.genai import types
from rich.console import Console

from src.judges.base import JudgeResult
from src.judges.prompts import NAIVE_JUDGE_PROMPT, build_prompt

console = Console()


def _parse_judge_response(text: str) -> dict:
    """Parse JSON response from judge."""
    cleaned = text.strip()
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "verdict": "tie",
            "score_a": 5.0,
            "score_b": 5.0,
            "reasoning": f"Failed to parse: {text[:200]}",
        }

    verdict = str(data.get("verdict", "tie")).strip().upper()
    if verdict not in ("A", "B", "TIE"):
        verdict = "tie"
    verdict = verdict.lower() if verdict == "TIE" else verdict

    return {
        "verdict": verdict,
        "score_a": float(data.get("score_a", data.get("total_a", 5.0))),
        "score_b": float(data.get("score_b", data.get("total_b", 5.0))),
        "reasoning": data.get("reasoning", data.get("step4_comparison", "")),
    }


class GeminiBatchProcessor:
    """Submit and manage batch evaluation jobs for Gemini models."""

    def __init__(
        self,
        project_id: str,
        model_id: str = "gemini-2.5-flash",
        location: str = "global",
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
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

    def submit_batch(
        self,
        requests: list[dict],
        system_prompt: str | None = None,
        display_name: str = "llm-judge-batch",
    ) -> str:
        """Submit a batch of evaluation requests.

        Args:
            requests: List of dicts with keys: id, question, response_a, response_b,
                      and optionally reference.
            system_prompt: Override prompt template.
            display_name: Name for the batch job.

        Returns:
            Batch job name (ID) for polling.
        """
        template = system_prompt or NAIVE_JUDGE_PROMPT
        inlined = []
        for req in requests:
            prompt = build_prompt(
                template,
                req["question"],
                req["response_a"],
                req["response_b"],
                req.get("reference"),
            )
            inlined.append(
                types.InlinedRequest(
                    model=self.model_id,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)],
                        )
                    ],
                    metadata={"id": req["id"]},
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        response_mime_type="application/json",
                    ),
                )
            )

        batch_job = self.client.batches.create(
            model=self.model_id,
            src=types.BatchJobSource(inlined_requests=inlined),
            config=types.CreateBatchJobConfig(display_name=display_name),
        )

        console.print(
            f"[bold]Batch submitted:[/bold] {batch_job.name} "
            f"({len(requests)} requests, model={self.model_id})"
        )
        return batch_job.name

    def poll_batch(
        self,
        batch_name: str,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> list[dict]:
        """Poll a batch job until completion and return results.

        Args:
            batch_name: Job name from submit_batch.
            poll_interval: Seconds between polls.
            timeout: Max seconds to wait.

        Returns:
            List of result dicts with id and parsed response.
        """
        start = time.monotonic()
        while True:
            job = self.client.batches.get(name=batch_name)
            elapsed = time.monotonic() - start

            if job.done:
                if job.state.name == "JOB_STATE_SUCCEEDED":
                    console.print(
                        f"[green]Batch completed:[/green] {batch_name} "
                        f"in {elapsed:.0f}s"
                    )
                    if job.completion_stats:
                        console.print(
                            f"  Success: {job.completion_stats.successful_count}, "
                            f"Failed: {job.completion_stats.failed_count}"
                        )
                    return self._fetch_results(job)
                else:
                    error_msg = job.error.message if job.error else "Unknown error"
                    raise RuntimeError(
                        f"Batch job {batch_name} failed: {job.state.name} - {error_msg}"
                    )

            if elapsed > timeout:
                raise TimeoutError(
                    f"Batch job {batch_name} timed out after {timeout}s "
                    f"(state: {job.state.name})"
                )

            console.print(
                f"  Polling {batch_name}: {job.state.name} ({elapsed:.0f}s elapsed)",
                style="dim",
            )
            time.sleep(poll_interval)

    def _fetch_results(self, job) -> list[dict]:
        """Extract results from a completed batch job."""
        results = []
        # Results are in the destination (GCS or inline depending on API)
        if job.dest and job.dest.gcs_uri:
            console.print(f"  Results at: {job.dest.gcs_uri}")
            # For GCS results, we'd need to download and parse
            # This is a placeholder for GCS-based result fetching
            console.print(
                "[yellow]GCS result fetching not yet implemented. "
                "Use inlined results or download from GCS manually.[/yellow]"
            )
        return results

    def submit_and_wait(
        self,
        requests: list[dict],
        system_prompt: str | None = None,
        display_name: str = "llm-judge-batch",
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> list[JudgeResult]:
        """Submit a batch and wait for results. Convenience method.

        Returns:
            List of JudgeResult objects.
        """
        batch_name = self.submit_batch(requests, system_prompt, display_name)
        raw_results = self.poll_batch(batch_name, poll_interval, timeout)

        judge_results = []
        for raw in raw_results:
            text = raw.get("response_text", "")
            parsed = _parse_judge_response(text)
            judge_results.append(
                JudgeResult(
                    verdict=parsed["verdict"],
                    score_a=parsed["score_a"],
                    score_b=parsed["score_b"],
                    reasoning=parsed["reasoning"],
                    model_id=self.model_id,
                    raw_response=text,
                    metadata={"batch_id": batch_name, "request_id": raw.get("id", "")},
                )
            )
        return judge_results

    def list_jobs(self, limit: int = 10) -> list[dict]:
        """List recent batch jobs."""
        jobs = []
        for job in self.client.batches.list():
            jobs.append(
                {
                    "name": job.name,
                    "display_name": job.display_name,
                    "state": job.state.name if job.state else "UNKNOWN",
                    "model": job.model,
                    "create_time": str(job.create_time) if job.create_time else None,
                    "done": job.done,
                }
            )
            if len(jobs) >= limit:
                break
        return jobs
