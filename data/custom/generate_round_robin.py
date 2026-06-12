#!/usr/bin/env python3
"""Generate round-robin MODEL_ORIGIN pairs covering all judge families.

The original MODEL_ORIGIN pairs (50 of them) are all Gemini 2.5 Pro vs Claude
Sonnet 4. This means GPT-4o, Llama, and Gemini Flash judges never see a
same-family option, so the existing self-preference metric is not interpretable
for them (reviewer p6d3 RC1 and reviewer GR4A "Critical" 3).

This script adds 100 new pairs covering combinations that include GPT-4o and
Llama, so every judge family has at least 25 same-family pairs to evaluate.

New pairs (100 total, 25 each):
  GPT-4o vs Gemini 2.5 Pro
  GPT-4o vs Claude Sonnet 4
  Llama 3.3-70B vs Gemini 2.5 Pro
  Llama 3.3-70B vs Claude Sonnet 4

For each pair we randomize which model goes in slot A vs B (per-instance fair
coin via numpy seed=42), so any preference for slot A/B is not confounded with
model identity. Self-preference is then computed as P(prefers own family) on the
subset of pairs where the judge's family is exactly one of the two responders.

Output is appended to data/custom/controlled_pairs.jsonl alongside the existing
LENGTH/POSITION/STYLE/MODEL_ORIGIN/length_truncated pairs.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import truststore

truststore.inject_into_ssl()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

console = Console()

PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "")
OUTPUT_PATH = Path(__file__).parent / "controlled_pairs.jsonl"

# Same 50-question pool as generate_controlled.py for consistency
QUESTIONS = [
    # Math / Reasoning (10)
    "Explain why the square root of 2 is irrational.",
    "A train leaves Station A at 60 mph and another leaves Station B at 80 mph toward each other. If the stations are 280 miles apart, when do they meet?",
    "What is the probability of rolling at least one six when rolling two fair dice?",
    "Explain the difference between correlation and causation with an example.",
    "Why is division by zero undefined?",
    "A store offers 20% off, then an additional 15% off the reduced price. What is the total discount?",
    "Explain the Monty Hall problem and why switching doors is advantageous.",
    "How many ways can you arrange the letters in the word MISSISSIPPI?",
    "What is the difference between permutations and combinations? Give an example of each.",
    "If a ball is thrown upward at 20 m/s, how long until it returns to the thrower's hand? Ignore air resistance.",
    # Coding / Technical (10)
    "Explain the difference between a stack and a queue with use cases for each.",
    "What is a race condition in concurrent programming and how can it be prevented?",
    "Explain how a hash table works and what happens during a collision.",
    "What is the time complexity of binary search and why?",
    "Explain the concept of recursion and provide an example of when it is useful.",
    "What is the difference between TCP and UDP? When would you use each?",
    "Explain what a database index is and how it improves query performance.",
    "What is the CAP theorem and what trade-offs does it describe?",
    "Explain the difference between compiled and interpreted languages with examples.",
    "What is dependency injection and why is it useful in software design?",
    # Creative Writing (10)
    "Write a short paragraph describing a sunset over the ocean.",
    "Create a brief character description for a detective in a noir story.",
    "Write an opening sentence for a science fiction novel set on Mars.",
    "Describe the taste of coffee to someone who has never tried it.",
    "Write a haiku about autumn.",
    "Create a brief dialogue between two strangers meeting on a train.",
    "Write a short metaphor comparing life to a river.",
    "Describe the sound of a thunderstorm from inside a cabin.",
    "Write a brief eulogy for a beloved family pet.",
    "Create a one-paragraph fairy tale with a moral lesson.",
    # Factual QA (10)
    "What causes the seasons on Earth?",
    "How does photosynthesis work?",
    "What is the theory of general relativity in simple terms?",
    "Why is the sky blue?",
    "How do vaccines work to protect against diseases?",
    "What caused the extinction of the dinosaurs?",
    "How does the human immune system fight infections?",
    "What is CRISPR and how is it used in gene editing?",
    "Why do we dream? Summarize the main scientific theories.",
    "How does blockchain technology work?",
    # Instruction Following (10)
    "List exactly 5 tips for improving public speaking skills.",
    "Explain quantum computing in exactly 3 sentences.",
    "Write a professional email declining a meeting invitation.",
    "Summarize the water cycle in under 50 words.",
    "Give step-by-step instructions for making a paper airplane.",
    "Compare and contrast renewable and non-renewable energy sources in a table format.",
    "Write a tweet (under 280 characters) promoting environmental awareness.",
    "Explain the scientific method to a 10-year-old.",
    "Create a pros and cons list for remote work.",
    "Write a formal apology letter for missing a deadline.",
]

CATEGORIES = [
    "math_reasoning",
    "coding_technical",
    "creative_writing",
    "factual_qa",
    "instruction_following",
]

# Pairings to add (each will use 25 questions).
# Llama pairings first because they only hit Vertex MaaS, not the OpenAI route;
# this lets the script make progress even if the OpenAI route is rate-limited
# by parallel work.
PAIRINGS = [
    ("llama-3.3-70b", "gemini-2.5-pro"),
    ("llama-3.3-70b", "claude-sonnet-4"),
    ("gpt-4o", "gemini-2.5-pro"),
    ("gpt-4o", "claude-sonnet-4"),
]


async def _generate_via_gemini_client(
    client: genai.Client, model_id: str, prompt: str
) -> str:
    """Generate using the google-genai client (works for Gemini and Llama MaaS)."""
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    return response.text or ""


async def _generate_via_anthropic(judge, prompt: str) -> str:
    """Generate using the AnthropicVertex client."""
    response = await asyncio.to_thread(
        judge.client.messages.create,
        model=judge.model_id,
        max_tokens=2048,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def _generate_via_openai(prompt: str) -> str:
    """Generate via OpenAI for GPT-4o.

    If a local override module exists at ``data.custom._local_openai_gen``
    exposing ``async def generate(prompt)``, it is used instead. Otherwise this
    falls through to the standard OpenAI SDK. Retries up to 5 times with
    exponential backoff on transient errors.
    """
    try:
        import importlib

        override = importlib.import_module("data.custom._local_openai_gen")
        return await override.generate(prompt)
    except ImportError:
        from openai import OpenAI

        oai = OpenAI()
        resp = await asyncio.to_thread(
            oai.chat.completions.create,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def _generate_via_gemini_client_retry(client, model_id, prompt):
    """Wrapper around _generate_via_gemini_client with retry."""
    return await asyncio.wait_for(
        _generate_via_gemini_client(client, model_id, prompt),
        timeout=120.0,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def _generate_via_anthropic_retry(judge, prompt):
    """Wrapper around _generate_via_anthropic with retry."""
    return await asyncio.wait_for(
        _generate_via_anthropic(judge, prompt),
        timeout=120.0,
    )


async def generate_for_model(
    model_name: str,
    prompt: str,
    gemini_client: genai.Client,
    anthropic_judge,
) -> str:
    """Dispatch generation to the right client based on model name (with retries)."""
    if model_name == "gpt-4o":
        return await _generate_via_openai(prompt)
    if model_name == "claude-sonnet-4":
        return await _generate_via_anthropic_retry(anthropic_judge, prompt)
    if model_name == "llama-3.3-70b":
        return await _generate_via_gemini_client_retry(
            gemini_client, "meta/llama-3.3-70b-instruct-maas", prompt
        )
    # Default: native Gemini
    return await _generate_via_gemini_client_retry(gemini_client, model_name, prompt)


def _category_for(question_index: int) -> str:
    return CATEGORIES[question_index // 10]


async def main() -> None:
    if not PROJECT_ID:
        console.print("[red]GOOGLE_PROJECT_ID not set in .env[/red]")
        return

    rng = np.random.default_rng(seed=42)

    gemini_client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
    )
    llama_client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="us-central1",
    )

    from src.judges.factory import JudgeFactory

    factory = JudgeFactory(project_id=PROJECT_ID)
    claude_judge = factory.create("claude-sonnet-4")

    # Helper that picks the right client for Llama (us-central1 vs global)
    async def gen(model_name: str, prompt: str) -> str:
        if model_name == "llama-3.3-70b":
            return await _generate_via_gemini_client(
                llama_client, "meta/llama-3.3-70b-instruct-maas", prompt
            )
        return await generate_for_model(model_name, prompt, gemini_client, claude_judge)

    new_pairs = []
    start = time.monotonic()

    # Resume from checkpoint if present
    pairings_done = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                d = json.loads(line)
                if d.get("bias_type") == "model_origin_rr":
                    pairings_done.add(d["metadata"]["pairing"])
                    new_pairs.append(d)
    if pairings_done:
        console.print(f"[yellow]Resuming: {len(pairings_done)} pairings already done: {pairings_done}[/yellow]")

    # Use 25 questions per pairing, drawn evenly across the 50 question pool
    pairing_question_indices = [
        list(range(0, 50, 2)),  # even indices: 0, 2, ..., 48 (25 questions)
        list(range(1, 50, 2)),  # odd indices: 1, 3, ..., 49 (25 questions)
        list(range(0, 50, 2)),
        list(range(1, 50, 2)),
    ]

    for pairing_idx, ((model_a_name, model_b_name), q_indices) in enumerate(
        zip(PAIRINGS, pairing_question_indices)
    ):
        pairing_key = f"{model_a_name}_vs_{model_b_name}"
        if pairing_key in pairings_done:
            console.print(f"[dim]Skipping {pairing_key} (already done)[/dim]")
            continue
        console.print(
            f"\n[bold]Pairing {pairing_idx + 1}/4: {model_a_name} vs {model_b_name}[/bold]"
        )

        with Progress(console=console) as progress:
            task = progress.add_task(
                f"{model_a_name} vs {model_b_name}", total=len(q_indices)
            )

            for q_idx in q_indices:
                question = QUESTIONS[q_idx]
                prompt = f"Answer the following question thoroughly and accurately.\n\nQuestion: {question}"

                # Generate from both models concurrently
                resp_a, resp_b = await asyncio.gather(
                    gen(model_a_name, prompt),
                    gen(model_b_name, prompt),
                )

                # Randomize slot assignment per pair (fair coin)
                swap = bool(rng.integers(0, 2))
                if swap:
                    response_a, response_b = resp_b, resp_a
                    model_in_a, model_in_b = model_b_name, model_a_name
                else:
                    response_a, response_b = resp_a, resp_b
                    model_in_a, model_in_b = model_a_name, model_b_name

                pair_id = f"model_origin_rr_{model_a_name.replace('-', '')}_{model_b_name.replace('-', '')}_{q_idx:03d}"
                new_pairs.append(
                    {
                        "id": pair_id,
                        "question": question,
                        "response_a": response_a,
                        "response_b": response_b,
                        "human_preference": None,
                        "reference": None,
                        "bias_type": "model_origin_rr",
                        "expected_verdict": "tie",
                        "manipulation": (
                            f"round-robin pair: response_a from {model_in_a}, "
                            f"response_b from {model_in_b}; slot assignment randomized "
                            f"to decouple model identity from position"
                        ),
                        "metadata": {
                            "category": _category_for(q_idx),
                            "model_a": model_in_a,
                            "model_b": model_in_b,
                            "pairing": f"{model_a_name}_vs_{model_b_name}",
                            "slot_swapped": swap,
                        },
                    }
                )
                progress.advance(task)

        # Checkpoint: persist progress after each pairing so a mid-run failure
        # (e.g., transient HTTP 500) does not throw away completed work.
        existing_pairs_chk = []
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH) as fh:
                for line in fh:
                    data = json.loads(line)
                    if data.get("bias_type") != "model_origin_rr":
                        existing_pairs_chk.append(data)
        with open(OUTPUT_PATH, "w") as fh:
            for pair in existing_pairs_chk + new_pairs:
                fh.write(json.dumps(pair) + "\n")
        console.print(f"  [dim]Checkpointed: {len(new_pairs)} round-robin pairs saved[/dim]")

    elapsed = time.monotonic() - start

    # Final write: ensure file reflects all pairings in case checkpoint missed any
    existing_pairs = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                data = json.loads(line)
                if data.get("bias_type") != "model_origin_rr":
                    existing_pairs.append(data)

    all_pairs = existing_pairs + new_pairs
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    console.print(
        f"\n[bold green]Done![/bold green] {len(new_pairs)} round-robin pairs generated in {elapsed:.0f}s"
    )
    console.print(f"Total pairs in dataset: {len(all_pairs)}")
    for pairing in PAIRINGS:
        n = sum(
            1
            for p in new_pairs
            if p["metadata"]["pairing"] == f"{pairing[0]}_vs_{pairing[1]}"
        )
        console.print(f"  {pairing[0]} vs {pairing[1]}: {n} pairs")


if __name__ == "__main__":
    asyncio.run(main())
