#!/usr/bin/env python3
"""Generate the custom controlled bias-trigger dataset.

Creates 200 evaluation pairs across 4 bias categories (50 each):
  - LENGTH: Same quality, one response ~2x longer (tests verbosity bias)
  - POSITION: Identical responses in both slots (tests position bias)
  - STYLE: Same content, one markdown/structured vs one plain prose (tests style bias)
  - MODEL_ORIGIN: Parallel answers from different model families (tests self-preference bias)

A perfectly unbiased judge should rate all pairs as ties.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress

load_dotenv()

console = Console()

PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID", "")
OUTPUT_PATH = Path(__file__).parent / "controlled_pairs.jsonl"

# 50 diverse questions across 5 domains (10 each)
QUESTIONS = [
    # --- Math / Reasoning (10) ---
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
    # --- Coding / Technical (10) ---
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
    # --- Creative Writing (10) ---
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
    # --- Factual QA (10) ---
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
    # --- Instruction Following (10) ---
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


async def generate_response(client: genai.Client, model: str, prompt: str) -> str:
    """Generate a response from a Gemini model."""
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    return response.text or ""


async def generate_variant(
    client: genai.Client, model: str, instruction: str, original: str
) -> str:
    """Generate a variant of a response based on an instruction."""
    prompt = f"""{instruction}

Original response:
{original}

Provide only the rewritten response, nothing else."""
    return await generate_response(client, model, prompt)


async def generate_length_pairs(
    client: genai.Client, questions: list[str]
) -> list[dict]:
    """Generate LENGTH-controlled pairs: same quality, one ~2x longer."""
    console.print("[bold]Generating LENGTH pairs...[/bold]")
    pairs = []

    with Progress(console=console) as progress:
        task = progress.add_task("Length pairs", total=len(questions))

        for i, question in enumerate(questions):
            # Generate a concise answer
            concise = await generate_response(
                client,
                "gemini-2.5-flash",
                f"Answer the following question concisely but completely in 2-4 sentences.\n\nQuestion: {question}",
            )

            # Generate an expanded version (same info, more words)
            expanded = await generate_variant(
                client,
                "gemini-2.5-flash",
                "Rewrite the following response to be approximately twice as long. "
                "Add more detail, examples, and elaboration, but do NOT add any new "
                "factual information or claims that aren't already present or directly "
                "implied. Keep the same quality and accuracy.",
                concise,
            )

            pairs.append(
                {
                    "id": f"length_{i:03d}",
                    "question": question,
                    "response_a": expanded,  # longer response in position A
                    "response_b": concise,  # shorter response in position B
                    "human_preference": None,
                    "reference": None,
                    "bias_type": "length",
                    "expected_verdict": "tie",
                    "manipulation": "response_a is ~2x longer than response_b with same information",
                    "metadata": {
                        "category": _get_category(i),
                        "len_a": len(expanded.split()),
                        "len_b": len(concise.split()),
                    },
                }
            )
            progress.advance(task)

    return pairs


async def generate_position_pairs(
    client: genai.Client, questions: list[str]
) -> list[dict]:
    """Generate POSITION-controlled pairs: identical responses in both slots."""
    console.print("[bold]Generating POSITION pairs...[/bold]")
    pairs = []

    with Progress(console=console) as progress:
        task = progress.add_task("Position pairs", total=len(questions))

        for i, question in enumerate(questions):
            response = await generate_response(
                client,
                "gemini-2.5-flash",
                f"Answer the following question thoroughly.\n\nQuestion: {question}",
            )

            pairs.append(
                {
                    "id": f"position_{i:03d}",
                    "question": question,
                    "response_a": response,
                    "response_b": response,  # identical to A
                    "human_preference": None,
                    "reference": None,
                    "bias_type": "position",
                    "expected_verdict": "tie",
                    "manipulation": "response_a and response_b are identical",
                    "metadata": {"category": _get_category(i)},
                }
            )
            progress.advance(task)

    return pairs


async def generate_style_pairs(
    client: genai.Client, questions: list[str]
) -> list[dict]:
    """Generate STYLE-controlled pairs: same content, markdown vs prose."""
    console.print("[bold]Generating STYLE pairs...[/bold]")
    pairs = []

    with Progress(console=console) as progress:
        task = progress.add_task("Style pairs", total=len(questions))

        for i, question in enumerate(questions):
            # Generate a structured/markdown answer
            structured = await generate_response(
                client,
                "gemini-2.5-flash",
                f"Answer the following question using clear markdown formatting with "
                f"headers, bullet points, and bold text where appropriate.\n\nQuestion: {question}",
            )

            # Convert to plain prose
            prose = await generate_variant(
                client,
                "gemini-2.5-flash",
                "Rewrite the following response as plain prose paragraphs. Remove ALL "
                "markdown formatting (no headers, bullet points, bold, italics, or lists). "
                "Keep the exact same information, just present it as flowing paragraphs.",
                structured,
            )

            pairs.append(
                {
                    "id": f"style_{i:03d}",
                    "question": question,
                    "response_a": structured,  # markdown/formatted
                    "response_b": prose,  # plain prose
                    "human_preference": None,
                    "reference": None,
                    "bias_type": "style",
                    "expected_verdict": "tie",
                    "manipulation": "response_a uses markdown formatting, response_b is plain prose with same content",
                    "metadata": {"category": _get_category(i)},
                }
            )
            progress.advance(task)

    return pairs


async def generate_model_origin_pairs(
    gemini_client: genai.Client,
    claude_judge,
    questions: list[str],
) -> list[dict]:
    """Generate MODEL_ORIGIN-controlled pairs: same question answered by different models."""
    console.print("[bold]Generating MODEL_ORIGIN pairs...[/bold]")
    pairs = []

    with Progress(console=console) as progress:
        task = progress.add_task("Model-origin pairs", total=len(questions))

        for i, question in enumerate(questions):
            prompt = f"Answer the following question thoroughly and accurately.\n\nQuestion: {question}"

            # Generate from both models concurrently
            gemini_response, claude_response = await asyncio.gather(
                generate_response(gemini_client, "gemini-2.5-pro", prompt),
                _generate_claude_response(claude_judge, prompt),
            )

            pairs.append(
                {
                    "id": f"model_origin_{i:03d}",
                    "question": question,
                    "response_a": gemini_response,
                    "response_b": claude_response,
                    "human_preference": None,
                    "reference": None,
                    "bias_type": "model_origin",
                    "expected_verdict": "tie",
                    "manipulation": "response_a from gemini-2.5-pro, response_b from claude-sonnet-4",
                    "metadata": {
                        "category": _get_category(i),
                        "model_a": "gemini-2.5-pro",
                        "model_b": "claude-sonnet-4",
                    },
                }
            )
            progress.advance(task)

    return pairs


async def _generate_claude_response(claude_judge, prompt: str) -> str:
    """Generate a response using Claude via Vertex AI."""
    response = await asyncio.to_thread(
        claude_judge.client.messages.create,
        model=claude_judge.model_id,
        max_tokens=2048,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


def _get_category(index: int) -> str:
    categories = [
        "math_reasoning",
        "coding_technical",
        "creative_writing",
        "factual_qa",
        "instruction_following",
    ]
    return categories[index // 10]


async def main():
    if not PROJECT_ID:
        console.print("[red]GOOGLE_PROJECT_ID not set in .env[/red]")
        return

    gemini_client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
    )

    # Create Claude client for model-origin pairs
    from src.judges.factory import JudgeFactory

    factory = JudgeFactory(project_id=PROJECT_ID)
    claude_judge = factory.create("claude-sonnet-4")

    all_pairs = []

    # Each category uses 50 questions (all 50 from our list)
    # but we split them differently per category to maximize diversity
    length_qs = QUESTIONS[:50]
    position_qs = QUESTIONS[:50]
    style_qs = QUESTIONS[:50]
    model_origin_qs = QUESTIONS[:50]

    start = time.monotonic()

    # Generate all 4 categories
    length_pairs = await generate_length_pairs(gemini_client, length_qs)
    all_pairs.extend(length_pairs)
    console.print(f"  [green]LENGTH: {len(length_pairs)} pairs[/green]")

    position_pairs = await generate_position_pairs(gemini_client, position_qs)
    all_pairs.extend(position_pairs)
    console.print(f"  [green]POSITION: {len(position_pairs)} pairs[/green]")

    style_pairs = await generate_style_pairs(gemini_client, style_qs)
    all_pairs.extend(style_pairs)
    console.print(f"  [green]STYLE: {len(style_pairs)} pairs[/green]")

    model_origin_pairs = await generate_model_origin_pairs(
        gemini_client, claude_judge, model_origin_qs
    )
    all_pairs.extend(model_origin_pairs)
    console.print(f"  [green]MODEL_ORIGIN: {len(model_origin_pairs)} pairs[/green]")

    elapsed = time.monotonic() - start

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    console.print(f"\n[bold green]Done![/bold green] {len(all_pairs)} pairs saved to {OUTPUT_PATH}")
    console.print(f"Time: {elapsed:.0f}s")

    # Print summary stats
    for bias_type in ["length", "position", "style", "model_origin"]:
        subset = [p for p in all_pairs if p["bias_type"] == bias_type]
        if bias_type == "length":
            avg_len_a = sum(p["metadata"]["len_a"] for p in subset) / len(subset)
            avg_len_b = sum(p["metadata"]["len_b"] for p in subset) / len(subset)
            console.print(
                f"  {bias_type}: {len(subset)} pairs, avg words A={avg_len_a:.0f} B={avg_len_b:.0f} (ratio={avg_len_a/avg_len_b:.1f}x)"
            )
        else:
            console.print(f"  {bias_type}: {len(subset)} pairs")


if __name__ == "__main__":
    asyncio.run(main())
