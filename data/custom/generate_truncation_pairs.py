#!/usr/bin/env python3
"""Generate 25 truncation-based LENGTH pairs to address the verbosity confound.

Unlike the expansion-based LENGTH pairs (where a short answer is expanded),
these start with a long, high-quality answer and truncate it mechanically.
The long answer is genuinely better (more complete), so a correct judge
should prefer the long version. If judges prefer the shorter (truncated)
version, it confirms conciseness bias independent of filler quality.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()

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

# 25 questions requiring detailed, multi-part answers
TRUNCATION_QUESTIONS = [
    # Math / Reasoning (5)
    "Explain the central limit theorem, why it matters, and give a real-world example.",
    "Walk through how to solve a system of three linear equations using Gaussian elimination.",
    "Explain Bayes' theorem with a medical testing example including calculations.",
    "What is the pigeonhole principle? Give three different applications.",
    "Explain the difference between NP, NP-hard, and NP-complete with examples.",
    # Coding / Technical (5)
    "Explain how garbage collection works in Java, including the different generations and GC algorithms.",
    "Describe the OAuth 2.0 authorization code flow step by step, including all parties involved.",
    "Explain how a B-tree index works in databases and why it is preferred over binary search trees.",
    "What are microservices? Explain the benefits, challenges, and when monoliths are preferable.",
    "Explain how TLS/SSL handshake works to establish a secure connection.",
    # Science (5)
    "Explain how mRNA vaccines work, from design through immune response.",
    "Describe the water cycle in detail, including all major processes and their drivers.",
    "Explain plate tectonics: what drives it, types of boundaries, and geological consequences.",
    "How does nuclear fusion work in stars? Describe the proton-proton chain.",
    "Explain the greenhouse effect, including the role of each major greenhouse gas.",
    # Humanities / Social Science (5)
    "Explain the causes and consequences of the French Revolution in detail.",
    "What is cognitive behavioral therapy? Explain its principles, techniques, and evidence base.",
    "Describe the key principles of supply and demand, including elasticity and market equilibrium.",
    "Explain the philosophical concept of utilitarianism, its variants, and major criticisms.",
    "What were the main causes of World War I? Explain each contributing factor.",
    # Instruction Following (5)
    "Write a comprehensive guide to preparing for a job interview, covering all stages.",
    "Explain the complete process of brewing coffee using the pour-over method.",
    "Describe a full disaster recovery plan for a small business IT infrastructure.",
    "Write detailed instructions for planning and executing a research literature review.",
    "Explain the complete process of buying a house, from pre-approval to closing.",
]

CATEGORIES = [
    "math_reasoning",
    "coding_technical",
    "science",
    "humanities",
    "instruction_following",
]


def truncate_response(text: str, target_ratio: float = 0.4) -> str:
    """Truncate a response to approximately target_ratio of its length.

    Truncates at sentence boundaries to keep the text coherent.
    """
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current.strip()) > 10:
            sentences.append(current)
            current = ""
    if current.strip():
        sentences.append(current)

    if len(sentences) <= 2:
        words = text.split()
        cut_point = max(1, int(len(words) * target_ratio))
        return " ".join(words[:cut_point])

    target_count = max(2, int(len(sentences) * target_ratio))
    return "".join(sentences[:target_count]).strip()


async def generate_long_response(client: genai.Client, question: str) -> str:
    """Generate a comprehensive, high-quality response."""
    prompt = (
        "Answer the following question thoroughly and comprehensively. "
        "Provide detailed explanations, examples, and cover all important aspects. "
        "Aim for a response of 300-500 words that demonstrates deep understanding.\n\n"
        f"Question: {question}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )
    return response.text or ""


async def main() -> None:
    if not PROJECT_ID:
        console.print("[red]GOOGLE_PROJECT_ID not set in .env[/red]")
        return

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
    )

    pairs = []
    start = time.monotonic()

    console.print("[bold]Generating 25 truncation-based LENGTH pairs...[/bold]")

    with Progress(console=console) as progress:
        task = progress.add_task("Truncation pairs", total=len(TRUNCATION_QUESTIONS))

        for i, question in enumerate(TRUNCATION_QUESTIONS):
            long_response = await generate_long_response(client, question)
            short_response = truncate_response(long_response, target_ratio=0.4)

            long_words = len(long_response.split())
            short_words = len(short_response.split())

            pairs.append(
                {
                    "id": f"length_truncated_{i:03d}",
                    "question": question,
                    "response_a": long_response,
                    "response_b": short_response,
                    "human_preference": "A",
                    "reference": None,
                    "bias_type": "length_truncated",
                    "expected_verdict": "A",
                    "manipulation": (
                        "response_b is a mechanical truncation of response_a; "
                        "response_a contains more complete information"
                    ),
                    "metadata": {
                        "category": CATEGORIES[i // 5],
                        "len_a": long_words,
                        "len_b": short_words,
                        "truncation_ratio": round(short_words / long_words, 2)
                            if long_words > 0
                            else 0,
                    },
                }
            )
            progress.advance(task)

    elapsed = time.monotonic() - start

    # Append to existing controlled_pairs.jsonl
    existing_pairs = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                data = json.loads(line)
                if data.get("bias_type") != "length_truncated":
                    existing_pairs.append(data)

    all_pairs = existing_pairs + pairs
    with open(OUTPUT_PATH, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    avg_long = sum(p["metadata"]["len_a"] for p in pairs) / len(pairs)
    avg_short = sum(p["metadata"]["len_b"] for p in pairs) / len(pairs)
    avg_ratio = sum(p["metadata"]["truncation_ratio"] for p in pairs) / len(pairs)

    console.print(
        f"\n[bold green]Done![/bold green] {len(pairs)} truncation pairs generated"
    )
    console.print(f"Time: {elapsed:.0f}s")
    console.print(
        f"Avg words: long={avg_long:.0f}, short={avg_short:.0f} "
        f"(ratio={avg_ratio:.2f})"
    )
    console.print(f"Total pairs in dataset: {len(all_pairs)}")


if __name__ == "__main__":
    asyncio.run(main())
