#!/usr/bin/env python3
"""Human annotation tool for the STYLE pairs human study.

Goal (per Reviewer p6d3 RC3): determine whether the LLM judges' strong
preference for markdown over plain prose reflects bias or a real readability
advantage that humans would also recognize. If humans also prefer the markdown
side at high rates, the LLM verdict is tracking a legitimate preference; if
humans are roughly 50/50 or prefer prose, the LLM verdict is bias.

Usage:
    # Annotator 1
    python scripts/human_study_style.py --annotator alice

    # Annotator 2 (sees the same 30 pairs in same randomized order)
    python scripts/human_study_style.py --annotator bob

    # After both have annotated, run the comparison:
    python scripts/human_study_style.py --report

Each pair shows both responses with formatting preserved (markdown side shows
its bullets/headers/bold markers as raw characters; prose side shows flowing
paragraphs). This is exactly what the LLM judges saw. The annotator is asked
which response they would prefer to read, mimicking the practical question of
"is the markdown side actually more readable, or just formatted?"

Pair ordering is randomized per annotator-name (different orders for alice
and bob), and within each pair we randomly swap which side is shown as
"Response 1" vs "Response 2" so the annotator does not learn that "Response 1
is always markdown." After collection, we map the annotator's "1/2/T" answer
back to the original A/B labels for the analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "custom" / "controlled_pairs.jsonl"
STUDY_DIR = ROOT / "data" / "human_study"
SAMPLE_SIZE = 30
SEED = 42


def strip_markdown(text: str) -> str:
    """Remove markdown formatting so both sides appear as flat prose.

    Strips: headers, bullets, bold/italic markers, code fences, links.
    Keeps the actual content text.
    """
    # Remove code fences (multi-line) first
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    # Remove header markers (# ## ###)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove bullet markers (-, *, +, 1., 2.)
    text = re.sub(r"^\s*[\-\*\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers (** *)
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove inline code markers
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove markdown link syntax: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_style_pairs() -> list[dict]:
    """Load STYLE pairs from the controlled dataset."""
    pairs = []
    with open(DATASET) as f:
        for line in f:
            d = json.loads(line)
            if d.get("bias_type") == "style":
                pairs.append(d)
    return pairs


def sample_pairs() -> list[dict]:
    """Sample SAMPLE_SIZE STYLE pairs with a fixed seed for reproducibility."""
    pairs = load_style_pairs()
    rng = np.random.default_rng(SEED)
    indices = sorted(rng.choice(len(pairs), size=min(SAMPLE_SIZE, len(pairs)), replace=False))
    return [pairs[i] for i in indices]


def _annotator_seed(name: str) -> int:
    """Per-annotator deterministic seed so each annotator sees a unique randomization."""
    return abs(hash(name)) % (2**31)


def annotate(annotator_name: str) -> None:
    """Walk the annotator through each pair and record their verdicts.

    Per pair, we randomly decide whether to show response_a as "Response 1"
    (slot1) or as "Response 2" (slot2). This blinds the annotator to which
    side is markdown so they cannot learn the pattern. The CSV stores the
    slot mapping so we can recover the original A/B verdict during analysis.
    """
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STUDY_DIR / f"{annotator_name}.csv"

    if out_path.exists():
        ans = input(f"{out_path} already exists. Continue from where you left off? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        with open(out_path) as f:
            done_ids = {row["pair_id"] for row in csv.DictReader(f)}
    else:
        done_ids = set()
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "slot1_is", "verdict_slot", "verdict_ab", "confidence"])

    pairs = sample_pairs()
    rng = np.random.default_rng(_annotator_seed(annotator_name))
    # Shuffle order per annotator so the two annotators see different sequences
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    # Per-pair coin: 0 means slot1=A (markdown), 1 means slot1=B (prose)
    swaps = rng.integers(0, 2, size=len(pairs))

    todo = [(p, s) for p, s in zip(pairs, swaps) if p["id"] not in done_ids]
    if not todo:
        print(f"All {len(pairs)} pairs already annotated. Run with --report to see results.")
        return

    print()
    print("=" * 70)
    print("HUMAN STUDY: which response is more useful / readable?")
    print("=" * 70)
    print(f"Annotator: {annotator_name}")
    print(f"Pairs to review: {len(todo)} (already done: {len(done_ids)})")
    print()
    print("Instructions:")
    print("  For each pair, read both responses carefully (formatting preserved).")
    print("  Decide which response you would prefer to READ AND USE.")
    print("  Both responses contain the same information, just presented differently.")
    print("  You are answering: which is more useful to you as a reader?")
    print()
    print("  This is exactly what the LLM judges saw, so your answer tells us")
    print("  whether the judges' markdown preference matches a real reading preference.")
    print()
    print("  Verdict: 1 = Response 1 is better, 2 = Response 2 is better, T = tie")
    print("  Confidence: 1 (low), 2 (medium), 3 (high)")
    print("  Type Q at any time to save and quit.")
    print()
    input("Press Enter to begin...")

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)

        for i, (pair, swap) in enumerate(todo, start=1):
            if swap == 0:
                slot1_text, slot2_text = pair["response_a"], pair["response_b"]
                slot1_is = "A"  # slot1 holds the markdown side
            else:
                slot1_text, slot2_text = pair["response_b"], pair["response_a"]
                slot1_is = "B"  # slot1 holds the prose side

            print()
            print("=" * 70)
            print(f"Pair {i} of {len(todo)} (id: {pair['id']})")
            print("=" * 70)
            print()
            print(f"QUESTION:")
            print(f"  {pair['question']}")
            print()
            print("-" * 70)
            print("RESPONSE 1:")
            print("-" * 70)
            print(slot1_text)
            print()
            print("-" * 70)
            print("RESPONSE 2:")
            print("-" * 70)
            print(slot2_text)
            print()

            while True:
                verdict_slot = input("Verdict (1/2/T/Q): ").strip().upper()
                if verdict_slot in ("1", "2", "T", "Q"):
                    break
                print("  Please enter 1, 2, T (tie), or Q (quit).")
            if verdict_slot == "Q":
                print("Saved progress. Re-run to continue.")
                return

            while True:
                conf = input("Confidence (1=low, 2=med, 3=high): ").strip()
                if conf in ("1", "2", "3"):
                    break
                print("  Please enter 1, 2, or 3.")

            # Recover the A/B verdict from the slot verdict
            if verdict_slot == "T":
                verdict_ab = "T"
            elif verdict_slot == "1":
                verdict_ab = slot1_is  # whatever was in slot 1
            else:  # verdict_slot == "2"
                verdict_ab = "B" if slot1_is == "A" else "A"

            writer.writerow([pair["id"], slot1_is, verdict_slot, verdict_ab, conf])
            f.flush()

    print()
    print(f"Done! All annotations saved to {out_path}")
    print("Run with --report to see the comparison against judge verdicts.")


def report() -> None:
    """Compare human annotations against LLM judge verdicts on the same pairs."""
    if not STUDY_DIR.exists() or not list(STUDY_DIR.glob("*.csv")):
        print(f"No annotations found in {STUDY_DIR}. Run with --annotator NAME first.")
        return

    annotators = {}
    for path in STUDY_DIR.glob("*.csv"):
        name = path.stem
        annotators[name] = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Prefer verdict_ab (new format); fall back to verdict (old format)
                verdict_ab = row.get("verdict_ab") or row.get("verdict")
                annotators[name][row["pair_id"]] = verdict_ab

    print("=" * 70)
    print("HUMAN STUDY REPORT")
    print("=" * 70)
    print()
    print(f"Annotators: {list(annotators.keys())}")
    print()

    # Per-annotator preference distribution
    print("Per-annotator verdict distribution (A = markdown, B = prose, T = tie):")
    for name, verdicts in annotators.items():
        n = len(verdicts)
        if n == 0:
            continue
        counts = {"A": 0, "B": 0, "T": 0}
        for v in verdicts.values():
            counts[v] = counts.get(v, 0) + 1
        print(f"  {name} (n={n}): A={counts['A']/n:.2f}, B={counts['B']/n:.2f}, T={counts['T']/n:.2f}")
    print()

    # Inter-annotator agreement
    if len(annotators) >= 2:
        names = sorted(annotators.keys())
        a_name, b_name = names[0], names[1]
        common_ids = set(annotators[a_name]) & set(annotators[b_name])
        if common_ids:
            agree = sum(annotators[a_name][i] == annotators[b_name][i] for i in common_ids)
            print(f"Inter-annotator agreement ({a_name} vs {b_name}): {agree}/{len(common_ids)} = {agree/len(common_ids):.2f}")
            print()

    # Compare to LLM judge verdicts on the same pairs
    raw_dir = ROOT / "results" / "raw"
    judges = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    judge_labels = {
        "gemini-2.5-pro": "Gemini Pro",
        "claude-sonnet-4": "Claude",
        "gpt-4o": "GPT-4o",
        "llama-3.3-70b": "Llama",
        "gemini-2.5-flash": "Flash",
    }

    # Use the first annotator's set of pair ids for the comparison
    first_annotator = next(iter(annotators.values()))
    pair_ids = list(first_annotator.keys())

    print("LLM judge B0 verdicts on the same pairs (A = markdown):")
    print(f"  {'Judge':<14}{'A%':>6}{'B%':>6}{'T%':>6}{'AvsHuman%':>11}")
    for judge in judges:
        verdicts = []
        for pid in pair_ids:
            cache = raw_dir / f"{judge}_B0_naive_custom_{pid}.json"
            if cache.exists():
                with open(cache) as f:
                    d = json.load(f)
                verdicts.append(d["result"]["verdict"])
        if not verdicts:
            print(f"  {judge_labels[judge]:<14}-- no cached results")
            continue
        n = len(verdicts)
        a_rate = sum(1 for v in verdicts if v == "A") / n
        b_rate = sum(1 for v in verdicts if v == "B") / n
        t_rate = sum(1 for v in verdicts if v == "tie") / n

        # Agreement with the first annotator on the same pairs
        if first_annotator:
            judge_letter_to_human = {"A": "A", "B": "B", "tie": "T"}
            agree = 0
            cmp_n = 0
            for pid, judge_v in zip(pair_ids, verdicts):
                if pid in first_annotator:
                    cmp_n += 1
                    if judge_letter_to_human.get(judge_v) == first_annotator[pid]:
                        agree += 1
            avh = agree / cmp_n if cmp_n else 0
        else:
            avh = 0

        print(f"  {judge_labels[judge]:<14}{a_rate:>6.2f}{b_rate:>6.2f}{t_rate:>6.2f}{avh:>11.2f}")
    print()

    # Aggregate human preference for markdown
    if annotators:
        all_human_a = 0
        all_human_total = 0
        for name, verdicts in annotators.items():
            for v in verdicts.values():
                all_human_total += 1
                if v == "A":
                    all_human_a += 1
        human_md_pref = all_human_a / all_human_total if all_human_total else 0
        print(f"Aggregate human preference for markdown content: {human_md_pref:.2f}")
        print(f"  (n = {all_human_total} verdicts across {len(annotators)} annotators)")
        print()
        print("Interpretation:")
        if human_md_pref > 0.65:
            print("  Humans also strongly prefer the markdown-side content. The LLM")
            print("  judges' markdown preference may track a real content/readability")
            print("  advantage rather than pure formatting bias.")
        elif human_md_pref < 0.35:
            print("  Humans prefer the prose-side content. The LLM judges' markdown")
            print("  preference is opposite to human preference: clear bias.")
        else:
            print("  Humans are roughly 50/50 on content preference. The LLM judges'")
            print("  systematic markdown preference is therefore bias, not content tracking.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human annotation study for STYLE pairs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--annotator", type=str, help="Your name; saves annotations to data/human_study/{name}.csv")
    group.add_argument("--report", action="store_true", help="Generate the comparison report after annotation is done")
    args = parser.parse_args()

    if args.report:
        report()
    else:
        annotate(args.annotator)


if __name__ == "__main__":
    main()
