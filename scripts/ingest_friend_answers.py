#!/usr/bin/env python3
"""Ingest a friend's filled-in answer template and convert to the study CSV.

The friend will reply with something that looks like:

    style_001: 1
    style_005: T (3)
    style_012: 2
    ...

This script:
  1. Parses that block (very forgiving: ignores extra whitespace, comments, blank lines).
  2. Looks up the per-pair slot mapping in {annotator}_key.json (which we kept private).
  3. Maps each "1/2/T" answer back to the original "A/B/T" verdict.
  4. Writes data/human_study/{annotator}.csv in the same format the report expects.

Usage:
    # Save the friend's reply text to a file (e.g., paste into priya_raw.txt)
    python scripts/ingest_friend_answers.py --annotator priya --raw priya_raw.txt

    # Or pipe stdin:
    cat reply.txt | python scripts/ingest_friend_answers.py --annotator priya

After ingest, run the comparison report:
    python scripts/human_study_style.py --report
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = ROOT / "data" / "human_study"

# Lines look like "style_001: 1" or "style_001: T (3)" or "style_001: 2 [confidence 2]" etc.
LINE_RE = re.compile(
    r"^\s*(?P<pid>\S+?)\s*[:=]\s*(?P<v>[12tT?])\s*(?:[\(\[]\s*(?P<c>[1-3])\s*[\)\]])?",
)


def parse_raw(text: str) -> dict[str, tuple[str, str]]:
    """Parse the friend's reply text into {pair_id: (verdict_slot, confidence)}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        pid = m.group("pid")
        v = m.group("v").upper()
        if v == "T":
            verdict = "T"
        elif v == "?":
            verdict = "?"
        else:
            verdict = v  # "1" or "2"
        conf = m.group("c") or "2"  # default to medium confidence
        out[pid] = (verdict, conf)
    return out


def ingest(annotator: str, raw_text: str, key_name: str | None = None) -> None:
    """Ingest a reply.

    annotator : the name to write the CSV under (data/human_study/{annotator}.csv)
    key_name  : the name of the key file to decode against. Defaults to annotator.
                Use this when multiple annotators reviewed the same HTML packet
                (e.g., two friends both reviewed friend_review.html).
    """
    key_basename = key_name or annotator
    key_path = STUDY_DIR / f"{key_basename}_key.json"
    if not key_path.exists():
        sys.exit(
            f"ERROR: {key_path} not found. Run scripts/generate_friend_review.py "
            f"--annotator {key_basename} first to create it."
        )

    key_data = json.loads(key_path.read_text())
    slot1_lookup = {p["pair_id"]: p["slot1_is"] for p in key_data["pairs"]}

    answers = parse_raw(raw_text)
    if not answers:
        sys.exit("ERROR: no parseable answer lines found in input. Expected lines like 'style_001: 1'.")

    out_path = STUDY_DIR / f"{annotator}.csv"
    n_written = 0
    n_skipped = 0
    n_unknown = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "slot1_is", "verdict_slot", "verdict_ab", "confidence"])
        for pid, (verdict_slot, conf) in answers.items():
            if pid not in slot1_lookup:
                print(f"  WARNING: pair_id {pid!r} not in answer key, skipping")
                n_unknown += 1
                continue
            if verdict_slot == "?":
                n_skipped += 1
                continue
            slot1_is = slot1_lookup[pid]
            if verdict_slot == "T":
                verdict_ab = "T"
            elif verdict_slot == "1":
                verdict_ab = slot1_is
            else:  # "2"
                verdict_ab = "B" if slot1_is == "A" else "A"
            writer.writerow([pid, slot1_is, verdict_slot, verdict_ab, conf])
            n_written += 1

    print(f"Ingested {n_written} answers from {annotator}.")
    if n_skipped:
        print(f"  Skipped {n_skipped} pairs the annotator marked '?'.")
    if n_unknown:
        print(f"  Skipped {n_unknown} unknown pair IDs.")
    expected = len(slot1_lookup)
    if n_written < expected:
        missing = set(slot1_lookup) - set(answers.keys())
        print(f"  Note: {len(missing)} of {expected} pairs are missing from this reply: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"  Saved to {out_path}")
    print()
    print("Run scripts/human_study_style.py --report once both annotators are ingested.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest friend's annotation reply")
    parser.add_argument("--annotator", required=True, help="Annotator name (used to name the output CSV: data/human_study/{annotator}.csv)")
    parser.add_argument("--key-name", help="Name of the key file to decode against (default: same as --annotator). Use when multiple annotators shared the same HTML packet.")
    parser.add_argument("--raw", help="Path to file containing the friend's raw answer text. If omitted, read from stdin.")
    args = parser.parse_args()

    if args.raw:
        text = Path(args.raw).read_text()
    else:
        text = sys.stdin.read()

    ingest(args.annotator, text, key_name=args.key_name)


if __name__ == "__main__":
    main()
