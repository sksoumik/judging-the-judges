#!/usr/bin/env python3
"""Generate a self-contained HTML review document for the human STYLE study.

Produces three files in data/human_study/:
  - {name}_review.html   : the document the friend opens in a browser
  - {name}_key.json      : private answer key (do NOT send to friend)
  - {name}_answer_template.txt : template the friend can copy-paste into a reply

The HTML renders the markdown side as actual HTML (so headers, bullets, bold,
etc.\\ are visible), and the prose side as flowing paragraphs. The pair order
and per-pair slot assignment are randomized per annotator name so two
annotators see different shuffles. Pair IDs are preserved so we can decode
their answers later.

Usage:
    # Make a review packet for your friend:
    python scripts/generate_friend_review.py --annotator priya

    # Make a review packet for yourself (different randomization):
    python scripts/generate_friend_review.py --annotator soumik
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import markdown
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "custom" / "controlled_pairs.jsonl"
STUDY_DIR = ROOT / "data" / "human_study"
SAMPLE_SIZE = 30
SEED = 42  # same global sample as the CLI tool, so both methods agree

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  max-width: 900px;
  margin: 2em auto;
  padding: 0 1em;
  line-height: 1.5;
  color: #222;
}
h1 { color: #1a3552; border-bottom: 2px solid #1a3552; padding-bottom: 0.3em; }
h2.pair-header {
  color: #1a3552;
  background: #eef3fa;
  padding: 0.6em 1em;
  border-left: 4px solid #1a3552;
  margin-top: 3em;
}
.question {
  background: #fff8e1;
  padding: 1em;
  border-left: 4px solid #f0b400;
  margin: 1em 0;
  font-size: 1.05em;
}
.response {
  background: #f7f7f9;
  padding: 1.2em;
  border-radius: 6px;
  margin: 0.8em 0 1.5em 0;
  border: 1px solid #ddd;
}
.response-label {
  font-weight: bold;
  color: #555;
  margin-bottom: 0.5em;
  font-size: 0.95em;
}
.intro {
  background: #e8f5e9;
  padding: 1.5em;
  border-radius: 6px;
  border-left: 4px solid #2e7d32;
  margin-bottom: 2em;
}
.answer-prompt {
  background: #fce4ec;
  padding: 0.8em 1.2em;
  border-radius: 6px;
  margin-top: 1em;
  font-weight: 500;
}
.answer-template {
  background: #263238;
  color: #eceff1;
  padding: 1.2em;
  border-radius: 6px;
  font-family: 'SF Mono', 'Monaco', 'Menlo', monospace;
  white-space: pre-wrap;
  margin-top: 2em;
  font-size: 0.95em;
}
code, pre { background: #eee; padding: 0.1em 0.3em; border-radius: 3px; }
pre { padding: 0.8em; overflow-x: auto; }
"""

INTRO_HTML = """
<h1>Reviewing AI-generated responses</h1>
<div class="intro">
<p><strong>Hi! Thanks for helping me with this.</strong></p>
<p>You will see {n} pairs of responses to questions. For each pair, please tell me which response <strong>you would prefer to read and use</strong>, treating both as if they were results from a chatbot you were querying.</p>
<p>Both responses contain the same information, but they're presented differently. Some are formatted with bullets, headers, and bold text; others are plain paragraphs. <strong>I want to know which presentation YOU find more useful.</strong></p>
<p><strong>Three answer choices per pair:</strong></p>
<ul>
  <li><code>1</code> = Response 1 is better for me</li>
  <li><code>2</code> = Response 2 is better for me</li>
  <li><code>T</code> = they're roughly equal (tie)</li>
</ul>
<p>Optional: also include a confidence number (1=low, 2=medium, 3=high). Example: <code>style_017: 2 (3)</code> means you prefer Response 2 with high confidence.</p>
<p><strong>How to send your answers back:</strong></p>
<ol>
  <li>Read all {n} pairs.</li>
  <li>Scroll to the bottom of this page where you'll find an answer template.</li>
  <li>Copy the template into a text file, email, or chat message.</li>
  <li>Replace each blank with your answer (1, 2, or T).</li>
  <li>Send it back to me.</li>
</ol>
<p>You don't need to do all {n} in one sitting. Take breaks. Skip a pair if you really cannot decide (just write <code>?</code> for that pair).</p>
<p>Estimated time: 1 to 2 hours.</p>
</div>
"""


def _annotator_seed(name: str) -> int:
    return abs(hash(name)) % (2**31)


def _md_to_html(text: str) -> str:
    """Render markdown to HTML using the standard markdown library."""
    return markdown.markdown(text, extensions=["extra", "sane_lists"])


def load_style_pairs() -> list[dict]:
    pairs = []
    with open(DATASET) as f:
        for line in f:
            d = json.loads(line)
            if d.get("bias_type") == "style":
                pairs.append(d)
    return pairs


def sample_pairs() -> list[dict]:
    """Same global sample of 30 pairs as the CLI tool (seed=42)."""
    pairs = load_style_pairs()
    rng = np.random.default_rng(SEED)
    indices = sorted(rng.choice(len(pairs), size=min(SAMPLE_SIZE, len(pairs)), replace=False))
    return [pairs[i] for i in indices]


def generate(annotator_name: str) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    pairs = sample_pairs()
    rng = np.random.default_rng(_annotator_seed(annotator_name))
    order = rng.permutation(len(pairs))
    swaps = rng.integers(0, 2, size=len(pairs))  # 0 = slot1 holds A; 1 = slot1 holds B
    ordered_pairs = [(pairs[i], int(swaps[i])) for i in order]

    # Build the HTML
    body_chunks = [INTRO_HTML.format(n=len(ordered_pairs))]
    template_lines = []
    answer_key = []  # private: maps pair_id -> what slot1 actually was

    for display_idx, (pair, swap) in enumerate(ordered_pairs, start=1):
        if swap == 0:
            slot1_text, slot2_text = pair["response_a"], pair["response_b"]
            slot1_is = "A"  # markdown
        else:
            slot1_text, slot2_text = pair["response_b"], pair["response_a"]
            slot1_is = "B"  # prose

        # Render BOTH sides through the markdown engine. The markdown side
        # will gain real <h1>, <ul>, <strong>, etc.; the prose side passes
        # through as paragraphs (no markdown markers to convert).
        slot1_html = _md_to_html(slot1_text)
        slot2_html = _md_to_html(slot2_text)

        body_chunks.append(f"""
<h2 class="pair-header">Pair {display_idx} of {len(ordered_pairs)} &mdash; <code>{pair['id']}</code></h2>
<div class="question"><strong>Question:</strong> {html.escape(pair['question'])}</div>

<div class="response">
<div class="response-label">Response 1</div>
{slot1_html}
</div>

<div class="response">
<div class="response-label">Response 2</div>
{slot2_html}
</div>

<div class="answer-prompt">
<strong>Your answer for {pair['id']}:</strong> &nbsp; <code>1</code>, <code>2</code>, or <code>T</code> &nbsp;
(write it in the answer template at the bottom of this document)
</div>
""")

        template_lines.append(f"{pair['id']}: ___")
        answer_key.append({"pair_id": pair["id"], "slot1_is": slot1_is, "display_idx": display_idx})

    template_text = "\n".join(template_lines)
    body_chunks.append(f"""
<h1 style="margin-top: 4em;">Answer template</h1>
<p>Copy the block below, fill in your answer (1, 2, or T) for each pair, and send back to Soumik.</p>
<div class="answer-template">{html.escape(template_text)}</div>
<p style="margin-top: 2em;">Thank you so much for taking the time to do this! &mdash; Soumik</p>
""")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Style preference review &mdash; {len(ordered_pairs)} pairs ({annotator_name})</title>
<style>{CSS}</style>
</head>
<body>
{''.join(body_chunks)}
</body>
</html>
"""

    html_path = STUDY_DIR / f"{annotator_name}_review.html"
    key_path = STUDY_DIR / f"{annotator_name}_key.json"
    template_path = STUDY_DIR / f"{annotator_name}_answer_template.txt"

    html_path.write_text(html_doc)
    key_path.write_text(json.dumps({"annotator": annotator_name, "pairs": answer_key}, indent=2))
    template_path.write_text(template_text)

    print(f"Generated:")
    print(f"  HTML for friend:        {html_path}")
    print(f"  Private answer key:     {key_path}  (DO NOT send to friend)")
    print(f"  Plain answer template:  {template_path}  (optional; the same template is embedded in the HTML)")
    print()
    print("Send the HTML file to your friend. Keep the JSON key on your side; the ingest script needs it.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a friend-shareable HTML review packet")
    parser.add_argument("--annotator", required=True, help="Name of the annotator (used to seed randomization and name files)")
    args = parser.parse_args()
    generate(args.annotator)


if __name__ == "__main__":
    main()
