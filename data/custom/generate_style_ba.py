#!/usr/bin/env python3
"""Generate STYLE pairs with positions reversed (markdown in B, prose in A).

The existing 50 STYLE pairs all place markdown in slot A and prose in slot B,
which means the reported style bias confounds formatting preference with any
residual position effect on these specific pairs (reviewer GR4A "Critical" 4).

This script reads the existing STYLE pairs and emits a mirrored copy where
markdown is in slot B and prose is in slot A. Combined, the two halves let us
report the position-averaged style bias.

The output is appended to data/custom/controlled_pairs.jsonl with
bias_type="style_ba" and ids prefixed "style_ba_". No new generation calls are
needed; this is just a structural mirror of the existing pairs.
"""

import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "controlled_pairs.jsonl"


def main() -> None:
    if not OUTPUT_PATH.exists():
        print(f"ERROR: {OUTPUT_PATH} does not exist; run generate_controlled.py first")
        return

    with open(OUTPUT_PATH) as f:
        all_pairs = [json.loads(line) for line in f]

    style_pairs = [p for p in all_pairs if p.get("bias_type") == "style"]
    if not style_pairs:
        print("ERROR: no STYLE pairs found in dataset")
        return

    # Drop any prior style_ba pairs so this script is idempotent
    other_pairs = [p for p in all_pairs if p.get("bias_type") != "style_ba"]

    new_pairs = []
    for p in style_pairs:
        new_id = "style_ba_" + p["id"].removeprefix("style_")
        mirrored = {
            "id": new_id,
            "question": p["question"],
            # SWAP: markdown was A, prose was B; now prose is A, markdown is B
            "response_a": p["response_b"],
            "response_b": p["response_a"],
            "human_preference": None,
            "reference": None,
            "bias_type": "style_ba",
            "expected_verdict": "tie",
            "manipulation": (
                "STYLE pair with positions reversed: response_a is plain prose, "
                "response_b is markdown formatting; mirrors the corresponding "
                f"{p['id']} pair to enable position-averaged style bias measurement"
            ),
            "metadata": {
                **p.get("metadata", {}),
                "mirror_of": p["id"],
            },
        }
        new_pairs.append(mirrored)

    final_pairs = other_pairs + new_pairs
    with open(OUTPUT_PATH, "w") as f:
        for pair in final_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(new_pairs)} STYLE pairs with positions reversed")
    print(f"Total pairs in dataset: {len(final_pairs)}")
    print(f"  Original style (markdown in A): {len(style_pairs)}")
    print(f"  Mirrored style_ba (markdown in B): {len(new_pairs)}")


if __name__ == "__main__":
    main()
