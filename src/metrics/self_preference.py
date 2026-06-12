"""Self-preference score: excess win rate when judging own model's outputs."""

import numpy as np


def self_preference_score(
    verdicts: list[str],
    response_a_origins: list[str],
    response_b_origins: list[str],
    judge_model: str,
    bootstrap_n: int = 1000,
) -> dict:
    """Compute self-preference score with CI.

    SPS = win_rate(own outputs) - win_rate(other outputs), controlling for quality.

    Args:
        verdicts: List of verdicts ("A", "B", "tie")
        response_a_origins: Model that generated each response A
        response_b_origins: Model that generated each response B
        judge_model: The judge model's identifier
        bootstrap_n: Number of bootstrap samples

    Returns:
        dict with "sps", "own_win_rate", "other_win_rate", "ci_lower", "ci_upper"
    """
    own_wins = []
    other_wins = []

    for verdict, origin_a, origin_b in zip(verdicts, response_a_origins, response_b_origins):
        a_is_own = _is_same_family(origin_a, judge_model)
        b_is_own = _is_same_family(origin_b, judge_model)

        if a_is_own and not b_is_own:
            own_wins.append(1 if verdict == "A" else 0)
        elif b_is_own and not a_is_own:
            own_wins.append(1 if verdict == "B" else 0)
        elif not a_is_own and not b_is_own:
            # Neither is own model, use as baseline
            other_wins.append(1 if verdict == "A" else 0)

    if not own_wins or not other_wins:
        return {
            "sps": 0.0,
            "own_win_rate": 0.0,
            "other_win_rate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_own": len(own_wins),
            "n_other": len(other_wins),
        }

    own_rate = sum(own_wins) / len(own_wins)
    other_rate = sum(other_wins) / len(other_wins)
    sps = own_rate - other_rate

    # Bootstrap CI for the difference
    rng = np.random.default_rng(42)
    own_arr = np.array(own_wins)
    other_arr = np.array(other_wins)
    boot_diffs = []
    for _ in range(bootstrap_n):
        own_sample = own_arr[rng.choice(len(own_arr), size=len(own_arr), replace=True)].mean()
        other_sample = other_arr[rng.choice(len(other_arr), size=len(other_arr), replace=True)].mean()
        boot_diffs.append(own_sample - other_sample)

    return {
        "sps": sps,
        "own_win_rate": own_rate,
        "other_win_rate": other_rate,
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
        "n_own": len(own_wins),
        "n_other": len(other_wins),
    }


def _is_same_family(model_a: str, model_b: str) -> bool:
    """Check if two model IDs belong to the same family."""
    families = {
        "gemini": ["gemini"],
        "claude": ["claude", "anthropic"],
        "llama": ["llama", "meta"],
        "gemma": ["gemma"],
        "mistral": ["mistral", "mixtral"],
    }
    a_lower = model_a.lower()
    b_lower = model_b.lower()
    for keywords in families.values():
        a_match = any(k in a_lower for k in keywords)
        b_match = any(k in b_lower for k in keywords)
        if a_match and b_match:
            return True
    return False
