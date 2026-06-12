"""Position consistency rate (PCR): how often verdict stays the same when positions swap."""

import numpy as np


def position_consistency_rate(
    verdicts_ab: list[str],
    verdicts_ba: list[str],
    bootstrap_n: int = 1000,
) -> dict:
    """Compute PCR with bootstrap confidence interval.

    Args:
        verdicts_ab: Verdicts when responses are in original order (A, B)
        verdicts_ba: Verdicts when responses are swapped (B, A).
            These should already be flipped back to AB frame
            (i.e., if the judge said "A" in BA ordering, record as "B" here).

    Returns:
        dict with "pcr", "ci_lower", "ci_upper"
    """
    assert len(verdicts_ab) == len(verdicts_ba)
    consistent = [ab == ba for ab, ba in zip(verdicts_ab, verdicts_ba)]
    pcr = sum(consistent) / len(consistent)

    rng = np.random.default_rng(42)
    arr = np.array(consistent)
    boot_means = [
        arr[rng.choice(len(arr), size=len(arr), replace=True)].mean()
        for _ in range(bootstrap_n)
    ]

    return {
        "pcr": pcr,
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }
