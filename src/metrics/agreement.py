"""Human agreement rate and Cohen's kappa."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def human_agreement_rate(
    predictions: list[str],
    gold_labels: list[str],
    bootstrap_n: int = 1000,
) -> dict:
    """Compute agreement rate with confidence interval.

    Args:
        predictions: List of predicted verdicts ("A", "B", "tie")
        gold_labels: List of gold-standard verdicts
        bootstrap_n: Number of bootstrap samples for CI

    Returns:
        dict with "agreement", "ci_lower", "ci_upper"
    """
    assert len(predictions) == len(gold_labels)
    matches = [p == g for p, g in zip(predictions, gold_labels)]
    agreement = sum(matches) / len(matches)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    arr = np.array(matches)
    boot_means = [
        arr[rng.choice(len(arr), size=len(arr), replace=True)].mean()
        for _ in range(bootstrap_n)
    ]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {"agreement": agreement, "ci_lower": ci_lower, "ci_upper": ci_upper}


def cohens_kappa(
    predictions: list[str],
    gold_labels: list[str],
) -> float:
    """Compute Cohen's kappa between predictions and gold labels."""
    return float(cohen_kappa_score(gold_labels, predictions))
