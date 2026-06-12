"""Verbosity bias: Spearman correlation between response length and judge score."""

from scipy import stats


def verbosity_bias(
    scores: list[float],
    token_counts: list[int],
) -> dict:
    """Compute Spearman correlation between token count and judge score.

    A positive correlation indicates verbosity bias (longer = higher score).

    Args:
        scores: Judge scores for responses
        token_counts: Token counts for the same responses

    Returns:
        dict with "rho", "p_value"
    """
    result = stats.spearmanr(token_counts, scores)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }
