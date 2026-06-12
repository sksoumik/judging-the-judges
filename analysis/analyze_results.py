#!/usr/bin/env python3
"""Analyze experiment results and generate figures/tables for the paper.

Run this after experiments complete:
    python analysis/analyze_results.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataLoader
from src.metrics.agreement import cohens_kappa, human_agreement_rate
from src.metrics.position_consistency import position_consistency_rate
from src.metrics.verbosity_correlation import verbosity_bias

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

STRATEGY_LABELS = {
    "B0_naive": "Baseline",
    "S1_position_swap": "Pos. Swap",
    "S2_ensemble_same": "Ensemble (Same)",
    "S3_ensemble_cross": "Ensemble (Cross)",
    "S4_calibrated_rubric": "Rubric",
    "S5_cot_forcing": "CoT",
    "S6_reference_guided": "Ref-Guided",
    "S7_combined_full": "Combined Full",
    "S8_combined_budget": "Combined Budget",
}

MODEL_LABELS = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-4": "Claude Sonnet 4",
    "gpt-4o": "GPT-4o",
    "llama-3.3-70b": "Llama 3.3-70B",
}

MT_BENCH_CATEGORIES = {
    **{qid: "writing" for qid in range(81, 91)},
    **{qid: "roleplay" for qid in range(91, 101)},
    **{qid: "reasoning" for qid in range(101, 111)},
    **{qid: "math" for qid in range(111, 121)},
    **{qid: "coding" for qid in range(121, 131)},
    **{qid: "extraction" for qid in range(131, 141)},
    **{qid: "stem" for qid in range(141, 151)},
    **{qid: "humanities" for qid in range(151, 161)},
}


def reaggregate_from_raw(
    model: str, strategy: str, benchmark: str, sample_size: int | None = None
) -> dict | None:
    """Rebuild aggregated metrics from raw results and current dataset.

    This fixes stale aggregated files by re-computing metrics against
    the current dataset (which may have changed sample size or IDs).
    """
    dl = DataLoader()
    bench_sizes = {"mt_bench": 400, "llmbar": 200}
    size = sample_size or bench_sizes.get(benchmark)
    try:
        instances = dl.load(benchmark, sample_size=size, seed=42)
    except FileNotFoundError:
        return None

    inst_map = {inst.id: inst for inst in instances}
    raw_dir = RESULTS_DIR / "raw"

    verdicts = []
    gold_labels = []
    total_cost = 0.0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for inst in instances:
        cache_path = raw_dir / f"{model}_{strategy}_{benchmark}_{inst.id}.json"
        if not cache_path.exists():
            continue
        with open(cache_path) as fh:
            raw = json.load(fh)
        result = raw["result"]
        verdicts.append(result["verdict"])
        gold_labels.append(inst.human_preference)
        total_cost += result.get("total_cost_usd", 0)
        total_latency += result.get("total_latency_ms", 0)
        total_input_tokens += result.get("total_input_tokens", 0)
        total_output_tokens += result.get("total_output_tokens", 0)

    if not verdicts:
        return None

    pairs = [(v, g) for v, g in zip(verdicts, gold_labels) if g is not None]
    agreement = sum(v == g for v, g in pairs) / len(pairs) if pairs else None

    agg = {
        "model": model,
        "strategy": strategy,
        "benchmark": benchmark,
        "n_instances": len(verdicts),
        "verdict_distribution": {
            "A": verdicts.count("A"),
            "B": verdicts.count("B"),
            "tie": verdicts.count("tie"),
        },
        "agreement_rate": agreement,
        "total_cost_usd": total_cost,
        "avg_latency_ms": total_latency / len(verdicts) if verdicts else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    # Save updated aggregated file
    agg_path = RESULTS_DIR / "aggregated" / f"{model}_{strategy}_{benchmark}_metrics.json"
    with open(agg_path, "w") as fh:
        json.dump(agg, fh, indent=2)

    return agg


def reaggregate_all() -> None:
    """Re-aggregate all existing model/strategy/benchmark combos from raw results."""
    models = list(MODEL_LABELS.keys())
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric",
                  "S5_cot_forcing", "S8_combined_budget"]
    benchmarks = ["mt_bench", "llmbar", "custom"]

    updated = 0
    for model in models:
        for strategy in strategies:
            for benchmark in benchmarks:
                result = reaggregate_from_raw(model, strategy, benchmark)
                if result:
                    updated += 1
                    print(
                        f"  {model}/{strategy}/{benchmark}: "
                        f"n={result['n_instances']}, "
                        f"agreement={result['agreement_rate']:.3f}"
                        if result['agreement_rate'] is not None
                        else f"  {model}/{strategy}/{benchmark}: n={result['n_instances']}"
                    )
    print(f"\nRe-aggregated {updated} configurations")


def load_all_aggregated() -> pd.DataFrame:
    """Load all aggregated results into a DataFrame."""
    rows = []
    agg_dir = RESULTS_DIR / "aggregated"
    for f in agg_dir.glob("*_metrics.json"):
        with open(f) as fh:
            data = json.load(fh)
        rows.append(data)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["model_label"] = df["model"].map(MODEL_LABELS)
        df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)
    return df


def load_raw_results(model: str, strategy: str, benchmark: str) -> list[dict]:
    """Load individual raw results for a configuration."""
    raw_dir = RESULTS_DIR / "raw"
    results = []
    for f in raw_dir.glob(f"{model}_{strategy}_{benchmark}_*.json"):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def compute_position_bias(model: str, benchmark: str) -> dict:
    """Compute position bias from raw results.

    For the custom dataset (position pairs), an unbiased judge should always say tie.
    We measure the fraction that say A vs B.
    """
    results = load_raw_results(model, "B0_naive", benchmark)
    if not results:
        return {"a_rate": 0, "b_rate": 0, "tie_rate": 0, "n": 0}

    verdicts = [r["result"]["verdict"] for r in results]
    n = len(verdicts)
    return {
        "a_rate": verdicts.count("A") / n,
        "b_rate": verdicts.count("B") / n,
        "tie_rate": verdicts.count("tie") / n,
        "n": n,
    }


def compute_verbosity_bias_from_custom(model: str, strategy: str) -> dict | None:
    """Compute verbosity bias from custom LENGTH pairs.

    Length-aware (not slot-aware): each verdict is mapped to whether the judge
    preferred the actually-longer response, using metadata.len_a vs metadata.len_b.
    The expanded version is in slot A in 34/50 pairs and slot B in 16/50, so a
    naive slot-A vs slot-B count would have the wrong sign on the 16 reversed pairs.
    """
    results = load_raw_results(model, strategy, "custom")
    if not results:
        return None

    dl = DataLoader()
    try:
        custom = dl.load("custom")
    except FileNotFoundError:
        return None

    # Build per-instance map of which slot holds the longer response
    longer_slot = {}  # instance_id -> "A" | "B" | None (if equal)
    for inst in custom:
        if inst.metadata.get("bias_type") != "length":
            continue
        len_a = inst.metadata.get("len_a")
        len_b = inst.metadata.get("len_b")
        if len_a is None or len_b is None:
            continue
        if len_a > len_b:
            longer_slot[inst.id] = "A"
        elif len_b > len_a:
            longer_slot[inst.id] = "B"
        else:
            longer_slot[inst.id] = None  # equal, exclude

    length_results = [r for r in results if r["instance_id"] in longer_slot and longer_slot[r["instance_id"]] is not None]
    if not length_results:
        return None

    n = len(length_results)
    prefers_longer = 0
    prefers_shorter = 0
    ties = 0
    for r in length_results:
        verdict = r["result"]["verdict"]
        longer = longer_slot[r["instance_id"]]
        if verdict == "tie":
            ties += 1
        elif verdict == longer:
            prefers_longer += 1
        else:
            prefers_shorter += 1

    longer_rate = prefers_longer / n
    shorter_rate = prefers_shorter / n
    tie_rate = ties / n

    return {
        "prefers_longer": longer_rate,
        "prefers_shorter": shorter_rate,
        "tie_rate": tie_rate,
        "verbosity_bias": longer_rate - shorter_rate,  # positive = prefers longer (length-aware)
        "n": n,
    }


def compute_position_bias_from_custom(model: str, strategy: str) -> dict | None:
    """Compute position bias from custom POSITION pairs (identical A=B)."""
    results = load_raw_results(model, strategy, "custom")
    if not results:
        return None

    dl = DataLoader()
    try:
        custom = dl.load("custom")
    except FileNotFoundError:
        return None

    pos_ids = {inst.id for inst in custom if inst.metadata.get("bias_type") == "position"}
    pos_results = [r for r in results if r["instance_id"] in pos_ids]
    if not pos_results:
        return None

    verdicts = [r["result"]["verdict"] for r in pos_results]
    n = len(verdicts)
    a_rate = verdicts.count("A") / n
    b_rate = verdicts.count("B") / n
    tie_rate = verdicts.count("tie") / n

    return {
        "prefers_first": a_rate,
        "prefers_second": b_rate,
        "tie_rate": tie_rate,
        "position_bias": a_rate - b_rate,  # positive = prefers first
        "n": n,
    }


def compute_style_bias_from_custom(model: str, strategy: str) -> dict | None:
    """Compute style bias from STYLE pairs, position-averaged across both orderings.

    STYLE pairs (markdown in slot A, prose in slot B): a verdict for A means prefers markdown.
    STYLE_BA pairs (prose in slot A, markdown in slot B): a verdict for B means prefers markdown.
    The style_bias score is computed as P(prefers markdown) - P(prefers prose),
    averaged across both orderings. This addresses the position-confound that
    Reviewer GR4A flagged ("Critical 4").
    """
    results = load_raw_results(model, strategy, "custom")
    if not results:
        return None

    dl = DataLoader()
    try:
        custom = dl.load("custom")
    except FileNotFoundError:
        return None

    inst_lookup = {inst.id: inst for inst in custom}

    prefers_markdown = 0
    prefers_prose = 0
    ties = 0
    n = 0
    for r in results:
        inst = inst_lookup.get(r["instance_id"])
        if inst is None:
            continue
        bias_type = inst.metadata.get("bias_type")
        verdict = r["result"]["verdict"]
        if bias_type == "style":
            # markdown in A, prose in B
            n += 1
            if verdict == "A":
                prefers_markdown += 1
            elif verdict == "B":
                prefers_prose += 1
            else:
                ties += 1
        elif bias_type == "style_ba":
            # prose in A, markdown in B
            n += 1
            if verdict == "B":
                prefers_markdown += 1
            elif verdict == "A":
                prefers_prose += 1
            else:
                ties += 1

    if n == 0:
        return None

    md_rate = prefers_markdown / n
    pr_rate = prefers_prose / n

    return {
        "prefers_markdown": md_rate,
        "prefers_prose": pr_rate,
        "tie_rate": ties / n,
        "style_bias": md_rate - pr_rate,  # positive = prefers markdown (position-averaged)
        "n": n,
        "n_style_only": sum(1 for r in results if inst_lookup.get(r["instance_id"]) and inst_lookup[r["instance_id"]].metadata.get("bias_type") == "style"),
        "n_style_ba": sum(1 for r in results if inst_lookup.get(r["instance_id"]) and inst_lookup[r["instance_id"]].metadata.get("bias_type") == "style_ba"),
    }


_FAMILY_KEYWORDS = {
    "gemini": ["gemini"],
    "claude": ["claude", "anthropic"],
    "llama": ["llama", "meta"],
    "openai": ["gpt", "openai"],
    "mistral": ["mistral", "mixtral"],
}


def _model_family(model_id: str) -> str | None:
    m = model_id.lower()
    for family, keywords in _FAMILY_KEYWORDS.items():
        if any(k in m for k in keywords):
            return family
    return None


def compute_self_preference_from_custom(model: str, strategy: str) -> dict | None:
    """Compute self-preference from MODEL_ORIGIN pairs.

    Uses both the legacy 50 Gemini-vs-Claude pairs (model_origin) and the new 100
    round-robin pairs (model_origin_rr). For the legacy pairs, only Gemini and
    Claude judges have a same-family option. For the round-robin pairs, every
    judge family has same-family pairs (per Reviewer p6d3 RC1 and GR4A Critical 3).

    Self-preference = P(prefers own family) on pairs where exactly one of the two
    responders shares the judge's family.
    """
    results = load_raw_results(model, strategy, "custom")
    if not results:
        return None

    dl = DataLoader()
    try:
        custom = dl.load("custom")
    except FileNotFoundError:
        return None

    judge_family = _model_family(model)
    inst_lookup = {inst.id: inst for inst in custom}

    legacy_results = []
    rr_same_family_pairs = []  # (verdict, slot_with_own_family)
    rr_cross_family_pairs = []  # (verdict, slot_with_a_model)
    for r in results:
        inst = inst_lookup.get(r["instance_id"])
        if inst is None:
            continue
        bias_type = inst.metadata.get("bias_type")
        verdict = r["result"]["verdict"]
        if bias_type == "model_origin":
            legacy_results.append(verdict)
        elif bias_type == "model_origin_rr":
            model_in_a = inst.metadata.get("model_a", "")
            model_in_b = inst.metadata.get("model_b", "")
            fam_a = _model_family(model_in_a)
            fam_b = _model_family(model_in_b)
            a_is_own = fam_a == judge_family
            b_is_own = fam_b == judge_family
            if a_is_own and not b_is_own:
                rr_same_family_pairs.append((verdict, "A"))
            elif b_is_own and not a_is_own:
                rr_same_family_pairs.append((verdict, "B"))
            elif not a_is_own and not b_is_own:
                rr_cross_family_pairs.append((verdict, "A"))
            # if both own (e.g., Gemini Pro judging Gemini Flash vs Pro), skip

    # Compute self-preference from round-robin pairs (preferred metric)
    if rr_same_family_pairs:
        prefers_own = sum(1 for v, slot in rr_same_family_pairs if v == slot)
        prefers_other = sum(1 for v, slot in rr_same_family_pairs if v != slot and v != "tie")
        ties = sum(1 for v, slot in rr_same_family_pairs if v == "tie")
        n_rr = len(rr_same_family_pairs)
        own_rate = prefers_own / n_rr
        other_rate = prefers_other / n_rr
        rr_self_pref = own_rate - other_rate
    else:
        rr_self_pref = None
        n_rr = 0
        own_rate = 0
        other_rate = 0

    if not legacy_results:
        return {
            "round_robin_self_preference": rr_self_pref,
            "n_round_robin_same_family": n_rr,
            "round_robin_own_win_rate": own_rate,
            "round_robin_other_win_rate": other_rate,
            "self_preference": rr_self_pref if rr_self_pref is not None else 0,
            "n": n_rr,
        }

    # Legacy: A is gemini, B is claude (for backward compat with Figure 1 etc.)
    verdicts = legacy_results
    n = len(verdicts)
    a_rate = verdicts.count("A") / n  # prefers gemini
    b_rate = verdicts.count("B") / n  # prefers claude

    is_gemini_judge = judge_family == "gemini"
    is_claude_judge = judge_family == "claude"
    if is_gemini_judge:
        legacy_self_pref = a_rate - b_rate
    elif is_claude_judge:
        legacy_self_pref = b_rate - a_rate
    else:
        legacy_self_pref = None  # not interpretable as self-preference for non-Gemini, non-Claude judges
    own_pref = a_rate if is_gemini_judge else (b_rate if is_claude_judge else 0)

    # Prefer round-robin score when available (works for all judges); fall back
    # to legacy score (only meaningful for Gemini and Claude judges).
    final_self_pref = rr_self_pref if rr_self_pref is not None else legacy_self_pref

    return {
        "prefers_gemini": a_rate,
        "prefers_claude": b_rate,
        "tie_rate": verdicts.count("tie") / n,
        "legacy_self_preference": legacy_self_pref,
        "round_robin_self_preference": rr_self_pref,
        "n_legacy": n,
        "n_round_robin_same_family": n_rr,
        "round_robin_own_win_rate": own_rate,
        "round_robin_other_win_rate": other_rate,
        "self_preference": final_self_pref if final_self_pref is not None else 0,
        "n": n + n_rr,
    }


# ============================================================
# Figure Generation
# ============================================================


def figure1_bias_heatmap():
    """Figure 1: Heatmap of baseline bias values (models x bias types).

    Reports SIGNED bias scores so the direction of preference is visible.
    The magnitude can be read from the absolute value; positive and negative
    are distinguished by color (RdBu diverging colormap centered at 0).
    """
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    bias_types = ["Position", "Verbosity", "Style", "Self-Pref"]

    data = np.zeros((len(models), len(bias_types)))
    for i, model in enumerate(models):
        pos = compute_position_bias_from_custom(model, "B0_naive")
        verb = compute_verbosity_bias_from_custom(model, "B0_naive")
        style = compute_style_bias_from_custom(model, "B0_naive")
        self_pref = compute_self_preference_from_custom(model, "B0_naive")

        # Signed values; positive = first/longer/markdown/own-family preference
        data[i, 0] = pos["position_bias"] if pos else 0
        data[i, 1] = verb["verbosity_bias"] if verb else 0
        data[i, 2] = style["style_bias"] if style else 0
        data[i, 3] = self_pref["self_preference"] if self_pref else 0

    fig, ax = plt.subplots(figsize=(5, 3))
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    sns.heatmap(
        data,
        xticklabels=bias_types,
        yticklabels=model_labels,
        annot=True,
        fmt="+.2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        cbar_kws={"label": "Bias Score (signed)"},
    )
    ax.set_title("Baseline Bias Scores by Model (B0 Naive)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_bias_heatmap.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig1_bias_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Generated Figure 1: Bias heatmap (signed values)")


def figure2_strategy_comparison():
    """Figure 2: Strategy effectiveness across bias types."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b"]
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric", "S5_cot_forcing", "S8_combined_budget"]

    # Collect bias reductions for each strategy
    bias_data = defaultdict(lambda: defaultdict(list))

    for model in models:
        for strategy in strategies:
            pos = compute_position_bias_from_custom(model, strategy)
            verb = compute_verbosity_bias_from_custom(model, strategy)
            style = compute_style_bias_from_custom(model, strategy)

            if pos:
                bias_data[strategy]["Position"].append(abs(pos["position_bias"]))
            if verb:
                bias_data[strategy]["Verbosity"].append(abs(verb["verbosity_bias"]))
            if style:
                bias_data[strategy]["Style"].append(abs(style["style_bias"]))

    # Build grouped bar chart
    fig, ax = plt.subplots(figsize=(6.75, 3.5))
    bias_types = ["Position", "Verbosity", "Style"]
    x = np.arange(len(bias_types))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    for j, strategy in enumerate(strategies):
        means = [np.mean(bias_data[strategy][bt]) if bias_data[strategy][bt] else 0 for bt in bias_types]
        ax.bar(x + j * width, means, width, label=STRATEGY_LABELS.get(strategy, strategy), color=colors[j])

    ax.set_ylabel("Bias Magnitude (lower is better)")
    ax.set_title("Bias Reduction by Strategy (averaged across frontier models)")
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(bias_types)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_strategy_comparison.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig2_strategy_comparison.png", bbox_inches="tight")
    plt.close()
    print("Generated Figure 2: Strategy comparison")


def figure3_cross_bias_interactions():
    """Figure 3: Cross-bias interaction heatmap.

    For each strategy, shows the change in ALL bias types relative to baseline.
    """
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b"]
    strategies = ["S1_position_swap", "S4_calibrated_rubric", "S5_cot_forcing", "S8_combined_budget"]
    bias_types = ["Position", "Verbosity", "Style"]

    # Get baseline biases
    baseline = defaultdict(list)
    for model in models:
        pos = compute_position_bias_from_custom(model, "B0_naive")
        verb = compute_verbosity_bias_from_custom(model, "B0_naive")
        style = compute_style_bias_from_custom(model, "B0_naive")
        if pos: baseline["Position"].append(abs(pos["position_bias"]))
        if verb: baseline["Verbosity"].append(abs(verb["verbosity_bias"]))
        if style: baseline["Style"].append(abs(style["style_bias"]))

    baseline_means = {bt: np.mean(v) if v else 0 for bt, v in baseline.items()}

    # Compute deltas
    delta = np.zeros((len(strategies), len(bias_types)))
    for i, strategy in enumerate(strategies):
        strat_biases = defaultdict(list)
        for model in models:
            pos = compute_position_bias_from_custom(model, strategy)
            verb = compute_verbosity_bias_from_custom(model, strategy)
            style = compute_style_bias_from_custom(model, strategy)
            if pos: strat_biases["Position"].append(abs(pos["position_bias"]))
            if verb: strat_biases["Verbosity"].append(abs(verb["verbosity_bias"]))
            if style: strat_biases["Style"].append(abs(style["style_bias"]))

        for j, bt in enumerate(bias_types):
            strat_mean = np.mean(strat_biases[bt]) if strat_biases[bt] else 0
            delta[i, j] = strat_mean - baseline_means.get(bt, 0)  # negative = reduced bias

    fig, ax = plt.subplots(figsize=(5, 3))
    strat_labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    sns.heatmap(
        delta,
        xticklabels=bias_types,
        yticklabels=strat_labels,
        annot=True,
        fmt="+.2f",
        cmap="RdYlGn_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Bias Change (negative = improvement)"},
    )
    ax.set_title("Cross-Bias Interactions: Change vs Baseline")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_cross_bias.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_cross_bias.png", bbox_inches="tight")
    plt.close()
    print("Generated Figure 3: Cross-bias interactions")


def figure4_cost_accuracy():
    """Figure 4: Cost vs accuracy Pareto frontier."""
    df = load_all_aggregated()
    if df.empty:
        print("No aggregated results found, skipping Figure 4")
        return

    # Focus on MT-Bench (has human labels)
    mt = df[df["benchmark"] == "mt_bench"].copy()
    if mt.empty:
        print("No MT-Bench results, skipping Figure 4")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Distinct markers per model for clarity
    model_markers = {
        "Gemini 2.5 Pro": "o",
        "Claude Sonnet 4": "s",
        "GPT-4o": "D",
        "Llama 3.3-70B": "^",
        "Gemini 2.5 Flash": "v",
    }
    strat_list = sorted(mt["strategy"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(strat_list)))
    strat_colors = dict(zip(strat_list, colors))

    for _, row in mt.iterrows():
        if row["agreement_rate"] is None:
            continue
        color = strat_colors.get(row["strategy"], "gray")
        cost = row["total_cost_usd"] / max(row["n_instances"], 1)
        marker = model_markers.get(row.get("model_label", ""), "o")
        ax.scatter(cost, row["agreement_rate"], c=[color], s=70, zorder=5, marker=marker, edgecolors="black", linewidths=0.3)

    # Use adjustText if available, otherwise offset manually
    from matplotlib.lines import Line2D
    strat_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=strat_colors[s],
               markersize=8, label=STRATEGY_LABELS.get(s, s))
        for s in strat_list if s in strat_colors
    ]
    model_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
               markersize=8, label=name, markeredgecolor="black", markeredgewidth=0.3)
        for name, m in model_markers.items()
        if name in mt["model_label"].values
    ]
    leg1 = ax.legend(handles=strat_handles, loc="lower right", fontsize=7, title="Strategy", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="upper left", fontsize=7, title="Model", title_fontsize=7)

    ax.set_xlabel("Cost per evaluation (USD)")
    ax.set_ylabel("Human agreement rate")
    ax.set_title("Cost vs Accuracy Tradeoff (MT-Bench)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_cost_accuracy.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig4_cost_accuracy.png", bbox_inches="tight")
    plt.close()
    print("Generated Figure 4: Cost vs accuracy")


def figure5_mt_bench_per_category():
    """Figure 5: Per-category MT-Bench agreement rates for baseline strategy."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    categories = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]

    data = {cat: {} for cat in categories}

    for model in models:
        results = load_raw_results(model, "B0_naive", "mt_bench")
        if not results:
            continue

        # Group by category using question_id from instance_id
        cat_verdicts = defaultdict(list)
        cat_golds = defaultdict(list)

        dl = DataLoader()
        instances = dl.load("mt_bench", sample_size=400, seed=42)
        inst_map = {inst.id: inst for inst in instances}

        for r in results:
            inst = inst_map.get(r["instance_id"])
            if not inst or not inst.human_preference:
                continue
            qid = inst.metadata.get("question_id", 0)
            cat = MT_BENCH_CATEGORIES.get(qid, "")
            if cat:
                verdict = r["result"]["verdict"]
                cat_verdicts[cat].append(verdict)
                cat_golds[cat].append(inst.human_preference)

        label = MODEL_LABELS.get(model, model)
        for cat in categories:
            if cat in cat_verdicts:
                agreements = sum(
                    v == g for v, g in zip(cat_verdicts[cat], cat_golds[cat])
                )
                n = len(cat_verdicts[cat])
                data[cat][label] = agreements / n if n > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(categories))
    width = 0.15
    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(models)))

    for j, label in enumerate(model_labels):
        values = [data[cat].get(label, 0) for cat in categories]
        ax.bar(x + j * width, values, width, label=label, color=colors[j])

    ax.set_ylabel("Human Agreement Rate")
    ax.set_title("MT-Bench Agreement by Category (B0 Baseline)")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([c.capitalize() for c in categories], rotation=30, ha="right")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_mt_bench_categories.pdf", bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_mt_bench_categories.png", bbox_inches="tight")
    plt.close()
    print("Generated Figure 5: MT-Bench per-category")


def compute_truncation_bias(model: str, strategy: str) -> dict | None:
    """Compute truncation-based verbosity bias.

    For truncation pairs, A is the long (complete) response, expected verdict is A.
    If the judge prefers B (shorter/truncated), that's conciseness bias.
    """
    results = load_raw_results(model, strategy, "custom")
    if not results:
        return None

    dl = DataLoader()
    try:
        custom = dl.load("custom")
    except FileNotFoundError:
        return None

    trunc_ids = {inst.id for inst in custom if inst.metadata.get("bias_type") == "length_truncated"}
    trunc_results = [r for r in results if r["instance_id"] in trunc_ids]
    if not trunc_results:
        return None

    verdicts = [r["result"]["verdict"] for r in trunc_results]
    n = len(verdicts)
    a_rate = verdicts.count("A") / n  # correct (prefers complete)
    b_rate = verdicts.count("B") / n  # conciseness bias

    return {
        "prefers_complete": a_rate,
        "prefers_truncated": b_rate,
        "tie_rate": verdicts.count("tie") / n,
        "accuracy": a_rate,
        "n": n,
    }


def table3_length_split():
    """Table 3: LENGTH results split by expansion vs truncation direction."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    strategies = ["B0_naive", "S5_cot_forcing", "S8_combined_budget"]

    rows = []
    for model in models:
        for strategy in strategies:
            # Expansion-based
            verb = compute_verbosity_bias_from_custom(model, strategy)
            # Truncation-based
            trunc = compute_truncation_bias(model, strategy)

            rows.append({
                "Model": MODEL_LABELS.get(model, model),
                "Strategy": STRATEGY_LABELS.get(strategy, strategy),
                "Expansion Bias": f"{verb['verbosity_bias']:+.2f}" if verb else "--",
                "Trunc. Accuracy": f"{trunc['accuracy']:.2f}" if trunc else "--",
                "Trunc. n": trunc["n"] if trunc else 0,
            })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(FIGURES_DIR / "table3_length_split.csv", index=False)
    print("Generated Table 3: LENGTH split")
    print(table_df.to_string(index=False))
    return table_df


def compute_agreement_with_ci(
    model: str, strategy: str, benchmark: str, sample_size: int | None = None
) -> dict | None:
    """Compute agreement rate with bootstrap 95% CI and Cohen's kappa."""
    dl = DataLoader()
    bench_sizes = {"mt_bench": 400, "llmbar": 200}
    size = sample_size or bench_sizes.get(benchmark)
    try:
        instances = dl.load(benchmark, sample_size=size, seed=42)
    except FileNotFoundError:
        return None

    inst_map = {inst.id: inst for inst in instances}
    results = load_raw_results(model, strategy, benchmark)
    if not results:
        return None

    result_map = {r["instance_id"]: r for r in results}

    predictions = []
    golds = []
    for inst in instances:
        if inst.id in result_map and inst.human_preference is not None:
            predictions.append(result_map[inst.id]["result"]["verdict"])
            golds.append(inst.human_preference)

    if not predictions:
        return None

    from src.metrics.agreement import human_agreement_rate as har, cohens_kappa as ck
    agreement_info = har(predictions, golds, bootstrap_n=2000)
    kappa = ck(predictions, golds)

    return {
        "agreement": agreement_info["agreement"],
        "ci_lower": agreement_info["ci_lower"],
        "ci_upper": agreement_info["ci_upper"],
        "kappa": kappa,
        "n": len(predictions),
    }


def compute_mcnemar(
    model: str,
    strategy_a: str,
    strategy_b: str,
    benchmark: str,
    sample_size: int | None = None,
) -> dict | None:
    """McNemar's test between two strategies on the same instances."""
    dl = DataLoader()
    bench_sizes = {"mt_bench": 400, "llmbar": 200}
    size = sample_size or bench_sizes.get(benchmark)
    try:
        instances = dl.load(benchmark, sample_size=size, seed=42)
    except FileNotFoundError:
        return None

    results_a = {r["instance_id"]: r for r in load_raw_results(model, strategy_a, benchmark)}
    results_b = {r["instance_id"]: r for r in load_raw_results(model, strategy_b, benchmark)}

    # Count discordant pairs
    b_correct_a_wrong = 0  # strategy_b correct, strategy_a wrong
    a_correct_b_wrong = 0  # strategy_a correct, strategy_b wrong

    for inst in instances:
        if inst.human_preference is None:
            continue
        if inst.id not in results_a or inst.id not in results_b:
            continue
        verdict_a = results_a[inst.id]["result"]["verdict"]
        verdict_b = results_b[inst.id]["result"]["verdict"]
        correct_a = verdict_a == inst.human_preference
        correct_b = verdict_b == inst.human_preference

        if correct_b and not correct_a:
            b_correct_a_wrong += 1
        elif correct_a and not correct_b:
            a_correct_b_wrong += 1

    n_discordant = b_correct_a_wrong + a_correct_b_wrong
    if n_discordant == 0:
        return {"chi2": 0, "p_value": 1.0, "n_discordant": 0}

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2 / n_discordant
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "n_discordant": n_discordant,
        "b_better": b_correct_a_wrong,
        "a_better": a_correct_b_wrong,
    }


def table1_main_results():
    """Table 1: Main results table with bootstrap CIs and Cohen's kappa."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric",
                  "S5_cot_forcing", "S8_combined_budget"]

    rows = []
    for model in models:
        row = {"Model": MODEL_LABELS.get(model, model)}
        for strategy in strategies:
            info = compute_agreement_with_ci(model, strategy, "mt_bench")
            if info:
                cell = f"{info['agreement']:.3f} [{info['ci_lower']:.3f}, {info['ci_upper']:.3f}]"
                row[STRATEGY_LABELS.get(strategy, strategy)] = cell
            else:
                row[STRATEGY_LABELS.get(strategy, strategy)] = "--"
        rows.append(row)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(FIGURES_DIR / "table1_main_results.csv", index=False)

    # Also generate kappa table
    kappa_rows = []
    for model in models:
        row = {"Model": MODEL_LABELS.get(model, model)}
        for strategy in strategies:
            info = compute_agreement_with_ci(model, strategy, "mt_bench")
            row[STRATEGY_LABELS.get(strategy, strategy)] = (
                f"{info['kappa']:.3f}" if info else "--"
            )
        kappa_rows.append(row)

    kappa_df = pd.DataFrame(kappa_rows)
    kappa_df.to_csv(FIGURES_DIR / "table1_kappa.csv", index=False)

    # McNemar's test: B0 vs S8 for each model
    print("\nMcNemar's test (B0 vs S8 on MT-Bench):")
    for model in models:
        result = compute_mcnemar(model, "B0_naive", "S8_combined_budget", "mt_bench")
        if result:
            sig = "*" if result["p_value"] < 0.05 else ""
            print(
                f"  {MODEL_LABELS.get(model, model)}: "
                f"chi2={result['chi2']:.2f}, p={result['p_value']:.4f}{sig}, "
                f"n_discordant={result['n_discordant']}"
            )

    print("\nGenerated Table 1: Main results with CIs")
    print(table_df.to_string(index=False))


def table2_custom_bias_summary():
    """Table 2: Bias measurements on custom controlled dataset."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric", "S5_cot_forcing", "S8_combined_budget"]

    rows = []
    for model in models:
        for strategy in strategies:
            pos = compute_position_bias_from_custom(model, strategy)
            verb = compute_verbosity_bias_from_custom(model, strategy)
            style = compute_style_bias_from_custom(model, strategy)

            rows.append({
                "Model": MODEL_LABELS.get(model, model),
                "Strategy": STRATEGY_LABELS.get(strategy, strategy),
                "Position Bias": f"{pos['position_bias']:+.2f}" if pos else "--",
                "Verbosity Bias": f"{verb['verbosity_bias']:+.2f}" if verb else "--",
                "Style Bias": f"{style['style_bias']:+.2f}" if style else "--",
            })

    table_df = pd.DataFrame(rows)
    latex = table_df.to_latex(
        index=False,
        caption="Bias measurements on custom controlled dataset. Values show preference direction (positive = prefers first/longer/markdown).",
        label="tab:bias_summary",
    )
    with open(FIGURES_DIR / "table2_bias_summary.tex", "w") as f:
        f.write(latex)
    table_df.to_csv(FIGURES_DIR / "table2_bias_summary.csv", index=False)
    print("Generated Table 2: Bias summary")
    print(table_df.to_string(index=False))


def table_llmbar_results():
    """LLMBar agreement results table."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric",
                  "S5_cot_forcing", "S8_combined_budget"]

    rows = []
    for model in models:
        row = {"Model": MODEL_LABELS.get(model, model)}
        for strategy in strategies:
            info = compute_agreement_with_ci(model, strategy, "llmbar")
            if info:
                cell = f"{info['agreement']:.3f} [{info['ci_lower']:.3f}, {info['ci_upper']:.3f}]"
                row[STRATEGY_LABELS.get(strategy, strategy)] = cell
            else:
                row[STRATEGY_LABELS.get(strategy, strategy)] = "--"
        rows.append(row)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(FIGURES_DIR / "table_llmbar_results.csv", index=False)
    print("Generated LLMBar results table")
    print(table_df.to_string(index=False))


def table_llmbar_with_stats():
    """LLMBar agreement with bootstrap CIs and McNemar tests (per reviewer p6d3)."""
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric",
                  "S5_cot_forcing", "S8_combined_budget"]

    rows = []
    for model in models:
        row = {"Model": MODEL_LABELS.get(model, model)}
        for strategy in strategies:
            info = compute_agreement_with_ci(model, strategy, "llmbar")
            if info:
                row[STRATEGY_LABELS.get(strategy, strategy)] = (
                    f"{info['agreement']:.3f} [{info['ci_lower']:.3f}, {info['ci_upper']:.3f}]"
                )
            else:
                row[STRATEGY_LABELS.get(strategy, strategy)] = "--"
        rows.append(row)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(FIGURES_DIR / "table_llmbar_with_ci.csv", index=False)
    print("\nLLMBar agreement with CIs:")
    print(table_df.to_string(index=False))

    # McNemar tests for B0 vs every other strategy on LLMBar
    print("\nMcNemar tests on LLMBar (B0 vs each strategy):")
    p_values = []
    for model in models:
        for strategy in strategies:
            if strategy == "B0_naive":
                continue
            result = compute_mcnemar(model, "B0_naive", strategy, "llmbar")
            if result:
                p_values.append((MODEL_LABELS.get(model, model), STRATEGY_LABELS.get(strategy, strategy), result))

    # Holm-Bonferroni correction across all LLMBar McNemar tests
    sorted_pvals = sorted(p_values, key=lambda x: x[2]["p_value"])
    m = len(sorted_pvals)
    for rank, (model_name, strat_name, result) in enumerate(sorted_pvals, start=1):
        adjusted_threshold = 0.05 / (m - rank + 1)
        sig = "*" if result["p_value"] < adjusted_threshold else ""
        print(
            f"  {model_name} / {strat_name}: chi2={result['chi2']:.2f}, "
            f"p={result['p_value']:.4f} (Holm threshold={adjusted_threshold:.4f}){sig}"
        )


def mixed_effects_regression():
    """Mixed-effects logistic regression with instance random effects (per reviewer p6d3 RC2).

    Model: judge_correct ~ strategy + (1|instance_id) per model, then aggregated.
    Replaces the sign test that p6d3 critiqued for non-independent observations.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("statsmodels not installed; skipping mixed-effects regression")
        return

    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    strategies = ["B0_naive", "S1_position_swap", "S4_calibrated_rubric",
                  "S5_cot_forcing", "S8_combined_budget"]

    dl = DataLoader()
    instances = dl.load("mt_bench", sample_size=400, seed=42)
    inst_lookup = {inst.id: inst for inst in instances}

    # Build long-format dataframe: one row per (judge, strategy, instance) with correctness
    rows = []
    for model in models:
        for strategy in strategies:
            results = load_raw_results(model, strategy, "mt_bench")
            for r in results:
                inst = inst_lookup.get(r["instance_id"])
                if not inst or inst.human_preference is None:
                    continue
                rows.append({
                    "judge": model,
                    "strategy": strategy,
                    "instance_id": r["instance_id"],
                    "correct": int(r["result"]["verdict"] == inst.human_preference),
                })

    if not rows:
        print("No data for mixed-effects regression")
        return

    df = pd.DataFrame(rows)
    print(f"\nMixed-effects regression on {len(df)} judge-strategy-instance observations")

    # Per-model models with instance random effects
    print("\nPer-model: judge_correct ~ C(strategy, Treatment('B0_naive')) + (1|instance_id)")
    for model in models:
        sub = df[df["judge"] == model]
        if len(sub) < 100:
            print(f"  {MODEL_LABELS.get(model, model)}: insufficient data ({len(sub)} obs)")
            continue
        try:
            md = smf.mixedlm(
                "correct ~ C(strategy, Treatment('B0_naive'))",
                data=sub,
                groups=sub["instance_id"],
            )
            mdf = md.fit(method="lbfgs", reml=False, disp=False)
            print(f"\n  {MODEL_LABELS.get(model, model)} (n={len(sub)}):")
            for name, coef, pval in zip(mdf.params.index, mdf.params, mdf.pvalues):
                if "C(strategy" in name:
                    strat_label = name.split(".")[1].rstrip("]")
                    sig = "*" if pval < 0.05 else ""
                    print(f"    {strat_label}: coef={coef:+.3f}, p={pval:.4f}{sig}")
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"  {MODEL_LABELS.get(model, model)}: regression failed ({e})")


def investigate_tie_discrepancy():
    """Investigate why S8 has fewer ties than S1 despite both using position swap.

    Per reviewer GR4A's question. Hypothesis: the merged CoT+rubric prompt in S8
    produces more consistent verdicts under position swap, so fewer disagreements
    trigger the tie-on-disagreement rule.
    """
    print("\n=== Tie-on-disagreement diagnostic (S1 vs S8) ===")
    models = ["gemini-2.5-pro", "claude-sonnet-4", "gpt-4o", "llama-3.3-70b", "gemini-2.5-flash"]
    for model in models:
        for strategy in ["S1_position_swap", "S8_combined_budget"]:
            agg_path = RESULTS_DIR / "aggregated" / f"{model}_{strategy}_mt_bench_metrics.json"
            if not agg_path.exists():
                continue
            with open(agg_path) as f:
                agg = json.load(f)
            vd = agg.get("verdict_distribution", {})
            n = sum(vd.values())
            tie_pct = vd.get("tie", 0) / n if n else 0
            print(f"  {MODEL_LABELS.get(model, model)} / {STRATEGY_LABELS.get(strategy, strategy)}: "
                  f"ties={vd.get('tie', 0)}/{n} ({tie_pct:.1%})")


def reaggregate_all_extended() -> None:
    """Re-aggregate everything including S2/S3/S6/S7 ensembles where data exists."""
    models = list(MODEL_LABELS.keys())
    strategies = ["B0_naive", "S1_position_swap", "S2_ensemble_same",
                  "S3_ensemble_cross", "S4_calibrated_rubric",
                  "S5_cot_forcing", "S6_reference_guided",
                  "S7_combined_full", "S8_combined_budget"]
    benchmarks = ["mt_bench", "llmbar", "custom"]

    updated = 0
    for model in models:
        for strategy in strategies:
            for benchmark in benchmarks:
                result = reaggregate_from_raw(model, strategy, benchmark)
                if result:
                    updated += 1
                    if result.get("agreement_rate") is not None:
                        print(f"  {model}/{strategy}/{benchmark}: n={result['n_instances']}, "
                              f"agreement={result['agreement_rate']:.3f}")
                    else:
                        print(f"  {model}/{strategy}/{benchmark}: n={result['n_instances']}")
    print(f"\nRe-aggregated {updated} configurations (including S2/S3/S6/S7 where present)")


def generate_all():
    """Generate all figures and tables."""
    print("=" * 60)
    print("ANALYSIS: Generating figures and tables")
    print("=" * 60)

    # Check available data
    agg_dir = RESULTS_DIR / "aggregated"
    n_files = len(list(agg_dir.glob("*_metrics.json")))
    raw_dir = RESULTS_DIR / "raw"
    n_raw = len(list(raw_dir.glob("*.json")))
    print(f"\nFound {n_files} aggregated configs, {n_raw} raw results\n")

    # Re-aggregate from raw results first (extended to include S2/S3/S6/S7)
    print("Re-aggregating from raw results (including S2/S3/S6/S7)...")
    reaggregate_all_extended()
    print()

    figure1_bias_heatmap()
    figure2_strategy_comparison()
    figure3_cross_bias_interactions()
    figure4_cost_accuracy()
    figure5_mt_bench_per_category()
    print()
    table1_main_results()
    print()
    table_llmbar_results()
    print()
    table_llmbar_with_stats()
    print()
    table2_custom_bias_summary()
    print()
    table3_length_split()
    print()
    investigate_tie_discrepancy()
    print()
    mixed_effects_regression()

    print(f"\nAll outputs saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all()
