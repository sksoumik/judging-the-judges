# Judging the Judges: LLM-as-a-Judge Bias and Reliability

A systematic evaluation of bias mitigation strategies in LLM-as-a-Judge pipelines.

## What This Project Does

LLM-as-a-Judge has become the standard way to evaluate LLM outputs (MT-Bench, AlpacaEval, Chatbot Arena). But LLM judges have systematic biases: they prefer responses in certain positions, favor longer responses, and show preference for outputs from their own model family.

This project measures those biases and tests 9 different debiasing strategies across multiple judge models and benchmarks in a unified framework. The goal is to answer:

1. **Which debiasing strategies actually work** for each type of bias?
2. **Do fixes for one bias make another worse?** (cross-bias interactions)
3. **What is the cost/accuracy tradeoff** of each strategy?
4. **Can we build a practical meta-debiasing pipeline** that works well across the board?

## Research Questions

- **RQ1:** How do different bias mitigation strategies compare in reducing position, verbosity, and self-preference biases across LLM judge models?
- **RQ2:** Do mitigation strategies that work for one bias type inadvertently worsen another?
- **RQ3:** What is the cost-accuracy tradeoff of each strategy?
- **RQ4:** Can we construct a simple "meta-debiasing" pipeline that outperforms any individual technique?

## Experiment Matrix

### Judge Models (4 active)

| Model | Provider | Category | Cost (input/output per 1K tokens) |
|-------|----------|----------|-----------------------------------|
| gemini-2.5-pro | Vertex AI | Frontier | $0.00125 / $0.005 |
| gemini-2.5-flash | Vertex AI | Mid-tier | $0.00015 / $0.0006 |
| gemini-2.5-flash-lite | Vertex AI | Budget | Free tier |
| claude-sonnet-4 | Vertex AI (Anthropic) | Frontier | $0.003 / $0.015 |

Open-source models (Llama, Gemma, Mistral) can be added via Vertex AI Model Garden endpoints. See `configs/models.yaml`.

### Debiasing Strategies (9 total)

| ID | Strategy | How it works | Cost multiplier |
|----|----------|-------------|-----------------|
| B0 | Baseline (Naive) | Single judge call, fixed position, default prompt | 1x |
| S1 | Position Swap | Run twice with swapped A/B positions, check consistency | 2x |
| S2 | Same-Family Ensemble | 3 calls at temperatures 0.0, 0.3, 0.7, majority vote | 3x |
| S3 | Cross-Family Ensemble | 3 different models (Gemini Pro + Claude + Gemini Flash), majority vote | 3x |
| S4 | Calibrated Rubric | Detailed 5-criteria rubric (accuracy, relevance, completeness, clarity, reasoning) | 1x |
| S5 | Chain-of-Thought | Force step-by-step analysis before verdict | 1x |
| S6 | Reference-Guided | Provide gold-standard reference answer for anchoring | 1x |
| S7 | Combined Full | S1 + S3 + S4 (position swap + cross-family ensemble + rubric) | 6x |
| S8 | Combined Budget | S1 + S5 + S4 (position swap + CoT + rubric) | 2x |

### Benchmarks (5 total)

| Benchmark | Size | What it tests |
|-----------|------|---------------|
| MT-Bench | 3,355 instances (sample 80) | Gold-standard pairwise preference with human labels |
| LLMBar | 419 instances (sample 200) | Robustness to adversarial distractors |
| AlpacaEval | 805 instances (sample 200) | Large-scale instruction following |
| FairEval | ~200 instances | Position bias (position-swapped annotations) |
| Custom Controlled | 200 instances | **Our novel contribution**: synthetic pairs with controlled bias triggers |

### Custom Controlled Dataset

200 pairs across 4 categories (50 each), where an unbiased judge should always say "tie":

- **LENGTH**: Same quality response, one is ~2.8x longer (tests verbosity bias)
- **POSITION**: Identical responses in both A and B slots (tests position bias)
- **STYLE**: Same content in markdown vs plain prose (tests style/formatting bias)
- **MODEL_ORIGIN**: Parallel answers from Gemini Pro vs Claude Sonnet (tests self-preference bias)

### Full Matrix

**4 models x 9 strategies x 5 benchmarks = 180 configurations**

With sampling: ~70,400 total API calls, estimated cost ~$336.

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Human Agreement Rate | % match with gold-standard human preferences |
| Cohen's Kappa | Agreement adjusted for chance |
| Position Consistency Rate (PCR) | % consistent verdicts when A/B positions are swapped |
| Verbosity Correlation | Spearman rho between response length and judge score |
| Self-Preference Score (SPS) | Excess win rate when judging own model's outputs |
| Cost per Evaluation | USD and latency per judgment |

All metrics include 95% confidence intervals via bootstrap (n=1000).

## Setup

```bash
# Prerequisites: Python 3.13, uv, gcloud CLI

# Install dependencies
uv sync

# Authenticate with GCP
gcloud auth application-default login

# Configure
cp .env.example .env
# Edit .env with your GOOGLE_PROJECT_ID
```

## Running Experiments

### Quick Test (verify everything works)

```bash
python scripts/run_single.py \
  --model gemini-2.5-flash \
  --strategy B0_naive \
  --benchmark custom \
  --sample-size 5
```

### Dry Run (estimate cost before committing)

```bash
python scripts/run_experiments.py --dry-run
```

### Priority 1: Baseline Bias Measurement

Measures raw bias levels for all models before any mitigation. This is the foundation for all analysis.

```bash
# ~1,120 API calls, ~$5-10
python scripts/run_experiments.py \
  --models gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite,claude-sonnet-4 \
  --strategies B0_naive \
  --benchmarks mt_bench,custom \
  --sample-size 80
```

**What this produces:** Baseline bias measurements (PCR, verbosity correlation, self-preference) for every model. This tells us how biased each model is out of the box.

### Priority 2: All Strategies on MT-Bench (Frontier Models)

Tests every debiasing strategy on the two strongest models. Core results table for the paper.

```bash
# ~3,200 API calls, ~$30-50
python scripts/run_experiments.py \
  --models gemini-2.5-pro,claude-sonnet-4 \
  --strategies B0_naive,S1_position_swap,S2_ensemble_same,S3_ensemble_cross,S4_calibrated_rubric,S5_cot_forcing,S6_reference_guided,S7_combined_full,S8_combined_budget \
  --benchmarks mt_bench \
  --sample-size 80
```

**What this produces:** Strategy comparison table showing which strategies reduce which biases. The main results figure of the paper.

### Priority 3: Expand to Other Benchmarks

Tests generalization across LLMBar and AlpacaEval.

```bash
# ~16,000 API calls, ~$80-120
python scripts/run_experiments.py \
  --models gemini-2.5-pro,claude-sonnet-4 \
  --strategies B0_naive,S1_position_swap,S2_ensemble_same,S3_ensemble_cross,S4_calibrated_rubric,S5_cot_forcing,S6_reference_guided,S7_combined_full,S8_combined_budget \
  --benchmarks llmbar \
  --sample-size 200
```

**What this produces:** Evidence that findings generalize beyond MT-Bench.

### Priority 4: Custom Dataset, All Models

Runs every strategy on our controlled dataset for all models. Key for the cross-bias interaction analysis.

```bash
# ~16,000 API calls, ~$80-120
python scripts/run_experiments.py \
  --models gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite,claude-sonnet-4 \
  --strategies B0_naive,S1_position_swap,S2_ensemble_same,S3_ensemble_cross,S4_calibrated_rubric,S5_cot_forcing,S6_reference_guided,S7_combined_full,S8_combined_budget \
  --benchmarks custom
```

**What this produces:** Controlled causal measurements of bias. Since all custom pairs have known ground truth (tie), any systematic preference reveals bias.

### Priority 5: Fill Remaining Matrix

Completes the picture for mid-tier and budget models.

```bash
# ~5,000 API calls, ~$10-20
python scripts/run_experiments.py \
  --models gemini-2.5-flash,gemini-2.5-flash-lite \
  --strategies B0_naive,S1_position_swap,S3_ensemble_cross,S8_combined_budget \
  --benchmarks mt_bench,custom \
  --sample-size 80
```

**What this produces:** Tests whether cheaper models benefit more from debiasing (Hypothesis H5).

### Batch Mode (for large Gemini runs)

For large runs with Gemini models, use batch mode for lower cost and higher throughput:

```bash
python scripts/run_experiments.py \
  --batch \
  --models gemini-2.5-flash,gemini-2.5-pro,gemini-2.5-flash-lite \
  --strategies B0_naive,S4_calibrated_rubric,S5_cot_forcing \
  --benchmarks mt_bench,custom \
  --sample-size 80 \
  --poll-interval 30
```

Batch mode is limited to Gemini models and single-call strategies (B0, S4, S5, S6). Use online mode for Claude and multi-call strategies (S1, S2, S3, S7, S8).

### Run Everything (full matrix)

```bash
python scripts/run_experiments.py
```

## Results and Caching

All results are cached in `results/raw/`. If a run is interrupted, rerun the same command and it skips completed instances.

Aggregated metrics are saved to `results/aggregated/{model}_{strategy}_{benchmark}_metrics.json`.

## From Experiments to Paper

### Step 1: Run experiments in priority order (P1 through P5)

P1 and P2 alone give you enough for a submittable paper. P3 through P5 strengthen results.

### Step 2: Analysis notebooks

After running experiments, use the analysis notebooks in `analysis/notebooks/`:

| Notebook | What it produces | Paper section |
|----------|-----------------|---------------|
| `01_bias_measurement.ipynb` | Heatmap of bias magnitudes per model (Figure 1) | Section 4.1 |
| `02_strategy_comparison.ipynb` | Strategy effectiveness bar chart + main results table (Figure 2, Table 1) | Section 4.2 |
| `03_cross_bias_interactions.ipynb` | Cross-bias interaction heatmap (Figure 3) | Section 4.3 |
| `04_cost_accuracy_tradeoff.ipynb` | Pareto frontier plot of cost vs accuracy (Figure 4) | Section 5 |
| `05_figures_for_paper.ipynb` | Publication-quality PDF figures for LaTeX | All figures |

### Step 3: Paper writing

Key sections:

1. **Introduction**: Motivation, gap in existing work, contributions
2. **Related Work**: LLM-as-judge literature, known biases, existing mitigation
3. **Methodology**: Framework design, models, benchmarks, strategies, metrics
4. **Experiments & Results**: Main results tables, ablations, analysis
5. **Discussion**: Cross-bias interactions, cost-accuracy tradeoffs, practical recommendations
6. **Conclusion**: Summary, limitations, future work

### Key contributions

1. **First unified benchmark** comparing 9 debiasing strategies across 4 LLM judges, 5 benchmarks, and 3 bias types
2. **Cross-bias interaction analysis** showing that strategies effective for one bias can worsen others
3. **A practical meta-debiasing pipeline** with cost-accuracy Pareto analysis
4. **Open-source framework and controlled dataset** for reproducible LLM judge evaluation

## Project Structure

```
llm-as-judge/
├── configs/
│   ├── models.yaml           # Judge model definitions
│   ├── strategies.yaml       # Debiasing strategy configs
│   └── experiments.yaml      # Experiment matrix and priorities
├── data/
│   ├── raw/                  # Downloaded benchmark data
│   │   ├── mt_bench/         # 3,355 human preference instances
│   │   └── llmbar/           # 419 adversarial evaluation instances
│   └── custom/
│       ├── generate_controlled.py  # Script to generate controlled dataset
│       └── controlled_pairs.jsonl  # 200 controlled bias-trigger pairs
├── src/
│   ├── judges/               # LLM judge implementations
│   │   ├── base.py           # Abstract judge interface
│   │   ├── gemini_judge.py   # Gemini via Vertex AI
│   │   ├── anthropic_judge.py # Claude via Vertex AI
│   │   ├── vertex_endpoint_judge.py # Open-source models
│   │   ├── batch.py          # Vertex AI batch prediction
│   │   ├── prompts.py        # All evaluation prompt templates
│   │   └── factory.py        # Judge factory from config
│   ├── strategies/           # Debiasing strategy implementations
│   │   ├── naive.py          # B0: Baseline
│   │   ├── position_swap.py  # S1: Position swapping
│   │   ├── ensemble_same.py  # S2: Same-family ensemble
│   │   ├── ensemble_cross.py # S3: Cross-family ensemble
│   │   ├── calibrated_rubric.py # S4: Rubric-based scoring
│   │   ├── cot_forcing.py    # S5: Chain-of-thought
│   │   ├── reference_guided.py # S6: Reference-guided
│   │   ├── combined.py       # S7/S8: Combined strategies
│   │   └── factory.py        # Strategy factory
│   ├── metrics/              # Evaluation metrics with bootstrap CIs
│   ├── data_loader.py        # Unified benchmark data loading
│   └── runner.py             # Experiment orchestrator (online + batch)
├── scripts/
│   ├── run_experiments.py    # CLI for full experiment matrix
│   ├── run_single.py         # Quick single-config test
│   └── download_data.sh      # Download benchmark datasets
├── analysis/notebooks/       # Jupyter notebooks for figures and tables
└── results/
    ├── raw/                  # Cached per-instance results (auto-generated)
    └── aggregated/           # Aggregated metrics per configuration
```
