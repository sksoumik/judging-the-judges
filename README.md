# Judging the Judges

Code and data for the TMLR paper "Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines" (https://openreview.net/forum?id=QF4IAmG4zc).

We measure four biases in LLM judges (position, verbosity, style, and self-preference) and compare nine debiasing strategies across five judge models (Gemini 2.5 Pro, Gemini 2.5 Flash, Claude Sonnet 4, GPT-4o, Llama 3.3-70B) and three benchmarks (MT-Bench, LLMBar, and a 375-pair controlled dataset we built).

## Setup

Needs Python 3.13 and [uv](https://github.com/astral-sh/uv). Gemini, Claude, and Llama run through Vertex AI; GPT-4o runs through the OpenAI API.

```bash
uv sync
cp .env.example .env          # add your API keys and GCP project id
gcloud auth application-default login
```

## Running

Single config on a small sample, to check the setup works:

```bash
python scripts/run_single.py --model gemini-2.5-flash --strategy B0_naive --benchmark custom --sample-size 5
```

A subset of the matrix (omit the flags to run everything):

```bash
python scripts/run_experiments.py \
  --models gemini-2.5-pro,claude-sonnet-4 \
  --strategies B0_naive,S5_cot_forcing,S8_combined_budget \
  --benchmarks mt_bench --sample-size 400
```

Results are cached per instance in `results/raw/`, so rerunning the same command resumes an interrupted run. Aggregated metrics are written to `results/aggregated/`. The figures and tables in the paper come from the scripts in `analysis/`.

## Layout

```
src/judges/      judge clients (Gemini, Claude, GPT-4o, Llama)
src/strategies/  the nine strategies (B0, S1 to S8)
src/metrics/     agreement, kappa, and bias scores with bootstrap CIs
data/custom/     the 375-pair controlled dataset
configs/         model, strategy, and experiment configs
analysis/        analysis and figure scripts
results/         cached outputs and aggregated metrics
```

## Citation

```bibtex
@article{soumik2026judging,
  title={Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines},
  author={Soumik, Sadman Kabir},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=QF4IAmG4zc}
}
```
