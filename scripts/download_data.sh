#!/usr/bin/env bash
# Download benchmark datasets using curl (bypasses Python SSL issues with corporate proxies)
set -e

DATA_DIR="data/raw"

echo "=== Downloading MT-Bench Human Judgments ==="
mkdir -p "$DATA_DIR/mt_bench"
curl -sL "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/human-00000-of-00001-15e4bfb3e0581eaa.parquet" \
  -o "$DATA_DIR/mt_bench/human.parquet"
echo "  Downloaded mt_bench/human.parquet"

echo "=== Downloading LLMBar ==="
mkdir -p "$DATA_DIR/llmbar"
for split in Natural Adversarial_Neighbor Adversarial_GPTInst Adversarial_GPTOut Adversarial_Manual; do
  curl -sL "https://huggingface.co/datasets/princeton-nlp/LLMBar/resolve/main/${split}/dataset.json" \
    -o "$DATA_DIR/llmbar/${split}.json" 2>/dev/null || echo "  Warning: $split not found"
done
echo "  Downloaded LLMBar splits"

echo "=== Done ==="
ls -lhR "$DATA_DIR"
