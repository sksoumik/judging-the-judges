"""Unified data loader for all benchmarks."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import truststore
truststore.inject_into_ssl()

import numpy as np


@dataclass
class EvalInstance:
    id: str
    question: str
    response_a: str
    response_b: str
    human_preference: str | None = None  # "A", "B", "tie", or None
    reference: str | None = None
    metadata: dict = field(default_factory=dict)


class DataLoader:
    """Loads benchmark data into a standard format."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load(
        self,
        benchmark: str,
        sample_size: int | None = None,
        seed: int = 42,
    ) -> list[EvalInstance]:
        loaders = {
            "mt_bench": self._load_mt_bench,
            "alpaca_eval": self._load_alpaca_eval,
            "llmbar": self._load_llmbar,
            "faireval": self._load_faireval,
            "custom": self._load_custom,
        }
        if benchmark not in loaders:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(loaders.keys())}")

        instances = loaders[benchmark]()

        if sample_size and sample_size < len(instances):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(instances), size=sample_size, replace=False)
            instances = [instances[i] for i in sorted(indices)]

        return instances

    def _load_mt_bench(self) -> list[EvalInstance]:
        """Load MT-Bench from HuggingFace datasets (cached locally)."""
        path = self.data_dir / "raw" / "mt_bench"
        jsonl_file = path / "mt_bench.jsonl"

        if jsonl_file.exists():
            return self._load_jsonl(jsonl_file)

        # Download from HuggingFace
        try:
            instances = self._download_mt_bench_hf(path, jsonl_file)
            return instances
        except Exception as e:
            raise FileNotFoundError(
                f"MT-Bench data not found at {jsonl_file} and download failed: {e}. "
                "Run: python -c \"from datasets import load_dataset; load_dataset('lmsys/mt_bench_human_judgments')\""
            )

    def _download_mt_bench_hf(self, path: Path, jsonl_file: Path) -> list[EvalInstance]:
        """Download MT-Bench from HuggingFace and extract conversation fields.

        The lmsys/mt_bench_human_judgments dataset stores data as structured
        conversations in 'conversation_a' and 'conversation_b' (lists of
        {role, content} dicts), not flat text fields.
        """
        import io
        import urllib.request

        import pandas as pd

        url = (
            "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/"
            "resolve/main/data/human-00000-of-00001-25f4910818759289.parquet"
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        df = pd.read_parquet(io.BytesIO(data))

        path.mkdir(parents=True, exist_ok=True)
        instances = []
        for i, row in df.iterrows():
            qid = row.get("question_id", 0)
            turn = row.get("turn", 1)
            category = _MT_BENCH_CATEGORIES.get(qid, "")

            conv_a = row.get("conversation_a", [])
            conv_b = row.get("conversation_b", [])

            # Turn 1: question at [0], response at [1]
            # Turn 2: question at [2], response at [3]
            q_idx = (turn - 1) * 2
            r_idx = q_idx + 1

            question = conv_a[q_idx]["content"] if len(conv_a) > q_idx else ""
            response_a = conv_a[r_idx]["content"] if len(conv_a) > r_idx else ""
            response_b = conv_b[r_idx]["content"] if len(conv_b) > r_idx else ""

            winner = row.get("winner", "")
            inst = EvalInstance(
                id=f"mt_bench_{qid}_{turn}_{i}",
                question=question,
                response_a=response_a,
                response_b=response_b,
                human_preference=_normalize_preference(winner),
                metadata={
                    "model_a": row.get("model_a", ""),
                    "model_b": row.get("model_b", ""),
                    "question_id": qid,
                    "category": category,
                    "turn": turn,
                },
            )
            instances.append(inst)

        self._save_jsonl(instances, jsonl_file)
        return instances

    def _load_alpaca_eval(self) -> list[EvalInstance]:
        """Load AlpacaEval 2.0."""
        path = self.data_dir / "raw" / "alpaca_eval"
        jsonl_file = path / "alpaca_eval.jsonl"

        if jsonl_file.exists():
            return self._load_jsonl(jsonl_file)

        try:
            from datasets import load_dataset
            ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
            path.mkdir(parents=True, exist_ok=True)
            instances = []
            for i, row in enumerate(ds):
                inst = EvalInstance(
                    id=f"alpaca_eval_{i}",
                    question=row.get("instruction", ""),
                    response_a=row.get("output", ""),
                    response_b=row.get("baseline_output", row.get("output_2", "")),
                    human_preference=_normalize_preference(row.get("preference")),
                    reference=row.get("reference", None),
                    metadata={"dataset": row.get("dataset", "")},
                )
                instances.append(inst)
            self._save_jsonl(instances, jsonl_file)
            return instances
        except Exception as e:
            raise FileNotFoundError(f"AlpacaEval data not found and download failed: {e}")

    def _load_llmbar(self) -> list[EvalInstance]:
        """Load LLMBar benchmark."""
        path = self.data_dir / "raw" / "llmbar"
        jsonl_file = path / "llmbar.jsonl"

        if jsonl_file.exists():
            return self._load_jsonl(jsonl_file)

        try:
            from datasets import load_dataset
            instances = []
            for split_name in ["Natural", "Adversarial_Neighbor", "Adversarial_GPTInst", "Adversarial_GPTOut", "Adversarial_Manual"]:
                try:
                    ds = load_dataset("princeton-nlp/LLMBar", split_name, split="test")
                    for i, row in enumerate(ds):
                        inst = EvalInstance(
                            id=f"llmbar_{split_name}_{i}",
                            question=row.get("input", ""),
                            response_a=row.get("output_1", ""),
                            response_b=row.get("output_2", ""),
                            human_preference=_normalize_preference(row.get("label")),
                            metadata={"split": split_name},
                        )
                        instances.append(inst)
                except Exception:
                    continue
            if instances:
                path.mkdir(parents=True, exist_ok=True)
                self._save_jsonl(instances, jsonl_file)
            return instances
        except Exception as e:
            raise FileNotFoundError(f"LLMBar data not found and download failed: {e}")

    def _load_faireval(self) -> list[EvalInstance]:
        """Load FairEval benchmark."""
        path = self.data_dir / "raw" / "faireval"
        jsonl_file = path / "faireval.jsonl"
        if jsonl_file.exists():
            return self._load_jsonl(jsonl_file)
        raise FileNotFoundError(
            f"FairEval data not found at {jsonl_file}. "
            "Place faireval.jsonl in data/raw/faireval/"
        )

    def _load_custom(self) -> list[EvalInstance]:
        """Load custom controlled dataset."""
        path = self.data_dir / "custom" / "controlled_pairs.jsonl"
        if path.exists():
            return self._load_jsonl(path)
        raise FileNotFoundError(
            f"Custom dataset not found at {path}. "
            "Run: python data/custom/generate_controlled.py"
        )

    def _load_jsonl(self, path: Path) -> list[EvalInstance]:
        instances = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                metadata = data.get("metadata", {})
                # Capture custom dataset fields in metadata
                for key in ("bias_type", "expected_verdict", "manipulation"):
                    if key in data:
                        metadata[key] = data[key]
                # Populate MT-Bench category from question_id if missing
                if not metadata.get("category") and metadata.get("question_id"):
                    metadata["category"] = _MT_BENCH_CATEGORIES.get(
                        metadata["question_id"], ""
                    )
                instances.append(EvalInstance(
                    id=data["id"],
                    question=data["question"],
                    response_a=data["response_a"],
                    response_b=data["response_b"],
                    human_preference=data.get("human_preference"),
                    reference=data.get("reference"),
                    metadata=metadata,
                ))
        return instances

    def _save_jsonl(self, instances: list[EvalInstance], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for inst in instances:
                data = {
                    "id": inst.id,
                    "question": inst.question,
                    "response_a": inst.response_a,
                    "response_b": inst.response_b,
                    "human_preference": inst.human_preference,
                    "reference": inst.reference,
                    "metadata": inst.metadata,
                }
                f.write(json.dumps(data) + "\n")


# Standard MT-Bench question_id to category mapping
_MT_BENCH_CATEGORIES = {
    **{qid: "writing" for qid in range(81, 91)},
    **{qid: "roleplay" for qid in range(91, 101)},
    **{qid: "reasoning" for qid in range(101, 111)},
    **{qid: "math" for qid in range(111, 121)},
    **{qid: "coding" for qid in range(121, 131)},
    **{qid: "extraction" for qid in range(131, 141)},
    **{qid: "stem" for qid in range(141, 151)},
    **{qid: "humanities" for qid in range(151, 161)},
}


def _normalize_preference(value) -> str | None:
    """Normalize various preference formats to 'A', 'B', or 'tie'."""
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in ("a", "1", "model_a", "response_a"):
        return "A"
    elif val in ("b", "2", "model_b", "response_b"):
        return "B"
    elif val in ("tie", "0", "draw", "both", "neither"):
        return "tie"
    return None
