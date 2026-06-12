"""Factory for creating judge instances from config."""

import importlib
import os

import yaml

from src.judges.anthropic_judge import AnthropicJudge
from src.judges.base import JudgeBase
from src.judges.gemini_judge import GeminiJudge
from src.judges.openai_judge import OpenAIJudge
from src.judges.vertex_endpoint_judge import VertexEndpointJudge


def _load_models_config(config_path: str = "configs/models.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _try_local_override(provider: str):
    """Return a local override class for the given provider, if one is installed.

    The factory looks for a private module at ``src.judges._local_{provider}``
    that exports a ``LocalJudge`` class. This is a hook for site-specific
    deployments where the public provider client must be replaced with an
    internal alternative; the public repo never ships such a module, so
    external reproducers always get the standard provider judge below.
    """
    try:
        module = importlib.import_module(f"src.judges._local_{provider}")
    except ImportError:
        return None
    return getattr(module, "LocalJudge", None) or getattr(
        module, f"Local{provider.capitalize()}Judge", None
    )


class JudgeFactory:
    """Creates judge instances from YAML configuration."""

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        project_id: str | None = None,
    ):
        self.config = _load_models_config(config_path)
        self.project_id = project_id or os.environ.get("GOOGLE_PROJECT_ID", "")
        self.defaults = self.config.get("defaults", {})

    def create(self, model_name: str) -> JudgeBase:
        """Create a judge for the given model name."""
        models = self.config["models"]
        if model_name not in models:
            available = ", ".join(models.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )

        cfg = models[model_name]
        provider = cfg["provider"]
        temperature = cfg.get("temperature", self.defaults.get("temperature", 0.1))
        max_tokens = cfg.get(
            "max_output_tokens", self.defaults.get("max_output_tokens", 8192)
        )
        cost_in = cfg.get("cost_per_1k_input_tokens", 0.0)
        cost_out = cfg.get("cost_per_1k_output_tokens", 0.0)

        if provider == "gemini":
            return GeminiJudge(
                project_id=self.project_id,
                model_id=cfg["model_id"],
                location=cfg.get("location", "global"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                cost_per_1k_input=cost_in,
                cost_per_1k_output=cost_out,
                json_mode=cfg.get("json_mode", True),
            )
        elif provider == "anthropic":
            return AnthropicJudge(
                project_id=self.project_id,
                model_id=cfg["model_id"],
                region=cfg.get("region", "us-east5"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                cost_per_1k_input=cost_in,
                cost_per_1k_output=cost_out,
            )
        elif provider == "openai":
            override = _try_local_override("openai")
            judge_cls = override or OpenAIJudge
            return judge_cls(
                model_id=cfg["model_id"],
                temperature=temperature,
                max_output_tokens=max_tokens,
                cost_per_1k_input=cost_in,
                cost_per_1k_output=cost_out,
            )
        elif provider == "vertex_endpoint":
            endpoint_id = cfg.get("endpoint_id")
            if not endpoint_id:
                raise ValueError(
                    f"Model '{model_name}' requires endpoint_id. "
                    "Deploy the model in Vertex AI Model Garden first."
                )
            return VertexEndpointJudge(
                project_id=self.project_id,
                endpoint_id=endpoint_id,
                model_id=cfg["model_id"],
                location=cfg.get("location", "us-central1"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                cost_per_1k_input=cost_in,
                cost_per_1k_output=cost_out,
            )
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{model_name}'")

    def list_available(self) -> list[str]:
        """List all configured model names."""
        return list(self.config["models"].keys())

    def list_ready(self) -> list[str]:
        """List models that are ready to use (no deployment required)."""
        ready = []
        for name, cfg in self.config["models"].items():
            if cfg["provider"] in ("gemini", "anthropic", "openai"):
                ready.append(name)
            elif cfg["provider"] == "vertex_endpoint" and cfg.get("endpoint_id"):
                ready.append(name)
        return ready
