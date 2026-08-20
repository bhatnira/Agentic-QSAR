"""Configuration loading and validation.

Configuration precedence (highest wins):
    1. explicit CLI arguments
    2. environment variables
    3. YAML config file
    4. built-in defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field

from cta_qsar.core.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


class LLMConfig(BaseModel):
    provider: str = "mock"
    model: str = "heuristic"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: int = 120


class ComputeConfig(BaseModel):
    max_minutes: float = 30.0
    max_experiments: int = 12
    max_memory_gb: float = 16.0
    gpu_required: bool = False


class DatasetConfig(BaseModel):
    smiles_column: str | None = None
    target_column: str | None = None
    max_rows: int = 100_000
    standardize: bool = True
    drop_invalid: bool = False


class ExperimentConfig(BaseModel):
    n_splits: int = 5
    n_repeats: int = 2
    test_fraction: float = 0.2
    random_seed: int = 42
    min_cv_score_improvement: float = 0.01
    hyperparameter_search: bool = False


class BudgetConfig(BaseModel):
    default_max_experiments: int = 12
    cheap_runtime_seconds: int = 60
    expensive_runtime_seconds: int = 600


class TrustConfig(BaseModel):
    required: list[str] = Field(default_factory=lambda: ["predictive", "generalization"])


class TrackingConfig(BaseModel):
    enabled: bool = True
    backend: str = "mlflow"
    mlflow_uri: str = ""


class Config(BaseModel):
    """Top-level CTA-QSAR configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    representations: dict[str, list[str]] = Field(
        default_factory=lambda: {"enabled": ["morgan", "rdkit_descriptors"]}
    )
    models: dict[str, list[str]] = Field(
        default_factory=lambda: {"enabled": ["ridge", "random_forest"]}
    )
    validation: dict[str, list[str]] = Field(
        default_factory=lambda: {"enabled": ["random", "scaffold"]}
    )
    trust: TrustConfig = Field(default_factory=TrustConfig)
    uncertainty: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    explainability: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    reporting: dict[str, Any] = Field(
        default_factory=lambda: {"format": "markdown", "output_dir": "runs"}
    )
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    env_overrides: ClassVar[dict[str, str]] = {
        "LLM_PROVIDER": "llm.provider",
        "LLM_MODEL": "llm.model",
        "MLFLOW_TRACKING_URI": "tracking.mlflow_uri",
        "CTA_QSAR_OUTPUT_DIR": "reporting.output_dir",
    }

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load config from a YAML file (or defaults) merged with env vars."""
        _load_dotenv()
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text()) or {}
        cfg = cls.model_validate(raw)
        cfg._apply_env_overrides()  # noqa: SLF001
        return cfg

    def save(self, path: str | Path) -> None:
        """Persist the effective configuration as YAML."""
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json")))

    def _apply_env_overrides(self) -> None:
        resolved: dict[str, str] = {}
        for env_name, dotted_path in self.env_overrides.items():
            value = os.getenv(env_name)
            if value is None or value == "":
                continue
            resolved[dotted_path] = value
        # NVIDIA_MODEL is a convenience alias for llm.model; LLM_MODEL wins.
        if "llm.model" not in resolved:
            nvidia_model = os.getenv("NVIDIA_MODEL")
            if nvidia_model:
                resolved["llm.model"] = nvidia_model
        for dotted_path, value in resolved.items():
            node: Any = self
            parts = dotted_path.split(".")
            for part in parts[:-1]:
                node = getattr(node, part)
            if isinstance(node, dict):
                node[parts[-1]] = value
            else:
                setattr(node, parts[-1], value)


def _load_dotenv() -> None:
    """Load .env from the project root into os.environ (never overwrites)."""
    from pathlib import Path as _Path

    import dotenv

    candidates = [
        _Path(os.environ.get("CTA_QSAR_ENV_FILE", "")),
        _Path.cwd() / ".env",
        DEFAULT_CONFIG_PATH.parents[1] / ".env",  # repo root
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                dotenv.load_dotenv(candidate, override=False)
                return
        except OSError:  # pragma: no cover
            continue


def build_config(
    config_path: str | Path | None = None,
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    max_minutes: float | None = None,
    max_experiments: int | None = None,
    smiles_column: str | None = None,
    target_column: str | None = None,
    seed: int | None = None,
    hyperparameter_search: bool | None = None,
) -> Config:
    """Build a Config merging CLI overrides on top of file + env."""
    cfg = Config.load(config_path)
    if llm_provider:
        cfg.llm.provider = llm_provider
    if llm_model:
        cfg.llm.model = llm_model
    if max_minutes:
        cfg.compute.max_minutes = max_minutes
    if max_experiments:
        cfg.compute.max_experiments = max_experiments
    if smiles_column:
        cfg.dataset.smiles_column = smiles_column
    if target_column:
        cfg.dataset.target_column = target_column
    if seed is not None:
        cfg.experiment.random_seed = seed
    if hyperparameter_search is not None:
        cfg.experiment.hyperparameter_search = hyperparameter_search
    return cfg