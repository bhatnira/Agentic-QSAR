"""AutoGluon plugin (optional heavy dependency; CPU-compatible)."""

from __future__ import annotations

from typing import Any

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate

AG_SKLEARN_DATAFRAME_TASK = ("binary", "multiclass", "regression")


def _ag_available() -> bool:
    try:
        import autogluon.tabular  # noqa: F401

        return True
    except ImportError:
        return False


class AutoGluonPlugin:
    name = "autogluon"
    version = "1.0.0"
    supports = ("regression", "binary", "multiclass")

    def applicability(self, task_type: str, representation_name: str) -> tuple[bool, str]:
        if not _ag_available():
            return False, "autogluon not installed (pip install autogluon)"
        if task_type not in self.supports:
            return False, f"task type {task_type} not supported"
        return True, "automated ML stack on CPU (heavy; used sparingly)"

    def estimate_cost(
        self, n_samples: int, n_features: int, representation_name: str
    ) -> CostEstimate:
        return CostEstimate(
            runtime_seconds=max(120.0, n_samples * 0.5),
            memory_gb=max(1.0, n_samples * n_features * 8e-9 * 5),
        )

    def build_estimator(
        self, task_type: str, n_classes: int | None = None, **hyperparams: Any
    ) -> Any:
        if not _ag_available():
            raise PluginUnavailableError("autogluon not installed")
        from autogluon.tabular import TabularPredictor

        predictor = TabularPredictor(
            label="y",
            problem_type="regression"
            if task_type == "regression"
            else ("binary" if task_type == "binary" else "multiclass"),
            verbosity=0,
        )
        return _AGWrapper(predictor)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {}


class _AGWrapper:
    """Minimal sklearn-duckpadapter so the runner can call fit/predict."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def fit(self, X: Any, y: Any) -> _AGWrapper:
        import pandas as pd

        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        frame["y"] = y
        self.predictor.fit(frame, time_limit=120, presets="medium_quality")
        return self

    def predict(self, X: Any) -> Any:
        import pandas as pd

        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        return self.predictor.predict(frame)

    def predict_proba(self, X: Any) -> Any:
        import pandas as pd

        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        proba = self.predictor.predict_proba(frame)
        return proba

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"predictor": self.predictor}

    def set_params(self, **params: Any) -> _AGWrapper:
        return self


PLUGINS = [AutoGluonPlugin]