"""Model plugin base classes.

Each model plugin builds a scikit-learn compatible estimator for a given task
type, reports applicability, estimates compute cost, and exposes a small
hyperparameter candidate space for cheap tuning.
"""

from __future__ import annotations

import abc
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate

TASK_TYPES = ("regression", "binary", "multiclass", "multitask_regression", "multitask_classification")


def _estimator_for(task_type: str, model_cls: Any, n_classes: int | None, try_balance: bool = False, **hp: Any) -> Any:
    if task_type == "regression":
        return model_cls.Regressor(**hp) if hasattr(model_cls, "Regressor") else model_cls(**hp)
    if task_type in ("binary", "multiclass"):
        if try_balance and hasattr(model_cls, "Classifier") and "class_weight" not in hp:
            hp["class_weight"] = "balanced"
        return model_cls.Classifier(**hp) if hasattr(model_cls, "Classifier") else model_cls(**hp)
    raise PluginUnavailableError(f"model does not support task type {task_type}")


class ScikitModelPlugin(abc.ABC):
    """Base for classical regressor/classifier plugins."""

    name: str = ""
    supports: tuple[str, ...] = ("regression", "binary", "multiclass")
    needs_scaling: bool = False
    default_hyperparams: dict[str, Any] = {}

    def applicability(self, task_type: str, representation_name: str) -> tuple[bool, str]:
        if task_type not in self.supports:
            return False, f"task type {task_type} not supported by {self.name}"
        if representation_name == "graph":
            return False, f"{self.name} cannot consume graph representations"
        return True, "default applicability"

    def estimate_cost(
        self, n_samples: int, n_features: int, representation_name: str
    ) -> CostEstimate:
        base = self.cheap_runtime_seconds(n_samples, n_features)
        return CostEstimate(runtime_seconds=base, memory_gb=max(0.1, n_samples * n_features * 8e-9 * 3))

    def cheap_runtime_seconds(self, n_samples: int, n_features: int) -> float:
        return max(1.0, n_samples / 200.0 + n_features / 2000.0)

    def build_estimator(
        self, task_type: str, n_classes: int | None = None, **hyperparams: Any
    ) -> Any:
        hp = {**self.default_hyperparams, **hyperparams}
        return self._build(task_type, n_classes, hp)

    @abc.abstractmethod
    def _build(self, task_type: str, n_classes: int | None, hp: dict[str, Any]) -> Any:
        """Construct the raw scikit-learn estimator."""

    def wrapped_estimator(self, task_type: str, n_classes: int | None = None, **hp: Any) -> Any:
        """Estimator wrapped in a scaler pipeline for scale-sensitive models."""
        raw = self.build_estimator(task_type, n_classes, **hp)
        if self.needs_scaling:
            return Pipeline([("scaler", StandardScaler()), ("model", raw)])
        return raw

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        """Candidate hyperparameter values for cheap grid search.  Empty = defaults only."""
        return {}


class RidgePlugin(ScikitModelPlugin):
    name = "ridge"
    supports = ("regression",)
    needs_scaling = True
    default_hyperparams = {}

    def _build(self, task_type, n_classes, hp):
        return Ridge(**hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"alpha": [0.1, 1.0, 10.0]}


class ElasticNetPlugin(ScikitModelPlugin):
    name = "elastic_net"
    supports = ("regression",)
    needs_scaling = True
    default_hyperparams = {"max_iter": 2000}

    def _build(self, task_type, n_classes, hp):
        return ElasticNet(**hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}


class RandomForestPlugin(ScikitModelPlugin):
    name = "random_forest"
    supports = ("regression", "binary", "multiclass")
    default_hyperparams = {"n_estimators": 300, "n_jobs": -1}

    def cheap_runtime_seconds(self, n_samples: int, n_features: int) -> float:
        return max(2.0, n_samples / 50.0 + n_features / 300.0)

    def _build(self, task_type, n_classes, hp):
        if task_type == "regression":
            return RandomForestRegressor(**hp)
        return RandomForestClassifier(**hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"n_estimators": [100, 300], "max_depth": [None, 20], "min_samples_leaf": [1, 4]}


class ExtraTreesPlugin(ScikitModelPlugin):
    name = "extra_trees"
    supports = ("regression", "binary", "multiclass")
    default_hyperparams = {"n_estimators": 300, "n_jobs": -1}

    def _build(self, task_type, n_classes, hp):
        if task_type == "regression":
            return ExtraTreesRegressor(**hp)
        return ExtraTreesClassifier(**hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"n_estimators": [100, 300], "max_depth": [None, 20], "min_samples_leaf": [1, 4]}


class SVRPlugin(ScikitModelPlugin):
    name = "svr"
    supports = ("regression", "binary")
    needs_scaling = True
    default_hyperparams = {"max_iter": 5000}

    def _build(self, task_type, n_classes, hp):
        if task_type == "regression":
            return SVR(**hp)
        return SVC(probability=True, class_weight="balanced", **hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"C": [0.1, 1.0, 10.0], "gamma": ["scale"]}


class MLPPlugin(ScikitModelPlugin):
    name = "mlp"
    supports = ("regression", "binary", "multiclass")
    needs_scaling = True
    default_hyperparams = {"hidden_layer_sizes": (128, 64), "max_iter": 300, "early_stopping": True}

    def _build(self, task_type, n_classes, hp):
        if task_type == "regression":
            return MLPRegressor(**hp)
        return MLPClassifier(**hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"hidden_layer_sizes": [(128, 64), (256, 128)], "alpha": [1e-4, 1e-3]}


class XGBoostPlugin(ScikitModelPlugin):
    name = "xgboost"
    supports = ("regression", "binary", "multiclass")

    def __init__(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise PluginUnavailableError("xgboost not installed") from exc

    def _build(self, task_type, n_classes, hp):
        import xgboost as xgb

        if task_type == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, **hp)
        if task_type == "binary":
            return xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                                     n_jobs=-1, **hp)
        return xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes or 3,
                                 n_jobs=-1, **hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"n_estimators": [100, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}


class LightGBMPlugin(ScikitModelPlugin):
    name = "lightgbm"
    supports = ("regression", "binary", "multiclass")

    def __init__(self) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:
            raise PluginUnavailableError("lightgbm not installed") from exc

    def _build(self, task_type, n_classes, hp):
        import lightgbm as lgb

        if task_type == "regression":
            return lgb.LGBMRegressor(n_jobs=-1, verbose=-1, **hp)
        if task_type == "binary":
            return lgb.LGBMClassifier(n_jobs=-1, verbose=-1, class_weight="balanced", **hp)
        return lgb.LGBMClassifier(n_jobs=-1, verbose=-1, **hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"n_estimators": [100, 300], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]}