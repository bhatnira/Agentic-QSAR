"""Model plugins package root."""

from cta_qsar.models.base import (
    ElasticNetPlugin,
    ExtraTreesPlugin,
    LightGBMPlugin,
    MLPPlugin,
    RandomForestPlugin,
    RidgePlugin,
    SVRPlugin,
    XGBoostPlugin,
)
from cta_qsar.models.deep.gcn import GCNPlugin, TorchGCN
from cta_qsar.models.registry import (
    available_models,
    build_estimator,
    estimate_model_cost,
    hyperparameter_space,
    wrapped_estimator,
)

__all__ = [
    "ElasticNetPlugin",
    "ExtraTreesPlugin",
    "GCNPlugin",
    "LightGBMPlugin",
    "MLPPlugin",
    "RandomForestPlugin",
    "RidgePlugin",
    "SVRPlugin",
    "TorchGCN",
    "XGBoostPlugin",
    "available_models",
    "build_estimator",
    "estimate_model_cost",
    "hyperparameter_space",
    "wrapped_estimator",
]