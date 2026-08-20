"""Model registry helpers: availability filtering and estimator building."""

from __future__ import annotations

from typing import Any

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.core.registry import PluginRegistry


def available_models(
    registry: PluginRegistry,
    enabled: list[str] | None,
    task_type: str,
    representation_name: str,
) -> dict[str, Any]:
    """Return {name: plugin} for enabled models that apply to the pair."""
    names = enabled if enabled is not None else registry.list("model")
    resolved: dict[str, Any] = {}
    for name in names:
        plugin = registry.get_or_none("model", name)
        if plugin is None:
            continue
        try:
            applicable, _ = plugin.applicability(task_type, representation_name)
        except Exception:  # noqa: BLE001
            applicable = False
        if applicable:
            resolved[name] = plugin
    return resolved


def build_estimator(
    registry: PluginRegistry,
    name: str,
    task_type: str,
    n_classes: int | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    plugin = registry.get_or_none("model", name)
    if plugin is None:
        raise PluginUnavailableError(f"model plugin {name!r} not registered")
    try:
        return plugin.build_estimator(task_type, n_classes=n_classes, **(hyperparams or {}))
    except PluginUnavailableError:
        raise
    except ImportError as exc:
        raise PluginUnavailableError(
            f"model {name} requires missing dependency: {exc}"
        ) from exc


def wrapped_estimator(
    registry: PluginRegistry,
    name: str,
    task_type: str,
    n_classes: int | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> Any:
    """Estimator, wrapped in a scaling pipeline for scale-sensitive models."""
    plugin = registry.get_or_none("model", name)
    if plugin is None:
        raise PluginUnavailableError(f"model plugin {name!r} not registered")
    wrapper = getattr(plugin, "wrapped_estimator", None)
    if wrapper is not None:
        try:
            return wrapper(task_type, n_classes=n_classes, **(hyperparams or {}))
        except PluginUnavailableError:
            raise
        except ImportError as exc:
            raise PluginUnavailableError(f"model {name} requires missing dependency: {exc}") from exc
    return build_estimator(registry, name, task_type, n_classes=n_classes, hyperparams=hyperparams)


def estimate_model_cost(
    registry: PluginRegistry,
    name: str,
    task_type: str,
    n_samples: int,
    n_features: int,
    representation_name: str,
) -> CostEstimate:
    plugin = registry.get_or_none("model", name)
    if plugin is None:
        return CostEstimate(runtime_seconds=60.0, memory_gb=0.5)
    try:
        return plugin.estimate_cost(n_samples, n_features, representation_name)
    except Exception:  # noqa: BLE001
        return CostEstimate(runtime_seconds=60.0, memory_gb=0.5)


def hyperparameter_space(registry: PluginRegistry, name: str) -> dict[str, list[Any]]:
    plugin = registry.get_or_none("model", name)
    if plugin is None:
        return {}
    return getattr(plugin, "hyperparameter_space", lambda: {})()