"""Representation registry helpers.

The plugin registry (core.registry) holds *instances*; here we add behavioral
helpers used by the planner: filtering by config + applicability, cost
estimates, and safe transform with availability guards.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.core.registry import PluginRegistry


def available_representations(
    registry: PluginRegistry,
    enabled: list[str] | None,
    task_type: str,
    dataset_props: dict[str, Any],
) -> dict[str, Any]:
    """Return {name: plugin_instance} for enabled+applicable representations."""
    names = enabled if enabled is not None else registry.list("representation")
    resolved: dict[str, Any] = {}
    for name in names:
        plugin = registry.get_or_none("representation", name)
        if plugin is None:
            continue
        try:
            applicable, _ = plugin.applicability(task_type, dataset_props)
        except Exception:  # noqa: BLE001
            applicable = False
        if applicable:
            resolved[name] = plugin
    return resolved


def representation_matrix(
    plugin: Any, smiles: list[str], *, fit: bool = True
) -> np.ndarray | Any:
    """Compute the feature matrix for a representation plugin.

    Raises ``PluginUnavailableError`` with a usable message when optional
    dependencies are missing.
    """
    try:
        if fit:
            return plugin.fit_transform(smiles)
        return plugin.transform(smiles)
    except PluginUnavailableError:
        raise
    except ImportError as exc:
        raise PluginUnavailableError(
            f"representation {plugin.name} requires missing dependency: {exc}"
        ) from exc


def estimate_rep_cost(plugin: Any, n_molecules: int) -> CostEstimate:
    try:
        cost = plugin.estimate_cost(n_molecules)
    except Exception:  # noqa: BLE001
        cost = CostEstimate(runtime_seconds=n_molecules * 0.01, memory_gb=0.2)
    return cost