"""Plugin discovery and registration.

Plugins are registered explicitly from their package modules.  New plugins are
added by writing a new module and calling ``PluginRegistry.register(...)`` (or
auto-discovering entry points) without modifying the orchestration engine.
"""

from __future__ import annotations

import contextlib
import importlib
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from cta_qsar.core.exceptions import PluginError

_PLUGIN_KINDS = (
    "endpoint",
    "preprocessing",
    "representation",
    "model",
    "validation",
    "trust",
    "uncertainty",
    "explainability",
    "diagnosis",
    "intervention",
    "llm_provider",
    "reporting",
)


class PluginRegistry:
    """Namespaced registry of plugin instances, keyed by kind then name."""

    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, Any]] = defaultdict(dict)
        self._factories: dict[str, dict[str, Callable[[], Any]]] = defaultdict(dict)

    # -- registration ------------------------------------------------------
    def register(self, kind: str, plugin: Any) -> None:
        if kind not in _PLUGIN_KINDS:
            raise PluginError(f"Unknown plugin kind: {kind}")
        name = getattr(plugin, "name", plugin.__class__.__name__)
        if name in self._plugins[kind]:
            raise PluginError(f"Duplicate plugin {kind}/{name}")
        self._plugins[kind][name] = plugin

    def register_factory(self, kind: str, name: str, factory: Callable[[], Any]) -> None:
        if kind not in _PLUGIN_KINDS:
            raise PluginError(f"Unknown plugin kind: {kind}")
        self._factories[kind][name] = factory

    # -- lookup ------------------------------------------------------------
    def get(self, kind: str, name: str) -> Any:
        if kind not in _PLUGIN_KINDS:
            raise PluginError(f"Unknown plugin kind: {kind}")
        if name in self._plugins[kind]:
            return self._plugins[kind][name]
        if name in self._factories[kind]:
            plugin = self._factories[kind][name]()
            self._plugins[kind][name] = plugin
            return plugin
        raise PluginError(f"Plugin not registered: {kind}/{name}")

    def get_or_none(self, kind: str, name: str) -> Any | None:
        try:
            return self.get(kind, name)
        except PluginError:
            return None

    def list(self, kind: str, *, include_unavailable: bool = False) -> list[str]:
        return sorted(set(self._plugins[kind]) | set(self._factories[kind]))

    def all(self, kind: str) -> list[Any]:
        return [self.get(kind, name) for name in self.list(kind)]

    def require(self, kind: str, names: list[str]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for name in names:
            plugin = self.get_or_none(kind, name)
            if plugin is not None:
                resolved[name] = plugin
        return resolved

    # -- convenience -------------------------------------------------------
    def register_module(self, module_name: str, kind: str) -> None:
        """Register every plugin named ``PLUGINS`` (instances or callables)."""
        module = importlib.import_module(module_name)
        for entry in getattr(module, "PLUGINS", []):
            name = entry.name if isinstance(entry, type) else getattr(entry, "name", None)
            if isinstance(entry, type):
                try:
                    instance = entry()
                except (PluginError, TypeError):
                    instantiator = entry
                    if name is not None:
                        self.register_factory(kind, name, instantiator)
                    continue
                self.register(kind, instance)
            else:
                self.register(kind, entry() if callable(entry) else entry)

    def auto_discover(self) -> None:
        """Discover plugin modules in cta_qsar plugin packages.

        Each discovered module is expected to define ``PLUGINS`` (a list of
        instances or zero-arg callables) and ``PLUGIN_KIND``.
        """
        package = importlib.import_module("cta_qsar")
        _ = package
        # Explicitly register from known plugin modules for determinism.
        _EXPLICIT: list[tuple[str, str]] = [
            ("cta_qsar.endpoints.regression", "endpoint"),
            ("cta_qsar.endpoints.classification", "endpoint"),
            ("cta_qsar.representations.fingerprints.morgan", "representation"),
            ("cta_qsar.representations.fingerprints.rdkit_fp", "representation"),
            ("cta_qsar.representations.fingerprints.maccs", "representation"),
            ("cta_qsar.representations.fingerprints.atompair", "representation"),
            ("cta_qsar.representations.fingerprints.torsion", "representation"),
            ("cta_qsar.representations.descriptors.rdkit_desc", "representation"),
            ("cta_qsar.representations.descriptors.mordred_desc", "representation"),
            ("cta_qsar.models.classical.ridge", "model"),
            ("cta_qsar.models.classical.elasticnet", "model"),
            ("cta_qsar.models.classical.randomforest", "model"),
            ("cta_qsar.models.classical.extratrees", "model"),
            ("cta_qsar.models.classical.svr", "model"),
            ("cta_qsar.models.classical.xgboost_model", "model"),
            ("cta_qsar.models.classical.lightgbm_model", "model"),
            ("cta_qsar.models.classical.mlp", "model"),
            ("cta_qsar.models.deep.gcn", "model"),
            ("cta_qsar.models.deep.gat", "model"),
            ("cta_qsar.models.deep.mpnn", "model"),
            ("cta_qsar.models.foundation.heads", "model"),
            ("cta_qsar.models.automl.autogluon", "model"),
            ("cta_qsar.representations.graph.plugin", "representation"),
            ("cta_qsar.representations.embeddings.foundation", "representation"),
            ("cta_qsar.validation.random_split", "validation"),
            ("cta_qsar.validation.stratified", "validation"),
            ("cta_qsar.validation.scaffold_split", "validation"),
            ("cta_qsar.validation.cluster_split", "validation"),
            ("cta_qsar.validation.temporal_split", "validation"),
            ("cta_qsar.trust.predictive", "trust"),
            ("cta_qsar.trust.generalization", "trust"),
            ("cta_qsar.trust.robustness", "trust"),
            ("cta_qsar.trust.applicability", "trust"),
            ("cta_qsar.trust.uncertainty", "trust"),
            ("cta_qsar.trust.explainability", "trust"),
            ("cta_qsar.diagnosis.failure", "diagnosis"),
            ("cta_qsar.diagnosis.interventions", "intervention"),
        ]
        for module_name, kind in _EXPLICIT:
            with contextlib.suppress(ImportError, PluginError):
                self.register_module(module_name, kind)


_default_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Return the process-wide default registry."""
    return _default_registry