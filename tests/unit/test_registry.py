"""Unit tests: plugin registry."""

from __future__ import annotations

import pytest

from cta_qsar.core.exceptions import PluginError
from cta_qsar.core.registry import PluginRegistry


class DummyPlugin:
    name = "dummy"


def test_register_and_get() -> None:
    registry = PluginRegistry()
    registry.register("model", DummyPlugin())
    assert registry.get("model", "dummy") is not None


def test_duplicate_registration_raises() -> None:
    registry = PluginRegistry()
    registry.register("model", DummyPlugin())
    with pytest.raises(PluginError):
        registry.register("model", DummyPlugin())


def test_unknown_kind_raises() -> None:
    registry = PluginRegistry()
    with pytest.raises(PluginError):
        registry.register("nonsense", DummyPlugin())
    with pytest.raises(PluginError):
        registry.get("nonsense", "dummy")


def test_unknown_name_raises() -> None:
    registry = PluginRegistry()
    with pytest.raises(PluginError):
        registry.get("model", "missing")
    assert registry.get_or_none("model", "missing") is None


def test_factory_registration_is_lazy() -> None:
    registry = PluginRegistry()
    created = []

    def factory() -> DummyPlugin:
        created.append(1)
        return DummyPlugin()

    registry.register_factory("model", "lazy_dummy", factory)
    assert registry.list("model") == ["lazy_dummy"]
    assert created == []  # not instantiated yet
    registry.get("model", "lazy_dummy")
    assert created == [1]


def test_require_resolves_available_only() -> None:
    registry = PluginRegistry()
    registry.register("model", DummyPlugin())
    resolved = registry.require("model", ["dummy", "missing"])
    assert list(resolved) == ["dummy"]


def test_auto_discover_registers_all_builtin_plugins() -> None:
    registry = PluginRegistry()
    registry.auto_discover()
    expected = {
        "representation": {"morgan", "rdkit_descriptors", "maccs", "atom_pair", "torsion", "rdkit_fp", "graph"},
        "model": {"ridge", "elastic_net", "random_forest", "extra_trees", "svr", "xgboost", "mlp", "gcn", "gat", "mpnn"},
        "validation": {"random", "scaffold", "stratified", "cluster", "temporal"},
        "trust": {"predictive", "generalization", "robustness", "uncertainty", "explainability"},
        "diagnosis": {"failure_rules"},
        "intervention": {"intervention_proposer"},
    }
    for kind, names in expected.items():
        registered = set(registry.list(kind))
        assert names.issubset(registered), f"missing {kind} plugins: {names - registered}"