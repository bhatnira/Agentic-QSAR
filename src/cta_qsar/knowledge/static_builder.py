"""Auto-generated registry facts: derive static edges from plugin metadata.

Facts are version-stamped by hashing the registry's plugin catalog, so a
registry change invalidates cached evidence and re-derives relationships.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any

from cta_qsar.knowledge.facts import Fact

GRAPH_MODELS = {"gcn", "gat", "mpnn"}
FOUNDATION_MODELS = {"embedding_ridge", "embedding_mlp"}
FOUNDATION_EMB = "foundation_embeddings"
GRAPH_EMB = "graph"
HEAVY_COST = 600.0
MEDIUM_COST = 60.0


def _cost_tier(runtime: float) -> str:
    if runtime >= HEAVY_COST:
        return "expensive"
    if runtime >= MEDIUM_COST:
        return "medium"
    return "cheap"


def build_registry_facts(registry: Any, *, estimate_cost=None) -> list[Fact]:
    """Walk model + representation registries, emit capability/requirement facts.

    Facts are (plugin_name, predicate, object) triples: is_kind, cost_tier,
    requires (graph/embedding), supports (task types) and applicability edges
    between model and representation plugins.
    """
    facts: list[Fact] = []

    model_plugins = _plugins(registry, "model")
    rep_plugins = _plugins(registry, "representation")

    if estimate_cost is None:
        estimate_cost = _rough_cost

    for rep in rep_plugins:
        name = rep.name
        kind = getattr(rep, "kind", "descriptor")
        facts.append(Fact(subject=name, predicate="is_kind", object=kind, level=3, source="registry"))
        runtime = None
        with contextlib.suppress(TypeError, ValueError, KeyError):
            runtime = estimate_cost(name, "ridge", 200, 2_000)
        facts.append(
            Fact(
                subject=name, predicate="cost_tier",
                object=_cost_tier(runtime if runtime is not None else 1.0),
                level=3, source="registry:estimate_cost",
            )
        )

    for model in model_plugins:
        name = model.name
        kind = getattr(model, "kind", "classical")
        facts.append(Fact(subject=name, predicate="model_kind", object=kind, level=3, source="registry"))
        if name in GRAPH_MODELS:
            facts.append(Fact(subject=name, predicate="requires", object=GRAPH_EMB, level=3, source="registry"))
        if name in FOUNDATION_MODELS:
            facts.append(
                Fact(
                    subject=name, predicate="requires",
                    object=FOUNDATION_EMB, level=3, source="registry",
                )
            )
        supports = getattr(model, "supports", ())
        if callable(supports):
            supports = supports()
        for task in supports or ():
            if isinstance(task, str):
                facts.append(Fact(subject=name, predicate="supports", object=task, level=3, source="registry"))
        runtime = None
        with contextlib.suppress(TypeError, ValueError, KeyError):
            runtime = estimate_cost("morgan", name, 200, 2_000)
        facts.append(
            Fact(
                subject=name, predicate="cost_tier",
                object=_cost_tier(runtime if runtime is not None else 1.0),
                level=3, source="registry:estimate_cost",
            )
        )

    # Cross edges: representation x model applicability
    for rep in rep_plugins:
        for model in model_plugins:
            if _applicable(rep.name, model.name):
                facts.append(
                    Fact(subject=f"{model.name}+{rep.name}", predicate="applicable", object="true", level=3, source="registry")
                )
    return facts


def _plugins(registry: Any, kind: str) -> list[Any]:
    """Pull plugin instances for the kind, skipping unavailable plugins."""
    list_fn = getattr(registry, "list", None)
    get_fn = getattr(registry, "get", None)
    if list_fn is None or get_fn is None:
        return []
    try:
        names = list_fn(kind)
    except TypeError:
        names = list_fn(kind, include_unavailable=False)
    items: list[Any] = []
    for name in names:
        try:
            items.append(get_fn(kind, name))
        except Exception:  # noqa: BLE001 - plugin import failure, skip
            continue
    return items


def _applicable(rep: str, model: str) -> bool:
    requires: list[str] = []
    if model in GRAPH_MODELS:
        requires.append(GRAPH_EMB)
    if model in FOUNDATION_MODELS:
        requires.append(FOUNDATION_EMB)
    return not (
        (requires and rep not in requires)
        or (rep == GRAPH_EMB and model not in GRAPH_MODELS)
        or (rep == FOUNDATION_EMB and model not in FOUNDATION_MODELS)
    )


def _rough_cost(rep: str, model: str, n_features: int, n_samples: int) -> float:
    """Heuristic seconds estimate when the real cost estimator is unavailable."""
    estimate = 1.0 + n_features / 200.0
    if model in {"xgboost", "lightgbm", "mpnn"}:
        estimate += 20.0
    if model in {"gcn", "gat"}:
        estimate += 60.0
    if rep == "graph":
        estimate += 30.0
    return estimate


def registry_version(registry: Any) -> str:
    """Stable fingerprint of the plugin catalog."""
    models = sorted(p.name for p in _plugins(registry, "model"))
    reps = sorted(p.name for p in _plugins(registry, "representation"))
    payload = {"models": models, "representations": reps}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]