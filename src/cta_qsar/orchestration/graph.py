"""LangGraph assembly of the cyclic QSAR scientist workflow.

All nodes are plain functions (see ``nodes.py``); this module only wires them
with conditional edges so the graph can return to ``plan_experiment`` after a
failure --- the defining loop of self-correction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from cta_qsar.core.config import Config
from cta_qsar.core.logging import get_logger
from cta_qsar.core.registry import PluginRegistry, get_registry
from cta_qsar.core.state import QSARState

logger = get_logger(__name__)


def _node(impl: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[QSARState], QSARState]:
    """Adapt a dict->dict node to the QSARState type annotation."""

    def adapter(state: QSARState) -> QSARState:
        return impl(dict(state))

    return adapter


def build_graph(
    *,
    config: Config,
    llm: Any = None,
    registry: PluginRegistry | None = None,
    output_root: str = "runs",
) -> Any:
    """Construct the compiled LangGraph workflow."""
    from cta_qsar.orchestration import nodes as n
    from cta_qsar.orchestration.routing import (
        route_from_diagnosis,
        route_from_execute,
        route_from_trust,
    )

    if registry is None:
        registry = get_registry()
        registry.auto_discover()

    graph = StateGraph(QSARState)
    graph.add_node("ingest_dataset", _node(n.ingest_dataset))
    graph.add_node("profile_dataset", _node(n.profile_dataset_node))
    graph.add_node("detect_endpoint", _node(n.detect_endpoint))
    graph.add_node("standardize_dataset", _node(n.standardize_dataset))
    graph.add_node("assess_data_quality", _node(n.assess_data_quality))
    graph.add_node("characterize_chemical_space", _node(n.characterize_chemical_space))
    graph.add_node("select_validation", _node(n.select_validation))
    graph.add_node("generate_candidate_representations", _node(n.generate_candidate_representations))
    graph.add_node("generate_candidate_models", _node(n.generate_candidate_models))
    graph.add_node("plan_experiment", _node(n.plan_experiment))
    graph.add_node("execute_experiment", _node(n.execute_experiment))
    graph.add_node("evaluate_performance", _node(n.evaluate_performance))
    graph.add_node("evaluate_trust", _node(n.evaluate_trust))
    graph.add_node("diagnose_failure", _node(n.diagnose_failure))
    graph.add_node("propose_intervention", _node(n.propose_intervention))
    graph.add_node("finalize_report", _node(n.finalize_report))

    graph.set_entry_point("ingest_dataset")

    graph.add_edge("ingest_dataset", "profile_dataset")
    graph.add_edge("profile_dataset", "detect_endpoint")
    graph.add_edge("detect_endpoint", "standardize_dataset")
    graph.add_edge("standardize_dataset", "assess_data_quality")
    graph.add_edge("assess_data_quality", "characterize_chemical_space")
    graph.add_edge("characterize_chemical_space", "select_validation")
    graph.add_edge("select_validation", "generate_candidate_representations")
    graph.add_edge("generate_candidate_representations", "generate_candidate_models")
    graph.add_edge("generate_candidate_models", "plan_experiment")

    graph.add_conditional_edges(
        "plan_experiment",
        _plan_done,
        {"plan": "execute_experiment", "finalize": "finalize_report"},
    )

    graph.add_conditional_edges(
        "execute_experiment",
        route_from_execute,
        {"evaluate_performance": "evaluate_performance", "plan_experiment": "plan_experiment"},
    )

    graph.add_edge("evaluate_performance", "evaluate_trust")
    graph.add_conditional_edges("evaluate_trust", route_from_trust, {"diagnose_failure": "diagnose_failure", "finalize_report": "finalize_report"})
    graph.add_conditional_edges("diagnose_failure", route_from_diagnosis, {"propose_intervention": "propose_intervention"})

    graph.add_conditional_edges(
        "propose_intervention",
        n.decide_next_action,
        {"plan_experiment": "plan_experiment", "finalize_report": "finalize_report"},
    )

    graph.add_edge("finalize_report", END)

    compiled = graph.compile()

    # Thread context (registry, llm, config) via contextvars: these objects
    # cannot travel in the state channels.
    from cta_qsar.core.context import ExecutionContext, set_context

    original = compiled.invoke

    def invoke_with_deps(state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        set_context(
            ExecutionContext(
                config=config,
                registry=registry,
                llm=llm,
                output_root=output_root,
            )
        )
        merged = {"config": config, "output_dir": output_root, **state}
        return original(merged, *args, **kwargs)

    compiled.invoke = invoke_with_deps  # type: ignore[method-assign]
    return compiled


def _plan_done(state: QSARState) -> str:
    if state.get("selected_candidate") is None:
        return "finalize"
    return "plan"