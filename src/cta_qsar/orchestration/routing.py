"""Conditional routing logic that connects graph nodes.

The scientist is a single agent with deterministic tool-like nodes; routing is
based on budget state and observed evidence, and remains independent of which
plugins exist.
"""

from __future__ import annotations

from typing import Any


def route_from_profile(state: dict[str, Any]) -> str:
    """After profiling: if target column is missing -> ask/user error else standardize."""
    if not state.get("target_column"):
        return "error_target_column"
    return "standardize_dataset"


def route_from_execute(state: dict[str, Any]) -> str:
    """After executing: evaluate if completed, else retry/plan."""
    if state.get("failed_experiment") is not None or state.get("error"):
        return "plan_experiment"  # consume failure; do not crash the loop
    return "evaluate_performance"


def route_from_trust(state: dict[str, Any]) -> str:
    if not state.get("current_experiment"):
        return "finalize_report"
    return "diagnose_failure"


def route_from_diagnosis(state: dict[str, Any]) -> str:
    return "propose_intervention"


def route_next(state: dict[str, Any]) -> str:
    """Main decision point: continue vs finalize."""
    decisions = state.get("stop_reasons", [])
    if decisions:
        return "finalize_report"
    budget = state.get("budget_state", {}) or {}
    done = len(state.get("experiments", []))
    if done >= (budget.get("max_experiments") or 12):
        return "finalize_report"
    return "plan_experiment"