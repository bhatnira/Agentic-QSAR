"""Orchestration policies: stopping rules and experiment selection."""

from __future__ import annotations

from typing import Any

from cta_qsar.core.logging import get_logger

logger = get_logger(__name__)


def evaluate_stopping(
    *,
    budget: dict[str, Any],
    plan_round: int,
    experiments: list[dict[str, Any]],
    no_improvement_rounds: int,
    settle_delta: float = 0.005,
) -> list[str]:
    """Return a list of stop reasons (empty => continue).

    ``settle_delta`` is the "negligible improvement" threshold below which the
    latest experiment counts as settled; the self-improving planner learns it
    per dataset class from the observed improvement distribution, otherwise
    the hardcoded 0.005 default is used.
    """
    reasons: list[str] = []

    max_experiments = budget.get("max_experiments", 12)
    if len(experiments) >= max_experiments:
        reasons.append("experiment budget exhausted")

    elapsed = budget.get("elapsed_minutes", 0.0)
    max_minutes = budget.get("max_minutes", 30.0)
    if elapsed >= max_minutes:
        reasons.append("compute time budget exhausted")

    if no_improvement_rounds >= 2 and len(experiments) >= 2:
        reasons.append("no meaningful performance improvement over recent experiments")

    if len(experiments) >= 2:
        last = experiments[-1].get("metrics", {})
        prev = experiments[-2].get("metrics", {})
        key = _primary_key(last, prev)
        if key and _improvement(last, prev, key) < settle_delta:
            reasons.append("improvement negligible in latest experiment")
            reasons.append(f"settle-delta policy threshold: {settle_delta}")

    if plan_round >= 4:
        reasons.append("no unexplored applicable strategies remain (round cap)")
    return reasons


def _primary_key(last: dict[str, Any], prev: dict[str, Any]) -> str | None:
    for key in ("rmse", "mae", "roc_auc", "mcc", "balanced_accuracy", "r2"):
        if key in last and key in prev:
            return key
    return None


def _improvement(last: dict[str, Any], prev: dict[str, Any], key: str) -> float:
    if key in ("rmse", "mae"):
        return prev[key] - last[key]  # lower better
    return last[key] - prev[key]


def compute_marginal_gain(experiments: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    """Realized improvement of the latest completed experiment over the previous one.

    Returns ``(gain, primary_key)`` where gain > 0 means better. Used by the
    self-improving planner to learn from the executed trace.
    """
    if len(experiments) < 2:
        return None, None
    last = experiments[-1].get("metrics", {})
    prev = experiments[-2].get("metrics", {})
    key = _primary_key(last, prev)
    if not key:
        return None, None
    return _improvement(last, prev, key), key


def decide_next(
    *,
    budget: dict[str, Any],
    plan_round: int,
    experiments: list[dict[str, Any]],
    no_improvement_rounds: int,
    candidates_remaining: int,
    llm_stop: dict[str, Any] | None = None,
    settle_delta: float = 0.005,
) -> str:
    """Return 'finalize_report' or 'plan_experiment'."""
    reasons = evaluate_stopping(
        budget=budget,
        plan_round=plan_round,
        experiments=experiments,
        no_improvement_rounds=no_improvement_rounds,
        settle_delta=settle_delta,
    )
    if candidates_remaining <= 0 and len(experiments) > 0:
        reasons.append("no unexplored applicable strategies remain")
    if llm_stop and llm_stop.get("should_stop") and len(experiments) >= 2:
        reasons.append(llm_stop.get("reason", "LLM judged further experimentation not worthwhile"))
    if reasons:
        logger.info("Stopping decision: %s", "; ".join(reasons))
        return "finalize_report"
    return "plan_experiment"