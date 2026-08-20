"""Explainability: machine-attributable decision traces and grounded prose.

Every planner decision records WHY (evidence consulted, fallback visited,
winner-boost deltas) so results are auditable without trusting a model.
Explanations are template-generated from structured trace fields - the LLM is
never the source of justification.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cta_qsar.knowledge.facts import EvidenceStore, Fact

DISPLAY_LOWER_BETTER = {"rmse", "mae", "mse"}


def _metric_label(predicate: str) -> str:
    return predicate or "primary"


def _fmt(fact: Fact) -> str:
    mean = fact.attrs.get("mean", 0.0)
    std = fact.attrs.get("std", 0.0)
    n = fact.attrs.get("n", 0)
    return f"{_metric_label(fact.predicate) or 'any'}[{fact.object}] mean={mean:.3f} (+/-{std:.3f}) n={n}"


def render_evidence_board(facts: list[Fact]) -> str:
    """Compact table of the evidence the planner was given this round."""
    if not facts:
        return "(no accumulated evidence for this dataset class)"
    lines = ["Evidence consulted (fine -> coarse):"]
    for fact in facts:
        lines.append(f"  - {fact.subject} / {fact.predicate or '*'} / {fact.object}: {_fmt(fact)}")
    return "\n".join(lines)


def explain_decision(trace: dict[str, Any]) -> str:
    """Template the decision trace into grounded prose (no LLM)."""
    chosen = trace.get("chosen", "")
    lines = [f"Round {trace.get('round', '?')}: chose strategy '{chosen}' ({trace.get('reason', '')})."]
    if trace.get("evidence"):
        lines.append("Reasoning used these evidence edges:")
        for fact in trace["evidence"]:
            lines.append(f"  - {fact['subject']} / {fact['predicate'] or '*'} / {fact['object']}: "
                         f"{fact['attrs'].get('mean', 0.0):.3f} (n={fact['attrs'].get('n', 0)})")
    if trace.get("winner_boost") is not None:
        lines.append(f"  winner-boost applied: +{trace['winner_boost']} to current best strategy {trace.get('winner')}")
    if trace.get("adjacency"):
        lines.append(f"  adjacency expansion considered: {', '.join(trace['adjacency'])}")
    return "\n".join(lines)


def counterfactual_report(store: EvidenceStore, dataset_class: str, predicate: str | None = None) -> str:
    """What would the planner have picked without the winning strategy (drop-1)."""
    if predicate is None:
        best = store.best_for(dataset_class, list(store._predicates(dataset_class))[0]) if store._predicates(dataset_class) else None
    else:
        best = store.best_for(dataset_class, predicate)
    if not best:
        return "no counterfactual available (insufficient evidence)"
    winner, attrs = best
    alternatives = store.facts_for(dataset_class, exclude_predicate=predicate)
    return (
        f"Without {winner}, the planner would rely on:\n"
        + (render_evidence_board(alternatives) if alternatives else "  (no other qualified evidence - falls back to heuristics)")
    )


def append_trace(trace: dict[str, Any], path: str | Path) -> None:
    """Append one machine-readable decision trace to a JSONL log (audit trail)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **trace}
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def build_trace(
    *,
    round_: int,
    chosen: str,
    reason: str,
    evidence: list[Fact],
    winner: str | None = None,
    winner_boost: float | None = None,
    adjacency: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "round": round_, "chosen": chosen, "reason": reason,
        "evidence": [f.to_dict() for f in evidence],
        "winner": winner, "winner_boost": winner_boost, "adjacency": adjacency or [],
    }