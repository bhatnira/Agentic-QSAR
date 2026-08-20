"""Heuristic + LLM-assisted experiment candidate selection.

The planner constructs candidate (representation, model, validation) triples,
scores them with a utility =
    (expected_improvement + information_gain + trust_gain) / compute_cost

and consults the LLM (when configured) for the final ranked selection.  The
heuristics alone are sufficient for a complete, working run without any API.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.core.interfaces import ExperimentCandidate, QSARCase
from cta_qsar.core.logging import get_logger
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.models.registry import estimate_model_cost
from cta_qsar.representations.registry import estimate_rep_cost

logger = get_logger(__name__)

KIND_ORDER = {"fingerprint": 0, "descriptors": 1, "graph": 2, "embedding": 3}


def generate_candidates(
    *,
    registry: PluginRegistry,
    case: QSARCase,
    enabled_representations: list[str] | None,
    enabled_models: list[str] | None,
    validated_splits: list[str],
    budget: BudgetState,
    history: list[dict[str, Any]],
    task_type: str,
    n_samples: int,
    dataset_props: dict[str, Any],
    hardware_tier: str,
    evidence: list[Any] | None = None,
    policy_weights: dict[str, float] | None = None,
) -> list[ExperimentCandidate]:
    """Generate and score candidate experiments (no LLM required).

    ``evidence`` is a list of knowledge-layer Facts (Finest-first) used to
    boost strategies with proven performance on this dataset class; it is
    advisory only and never blocks a candidate.  ``policy_weights`` are the
    learned (or default) self-improving-planner multipliers applied to the
    three value signals when ranking.
    """
    reps = _resolve_reps(registry, enabled_representations, task_type, n_samples, dataset_props)
    models = _resolve_models(registry, enabled_models, task_type)
    splits = validated_splits or ["random"]
    seen = {_sig(_as_plain(h)) for h in history}

    evidence_bonus = _evidence_bonus_map(evidence or [])
    winner_boost = _winner_boost_map(history)

    candidates: list[ExperimentCandidate] = []
    for rep_name, rep_plugin in reps:
        for model_name, model_plugin in models.items():
            applicable, reason = model_plugin.applicability(task_type, rep_name)
            if not applicable:
                continue
            if model_name in ("gcn", "gat", "mpnn") and rep_name != "graph":
                continue
            n_features = _n_features(rep_plugin, dataset_props, n_samples)
            model_cost = estimate_model_cost(
                registry, model_name, task_type, n_samples, n_features, rep_name
            )
            rep_cost = estimate_rep_cost(rep_plugin, n_samples)
            for split in splits:
                candidate = _score_candidate(
                    rep_name=rep_name,
                    rep_plugin=rep_plugin,
                    model_name=model_name,
                    model_plugin=model_plugin,
                    split=split,
                    rep_cost=rep_cost,
                    model_cost=model_cost,
                    task_type=task_type,
                    n_samples=n_samples,
                    n_features=n_features,
                    history=seen,
                    dataset_props=dataset_props,
                    hardware_tier=hardware_tier,
                    evidence_bonus=evidence_bonus.get(f"{rep_name}|{model_name}", 0.0)
                    + winner_boost.get(f"{rep_name}|{model_name}", 0.0),
                    policy_weights=policy_weights,
                )
                candidates.append(candidate)
    candidates.sort(key=lambda c: c.utility, reverse=True)
    for rank, candidate in enumerate(candidates):
        candidate.rank = rank + 1
    return candidates


def _resolve_reps(
    registry: PluginRegistry,
    enabled: list[str] | None,
    task_type: str,
    n_samples: int,
    dataset_props: dict[str, Any],
) -> list[Any]:
    from cta_qsar.representations.registry import available_representations

    available = available_representations(registry, enabled, task_type, dataset_props)
    ordered = sorted(
        available.items(),
        key=lambda item: (KIND_ORDER.get(getattr(item[1], "kind", "fingerprint"), 9), item[0]),
    )
    if n_samples > 5000:
        ordered = [item for item in ordered if getattr(item[1], "kind", "") not in ("graph", "embedding")]
    return ordered


def _resolve_models(
    registry: PluginRegistry, enabled: list[str] | None, task_type: str
) -> dict[str, Any]:
    from cta_qsar.models.registry import available_models

    return available_models(
        registry, enabled, task_type, "morgan"
    )  # placeholder rep for broad availability


def _n_features(rep_plugin: Any, dataset_props: dict[str, Any], n_samples: int) -> int:
    if getattr(rep_plugin, "kind", "") == "fingerprint":
        return int(getattr(rep_plugin, "n_bits", 2048))
    name = getattr(rep_plugin, "name", "")
    if name in ("rdkit_descriptors", "mordred"):
        return 200 if name == "rdkit_descriptors" else 1600
    if name == "geometry":
        return 2
    return 128


def _score_candidate(
    *,
    rep_name: str,
    rep_plugin: Any,
    model_name: str,
    model_plugin: Any,
    split: str,
    rep_cost: Any,
    model_cost: Any,
    task_type: str,
    n_samples: int,
    n_features: int,
    history: set[str],
    dataset_props: dict[str, Any],
    hardware_tier: str,
    evidence_bonus: float = 0.0,
    policy_weights: dict[str, float] | None = None,
) -> ExperimentCandidate:
    runtime = rep_cost.runtime_seconds + model_cost.runtime_seconds
    memory = rep_cost.memory_gb + model_cost.memory_gb
    compute_cost = runtime + memory * 10.0

    hyper_budget = (
        3 if (model_plugin.hyperparameter_space() or {}) and compute_cost < 30 else 1
    )

    expected_improvement, info_gain, trust_gain, reason = _heuristic_expectations(
        rep_name=rep_name,
        rep_plugin=rep_plugin,
        model_name=model_name,
        model_plugin=model_plugin,
        split=split,
        task_type=task_type,
        n_samples=n_samples,
        history=history,
        dataset_props=dataset_props,
    )
    if evidence_bonus:
        expected_improvement += evidence_bonus
        reason = f"{reason}; evidence: {evidence_bonus:+.2f}"
    candidate = ExperimentCandidate(
        representation=rep_name,
        model=model_name,
        validation=split,
        hyperparameter_budget=hyper_budget,
        estimated_runtime_seconds=round(runtime, 2),
        estimated_memory_gb=round(memory, 3),
        expected_improvement=expected_improvement,
        expected_information_gain=info_gain,
        expected_trustworthiness_gain=trust_gain,
        compute_cost=round(compute_cost, 3),
        reason=reason,
    )
    candidate.compute_utility(policy_weights=policy_weights)
    return candidate


def _heuristic_expectations(
    *,
    rep_name: str,
    rep_plugin: Any,
    model_name: str,
    model_plugin: Any,
    split: str,
    task_type: str,
    n_samples: int,
    history: set[str],
    dataset_props: dict[str, Any],
) -> tuple[float, float, float, str]:
    """Deterministic priors on scientific value.

    Cheap + expressive baseline first; diversity as the loop proceeds.
    """
    done = len(history)
    reason_parts: list[str] = []

    if done == 0:
        if rep_name == "morgan" and model_name == "ridge":
            return 0.6, 0.5, 0.4, "scientific baseline: circular fingerprints + regularized linear model"
        if rep_name == "morgan" and model_name == "random_forest":
            return 0.5, 0.4, 0.5, "cheap non-linear baseline on fingerprints"
        if rep_name == "morgan" and model_name == "xgboost":
            return 0.5, 0.4, 0.5, "gradient boosting on fingerprints"

    kind = getattr(rep_plugin, "kind", "fingerprint")
    rep_novel = done > 0 and not any(h.startswith(f"{rep_name}|") for h in history)
    model_novel = done > 0 and not any(f"|{model_name}|" in h for h in history)

    expected_improvement = 0.15
    info_gain = 0.2
    trust_gain = 0.2

    if task_type == "binary" and dataset_props.get("imbalance_ratio", 1) >= 10 and "class_weight" in dir(model_plugin):
        expected_improvement += 0.15
        reason_parts.append("imbalanced classes: prefer class-weighted models")

    if n_samples < 200 and model_name in ("xgboost", "random_forest"):
        expected_improvement -= 0.05
        reason_parts.append("small data: prefer simpler models")

    if split == "scaffold":
        trust_gain += 0.3
        reason_parts.append("scaffold split: key generalization evidence")
    if split == "random" and done == 0:
        trust_gain += 0.15

    if rep_novel:
        info_gain += 0.2
        reason_parts.append(f"untried representation {rep_name}")
    if model_novel:
        info_gain += 0.15
        reason_parts.append(f"untried model family {model_name}")

    if model_name in ("svr", "xgboost") and rep_name in ("rdkit_fingerprints", "maccs"):
        expected_improvement += 0.05
    if kind in ("descriptors", "embedding") and split == "scaffold":
        expected_improvement += 0.1

    reason = "; ".join(reason_parts) or "baseline exploration"
    return (
        min(expected_improvement, 0.95),
        min(info_gain, 0.95),
        min(trust_gain, 0.95),
        reason,
    )


def _evidence_bonus_map(evidence: list[Any]) -> dict[str, float]:
    """Rank strategies with proven performance (fine facts) over class baseline.

    Facts carry attrs {mean, std, n, sign}: score = signed mean; boost is the
    margin over the class aggregate, scaled by evidence volume.
    """
    if not evidence:
        return {}
    facts = [f for f in evidence if getattr(f, "attrs", {})]
    baseline: float | None = None
    for fact in facts:
        if getattr(fact, "object", "") == "*" and getattr(fact, "predicate", "") == "*":
            baseline = fact.attrs.get("mean", 0.0) * fact.attrs.get("sign", 1.0)
    best: dict[str, float] = {}
    for fact in facts:
        if getattr(fact, "object", "") in ("*", ""):
            continue
        n = fact.attrs.get("n", 0)
        if n < 2:
            continue
        score = fact.attrs.get("mean", 0.0) * fact.attrs.get("sign", 1.0)
        obj = str(fact.object)
        rep, _, model = obj.partition("+")
        model = model.split("[")[0] if "[" in model else model
        margin = score - (baseline if baseline is not None else 0.0)
        if baseline is not None and margin >= 0:
            boost = min(0.25, 0.05 * n + 0.05 * margin)
        else:
            boost = 0.03 * n if rep and model else 0.0
        key = f"{rep}|{model}"
        best[key] = max(best.get(key, 0.0), boost)
    return best


def _winner_boost_map(history: list[Any]) -> dict[str, float]:
    """Adjacency + winner boost around the current best completed experiment.

    The winner's exact (representation, model) gets a small boost so the loop
    refines near proven performance; sibling splits of the same strategy are
    already present in the candidate pool (adjacency by construction).
    """
    plain = [_as_plain(h) for h in history if hasattr(h, "model_dump") or isinstance(h, dict)]
    completed = [r for r in plain if r.get("result") == "completed"]
    if not completed:
        return {}
    best_run: dict[str, Any] | None = None
    for record in completed:
        value = record.get("primary_value")
        if value is None:
            continue
        primary = record.get("primary") or "rmse"
        sign = 1.0 if _higher_is_better(primary) else -1.0
        score = sign * float(value)
        if best_run is None or score > best_run["_score"]:
            best_run = {**record, "_score": score}
    if best_run is None:
        return {}
    winner_rep = best_run.get("representation", "")
    winner_model = best_run.get("model", "")
    if not winner_rep or not winner_model:
        return {}
    return {f"{winner_rep}|{winner_model}": 0.15}


def _higher_is_better(primary: str) -> bool:
    primary = (primary or "").lower()
    return any(
        token in primary
        for token in ("auc", "acc", "f1", "recall", "kappa", "truth", "balanced", "precision")
    ) and not any(t in primary for t in ("rmse", "mae", "mse"))


def _sig(history_row: dict[str, Any]) -> str:
    return (
        f"{history_row.get('representation')}|{history_row.get('model')}|"
        f"{history_row.get('split')}|{history_row.get('seed', 42)}"
    )


def _as_plain(row: Any) -> dict[str, Any]:
    """Normalize pydantic records (or dicts) into plain dicts."""
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return {}


def pick_first_usable(
    candidates: list[ExperimentCandidate],
    budget: BudgetState,
    history: list[Any],
) -> ExperimentCandidate | None:
    """Pick the highest-utility candidate not already executed or forbidden.

    Returns None once the experiment or compute-time budget is exhausted.
    """
    if budget.experiments_done >= budget.max_experiments:
        return None
    seen = {_sig(_as_plain(h)) for h in history}
    for candidate in candidates:
        sig = f"{candidate.representation}|{candidate.model}|{candidate.validation}|42"
        if sig in seen:
            continue
        if candidate.estimated_runtime_seconds > budget.max_minutes * 60 * 0.8:
            continue
        return candidate
    return None


def explain_decisions(candidates: list[ExperimentCandidate], top_k: int = 5) -> str:
    """Human-readable summary of candidate ranking decisions."""
    lines = []
    for candidate in candidates[:top_k]:
        lines.append(
            f"  #{candidate.rank} {candidate.representation}+{candidate.model} "
            f"[{candidate.validation}] value={candidate.utility:.3f} "
            f"cost={candidate.compute_cost:.1f}s-eq reason={candidate.reason}"
        )
    return "\n".join(lines) if lines else "  (no candidates)"