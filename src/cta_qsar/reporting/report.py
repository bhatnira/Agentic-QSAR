"""Final scientific report generation.

The report is a reproducible summary that explicitly states *why the agent
stopped* and uses only claims supported by executed experiments.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.core.logging import get_logger

logger = get_logger(__name__)


def build_report(state: dict[str, Any], *, stop_reasons: list[str], llm_summary: str = "") -> dict[str, Any]:
    """Assemble the full report dict from graph state."""
    experiments = [_plain(e) for e in state.get("experiments", [])]
    completed = [e for e in experiments if e.get("result") == "completed"]
    best = _best_experiment(completed, state.get("endpoint", {}).get("task_type", "regression"))

    report: dict[str, Any] = {
        "run_id": state.get("run_id", ""),
        "dataset": state.get("data_path", ""),
        "dataset_profile": {
            "n_rows": state.get("profile", {}).get("n_rows", 0),
            "n_columns": state.get("profile", {}).get("n_columns", 0),
            "smiles_column": state.get("smiles_column", ""),
            "target_column": state.get("target_column", ""),
        },
        "endpoint": state.get("endpoint", {}),
        "data_quality": state.get("quality_report", {}),
        "standardization": state.get("standardization_log", {}),
        "chemical_space": state.get("chemical_space", {}),
        "validation_strategy": {
            "enabled": state.get("validated_splits", []),
            "rationale": state.get("validation_rationale", ""),
        },
        "representations_considered": state.get("representation_candidates", []),
        "models_considered": state.get("model_candidates", []),
        "experiments": experiments,
        "performance_comparison": _performance_table(completed),
        "generalization_results": _plugin_summary(completed, "generalization"),
        "applicability_domain_results": _plugin_summary(completed, "applicability_domain"),
        "uncertainty_results": _plugin_summary(completed, "uncertainty"),
        "explainability_results": _plugin_summary(completed, "explainability"),
        "failure_diagnoses": [_plain(d) for d in state.get("diagnoses", [])],
        "self_correction_experiments": [
            e for e in experiments if e.get("llm_decision") == "self_correcting"
        ],
        "computational_cost": {
            "experiments_done": len(completed),
            "total_experiments": len(experiments),
            "max_experiments": state.get("config", {}).compute.max_experiments
            if hasattr(state.get("config"), "compute")
            else 12,
            "stop_reasons": stop_reasons,
        },
        "best_experiment": best,
        "final_model_selection_rationale": _selection_rationale(best, completed, state),
        "limitations": _limitations(state, completed),
        "recommended_next_experiments": _recommendations(state, best),
    }
    if llm_summary:
        report["executive_summary"] = llm_summary
    return report


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _best_experiment(completed: list[dict[str, Any]], task_type: str) -> dict[str, Any] | None:
    if not completed:
        return None
    metric = "rmse" if task_type == "regression" else "roc_auc" if task_type == "binary" else "mcc"
    lower_better = metric in ("rmse", "mae")
    scored = [e for e in completed if metric in e.get("metrics", {})]
    if not scored:
        return completed[-1]
    scored.sort(key=lambda e: e["metrics"][metric], reverse=not lower_better)
    return scored[0]


def _performance_table(completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for e in completed:
        metrics = e.get("metrics", {})
        rows.append(
            {
                "experiment": f"{e.get('representation')}+{e.get('model')}[{e.get('split')}]",
                "metrics": metrics,
                "runtime_seconds": e.get("runtime_seconds", 0.0),
            }
        )
    return rows


def _plugin_summary(completed: list[dict[str, Any]], plugin: str) -> list[dict[str, Any]]:
    rows = []
    for e in completed:
        block = e.get("trust", {}).get(plugin, {})
        if not block:
            continue
        rollup: dict[str, Any] = {}
        for key, value in block.items():
            if isinstance(value, dict) and "mean" in value:
                rollup[key] = round(float(value["mean"]), 4)
            elif key in ("interpretation", "evaluated"):
                rollup[key] = value
        rows.append(
            {
                "experiment": f"{e.get('representation')}+{e.get('model')}[{e.get('split')}]",
                **rollup,
            }
        )
    return rows


def _selection_rationale(
    best: dict[str, Any] | None, completed: list[dict[str, Any]], state: dict[str, Any]
) -> str:
    if best is None:
        return "No completed experiments; a scientific claim cannot be made."
    task_type = state.get("endpoint", {}).get("task_type", "regression")
    metric = "RMSE" if task_type == "regression" else "ROC-AUC" if task_type == "binary" else "MCC"
    value = best.get("metrics", {}).get("rmse" if metric == "RMSE" else "roc_auc" if metric == "ROC-AUC" else "mcc")
    return (
        f"Best observed strategy under the specified budget: "
        f"{best.get('representation')} + {best.get('model')} with "
        f"{best.get('split')} validation ({metric}={value}). "
        f"Selection is budget-constrained; it is not claimed to be globally optimal."
    )


def _limitations(state: dict[str, Any], completed: list[dict[str, Any]]) -> list[str]:
    limitations: list[str] = []
    endpoint = state.get("endpoint", {})
    if endpoint.get("confidence", 1.0) < 0.7:
        limitations.append("Endpoint type/units inferred with limited confidence.")
    quality = state.get("quality_report", {})
    if quality.get("invalid_smiles", 0):
        limitations.append(f"{quality['invalid_smiles']} molecules had invalid SMILES and were excluded.")
    if quality.get("conflicting_labels", {}).get("n_conflicting_groups", 0):
        limitations.append("Conflicting labels were present; model may inherit label noise.")
    if len(completed) < 3:
        limitations.append("Few experiments were feasible under the budget; conclusions have low coverage.")
    return limitations


def _recommendations(state: dict[str, Any], best: dict[str, Any] | None) -> list[str]:
    recs: list[str] = []
    if best:
        recs.append(
            f"Re-run the best strategy ({best.get('representation')}+{best.get('model')}) "
            "with repeated CV and multiple seeds for tighter confidence intervals."
        )
    diagnoses = state.get("diagnoses", [])
    for diagnosis in diagnoses[:2]:
        if hasattr(diagnosis, "recommended_actions"):
            recs.extend(diagnosis.recommended_actions[:2])
    recs.append("Expand the applicability domain with structurally closer examples.")
    return recs