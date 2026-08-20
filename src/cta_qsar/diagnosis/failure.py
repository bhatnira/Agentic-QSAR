"""Deterministic failure-diagnosis rules.

Each rule consumes the experiment's trust report and produces a
:class:`FailureDiagnosis` with evidence, hypothesis, confidence, and
recommended actions.  LLM refinement happens at the diagnosis agent level.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.core.interfaces import ExperimentRecord, FailureDiagnosis


def diagnose(experiment: ExperimentRecord, trust_report: dict[str, Any]) -> list[FailureDiagnosis]:
    """Run all rules; return deterministic diagnoses ordered by confidence."""
    diagnoses: list[FailureDiagnosis] = []
    for rule in (
        r1_random_vs_scaffold,
        r2_overfitting,
        r3_class_imbalance,
        r4_limited_ad,
        r5_instability,
    ):
        result = rule(experiment, trust_report)
        if result is not None:
            diagnoses.append(result)
    diagnoses.sort(key=lambda d: d.confidence, reverse=True)
    return diagnoses


def _metric(trust_report: dict[str, Any], plugin: str, key: str) -> float | None:
    block = trust_report.get(plugin, {})
    value = block.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _summary_value(trust_report: dict[str, Any], plugin: str, key: str, default: float = 0.0) -> float:
    value = _metric(trust_report, plugin, key)
    if value is None or (isinstance(value, float) and value != value):
        return default
    return float(value)


def r1_random_vs_scaffold(experiment: ExperimentRecord, trust_report: dict[str, Any]) -> FailureDiagnosis | None:
    """Random split strong + scaffold split weak => possible chemical-series dependence."""
    random_r2 = _summary_value(trust_report, "predictive", "r2", 0.0)
    scaffold_r2 = _summary_value(trust_report, "generalization", "r2", 0.0)
    if random_r2 >= 0.6 and scaffold_r2 < 0.3:
        return FailureDiagnosis(
            failure_type="chemical_series_dependence",
            evidence={"random_split_r2": random_r2, "scaffold_split_r2": scaffold_r2},
            hypothesis=(
                "Performance collapses under scaffold-exclusive splits: predictions "
                "likely rely on series-internal patterns rather than learned chemistry."
            ),
            confidence=0.8,
            recommended_actions=[
                "switch to a more transferable representation (descriptors, graph)",
                "increase regularization (reduce model capacity)",
                "use scaffold-aware model selection (report scaffold as primary)",
            ],
        )
    return None


def r2_overfitting(experiment: ExperimentRecord, trust_report: dict[str, Any]) -> FailureDiagnosis | None:
    """Train strong + validation weak => possible overfitting."""
    train = _summary_value(trust_report, "train", "r2", None)
    val = _summary_value(trust_report, "predictive", "r2", 0.0)
    if train is None:
        return None
    delta = train - val
    if delta > 0.25 and val < 0.5:
        return FailureDiagnosis(
            failure_type="overfitting",
            evidence={"train_r2": train, "validation_r2": val, "gap": delta},
            hypothesis=f"Training/validation gap of {delta:.2f} R2 indicates memorization.",
            confidence=0.7,
            recommended_actions=[
                "stronger regularization (lower C/alpha, lower max_depth)",
                "fewer features (subset of descriptors or feature selection)",
                "more conservative CV (repeated folds, seed averaging)",
            ],
        )
    return None


def r3_class_imbalance(experiment: ExperimentRecord, trust_report: dict[str, Any]) -> FailureDiagnosis | None:
    """Accuracy strong + PR-AUC poor => possible class imbalance."""
    accuracy = _summary_value(trust_report, "predictive", "accuracy", 0.0)
    pr_auc = _summary_value(trust_report, "predictive", "pr_auc", 0.0)
    if accuracy > 0.8 and 0 < pr_auc < 0.4:
        return FailureDiagnosis(
            failure_type="class_imbalance",
            evidence={"accuracy": accuracy, "pr_auc": pr_auc},
            hypothesis=(
                "High accuracy with weak PR-AUC characterizes models that ignore "
                "the minority class."
            ),
            confidence=0.75,
            recommended_actions=[
                "use class_weight='balanced' or resampling",
                "report PR-AUC / MCC as primary metrics",
                "optimize the decision threshold on validation PR curves",
            ],
        )
    return None


def r4_limited_ad(experiment: ExperimentRecord, trust_report: dict[str, Any]) -> FailureDiagnosis | None:
    """Performance strong + OOD/AD poor => limited applicability domain."""
    perf = max(
        _summary_value(trust_report, "predictive", "r2", 0.0),
        _summary_value(trust_report, "predictive", "roc_auc", 0.0),
    )
    ad = trust_report.get("applicability_domain", {})
    if not ad.get("evaluated"):
        return None
    nn = ad.get("nn_tanimoto", {})
    median = float(nn.get("median", 1.0)) if isinstance(nn, dict) else 1.0
    if perf >= 0.6 and median < 0.35:
        return FailureDiagnosis(
            failure_type="limited_applicability_domain",
            evidence={"median_nn_tanimoto": median, "performance": perf},
            hypothesis=(
                "Test molecules are chemically distant from the training set; "
                "high average performance hides low-confidence regions."
            ),
            confidence=0.7,
            recommended_actions=[
                "report per-prediction confidence via uncertainty plugin",
                "restrict the model to its applicability domain when deployed",
                "add structurally similar examples to training",
            ],
        )
    return None


def r5_instability(experiment: ExperimentRecord, trust_report: dict[str, Any]) -> FailureDiagnosis | None:
    """Explanation unstable + performance variable => unstable representation/model."""
    robustness = trust_report.get("robustness", {})
    expl = trust_report.get("explainability", {})
    if not robustness.get("evaluated") or not expl.get("evaluated"):
        return None
    cv = float(robustness.get("coefficient_of_variation", 0.0))
    unstable_top = any(
        f.get("stability", 0.0) > 3.0 for f in expl.get("top_features", [])[:5]
    )
    if cv > 0.15 and unstable_top:
        return FailureDiagnosis(
            failure_type="unstable_representation",
            evidence={"cv": cv, "top_feature_stability_unstable": unstable_top},
            hypothesis=(
                "Both performance and attributions vary across seeds/folds; "
                "the representation+model pairing is unstable."
            ),
            confidence=0.6,
            recommended_actions=[
                "change representation family",
                "average predictions over seeds (ensemble)",
                "prefer simpler models with stable attributions",
            ],
        )
    return None


class FailureDiagnosisPlugin:
    """Registry-facing wrapper around the deterministic rules."""

    name = "failure_rules"

    def diagnose(self, trust_report: dict[str, Any], experiment: ExperimentRecord) -> list[FailureDiagnosis]:
        return diagnose(experiment, trust_report)


PLUGINS = [FailureDiagnosisPlugin]