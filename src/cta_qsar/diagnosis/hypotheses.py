"""Hypothesis templates for failure diagnoses (shared vocabulary)."""

from __future__ import annotations

HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "chemical_series_dependence": (
        "Predictive power comes from within-chemical-series patterns; "
        "model generalizes poorly to unseen scaffolds."
    ),
    "overfitting": (
        "Model capacity exceeds the information content of the dataset; "
        "training signal is memorized rather than learned."
    ),
    "class_imbalance": (
        "Minority class is under-represented; accuracy is dominated by "
        "majority-class prediction."
    ),
    "limited_applicability_domain": (
        "The test space lies partly outside the chemically populated training "
        "space; extrapolation is unconstrained."
    ),
    "unstable_representation": (
        "The representation exposes noise to the model; attributions and "
        "performance fluctuate across resampling."
    ),
    "insufficient_data": (
        "Dataset size limits statistical power for complex models."
    ),
    "label_noise": (
        "Assayed values contain inconsistencies (duplicate/conflicting "
        "labels, aggregation artifacts)."
    ),
}

EVIDENCE_KEYS: dict[str, tuple[str, ...]] = {
    "chemical_series_dependence": ("random_split_r2", "scaffold_split_r2"),
    "overfitting": ("train_r2", "validation_r2", "gap"),
    "class_imbalance": ("accuracy", "pr_auc"),
    "limited_applicability_domain": ("median_nn_tanimoto", "performance"),
    "unstable_representation": ("cv", "top_feature_stability_unstable"),
}


def hypothesis_for(failure_type: str) -> str:
    return HYPOTHESIS_TEMPLATES.get(
        failure_type,
        "Failure type not in the predefined vocabulary; needs manual examination.",
    )