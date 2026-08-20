"""Deterministic chemical data-quality checks.

Distinguishes *data errors* from *scientifically valid extreme values*:
outliers are flagged and characterized, never auto-removed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def quality_report(
    df: pd.DataFrame,
    *,
    smiles_column: str,
    target_column: str | None,
    task_type: str,
    endpoint: dict[str, Any],
) -> dict[str, Any]:
    """Compute the full chemical/data quality report."""
    n_rows = len(df)
    report: dict[str, Any] = {
        "n_rows": n_rows,
        "n_columns": df.shape[1],
        "missing_values": _missing_values(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_molecules": _duplicate_molecules(df, smiles_column),
        "conflicting_labels": _conflicting_labels(df, smiles_column, target_column),
        "invalid_smiles": int((~df.get("smiles_valid", pd.Series(True, index=df.index))).sum())
        if "smiles_valid" in df
        else 0,
        "class_balance": _class_balance(df, target_column, task_type),
        "target_distribution": _target_distribution(df, target_column, task_type),
        "outliers": _outliers(df, target_column, task_type),
        "missing_value_columns": sorted(
            {
                col
                for col in df.columns
                if df[col].isna().sum() > 0
            }
        ),
    }
    return report


def _missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum()
    total = {
        "n_cells_missing": int(missing.sum()),
        "n_rows_with_missing": int(df.isna().any(axis=1).sum()),
    }
    columns = {
        col: int(n) for col, n in missing.items() if n > 0
    }
    total["columns"] = columns
    return total


def _duplicate_molecules(df: pd.DataFrame, smiles_column: str) -> dict[str, Any]:
    if smiles_column not in df.columns or df.empty:
        return {"use_canonical": False, "n_duplicates": 0}
    canonical_col = "standardized_smiles" if "standardized_smiles" in df else smiles_column
    counts = df[canonical_col].value_counts()
    dupes = counts[counts > 1]
    return {
        "use_canonical": canonical_col != smiles_column,
        "n_duplicates": int((counts > 1).sum()),
        "n_duplicate_rows": int((dupes - 1).sum()),
        "most_duplicated": [
            {"smiles": k, "times": int(v)} for k, v in dupes.head(5).items()
        ],
    }


def _conflicting_labels(
    df: pd.DataFrame, smiles_column: str, target_column: str | None
) -> dict[str, Any]:
    if target_column is None or target_column not in df.columns:
        return {"n_conflicting_groups": 0}
    canonical_col = "standardized_smiles" if "standardized_smiles" in df else smiles_column
    if canonical_col not in df.columns:
        return {"n_conflicting_groups": 0}
    groups = df.groupby(canonical_col, dropna=False)[target_column].nunique()
    conflicted = groups[groups > 1]
    return {
        "n_conflicting_groups": int(len(conflicted)),
        "n_conflicting_rows": int(df[canonical_col].isin(conflicted.index).sum()),
        "examples": [
            {
                "smiles": smiles,
                "targets": df.loc[df[canonical_col] == smiles, target_column]
                .astype(str)
                .unique()
                .tolist(),
            }
            for smiles in conflicted.index[:5]
        ],
    }


def _class_balance(
    df: pd.DataFrame, target_column: str | None, task_type: str
) -> dict[str, Any]:
    if target_column is None or target_column not in df.columns:
        return {"applicable": False}
    if task_type not in ("binary", "multiclass", "multitask_classification"):
        return {"applicable": False}
    counts = df[target_column].value_counts().sort_index()
    return {
        "applicable": True,
        "classes": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else float("inf"),
        "minority_fraction": float(counts.min() / counts.sum()),
    }


def _target_distribution(
    df: pd.DataFrame, target_column: str | None, task_type: str
) -> dict[str, Any]:
    if target_column is None or target_column not in df.columns:
        return {"applicable": False}
    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.notna().sum() == 0:
        return {"applicable": False}
    return {
        "applicable": True,
        "task_type": task_type,
        "n": int(y.notna().sum()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "skewness": float(stats.skew(y.dropna())),
        "kurtosis": float(stats.kurtosis(y.dropna())),
    }


def _outliers(
    df: pd.DataFrame, target_column: str | None, task_type: str
) -> dict[str, Any]:
    """Flag distributional extremes WITHOUT recommending removal."""
    if target_column is None or target_column not in df.columns:
        return {"applicable": False}
    y = pd.to_numeric(df[target_column], errors="coerce").dropna()
    if len(y) < 10:
        return {"applicable": False}
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return {"applicable": False}
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    extreme = y[(y < lower) | (y > upper)]
    return {
        "applicable": True,
        "method": "iqr_tukey",
        "n_extreme": int(len(extreme)),
        "fraction_extreme": float(len(extreme) / len(y)),
        "extreme_bounds": {"lower": float(lower), "upper": float(upper)},
        "note": "Extreme values are flagged, NOT removed; they may be scientifically valid.",
        "extremes": [
            {"index": int(i), "value": float(v)} for i, v in extreme.head(10).items()
        ],
    }