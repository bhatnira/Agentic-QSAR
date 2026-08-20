"""Report export: JSON + Markdown on disk (MLflow handled in node)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_report(report: dict[str, Any], run_dir: Path, formats: tuple[str, ...] = ("json", "markdown")) -> dict[str, Path]:
    """Write report artifacts; returns {format: path}."""
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if "json" in formats:
        path = run_dir / "report.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        paths["json"] = path
    if "markdown" in formats:
        path = run_dir / "report.md"
        path.write_text(markdown_report(report))
        paths["markdown"] = path
    return paths


def markdown_report(report: dict[str, Any]) -> str:
    """Render the report dict as a readable scientific report."""
    lines: list[str] = []
    lines.append("# Chemically Trustworthy Agentic QSAR — Final Report")
    lines.append("")
    meta = report.get("dataset_profile", {})
    lines.append(f"**Run ID:** {report.get('run_id', '')}")
    lines.append(f"**Dataset:** {report.get('dataset', '')}")
    lines.append(f"**Rows:** {meta.get('n_rows', 0)} | **Columns:** {meta.get('n_columns', 0)}")
    lines.append("")

    summary = report.get("executive_summary")
    if summary:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(str(summary))
        lines.append("")

    endpoint = report.get("endpoint", {})
    lines.append("## 1. Endpoint Identification")
    lines.append("")
    lines.append(f"- **Task type:** {endpoint.get('task_type', 'unknown')}")
    lines.append(f"- **Endpoint:** {endpoint.get('endpoint_name', 'unknown')}")
    lines.append(f"- **Confidence:** {endpoint.get('confidence', 0.0):.2f}")
    lines.append(f"- **Reasoning:** {endpoint.get('reasoning', '')}")
    lines.append("")

    quality = report.get("data_quality", {})
    lines.append("## 2. Data Quality")
    lines.append("")
    lines.append(f"- Duplicate rows: {quality.get('duplicate_rows', 0)}")
    dupes = quality.get("duplicate_molecules", {})
    lines.append(f"- Duplicate molecules: {dupes.get('n_duplicate_rows', 0)}")
    conflicts = quality.get("conflicting_labels", {})
    lines.append(f"- Conflicting-label groups: {conflicts.get('n_conflicting_groups', 0)}")
    lines.append(f"- Invalid SMILES: {quality.get('invalid_smiles', 0)}")
    outliers = quality.get("outliers", {})
    if outliers.get("applicable"):
        lines.append(
            f"- Extreme target values (flagged, NOT removed): {outliers.get('n_extreme', 0)}"
        )
    lines.append("")

    std = report.get("standardization", {})
    lines.append("## 3. Chemical Standardization")
    lines.append("")
    lines.append(f"- Valid molecules: {std.get('n_valid', 0)}")
    lines.append(f"- Invalid molecules: {std.get('n_invalid', 0)}")
    lines.append(f"- Desalted: {std.get('n_desalted', 0)} | Neutralized: {std.get('n_neutralized', 0)}")
    lines.append("")

    chem = report.get("chemical_space", {})
    lines.append("## 4. Chemical Space")
    lines.append("")
    lines.append(f"- Unique molecules: {chem.get('n_unique_molecules', 0)}")
    if chem.get("mean_nn_similarity") is not None:
        lines.append(f"- Mean nearest-neighbor similarity: {chem['mean_nn_similarity']:.3f}")
    lines.append("")

    validation = report.get("validation_strategy", {})
    lines.append("## 5. Validation Strategy")
    lines.append("")
    lines.append(f"- Enabled splits: {', '.join(validation.get('enabled', []))}")
    lines.append(f"- Rationale: {validation.get('rationale', '')}")
    lines.append("")

    lines.append("## 6. Experiments Performed")
    lines.append("")
    for exp in report.get("experiments", []):
        lines.append(
            f"### {exp.get('representation')} + {exp.get('model')} [{exp.get('split')}] "
            f"({exp.get('result', '')})"
        )
        metrics = exp.get("metrics", {})
        if metrics:
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in metrics.items():
                lines.append(f"| {key} | {value} |")
        lines.append("")

    rows = report.get("performance_comparison", [])
    if rows:
        lines.append("## 7. Performance Comparison (best observed under budget)")
        lines.append("")
        for row in rows:
            lines.append(f"- **{row['experiment']}**: {row.get('metrics', {})} (runtime {row.get('runtime_seconds', 0):.1f}s)")
        lines.append("")

    best = report.get("best_experiment")
    if best:
        lines.append("## 8. Final Model (budget-observed best)")
        lines.append("")
        lines.append(f"**{best.get('representation')} + {best.get('model')} [{best.get('split')}]**")
        lines.append("")
        lines.append(report.get("final_model_selection_rationale", ""))
        lines.append("")

    diagnoses = report.get("failure_diagnoses", [])
    if diagnoses:
        lines.append("## 9. Failure Diagnoses")
        lines.append("")
        for d in diagnoses[:5]:
            lines.append(f"- **{d.get('failure_type', 'unknown')}** (conf {d.get('confidence', 0):.2f}): {d.get('hypothesis', '')}")
        lines.append("")

    cost = report.get("computational_cost", {})
    lines.append("## 10. Stopping & Compute")
    lines.append("")
    lines.append(f"- Experiments performed: {cost.get('experiments_done', 0)}")
    lines.append(f"- Stop reasons: {'; '.join(cost.get('stop_reasons', []))}")
    lines.append("")

    limitations = report.get("limitations", [])
    if limitations:
        lines.append("## 11. Limitations")
        lines.append("")
        for limitation in limitations:
            lines.append(f"- {limitation}")
        lines.append("")

    recs = report.get("recommended_next_experiments", [])
    if recs:
        lines.append("## 12. Recommended Next Experiments")
        lines.append("")
        for rec in recs:
            lines.append(f"- {rec}")
        lines.append("")
    return "\n".join(lines)