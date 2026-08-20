"""Aggregate benchmark results into paper-ready tables + statistics.

Usage (from repo root):

    python3 benchmarks/analyze_results.py [--stamp 20260820-...] [--seeds 0,1,2,3,4]

Loads ONE results stamp (default: the newest stamp whose results.csv covers
at least 90% of the expected dataset x scenario x seed grid), so series
with different configurations (e.g. trust gate on/off) are never mixed.
Outputs to benchmarks/analysis/<stamp>/:
    - table_primary.csv    mean +/- std per (dataset, scenario)
    - table_efficiency.csv runtime / experiments / speedup vs grid
    - table_ablation.csv   paired deltas per listed ablation
    - stats_paired.txt     Wilcoxon signed-rank summaries
    - summary.json         machine-readable aggregates (for the paper build)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmarks" / "results"
ANALYSIS_ROOT = ROOT / "benchmarks" / "analysis"

SCENARIOS = ["grid", "agent-mock", "agent-nosearch", "agent-nogate", "agent-nvidia", "agent-evolve"]
TASKS = {"esol": "regression", "freesolv": "regression", "lipophilicity": "regression",
         "bace": "binary", "bbbp": "binary", "clintox": "binary"}


def _load_stamp(stamp: Path) -> pd.DataFrame:
    path = stamp / "results.csv"
    if not path.exists():
        raise SystemExit(f"missing results.csv in {stamp}")
    return pd.read_csv(path)


def _pick_stamp(stamps: list[Path], expected_grid: int, args_seeds: list[int]) -> tuple[Path, pd.DataFrame, list[int]]:
    for stamp in reversed(sorted(stamps)):
        df = _load_stamp(stamp)
        df = df[df["primary_value"].notna()]
        if df.empty:
            continue
        seeds = args_seeds or sorted({int(s) for s in df["seed"].dropna().astype(int).unique()})
        df = df[df["seed"].astype(int).isin(seeds)]
        present = len(df)
        if present >= expected_grid:
            return stamp, df, seeds
    raise SystemExit(
        f"no results stamp covers the full grid (need >= {expected_grid} non-null rows); "
        "use --stamp or --seeds to narrow"
    )


def _signed(series: pd.Series, dataset: str) -> pd.Series:
    """Reflect onto a higher-is-better axis for deltas."""
    return series if TASKS.get(dataset) == "binary" else -series


def _paired(selected: pd.DataFrame, seeds: list[int], dataset: str, a: str, b: str) -> pd.Series:
    a_map = {int(r.seed): r.primary_value for r in selected[(selected["scenario"] == a) & (selected["dataset"] == dataset)].itertuples()}
    b_map = {int(r.seed): r.primary_value for r in selected[(selected["scenario"] == b) & (selected["dataset"] == dataset)].itertuples()}
    common = [s for s in seeds if s in a_map and s in b_map]
    if not common:
        return pd.Series(dtype=float)
    a_vals = pd.Series([a_map[s] for s in common], dtype=float)
    b_vals = pd.Series([b_map[s] for s in common], dtype=float)
    return _signed(a_vals, dataset) - _signed(b_vals, dataset)


def _wilcoxon(deltas: pd.Series) -> dict[str, Any]:
    deltas = deltas.dropna().astype(float)
    if len(deltas) < 2:
        return {"n": len(deltas), "p": None, "w": None}
    if (deltas == 0).all():
        return {"n": len(deltas), "p": 1.0, "w": 0.0}
    result = stats.wilcoxon(deltas, zero_method="wilcox")
    return {"n": len(deltas), "p": float(result.pvalue), "w": float(result.statistic)}


def build(args: argparse.Namespace) -> int:
    stamps = sorted(RESULTS_ROOT.glob("2026*"))
    if args.stamp:
        stamps = [s for s in stamps if s.name == args.stamp]
    if not stamps:
        print("no results stamps under", RESULTS_ROOT, file=sys.stderr)
        return 1

    # coverage probe with the first stamp's seed set
    probe = _load_stamp(stamps[0])
    probe_seeds = args.seeds or sorted({int(s) for s in probe["seed"].dropna().astype(int).unique()})
    expected = len(probe_seeds) * len(SCENARIOS) * 6
    stamp, selected, seeds = _pick_stamp(stamps, int(expected * 0.9), args.seeds)

    out_dir = ANALYSIS_ROOT / stamp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"stamp": stamp.name, "seeds": seeds}

    datasets = sorted(d for d in selected["dataset"].unique() if d in TASKS)

    # ---- Table 1: primary metric mean +/- std per (dataset, scenario) ----
    lines = ["dataset,scenario,task,primary,mean,std,n_seeds"]
    for dataset in datasets:
        for scenario in SCENARIOS:
            rows = selected[(selected["scenario"] == scenario) & (selected["dataset"] == dataset)]
            if rows.empty:
                continue
            values = rows["primary_value"].astype(float)
            lines.append(
                f"{dataset},{scenario},{TASKS[dataset]},{rows['primary'].iloc[0]},"
                f"{values.mean():.4f},{values.std(ddof=0):.4f},{len(values)}"
            )
    (out_dir / "table_primary.csv").write_text("\n".join(lines) + "\n")
    out["table_primary"] = lines

    # ---- Table 2: efficiency ----
    lines = ["dataset,scenario,mean_runtime_s,median_runtime_s,mean_experiments,speedup_vs_grid"]
    grid_rt = {
        dataset: selected[(selected["scenario"] == "grid") & (selected["dataset"] == dataset)]["runtime_seconds"].astype(float).mean()
        for dataset in datasets
    }
    for dataset in datasets:
        for scenario in SCENARIOS:
            rows = selected[(selected["scenario"] == scenario) & (selected["dataset"] == dataset)]
            if rows.empty or "runtime_seconds" not in rows.columns:
                continue
            rt = rows["runtime_seconds"].astype(float)
            exp_col = rows["n_experiments"].astype(float) if "n_experiments" in rows.columns else pd.Series([float("nan")] * len(rows))
            speed = f"{grid_rt[dataset] / rt.mean():.2f}x" if grid_rt[dataset] and grid_rt[dataset] > 0 else "n/a"
            lines.append(
                f"{dataset},{scenario},{rt.mean():.1f},{rt.median():.1f},{exp_col.mean():.1f},{speed}"
            )
    (out_dir / "table_efficiency.csv").write_text("\n".join(lines) + "\n")
    out["table_efficiency"] = lines

    # ---- Table 3: ablations (paired deltas, higher-is-better axis) ----
    ablations = [
        ("agent-vs-grid", "agent-mock", "grid"),
        ("nosearch-vs-grid", "agent-nosearch", "grid"),
        ("nogate-vs-grid", "agent-nogate", "grid"),
        ("search-on-vs-off", "agent-mock", "agent-nosearch"),
        ("gate-on-vs-off", "agent-mock", "agent-nogate"),
        ("evolve-vs-frozen", "agent-evolve", "agent-mock"),
        ("nvidia-vs-mock", "agent-nvidia", "agent-mock"),
    ]
    lines = ["ablation,dataset,delta_mean,delta_median,delta_std,n_pairs"]
    for label, a, b in ablations:
        for dataset in datasets:
            deltas = _paired(selected, seeds, dataset, a, b)
            if deltas.empty:
                continue
            lines.append(
                f"{label},{dataset},{deltas.mean():+.4f},{deltas.median():+.4f},{deltas.std(ddof=0):.4f},{len(deltas)}"
            )
    (out_dir / "table_ablation.csv").write_text("\n".join(lines) + "\n")
    out["table_ablation"] = lines

    # ---- Paired significance summary per ablation (dataset-level deltas) ----
    lines = ["ablation,n_datasets,mean_delta,median_delta,p_wilcoxon"]
    for label, a, b in ablations:
        deltas = pd.concat([_paired(selected, seeds, d, a, b) for d in datasets if not _paired(selected, seeds, d, a, b).empty])
        if deltas.empty:
            continue
        stat = _wilcoxon(deltas)
        lines.append(
            f"{label},{deltas.notna().sum()},{deltas.mean():+.4f},{deltas.median():+.4f},"
            f"{stat['p'] if stat['p'] is not None else 'n/a'}"
        )
    (out_dir / "stats_paired.txt").write_text("\n".join(lines) + "\n")
    out["stats_paired"] = lines

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"analysis written to {out_dir}/ using stamp {stamp.name}")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="", help="specific results stamp dir name")
    parser.add_argument("--seeds", default="", help="comma-separated seed list override")
    args = parser.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else []
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())