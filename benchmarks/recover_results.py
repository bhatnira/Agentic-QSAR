"""Recover a campaign's results.csv from persisted run artifacts.

Recovery path used after a campaign crash at the final DataFrame write (the
stamp directory had been removed by an earlier cleanup). Every completed run
mints a run dir under benchmarks/runs/ containing provenance.json (config +
dataset path) and report.json (full-precision per-experiment metrics), which
is enough to reconstruct the exact rows run_benchmark.run_agent would have
written:

    dataset, seed, scenario, primary, primary_value, best_model,
    best_hyperparams, n_experiments, runtime_seconds, run_id

Scenario labels are re-derived from the stored config (llm provider, search
flag, trust gate, adaptive policy). Per (dataset, seed, scenario) the run
with the lexicographically newest run id wins (campaign runs came after all
earlier smoke runs). runtime_seconds is not recoverable from artifacts and is
written as 0.0; run_benchmark is not modified.

    python3 benchmarks/recover_results.py --out benchmarks/results/<stamp>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import run_benchmark as rb

RUNS_ROOT = rb.RUNS_ROOT
DATASET_NAMES = sorted(rb.DATASETS)
CSV_PATTERN = re.compile(rf"^(?P<ds>{'|'.join(DATASET_NAMES)})_seed(?P<seed>\d+)(?:_(?P<card>.*))?\.csv$")


def infer_scenario(config: dict) -> str:
    llm = (config.get("llm") or {}).get("provider", "")
    if llm == "nvidia":
        return "agent-nvidia"
    policy = (config.get("policy") or {})
    if policy.get("adaptive"):
        return "agent-evolve"
    if not (config.get("experiment") or {}).get("hyperparameter_search", True):
        return "agent-nosearch"
    trust = (config.get("trust") or {}).get("required", ["predictive"])
    if not trust:
        return "agent-nogate"
    return "agent-mock"


def best_experiment(report: dict, task_type: str) -> tuple[dict, dict, list[dict]] | None:
    """Mirror run_agent's selection: best completed experiment on the primary."""
    primary = rb._primary(task_type)  # noqa: SLF001
    best = None
    completed = [e for e in report.get("experiments", []) if e.get("result") == "completed"]
    for exp in completed:
        metrics = exp.get("metrics", {})
        if primary not in metrics:
            continue
        value = float(metrics[primary])
        if best is None or rb._better(value, best["metrics"][primary], task_type):  # noqa: SLF001
            best = exp
    if best is None:
        return None
    return best, best["metrics"], completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="results stamp dir to write results.csv into")
    args = parser.parse_args(argv)

    rows: dict[tuple, dict] = {}
    for run_dir in sorted(RUNS_ROOT.glob("*")):
        prov_path = run_dir / "provenance.json"
        report_path = run_dir / "report.json"
        if not prov_path.exists() or not report_path.exists():
            continue
        provenance = json.loads(prov_path.read_text())
        config = (provenance.get("config") or {})
        dataset_path = str(provenance.get("dataset", "") or "")
        match = CSV_PATTERN.match(Path(dataset_path).name)
        if not match or match.group("card"):
            continue  # challenge runs have card-suffixed csvs; excluded by design
        dataset_name = match.group("ds")
        seed = int(match.group("seed"))
        table = rb.DATASETS[dataset_name]
        task_type = table.task_type
        if (config.get("knowledge") or {}).get("evidence_path", "") != str(rb.KNOWLEDGE_EVIDENCE):
            continue  # only include runs of the standard campaign pipeline
        report = json.loads(report_path.read_text())
        chosen = best_experiment(report, task_type)
        if chosen is None:
            print(f"skip {run_dir.name}: no completed experiments", file=sys.stderr)
            continue
        best, metrics, completed = chosen
        scenario = infer_scenario(config)
        key = (dataset_name, seed, scenario)
        prior = rows.get(key)
        if prior is not None and prior["run_id"] >= run_dir.name:
            continue
        rows[key] = {
            "dataset": dataset_name,
            "task_type": task_type,
            "rows": len(rb.load_dataset(table)),
            "seed": seed,
            "scenario": scenario,
            "primary": rb._primary(task_type),  # noqa: SLF001
            "primary_value": round(float(metrics[rb._primary(task_type)]), 4),  # noqa: SLF001
            "best_model": f"{best['representation']}+{best['model']}[{best['split']}]",
            "best_hyperparams": json.dumps(best.get("hyperparameters", {})),
            "n_experiments": len(completed),
            "runtime_seconds": round(sum(float(e.get("runtime_seconds", 0.0) or 0.0) for e in completed), 1),
            "run_id": run_dir.name,
        }

    if not rows:
        print("no recoverable runs found (expected 120 core + 30 nvidia)", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else RUNS_ROOT.parent / "results" / "recovered"
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv

    columns = ["dataset", "task_type", "rows", "seed", "scenario", "primary",
               "primary_value", "best_model", "best_hyperparams",
               "n_experiments", "runtime_seconds", "run_id"]
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    (out_dir / "meta.json").write_text(
        json.dumps({
            "note": "recovered from run artifacts after results.csv write crash",
            "n_rows": len(rows),
            "scenarios": sorted({v["scenario"] for v in rows.values()}),
            "runtime_seconds_unavailable": True,
        }, indent=2) + "\n"
    )
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row["scenario"]] = counts.get(row["scenario"], 0) + 1
    print(f"recovered {len(rows)} rows -> {out_dir/'results.csv'}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())