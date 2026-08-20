"""Multi-dataset benchmark harness for CTA-QSAR.

Runs the autonomous pipeline (and static grid baselines) across MoleculeNet
datasets and random seeds, then aggregates results into CSV/JSON.

Typical usage (from the repo root):

    python3 benchmarks/run_benchmark.py \\
        --datasets esol,freesolv --seeds 0,1,2 --scenarios grid,agent-mock

Metrics: predictive CV performance using the same fold generation, estimator
wrapping, and metric functions as the production pipeline (primary metric:
RMSE for regression, ROC-AUC for binary classification).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from cta_qsar.core.config import build_config
from cta_qsar.core.registry import get_registry

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "data"
RUNS_ROOT = ROOT / "benchmarks" / "runs"
RESULTS_ROOT = ROOT / "benchmarks" / "results"
DATASET_DIR_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    file: str
    smiles_column: str
    target_column: str
    task_type: str
    note: str


DATASETS: dict[str, BenchmarkDataset] = {
    "esol": BenchmarkDataset(
        "esol", "delaney-processed.csv", "smiles",
        "measured log solubility in mols per litre", "regression",
        "Delaney ESOL aqueous solubility (1128 compounds)",
    ),
    "freesolv": BenchmarkDataset(
        "freesolv", "freesolv.csv.gz", "smiles", "y", "regression",
        "FreeSolv hydration free energies (642 compounds)",
    ),
    "lipophilicity": BenchmarkDataset(
        "lipophilicity", "Lipophilicity.csv", "smiles", "exp", "regression",
        "MoleculeNet Lipophilicity (4200 compounds)",
    ),
    "bace": BenchmarkDataset(
        "bace", "bace.csv", "mol", "Class", "binary",
        "BACE-1 binding classification (1513 compounds)",
    ),
    "bbbp": BenchmarkDataset(
        "bbbp", "BBBP.csv", "smiles", "p_np", "binary",
        "Blood-brain barrier penetration (2050 compounds)",
    ),
    "clintox": BenchmarkDataset(
        "clintox", "clintox.csv.gz", "smiles", "CT_TOX", "binary",
        "Clinical trial toxicity (1478 compounds)",
    ),
}

# Static "standard practice" baseline grid: (representation, model, hyperparams).
# Mirrors the plugin hyperparameter spaces used by the autonomous agent.
BASELINE_GRID: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
    "regression": [
        ("morgan", "ridge", {"alpha": 0.1}),
        ("morgan", "ridge", {"alpha": 1.0}),
        ("morgan", "ridge", {"alpha": 10.0}),
        ("morgan", "elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}),
        ("morgan", "elastic_net", {"alpha": 0.1, "l1_ratio": 0.5}),
        ("morgan", "random_forest", {"n_estimators": 100, "max_depth": None}),
        ("morgan", "random_forest", {"n_estimators": 300, "max_depth": 20}),
        ("morgan", "extra_trees", {"n_estimators": 300, "max_depth": None}),
    ],
    "binary": [
        ("morgan", "random_forest", {"n_estimators": 100, "max_depth": None}),
        ("morgan", "random_forest", {"n_estimators": 300, "max_depth": 20}),
        ("morgan", "extra_trees", {"n_estimators": 300, "max_depth": None}),
        ("morgan", "mlp", {"hidden_layer_sizes": (128, 64), "max_iter": 300, "early_stopping": True}),
    ],
}


def ensure_dataset(dataset: BenchmarkDataset) -> Path:
    """Download (once) and return the cached local path for a dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / (dataset.file[:-3] if dataset.file.endswith(".gz") else dataset.file)
    if not dest.exists():
        url = f"{DATASET_DIR_URL}/{dataset.file}"
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        if dataset.file.endswith(".gz"):
            dest.write_bytes(gzip.decompress(response.content))
        else:
            dest.write_bytes(response.content)
    return dest


def load_dataset(dataset: BenchmarkDataset) -> pd.DataFrame:
    path = ensure_dataset(dataset)
    df = pd.read_csv(path, low_memory=False)
    if dataset.smiles_column not in df.columns or dataset.target_column not in df.columns:
        raise ValueError(f"columns {dataset.smiles_column}/{dataset.target_column} missing from {path}")
    df = df[[dataset.smiles_column, dataset.target_column]].dropna()
    df.columns = ["smiles", "target"]
    return df.reset_index(drop=True)


def _primary(task_type: str) -> str:
    return "rmse" if task_type == "regression" else "roc_auc"


def _better(a: float, b: float, task_type: str) -> bool:
    return (a < b) if task_type == "regression" else (a > b)


def run_baseline(df: pd.DataFrame, dataset: BenchmarkDataset, seed: int) -> dict[str, Any]:
    """Best static-grid combination evaluated with identical CV folds."""
    from rdkit import Chem

    from cta_qsar.experiments.runner import _fold_primary_score
    from cta_qsar.models.registry import wrapped_estimator
    from cta_qsar.representations.registry import representation_matrix
    from cta_qsar.validation.base import make_cv_folds

    registry = get_registry()
    smiles = df["smiles"].astype(str).tolist()
    valid = np.asarray([Chem.MolFromSmiles(s) is not None for s in smiles])
    n_dropped = int((~valid).sum())
    if n_dropped:
        smiles = [s for s, ok in zip(smiles, valid, strict=False) if ok]
        df = df[valid].reset_index(drop=True)
    y = pd.to_numeric(df["target"], errors="coerce").to_numpy()
    if dataset.task_type != "regression":
        classes = pd.unique(df["target"])
        class_map = {c: i for i, c in enumerate(sorted(classes))}
        y = np.asarray([class_map[v] for v in df["target"]])
    folds = make_cv_folds(
        len(df), y, strategy="random", n_splits=5, n_repeats=1,
        random_seed=seed, test_fraction=0.2,
    )
    rep_plugin = registry.get("representation", "morgan")
    X = representation_matrix(rep_plugin, smiles, fit=True)
    started = time.time()
    best: dict[str, Any] | None = None
    for _rep, model, hp in BASELINE_GRID[dataset.task_type]:
        try:
            estimator = wrapped_estimator(registry, model, dataset.task_type, n_classes=None, hyperparams=hp)
            score = _fold_primary_score(estimator, X, y, folds, dataset.task_type)
        except Exception as exc:  # noqa: BLE001
            print(f"    baseline {model} failed: {exc}")
            continue
        row = {"model": model, "hyperparams": hp, "score": score}
        if best is None or _better(score, best["score"], dataset.task_type):
            best = row
    if best is None:
        raise RuntimeError(f"no baseline combo succeeded for {dataset.name}")
    primary = _primary(dataset.task_type)
    value = float(best["score"])
    if primary == "rmse":
        value = -value
    return {
        "scenario": "grid",
        "best_model": best["model"],
        "best_hyperparams": json.dumps(best["hyperparams"]),
        "primary": primary,
        "primary_value": round(value, 4),
        "n_experiments": len(BASELINE_GRID[dataset.task_type]),
        "runtime_seconds": round(time.time() - started, 1),
        "seed": seed,
    }


def run_agent(df: pd.DataFrame, dataset: BenchmarkDataset, seed: int, *, search: bool, llm_provider: str = "mock") -> dict[str, Any]:
    """One full autonomous run at the given seed (provider mock|nvidia)."""
    from cta_qsar.agents.scientist import QSARScientist

    config = build_config(
        seed=seed,
        hyperparameter_search=True,
        llm_provider=llm_provider,
    )
    config.experiment.hyperparameter_search = search
    config.experiment.n_repeats = 1
    config.compute.max_experiments = 6
    config.compute.max_minutes = 30.0
    config.dataset.smiles_column = dataset.smiles_column
    config.dataset.target_column = dataset.target_column
    config.reporting["output_dir"] = str(RUNS_ROOT)
    data_csv = DATA_DIR / f"{dataset.name}_seed{seed}.csv"
    df.to_csv(data_csv, index=False)
    scientist = QSARScientist(config)
    started = time.time()
    final = scientist.run(data_csv)
    runtime = time.time() - started
    experiments = final.get("experiments", []) if isinstance(final, dict) else []
    completed = [e for e in experiments if e.get("result") == "completed"]
    primary = _primary(dataset.task_type)
    best_exp = None
    for exp in completed:
        metrics = exp.get("metrics", {})
        if primary not in metrics:
            continue
        value = float(metrics[primary])
        if best_exp is None or _better(value, best_exp["metrics"][primary], dataset.task_type):
            best_exp = exp
    if best_exp is None:
        raise RuntimeError(f"no completed experiments for {dataset.name} seed {seed}")
    return {
        "scenario": "agent-nvidia" if llm_provider == "nvidia" else ("agent-mock" if search else "agent-nosearch"),
        "best_model": f"{best_exp['representation']}+{best_exp['model']}[{best_exp['split']}]",
        "best_hyperparams": json.dumps(best_exp.get("hyperparameters", {})),
        "primary": primary,
        "primary_value": float(best_exp["metrics"][primary]),
        "n_experiments": len(completed),
        "runtime_seconds": round(runtime, 1),
        "run_id": final.get("run_id", ""),
        "seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="esol", help="comma-separated subset of esol,freesolv,lipophilicity,bace,bbbp,clintox")
    parser.add_argument("--seeds", default="0,1", help="comma-separated random seeds")
    parser.add_argument("--scenarios", default="grid,agent-mock", help="comma-separated: grid,agent-mock,agent-nosearch,agent-nvidia")
    args = parser.parse_args(argv)

    datasets = [DATASETS[name] for name in args.datasets.split(",") if name in DATASETS]
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = [s for s in args.scenarios.split(",")
                 if s in ("grid", "agent-mock", "agent-nosearch", "agent-nvidia")]
    if not datasets or not seeds or not scenarios:
        parser.error("need valid --datasets, --seeds, --scenarios")
    if "agent-nvidia" in scenarios and not any(
        key in ("NVIDIA_API_KEY", "OPENAI_API_KEY") for key in dict(__import__("os").environ)
    ):
        parser.error("agent-nvidia requires NVIDIA_API_KEY/OPENAI_API_KEY in the environment")
    get_registry().auto_discover()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": stamp,
        "datasets": [d.name for d in datasets],
        "seeds": seeds,
        "scenarios": scenarios,
        "cv": "random 5-fold x 1 repeat, 20% test fraction",
        "primary_metric": "rmse (regression) / roc_auc (binary)",
        "notes": ["agent scenarios: heuristic planner (mock LLM) or real NVIDIA LLM, hyperparameter search on",
                  "agent-nosearch: agent with hyperparameter search disabled (ablation)",
                  "grid scenario: static hyperparameter grid, identical folds/metrics"],
    }
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        print(f"== {dataset.name}: {dataset.note}")
        df = load_dataset(dataset)
        print(f"   rows={len(df)} task={dataset.task_type}")
        for seed in seeds:
            for scenario in scenarios:
                started = time.time()
                print(f"   seed={seed} scenario={scenario} ...", flush=True)
                try:
                    if scenario == "grid":
                        result = run_baseline(df, dataset, seed)
                    else:
                        result = run_agent(
                            df, dataset, seed,
                            search=(scenario == "agent-mock"),
                            llm_provider=("nvidia" if scenario == "agent-nvidia" else "mock"),
                        )
                    result.update({"dataset": dataset.name, "task_type": dataset.task_type, "rows": len(df)})
                    rows.append(result)
                    print(f"     -> primary={result['primary_value']} "
                          f"model={result.get('best_model')} in {result['runtime_seconds']}s")
                except Exception as exc:  # noqa: BLE001
                    print(f"     FAILED: {exc}", file=sys.stderr)
                    rows.append({
                        "dataset": dataset.name, "task_type": dataset.task_type, "rows": len(df),
                        "seed": seed, "scenario": scenario, "failed": str(exc),
                        "runtime_seconds": round(time.time() - started, 1),
                    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "results.csv", index=False)
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    summary: dict[str, Any] = {}
    for (dataset, scenario), group in results_df.groupby(["dataset", "scenario"]):
        if "primary_value" not in group or group["primary_value"].isna().all():
            summary[f"{dataset}/{scenario}"] = {"error": True}
            continue
        values = group["primary_value"].astype(float)
        task = DATASETS[dataset].task_type
        summary[f"{dataset}/{scenario}"] = {
            "primary": _primary(task),
            "task_type": task,
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=0)), 4),
            "n_seeds": len(values),
        }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nresults written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())