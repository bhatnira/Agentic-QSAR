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
KNOWLEDGE_EVIDENCE = ROOT / "benchmarks" / "knowledge" / "evidence.jsonl"
DATASET_DIR_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    file: str
    smiles_column: str
    target_column: str
    task_type: str
    note: str
    target_columns: list[str] | None = None
    url: str | None = None



# -- extension datasets (multiclass / multitask / OpenADMET) ----------------

TOX21_TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _tox21_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        "tox21",
        "tox21.csv.gz",
        "smiles",
        "NR-AR",
        "multitask_binary",
        "Tox21 12-target nuclear receptor / stress response assays (8014 compounds)",
        target_columns=TOX21_TARGETS,
    )


def _pampa3_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        "pampa3",
        "permeability_NCATS-PAMPA-pH7.4.csv",
        "canonical_smiles",
        "Phenotype",
        "multiclass",
        "NCATS PAMPA permeability at pH 7.4: 3 classes (Low/Moderate/High, 2033 compounds)",
        url="https://netknowledge.github.io/ADMET/datasets/permeability_NCATS-PAMPA-pH7.4.csv",
    )


def _pxr_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        "pxr",
        "pxr-challenge_TRAIN.csv",
        "SMILES",
        "pEC50",
        "regression",
        "OpenADMET PXR induction pEC50, training split (4139 compounds)",
        url="https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main/pxr-challenge_TRAIN.csv",
    )

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
    "tox21": _tox21_dataset(),
    "pampa3": _pampa3_dataset(),
    "pxr": _pxr_dataset(),
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
    "multiclass": [
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
    if dest.exists():
        return dest
    if dataset.url:
        response = requests.get(dataset.url, timeout=600)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest
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
    if dataset.name == "tox21":
        df = pd.read_csv(path, low_memory=False)
        keep = [dataset.smiles_column] + [c for c in dataset.target_columns or [] if c in df.columns]
        df = df[keep].dropna().reset_index(drop=True)
        df.columns = ["smiles"] + [c for c in dataset.target_columns or [] if len(keep) > 1]
        return df
    if dataset.name == "pampa3":
        df = pd.read_csv(path, low_memory=False)
        df = df[[dataset.smiles_column, dataset.target_column]].dropna()
        df.columns = ["smiles", dataset.target_column]
        return df.reset_index(drop=True)
    df = pd.read_csv(path, low_memory=False)
    if dataset.smiles_column not in df.columns or dataset.target_column not in df.columns:
        raise ValueError(f"columns {dataset.smiles_column}/{dataset.target_column} missing from {path}")
    df = df[[dataset.smiles_column, dataset.target_column]].dropna()
    df.columns = ["smiles", dataset.target_column]
    return df.reset_index(drop=True)


def _base_task(task_type: str) -> str:
    return task_type.removeprefix("multitask_")


def _primary(task_type: str) -> str:
    if task_type == "regression":
        return "rmse"
    if task_type in ("binary", "multitask_binary"):
        return "roc_auc"
    return "mcc"


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
    base = _base_task(dataset.task_type)
    grid = BASELINE_GRID[base]
    target_cols = dataset.target_columns or [dataset.target_column]
    needs_encode = base in ("binary", "multiclass")

    def _encode(values: Any) -> np.ndarray:
        if not needs_encode:
            return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        classes = pd.unique(values)
        class_map = {c: i for i, c in enumerate(sorted(classes))}
        return np.asarray([class_map[v] for v in values])

    smiles = df["smiles"].astype(str).tolist()
    valid = np.asarray([Chem.MolFromSmiles(s) is not None for s in smiles])
    n_dropped = int((~valid).sum())
    if n_dropped:
        smiles = [s for s, ok in zip(smiles, valid, strict=False) if ok]
        df = df[valid].reset_index(drop=True)
    ys = [_encode(df[col]) for col in target_cols]
    folds = make_cv_folds(
        len(df), ys[0], strategy="random", n_splits=5, n_repeats=1,
        random_seed=seed, test_fraction=0.2,
    )
    rep_plugin = registry.get("representation", "morgan")
    X = representation_matrix(rep_plugin, smiles, fit=True)
    started = time.time()
    best: dict[str, Any] | None = None
    for _rep, model, hp in grid:
        try:
            estimator = wrapped_estimator(registry, model, base, n_classes=None, hyperparams=hp)
            if dataset.task_type.startswith("multitask_"):
                run_scores = [
                    _fold_primary_score(estimator, X, y, folds, base) for y in ys
                ]
                score = float(np.mean(run_scores))
            else:
                score = _fold_primary_score(estimator, X, ys[0], folds, base)
        except Exception as exc:  # noqa: BLE001
            print(f"    baseline {model} failed: {exc}")
            continue
        row = {"model": model, "hyperparams": hp, "score": score}
        if best is None or _better(score, best["score"], base):
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
        "n_experiments": len(grid),
        "runtime_seconds": round(time.time() - started, 1),
        "seed": seed,
    }


def run_agent(
    df: pd.DataFrame,
    dataset: BenchmarkDataset,
    seed: int,
    *,
    search: bool,
    llm_provider: str = "mock",
    trust_gate: bool = True,
) -> dict[str, Any]:
    """One full autonomous run at the given seed (provider mock|nvidia).

    ``trust_gate=False`` disables the trust gate (ablation): the agent may
    finalize as soon as the stopping policy triggers, regardless of whether
    the latest validated strategy is trustworthy.
    """
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
    if dataset.target_columns:
        config.dataset.target_columns = list(dataset.target_columns)
    config.reporting["output_dir"] = str(RUNS_ROOT)
    config.knowledge.evidence_path = str(KNOWLEDGE_EVIDENCE)
    if not trust_gate:
        config.trust.required = []
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
    if llm_provider == "nvidia":
        scenario = "agent-nvidia"
    elif not search:
        scenario = "agent-nosearch"
    elif not trust_gate:
        scenario = "agent-nogate"
    else:
        scenario = "agent-mock"
    return {
        "scenario": scenario,
        "best_model": f"{best_exp['representation']}+{best_exp['model']}[{best_exp['split']}]",
        "best_hyperparams": json.dumps(best_exp.get("hyperparameters", {})),
        "primary": primary,
        "primary_value": float(best_exp["metrics"][primary]),
        "n_experiments": len(completed),
        "runtime_seconds": round(runtime, 1),
        "run_id": final.get("run_id", ""),
        "seed": seed,
    }


def refresh_evidence(new_results: list[Path] | None = None, *, roots: list[Path] | None = None) -> Any:
    """Reload + re-ingest results into the knowledge evidence store.

    Idempotent: merges are keyed by (triple, run_id) with a rolling window, so
    re-running a benchmark only appends genuinely new runs. ``roots`` limits
    ingestion to specific results dirs (used with --fresh-evidence).
    """
    from cta_qsar.knowledge.facts import EvidenceStore
    from cta_qsar.knowledge.ingestor import ingest_results_file, ingest_results_glob

    store = EvidenceStore.load(KNOWLEDGE_EVIDENCE)
    for root in roots or [RESULTS_ROOT]:
        ingest_results_glob(store, root)
    for results_file in new_results or []:
        ingest_results_file(store, results_file)
    store.save(KNOWLEDGE_EVIDENCE)
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="esol", help="comma-separated subset of esol,freesolv,lipophilicity,bace,bbbp,clintox")
    parser.add_argument("--seeds", default="0,1", help="comma-separated random seeds")
    parser.add_argument("--scenarios", default="grid,agent-mock",
                        help="comma-separated: grid,agent-mock,agent-nosearch,agent-nvidia,agent-nogate")
    parser.add_argument("--fresh-evidence", action="store_true",
                        help="start the knowledge evidence store from zero (ignore prior results dirs)")
    parser.add_argument("--resume", action="store_true",
                        help="skip (dataset, seed, scenario) combos already present in the most recent results.csv")
    args = parser.parse_args(argv)

    datasets = [DATASETS[name] for name in args.datasets.split(",") if name in DATASETS]
    seeds = [int(s) for s in args.seeds.split(",")]
    scenarios = [s for s in args.scenarios.split(",")
                 if s in ("grid", "agent-mock", "agent-nosearch", "agent-nvidia", "agent-nogate")]
    if not datasets or not seeds or not scenarios:
        parser.error("need valid --datasets, --seeds, --scenarios")
    if "agent-nvidia" in scenarios and not any(
        key in ("NVIDIA_API_KEY", "OPENAI_API_KEY") for key in dict(__import__("os").environ)
    ):
        parser.error("agent-nvidia requires NVIDIA_API_KEY/OPENAI_API_KEY in the environment")
    get_registry().auto_discover()
    resume_done: set[tuple[str, int, str]] = set()
    if args.resume:
        stamps = sorted(RESULTS_ROOT.glob("*/results.csv"))
        if not stamps:
            parser.error("--resume requested but no previous results.csv found")
        prev = pd.read_csv(stamps[-1]).dropna(subset=["primary_value"])
        resume_done = {
            (str(row.dataset), int(row.seed), str(row.scenario))
            for row in prev.itertuples(index=False)
        }
        print(f"resume: {len(resume_done)} completed combos will be skipped", flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh_evidence:
        from cta_qsar.knowledge.facts import EvidenceStore

        EvidenceStore().save(KNOWLEDGE_EVIDENCE)
        print("fresh evidence: knowledge store reset to empty", flush=True)
        refresh_evidence(roots=[out_dir])
    else:
        refresh_evidence()

    meta = {
        "timestamp": stamp,
        "datasets": [d.name for d in datasets],
        "seeds": seeds,
        "scenarios": scenarios,
        "cv": "random 5-fold x 1 repeat, 20% test fraction",
        "primary_metric": "rmse (regression) / roc_auc (binary)",
        "notes": ["agent scenarios: heuristic planner (mock LLM) or real NVIDIA LLM, hyperparameter search on",
                  "agent-nosearch: agent with hyperparameter search disabled (ablation)",
                  "agent-nogate: agent with the trust gate disabled (ablation)",
                  "trust gate enforced unless trust.required is empty",
                  "grid scenario: static hyperparameter grid, identical folds/metrics"],
    }
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        print(f"== {dataset.name}: {dataset.note}")
        df = load_dataset(dataset)
        print(f"   rows={len(df)} task={dataset.task_type}")
        for seed in seeds:
            for scenario in scenarios:
                if (dataset.name, seed, scenario) in resume_done:
                    print(f"   seed={seed} scenario={scenario} (resumed: skip)", flush=True)
                    continue
                started = time.time()
                print(f"   seed={seed} scenario={scenario} ...", flush=True)
                try:
                    if scenario == "grid":
                        result = run_baseline(df, dataset, seed)
                    else:
                        result = run_agent(
                            df, dataset, seed,
                            search=(scenario in ("agent-mock", "agent-nogate", "agent-nvidia")),
                            llm_provider=("nvidia" if scenario == "agent-nvidia" else "mock"),
                            trust_gate=(scenario != "agent-nogate"),
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

    refresh_evidence(new_results=[out_dir / "results.csv"])

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