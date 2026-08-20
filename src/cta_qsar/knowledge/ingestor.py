"""Ingest run evidence (benchmark results.csv, experiments.jsonl, report.json)
into the EvidenceStore so planners can condition on accumulated experience.

Evidence discipline:
  - append-only merges keyed by (triple, run_id) -> idempotent re-ingestion
  - rolling window keeps the most recent WINDOW_SIZE runs per triple
  - per-aspect path cached via .cache destination (default: benchmark dir)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cta_qsar.knowledge.facts import MIN_EVIDENCE, EvidenceStore, dataset_class


def ingest_results_file(store: EvidenceStore, path: str | Path, *, task_type: str = "", rows: int = 0) -> int:
    """Ingest one benchmark results.csv. Returns number of triples updated."""
    path = Path(path)
    if not path.exists():
        return 0
    n_updated = 0
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dataset = row.get("dataset", "")
            scenario = row.get("scenario", "")
            primary_value = row.get("primary_value", "")
            primary = row.get("primary", "")
            _type = row.get("task_type", "") or task_type
            _rows = int(row.get("rows", rows) or rows or 0)
            seed = row.get("seed", "")
            run_id = row.get("run_id", "") or f"{dataset}|{seed}"
            best_model = row.get("best_model", "") or "unknown"
            if not dataset or not scenario or not primary_value:
                continue
            try:
                value = float(primary_value)
            except ValueError:
                continue
            cls = dataset_class(_type, _rows)
            sign = 1.0 if _higher_is_better(primary) else -1.0
            # fine: class|scenario|best_model
            store.add_value(cls, scenario, best_model, value, run_id=run_id, level=4, source=f"benchmark:{path.stem}", sign=sign)
            # coarse: class|scenario|any (+ exact-object aggregates)
            store.add_value(cls, scenario, "*", value, run_id=run_id, level=2, source=f"benchmark:{path.stem}", sign=sign)
            # coarse: class|any|any
            store.add_value(cls, "*", "*", value, run_id=run_id, level=1, source=f"benchmark:{path.stem}", sign=sign)
            n_updated += 3
    return n_updated


def ingest_jsonl(store: EvidenceStore, path: str | Path, *, task_type: str = "", rows: int = 0, primary_key: str = "primary_value") -> int:
    """Ingest run records (experiments.jsonl) with per-record primary values."""
    path = Path(path)
    if not path.exists():
        return 0
    n_updated = 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("result") != "completed":
                continue
            dataset = rec.get("dataset", "")
            scenario = "internal-run"
            value = rec.get(primary_key)
            primary = rec.get("primary", "")
            seed = rec.get("seed", "")
            run_id = str(rec.get("run_id", "")) or f"{dataset}|{seed}"
            if scenario and value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                cls = dataset_class(task_type or "regression", rows)
                sign = 1.0 if _higher_is_better(primary) else -1.0
                store.add_value(cls, scenario, rec.get("model", "unknown"), value, run_id=run_id, level=4, source=f"run:{path.stem}", sign=sign)
                n_updated += 1
    return n_updated


def ingest_results_glob(store: EvidenceStore, root: str | Path, *, min_n: int = MIN_EVIDENCE) -> int:
    """Ingest every results.csv and experiments.jsonl under ``root``."""
    root = Path(root)
    updated = 0
    for results_file in sorted(root.glob("**/results.csv")):
        updated += ingest_results_file(store, results_file)
    for jsonl_file in sorted(root.glob("**/experiments.jsonl")):
        updated += ingest_jsonl(store, jsonl_file)
    return updated


def _higher_is_better(primary: str) -> bool:
    primary = (primary or "").lower()
    return any(
        token in primary
        for token in ("auc", "acc", "f1", "recall", "kappa", "true", "precision", "balanced")
    ) and "rmse" not in primary and "mae" not in primary and "mse" not in primary