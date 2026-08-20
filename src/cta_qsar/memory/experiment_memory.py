"""Scientific memory: the run store and cross-run experiment memory.

Every experiment is persisted to ``runs/<run_id>/experiments.jsonl`` and
indexed in-process so the planner can avoid repeating identical
configurations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cta_qsar.core.interfaces import ExperimentRecord


@dataclass
class ExperimentMemory:
    """In-process + on-disk experiment memory for one run."""

    run_dir: Path
    records: list[ExperimentRecord] = field(default_factory=list)

    @property
    def jsonl_path(self) -> Path:
        return self.run_dir / "experiments.jsonl"

    def add(self, record: ExperimentRecord) -> None:
        self.records.append(record)
        self._flush(record)

    def _flush(self, record: ExperimentRecord) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(mode="json")
        payload["signature"] = record.signature
        with self.jsonl_path.open("a") as handler:
            handler.write(json.dumps(payload) + "\n")

    def signatures_seen(self) -> set[str]:
        return {record.signature for record in self.records}

    def all_as_dicts(self) -> list[dict[str, Any]]:
        return [record.model_dump(mode="json") for record in self.records]

    def best_experiment(self, metric_priority: tuple[str, ...]) -> ExperimentRecord | None:
        """Best completed experiment by the first metric that all reports share."""
        best: ExperimentRecord | None = None
        best_score: float | None = None
        for record in self.records:
            if record.result != "completed":
                continue
            for metric in metric_priority:
                if metric in record.metrics and record.metrics[metric] is not None:
                    score = float(record.metrics[metric])
                    better = best_score is None or (
                        score < best_score if metric in ("rmse", "mae") else score > best_score
                    )
                    if better:
                        best, best_score = record, score
                    break
        return best

    @classmethod
    def load(cls, run_dir: str | Path) -> ExperimentMemory:
        run_dir = Path(run_dir)
        memory = cls(run_dir=run_dir)
        path = run_dir / "experiments.jsonl"
        if path.exists():
            with path.open() as handler:
                for line in handler:
                    if line.strip():
                        memory.records.append(ExperimentRecord.model_validate(json.loads(line)))
        return memory


def run_output_dir(output_root: str | Path, run_id: str) -> Path:
    return Path(output_root) / run_id


def make_run_id(dataset_name: str | None = None, suffix: str = "") -> str:
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (dataset_name or "run").replace("/", "_").replace(".", "_")[:40]
    tag = f"-{suffix}" if suffix else ""
    return f"{stamp}-{name}{tag}"


def write_provenance(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "provenance.json").open("w") as handler:
        json.dump(payload, handler, indent=2, default=str)
    env = dict(os.environ)
    env = {
        k: v for k, v in env.items() if not k.startswith(("OPENROUTER", "HF_", "NVIDIA", "TOKEN"))
    }
    with (run_dir / "environment.txt").open("w") as handler:
        for key, value in sorted(env.items()):
            handler.write(f"{key}={value}\n")