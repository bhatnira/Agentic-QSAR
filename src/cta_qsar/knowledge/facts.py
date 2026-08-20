"""Knowledge graph evidence layer: attributed facts and a windowed store.

Design rules (see docs/architecture.md):
  - facts are (subject, predicate, object) triples with statistics attached;
    nodes are entities (plugin names, dataset classes, endpoint classes)
  - the store is append-only with a rolling window; merges are monotonic and
    deduplicated by run id
  - the graph is read-only advice for planners: it never triggers
    evaluations, only reallocates the next one
  - retrieval falls back from fine-grained cells to coarser aggregates
    (fine cells require enough evidence, n >= min_n)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WINDOW_SIZE = 20
MIN_EVIDENCE = 2


@dataclass
class Fact:
    """One attributed triple, e.g. subject=dataset-class, predicate=scenario,
    object='maccs+svr[scaffold]', attrs={'mean': 1.21, 'std': 0.08, 'n': 3}."""

    subject: str
    predicate: str
    object: str
    level: int = 2
    source: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.object}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "level": self.level,
            "source": self.source,
            "attrs": self.attrs,
        }


def dataset_class(task_type: str, n_rows: int) -> str:
    """Bucket datasets into coarse classes so evidence transfers between them."""
    if n_rows < 500:
        size = "tiny"
    elif n_rows < 2000:
        size = "small"
    elif n_rows < 5000:
        size = "medium"
    else:
        size = "large"
    return f"{task_type}|{size}"


class EvidenceStore:
    """Windowed, append-only triple store persisted as JSONL.

    Internally keyed by ``(subject, predicate, object)`` tuples so that
    datasets classes containing ``|`` in their name never collide.
    """

    def __init__(self) -> None:
        self._windows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}  # (s,p,o) -> [{value, run_id, sign}]
        self.meta: dict[str, Any] = {"fact_versions": {}, "last_ingested": None}

    def merge(self, other: EvidenceStore) -> None:
        for key, entries in other._windows.items():
            known = {e["run_id"] for e in self._windows.get(key, [])}
            new = [e for e in entries if e["run_id"] not in known]
            self._windows.setdefault(key, []).extend(new)
            self._windows[key] = self._windows[key][-WINDOW_SIZE:]

    def add_value(self, subject: str, predicate: str, object: str, value: float, *, run_id: str, level: int, source: str = "", sign: float = 1.0) -> None:
        key = (subject, predicate, object)
        entries = self._windows.setdefault(key, [])
        entries = [e for e in entries if e["run_id"] != run_id]  # idempotent re-merge
        entries.append({"value": float(value), "run_id": run_id, "sign": float(sign)})
        self._windows[key] = entries[-WINDOW_SIZE:]
        self.meta["last_ingested"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def fact(self, subject: str, predicate: str, object: str) -> Fact | None:
        entries = self._windows.get((subject, predicate, object))
        if not entries:
            return None
        values = [e["value"] for e in entries]
        sign = entries[0].get("sign", 1.0)
        fact = _fact_from_values(subject, predicate, object, len(entries), values)
        fact.attrs["sign"] = sign
        return fact

    def facts_for(self, dataset_class: str, *, min_n: int = MIN_EVIDENCE, exclude_predicate: str | None = None) -> list[Fact]:
        """Finest-grained facts first (level 4), degrading to coarser aggregates.

        Returns the finest evidence available per predicate; a predicate-level
        or class-level aggregate is included only when no fine (n >= min_n)
        fact exists for that predicate. Sorted most-relevant (level, n) first."""
        fine: list[Fact] = []
        for predicate in sorted(self._predicates(dataset_class)):
            if predicate == "*" or predicate == exclude_predicate:
                continue
            for obj in sorted(self._objects(dataset_class, predicate)):
                if obj == "*":
                    continue
                fact = self.fact(dataset_class, predicate, obj)
                if fact and fact.attrs.get("n", 0) >= min_n:
                    fine.append(fact)
        seen_predicates = {f.predicate for f in fine}
        aggregates: list[Fact] = []
        for predicate in sorted(self._predicates(dataset_class)):
            if predicate in ("*",) or predicate == exclude_predicate or predicate in seen_predicates:
                continue
            agg = self.fact(dataset_class, predicate, "*")
            if agg and agg.attrs.get("n", 0) >= min_n:
                aggregates.append(agg)
        if not fine and not aggregates:
            class_agg = self.fact(dataset_class, "*", "*")
            if class_agg and class_agg.attrs.get("n", 0) >= min_n:
                aggregates.append(class_agg)
        result = fine + aggregates
        result.sort(key=lambda f: (f.level, f.attrs.get("n", 0)), reverse=True)
        return result

    def best_for(self, dataset_class: str, predicate: str) -> tuple[str, dict[str, Any]] | None:
        """Highest mean primary metric for a scenario within a dataset class."""
        best: tuple[str, dict[str, Any]] | None = None
        for obj in self._objects(dataset_class, predicate):
            fact = self.fact(dataset_class, predicate, obj)
            if fact is None or fact.attrs.get("n", 0) < MIN_EVIDENCE or obj == "*":
                continue
            sign = fact.attrs.get("sign", 1.0)
            score = sign * fact.attrs.get("mean", 0.0)
            if best is None or score > best[1].get("score", -1e18):
                best = (obj, {**fact.attrs, "score": round(score, 4)})
        return best

    def edge_exists(self, subject: str, predicate: str, object: str) -> bool:
        return self.fact(subject, predicate, object) is not None

    def _predicates(self, subject: str) -> set[str]:
        return {p for (s, p, o) in self._windows if s == subject}

    def _objects(self, subject: str, predicate: str) -> set[str]:
        return {o for (s, p, o) in self._windows if s == subject and p == predicate}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for (subject, predicate, object), entries in sorted(self._windows.items()):
                values = [e["value"] for e in entries]
                fact = _fact_from_values(subject, predicate, object, len(entries), values)
                record = {
                    "key": f"{subject}\u241f{predicate}\u241f{object}",
                    "fact": fact.to_dict(),
                    "values": values,
                    "signs": [e.get("sign", 1.0) for e in entries],
                    "run_ids": [e["run_id"] for e in entries],
                }
                fh.write(json.dumps(record) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> EvidenceStore:
        store = cls()
        path = Path(path)
        if not path.exists():
            return store
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                fact = Fact(**record["fact"])
                signs = record.get("signs", [1.0] * len(record.get("run_ids", [])))
                store._windows[(fact.subject, fact.predicate, fact.object)] = [
                    {"value": float(v), "run_id": rid, "sign": float(s)}
                    for v, rid, s in zip(record.get("values", []), record.get("run_ids", []), signs, strict=False)
                ][-WINDOW_SIZE:]
        return store

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._windows.values())


def _fact_from_values(subject: str, predicate: str, object: str, n: int, values: list[float]) -> Fact:
    mean = sum(values) / n if n else 0.0
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n else 0.0
    return Fact(
        subject=subject, predicate=predicate, object=object,
        level=4 if object != "*" and predicate != "*" else (2 if predicate != "*" else 1),
        attrs={"mean": round(mean, 4), "std": round(std, 4), "n": n, "window": WINDOW_SIZE},
    )