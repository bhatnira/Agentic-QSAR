"""Federated challenge runner: multiple independent strategy agents compete.

Each dataset x seed run launches every strategy card as an independent agent
(separate run dir, own config). Outcomes are ranked on the primary metric
(leaderboard), and in-progress results are digested back into the shared
evidence store + in-memory knowledge graph after every agent, so knowledge
accumulates across the whole challenge (and across challenges on disk).

Usage:
    python3 benchmarks/run_challenge.py --datasets esol --seeds 0,1 --cards gate:aggressive,evolve
    python3 benchmarks/run_challenge.py --datasets esol,bace --seeds 0-4 --with-nvidia --fresh-evidence
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import run_benchmark as rb

from cta_qsar.federation.cards import default_cards
from cta_qsar.federation.core import ChallengeSession
from cta_qsar.knowledge.facts import EvidenceStore
from cta_qsar.policy.state import PolicyStore

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "data"
RUNS_ROOT = ROOT / "benchmarks" / "runs"
CHALLENGES_ROOT = ROOT / "benchmarks" / "challenges"
KNOWLEDGE_EVIDENCE = ROOT / "benchmarks" / "knowledge" / "evidence.jsonl"
POLICY_STATE = KNOWLEDGE_EVIDENCE.parent / "policy_state.jsonl"

LEADERBOARD_COLUMNS = [
    "dataset", "seed", "rank", "card", "primary", "primary_value", "best_model",
    "n_experiments", "runtime_seconds", "run_id", "failed",
]


def _agent_fn(seed: int, dataset: rb.BenchmarkDataset, *, evidence_path: Path, policy_path: Path) -> Any:
    """Real autonomous agent runner bound to one dataset + seed."""
    from cta_qsar.agents.scientist import QSARScientist
    from cta_qsar.core.config import build_config

    dataset_name = dataset.name
    task_type = dataset.task_type

    def run(card: Any, df: Any, **kwargs: Any) -> dict[str, Any]:
        config = build_config(
            seed=seed,
            hyperparameter_search=True,
            llm_provider=card.llm_provider,
        )
        config.experiment.hyperparameter_search = card.search
        config.experiment.n_repeats = 1
        config.compute.max_experiments = 6
        config.compute.max_minutes = 30.0
        config.dataset.smiles_column = dataset.smiles_column
        config.dataset.target_column = dataset.target_column
        if dataset.target_columns:
            config.dataset.target_columns = list(dataset.target_columns)
        config.reporting["output_dir"] = str(RUNS_ROOT)
        config.knowledge.evidence_path = str(evidence_path)
        if card.adaptive_policy:
            config.policy.adaptive = True
            config.policy.policy_state_path = str(policy_path)
        if not card.trust_gate:
            config.trust.required = []
        if card.enabled_representations is not None:
            config.representations["enabled"] = list(card.enabled_representations)
        if card.enabled_models is not None:
            config.models["enabled"] = list(card.enabled_models)

        data_csv = DATA_DIR / f"{dataset_name}_seed{seed}_{card.name}.csv"
        df.to_csv(data_csv, index=False)
        scientist = QSARScientist(config)
        started = time.time()
        final = scientist.run(data_csv)
        runtime = round(time.time() - started, 1)

        experiments = final.get("experiments", []) if isinstance(final, dict) else []
        completed = [e for e in experiments if e.get("result") == "completed"]
        primary = rb._primary(task_type)  # noqa: SLF001
        best = None
        for exp in completed:
            metrics = exp.get("metrics", {})
            if primary not in metrics:
                continue
            value = float(metrics[primary])
            if best is None or rb._better(value, best["metrics"][primary], task_type):  # noqa: SLF001
                best = exp
        if best is None:
            raise RuntimeError(f"no completed experiments for {dataset_name} seed {seed} card {card.name}")
        return {
            "card": card.name,
            "seed": seed,
            "primary": primary,
            "primary_value": float(best["metrics"][primary]),
            "best_model": f"{best['representation']}+{best['model']}[{best['split']}]",
            "best_hyperparams": json.dumps(best.get("hyperparameters", {})),
            "n_experiments": len(completed),
            "runtime_seconds": runtime,
            "run_id": final.get("run_id", ""),
            "failed": "",
        }

    return run


def _resolve_cards(names: str, with_nvidia: bool) -> list[Any]:
    if not names:
        return default_cards(with_nvidia=with_nvidia)
    by_name = {c.name: c for c in default_cards(with_nvidia=True)}
    cards = []
    for name in names.split(","):
        card = by_name.get(name.strip())
        if card is not None:
            cards.append(card)
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="esol", help="comma-separated subset of the benchmark datasets")
    parser.add_argument("--seeds", default="0,1", help="comma-separated seeds (0-4 supported)")
    parser.add_argument("--cards", default="",
                        help="comma-separated strategy cards; empty = all default cards")
    parser.add_argument("--with-nvidia", action="store_true",
                        help="include the llm:nvidia card (requires NVIDIA_API_KEY/OPENAI_API_KEY)")
    parser.add_argument("--fresh-evidence", action="store_true",
                        help="reset the shared evidence + policy state before the challenge")
    parser.add_argument("--evidence", default=str(KNOWLEDGE_EVIDENCE),
                        help="evidence store path (default: shared; use an isolated path while "
                             "other campaigns are running to avoid cross-contamination)")
    args = parser.parse_args(argv)

    evidence_path = Path(args.evidence)
    policy_path = evidence_path.with_name(evidence_path.stem + "_policy.jsonl")

    datasets = [rb.DATASETS[name] for name in args.datasets.split(",") if name in rb.DATASETS]
    seeds = [int(s) for s in args.seeds.replace("-", ",").split(",") if s.strip() != ""]
    cards = _resolve_cards(args.cards, args.with_nvidia)
    if not datasets or not seeds or not cards:
        parser.error("need valid --datasets, --seeds and at least one card")
    if "llm:nvidia" in [c.name for c in cards] and not any(
        key in ("NVIDIA_API_KEY", "OPENAI_API_KEY") for key in dict(__import__("os").environ)
    ):
        parser.error("llm:nvidia requires NVIDIA_API_KEY/OPENAI_API_KEY in the environment")

    if args.fresh_evidence:
        EvidenceStore().save(evidence_path)
        PolicyStore(policy_path).save()
        print("fresh evidence: knowledge + policy state reset to empty (cold-start)", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = CHALLENGES_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    inprogress = out_dir / "inprogress.csv"

    ranked_rows: list[dict[str, Any]] = []
    from cta_qsar.core.registry import get_registry

    registry = get_registry()
    registry.auto_discover()
    for dataset in datasets:
        df = rb.load_dataset(dataset)
        for seed in seeds:
            print(f"\n=== challenge {dataset.name} seed {seed} ===", flush=True)
            session = ChallengeSession(
                evidence_path=evidence_path,
                agent_fn=_agent_fn(seed, dataset, evidence_path=evidence_path, policy_path=policy_path),
                registry=registry,
            )
            try:
                report = session.run_challenge(
                    df=df,
                    dataset_name=dataset.name,
                    task_type=dataset.task_type,
                    n_rows=len(df),
                    seed=seed,
                    cards=cards,
                    inprogress_path=inprogress,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  challenge FAILED: {exc}", file=sys.stderr)
                continue
            (out_dir / f"challenge_{dataset.name}_seed{seed}.json").write_text(
                json.dumps(report.to_dict(), indent=2) + "\n"
            )
            print(f"  leaderboard ({report.primary}, dataset class {report.dataset_class}):")
            for row in report.leaderboard:
                rank = row.get("rank")
                tag = f"#{rank}" if rank else "FAILED"
                print(f"    {tag} {row['card']:<18} primary={row['primary_value']:<8} "
                      f"{row['best_model']:<32} {row['runtime_seconds']}s")
            if report.winner:
                print(f"  WINNER: {report.winner['card']} "
                      f"({report.winner['best_model']}, primary={report.winner['primary_value']})")
            print(f"  KG: {report.kg_before['n_edges']} -> {report.kg_after['n_edges']} edges, "
                  f"+{len(report.kg_diff['added_edges'])} added")
            session.kg.to_jsonl(out_dir / f"kg_{dataset.name}_seed{seed}.jsonl")
            for row in report.leaderboard:
                row.update({"dataset": dataset.name, "seed": seed})
                ranked_rows.append(row)

    ch_csv = out_dir / "challenge_leaderboard.csv"
    if ranked_rows:
        with ch_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in sorted(ranked_rows, key=lambda r: (r["dataset"], r["seed"], r.get("rank") or 99)):
                writer.writerow(row)
        print(f"\nleaderboard: {ch_csv}")
    print(f"challenge artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())