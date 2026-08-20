"""LangGraph nodes implementing the 16-step QSAR workflow.

Each node is a plain callable `(state: dict) -> dict` so the graph remains
testable without LangGraph.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from cta_qsar.core.config import Config
from cta_qsar.core.context import get_context
from cta_qsar.core.logging import agent_log, get_logger
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.endpoints.detector import EndpointDetector
from cta_qsar.memory.experiment_memory import make_run_id, write_provenance

logger = get_logger(__name__)

TEMP_DIR = Path(os.environ.get("CTA_QSAR_TMP", "/tmp"))

_LOADERS = {
    ".csv": lambda path, kwargs: pd.read_csv(path, **kwargs),
    ".tsv": lambda path, kwargs: pd.read_csv(path, sep="\t", **kwargs),
    ".parquet": lambda path, kwargs: pd.read_parquet(path, **kwargs),
}


def ingest_dataset(state: dict[str, Any]) -> dict[str, Any]:
    """1. Load the dataset from disk."""
    data_path = Path(state["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")
    suffix = data_path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"unsupported dataset format {suffix!r}; use CSV, TSV, or Parquet"
        )
    config: Config = state.get("config")
    kwargs = {}
    if suffix in (".csv", ".tsv"):
        kwargs = {"on_bad_lines": "skip"}
    df = loader(data_path, kwargs)
    if config is not None and config.dataset.max_rows and len(df) > config.dataset.max_rows:
        df = df.head(config.dataset.max_rows)
    state = dict(state)
    state["raw_df"] = df
    state["run_id"] = make_run_id(data_path.stem)
    agent_log("ingest", f"loaded {len(df)} rows x {df.shape[1]} cols from {data_path}")
    return state


def profile_dataset_node(state: dict[str, Any]) -> dict[str, Any]:
    """2. Profile the dataset and detect columns."""
    df = state["raw_df"]
    smiles_column = _resolve_smiles_column(df, state.get("smiles_column") or "")
    target_column = _resolve_target_column(df, state.get("target_column") or "")
    state = dict(state)
    state["smiles_column"] = smiles_column
    state["target_column"] = target_column
    state["profile"] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "n_numeric_columns": int(df.select_dtypes("number").shape[1]),
        "n_categorical_columns": int(df.select_dtypes("object").shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_candidates": {"smiles": _smiles_candidates(df), "target": _target_candidates(df)},
    }
    agent_log("profile", "dataset profiled", columns=len(df.columns))
    return state


def _resolve_smiles_column(df: pd.DataFrame, user_hint: str) -> str:
    if user_hint in df.columns:
        return user_hint
    for col in df.columns:
        low = str(col).lower()
        if low in ("smiles", "smi", "smiles_string", "canonical smiles"):
            return str(col)
    scores: dict[str, int] = {}
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(100)
        if sample.empty:
            continue
        parseable = sum(_parseable(s) for s in sample)
        scores[str(col)] = parseable / len(sample)
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0.8:
            return best
    raise ValueError(
        "could not auto-detect the SMILES column; pass --smiles-column explicitly"
    )


def _parseable(smiles: str) -> bool:
    from rdkit import Chem

    return Chem.MolFromSmiles(str(smiles)) is not None


def _resolve_target_column(df: pd.DataFrame, user_hint: str) -> str:
    if user_hint in df.columns:
        return user_hint
    candidates = _target_candidates(df)
    if candidates:
        return candidates[0]
    raise ValueError(
        "could not auto-detect the target column; pass --target-column explicitly"
    )


def _smiles_candidates(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if any(k in str(c).lower() for k in ("smiles", "smi", "canonical"))
    ]


def _target_candidates(df: pd.DataFrame) -> list[str]:
    names = [
        c
        for c in df.columns
        if any(k in str(c).lower() for k in ("activity", "value", "target", "ic50", "pic50", "ec50", "inhib", "logp", "logs", "ki", "kd", "tox", "potency", "endpoint", "label", "y"))
    ]
    if names:
        return names
    numeric = [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and len(df[c].dropna()) > 0
    ]
    return numeric


def standardize_dataset(state: dict[str, Any]) -> dict[str, Any]:
    """3b. Chemical standardization (never destroys original SMILES)."""
    from cta_qsar.chemistry.standardization import STANDARDIZATION_VERSION, MolecularStandardizer

    df = state["raw_df"]
    smiles_column = state["smiles_column"]
    standardizer = MolecularStandardizer(
        canonical=True,
        neutralise=True,
        desalt=True,
        remove_tautomers=False,
    )
    standardized, report = standardizer.fit_transform(df, smiles_column)
    state = dict(state)
    state["standardized_df"] = standardized
    state["preprocessing_version"] = STANDARDIZATION_VERSION
    state["standardization_log"] = report.to_dict()
    agent_log(
        "standardize",
        f"valid={report.n_valid} invalid={report.n_invalid} desalted={report.n_desalted}",
    )
    return state


def detect_endpoint(state: dict[str, Any]) -> dict[str, Any]:
    """3. Detect the endpoint / task type."""
    df = state["raw_df"]
    target_column = state.get("target_column") or ""
    detector = EndpointDetector()
    detection = detector.detect(df, target_column)
    state = dict(state)
    state["endpoint"] = detection.to_dict()
    agent_log(
        "endpoint",
        f"task={detection.task_type} conf={detection.confidence:.2f} "
        f"reason={detection.reasoning}",
    )
    return state


def assess_data_quality(state: dict[str, Any]) -> dict[str, Any]:
    """4. Quality control on top of standardization."""
    from cta_qsar.chemistry.validation import quality_report

    std_df = state.get("standardized_df")
    df = std_df if std_df is not None else state["raw_df"]
    target = state.get("target_column") or state["profile"].get("target_column", "")
    endpoint = state.get("endpoint", {})
    report = quality_report(
        df,
        smiles_column=state["smiles_column"],
        target_column=target or None,
        task_type=endpoint.get("task_type", "regression"),
        endpoint=endpoint,
    )
    state = dict(state)
    state["quality_report"] = report
    n_invalid = report.get("invalid_smiles", 0)
    if n_invalid:
        agent_log("quality", f"{n_invalid} invalid SMILES flagged")
    return state


def characterize_chemical_space(state: dict[str, Any]) -> dict[str, Any]:
    """5. Chemical-space characterization."""
    from cta_qsar.chemistry.chemical_space import summarize_chemical_space

    std_df = state.get("standardized_df")
    df = std_df if std_df is not None else state["raw_df"]
    summary = summarize_chemical_space(df, state["smiles_column"])
    state = dict(state)
    state["chemical_space"] = summary
    return state


def select_validation(state: dict[str, Any]) -> dict[str, Any]:
    """6. Select applicable validation strategies (plugins)."""
    from cta_qsar.validation.cluster_split import ClusterSplit
    from cta_qsar.validation.random_split import RandomSplit
    from cta_qsar.validation.scaffold_split import ScaffoldSplit
    from cta_qsar.validation.stratified import StratifiedSplit
    from cta_qsar.validation.temporal_split import TemporalSplit

    endpoint = state.get("endpoint", {})
    task_type = endpoint.get("task_type", "regression")
    profile = state.get("profile", {})
    config: Config = state["config"]
    enabled_splits = set(config.validation.get("enabled") or [])
    dataset_props = {
        "task_type": task_type,
        "has_temporal_column": bool(profile.get("temporal_columns")),
        "n_rows": profile.get("n_rows", 0),
    }
    plugins = [RandomSplit(), StratifiedSplit(), ScaffoldSplit(), ClusterSplit(), TemporalSplit()]
    chosen: list[str] = []
    for plugin in plugins:
        if plugin.name not in enabled_splits:
            continue
        applicable, _ = plugin.applicability(dataset_props)
        if applicable:
            chosen.append(plugin.name)
    if "scaffold" in chosen and "random" in chosen:
        pass
    state = dict(state)
    state["validated_splits"] = chosen
    state["validation_rationale"] = (
        "random+scaffold always evaluated; stratified for classification; "
        "cluster/temporal gated by data"
    )
    agent_log("validation", f"enabled splits: {chosen}")
    return state


def generate_candidate_representations(state: dict[str, Any]) -> dict[str, Any]:
    """7. Representation candidates for this endpoint/data scale."""
    config: Config = state["config"]
    enabled = config.representations.get("enabled")
    state = dict(state)
    state["representation_candidates"] = list(
        dict.fromkeys(enabled or []) if enabled else []
    )
    return state


def generate_candidate_models(state: dict[str, Any]) -> dict[str, Any]:
    """8. Model candidates for this task type."""
    config: Config = state["config"]
    enabled = config.models.get("enabled")
    state = dict(state)
    state["model_candidates"] = list(dict.fromkeys(enabled or []))
    return state


def plan_experiment(state: dict[str, Any]) -> dict[str, Any]:
    """9. Score candidates and select the next experiment."""
    from cta_qsar.agents.planner_agent import PlannerAgent
    from cta_qsar.core.interfaces import QSARCase
    from cta_qsar.experiments.budget import BudgetState
    from cta_qsar.experiments.planner import pick_first_usable

    registry: PluginRegistry = get_context().registry
    llm = get_context().llm
    config: Config = state["config"]
    endpoint = state.get("endpoint", {})
    case = QSARCase(
        dataset_size=state["profile"].get("n_rows", 0),
        n_unique_molecules=state.get("chemical_space", {}).get("n_unique_molecules", 0),
        task_type=endpoint.get("task_type", "regression"),
        endpoint_name=endpoint.get("endpoint_name", ""),
        endpoint_confidence=endpoint.get("confidence", 0.0),
        endpoint_reasoning=endpoint.get("reasoning", ""),
        risks=state.get("quality_report", {}).get("risks", []),
    )
    history = state.get("experiments", [])
    budget = BudgetState(
        max_experiments=config.compute.max_experiments,
        max_minutes=config.compute.max_minutes,
        max_memory_gb=config.compute.max_memory_gb,
    )
    budget.experiments_done = len(
        [e for e in history if _plain(e).get("result") == "completed"]
    )

    planner = PlannerAgent(registry, llm)
    candidates, llm_refined = planner.plan(
        case=case,
        enabled_representations=config.representations.get("enabled"),
        enabled_models=config.models.get("enabled"),
        validated_splits=state.get("validated_splits", ["random"]),
        budget=budget,
        history=history,
        dataset_props={"task_type": case.task_type, "n_rows": case.dataset_size},
        n_samples=case.dataset_size,
        hardware_tier="cpu",
    )
    if llm_refined:
        candidates = llm_refined + [c for c in candidates if c not in llm_refined]
    chosen = pick_first_usable(candidates, budget, history)
    state = dict(state)
    state["candidates"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in candidates]
    state["plan_round"] = state.get("plan_round", 0) + 1
    state["budget_state"] = budget.to_dict()
    if chosen is None:
        if budget.experiments_done >= budget.max_experiments:
            state["stop_reasons"] = ["experiment budget exhausted"]
        elif budget.exhausted:
            state["stop_reasons"] = ["compute budget exhausted"]
        else:
            state["stop_reasons"] = ["no feasible experiment candidates remain"]
        state["experiments_remaining"] = 0
        state["selected_candidate"] = None
        agent_log("plan", f"no candidate selected: {state['stop_reasons'][0]}")
        return state
    state["selected_candidate"] = chosen.model_dump()
    agent_log(
        "plan",
        f"selected #{chosen.rank} {chosen.representation}+{chosen.model}"
        f"[{chosen.validation}] utility={chosen.utility:.3f} cost={chosen.compute_cost:.1f}",
        reason=chosen.reason,
    )
    return state


def execute_experiment(state: dict[str, Any]) -> dict[str, Any]:
    """10. Run the selected experiment."""
    from cta_qsar.core.interfaces import ExperimentCandidate
    from cta_qsar.experiments.budget import BudgetState
    from cta_qsar.experiments.runner import ExperimentRunner

    config: Config = state["config"]
    candidate = ExperimentCandidate.model_validate(state["selected_candidate"])
    std_df = state.get("standardized_df") if state.get("standardized_df") is not None else state["raw_df"]
    smiles = std_df[state["smiles_column"]].astype(str).tolist()
    target = state.get("target_column") or state["profile"].get("target_column", "")

    if target not in std_df.columns:
        raise ValueError(f"target column {target!r} missing after standardization")

    from cta_qsar.experiments.runner import dataset_hash

    runner = ExperimentRunner(
        registry=get_context().registry,
        task_type=state["endpoint"].get("task_type", "regression"),
        n_splits=config.experiment.n_splits,
        n_repeats=config.experiment.n_repeats,
        test_fraction=config.experiment.test_fraction,
        random_seed=config.experiment.random_seed,
        dataset_hash=dataset_hash(std_df),
        preprocessing_version=state.get("preprocessing_version", "unknown"),
    )
    budget = BudgetState(
        max_experiments=config.compute.max_experiments,
        max_minutes=config.compute.max_minutes,
        max_memory_gb=config.compute.max_memory_gb,
    )
    budget.experiments_done = len(
        [e for e in state.get("experiments", []) if _plain(e).get("result") == "completed"]
    )
    try:
        record = runner.run(
            candidate,
            smiles=smiles,
            df=std_df,
            target_column=target,
            budget=budget,
            llm_decision="heuristic",
            rationale=f"plan round {state.get('plan_round', 1)}",
        )
        state = dict(state)
        state["experiments"] = [*(state.get("experiments") or []), record]
        state["current_experiment"] = record
        state["experiments_remaining"] = max(
            0, config.compute.max_experiments - budget.experiments_done - 1
        )
        agent_log(
            "execute",
            f"completed {record.representation}+{record.model}[{record.split}] "
            f"in {record.runtime_seconds:.1f}s",
        )
        return state
    except Exception as exc:  # noqa: BLE001
        logger.exception("experiment failed")
        failed = {
            "id": uuid.uuid4().hex[:8],
            "dataset_hash": dataset_hash(std_df) if std_df is not None else "",
            "preprocessing_version": state.get("preprocessing_version", "unknown"),
            "representation": candidate.representation,
            "model": candidate.model,
            "hyperparameters": {},
            "split": candidate.validation,
            "random_seed": config.experiment.random_seed,
            "result": "failed",
            "tags": {"error": str(exc)},
            "metrics": {},
            "trust": {},
            "runtime_seconds": 0.0,
            "memory_gb": 0.0,
            "llm_decision": "heuristic",
            "rationale": f"plan round {state.get('plan_round', 1)}",
            "failure_diagnosis": [],
            "intervention": [],
        }
        state = dict(state)
        # Record the failure in scientific memory so the planner never
        # re-selects an identical (already failed) experiment.
        state["experiments"] = [*(state.get("experiments") or []), failed]
        state["failed_experiment"] = failed
        state["error"] = str(exc)
        agent_log("execute", f"experiment FAILED: {exc}")
        return state


def evaluate_performance(state: dict[str, Any]) -> dict[str, Any]:
    """11. Assess predictive performance from the experiment record."""
    record = state.get("current_experiment")
    if record is None:
        return state
    state = dict(state)
    state["trust_reports"] = [*(state.get("trust_reports") or []), record.trust]
    agent_log(
        "evaluate",
        f"metrics: {json.dumps(record.metrics, default=str)[:200]}",
    )
    return state


def evaluate_trust(state: dict[str, Any]) -> dict[str, Any]:
    """12. Overall trustworthiness assessment."""
    record = state.get("current_experiment")
    if record is None:
        return state
    trust = record.trust
    verdicts: dict[str, str] = {}
    predictive = trust.get("predictive", {})
    if predictive.get("evaluated", True):
        primary = predictive.get("primary_metric", "rmse")
        value = None
        if isinstance(predictive.get(primary), dict):
            value = predictive[primary].get("mean")
        r2 = predictive.get("r2", {}).get("mean") if isinstance(predictive.get("r2"), dict) else None
        if primary == "rmse" and value is not None:
            target_std = state.get("quality_report", {}).get("target_distribution", {}).get("std", 1.0) or 1.0
            verdicts["predictive"] = "good" if value / max(target_std, 1e-9) < 0.6 else "moderate" if value / max(target_std, 1e-9) < 0.9 else "weak"
        elif r2 is not None:
            verdicts["predictive"] = "good" if r2 >= 0.7 else "moderate" if r2 >= 0.4 else "weak"
    generalization = trust.get("generalization", {})
    if generalization.get("evaluated"):
        g_r2 = generalization.get("r2", {}).get("mean") if isinstance(generalization.get("r2"), dict) else None
        verdicts["generalization"] = "good" if (g_r2 or 0) >= 0.6 else "moderate" if (g_r2 or 0) >= 0.3 else "weak"
    ad = trust.get("applicability_domain", {})
    if ad.get("evaluated"):
        median = ad.get("nn_tanimoto", {}).get("median", 0.5)
        verdicts["applicability_domain"] = "wide" if median >= 0.5 else "narrow"
    state = dict(state)
    state["trust_verdicts"] = verdicts
    agent_log("trust", f"verdicts: {verdicts}")
    return state


def diagnose_failure(state: dict[str, Any]) -> dict[str, Any]:
    """13. Diagnose weaknesses (deterministic + LLM)."""
    from cta_qsar.agents.diagnosis_agent import DiagnosisAgent

    record = state.get("current_experiment")
    if record is None:
        return state
    agent = DiagnosisAgent(get_context().llm)
    diagnoses, interventions = agent.run(
        record, {"experiments": state.get("experiments", [])}
    )
    state = dict(state)
    state["diagnoses"] = [*(state.get("diagnoses") or []), *diagnoses]
    state["interventions"] = interventions
    if diagnoses:
        agent_log(
            "diagnose",
            "found: " + ", ".join(d.failure_type for d in diagnoses),
        )
    return state


def propose_intervention(state: dict[str, Any]) -> dict[str, Any]:
    """14. Ranked interventions for the next experiment."""
    interventions = state.get("interventions", [])
    state = dict(state)
    state["ranked_interventions"] = [
        i.model_dump() for i in interventions[:5]
    ]
    top = interventions[0] if interventions else None
    if top is not None:
        agent_log(
            "intervene",
            f"proposed {top.name} (value/cost={(top.expected_improvement + top.expected_trust_gain) / max(top.compute_cost, 1e-6):.2f})",
        )
    return state


def decide_next_action(state: dict[str, Any]) -> str:
    """15. Route back to planning or finalize (stopping policy + optional LLM veto)."""
    from cta_qsar.orchestration.policies import decide_next

    config: Config = state["config"]
    experiments = state.get("experiments", [])
    no_improvement_rounds = _no_improvement_rounds(experiments)
    candidates_remaining = len(
        [c for c in state.get("candidates", []) if c.get("utility", 0) > 0]
    ) - len(experiments)
    budget_info = state.get("budget_state", {})
    if not budget_info:
        budget_info = {
            "max_experiments": config.compute.max_experiments,
            "max_minutes": config.compute.max_minutes,
            "elapsed_minutes": 0.0,
        }
    llm = get_context().llm
    llm_stop: dict[str, Any] | None = None
    if llm is not None:
        try:
            recent = [
                {
                    "id": r.get("id"),
                    "representation": r.get("representation"),
                    "model": r.get("model"),
                    "split": r.get("split"),
                    "result": r.get("result"),
                    "metrics": r.get("metrics", {}),
                }
                for r in (_plain(e) for e in experiments)
            ][-3:]
            stop = llm.decide_stop(
                {
                    "experiments_remaining": max(
                        0,
                        int(budget_info.get("max_experiments", 12))
                        - len([e for e in experiments if _plain(e).get("result") == "completed"]),
                    ),
                    "no_improvement_rounds": no_improvement_rounds,
                    "budget": budget_info,
                    "recent_experiments": recent,
                }
            )
            if stop.should_stop:
                llm_stop = {"should_stop": True, "reason": stop.reason}
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM stopping assessment unavailable: %s", exc)
    decision = decide_next(
        budget=budget_info,
        plan_round=state.get("plan_round", 0),
        experiments=[e.model_dump() if hasattr(e, "model_dump") else e for e in experiments],
        no_improvement_rounds=no_improvement_rounds,
        candidates_remaining=max(candidates_remaining, 0),
        llm_stop=llm_stop,
    )
    agent_log("route", f"decision: {decision}")
    return decision


def _no_improvement_rounds(experiments: list[Any]) -> int:
    if len(experiments) < 2:
        return 0
    completed = [
        e.model_dump() if hasattr(e, "model_dump") else e
        for e in experiments
        if getattr(e, "result", (e.model_dump() if hasattr(e, "model_dump") else e).get("result"))
        == "completed"
    ]
    if len(completed) < 2:
        return 0
    key = None
    for k in ("rmse", "roc_auc", "mcc", "balanced_accuracy"):
        if k in completed[-1].get("metrics", {}) and k in completed[-2].get("metrics", {}):
            key = k
            break
    if key is None:
        return 0
    last = completed[-1]["metrics"][key]
    prev = completed[-2]["metrics"][key]
    delta = (prev - last) if key in ("rmse", "mae") else (last - prev)
    return 0 if delta >= 0.01 else _no_improvement_rounds(completed[:-1]) + 1


def finalize_report(state: dict[str, Any]) -> dict[str, Any]:
    """16. Scientific report + MLflow tracking + persistence."""
    from cta_qsar.reporting.export import export_report
    from cta_qsar.reporting.report import build_report

    config: Config = state["config"]
    experiments = state.get("experiments", [])
    plain = [e.model_dump() if hasattr(e, "model_dump") else e for e in experiments]
    completed = [e for e in plain if e.get("result") == "completed"]
    stop_reasons = state.get("stop_reasons", []) or []
    if not stop_reasons:
        from cta_qsar.orchestration.policies import evaluate_stopping

        budget_info = state.get("budget_state", {}) or {
            "max_experiments": config.compute.max_experiments,
            "max_minutes": config.compute.max_minutes,
            "elapsed_minutes": 0.0,
        }
        budget_info = {**budget_info, "elapsed_minutes": budget_info.get("elapsed_minutes", 0.0)}
        stop_reasons = evaluate_stopping(
            budget=budget_info,
            plan_round=state.get("plan_round", 0),
            experiments=completed,
            no_improvement_rounds=_no_improvement_rounds(experiments),
        )
    run_id = state.get("run_id", make_run_id())
    output_dir = Path(state.get("output_dir") or config.reporting.get("output_dir", "runs"))
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    llm = get_context().llm
    summary = ""
    report = build_report(
        state=state,
        stop_reasons=stop_reasons or ["stopping rule triggered"],
        llm_summary=summary,
    )
    if llm is not None:
        try:
            report["executive_summary"] = llm.summarize(
                {
                    "experiments": completed,
                    "best_experiment": report.get("best_experiment", {}),
                    "stop_reasons": stop_reasons,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM summary failed: %s", exc)
            report["executive_summary"] = summary

    export_report(report, run_dir, formats=("json", "markdown"))
    with open(run_dir / "experiments.jsonl", "w", encoding="utf-8") as fh:
        for rec in plain:
            fh.write(json.dumps(rec, default=str) + "\n")
    provenance = {
        "run_id": run_id,
        "dataset": state.get("data_path"),
        "dataset_hash": _dataset_hash_digest(state),
        "endpoint": state.get("endpoint", {}),
        "config": config.model_dump(mode="json"),
        "hardware": _hardware_probe(),
        "llm": {"provider": getattr(llm, "provider_name", "mock"), "model": getattr(llm, "model", "heuristic")}
        if llm
        else {"provider": "mock", "model": "heuristic"},
        "constraints": {"max_experiments": config.compute.max_experiments, "max_minutes": config.compute.max_minutes},
    }
    write_provenance(run_dir, provenance)
    _maybe_mlflow(state, report, run_dir)
    state = dict(state)
    state["final_report"] = report
    state["output_dir"] = str(run_dir)
    state["stop_reasons"] = [*(state.get("stop_reasons") or []), *stop_reasons]
    agent_log("finalize", f"report written to {run_dir}")
    return state


def _plain(record: Any) -> dict[str, Any]:
    if record is None:
        return {}
    return record.model_dump() if hasattr(record, "model_dump") else record


def _dataset_hash_digest(state: dict[str, Any]) -> str:
    std_df = state.get("standardized_df")
    if std_df is None:
        return ""
    from cta_qsar.experiments.runner import dataset_hash

    return dataset_hash(std_df)


def _hardware_probe() -> dict[str, Any]:
    from cta_qsar.hardware.profiler import probe

    return probe().to_dict()


def _maybe_mlflow(state: dict[str, Any], report: dict[str, Any], run_dir: Path) -> None:
    config: Config = state["config"]
    if not config.tracking.enabled or config.tracking.backend != "mlflow":
        return
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; skipping tracking")
        return
    try:
        # SQLite backend: the legacy filesystem store is deprecated in MLflow 3.x
        default_uri = "sqlite:///" + str((run_dir.parent / "mlflow.db").resolve())
        uri = os.getenv("MLFLOW_TRACKING_URI") or default_uri
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=f"cta-qsar-{state.get('run_id', 'run')}"):
            for experiment in report.get("experiments", []):
                with mlflow.start_run(run_name=f"{experiment.get('representation')}+{experiment.get('model')}", nested=True):
                    mlflow.log_params(
                        {
                            "representation": experiment.get("representation", ""),
                            "model": experiment.get("model", ""),
                            "split": experiment.get("split", ""),
                            "seed": experiment.get("random_seed", 42),
                        }
                    )
                    mlflow.log_metrics(experiment.get("metrics", {}))
                    mlflow.log_dict(experiment, f"experiment_{experiment.get('id', 'x')}.json")
            mlflow.log_dict(report, "report.json")
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow tracking failed: %s", exc)