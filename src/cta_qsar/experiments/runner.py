"""Experiment executor: trains/evaluates a (representation, model, split)
triple and produces a complete scientific ExperimentRecord."""

from __future__ import annotations

import hashlib
import itertools
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from cta_qsar.core.interfaces import ExperimentCandidate, ExperimentRecord
from cta_qsar.core.logging import get_logger
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState
from cta_qsar.trust.base import primary_metric

logger = get_logger(__name__)


def dataset_hash(df: Any) -> str:
    """Stable hash from canonical SMILES + target values."""
    canonical = df.get("standardized_smiles")
    if canonical is None:
        canonical = df.iloc[:, 0]
    target = df.get("target_column")
    payload = ";".join(
        f"{s}|{t}" for s, t in zip(canonical.astype(str), (target or df.columns[0]), strict=False)
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


class ExperimentRunner:
    def __init__(
        self,
        registry: PluginRegistry,
        *,
        task_type: str,
        n_splits: int = 5,
        n_repeats: int = 2,
        test_fraction: float = 0.2,
        random_seed: int = 42,
        dataset_hash: str = "",
        preprocessing_version: str = "",
        hyperparameter_search: bool = False,
    ) -> None:
        self.registry = registry
        self.is_multitask = task_type.startswith("multitask_")
        self.task_type = task_type.removeprefix("multitask_")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.test_fraction = test_fraction
        self.random_seed = random_seed
        self.dataset_hash = dataset_hash
        self.preprocessing_version = preprocessing_version
        self.hyperparameter_search = hyperparameter_search

    def run(
        self,
        candidate: ExperimentCandidate,
        *,
        smiles: list[str],
        df: Any,
        target_column: str,
        budget: BudgetState,
        llm_decision: str = "",
        rationale: str = "",
        extra_tags: dict[str, Any] | None = None,
        target_columns: list[str] | None = None,
    ) -> ExperimentRecord:
        if self.is_multitask and target_columns and len(target_columns) > 1:
            return self._run_multitask(
                candidate,
                smiles=smiles,
                df=df,
                target_columns=target_columns,
                budget=budget,
                llm_decision=llm_decision,
                rationale=rationale,
                extra_tags=extra_tags,
            )
        started = time.time()
        start_mem = _mem_gb()

        rep_plugin = self.registry.get("representation", candidate.representation)
        model_plugin = self.registry.get("model", candidate.model)

        # Modeling rows: drop invalid-SMILES and missing-target rows *with
        # provenance* (the standardized dataset itself is never modified).
        # Classification labels may be strings: only blank/NaN rows are
        # missing; labels are label-encoded downstream, never float-coerced.
        drop_reasons: dict[str, int] = {}
        if df is not None:
            raw = df[target_column]
            if self.task_type in ("binary", "multiclass"):
                missing = raw.isna() | (raw.astype(str).str.strip() == "")
                y_raw: Any = raw.astype(object)
            else:
                y_raw = pd.to_numeric(raw, errors="coerce")
                missing = y_raw.isna()
            valid = (~missing).to_numpy()
            if valid.sum() < len(valid):
                drop_reasons["missing_target"] = int((~valid).sum())
            if "smiles_valid" in df.columns:
                mask = df["smiles_valid"].fillna(False).to_numpy()
                missing_smiles = int((~mask).sum())
                if missing_smiles:
                    drop_reasons["invalid_smiles"] = missing_smiles
                valid &= mask
            if ~valid.all():
                df = df[valid].reset_index(drop=True)
                smiles = [s for s, ok in zip(smiles, valid, strict=False) if ok]
                y_raw = y_raw[valid]

        # 1. feature matrix
        X = self._featurize(rep_plugin, smiles, candidate.representation)

        # 2. target vector
        y = y_raw.to_numpy()
        if self.task_type in ("binary", "multiclass"):
            y = _encode_labels(y)

        # 3. splits
        validation_plugin = self.registry.get("validation", candidate.validation)
        splits = self._make_splits(
            validation_plugin, df, y, candidate.validation
        )
        self._folds_cache = splits

        # 4. model
        if hasattr(X, "shape"):
            n_features = X.shape[1]
        else:
            n_features = getattr(X[0], "n_atoms", None) or len(X[0])
        hyperparams = self._pick_hyperparams(candidate, n_features, budget, X=X, y=y)
        estimator = self._build_estimator(
            model_plugin, candidate.model, n_features, hyperparams, y=y
        )

        # 5. trust evaluation
        trust = self._evaluate_trust(
            estimator=estimator,
            rep_plugin=rep_plugin,
            X=X,
            y=y,
            split=validation_plugin,
            candidate=candidate,
            smiles=smiles,
        )

        # 6. wrap up record
        runtime = time.time() - started
        memory = max(_mem_gb() - start_mem, 0.05)
        tags = dict(extra_tags or {})
        if drop_reasons:
            tags["dropped_rows"] = drop_reasons
            logger.info(
                "dropped rows for modeling (with reasons): %s", drop_reasons,
            )
        record = ExperimentRecord(
            id=str(uuid.uuid4())[:8],
            dataset_hash=self.dataset_hash,
            preprocessing_version=self.preprocessing_version,
            representation=candidate.representation,
            model=candidate.model,
            hyperparameters=hyperparams,
            split=candidate.validation,
            random_seed=self.random_seed,
            metrics=_extract_primary(trust),
            trust=trust,
            runtime_seconds=round(runtime, 2),
            memory_gb=round(memory, 3),
            llm_decision=llm_decision,
            rationale=rationale,
            result="completed",
            tags=tags,
        )
        budget.record(runtime, memory)
        return record

    def _run_multitask(
        self,
        candidate: ExperimentCandidate,
        *,
        smiles: list[str],
        df: Any,
        target_columns: list[str],
        budget: BudgetState,
        llm_decision: str = "",
        rationale: str = "",
        extra_tags: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Train one estimator per aligned target column; metrics are the
        mean across targets (folds and hyperparameters are shared)."""
        started = time.time()
        start_mem = _mem_gb()

        rep_plugin = self.registry.get("representation", candidate.representation)
        model_plugin = self.registry.get("model", candidate.model)

        drop_reasons: dict[str, int] = {}
        valid = np.ones(len(df), dtype=bool)
        if "smiles_valid" in df.columns:
            mask = df["smiles_valid"].fillna(False).to_numpy()
            n_bad = int((~mask).sum())
            if n_bad:
                drop_reasons["invalid_smiles"] = n_bad
            valid &= mask
        for col in target_columns:
            raw = df[col]
            if self.task_type in ("binary", "multiclass"):
                missing = raw.isna() | (raw.astype(str).str.strip() == "")
            else:
                missing = pd.to_numeric(raw, errors="coerce").isna()
            n_missing = int(missing.sum())
            if n_missing:
                drop_reasons[f"missing_target:{col}"] = n_missing
            valid &= (~missing).to_numpy()
        if ~valid.all():
            df = df[valid].reset_index(drop=True)
            smiles = [s for s, ok in zip(smiles, valid, strict=False) if ok]
        y_list: list[np.ndarray] = []
        for col in target_columns:
            if self.task_type in ("binary", "multiclass"):
                y_list.append(_encode_labels(df[col].to_numpy()))
            else:
                y_list.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))

        X = self._featurize(rep_plugin, smiles, candidate.representation)
        validation_plugin = self.registry.get("validation", candidate.validation)
        splits = self._make_splits(validation_plugin, df, y_list[0], candidate.validation)
        self._folds_cache = splits

        if hasattr(X, "shape"):
            n_features = X.shape[1]
        else:
            n_features = getattr(X[0], "n_atoms", None) or len(X[0])
        hyperparams = self._pick_hyperparams(
            candidate, n_features, budget, X=X, y=y_list[0], targets=y_list
        )
        trust_list: list[dict[str, Any]] = []
        for y in y_list:
            estimator = self._build_estimator(
                model_plugin, candidate.model, n_features, hyperparams, y=y
            )
            trust_list.append(
                self._evaluate_trust(
                    estimator=estimator,
                    rep_plugin=rep_plugin,
                    X=X,
                    y=y,
                    split=validation_plugin,
                    candidate=candidate,
                    smiles=smiles,
                )
            )
        trust = _merge_multitask_trust(trust_list)

        runtime = time.time() - started
        memory = max(_mem_gb() - start_mem, 0.05)
        tags = dict(extra_tags or {})
        tags["targets"] = list(target_columns)
        tags["per_target_primary"] = {}
        for col, t in zip(target_columns, trust_list, strict=False):
            pred = t.get("predictive", {})
            key = pred.get("primary_metric", primary_metric(self.task_type))
            value = pred.get(key, {}).get("mean")
            tags["per_target_primary"][col] = (
                round(float(value), 4) if isinstance(value, (int, float)) else None
            )
        record = ExperimentRecord(
            id=str(uuid.uuid4())[:8],
            dataset_hash=self.dataset_hash,
            preprocessing_version=self.preprocessing_version,
            representation=candidate.representation,
            model=candidate.model,
            hyperparameters=hyperparams,
            split=candidate.validation,
            random_seed=self.random_seed,
            metrics=_extract_primary(trust),
            trust=trust,
            runtime_seconds=round(runtime, 2),
            memory_gb=round(memory, 3),
            llm_decision=llm_decision,
            rationale=rationale,
            result="completed",
            tags=tags,
        )
        budget.record(runtime, memory)
        return record

    # -- internals ---------------------------------------------------------
    def _featurize(self, rep_plugin: Any, smiles: list[str], rep_name: str) -> Any:
        from cta_qsar.representations.registry import representation_matrix

        return representation_matrix(rep_plugin, smiles, fit=True)

    def _make_splits(self, validation_plugin: Any, df: Any, y: np.ndarray, split_name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        from cta_qsar.validation.base import make_cv_folds

        groups: np.ndarray | None = None
        if split_name in ("scaffold", "cluster"):
            canonical = df["standardized_smiles"].tolist() if "standardized_smiles" in df.columns else df[df.columns[0]].tolist()
            if split_name == "scaffold":
                from cta_qsar.validation.scaffold_split import scaffold_id

                groups = np.asarray([scaffold_id(s) or f"none-{i}" for i, s in enumerate(canonical)])
            else:
                from cta_qsar.validation.cluster_split import cluster_groups

                groups = cluster_groups(list(canonical))
        if split_name == "temporal":
            groups = df["_temporal"].to_numpy() if "_temporal" in df.columns else None
        strategy = {
            "random": "random",
            "stratified": "stratified",
            "repeated_cv": "repeated_cv",
            "scaffold": "scaffold",
            "cluster": "cluster",
            "temporal": "temporal",
        }.get(split_name, "random")
        return make_cv_folds(
            len(df),
            y,
            strategy=strategy,
            n_splits=self.n_splits,
            n_repeats=1 if split_name == "scaffold" else self.n_repeats,
            random_seed=self.random_seed,
            test_fraction=self.test_fraction,
            groups=groups,
        )

    def _pick_hyperparams(
        self,
        candidate: ExperimentCandidate,
        n_features: int,
        budget: BudgetState,
        X: Any | None = None,
        y: Any | None = None,
        targets: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        space = _model_space(self.registry, candidate.model)
        if not space or candidate.hyperparameter_budget <= 1:
            return {}
        first_choice = {k: v[0] for k, v in space.items()}
        if not self.hyperparameter_search or X is None or y is None:
            return first_choice
        if isinstance(X, list):  # graph inputs are too costly to grid-search
            return first_choice
        combos = list(itertools.product(*space.values()))
        if len(combos) > 16:
            combos = combos[:16]
        folds = getattr(self, "_folds_cache", None) or []
        if not folds:
            return first_choice
        names = list(space.keys())
        best_params, best_score = first_choice, None
        for combo in combos:
            params = dict(zip(names, combo, strict=False))
            estimator = self._build_estimator(
                self.registry.get("model", candidate.model), candidate.model,
                n_features, params, y=(targets[0] if targets else y),
            )
            if targets:
                run_scores = [
                    _fold_primary_score(estimator, X, np.asarray(t).ravel(), folds, self.task_type)
                    for t in targets
                ]
                score = float(np.mean(run_scores))
            else:
                score = _fold_primary_score(
                    estimator, X, np.asarray(y).ravel(), folds, self.task_type
                )
            if best_score is None or score > best_score:
                best_score, best_params = score, params
        logger.info("grid search %s -> %s (score=%.4f)", candidate.model, best_params, best_score or 0.0)
        return best_params

    def _build_estimator(
        self, model_plugin: Any, model_name: str, n_features: int,
        hyperparams: dict[str, Any], y: Any | None = None,
    ) -> Any:
        from cta_qsar.models.registry import build_estimator, wrapped_estimator

        n_classes = None
        if self.task_type == "binary":
            n_classes = 2
        elif self.task_type == "multiclass" and y is not None:
            n_classes = int(len(np.unique(np.asarray(y).ravel())))
        if model_name in ("gcn", "gat", "mpnn"):
            return build_estimator(self.registry, model_name, self.task_type, n_classes=n_classes, hyperparams=hyperparams)
        return wrapped_estimator(self.registry, model_name, self.task_type, n_classes=n_classes, hyperparams=hyperparams)

    def _evaluate_trust(
        self, *, estimator: Any, rep_plugin: Any, X: Any, y: Any,
        split: Any, candidate: ExperimentCandidate, smiles: list[str],
    ) -> dict[str, Any]:
        trust_output: dict[str, Any] = {}
        folds = getattr(self, "_folds_cache", None) or []
        if not folds:
            trust_output["predictive"] = {"evaluated": False, "reason": "no folds"}
            return trust_output
        # predictive on the chosen split type
        predictive = self.registry.get_or_none("trust", "predictive")
        if predictive is not None:
            trust_output["predictive"] = predictive.evaluate(
                model=estimator, task_type=self.task_type, representation=rep_plugin,
                X=X, y=y, splits=folds, smiles=smiles,
            )
        else:
            trust_output["predictive"] = {"evaluated": False}

        # generalization with scaffold folds when the split is scaffold
        if candidate.validation == "scaffold":
            generalization = self.registry.get_or_none("trust", "generalization")
            if generalization is not None and folds:
                trust_output["generalization"] = generalization.evaluate(
                    model=estimator, task_type=self.task_type, representation=rep_plugin,
                    X=X, y=y, splits=folds, smiles=smiles,
                )
        elif candidate.validation == "random":
            generalization = self.registry.get_or_none("trust", "generalization")
            scaffold_folds = self._scaffold_folds(smiles, y)
            if generalization is not None and scaffold_folds:
                trust_output["generalization"] = generalization.evaluate(
                    model=estimator, task_type=self.task_type, representation=rep_plugin,
                    X=X, y=y, splits=scaffold_folds, smiles=smiles,
                )

        applicability = self.registry.get_or_none("trust", "applicability_domain")
        if applicability is not None and folds:
            trust_output["applicability_domain"] = applicability.evaluate(
                model=estimator, task_type=self.task_type, representation=rep_plugin,
                X=X, y=y, splits=folds, smiles=smiles,
            )
        robustness = self.registry.get_or_none("trust", "robustness")
        if robustness is not None and folds:
            trust_output["robustness"] = robustness.evaluate(
                model=estimator, task_type=self.task_type, representation=rep_plugin,
                X=X, y=y, splits=folds, smiles=smiles,
            )
        uncertainty = self.registry.get_or_none("trust", "uncertainty")
        if uncertainty is not None and folds:
            trust_output["uncertainty"] = uncertainty.evaluate(
                model=estimator, task_type=self.task_type, representation=rep_plugin,
                X=X, y=y, splits=folds, smiles=smiles,
            )
        explainability = self.registry.get_or_none("trust", "explainability")
        if explainability is not None and folds and hasattr(X, "shape") and X.shape[1] <= 3000:
            trust_output["explainability"] = explainability.evaluate(
                model=estimator, task_type=self.task_type, representation=rep_plugin,
                X=X, y=y, splits=folds, smiles=smiles,
            )
        return trust_output

    def _scaffold_folds(self, smiles: list[str], y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        from cta_qsar.validation.base import make_cv_folds
        from cta_qsar.validation.scaffold_split import scaffold_id

        groups = np.asarray([scaffold_id(s) or f"none-{i}" for i, s in enumerate(smiles)])
        return make_cv_folds(
            len(smiles), y, strategy="scaffold", n_splits=min(3, self.n_splits),
            n_repeats=1, random_seed=self.random_seed, test_fraction=self.test_fraction,
            groups=groups,
        )


def _model_space(registry: PluginRegistry, model_name: str) -> dict[str, list[Any]]:
    plugin = registry.get_or_none("model", model_name)
    if plugin is None:
        return {}
    return getattr(plugin, "hyperparameter_space", lambda: {})()


def _encode_labels(y: np.ndarray) -> np.ndarray:
    uniques, inverse = np.unique(np.asarray(y), return_inverse=True)
    return inverse.astype(int)


def _merge_multitask_trust(trust_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-target trust outputs: metric means are averaged, stds take
    the maximum (conservative), scalar flags come from the first output."""
    if not trust_list:
        return {}
    if len(trust_list) == 1:
        return trust_list[0]
    merged = dict(trust_list[0])
    for plugin_key in merged:
        if not all(isinstance(t.get(plugin_key), dict) for t in trust_list):
            continue
        first = merged[plugin_key]
        if not isinstance(first, dict) or not first:
            continue
        out: dict[str, Any] = {}
        for metric_key, metric_val in first.items():
            if isinstance(metric_val, dict) and "mean" in metric_val:
                vals = [
                    t[plugin_key][metric_key]
                    for t in trust_list
                    if isinstance(t.get(plugin_key, {}).get(metric_key), dict)
                    and "mean" in t[plugin_key][metric_key]
                    and "std" in t[plugin_key][metric_key]
                ]
                if not vals:
                    out[metric_key] = metric_val
                    continue
                means = [v["mean"] for v in vals]
                stds = [v["std"] for v in vals]
                out[metric_key] = {
                    "mean": float(np.mean(means)),
                    "std": float(np.max(stds)),
                }
            else:
                out[metric_key] = metric_val
        merged[plugin_key] = out
    return merged


def _extract_primary(trust: dict[str, Any]) -> dict[str, Any]:
    predictive = trust.get("predictive", {})
    out: dict[str, Any] = {}
    for key in ("rmse", "mae", "r2", "roc_auc", "pr_auc", "balanced_accuracy", "mcc", "f1"):
        if isinstance(predictive.get(key), dict) and "mean" in predictive[key]:
            out[key] = round(float(predictive[key]["mean"]), 4)
            out[f"{key}_std"] = round(float(predictive[key]["std"]), 4)
    return out


def _fold_primary_score(
    estimator: Any, X: Any, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], task_type: str
) -> float:
    """Mean primary metric across folds; higher is better (rmse is negated)."""
    from sklearn import clone

    from cta_qsar.trust.base import (
        aggregate_folds,
        classification_metrics,
        primary_metric,
        regression_metrics,
    )

    fold_metrics: list[dict[str, Any]] = []
    for train_idx, test_idx in folds:
        if len(test_idx) == 0:
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        try:
            fitted = clone(estimator).fit(X_tr, y_tr)
            pred = fitted.predict(X_te)
            if task_type == "regression":
                fold_metrics.append(regression_metrics(y_te, pred))
            else:
                proba = None
                if hasattr(fitted, "predict_proba"):
                    proba = fitted.predict_proba(X_te)
                    if task_type == "binary" and proba.ndim > 1:
                        proba = proba[:, 1]
                fold_metrics.append(classification_metrics(y_te, pred, proba))
        except Exception:  # noqa: BLE001
            continue
    if not fold_metrics:
        return float("-inf")
    agg = aggregate_folds(fold_metrics)
    key = primary_metric(task_type)
    score = agg.get(key, {}).get("mean")
    if score is None or (isinstance(score, float) and score != score):
        fallback = agg.get("balanced_accuracy", {}).get("mean") or 0.0
        return float(fallback)
    if task_type == "regression":
        return -float(score)
    return float(score)


def _mem_gb() -> float:
    """Approximate current process memory (very rough, avoids psutil)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
    except (AttributeError, ValueError):  # pragma: no cover
        return 0.0


def _df_of(y: np.ndarray) -> Any:
    import pandas as pd

    return pd.DataFrame({"y": y})