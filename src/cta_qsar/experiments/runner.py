"""Experiment executor: trains/evaluates a (representation, model, split)
triple and produces a complete scientific ExperimentRecord."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from cta_qsar.core.interfaces import ExperimentCandidate, ExperimentRecord
from cta_qsar.core.logging import get_logger
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.experiments.budget import BudgetState

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
    ) -> None:
        self.registry = registry
        self.task_type = task_type
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.test_fraction = test_fraction
        self.random_seed = random_seed
        self.dataset_hash = dataset_hash
        self.preprocessing_version = preprocessing_version

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
    ) -> ExperimentRecord:
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
        hyperparams = self._pick_hyperparams(candidate, n_features, budget)
        estimator = self._build_estimator(
            model_plugin, candidate.model, n_features, hyperparams
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

    def _pick_hyperparams(self, candidate: ExperimentCandidate, n_features: int, budget: BudgetState) -> dict[str, Any]:
        space = _model_space(self.registry, candidate.model)
        if not space or candidate.hyperparameter_budget <= 1:
            return {}
        # cheap mode: pick the first candidate combination deterministically
        choices = {k: v[0] for k, v in space.items()} if space else {}
        return choices

    def _build_estimator(self, model_plugin: Any, model_name: str, n_features: int, hyperparams: dict[str, Any]) -> Any:
        from cta_qsar.models.registry import build_estimator, wrapped_estimator

        n_classes = None
        if self.task_type in ("binary", "multiclass"):
            n_classes = 2 if self.task_type == "binary" else None
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


def _extract_primary(trust: dict[str, Any]) -> dict[str, Any]:
    predictive = trust.get("predictive", {})
    out: dict[str, Any] = {}
    for key in ("rmse", "mae", "r2", "roc_auc", "pr_auc", "balanced_accuracy", "mcc", "f1"):
        if isinstance(predictive.get(key), dict) and "mean" in predictive[key]:
            out[key] = round(float(predictive[key]["mean"]), 4)
            out[f"{key}_std"] = round(float(predictive[key]["std"]), 4)
    return out


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