# Reproducibility

CTA-QSAR treats every run as a scientific record. Each run directory
(`runs/<Ymd-HMS>-<dataset>/`) contains complete provenance.

## What is recorded

Every run persists:

- **timestamp** and run id (`Ymd-HMS-dataset`)
- **dataset hash** — SHA-1 over canonical SMILES + target values
- **Python version**, **package versions** (via MLflow when enabled)
- **configuration** — the effective YAML config (`provenance.json`)
- **hardware profile** — CPU cores, RAM, GPU/CUDA availability, GPU memory
- **LLM provider and model** (never the key)
- **random seeds** — `experiment.random_seed` (default 42)
- **experiment results** — `experiments.jsonl`: dataset hash, preprocessing
  version, representation, model, hyperparameters, split, seed, metrics, trust
  metrics, runtime, memory, LLM decision, rationale, result, failure
  diagnoses, interventions
- **environment** — `environment.txt` (sensitive env vars stripped)
- **dependency lock** — `requirements.lock` pins the exact environment used
  for published benchmark results (`pip install -r requirements.lock`)

## Seeding

`experiment.random_seed` (default 42) threads through everything stochastic:
dataset splitting, repeated-CV fold generation, grid hyperparameter search,
tree-model internal sampling, and GNN weight init (`--seed` on the CLI).
Benchmark runs must use ≥3 seeds and report mean ± std.

## Hyperparameter searching

`experiment.hyperparameter_search` (default `false`; `--hyperparameter-search`
on the CLI) runs a budgeted grid search over each model plugin's
`hyperparameter_space` before the acceptance experiments, optimizing the same
folds and primary metric used for reporting (RMSE / ROC-AUC / MCC). With the
flag off, plugins use their first-space hyperparameters.

## Experiment signature (repeat prevention)

Each `ExperimentRecord.signature` is a SHA-1 over:

```
dataset hash + preprocessing version + representation + model
+ sorted hyperparameters + split + random seed
```

The planner excludes any signature already in memory. Identical *failed*
experiments are also never repeated — a failure is evidence, not a retry.

## Determinism

- All scikit-learn / XGBoost estimators are built with a fixed
  `experiment.random_seed` (default 42).
- GNN modules call `torch.manual_seed()` before training.
- Explainability uses `permutation_importance` with `random_state=42`.
- Robustness reuses fixed seeds (42, 1337, 2024).

Full bit-for-bit determinism is **not** guaranteed (threaded sklearn/XGBoost
and BLAS numerics vary across builds), so reports use
"best observed strategy under the specified budget" — never "global optimum".

## Re-running

```bash
cta-qsar run --data data.csv --budget 12 --max-minutes 30
cta-qsar report <RUN_ID>            # re-render the Markdown report
```

A tracker can also be rebuilt from persisted memory:

```python
from cta_qsar.experiments.tracker import ExperimentTracker
tracker = ExperimentTracker.load("runs/<RUN_ID>")
print(tracker.summary())
print(tracker.best())
```

## MLflow

When `tracking.enabled` and `backend=mlflow`, experiments and the full report
are logged with per-experiment params/metrics. `MLFLOW_TRACKING_URI` sets the
tracking server; the default is a local `./mlruns` store.

## Scientific honesty

The system never fabricates results, chemical interpretations, or literature
references. The LLM is instructed to cite the evidence it received and to use
budget-observed-best rather than globally-optimal phrasing.

## Planning transparency (knowledge & evidence layer)

Planner decisions are grounded in an attributed, read-only knowledge graph and
every decision is auditable (module `cta_qsar.knowledge`):

- **Evidence accumulation.** Every agent run's outcome (primary metric, chosen
  triple) is ingested into a windowed, append-only store — fine-grained cells
  keyed by dataset class (`task × size-bucket`), scenario, and strategy, plus
  coarse aggregates. Merges are idempotent per `run_id` and keep the most
  recent `WINDOW_SIZE` runs; storage grows only with evidence, never with
  runner history.
- **Fine-to-coarse fallback.** Retrieval returns the finest facts with at least
  `min_n` qualified runs, falling back to scenario-level and class-level
  aggregates otherwise — so a planner never over-trusts a two-run cell, but
  still benefits from transferable priors.
- **Grounding, not authority.** Evidence reallocates the *next evaluation*; it
  never vetoes one. The LLM prompt's `evidence_context` and the heuristic
  utility (evidence margin + winner-boost toward the best completed strategy)
  both remain advisory.
- **Audit trail.** Each planning round writes a machine-attributable trace to
  `run_dir/plan_trace.jsonl` (chosen candidate, evidence consulted with source
  and signed statistics, winner-boost/adjacency deltas), and the final report
  carries `planning_evidence` (dataset class, rendered evidence board, facts).
  Counterfactual drop-1 queries are available via
  `counterfactual_report(store, dataset_class, predicate)`.
- **Invariants for running experiments.** With no evidence file configured, or
  for a dataset class with no qualified facts, the planners behave exactly as
  the pure-heuristic baseline — the knowledge layer is a side-car and cannot
  change results when absent.