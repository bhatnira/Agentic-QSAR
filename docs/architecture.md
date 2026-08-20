# Architecture

CTA-QSAR is an autonomous QSAR scientist organized around a **plugin registry**
and a **cyclic LangGraph workflow**. The central research idea:

> The agent does not merely select a machine-learning algorithm. It autonomously
> discovers and evaluates *representation space + modeling paradigm + validation
> strategy + trustworthiness strategy*, then self-corrects from observed
> failures.

## Layered design

```
                    USER DATASET
                         |
                         v
                DATASET PROFILER
                         |
                         v
                 QSAR CASE AGENT
                         |
                         v
               STRATEGY PLANNER
                         |
              +----------+----------+
              |                     |
              v                     v
       REPRESENTATION          MODEL STRATEGY
          REGISTRY                 REGISTRY
              |                     |
              +----------+----------+
                         |
                         v
                 EXPERIMENT PLANNER
                         |
                         v
                  EXPERIMENT RUNNER
                         |
                         v
                  TRUST EVALUATOR
                         |
                         v
                 FAILURE DIAGNOSIS
                         |
                         v
                 SELF-CORRECTION
                         |
                         +----------+
                                    |
                                    v
                              NEXT EXPERIMENT
                                    |
                                    +----> loop
                                    |
                                    v
                              FINAL REPORT
```

## Modules

| Layer | Location | Responsibility |
|---|---|---|
| State & config | `core/state.py`, `core/config.py` | Typed LangGraph state, YAML+env configuration |
| Plugin contracts | `core/interfaces.py` | Protocols for every extensible capability |
| Registry | `core/registry.py` | Namespaced plugin discovery (`endpoint`, `representation`, `model`, `validation`, `trust`, `diagnosis`, `intervention`, …) |
| Orchestration | `orchestration/` | LangGraph graph, node functions, conditional routing, stop policies |
| Agents | `agents/` | Planner, diagnosis, profiler agents (single scientist + deterministic tools) |
| LLM | `llm/` | Abstract `ReasoningModel` + NVIDIA/OpenRouter/HuggingFace/mock providers |
| Chemistry | `chemistry/` | Standardization, SMILES validation, chemical-space stats |
| Endpoints | `endpoints/` | Regression/classification/multitask endpoint detection |
| Representations | `representations/` | Fingerprints, descriptors, graphs, embeddings as plugins |
| Models | `models/` | Classical, deep (GCN/GAT/MPNN), foundation, AutoML as plugins |
| Validation | `validation/` | Random, stratified, scaffold, cluster, temporal splits |
| Trust | `trust/` | Predictive, generalization, robustness, uncertainty, applicability, explainability |
| Experiments | `experiments/` | Candidates, planner (utility scoring), runner, budget, tracker |
| Diagnosis | `diagnosis/` | Failure rules, hypotheses, interventions |
| Memory | `memory/` | `experiments.jsonl` records, provenance |
| Reporting | `reporting/` | Markdown/JSON scientific reports |
| Hardware | `hardware/` | CPU/RAM/GPU/CUDA probe |
| CLI | `cli/main.py` | `cta-qsar` commands |

## The LangGraph workflow

The graph in `orchestration/graph.py` is stateful and cyclic. Nodes are plain
functions `(state: dict) -> dict`; conditional edges decide routing:

- `plan_experiment` → `execute_experiment` or `finalize_report`
- `execute_experiment` → `evaluate_performance` or back to `plan_experiment` (on failure)
- `evaluate_trust` → `diagnose_failure` or `finalize_report`
- `diagnose_failure` → `propose_intervention`
- `propose_intervention` → `plan_experiment` (loop) or `finalize_report`

The graph can therefore **return to planning after a failure**, which is the
defining loop of self-correction. Stopping is governed by `orchestration/policies.py`.

## Key design decisions

1. **The LLM never calculates.** All descriptor, fingerprint, training, metric,
   and statistical computation happens in deterministic Python tools; the LLM
   receives structured evidence and returns structured JSON decisions.
2. **Compute-aware planning.** Candidates are scored by
   `utility = (expected_improvement + information_gain + trust_gain) / compute_cost`;
   cheap experiments (Morgan + Ridge) are preferred unless science says otherwise.
3. **Graph-aware GNN handling.** Graph representations return `MolGraph` lists;
   GCN/GAT/MPNN estimators consume them directly, and trust plugins index
   graph inputs by position rather than numpy fancy-indexing.
4. **Memory prevents repetition.** `ExperimentRecord.signature` hashes
   dataset + preprocessing + representation + model + hyperparameters + split +
   seed; identical (including failed) experiments are never re-run.
