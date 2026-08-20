# Workflow

CTA-QSAR executes a 16-step cyclic workflow. The graph nodes in
`src/cta_qsar/orchestration/nodes.py` implement each step.

## The 16 steps

1. **ingest_dataset** — Load CSV / TSV / Parquet (missing files and unsupported
   formats raise clear errors).
2. **profile_dataset** — Auto-detect the SMILES column and the target column;
   report rows, columns, dtypes, missing values, duplicates, candidate columns.
3. **detect_endpoint** — Infer task type (regression / binary / multiclass /
   multitask) from values, datatype, uniqueness, and column name. Ambiguity is
   reported with a confidence score rather than silently assumed.
4. **standardize_dataset** — RDKit standardization (canonical SMILES, desalting,
   neutralization) storing `original_smiles`, `standardized_smiles`, and a
   transformation log. Original SMILES are never destroyed.
5. **assess_data_quality** — Invalid SMILES, duplicate rows/molecules,
   conflicting labels, class balance, target distribution, outlier flags —
   outliers are flagged, never auto-removed.
6. **characterize_chemical_space** — Unique molecules, diversity, and
   nearest-neighbor similarity statistics.
7. **select_validation** — Choose applicable validation plugins (random,
   stratified, scaffold, cluster, temporal) given task type and data.
8. **generate_candidate_representations** — List enabled representation plugins
   (Morgan, RDKit/MACCS/AtomPair/Torsion fingerprints, RDKit/Mordred
   descriptors, graph, foundation embeddings).
9. **generate_candidate_models** — List enabled model plugins (Ridge, ElasticNet,
   RandomForest, ExtraTrees, SVR, XGBoost, LightGBM, MLP, GCN, GAT, MPNN,
   AutoGluon if installed).
10. **plan_experiment** — Score every feasible (representation, model, validation)
    triple by
    `utility = (expected_improvement + information_gain + trust_gain) / compute_cost`,
    rank them, and let the LLM re-rank the top candidates when configured.
11. **execute_experiment** — Run the chosen candidate: featurize, split, train,
    evaluate, and record a complete `ExperimentRecord`. Failures are recorded in
    scientific memory so they are never repeated.
12. **evaluate_performance** — Extract primary metrics (RMSE/R² for regression;
    ROC-AUC/MCC/balanced accuracy for classification).
13. **evaluate_trust** — Predictive, scaffold/generalization, robustness,
    applicability domain, uncertainty, and explainability verdicts.
14. **diagnose_failure** — Rule-based diagnosis of trust gaps (random-strong /
    scaffold-weak ⇒ chemical-series dependence; train-strong / validation-weak ⇒
    overfitting; good accuracy / poor PR-AUC ⇒ imbalance; good performance / poor
    OOD ⇒ narrow applicability domain).
15. **propose_intervention** — Rank interventions by
    `(expected_improvement + expected_trust_gain) / compute_cost`.
16. **decide_next_action / finalize_report** — Decide whether another experiment
    is worthwhile; when stopping, write the final scientific report.

## Self-correction loop

```
plan_experiment ──► execute_experiment ──► evaluate_performance
      ▲                                        │
      │                                        v
      │                                   evaluate_trust
      │                                        │
      │                                        v
      │                                   diagnose_failure
      │                                        │
      │                                        v
      └────────── propose_intervention ◄───────┘
```

After `propose_intervention`, routing sends the graph back to `plan_experiment`
(or to `finalize_report` if a stopping rule has fired).

## Stopping rules

The agent stops when any of these hold (reported explicitly in the final
report):

1. compute budget exhausted (`compute.max_minutes`)
2. experiment budget exhausted (`compute.max_experiments`)
3. no feasible candidate remaining (utility ≤ 0)
4. no unexplored applicable strategies remain
5. expected improvement / compute cost drops below threshold (LLM stop decision)

## First vertical slice

The initial complete working slice is:

CSV → dataset profiling → endpoint detection → RDKit validation → Morgan
fingerprints → RDKit descriptors → Ridge → RandomForest → XGBoost → random split
→ scaffold split → trust evaluation → failure diagnosis → next experiment →
final report.

Only after the slice works are dependency-heavy options enabled: Mordred, GNNs,
foundation embeddings, uncertainty, SHAP, additional validation, and advanced
planning.