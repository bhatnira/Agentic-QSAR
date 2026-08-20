# CTA-QSAR — Chemically Trustworthy Agentic QSAR

An autonomous scientific agent that receives an unseen QSAR dataset and decides
**how it should be modeled**. It profiles the data, detects the endpoint,
validates chemistry, selects representations and models by expected scientific
value per unit compute, evaluates trustworthiness, diagnoses failures, and
self-corrects — then writes a reproducible scientific report.

This is **not** AutoML. The agent autonomously discovers and evaluates the
combination of *representation space + modeling paradigm + validation strategy
+ trustworthiness strategy*, and corrects its course from observed evidence.

## Highlights

- **Modular plugin system** — endpoints, preprocessing, representations,
  models, validation, trust, uncertainty, explainability, diagnosis, and
  interventions are all plugins; the LangGraph orchestration engine never
  changes.
- **Cyclic, stateful LangGraph workflow** — the graph routes back to
  `plan_experiment` after a failure (self-correction loop), with conditional
  routing and explicit stopping rules.
- **CPU-first** — runs entirely without a GPU (RDKit, scikit-learn, XGBoost,
  CPU-capable PyTorch GNNs). GPU/foundation-model methods are optional plugins.
- **Replaceable LLM brain** — NVIDIA (default), OpenRouter, Hugging Face, or a
  deterministic mock/heuristic model. The LLM only reasons over structured
  evidence; all chemistry and statistics are deterministic Python tools.
- **Trustworthiness engine** — predictive performance, scaffold/generalization,
  seed robustness, applicability domain, uncertainty, explainability.
- **Scientific memory + provenance** — every experiment, decision, and
  rationale is persisted; identical experiments are never repeated.
- **MLflow tracking** and Markdown/JSON scientific reports.

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env   # add your NVIDIA_API_KEY (nvapi-...)
cta-qsar profile examples/regression.csv
cta-qsar run --data examples/regression.csv --budget 6
cta-qsar report 20260818-193040-regression
cta-qsar list-models
cta-qsar list-representations
cta-qsar list-validation
```

The end-to-end test run requires no GPU and no live API:

```bash
pytest
```

## LLM brain

The default provider is NVIDIA (NVIDIA NIM / build.nvidia.com, OpenAI-compatible
chat completions). Configure via `.env`:

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=openai/gpt-oss-20b
```

Switch providers without code changes:

```dotenv
LLM_PROVIDER=openrouter   LLM_MODEL=deepseek/deepseek-chat
LLM_PROVIDER=huggingface  LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_PROVIDER=mock         # deterministic heuristics, no API
```

## Command line

```bash
cta-qsar profile data.csv                      # inspect a dataset only
cta-qsar run --data data.csv --smiles-column SMILES \
             --target-column pIC50 --budget 30 # full autonomous workflow
cta-qsar report RUN_ID                         # re-render a report
cta-qsar list-models | list-representations | list-validation
```

Reproducible runs: pass a seed and (optionally) enable hyperparameter search:

```bash
cta-qsar run --data data.csv --seed 42 --hyperparameter-search \
             --budget 10
```

Each run snapshots the effective config in `provenance.json`, records
dependency versions in `environment.txt`, and stores per-experiment results as
JSONL.

## Benchmarks

Multi-dataset benchmark harness over six MoleculeNet benchmarks (ESOL,
FreeSolv, Lipophilicity, BACE, BBBP, ClinTox) with identical CV folds and
metric definitions across scenarios — static hyperparameter grid, autonomous
agent with heuristic planner, and optionally the agent with a real LLM:

```bash
python3 benchmarks/run_benchmark.py \
    --datasets esol,freesolv,bace,bbbp,clintox,lipophilicity \
    --seeds 0,1,2 --scenarios grid,agent-mock,agent-nosearch
```

Results land in `benchmarks/results/` (CSV + per-dataset summary JSON); raw
run artifacts live in `benchmarks/runs/`. `agent-nvidia` additionally requires
`NVIDIA_API_KEY`.

CSV, TSV, and Parquet are supported. `--smiles-column` and `--target-column`
are optional — automatic detection is attempted first.

## Knowledge & evidence layer

The pipeline's planners (heuristic and LLM) are guided by an optional,
read-only knowledge graph of attributed triples — dataset-class evidence,
registry capabilities, and curated chemistry priors:

- **Evidence facts** are accumulated from every benchmark/agent run
  (`results.csv`, `experiments.jsonl`) into a windowed, append-only store
  (`benchmarks/knowledge/evidence.jsonl`), keyed by dataset class
  (`task × size-bucket`) so knowledge transfers across similar datasets.
- **Retrieval** degrades from fine-grained cells (`class × scenario × triple`)
  to coarse aggregates when evidence is thin (`min_n` qualified runs), so
  planners always get the finest *trustworthy* prior.
- **Curated priors** (immuno/physchem heuristics with citable sources) and
  **registry facts** (plugin capabilities/requirements) are merged in.
- **Grounding** — the evidence board is templated into the planner prompt
  (`evidence_context`) and into the heuristic utility score; it never triggers
  evaluations, only reallocates the next one.
- **Explainability** — every decision is recorded as a machine-attributable
  trace (`run_dir/plan_trace.jsonl`) plus a rendered evidence board and
  drop-1 counterfactuals in the final report; the LLM is never the source of
  justification.

Enable with `knowledge.evidence_path` in the config (the benchmark harness
maintains the store automatically); when absent the planners fall back to
pure heuristics and behavior is unchanged.

## Project layout

```
src/cta_qsar/
    core/           state, config, interfaces, registry, exceptions, logging
    orchestration/  LangGraph graph, nodes, routing, policies
    agents/         scientist, planner, diagnosis, critic, profiler agents
    llm/            base, nvidia, openrouter, huggingface, mock, structured output
    chemistry/      standardization, validation, chemical space, fingerprints, descriptors
    endpoints/      base, detector, regression, classification, multitask
    representations/ fingerprints, descriptors, graph, embeddings (plugins)
    models/         classical, deep (GCN/GAT/MPNN), foundation, automl (plugins)
    validation/     random, stratified, scaffold, cluster, temporal (plugins)
    trust/          predictive, generalization, robustness, uncertainty,
                    applicability, explainability, chemical consistency (plugins)
    experiments/    candidate, planner, runner, budget, tracker
    knowledge/      facts store, static/curated/evidence builders, explain
    diagnosis/      failure rules, hypotheses, interventions
    memory/         experiment memory, provenance
    reporting/      report builder, export
    hardware/       CPU/RAM/GPU profiler
    cli/            cta-qsar entry point
```

## Documentation

- [Architecture](docs/architecture.md)
- [Workflow](docs/workflow.md)
- [Plugin development](docs/plugin_development.md)
- [LLM providers](docs/llm_providers.md)
- [Reproducibility](docs/reproducibility.md)

## Reproducibility

- `requirements.lock` pins the exact dependency set used for the benchmark
  results (install with `pip install -r requirements.lock`).
- Every run stores its effective config (`provenance.json`), dependency
  versions (`environment.txt`), and experiment records (`experiments.jsonl`).
- Seeds are fully configurable (`--seed` / `experiment.random_seed`) and thread
  through splitting, grid-search, and model init.
- If you use CTA-QSAR in a publication, please cite it — see `CITATION.cff`.

## Scope

QSAR / molecular property prediction only: molecular structure, chemical
representations, ML, deep molecular learning, validation, uncertainty,
applicability domain, explainability, self-correction, computational
efficiency. No docking, AlphaFold, dynamics, or virtual screening.
