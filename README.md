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

CSV, TSV, and Parquet are supported. `--smiles-column` and `--target-column`
are optional — automatic detection is attempted first.

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

## Scope

QSAR / molecular property prediction only: molecular structure, chemical
representations, ML, deep molecular learning, validation, uncertainty,
applicability domain, explainability, self-correction, computational
efficiency. No docking, AlphaFold, dynamics, or virtual screening.
