# LLM providers

The scientist reasons through an abstract `ReasoningModel` interface
(`src/cta_qsar/llm/base.py`) with six capabilities:

- `classify_case` — understand the QSAR problem from profiled evidence
- `select_strategy` — choose the next (representation, model, validation)
  experiment, given ranked candidates
- `diagnose` — interpret trust evidence into failure hypotheses
- `propose_intervention` — pick the highest-value corrective action
- `decide_stop` — decide whether another experiment is worthwhile
- `summarize` — write the scientific executive summary

The LLM **only reasons over structured scientific information** produced by
deterministic Python tools. It never runs RDKit, never trains models, and never
calculates metrics — those are deterministic, tested code paths.

## Provider selection

Set `LLM_PROVIDER` in `.env` or via `--llm-provider`:

| Provider | Env vars | Notes |
|---|---|---|
| `nvidia` (default) | `NVIDIA_API_KEY` (`nvapi-…`), `NVIDIA_MODEL`, `NVIDIA_BASE_URL` | https://integrate.api.nvidia.com/v1, OpenAI-compatible |
| `openrouter` | `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `LLM_MODEL` | large open-model catalogue |
| `huggingface` | `HF_TOKEN`, `HF_MODEL`, `HF_ENDPOINT` | serverless Inference API |
| `mock` | — | deterministic `HeuristicModel`, no network, ideal for tests |

Unknown providers fall back to the deterministic heuristic model so a run never
blocks on a missing key.

## NVIDIA (default)

NVIDIA NIM / build.nvidia.com is OpenAI-compatible. Keys start with `nvapi-`
and are created at <https://build.nvidia.com>. The provider

- is selected when `LLM_PROVIDER=nvidia` (default in `configs/default.yaml`);
- reads the key from `NVIDIA_API_KEY`;
- defaults the model to `openai/gpt-oss-20b` (fast, verifiable on CPU-first
  workflows) or whatever `NVIDIA_MODEL` / `LLM_MODEL` specifies;
- supports `response_format={"type":"json_object"}` for structured reasoning.

The catalog backing your key can be listed programmatically:

```bash
curl https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY"
```

Models that are excellent QSAR reasoning choices include `openai/gpt-oss-20b`,
`openai/gpt-oss-120b`, and the NVIDIA Nemotron series.

## Structured output contract

Every prompt (`llm/base.py::PROMPTS`) demands JSON. `structured_output.py`
extracts and validates JSON against a pydantic schema, and leniently coerces
common LLM near-misses (list-vs-string, comma-separated lists, numeric strings,
bool strings) so a slightly loose model still yields valid decisions.

## Security

Keys are read from environment variables / `.env` (git-ignored). Never commit
credentials. Run provenance strips `OPENROUTER*`, `HF_*`, `NVIDIA*`, and
`TOKEN*` environment entries before persisting `environment.txt`.

## Adding a new LLM provider (pluggable)

Providers self-register; you never edit the factory. The registry lives in
`src/cta_qsar/llm/providers.py`, and `build_llm()` / the `list-providers` CLI
command read from it. To add a backend:

1. Create `src/cta_qsar/llm/yourprovider.py`.
2. Subclass `ReasoningModel` and implement the six capabilities
   (`classify_case`, `select_strategy`, `diagnose`, `propose_intervention`,
   `decide_stop`, `summarize`); reuse `cta_qsar.llm.base.PROMPTS` and
   `cta_qsar.llm.structured_output` for JSON parsing.
3. Add a zero-arg-friendly factory plus a self-registration call:

```python
from cta_qsar.llm.base import ReasoningModel
from cta_qsar.llm.providers import ProviderSpec, register_provider

class MyProvider(ReasoningModel):
    provider_name = "myprovider"
    # ... implement the six capabilities ...

def build(model="", temperature=0.1, max_tokens=4096, **kwargs):
    return MyProvider(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)

register_provider(ProviderSpec(
    name="myprovider",                 # LLM_PROVIDER value
    build=build,
    requires_env=("MY_API_KEY",),      # shown by `cta-qsar list-providers`
    description="Describe your endpoint",
    aliases=("myp",),                  # optional extra names
))
```

4. Use it: `LLM_PROVIDER=myprovider` (or `cta-qsar run --llm-provider myprovider`).

Providers in `_BUILTIN_MODULES` are discovered automatically; a provider in a
third-party package can be registered by calling `register_provider(...)` at
import time (for example from a `llm_provider` plugin in the plugin registry,
whose `build` method is also honored by `build_llm`).

Discover available backends at any time:

```bash
cta-qsar list-providers
```