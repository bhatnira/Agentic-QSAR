# Plugin development

CTA-QSAR is modular: a new QSAR capability is implemented as a plugin and
registered — the LangGraph orchestration engine unchanged.

## Plugin kinds

| Kind | Interface | Examples |
|---|---|---|
| `endpoint` | `EndpointPlugin` | regression, classification |
| `preprocessing` | `PreprocessingPlugin` | standardization steps |
| `representation` | `RepresentationPlugin` | morgan, maccs, graph, foundation_embeddings |
| `model` | `ModelPlugin` | ridge, random_forest, gcn, gat, mpnn, autogluon |
| `validation` | `ValidationPlugin` | random, stratified, scaffold, cluster, temporal |
| `trust` | `TrustPlugin` | predictive, generalization, robustness, uncertainty, applicability, explainability |
| `uncertainty` | `UncertaintyPlugin` | per-sample uncertainty |
| `explainability` | `ExplainabilityPlugin` | permutation/SHAP importance |
| `diagnosis` | `DiagnosisPlugin` | `failure_rules` |
| `intervention` | `InterventionPlugin` | `intervention_proposer` |
| `llm_provider` | `ReasoningModel` subclasses | nvidia, openrouter, huggingface |
| `reporting` | — | export.md / export.json |

## Contract examples

### Representation plugin

```python
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import RepresentPluginBase

class MyFingerprint(RepresentPluginBase):
    name = "my_fp"
    version = "1.0.0"

    def applicability(self, task_type, dataset_props):
        return True, "works everywhere"

    def estimate_cost(self, n_molecules):
        return CostEstimate(runtime_seconds=n_molecules * 0.01, memory_gb=0.2)

    def fit(self, smiles): ...      # learn vocab, return self
    def transform(self, smiles) -> np.ndarray: ...
    def metadata(self): ...
```

### Model plugin

```python
from cta_qsar.models.base import ModelPlugin

class MyModel(ModelPlugin):
    name = "my_model"
    supports = ("regression", "binary", "multiclass")

    def applicability(self, task_type, representation_name):
        return True, "compatible with matrix representations"

    def estimate_cost(self, n_samples, n_features, representation_name): ...
    def build_estimator(self, task_type, n_classes=None, **hyperparams): ...
    def hyperparameter_space(self): ...
```

### Registering a plugin

Every module that ships plugins defines `PLUGINS` (a list of instances or
zero-arg callables). Plugins are then registered either explicitly:

```python
from cta_qsar.core.registry import get_registry
registry = get_registry()
registry.register("model", MyModel())          # instance
registry.register_factory("model", "lazy_x", lambda: MyModel())  # lazy
```

or automatically by adding the module to the `_EXPLICIT` list in
`src/cta_qsar/core/registry.py` `auto_discover()`.

New plugins that follow an existing contract never require changes to
`orchestration/nodes.py`, `orchestration/graph.py`, `routing.py`, or
`policies.py`.

## Adding a GNN (worked example)

1. Write `src/cta_qsar/models/deep/mygnn.py` with:
   - a torch module, a scikit-learn-style estimator (`fit`/`predict`/
     `predict_proba`) that consumes `list[MolGraph]`,
   - a `MyGNNPlugin` with `name`, `supports`, `applicability` (requires
     `representation_name == "graph"` when torch is installed),
   - `PLUGINS = [MyGNNPlugin]`.
2. Add `("cta_qsar.models.deep.mygnn", "model")` to `auto_discover`.
3. Export the plugin from `models/deep/__init__.py`.
4. Add `mygnn` to `configs/*.yaml` `models.enabled`.

No other files change — trust plugins, the planner, and the runner already
handle graph inputs by position.

## Testing plugin additions

```bash
python -m pytest tests/unit/test_registry.py   # registry auto-discovery
python -m pytest tests/unit/test_planner.py    # candidate generation
python -m pytest tests/integration             # full graph runs
```