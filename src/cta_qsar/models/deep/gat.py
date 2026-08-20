"""A compact, CPU-friendly Graph Attention Network.

Implements a scikit-learn compatible estimator over :class:`MolGraph` inputs
(identical contract to the GCN plugin) with a single-head additive-attention
layer.  Torch is optional: the plugin only becomes available when torch is
installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class _GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.attn = nn.Parameter(torch.zeros(2 * out_dim, 1))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)  # (batch, n, out)
        bsz, n, out = h.shape
        left = h.unsqueeze(2).expand(bsz, n, n, out)  # (b, i, j, d) rows
        right = h.unsqueeze(1).expand(bsz, n, n, out)  # (b, i, j, d) cols
        scores = torch.cat([left, right], dim=-1) @ self.attn  # (b, n, n, 1)
        scores = F.leaky_relu(scores.squeeze(-1), negative_slope=0.2)
        mask = adj > 0
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out_h = torch.bmm(attn, h)
        return F.elu(out_h)


class _GATModule(nn.Module):
    def __init__(
        self, hidden: int = 64, n_out: int = 1, n_layers: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = 6  # matches cta_qsar.representations.graph.featurizer.N_FEATURES
        for _ in range(n_layers):
            self.layers.append(_GATLayer(in_dim, hidden, dropout=dropout))
            in_dim = hidden
        self.head = nn.Linear(hidden, n_out)

    def forward(self, graphs: list[Any]) -> torch.Tensor:
        outs = []
        for graph in graphs:
            x, adj = graph.to_torch()
            for layer in self.layers:
                x = layer(x.unsqueeze(0), adj.unsqueeze(0)).squeeze(0)
            outs.append(self.head(x.mean(dim=0)).squeeze())
        return torch.stack(outs)


class TorchGAT:
    """sklearn-style estimator (`fit`/`predict`) wrapping the GAT module."""

    def __init__(
        self,
        task_type: str = "regression",
        n_classes: int | None = None,
        hidden: int = 64,
        n_layers: int = 2,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        if not _TORCH_OK:
            raise PluginUnavailableError(
                "torch is not installed; the GAT plugin is unavailable"
            )
        self.task_type = task_type
        self.n_classes = n_classes
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._module: _GATModule | None = None
        self._fitted = False

    def fit(self, X: list[Any], y: np.ndarray) -> TorchGAT:
        if self._module is None:
            n_out = self.n_classes if self.task_type in ("binary", "multiclass") else 1
            self._module = _GATModule(hidden=self.hidden, n_out=n_out, n_layers=self.n_layers)
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self._module.parameters(), lr=self.lr)
        yt = torch.tensor(np.asarray(y, dtype=np.float32))
        if self.task_type == "binary":
            yt = yt if yt.dim() == 1 else yt[:, 0]
        n = len(X)
        for _ in range(self.epochs):
            self._module.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size].tolist()
                batch = [X[i] for i in idx]
                out = self._module(batch)
                target = yt[idx]
                if self.task_type == "binary":
                    loss = F.binary_cross_entropy_with_logits(out, target)
                elif self.task_type == "multiclass":
                    loss = F.cross_entropy(out, target.to(torch.long))
                else:
                    loss = F.mse_loss(out, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.detach()) * len(batch)
            if total_loss / n < 1e-5 and self.task_type == "regression":
                break
        self._fitted = True
        return self

    def predict(self, X: list[Any]) -> np.ndarray:
        if not self._fitted or self._module is None:
            raise RuntimeError("TorchGAT.predict called before fit")
        self._module.eval()
        with torch.no_grad():
            out = self._module(X)
        if self.task_type == "binary":
            return (torch.sigmoid(out) > 0.5).numpy().astype(int)
        if self.task_type == "multiclass":
            return out.argmax(dim=1).numpy()
        return out.numpy()

    def predict_proba(self, X: list[Any]) -> np.ndarray:
        if not self._fitted or self._module is None:
            raise RuntimeError("TorchGAT.predict_proba called before fit")
        self._module.eval()
        with torch.no_grad():
            out = torch.sigmoid(self._module(X)) if self.task_type == "binary" else F.softmax(
                self._module(X), dim=1
            )
        return out.numpy()


class GATPlugin:
    name = "gat"
    version = "1.0.0"
    supports = ("regression", "binary", "multiclass")

    def applicability(self, task_type: str, representation_name: str) -> tuple[bool, str]:
        if not _TORCH_OK:
            return False, "torch not installed"
        if task_type not in self.supports:
            return False, f"task type {task_type} not supported"
        if representation_name != "graph":
            return False, "GAT requires the graph representation"
        return True, "CPU-capable graph attention network over molecular graphs"

    def estimate_cost(
        self, n_samples: int, n_features: int, representation_name: str
    ) -> CostEstimate:
        runtime = n_samples / 25.0 + 50.0
        return CostEstimate(runtime_seconds=runtime, memory_gb=max(0.3, n_samples * 5000 * 4e-9))

    def build_estimator(
        self, task_type: str, n_classes: int | None = None, **hyperparams: Any
    ) -> TorchGAT:
        hp = {"task_type": task_type, "n_classes": n_classes, **hyperparams}
        return TorchGAT(**hp)

    def hyperparameter_space(self) -> dict[str, list[Any]]:
        return {"hidden": [32, 64], "epochs": [40, 80]}

PLUGINS = [GATPlugin]
