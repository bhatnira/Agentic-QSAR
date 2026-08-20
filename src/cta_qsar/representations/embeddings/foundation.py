"""Foundation-model SMILES embeddings (optional, abstraction only).

CTA-QSAR never *requires* a GPU or pretrained weights.  This module defines
the plugin contract and a concrete transformer-based plugin that activates
only when a local/token-based model is available.  The planner treats these
embeddings as an expensive, high-value representation when enabled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cta_qsar.core.exceptions import PluginUnavailableError
from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.representations.base import EmbeddingRepresentation


class FoundationEmbeddings(EmbeddingRepresentation):
    name = "foundation_embeddings"
    version = "1.0.0"

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name=model_name)
        self.model_name = model_name or ""
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    # -- availability ------------------------------------------------------
    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def applicability(self, task_type: str, dataset_props: dict[str, Any]) -> tuple[bool, str]:
        if not self.is_available():
            return False, "transformers not installed"
        if not self.model_name:
            return False, "no model configured (set representations.foundation_model)"
        return True, "pretrained transformer SMILES embeddings"

    def estimate_cost(self, n_molecules: int) -> CostEstimate:
        return CostEstimate(
            runtime_seconds=30.0 + n_molecules * 0.2,
            memory_gb=4.0,
            gpu_required=False,
        )

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        if not self.is_available():
            raise PluginUnavailableError(
                "transformers is not installed; embeddings unavailable"
            )
        if not self.model_name:
            raise PluginUnavailableError("foundation_embeddings.model_name not configured")
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    # -- transform ---------------------------------------------------------
    def fit(self, smiles: list[str]) -> FoundationEmbeddings:
        self._load()
        return self

    def transform(self, smiles: list[str]) -> np.ndarray:
        import torch

        if self._model is None:
            self.fit(smiles)
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for s in smiles:
                encoded = self._tokenizer(
                    f"{s}<eos>", return_tensors="pt", truncation=True, max_length=128
                )
                out = self._model(**encoded)
                emb = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                vectors.append(emb)
        return np.vstack(vectors)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "available": self.is_available(),
            "model": self.model_name,
        }

PLUGINS = [FoundationEmbeddings]
