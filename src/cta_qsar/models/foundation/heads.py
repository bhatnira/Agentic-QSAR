"""Foundation-model downstream models.

These plugins consume dense embedding vectors (e.g., from
:class:`cta_qsar.representations.embeddings.foundation.FoundationEmbeddings`)
with classical heads.  They require no GPU.
"""

from __future__ import annotations

from cta_qsar.core.interfaces import CostEstimate
from cta_qsar.models.base import MLPPlugin, RidgePlugin

ALLOWED_REPRESENTATIONS = ("foundation_embeddings", "mordred", "rdkit_descriptors")


class EmbeddingRidge(RidgePlugin):
    """Ridge head over embedding vectors (name: ``embedding_ridge``)."""

    name = "embedding_ridge"


class EmbeddingMLP(MLPPlugin):
    """MLP head over embedding vectors (name: ``embedding_mlp``)."""

    name = "embedding_mlp"

    def applicability(self, task_type: str, representation_name: str) -> tuple[bool, str]:
        if representation_name not in ALLOWED_REPRESENTATIONS:
            return False, f"{self.name} consumes only {ALLOWED_REPRESENTATIONS}"
        return True, "MLP head on precomputed embeddings"

    def estimate_cost(
        self, n_samples: int, n_features: int, representation_name: str
    ) -> CostEstimate:
        runtime = max(5.0, n_samples / 50.0)
        return CostEstimate(runtime_seconds=runtime, memory_gb=0.5)


PLUGINS = [EmbeddingRidge, EmbeddingMLP]