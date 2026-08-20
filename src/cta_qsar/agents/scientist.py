"""QSAR Scientist agent: the primary scientific reasoning agent.

Wraps the full LangGraph workflow as a programmatic API (used by the CLI and
tests), keeping the LangGraph dependency optional at import time for
lightweight components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cta_qsar.core.config import Config
from cta_qsar.core.logging import agent_log, configure_logging, get_logger
from cta_qsar.llm.base import ReasoningModel
from cta_qsar.llm.factory import build_llm

logger = get_logger(__name__)


class QSARScientist:
    """Programmatic entry point for a full autonomous QSAR run."""

    def __init__(
        self,
        config: Config,
        *,
        llm: ReasoningModel | None = None,
        registry: Any = None,
        output_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or build_llm(
            config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        if registry is None:
            from cta_qsar.core.registry import get_registry

            registry = get_registry()
            registry.auto_discover()
        self.registry = registry
        self.output_root = Path(output_root or config.reporting.get("output_dir", "runs"))

    def run(
        self,
        data_path: str | Path,
        *,
        smiles_column: str | None = None,
        target_column: str | None = None,
        max_experiments: int | None = None,
        max_minutes: float | None = None,
    ) -> dict[str, Any]:
        """Execute the full workflow; returns the final report dict."""
        configure_logging()
        from cta_qsar.orchestration.graph import build_graph

        smiles_column = smiles_column or self.config.dataset.smiles_column
        target_column = target_column or self.config.dataset.target_column
        if max_experiments:
            self.config.compute.max_experiments = max_experiments
        if max_minutes:
            self.config.compute.max_minutes = max_minutes

        agent_log(
            "start",
            "launching CTA-QSAR run",
            provider=self.llm.provider_name,
            model=self.llm.model,
            data=str(data_path),
        )
        graph_app = build_graph(
            config=self.config,
            llm=self.llm,
            registry=self.registry,
            output_root=str(self.output_root),
        )
        result = graph_app.invoke(
            {
                "data_path": str(data_path),
                "smiles_column": smiles_column or "",
                "target_column": target_column or "",
            }
        )
        final = result.get("final_report", {})
        agent_log(
            "finish",
            "run completed",
            run_id=result.get("run_id", "?"),
            experiments=len(result.get("experiments", [])),
        )
        return final

    def profile(
        self,
        data_path: str | Path,
        *,
        smiles_column: str | None = None,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Profile-only path (no experiments)."""
        configure_logging()
        from cta_qsar.orchestration.nodes import (
            assess_data_quality,
            characterize_chemical_space,
            detect_endpoint,
            ingest_dataset,
            profile_dataset_node,
        )

        state: dict[str, Any] = {
            "data_path": str(data_path),
            "smiles_column": smiles_column or self.config.dataset.smiles_column or "",
            "target_column": target_column or self.config.dataset.target_column or "",
            "config": self.config,
        }
        node = ingest_dataset(state)

        agent_log("profile", f"profiling {data_path}")
        node = profile_dataset_node(node)

        if node.get("target_column"):
            try:
                node = detect_endpoint(node)
                node = assess_data_quality(node)
                node = characterize_chemical_space(node)
            except Exception as exc:  # noqa: BLE001
                agent_log("profile", f"endpoint/quality checks skipped: {exc}")
        return node