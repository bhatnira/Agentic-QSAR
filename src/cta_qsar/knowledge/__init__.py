"""Knowledge graph evidence layer for planner guidance and explanation.

Public API:
    - facts.EvidenceStore / Fact / dataset_class
    - static_builder.build_registry_facts / registry_version
    - ingestor.ingest_results_file / ingest_jsonl
    - curated_loader.load_curated_facts
    - explain.render_evidence_board / explain_decision / counterfactual_report
"""

from __future__ import annotations

from cta_qsar.knowledge.curated_loader import load_curated_facts
from cta_qsar.knowledge.facts import (
    MIN_EVIDENCE,
    WINDOW_SIZE,
    EvidenceStore,
    Fact,
    dataset_class,
)
from cta_qsar.knowledge.graph import KnowledgeGraph, render_graph_summary
from cta_qsar.knowledge.ingestor import ingest_jsonl, ingest_results_file
from cta_qsar.knowledge.static_builder import build_registry_facts, registry_version

__all__ = [
    "Fact",
    "EvidenceStore",
    "MIN_EVIDENCE",
    "WINDOW_SIZE",
    "dataset_class",
    "KnowledgeGraph",
    "render_graph_summary",
    "build_registry_facts",
    "registry_version",
    "ingest_results_file",
    "ingest_jsonl",
    "load_curated_facts",
]