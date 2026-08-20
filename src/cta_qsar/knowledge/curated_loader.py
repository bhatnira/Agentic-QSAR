"""Load curated chemistry priors from the bundled YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from cta_qsar.knowledge.facts import Fact

CURATED_PATH = Path(__file__).parent / "curated.yaml"


def load_curated_facts(path: str | Path | None = None) -> list[Fact]:
    """Parse the curated YAML into Facts. Missing sources are rejected."""
    path = Path(path) if path else CURATED_PATH
    with path.open() as fh:
        docs = yaml.safe_load(fh) or []
    facts: list[Fact] = []
    for row in docs:
        subject = row.get("subject", "")
        predicate = row.get("predicate", "")
        object_ = row.get("object", "")
        source = row.get("source", "")
        if not (subject and predicate and object_):
            raise ValueError(f"curated fact missing subject/predicate/object: {row!r}")
        facts.append(
            Fact(subject=str(subject), predicate=str(predicate), object=str(object_), level=3, source=str(source))
        )
    return facts