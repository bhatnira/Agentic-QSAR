"""Per-run execution context (config, registry, LLM).

These objects must not travel through the LangGraph state (serialization +
schema constraints), so they are attached to the current run via contextvars.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

ctx_var: contextvars.ContextVar[ExecutionContext | None] = contextvars.ContextVar(
    "cta_qsar_context", default=None
)


@dataclass
class ExecutionContext:
    config: Any = None
    registry: Any = None
    llm: Any = None
    output_root: str = "runs"
    extra: dict[str, Any] = field(default_factory=dict)


def set_context(ctx: ExecutionContext) -> None:
    ctx_var.set(ctx)


def get_context() -> ExecutionContext:
    ctx = ctx_var.get()
    if ctx is None:
        raise RuntimeError("no ExecutionContext set; run inside build_graph.invoke()")
    return ctx


def get_or_none() -> ExecutionContext | None:
    return ctx_var.get()