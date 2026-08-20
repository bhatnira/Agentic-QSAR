"""LLM provider factory + registry wiring.

Providers self-register through ``cta_qsar/llm/providers.py`` so adding a new
backend never requires editing this file: write a ``ReasoningModel`` subclass,
register a ``ProviderSpec`` (or expose a ``build`` on a plugin registered in
the ``llm_provider`` plugin kind), and ``build_llm`` picks it up by name.
"""

from __future__ import annotations

from typing import Any

from cta_qsar.core.logging import get_logger
from cta_qsar.core.registry import PluginRegistry
from cta_qsar.llm.base import ReasoningModel
from cta_qsar.llm.mock import HeuristicModel
from cta_qsar.llm.providers import discover_builtin_providers, get_provider_spec, provider_specs

logger = get_logger(__name__)


def build_llm(
    provider: str,
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    registry: PluginRegistry | None = None,
) -> ReasoningModel:
    """Construct a ReasoningModel for the provider name.

    Provider resolution order:
        1. the self-registered built-in providers (nvidia, openrouter,
           huggingface, mock + any module imported by discover_builtin_providers),
        2. a ``llm_provider`` plugin registered in the plugin registry,
        3. fallback to the deterministic heuristic model so the agent always
           remains runnable.
    """
    provider = (provider or "mock").lower()

    if provider == "mock":
        return HeuristicModel(model=model or "heuristic")

    discover_builtin_providers()
    spec = get_provider_spec(provider)
    if spec is not None:
        return spec.build(model=model, temperature=temperature, max_tokens=max_tokens)

    if registry is not None:
        plugin = registry.get_or_none("llm_provider", provider)
        if plugin is not None:
            construct = getattr(plugin, "build", None)
            if construct is not None:
                return construct(model=model, temperature=temperature, max_tokens=max_tokens)

    logger.warning("Unknown LLM provider %r; falling back to heuristic model", provider)
    return HeuristicModel(model=model or "heuristic")


def list_providers() -> list[dict[str, Any]]:
    """Describe every registered provider (name, env vars required, aliases)."""
    discover_builtin_providers()
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "requires_env": list(spec.requires_env),
            "aliases": list(spec.aliases),
        }
        for spec in provider_specs()
    ]


def describe(model: ReasoningModel) -> dict[str, Any]:
    return {"provider": model.provider_name, "model": model.model}