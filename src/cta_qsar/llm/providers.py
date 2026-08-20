"""LLM provider registry — the pluggable point for new LLM backends.

Adding a new provider requires writing exactly ONE module:

    1. subclass ``ReasoningModel`` (see ``cta_qsar/llm/base.py``),
    2. define a ``build(model, temperature, max_tokens)`` factory,
    3. self-register with :func:`register_provider`.

The factory, the CLI, the orchestration engine, and the config never change.
Providers are resolved by name (case-insensitive, aliases allowed); unknown
names fall back to the deterministic heuristic model so a run never blocks.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ModelFactory = Callable[..., Any]

_PROVIDERS: dict[str, ProviderSpec] = {}
_ALIASES: dict[str, str] = {}

_BUILTIN_MODULES = (
    "cta_qsar.llm.nvidia",
    "cta_qsar.llm.openrouter",
    "cta_qsar.llm.huggingface",
    "cta_qsar.llm.mock",
)


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata + factory for one LLM provider backend."""

    name: str
    build: ModelFactory
    requires_env: tuple[str, ...] = ()
    description: str = ""
    aliases: tuple[str, ...] = ()


def register_provider(spec: ProviderSpec) -> None:
    """Register a provider so ``build_llm`` can construct it by name."""
    key = spec.name.lower().strip()
    if not key:
        raise ValueError("provider name must not be empty")
    if key in _PROVIDERS:
        raise ValueError(f"LLM provider {spec.name!r} is already registered")
    _PROVIDERS[key] = spec
    for alias in spec.aliases:
        _ALIASES[alias.lower().strip()] = key


def unregister_provider(name: str) -> ProviderSpec | None:
    """Remove a provider (mainly for tests); returns the removed spec."""
    spec = _PROVIDERS.pop(name.lower().strip(), None)
    if spec is not None:
        for alias, target in list(_ALIASES.items()):
            if target == spec.name.lower():
                del _ALIASES[alias]
    return spec


def get_provider_spec(name: str) -> ProviderSpec | None:
    """Resolve a provider spec by name or alias (case-insensitive)."""
    key = name.lower().strip()
    if key in _ALIASES:
        key = _ALIASES[key]
    return _PROVIDERS.get(key)


def provider_specs() -> list[ProviderSpec]:
    """All registered providers, sorted by name (after built-in discovery)."""
    discover_builtin_providers()
    return sorted(_PROVIDERS.values(), key=lambda spec: spec.name)


def discover_builtin_providers() -> None:
    """Import the shipped provider modules so they self-register.

    Import errors are swallowed: a provider with missing optional
    dependencies simply does not appear in the registry.
    """
    for module_name in _BUILTIN_MODULES:
        with contextlib.suppress(ImportError):  # optional dependency missing
            importlib.import_module(module_name)


def build_provider(name: str, **kwargs: Any) -> Any:
    """Construct a provider instance from the registry (no fallback)."""
    spec = get_provider_spec(name)
    if spec is None:
        raise KeyError(f"no LLM provider registered under {name!r}")
    return spec.build(**kwargs)