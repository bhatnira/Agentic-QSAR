"""Unit tests: LLM provider switching."""

from __future__ import annotations

import pytest

from cta_qsar.llm.factory import build_llm, list_providers
from cta_qsar.llm.huggingface import HuggingFaceModel
from cta_qsar.llm.mock import HeuristicModel
from cta_qsar.llm.nvidia import NVidiaModel
from cta_qsar.llm.openrouter import OpenRouterModel
from cta_qsar.llm.providers import (
    ProviderSpec,
    get_provider_spec,
    register_provider,
    unregister_provider,
)
from cta_qsar.llm.structured_output import extract_json, parse_model


def test_mock_provider() -> None:
    model = build_llm("mock")
    assert isinstance(model, HeuristicModel)


def test_openrouter_switching() -> None:
    model = build_llm("openrouter", model="qwen/qwen2.5-72b-instruct")
    assert isinstance(model, OpenRouterModel)
    assert model.model == "qwen/qwen2.5-72b-instruct"


def test_huggingface_switching() -> None:
    model = build_llm("huggingface", model="HuggingFaceH4/zephyr-7b-beta")
    assert isinstance(model, HuggingFaceModel)
    assert model.model == "HuggingFaceH4/zephyr-7b-beta"


def test_unknown_provider_falls_back_to_heuristic() -> None:
    model = build_llm("does-not-exist")
    assert isinstance(model, HeuristicModel)


def test_provider_is_configurable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "deepseek/deepseek-chat")
    from cta_qsar.core.config import Config

    config = Config.load()
    assert config.llm.provider == "openrouter"
    assert config.llm.model == "deepseek/deepseek-chat"
    model = build_llm(config.llm.provider, model=config.llm.model)
    assert isinstance(model, OpenRouterModel)


def test_mock_model_decisions_are_deterministic() -> None:
    model = HeuristicModel()
    context = {"endpoint": {"task_type": "regression"}, "quality_report": {}}
    classification = model.classify_case(context)
    assert classification.task_type == "regression"
    assert "morgan" in classification.representation_strategy
    repo_class = model.classify_case(context)
    assert repo_class.model_dump() == classification.model_dump()


def test_nvidia_switching() -> None:
    model = build_llm("nvidia", model="NVIDIA/Autogen-40")
    assert isinstance(model, NVidiaModel)
    assert model.model == "NVIDIA/Autogen-40"
    model = NVidiaModel(model="NVIDIA/Autogen-40", api_key="nvapi-test")
    assert model.api_key == "nvapi-test"


def test_nvidia_requires_key_for_chat() -> None:
    import pytest

    from cta_qsar.core.exceptions import LLMError

    model = NVidiaModel(model="NVIDIA/Autogen-40", api_key="")
    with pytest.raises(LLMError):
        model._chat("hello")


def test_default_config_uses_nvidia_provider() -> None:
    from cta_qsar.core.config import Config

    config = Config.load()
    assert config.llm.provider == "nvidia"


def test_structured_output_parsing() -> None:
    from cta_qsar.llm.base import CaseClassification

    text = '```json\n{"problem_statement": "predict pIC50", "task_type": "regression", "risks": ["leakage"], "validation_strategy": ["scaffold"], "representation_strategy": ["morgan"], "model_strategy": ["ridge"], "reasoning": "small dataset"}\n```'
    parsed = parse_model(text, CaseClassification)
    assert parsed.task_type == "regression"
    assert parsed.representation_strategy == ["morgan"]
    assert extract_json(text)["task_type"] == "regression"


def test_optional_dict_none_string_coerced() -> None:
    """LLM emitting 'None' as a string must map to a real null."""
    from cta_qsar.llm.base import StopDecision

    parsed = parse_model(
        '{"should_stop": false, "reason": "continue", "next_candidate": "None"}',
        StopDecision,
    )
    assert parsed.should_stop is False
    assert parsed.next_candidate is None

    parsed = parse_model(
        '{"should_stop": true, "reason": "done", "next_candidate": null}',
        StopDecision,
    )
    assert parsed.next_candidate is None


def test_dict_list_values_are_preserved_and_string_items_dropped() -> None:
    """A well-formed list-of-dicts must survive coercion untouched;
    an all-string plan falls back to an empty plan instead of crashing."""
    from cta_qsar.llm.base import StrategySelection

    good = parse_model(
        '{"experiment_plan": [{"representation": "morgan", "model": "ridge", '
        '"validation": "random", "hyperparameter_budget": 1, "reason": "base"}], '
        '"rationale": "x", "evidence_cited": []}',
        StrategySelection,
    )
    assert good.experiment_plan[0]["representation"] == "morgan"

    fallback = parse_model(
        '{"experiment_plan": ["baseline"], "rationale": "echo", "evidence_cited": []}',
        StrategySelection,
    )
    assert fallback.experiment_plan == []


def test_bad_shaped_fields_dropped_instead_of_crashing() -> None:
    """If a field cannot be salvaged, parse_model drops it (defaults apply)
    rather than raising."""
    from cta_qsar.llm.base import CaseClassification

    parsed = parse_model(
        '{"problem_statement": "p", "task_type": 123, "risks": {"x": 1}, '
        '"validation_strategy": null, "representation_strategy": "a, b", '
        '"model_strategy": [], "reasoning": "r"}',
        CaseClassification,
    )
    assert parsed.task_type == "123"  # numeric task type stringified
    assert parsed.risks == []
    assert parsed.validation_strategy == []
    assert parsed.representation_strategy == ["a", "b"]


def test_builtin_providers_are_registered_and_listable() -> None:
    names = {p["name"] for p in list_providers()}
    assert {"mock", "nvidia", "openrouter", "huggingface"} <= names
    for name in names:
        assert get_provider_spec(name) is not None


def test_provider_alias_resolves() -> None:
    spec = get_provider_spec("nvapi")
    assert spec is not None and spec.name == "nvidia"
    assert get_provider_spec("Heuristic").name == "mock"


def test_build_llm_uses_registry_provider() -> None:
    """A provider added at runtime is picked up without touching the factory."""

    class LocalBrain:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.called = kwargs

        _provider_calls = 0

    def _build(**kwargs):
        LocalBrain._provider_calls += 1
        return LocalBrain(**kwargs)

    register_provider(
        ProviderSpec(
            name="local-brain",
            build=_build,
            aliases=("local",),
            description="test-only provider",
        )
    )
    try:
        assert isinstance(build_llm("local-brain", model="tiny"), LocalBrain)
        assert LocalBrain._provider_calls == 1
        # alias path
        assert isinstance(build_llm("Local", model="tiny"), LocalBrain)
    finally:
        assert unregister_provider("local-brain") is not None
    # fully removed: falls back to heuristic again
    assert isinstance(build_llm("local-brain"), HeuristicModel)


def test_duplicate_provider_registration_rejected() -> None:
    with pytest.raises(ValueError):
        register_provider(ProviderSpec(name="nvidia", build=lambda **kwargs: None))