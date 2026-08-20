"""NVIDIA NIM / build.nvidia.com LLM provider (OpenAI-compatible endpoint).

Authenticates with an ``nvapi-...`` key stored in ``NVIDIA_API_KEY``.  The
catalog (build.nvidia.com) exposes open-weight models (Llama, Nemotron, Qwen,
DeepSeek, gpt-oss, etc.) through a chat-completions compatible API at
``https://integrate.api.nvidia.com/v1``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from cta_qsar.core.exceptions import LLMError, LLMOutputError
from cta_qsar.llm.base import (
    CaseClassification,
    DiagnosisOutput,
    InterventionOutput,
    LLMResponse,
    ReasoningModel,
    StopDecision,
    StrategySelection,
    prompt_for,
)
from cta_qsar.llm.providers import ProviderSpec, register_provider
from cta_qsar.llm.structured_output import parse_model, parse_models

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", os.getenv("LLM_MODEL", "nvidia/llama-3.3-70b-instruct"))


def build(
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> NVidiaModel:
    """Factory used by the provider registry; extra kwargs pass through."""
    return NVidiaModel(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)


register_provider(
    ProviderSpec(
        name="nvidia",
        build=build,
        requires_env=("NVIDIA_API_KEY", "NVIDIA_BASE_URL"),
        description="NVIDIA NIM / build.nvidia.com (OpenAI-compatible chat completions)",
        aliases=("nvapi", "nvidia-api"),
    )
)


class NVidiaModel(ReasoningModel):
    provider_name = "nvidia"

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL)
        self.timeout = timeout
        if not self.model:
            self.model = DEFAULT_MODEL

    # -- transport ---------------------------------------------------------
    def _chat(self, prompt: str, response_format: dict[str, Any] | None = None) -> str:
        if not self.api_key:
            raise LLMError(
                "NVIDIA_API_KEY is not set; set it or use LLM_PROVIDER=mock/heuristic"
            )
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"NVIDIA request failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(
                f"NVIDIA error {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError) as exc:
            raise LLMOutputError(f"unexpected NVIDIA payload: {data}") from exc

    def _call_json(self, prompt: str, schema: str = "json_object") -> LLMResponse:
        from cta_qsar.llm.structured_output import extract_json

        text = self._chat(prompt, response_format={"type": schema})
        try:
            structured = extract_json(text)
        except LLMOutputError:
            structured = {}
        return LLMResponse(
            content=text, structured=structured, raw=text,
            provider=self.provider_name, model=self.model,
        )

    # -- ReasoningModel contract -------------------------------------------
    def classify_case(self, case_context: dict[str, Any]) -> CaseClassification:
        text = self._chat(
            prompt_for("case"), response_format={"type": "json_object"}
        ).replace("{{CONTEXT}}", _dump(case_context))
        return parse_model(text, CaseClassification)

    def select_strategy(self, strategy_context: dict[str, Any]) -> StrategySelection:
        text = self._chat(
            prompt_for("strategy"), response_format={"type": "json_object"}
        ).replace("{{CONTEXT}}", _dump(strategy_context))
        return parse_model(text, StrategySelection)

    def diagnose(self, experiment_context: dict[str, Any]) -> list[DiagnosisOutput]:
        text = self._chat(
            prompt_for("diagnosis"), response_format={"type": "json_object"}
        ).replace("{{CONTEXT}}", _dump(experiment_context))
        return parse_models(text, DiagnosisOutput)

    def propose_intervention(self, diagnosis_context: dict[str, Any]) -> InterventionOutput:
        text = self._chat(
            prompt_for("intervention"), response_format={"type": "json_object"}
        ).replace("{{CONTEXT}}", _dump(diagnosis_context))
        return parse_model(text, InterventionOutput)

    def decide_stop(self, budget_context: dict[str, Any]) -> StopDecision:
        text = self._chat(
            prompt_for("stop"), response_format={"type": "json_object"}
        ).replace("{{CONTEXT}}", _dump(budget_context))
        return parse_model(text, StopDecision)

    def summarize(self, report_context: dict[str, Any]) -> str:
        return self._chat(prompt_for("report").replace("{{CONTEXT}}", _dump(report_context)))


def _dump(context: dict[str, Any]) -> str:
    return json.dumps(context, default=_json_default, sort_keys=False)


def _json_default(obj: Any) -> str:
    return str(obj)