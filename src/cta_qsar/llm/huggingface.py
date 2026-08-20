"""Hugging Face LLM provider (serverless Inference API, HF_TOKEN)."""

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

DEFAULT_ENDPOINT = "https://api-inference.huggingface.co/models/{model}"


def build(
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> HuggingFaceModel:
    """Factory used by the provider registry; extra kwargs pass through."""
    return HuggingFaceModel(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)


register_provider(
    ProviderSpec(
        name="huggingface",
        build=build,
        requires_env=("HF_TOKEN", "HF_ENDPOINT"),
        description="Hugging Face Inference API (serverless)",
    )
)


class HuggingFaceModel(ReasoningModel):
    provider_name = "huggingface"

    def __init__(
        self,
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        token: str | None = None,
        endpoint: str | None = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.token = token or os.getenv("HF_TOKEN", "")
        self.endpoint = endpoint or os.getenv(
            "HF_ENDPOINT", DEFAULT_ENDPOINT.format(model=self.model or "Qwen/Qwen2.5-7B-Instruct")
        )
        self.timeout = timeout
        if not self.model:
            self.model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    def _chat(self, prompt: str) -> str:
        if not self.token:
            raise LLMError("HF_TOKEN is not set")
        url = self.endpoint.format(model=self.model)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
                "return_full_text": False,
            },
        }
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"HuggingFace request failed: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"HuggingFace error {response.status_code}: {response.text[:300]}")
        data = response.json()
        if isinstance(data, list) and data:
            return str(data[0].get("generated_text", "")).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        raise LLMOutputError(f"unexpected HuggingFace payload: {json.dumps(data)[:300]}")

    def _call_json(self, prompt: str, schema: str = "json") -> LLMResponse:
        from cta_qsar.llm.structured_output import extract_json

        text = self._chat(prompt)
        try:
            structured = extract_json(text)
        except LLMOutputError:
            structured = {}
        return LLMResponse(content=text, structured=structured, raw=text, provider=self.provider_name, model=self.model)

    def classify_case(self, case_context: dict[str, Any]) -> CaseClassification:
        text = self._chat(prompt_for("case").replace("{{CONTEXT}}", json.dumps(case_context)))
        return parse_model(text, CaseClassification)

    def select_strategy(self, strategy_context: dict[str, Any]) -> StrategySelection:
        text = self._chat(prompt_for("strategy").replace("{{CONTEXT}}", json.dumps(strategy_context)))
        return parse_model(text, StrategySelection)

    def diagnose(self, experiment_context: dict[str, Any]) -> list[DiagnosisOutput]:
        text = self._chat(prompt_for("diagnosis").replace("{{CONTEXT}}", json.dumps(experiment_context)))
        return parse_models(text, DiagnosisOutput)

    def propose_intervention(self, diagnosis_context: dict[str, Any]) -> InterventionOutput:
        text = self._chat(prompt_for("intervention").replace("{{CONTEXT}}", json.dumps(diagnosis_context)))
        return parse_model(text, InterventionOutput)

    def decide_stop(self, budget_context: dict[str, Any]) -> StopDecision:
        text = self._chat(prompt_for("stop").replace("{{CONTEXT}}", json.dumps(budget_context)))
        return parse_model(text, StopDecision)

    def summarize(self, report_context: dict[str, Any]) -> str:
        return self._chat(prompt_for("report").replace("{{CONTEXT}}", json.dumps(report_context)))