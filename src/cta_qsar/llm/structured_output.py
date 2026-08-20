"""Structured-output helpers: JSON extraction + schema validation."""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from cta_qsar.core.exceptions import LLMOutputError

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM text output."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    # strip prose around the first [ or {
    start = min(
        [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1] or [-1]
    )
    if start > 0:
        cleaned = cleaned[start:]
    if not cleaned:
        raise LLMOutputError("empty LLM output")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # try to salvage a list of dicts from embedded objects (many models echo prose)
    objects = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if objects:
        try:
            return [json.loads(o) for o in objects]
        except json.JSONDecodeError:
            raise LLMOutputError(f"unparseable JSON in LLM output: {cleaned[:200]}...") from None
    raise LLMOutputError(f"unparseable JSON in LLM output: {cleaned[:200]}...")


def _dict_item_to_text(item: dict[str, Any]) -> str:
    """Render a nested dict (e.g. an echoed candidate) as compact evidence text."""
    for key in ("name", "reason", "description"):
        if isinstance(item.get(key), str):
            return item[key]
    return json.dumps(item, sort_keys=True)


def _coerce(data: dict[str, Any], model: type[T]) -> dict[str, Any]:
    """Coerce common LLM near-misses against the pydantic schema."""
    coerced = dict(data)
    for name, field in model.model_fields.items():
        if name not in coerced:
            continue
        value = coerced[name]
        annotation = str(field.annotation)
        is_list = annotation.startswith("list")
        is_dict_list = is_list and "dict" in annotation
        is_optional = "None" in annotation
        # exact str-typed fields only; "dict[str, ...]" contains 'str' as a
        # substring and must not trigger stringification
        is_str_field = annotation in ("<class 'str'>", "str")
        # LLMs often emit "None"/"null" as strings for optional fields
        if is_optional:
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("", "none", "null", "n/a", "na", "-", "null object", "none."):
                    coerced[name] = None
                    continue
            elif value is None:
                continue
        # a bare null for a field that has a default (e.g. list[str] = [],
        # float = 0.0) maps to that default instead of failing validation
        if value is None:
            if field.default_factory is not None:
                coerced[name] = field.default_factory()
                continue
            if field.default is not None:
                coerced[name] = field.default
                continue
        if is_list and isinstance(value, str):
            if value.strip().startswith("["):
                try:
                    coerced[name] = json.loads(value)
                except json.JSONDecodeError:
                    coerced[name] = [value]
            elif "," in value:
                coerced[name] = [item.strip() for item in value.split(",") if item.strip()]
            elif value.strip():
                coerced[name] = [value]
        elif is_list and isinstance(value, dict):
            # model wrapped a single object where a list is expected
            coerced[name] = [value] if value else []
        elif is_list and not is_dict_list and isinstance(value, list) and value and all(
            isinstance(v, dict) for v in value
        ):
            coerced[name] = [_dict_item_to_text(v) for v in value]
        elif is_dict_list and isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(value)
                if isinstance(parsed, list) and all(isinstance(v, dict) for v in parsed):
                    coerced[name] = parsed
        elif is_dict_list and isinstance(value, list):
            kept: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    kept.append(item)
                elif isinstance(item, str):
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed = json.loads(item)
                        if isinstance(parsed, dict):
                            kept.append(parsed)
            # always assign: an all-string list (e.g. a plan echoed as
            # ["baseline"]) cannot be validated and should not block parsing
            coerced[name] = kept
        elif not is_list and isinstance(value, dict) and is_str_field:
            coerced[name] = str(value)
        elif not is_list and isinstance(value, list) and is_str_field:
            coerced[name] = str(value[0]) if len(value) == 1 else ", ".join(str(v) for v in value)
        elif "float" in annotation and isinstance(value, (int, float)):
            coerced[name] = float(value)
        elif "float" in annotation and isinstance(value, list):
            # model wrapped a scalar in a list
            if value:
                with contextlib.suppress(ValueError, TypeError):
                    coerced[name] = float(value[0])
        elif "float" in annotation and isinstance(value, str):
            low = value.strip().lower()
            if low in ("", "none", "null", "n/a", "na", "-"):
                coerced[name] = field.default if field.default is not None else 0.0
            else:
                with contextlib.suppress(ValueError):
                    coerced[name] = float(value)
        elif "bool" in annotation and isinstance(value, str):
            low = value.strip().lower()
            if low in ("", "none", "null", "n/a", "na"):
                coerced[name] = field.default if field.default is not None else False
            else:
                coerced[name] = low in ("true", "yes", "1", "y")
        elif not is_list and not isinstance(value, str) and is_str_field:
            coerced[name] = str(value)
    return coerced


def parse_model(text: str, model: type[T]) -> T:
    """Parse LLM text into a pydantic model (clamped defaults on error keys)."""
    data = extract_json(text)
    if isinstance(data, list) and hasattr(model, "model_validate") and len(data) == 1:
        data = data[0]
    if isinstance(data, dict):
        data = _coerce(data, model)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        # Lenient fallback: drop the offending fields (they take their
        # defaults) rather than failing the whole structured decision.
        dropped = _drop_invalid(data, exc)
        if dropped is not None:
            return model.model_validate(dropped)
        raise LLMOutputError(f"schema validation failed: {exc}") from exc


def _drop_invalid(data: dict[str, Any], exc: ValidationError) -> dict[str, Any] | None:
    """Return data without the failing top-level keys, or None on retry failure."""
    bad = {err["loc"][0] for err in exc.errors() if err.get("loc")}
    if not bad:
        return None
    reduced = {k: v for k, v in data.items() if k not in bad}
    if len(reduced) == len(data):
        return None
    return reduced


def parse_models(text: str, model: type[T]) -> list[T]:
    """Parse LLM text into a list of pydantic models (lenient on failure)."""
    data = extract_json(text)
    if isinstance(data, dict):
        for key in ("diagnoses", "items", "candidates"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        data = [data]
    result: list[T] = []
    for item in data:
        try:
            result.append(model.model_validate(item))
        except ValidationError:
            continue
    return result