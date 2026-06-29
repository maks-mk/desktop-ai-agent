"""Shared helpers for provider adapter factories.

These utilities are provider-agnostic: model-name normalization, reasoning-effort
normalization, and pydantic-field introspection for chat-model kwargs.
"""

from __future__ import annotations

from typing import Any


def normalized_model_name(model_name: str | None) -> str:
    """Return a trimmed, lowercased model name."""
    return str(model_name or "").strip().lower()


def normalized_gemini_model_name(model_name: str | None) -> str:
    """Strip the ``models/`` prefix used by some Gemini endpoints."""
    normalized = normalized_model_name(model_name)
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    return normalized


def gemini_model_supports_thinking_budget(model_name: str | None) -> bool:
    """Whether a Gemini model accepts the ``thinking_budget`` parameter."""
    normalized = normalized_gemini_model_name(model_name)
    return normalized.startswith("gemini-2.5") or normalized in {
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
    }


def gemini_model_supports_thinking_level(model_name: str | None) -> bool:
    """Whether a Gemini model accepts the ``thinking_level`` parameter."""
    return normalized_gemini_model_name(model_name).startswith("gemini-3")


def chat_model_accepts_kwarg(model_cls: type, name: str) -> bool:
    """Check whether *model_cls* declares *name* as a pydantic field.

    Falls back to ``True`` when the class does not expose ``model_fields``
    (e.g. non-pydantic base), so callers can still attempt to pass the kwarg.
    """
    fields = getattr(model_cls, "model_fields", None)
    if isinstance(fields, dict):
        return name in fields
    return True


def normalized_reasoning_effort(value: str | None) -> str:
    """Clamp *value* to a known reasoning-effort label, defaulting to ``medium``."""
    normalized = str(value or "").strip().lower()
    if normalized in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        return normalized
    return "medium"


def normalized_gemini_thinking_level(value: str | None) -> str:
    """Map a reasoning-effort label to Gemini's ``thinking_level`` vocabulary."""
    normalized = normalized_reasoning_effort(value)
    if normalized in {"none", "minimal"}:
        return "minimal"
    if normalized == "xhigh":
        return "high"
    return normalized
