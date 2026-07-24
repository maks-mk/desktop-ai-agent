"""Anthropic provider adapter based on LangChain's ChatAnthropic."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel


class _ModelDumpCompatibleDict(dict[str, Any]):
    """Dictionary accepted by SDK-compatible proxies where a Pydantic object is expected."""

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return dict(self)


def _normalize_anthropic_message_delta_event(event: Any) -> Any:
    """Make malformed proxy metadata compatible with ``langchain-anthropic``.

    The official Anthropic SDK represents ``context_management`` and
    ``message_delta.delta.container`` as Pydantic models. Some Anthropic-compatible
    proxies return plain dictionaries instead, while ``ChatAnthropic`` calls
    ``.model_dump()`` on both fields in its final ``message_delta`` handler.
    """
    if getattr(event, "type", None) != "message_delta":
        return event

    event_updates: dict[str, Any] = {}
    usage = getattr(event, "usage", None)
    cache_creation = getattr(usage, "cache_creation", None)
    if isinstance(cache_creation, dict) and not hasattr(cache_creation, "model_dump"):
        compatible_usage = usage
        usage_updates = {"cache_creation": _ModelDumpCompatibleDict(cache_creation)}
        usage_copy = getattr(usage, "model_copy", None)
        if callable(usage_copy):
            compatible_usage = usage_copy(update=usage_updates)
        else:
            try:
                setattr(compatible_usage, "cache_creation", usage_updates["cache_creation"])
            except (AttributeError, TypeError):
                compatible_usage = None
        if compatible_usage is not None:
            event_updates["usage"] = compatible_usage

    context_management = getattr(event, "context_management", None)
    if isinstance(context_management, dict) and not hasattr(context_management, "model_dump"):
        event_updates["context_management"] = _ModelDumpCompatibleDict(context_management)

    delta = getattr(event, "delta", None)
    container = getattr(delta, "container", None)
    if isinstance(container, dict) and not hasattr(container, "model_dump"):
        compatible_delta = delta
        delta_updates = {"container": _ModelDumpCompatibleDict(container)}
        delta_copy = getattr(delta, "model_copy", None)
        if callable(delta_copy):
            compatible_delta = delta_copy(update=delta_updates)
        else:
            try:
                setattr(compatible_delta, "container", delta_updates["container"])
            except (AttributeError, TypeError):
                compatible_delta = None
        if compatible_delta is not None:
            event_updates["delta"] = compatible_delta

    if not event_updates:
        return event
    event_copy = getattr(event, "model_copy", None)
    if callable(event_copy):
        return event_copy(update=event_updates)
    for field, value in event_updates.items():
        try:
            setattr(event, field, value)
        except (AttributeError, TypeError):
            return event
    return event


def _build_anthropic_compatible_stream_adapter(base_cls: type) -> type:
    """Wrap ChatAnthropic's final stream-event conversion for compatible proxies."""

    class CompatibleStreamChatAnthropic(base_cls):
        def _make_message_chunk_from_anthropic_event(self, event: Any, *args: Any, **kwargs: Any):
            return super()._make_message_chunk_from_anthropic_event(
                _normalize_anthropic_message_delta_event(event), *args, **kwargs
            )

    return CompatibleStreamChatAnthropic

from core.config import AgentConfig
from core.http_headers import load_provider_headers
from core.reasoning_debug import debug_event

reasoning_logger = logging.getLogger("agent.reasoning_debug")


_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)
_ADAPTIVE_THINKING_WITHOUT_SAMPLING_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)


def _normalized_anthropic_model_name(model: str | None) -> str:
    return str(model or "").strip().lower()


def _anthropic_model_matches_family(model: str | None, families: tuple[str, ...]) -> bool:
    normalized_model = _normalized_anthropic_model_name(model)
    return any(normalized_model == family or normalized_model.startswith(f"{family}-") for family in families)


def _anthropic_model_uses_adaptive_thinking(model: str | None) -> bool:
    return _anthropic_model_matches_family(model, _ADAPTIVE_THINKING_MODELS)


def _anthropic_model_disallows_sampling(model: str | None) -> bool:
    return _anthropic_model_matches_family(model, _ADAPTIVE_THINKING_WITHOUT_SAMPLING_MODELS)


def create_anthropic_chat_model(
    config: AgentConfig,
    *,
    api_key_override: str | None = None,
) -> BaseChatModel:
    """Create a native Anthropic chat model with configured thinking support."""
    from langchain_anthropic import ChatAnthropic

    if api_key_override is None:
        api_key = config.anthropic_api_key.get_secret_value() if config.anthropic_api_key else None
    else:
        api_key = str(api_key_override or "")

    max_tokens = max(1, int(getattr(config, "anthropic_max_tokens", 8192)))
    kwargs: dict[str, Any] = {
        "model": config.anthropic_model,
        "anthropic_api_key": api_key,
        "max_tokens": max_tokens,
        "max_retries": 0,
        "default_headers": load_provider_headers(),
    }
    if not _anthropic_model_disallows_sampling(config.anthropic_model):
        kwargs["temperature"] = config.temperature
    if config.anthropic_base_url:
        # The Anthropic SDK appends "/v1/messages" itself, so the base URL must
        # NOT end with "/v1" (otherwise requests go to ".../v1/v1/messages").
        base_url = config.anthropic_base_url.strip().rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        kwargs["anthropic_api_url"] = base_url

    reasoning_level = str(getattr(config, "anthropic_reasoning", "") or "").strip().lower()
    if not bool(getattr(config, "enable_model_reasoning", True)) or reasoning_level in {"off", "none"}:
        kwargs["thinking"] = {"type": "disabled"}
        mode = "disabled"
        budget = None
        effort = None
    elif reasoning_level:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["effort"] = reasoning_level
        mode = "effort"
        budget = None
        effort = reasoning_level
    elif _anthropic_model_uses_adaptive_thinking(config.anthropic_model):
        # These models require adaptive thinking rather than a manual budget.
        kwargs["thinking"] = {"type": "adaptive"}
        mode = "adaptive"
        budget = None
        effort = None
    else:
        budget = int(getattr(config, "anthropic_thinking_budget", 4096))
        if max_tokens <= 1024:
            budget = 0
        else:
            budget = max(1024, min(budget, max_tokens - 1))
        if budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            mode = "budget"
        else:
            kwargs["thinking"] = {"type": "disabled"}
            mode = "disabled"
        effort = None

    reasoning_logger.debug(
        "anthropic reasoning config model=%s mode=%s effort=%s budget=%s",
        config.anthropic_model,
        mode,
        effort,
        budget,
    )
    debug_event(
        "reasoning_request",
        provider="anthropic",
        base_url=config.anthropic_base_url,
        model=config.anthropic_model,
        reasoning_enabled=bool(getattr(config, "enable_model_reasoning", True)),
        reasoning_effort=effort or "",
        mode=mode,
        budget=budget,
    )
    compatible_chat_model = _build_anthropic_compatible_stream_adapter(ChatAnthropic)
    return compatible_chat_model(**kwargs)
