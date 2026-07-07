from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable


logger = logging.getLogger("agent.reasoning_debug")

KNOWN_STREAM_FIELDS = frozenset(
    {
        "__interrupt__",
        "additional_kwargs",
        "agent",
        "args",
        "content",
        "content_blocks",
        "created",
        "data",
        "delta",
        "finish_reason",
        "id",
        "index",
        "invalid_tool_calls",
        "langgraph_node",
        "message",
        "messages",
        "model",
        "name",
        "object",
        "response_metadata",
        "role",
        "summarize",
        "text",
        "tool_call_chunks",
        "tool_calls",
        "tools",
        "type",
        "usage",
    }
)

REASONING_FIELD_HINTS = frozenset(
    {
        "analysis",
        "reasoning",
        "reasoning_content",
        "reasoning_delta",
        "reasoning_details",
        "thinking",
        "thinking_content",
        "thought",
        "thought_signature",
    }
)

STRUCTURED_OUTPUT_PARSED_EXCLUDE = {
    "parsed": True,
    "choices": {"__all__": {"message": {"parsed"}}},
}


def reasoning_debug_enabled() -> bool:
    return bool(logger.isEnabledFor(logging.DEBUG) and not logger.disabled)


def now() -> float:
    return time.perf_counter()


def elapsed_since(start: float | None) -> float | None:
    if start is None:
        return None
    return max(0.0, time.perf_counter() - start)


def preview_value(value: Any, *, limit: int = 600) -> str:
    try:
        rendered = json.dumps(_json_safe(value, depth=3), ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}…(+{len(rendered) - limit} chars)"


def debug_event(event: str, **fields: Any) -> None:
    if not reasoning_debug_enabled():
        return
    payload = {"event": event, **fields}
    try:
        rendered = json.dumps(_json_safe(payload, depth=4), ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = repr(payload)
    logger.debug(rendered)


def log_unknown_fields(
    source: str,
    value: Any,
    *,
    known_fields: Iterable[str] = KNOWN_STREAM_FIELDS,
    path: str = "",
) -> None:
    if not reasoning_debug_enabled():
        return
    known = set(known_fields)
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}" if path else key_text
            normalized = key_text.strip().lower()
            if normalized not in known or normalized in REASONING_FIELD_HINTS:
                debug_event(
                    "stream_field_observed",
                    source=source,
                    field=item_path,
                    field_kind="reasoning_hint" if normalized in REASONING_FIELD_HINTS else "unknown",
                    value_preview=preview_value(item, limit=200),
                )
            log_unknown_fields(source, item, known_fields=known, path=item_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            log_unknown_fields(source, item, known_fields=known, path=f"{path}[{index}]")


def _json_safe(value: Any, *, depth: int) -> Any:
    if depth <= 0:
        return _short_type(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth=depth - 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth - 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json", exclude=STRUCTURED_OUTPUT_PARSED_EXCLUDE)
        except TypeError:
            try:
                dumped = value.model_dump(mode="json")
            except Exception:
                dumped = None
        except Exception:
            dumped = None
        if dumped is not None:
            return _json_safe(dumped, depth=depth - 1)
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict(), depth=depth - 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            public = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
            return {"__type__": type(value).__name__, **_json_safe(public, depth=depth - 1)}
        except Exception:
            pass
    return _short_type(value)


def _short_type(value: Any) -> str:
    return f"<{type(value).__name__}>"
