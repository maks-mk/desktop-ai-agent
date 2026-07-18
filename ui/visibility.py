from __future__ import annotations

from typing import Any

from core.constants import DEFAULT_INTERNAL_UI_NOTICE


def get_agent_internal_metadata(message: Any) -> dict[str, Any]:
    metadata = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(metadata, dict):
        return {}
    internal = metadata.get("agent_internal")
    return dict(internal) if isinstance(internal, dict) else {}


def is_hidden_internal_message(message: Any) -> bool:
    internal = get_agent_internal_metadata(message)
    return bool(internal) and internal.get("visible_in_ui") is False


def get_internal_ui_notice(message: Any) -> str:
    internal = get_agent_internal_metadata(message)
    if not internal or internal.get("visible_in_ui") is not False:
        return ""
    if internal.get("silent_in_ui") is True:
        return ""
    notice = str(internal.get("ui_notice") or "").strip()
    return notice or DEFAULT_INTERNAL_UI_NOTICE
