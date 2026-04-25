import re
from typing import Any

from langchain_core.messages import ToolMessage


_STRUCTURED_TOOL_ERROR_RE = re.compile(r"^\s*ERROR\[[A-Z_]+\]:", re.IGNORECASE)
_LEGACY_INTERRUPTED_TOOL_ERROR_RE = re.compile(
    r"^\s*Error:\s*Execution interrupted\b",
    re.IGNORECASE,
)
_TOOL_MESSAGE_ERROR_STATUS = "error"
_TOOL_MESSAGE_SUCCESS_STATUS = "success"


def _stringify_content_item(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "text" in item:
            return str(item.get("text") or "")
        if "refusal" in item:
            return str(item.get("refusal") or "")
        if "content" in item:
            return _stringify_content_item(item.get("content"))
        return ""
    if isinstance(item, list):
        return "".join(_stringify_content_item(part) for part in item)
    return str(item)


def stringify_content(content: Any) -> str:
    return _stringify_content_item(content)


def compact_text(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 15] + "... [truncated]"


def is_error_text(text: Any) -> bool:
    normalized = stringify_content(text).strip()
    return bool(
        _STRUCTURED_TOOL_ERROR_RE.match(normalized)
        or _LEGACY_INTERRUPTED_TOOL_ERROR_RE.match(normalized)
    )


def tool_message_status(message: ToolMessage) -> str:
    status = str(getattr(message, "status", "") or "").strip().lower()
    if status in {_TOOL_MESSAGE_SUCCESS_STATUS, _TOOL_MESSAGE_ERROR_STATUS}:
        return status
    return _TOOL_MESSAGE_ERROR_STATUS if is_error_text(message.content) else _TOOL_MESSAGE_SUCCESS_STATUS


def is_tool_message_error(message: ToolMessage) -> bool:
    return tool_message_status(message) == _TOOL_MESSAGE_ERROR_STATUS
