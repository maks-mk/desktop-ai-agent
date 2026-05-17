from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

_TEXT_TOOL_CALL_RE = re.compile(
    r"call:(?P<name>[A-Za-z_][\w.-]*)\s*\{(?P<args>.*?)\}\s*<tool_call\|>",
    re.DOTALL,
)
_UNQUOTED_KEY_RE = re.compile(r"(?<![\"'])\b([A-Za-z_][\w-]*)\s*:")
_QUOTE_TOKEN = '<|"|>'


@dataclass(frozen=True)
class TextToolCallExtraction:
    cleaned_text: str
    tool_calls: list[dict[str, Any]]


def extract_text_tool_calls(
    text: str,
    *,
    allowed_tool_names: Iterable[str] | None = None,
    id_prefix: str = "txtcall",
) -> TextToolCallExtraction:
    """Extract provider-emitted text tool calls like call:read_file{path:"x"}<tool_call|>."""
    raw_text = str(text or "")
    if "call:" not in raw_text or "<tool_call|>" not in raw_text:
        return TextToolCallExtraction(cleaned_text=raw_text, tool_calls=[])

    allowed = {str(name or "").strip() for name in (allowed_tool_names or []) if str(name or "").strip()}
    tool_calls: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        tool_name = str(match.group("name") or "").strip()
        if allowed and tool_name not in allowed:
            return match.group(0)
        args = _parse_text_tool_call_args(match.group("args") or "")
        if args is None:
            return match.group(0)
        tool_id = f"{id_prefix}{len(tool_calls) + 1:02d}"
        tool_calls.append({"id": tool_id, "name": tool_name, "args": args})
        return ""

    cleaned = _TEXT_TOOL_CALL_RE.sub(replace, raw_text).strip()
    return TextToolCallExtraction(cleaned_text=cleaned, tool_calls=tool_calls)


def _parse_text_tool_call_args(raw_args: str) -> dict[str, Any] | None:
    normalized = _normalize_text_tool_call_args(raw_args)
    candidates = [
        normalized,
        "{" + normalized + "}",
        "{" + _UNQUOTED_KEY_RE.sub(r'"\1":', normalized) + "}",
    ]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    parsed_pairs: dict[str, Any] = {}
    for key, value in re.findall(r"([A-Za-z_][\w-]*)\s*:\s*([^,}]+)", normalized):
        parsed_pairs[key] = _parse_scalar(value)
    return parsed_pairs or None


def _normalize_text_tool_call_args(raw_args: str) -> str:
    return str(raw_args or "").replace(_QUOTE_TOKEN, '"').strip()


def _parse_scalar(value: str) -> Any:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
