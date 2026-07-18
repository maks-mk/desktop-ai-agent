import re
from dataclasses import dataclass
from typing import Any

from core.message_utils import stringify_content

_ERROR_RE = re.compile(r"^ERROR\[(?P<error_type>[A-Z_]+)\]:\s*(?P<message>.*)$", re.DOTALL)


@dataclass(frozen=True)
class ToolExecutionResult:
    raw: str
    ok: bool
    error_type: str = ""
    message: str = ""
    retryable: bool = False


def parse_tool_execution_result(raw: Any) -> ToolExecutionResult:
    text = stringify_content(raw)
    match = _ERROR_RE.match(text.strip())
    if not match:
        return ToolExecutionResult(raw=text, ok=True, message=text, retryable=False)
    error_type = match.group("error_type")
    return ToolExecutionResult(
        raw=text,
        ok=False,
        error_type=error_type,
        message=match.group("message").strip(),
        retryable=error_type in {"VALIDATION", "LOOP_DETECTED", "TIMEOUT", "NETWORK"},
    )
