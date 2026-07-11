from __future__ import annotations

import re
from typing import Any, Dict

from core.tool_policy import ToolMetadata


# ---------------------------------------------------------------------------
# Combined regex patterns — one alternation per category instead of N
# individual patterns.  This reduces classify_shell_command from ~50
# sequential re.search calls to a single re.search per category.
# ---------------------------------------------------------------------------

_INSPECT_ONLY_COMMAND_RE = re.compile(
    r"\b(?:get-process|tasklist|where-object|select-object|findstr|"
    r"get-childitem|get-content|select-string|dir|type|where|netstat|ss|ps)\b",
    re.IGNORECASE,
)
_NETWORK_DIAGNOSTIC_COMMAND_RE = re.compile(
    r"\b(?:ping(?:\.exe)?|test-netconnection|resolve-dnsname|"
    r"nslookup(?:\.exe)?|tracert(?:\.exe)?|pathping(?:\.exe)?)\b",
    re.IGNORECASE,
)
_MUTATING_COMMAND_RE = re.compile(
    r"\b(?:taskkill|stop-process|remove-item|rm|del|rmdir|move-item|rename-item|"
    r"copy-item|set-content|add-content|out-file|new-item|npm\s+install|npm\s+uninstall|"
    r"pip\s+install|python\s+-m\s+pip\s+install|git\s+checkout|git\s+reset)\b"
    r"|python\s+-c\b.*\b(?:write_text|write_bytes|open\s*\([^)]*,\s*['\"]?[wax])",
    re.IGNORECASE,
)
_DESTRUCTIVE_COMMAND_RE = re.compile(
    r"\b(?:taskkill|stop-process|remove-item|rm|del|rmdir)\b",
    re.IGNORECASE,
)
_LONG_RUNNING_SERVICE_RE = re.compile(
    r"\b(?:python(?:3(?:\.\d+)?)?\s+-m\s+http\.server|http-server|npm\s+start|"
    r"npm\s+run\s+dev|npm\s+exec\b.*\bhttp-server\b|npx\b.*\bhttp-server\b|"
    r"uvicorn|flask\s+run|webpack(?:\.cmd)?\s+serve|serve)\b",
    re.IGNORECASE,
)
_HTTP_PROBE_COMMAND_RE = re.compile(
    r"\b(?:curl(?:\.exe)?|invoke-webrequest|iwr|invoke-restmethod|irm)\b",
    re.IGNORECASE,
)
_HTTP_WRITE_METHOD_RE = re.compile(
    r"(?:--request|-x|\b-method\b)\s*[:=]?[\'\"]?(post|put|patch|delete)\b",
    re.IGNORECASE,
)
_HTTP_WRITE_FLAG_RE = re.compile(
    r"(?:^|[\s;|&])-(?:d|f|t)\b"
    r"|--(?:data(?:-raw|-binary|-ascii|-urlencode)?|form|string|upload-file|json)\b",
    re.IGNORECASE,
)
_RIPGREP_COMMAND_NAMES = {"rg", "rg.exe"}


def _is_http_probe_command(command: str) -> bool:
    return bool(_HTTP_PROBE_COMMAND_RE.search(command))


def _is_http_write_command(command: str) -> bool:
    if not _is_http_probe_command(command):
        return False
    if _HTTP_WRITE_METHOD_RE.search(command):
        return True
    return bool(_HTTP_WRITE_FLAG_RE.search(command))


def _has_unquoted_shell_operator(command: str) -> bool:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if quote is not None:
            continue
        if char in (";", "|", "<", ">"):
            return True
        if char == "&":
            stripped_before = command[:index].strip()
            if stripped_before:
                return True
        if char == "`":
            return True
    return False


def _first_command_token(command: str) -> str:
    raw = command.strip()
    if raw.startswith("&"):
        raw = raw[1:].lstrip()
    if not raw:
        return ""

    quote: str | None = None
    token_chars: list[str] = []
    for char in raw:
        if char in ("'", '"'):
            if quote == char:
                quote = None
                continue
            if quote is None:
                quote = char
                continue
        if quote is None and char.isspace():
            break
        token_chars.append(char)
    return "".join(token_chars).strip()


def _is_ripgrep_read_only_command(command: str) -> bool:
    normalized = str(command or "").strip()
    if not normalized or _has_unquoted_shell_operator(normalized):
        return False
    if "$(" in normalized or "%{" in normalized:
        return False

    token = _first_command_token(normalized).strip('"\'')
    if not token:
        return False
    executable = token.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    return executable in _RIPGREP_COMMAND_NAMES


def classify_shell_command(command: str) -> Dict[str, Any]:
    normalized = str(command or "").strip()
    is_long_running_service = bool(_LONG_RUNNING_SERVICE_RE.search(normalized))
    is_destructive = bool(_DESTRUCTIVE_COMMAND_RE.search(normalized))
    is_http_write = _is_http_write_command(normalized)
    is_mutating = is_long_running_service or is_destructive or is_http_write or bool(
        _MUTATING_COMMAND_RE.search(normalized)
    )
    is_network_diagnostic = bool(_NETWORK_DIAGNOSTIC_COMMAND_RE.search(normalized))
    is_inspect_only = (
        not is_mutating
        and not is_long_running_service
        and (
            bool(_INSPECT_ONLY_COMMAND_RE.search(normalized))
            or is_network_diagnostic
            or _is_http_probe_command(normalized)
            or _is_ripgrep_read_only_command(normalized)
        )
    )
    return {
        "normalized": normalized,
        "inspect_only": is_inspect_only,
        "mutating": is_mutating,
        "destructive": is_destructive,
        "long_running_service": is_long_running_service,
        "network_diagnostic": is_network_diagnostic,
        "http_probe": _is_http_probe_command(normalized),
    }


def shell_command_requires_approval(command: str) -> bool:
    profile = classify_shell_command(command)
    if profile.get("inspect_only") and not profile.get("long_running_service"):
        return False
    return True


def tool_requires_approval(
    tool_name: str,
    tool_args: Dict[str, Any] | None = None,
    *,
    metadata: ToolMetadata | None = None,
    approvals_enabled: bool = True,
) -> bool:
    if not approvals_enabled:
        return False

    normalized_name = str(tool_name or "").strip().lower()
    effective_metadata = metadata or ToolMetadata(name=str(tool_name or "unknown_tool"))
    if normalized_name == "cli_exec":
        command = str(((tool_args or {}).get("command")) or "").strip()
        if not command:
            return True
        return shell_command_requires_approval(command)

    return bool(
        effective_metadata.requires_approval
        or effective_metadata.destructive
        or effective_metadata.mutating
    )
