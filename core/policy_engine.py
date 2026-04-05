from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from langchain_core.messages import BaseMessage

from core.message_context import IsInternalRetry
from core.tool_policy import ToolMetadata


_INSPECT_ONLY_COMMAND_PATTERNS = (
    re.compile(r"\bget-process\b", re.IGNORECASE),
    re.compile(r"\btasklist\b", re.IGNORECASE),
    re.compile(r"\bwhere-object\b", re.IGNORECASE),
    re.compile(r"\bselect-object\b", re.IGNORECASE),
    re.compile(r"\bfindstr\b", re.IGNORECASE),
    re.compile(r"\bget-childitem\b", re.IGNORECASE),
    re.compile(r"\bget-content\b", re.IGNORECASE),
    re.compile(r"\bselect-string\b", re.IGNORECASE),
    re.compile(r"\bdir\b", re.IGNORECASE),
    re.compile(r"\btype\b", re.IGNORECASE),
    re.compile(r"\bwhere\b", re.IGNORECASE),
    re.compile(r"\bnetstat\b", re.IGNORECASE),
    re.compile(r"\bss\b", re.IGNORECASE),
    re.compile(r"\bps\b", re.IGNORECASE),
)
_NETWORK_DIAGNOSTIC_COMMAND_PATTERNS = (
    re.compile(r"\bping(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\btest-netconnection\b", re.IGNORECASE),
    re.compile(r"\bresolve-dnsname\b", re.IGNORECASE),
    re.compile(r"\bnslookup(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\btracert(?:\.exe)?\b", re.IGNORECASE),
    re.compile(r"\bpathping(?:\.exe)?\b", re.IGNORECASE),
)
_MUTATING_COMMAND_PATTERNS = (
    re.compile(r"\btaskkill\b", re.IGNORECASE),
    re.compile(r"\bstop-process\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\bdel\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bmove-item\b", re.IGNORECASE),
    re.compile(r"\brename-item\b", re.IGNORECASE),
    re.compile(r"\bcopy-item\b", re.IGNORECASE),
    re.compile(r"\bset-content\b", re.IGNORECASE),
    re.compile(r"\badd-content\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+uninstall\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\b", re.IGNORECASE),
)
_DESTRUCTIVE_COMMAND_PATTERNS = (
    re.compile(r"\btaskkill\b", re.IGNORECASE),
    re.compile(r"\bstop-process\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\bdel\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
)
_LONG_RUNNING_SERVICE_PATTERNS = (
    re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+http\.server\b", re.IGNORECASE),
    re.compile(r"\bhttp-server\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+start\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+run\s+dev\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+exec\b.*\bhttp-server\b", re.IGNORECASE),
    re.compile(r"\bnpx\b.*\bhttp-server\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bflask\s+run\b", re.IGNORECASE),
    re.compile(r"\bwebpack(?:\.cmd)?\s+serve\b", re.IGNORECASE),
    re.compile(r"\bserve\b", re.IGNORECASE),
)
_HTTP_PROBE_COMMAND_RE = re.compile(
    r"\b(?:curl(?:\.exe)?|invoke-webrequest|iwr|invoke-restmethod|irm)\b",
    re.IGNORECASE,
)
_HTTP_WRITE_METHOD_RE = re.compile(
    r"(?:--request|-x|\b-method\b)\s*[:=]?\s*['\"]?(post|put|patch|delete)\b",
    re.IGNORECASE,
)
_HTTP_WRITE_FLAG_PATTERNS = (
    re.compile(r"(^|[\s;|&])-(?:d|f|t)\b", re.IGNORECASE),
    re.compile(r"--(?:data(?:-raw|-binary|-ascii|-urlencode)?|form|string|upload-file|json)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class TurnPolicyDecision:
    inspect_only: bool
    requires_operational_evidence: bool
    prefer_read_only_fallback: bool
    intent: str
    should_force_tools: bool


@dataclass(frozen=True)
class ToolCallPolicyDecision:
    allowed: bool
    reason: str = ""


def _is_http_probe_command(command: str) -> bool:
    return bool(_HTTP_PROBE_COMMAND_RE.search(command))


def _is_http_write_command(command: str) -> bool:
    if not _is_http_probe_command(command):
        return False
    if _HTTP_WRITE_METHOD_RE.search(command):
        return True
    return any(pattern.search(command) for pattern in _HTTP_WRITE_FLAG_PATTERNS)


def classify_shell_command(command: str) -> Dict[str, Any]:
    normalized = str(command or "").strip()
    is_long_running_service = any(pattern.search(normalized) for pattern in _LONG_RUNNING_SERVICE_PATTERNS)
    is_destructive = any(pattern.search(normalized) for pattern in _DESTRUCTIVE_COMMAND_PATTERNS)
    is_http_write = _is_http_write_command(normalized)
    is_mutating = is_long_running_service or is_destructive or is_http_write or any(
        pattern.search(normalized) for pattern in _MUTATING_COMMAND_PATTERNS
    )
    is_network_diagnostic = any(pattern.search(normalized) for pattern in _NETWORK_DIAGNOSTIC_COMMAND_PATTERNS)
    is_inspect_only = (
        not is_mutating
        and not is_long_running_service
        and (
            any(pattern.search(normalized) for pattern in _INSPECT_ONLY_COMMAND_PATTERNS)
            or is_network_diagnostic
            or _is_http_probe_command(normalized)
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
    return bool(
        profile.get("mutating")
        or profile.get("destructive")
        or profile.get("long_running_service")
    )


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


class PolicyEngine:
    def evaluate_turn(
        self,
        *,
        task: str,
        messages: List[BaseMessage],
        current_turn_id: int,
        is_internal_retry: IsInternalRetry,
    ) -> TurnPolicyDecision:
        _ = task
        _ = messages
        _ = current_turn_id
        _ = is_internal_retry
        return TurnPolicyDecision(
            inspect_only=False,
            requires_operational_evidence=False,
            prefer_read_only_fallback=False,
            intent="chat",
            should_force_tools=False,
        )

    @staticmethod
    def tool_call_allowed_for_turn(
        tool_call: Dict[str, Any],
        *,
        inspect_only: bool,
        tool_is_read_only: Callable[[str], bool],
    ) -> ToolCallPolicyDecision:
        if not inspect_only:
            return ToolCallPolicyDecision(allowed=True)

        tool_name = str(tool_call.get("name") or "").strip()
        if not tool_name:
            return ToolCallPolicyDecision(allowed=False, reason="missing_tool_name")

        if tool_name == "cli_exec":
            command = str((tool_call.get("args") or {}).get("command", "") or "")
            profile = classify_shell_command(command)
            if profile.get("inspect_only") and not profile.get("long_running_service"):
                return ToolCallPolicyDecision(allowed=True)
            return ToolCallPolicyDecision(allowed=False, reason="cli_exec_non_inspect")

        if tool_is_read_only(tool_name):
            return ToolCallPolicyDecision(allowed=True)
        return ToolCallPolicyDecision(allowed=False, reason="mutating_tool_in_inspect_turn")

    def tool_calls_violate_turn_intent(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        inspect_only: bool,
        tool_is_read_only: Callable[[str], bool],
    ) -> bool:
        if not tool_calls or not inspect_only:
            return False
        return any(
            not self.tool_call_allowed_for_turn(
                tool_call,
                inspect_only=inspect_only,
                tool_is_read_only=tool_is_read_only,
            ).allowed
            for tool_call in tool_calls
        )

    @staticmethod
    def _tool_name_tokens(tool_name: str) -> set[str]:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(tool_name or "")).lower()
        return {token for token in re.split(r"[^a-z0-9а-яё]+", normalized) if token}

    @staticmethod
    def choose_fallback_tool_names(
        *,
        preferred_tool_names: List[str],
        recent_tool_names: List[str],
        all_tool_names: List[str],
        prefer_read_only: bool,
        tool_is_read_only: Callable[[str], bool],
    ) -> List[str]:
        candidates: List[str] = []
        for source_names in (preferred_tool_names, recent_tool_names):
            for name in source_names:
                if name and name not in candidates:
                    candidates.append(name)

        if prefer_read_only:
            candidates = [name for name in candidates if name == "cli_exec" or tool_is_read_only(name)]

        if recent_tool_names and not preferred_tool_names and not prefer_read_only:
            recent_tokens = set()
            for name in recent_tool_names:
                recent_tokens.update(PolicyEngine._tool_name_tokens(name))
            for name in all_tool_names:
                if name in candidates:
                    continue
                if PolicyEngine._tool_name_tokens(name) & recent_tokens:
                    candidates.append(name)

        if not candidates:
            candidates = [
                name
                for name in all_tool_names
                if not prefer_read_only or name == "cli_exec" or tool_is_read_only(name)
            ]

        return candidates or list(all_tool_names)
