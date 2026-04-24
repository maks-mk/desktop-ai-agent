from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from core.config import AgentConfig
from core.constants import AGENT_VERSION
from core.message_utils import is_tool_message_error, stringify_content
from core.model_profiles import find_active_profile, normalize_profiles_payload
from core.multimodal import extract_user_turn_data, resolve_model_capabilities
from core.session_store import (
    DEFAULT_CHAT_TITLE,
    SessionListEntry,
    SessionSnapshot,
    SessionStore,
    normalize_project_path,
)
from core.text_utils import build_tool_ui_labels, format_tool_output, parse_thought, prepare_markdown_for_render
from core.tool_args import canonicalize_tool_args
from core.tool_policy import ToolMetadata
from ui.tool_message_utils import extract_tool_args
from ui.visibility import is_hidden_internal_message

APPROVAL_MODE_PROMPT = "prompt"
APPROVAL_MODE_ALWAYS = "always"
CHAT_TITLE_MAX_LENGTH = 50
CHAT_TITLE_FALLBACK = DEFAULT_CHAT_TITLE
TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:пожалуйста|плиз|please)\s+)?"
    r"(?:(?:помоги(?:те)?|можешь(?:\s+ли)?|сделай(?:те)?|подскажи(?:те)?|нужно|надо|хочу|help|can you|could you|please)\s+)+",
    re.IGNORECASE,
)
TITLE_STRIP_RE = re.compile(r"^[\s\-\.,:;!?\"'`~()\[\]{}<>/\\]+|[\s\-\.,:;!?\"'`~()\[\]{}<>/\\]+$")
DIFF_BLOCK_RE = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ApprovalSummary:
    destructive_count: int
    mutating_count: int
    networked_count: int
    default_approve: bool
    risk_level: str
    impacts: tuple[str, ...]


def _provider_model(config: AgentConfig) -> tuple[str, str]:
    if config.provider == "gemini":
        return "Gemini", config.gemini_model
    return "OpenAI", config.openai_model


def _short_id(value: str, length: int = 16) -> str:
    if len(value) <= length:
        return value
    return f"{value[:length]}…"


def _tool_is_mcp(tool, metadata: ToolMetadata | None) -> bool:
    return bool((metadata and metadata.source == "mcp") or hasattr(tool, "_is_mcp") or ":" in tool.name)


def _tool_group(tool, metadata: ToolMetadata | None) -> str:
    if _tool_is_mcp(tool, metadata):
        return "MCP"
    if metadata and (metadata.mutating or metadata.destructive or metadata.requires_approval):
        return "Protected"
    return "Read-only"


def _format_tool_flags(metadata: ToolMetadata | None, *, is_mcp: bool) -> str:
    metadata = metadata or ToolMetadata(name="unknown", read_only=True)
    flags: list[str] = []
    if is_mcp:
        flags.append("mcp")
    if metadata.read_only and not metadata.mutating and not metadata.destructive:
        flags.append("read-only")
    if metadata.requires_approval:
        flags.append("approval")
    if metadata.mutating:
        flags.append("mutating")
    if metadata.destructive:
        flags.append("destructive")
    if metadata.networked:
        flags.append("network")
    return ", ".join(flags)


def _enabled_mcp_servers(tool_registry) -> list[str]:
    names = []
    for status in getattr(tool_registry, "mcp_server_status", []):
        server = status.get("server", "unknown")
        names.append(f"{server} (error)" if status.get("error") else server)
    return names


def _runtime_issue_count(tool_registry) -> int:
    lines = tool_registry.get_runtime_status_lines()
    return sum(
        1
        for line in lines
        if any(keyword in line.lower() for keyword in ("error", "warning", "failed", "unavailable"))
    )


def build_tools_snapshot(tool_registry) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metadata_map = getattr(tool_registry, "tool_metadata", {})
    tools = getattr(tool_registry, "tools", [])

    for group_name in ("Read-only", "Protected", "MCP"):
        items = []
        for tool in tools:
            metadata = metadata_map.get(tool.name)
            if _tool_group(tool, metadata) == group_name:
                items.append((tool, metadata))
        for tool, metadata in sorted(items, key=lambda item: item[0].name):
            rows.append(
                {
                    "group": group_name,
                    "name": tool.name,
                    "description": tool.description or "No description",
                    "flags": _format_tool_flags(metadata, is_mcp=_tool_is_mcp(tool, metadata)),
                }
            )
    return rows


def build_runtime_snapshot(config: AgentConfig, tool_registry, snapshot: SessionSnapshot) -> dict[str, Any]:
    provider_label, model_name = _provider_model(config)
    checkpoint_info = getattr(tool_registry, "checkpoint_info", {}) or {}
    backend = checkpoint_info.get("resolved_backend", config.checkpoint_backend)
    approvals = "off"
    if config.enable_approvals:
        approvals = "on"
        if getattr(snapshot, "approval_mode", APPROVAL_MODE_PROMPT) == APPROVAL_MODE_ALWAYS:
            approvals = "on (always for this session)"

    mcp_servers = _enabled_mcp_servers(tool_registry)
    issue_count = _runtime_issue_count(tool_registry)
    status = "ready" if issue_count == 0 else f"degraded ({issue_count} issue{'s' if issue_count != 1 else ''})"
    tools = build_tools_snapshot(tool_registry)
    return {
        "version": AGENT_VERSION,
        "provider": provider_label,
        "model": model_name,
        "backend": backend,
        "tools_count": len(getattr(tool_registry, "tools", [])),
        "session_id": snapshot.session_id,
        "session_short": _short_id(snapshot.session_id),
        "session_title": snapshot.title,
        "thread_id": snapshot.thread_id,
        "thread_short": _short_id(snapshot.thread_id),
        "project_path": snapshot.project_path,
        "approvals": approvals,
        "mcp_servers": mcp_servers,
        "mcp_text": ", ".join(mcp_servers) if mcp_servers else "none",
        "status": status,
        "config_mode": "debug" if config.debug else "standard",
        "runtime_lines": tool_registry.get_runtime_status_lines(),
        "tools": tools,
    }


def build_help_markdown() -> str:
    return (
        "## Workflow\n"
        "- Type a request and press **Enter**.\n"
        "- Use **Shift+Enter** for a new line.\n"
        "- Use **Up/Down** in an empty composer to browse earlier prompts from the current chat.\n"
        "- Type **@** to mention files from the current workspace.\n"
        "- Use the **+** button to attach images or Add files.\n"
        "- You can also paste images or files directly into the composer.\n"
        "- Use **Ctrl+N** for **New Session**.\n"
        "- Use **Ctrl+B** to show or hide the chat history sidebar.\n"
        "- Use **Ctrl+I** to open the runtime information popup.\n"
        "- Use **Open project folder** to switch the working directory and start a fresh chat for that folder.\n"
        "- Open **Settings** to add models, switch the active profile, or enable image support for a profile.\n"
        "- Open **Tools** to inspect read-only, protected, and MCP capabilities.\n"
        "- Open **Session** to review provider, backend, session, and MCP runtime state.\n"
        "- Right-click a chat in the sidebar to delete it from history.\n"
        "- Use **New Session** to reset the active session and clear session-scoped approvals.\n"
        "- Approval dialogs support **Approve**, **Deny**, and **Always for this session**.\n"
    )


def _plain_summary_text(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text or "").strip()


def generate_chat_title(user_text: str) -> str:
    text = " ".join(str(user_text or "").replace("\r", " ").replace("\n", " ").split()).strip()
    text = TITLE_PREFIX_RE.sub("", text)
    text = TITLE_STRIP_RE.sub("", text).strip()
    if not text:
        return CHAT_TITLE_FALLBACK
    if len(text) > CHAT_TITLE_MAX_LENGTH:
        text = text[: CHAT_TITLE_MAX_LENGTH - 1].rstrip() + "…"
    return text[:1].upper() + text[1:] if text else CHAT_TITLE_FALLBACK


def short_project_label(project_path: str | Path | None) -> str:
    normalized = normalize_project_path(project_path)
    path = Path(normalized)
    parts = [part for part in path.parts if part and part not in {path.anchor, "/", "\\"}]
    if not parts:
        return path.drive or normalized
    if len(parts) == 1:
        return parts[0]
    return "/".join(parts[-2:])


def append_project_label(title: str, project_path: str | Path | None) -> str:
    base_title = str(title or CHAT_TITLE_FALLBACK).strip() or CHAT_TITLE_FALLBACK
    label = short_project_label(project_path)
    if not label:
        return base_title
    suffix = f" [{label}]"
    if base_title.endswith(suffix):
        return base_title
    return f"{base_title}{suffix}"


def serialize_session_entries(entries: list[SessionListEntry]) -> list[dict[str, str]]:
    return [
        {
            "session_id": entry.session_id,
            "thread_id": entry.thread_id,
            "project_path": entry.project_path,
            "title": entry.title,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }
        for entry in entries
    ]


def _extract_ai_text(message: AIMessage | AIMessageChunk) -> str:
    return stringify_content(message.content)


def _diff_from_tool_content(content: str) -> str:
    match = DIFF_BLOCK_RE.search(content or "")
    return match.group(1).strip() if match else ""


def build_transcript_payload(state_values: dict[str, Any] | None) -> dict[str, Any]:
    values = state_values or {}
    summary_text = str(values.get("summary") or "").strip()
    turns: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    current_turn: dict[str, Any] | None = None

    for message in values.get("messages", []) or []:
        if isinstance(message, HumanMessage):
            text, attachments = extract_user_turn_data(message.content)
            text = text.strip()
            if not text and not attachments:
                continue
            current_turn = {"user_text": text, "attachments": attachments, "blocks": []}
            turns.append(current_turn)
            continue

        if current_turn is None:
            continue

        if isinstance(message, (AIMessage, AIMessageChunk)):
            for tool_call in getattr(message, "tool_calls", []) or []:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    pending_tool_calls[tool_call_id] = {
                        "name": tool_call.get("name", "tool"),
                        "args": canonicalize_tool_args(tool_call.get("args")),
                    }
            if is_hidden_internal_message(message):
                continue
            text = _extract_ai_text(message)
            if text.strip():
                _thought, clean_text, _has_thought = parse_thought(text)
                markdown = prepare_markdown_for_render(clean_text.strip()) if clean_text.strip() else ""
                if markdown:
                    current_turn["blocks"].append({"type": "assistant", "markdown": markdown})
            continue

        if isinstance(message, ToolMessage):
            tool_meta = pending_tool_calls.get(message.tool_call_id, {})
            tool_name = tool_meta.get("name") or message.name or "tool"
            tool_args = canonicalize_tool_args(tool_meta.get("args"))
            if not tool_args:
                tool_args = extract_tool_args(message)
            content = stringify_content(message.content)
            is_error = is_tool_message_error(message)
            labels = build_tool_ui_labels(tool_name, tool_args, phase="finished", is_error=is_error)
            current_turn["blocks"].append(
                {
                    "type": "tool",
                    "payload": {
                        "tool_id": message.tool_call_id,
                        "name": tool_name,
                        "args": tool_args,
                        "display": labels.get("title") or tool_name,
                        "subtitle": labels.get("subtitle", ""),
                        "raw_display": labels.get("raw_display", tool_name),
                        "args_state": labels.get("args_state", "complete"),
                        "display_state": "finished",
                        "phase": "finished",
                        "source_kind": labels.get("source_kind", "tool"),
                        "summary": _plain_summary_text(format_tool_output(tool_name, content, is_error)),
                        "content": content,
                        "is_error": is_error,
                        "duration": None,
                        "diff": _diff_from_tool_content(content),
                    },
                }
            )

    return {
        "summary_notice": (
            "Early messages were compressed automatically; the restored chat may be incomplete."
            if summary_text
            else ""
        ),
        "turns": turns,
    }


async def load_transcript_payload(agent_app, thread_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    async_get_state = getattr(agent_app, "aget_state", None)
    if callable(async_get_state):
        state = await async_get_state(config)
    else:
        state = agent_app.get_state(config)
    values = getattr(state, "values", {}) if state is not None else {}
    return build_transcript_payload(values if isinstance(values, dict) else {})


def summarize_approval_request(req_tools: list[dict]) -> ApprovalSummary:
    destructive_count = 0
    mutating_count = 0
    networked_count = 0
    impacts: set[str] = set()

    for tool in req_tools:
        policy = tool.get("policy") or {}
        name = (tool.get("name") or "").lower()
        destructive = bool(policy.get("destructive"))
        mutating = bool(policy.get("mutating"))
        networked = bool(policy.get("networked"))

        destructive_count += int(destructive)
        mutating_count += int(mutating)
        networked_count += int(networked)

        if destructive or mutating:
            if any(token in name for token in ("process", "pid", "port", "shell", "exec", "command")):
                impacts.add("processes")
            elif any(token in name for token in ("file", "directory", "path", "write", "edit", "delete", "download")):
                impacts.add("files")
            else:
                impacts.add("local state")
        if networked:
            impacts.add("network")

    risk_kinds = sum(int(count > 0) for count in (destructive_count, mutating_count, networked_count))
    mixed_risk = risk_kinds > 1
    default_approve = destructive_count == 0 and not mixed_risk
    risk_level = "high" if destructive_count or mixed_risk else "medium" if (mutating_count or networked_count) else "low"

    return ApprovalSummary(
        destructive_count=destructive_count,
        mutating_count=mutating_count,
        networked_count=networked_count,
        default_approve=default_approve,
        risk_level=risk_level,
        impacts=tuple(sorted(impacts)),
    )


def normalize_approval_mode(value: str | None) -> str:
    if value == APPROVAL_MODE_ALWAYS:
        return APPROVAL_MODE_ALWAYS
    return APPROVAL_MODE_PROMPT


def build_user_choice_payload(interrupt_payload: dict) -> dict[str, Any]:
    options_payload: list[dict[str, Any]] = []
    raw_options = interrupt_payload.get("options", []) if isinstance(interrupt_payload, dict) else []
    recommended = str((interrupt_payload or {}).get("recommended", "") or "").strip()
    normalized_recommended = recommended.casefold()
    matched_recommended_label = ""

    for index, raw_option in enumerate(raw_options, start=1):
        if isinstance(raw_option, dict):
            label = str(raw_option.get("label") or raw_option.get("value") or "").strip()
            submit_text = str(raw_option.get("submit_text") or raw_option.get("value") or label).strip()
            key = str(raw_option.get("key") or f"option_{index}").strip()
        else:
            label = str(raw_option or "").strip()
            submit_text = label
            key = f"option_{index}"
        if not label or not submit_text:
            continue
        candidates = {key.casefold(), submit_text.casefold(), label.casefold()}
        is_recommended = bool(normalized_recommended and normalized_recommended in candidates)
        if is_recommended and not matched_recommended_label:
            matched_recommended_label = label
        options_payload.append(
            {
                "key": key,
                "label": label,
                "submit_text": submit_text,
                "recommended": is_recommended,
            }
        )

    return {
        "kind": "user_choice",
        "question": interrupt_payload.get("question", "How would you like to proceed?"),
        "options": options_payload,
        "recommended_key": matched_recommended_label,
    }


def build_approval_payload(interrupt_payload: dict, current_session: SessionSnapshot) -> dict[str, Any]:
    req_tools = interrupt_payload.get("tools", []) if isinstance(interrupt_payload, dict) else []
    summary = summarize_approval_request(req_tools)
    return {
        "kind": interrupt_payload.get("kind", ""),
        "tools": req_tools,
        "summary": {
            "destructive_count": summary.destructive_count,
            "mutating_count": summary.mutating_count,
            "networked_count": summary.networked_count,
            "default_approve": summary.default_approve,
            "risk_level": summary.risk_level,
            "impacts": list(summary.impacts),
        },
        "approval_mode": normalize_approval_mode(getattr(current_session, "approval_mode", APPROVAL_MODE_PROMPT)),
    }


async def build_ui_payload(
    config: AgentConfig,
    tool_registry,
    store: SessionStore,
    snapshot: SessionSnapshot,
    model_profiles: dict[str, Any] | None = None,
    model_capabilities: dict[str, Any] | None = None,
    *,
    agent_app=None,
    include_transcript: bool = False,
) -> dict[str, Any]:
    runtime_snapshot = build_runtime_snapshot(config, tool_registry, snapshot)
    normalized_profiles = normalize_profiles_payload(model_profiles or {})
    effective_capabilities = resolve_model_capabilities(
        find_active_profile(normalized_profiles),
        model_capabilities if model_capabilities is not None else getattr(tool_registry, "model_capabilities", None),
    )
    payload = {
        "snapshot": runtime_snapshot,
        "tools": runtime_snapshot["tools"],
        "help_markdown": build_help_markdown(),
        "sessions": serialize_session_entries(store.list_sessions()),
        "active_session_id": snapshot.session_id,
        "model_profiles": normalized_profiles,
        "model_capabilities": effective_capabilities,
    }
    if include_transcript and agent_app is not None:
        payload["transcript"] = await load_transcript_payload(agent_app, snapshot.thread_id)
    return payload


async def close_runtime_resources(tool_registry) -> None:
    if tool_registry:
        await tool_registry.cleanup()
    try:
        from tools.system_tools import _net_client

        if _net_client:
            await _net_client.aclose()
    except ImportError:
        pass
