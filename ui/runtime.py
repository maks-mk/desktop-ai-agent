from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import SecretStr

from agent import build_agent_app
from core.config import AgentConfig
from core.constants import AGENT_VERSION, BASE_DIR
from core.logging_config import setup_logging
from core.message_utils import is_tool_message_error, stringify_content
from core.multimodal import (
    DEFAULT_MODEL_CAPABILITIES,
    build_user_message_content,
    extract_user_turn_data,
    normalize_model_capabilities,
    normalize_request_payload,
    resolve_model_capabilities,
    request_has_content,
    request_task_text,
    request_user_text,
)
from core.run_logger import JsonlRunLogger
from core.model_profiles import ModelProfileStore, find_active_profile, normalize_profiles_payload
from core.session_store import (
    DEFAULT_CHAT_TITLE,
    SessionListEntry,
    SessionSnapshot,
    SessionStore,
    normalize_project_path,
)
from core.session_utils import repair_session_if_needed
from core.text_utils import format_tool_display, format_tool_output, parse_thought, prepare_markdown_for_render
from core.tool_policy import ToolMetadata
from ui.streaming import StreamEvent, StreamProcessor
from ui.tool_message_utils import extract_tool_args
from ui.visibility import get_internal_ui_notice, is_hidden_internal_message

logger = logging.getLogger("agent")

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


def setup_runtime() -> AgentConfig:
    if getattr(sys, "frozen", False):
        os.chdir(os.getcwd())
    config = AgentConfig()
    setup_logging(level=config.log_level, log_file=config.log_file)
    return config


def build_initial_state(user_input: Any, session_id: str, safety_mode: str = "default") -> dict:
    request_payload = normalize_request_payload(user_input)
    user_text = request_payload["text"]
    attachments = request_payload["attachments"]
    current_task = request_task_text(request_payload)
    return {
        "messages": [HumanMessage(content=build_user_message_content(user_text, attachments))],
        "steps": 0,
        "token_usage": {},
        "current_task": current_task,
        "session_id": session_id,
        "run_id": uuid.uuid4().hex,
        "turn_id": 1,
        "pending_approval": None,
        "open_tool_issue": None,
        "recovery_state": {
            "turn_id": 1,
            "active_issue": None,
            "active_strategy": None,
            "strategy_queue": [],
            "attempts_by_strategy": {},
            "progress_markers": [],
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": [],
        },
        "has_protocol_error": False,
        "self_correction_retry_count": 0,
        "self_correction_retry_turn_id": 0,
        "self_correction_fingerprint_history": [],
        "last_tool_error": "",
        "last_tool_result": "",
        "safety_mode": safety_mode,
    }


def build_graph_config(thread_id: str, max_loops: int) -> dict:
    # LangGraph recursion_limit counts graph supersteps, not our logical `steps`.
    # Keep this as a technical translation layer so MAX_LOOPS remains the only
    # user-facing step budget enforced by the workflow state.
    graph_supersteps_per_logical_step = 6
    graph_superstep_overhead = 8
    recursion_limit = max(
        16,
        int(max_loops or 0) * graph_supersteps_per_logical_step + graph_superstep_overhead,
    )
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


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
        "- Open **Tools** to inspect read-only, protected, and MCP capabilities.\n"
        "- Open **Session** to review provider, backend, session, and MCP runtime state.\n"
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
                        "args": tool_call.get("args", {}),
                    }
            if is_hidden_internal_message(message):
                notice = get_internal_ui_notice(message)
                if notice:
                    current_turn["blocks"].append(
                        {
                            "type": "notice",
                            "message": notice,
                            "level": "warning",
                        }
                    )
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
            tool_args = tool_meta.get("args") or {}
            if not tool_args:
                tool_args = extract_tool_args(message)
            content = stringify_content(message.content)
            is_error = is_tool_message_error(message)
            current_turn["blocks"].append(
                {
                    "type": "tool",
                    "payload": {
                        "tool_id": message.tool_call_id,
                        "name": tool_name,
                        "args": tool_args,
                        "display": format_tool_display(tool_name, tool_args),
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

    recommended_key = matched_recommended_label

    return {
        "kind": "user_choice",
        "question": interrupt_payload.get("question", "How would you like to proceed?"),
        "options": options_payload,
        "recommended_key": recommended_key,
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


class AgentRunWorker(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    user_choice_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)
    shutdown_complete = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_task: asyncio.Task | None = None
        self.config: AgentConfig | None = None
        self.base_config: AgentConfig | None = None
        self.store: SessionStore | None = None
        self.profile_store: ModelProfileStore | None = None
        self.model_profiles: dict[str, Any] = {"active_profile": None, "profiles": []}
        self.runtime_model_capabilities: dict[str, Any] = dict(DEFAULT_MODEL_CAPABILITIES)
        self.model_capabilities: dict[str, Any] = dict(DEFAULT_MODEL_CAPABILITIES)
        self._runtime_profile_id: str = ""
        self.agent_app = None
        self.tool_registry = None
        self.current_session: SessionSnapshot | None = None
        self.ui_run_logger: JsonlRunLogger | None = None
        self._is_busy = False
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        self._active_run_elapsed_seconds = 0.0
        self._active_request_has_images = False

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()

        async def _wrapper():
            # Expose the running task so stop_run() can cancel it from outside.
            self._current_task = asyncio.current_task()
            try:
                return await coro
            finally:
                self._current_task = None

        try:
            return loop.run_until_complete(_wrapper())
        except asyncio.CancelledError:
            # Task was cancelled via stop_run() — notify the UI and clean up.
            self.event_emitted.emit(StreamEvent("run_failed", {"message": "Stopped by user."}))
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if self._is_busy == busy:
            return
        self._is_busy = busy
        self.busy_changed.emit(busy)

    def _current_project_path(self) -> str:
        return normalize_project_path(Path.cwd())

    @staticmethod
    def _project_path_is_valid(project_path: str | Path | None) -> bool:
        if not project_path:
            return False
        path = Path(str(project_path))
        try:
            resolved = path.resolve()
        except Exception:
            return False
        return resolved.exists() and resolved.is_dir()

    def _create_new_session_for_project(self, project_path: str, *, with_project_label: bool = False) -> SessionSnapshot:
        checkpoint_info = self.tool_registry.checkpoint_info
        title = append_project_label(CHAT_TITLE_FALLBACK, project_path) if with_project_label else CHAT_TITLE_FALLBACK
        return self.store.new_session(
            checkpoint_backend=checkpoint_info.get("resolved_backend", self.config.checkpoint_backend),
            checkpoint_target=checkpoint_info.get("target", "unknown"),
            project_path=project_path,
            title=title,
        )

    def _sync_tool_registry_workdir(self, target_project_path: str) -> None:
        sync_fn = getattr(self.tool_registry, "sync_working_directory", None)
        if callable(sync_fn):
            sync_fn(target_project_path)

    def _set_current_session_active(self, session: SessionSnapshot, *, touch: bool = False) -> None:
        self.current_session = session
        self.current_session.approval_mode = normalize_approval_mode(self.current_session.approval_mode)
        self.store.save_active_session(self.current_session, touch=touch, set_active=True)

    def _try_change_workdir(self, target_project_path: str) -> tuple[bool, str]:
        normalized_target = normalize_project_path(target_project_path)
        if normalize_project_path(Path.cwd()) == normalized_target:
            return True, ""
        try:
            os.chdir(normalized_target)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""

    def _fallback_to_current_project_session(
        self,
        *,
        event_type: str,
        reason: str,
        notice_message: str,
        target_session_id: str = "",
        target_project_path: str = "",
        error: str = "",
    ) -> SessionSnapshot:
        fallback_project = self._current_project_path()
        fallback_session = self._create_new_session_for_project(fallback_project, with_project_label=True)
        self._set_current_session_active(fallback_session, touch=False)
        self._sync_tool_registry_workdir(fallback_project)

        payload: dict[str, Any] = {
            "reason": reason,
            "fallback_project_path": fallback_project,
        }
        if target_session_id:
            payload["target_session_id"] = target_session_id
        if target_project_path:
            payload["target_project_path"] = target_project_path
        if error:
            payload["error"] = error
            payload["error_type"] = error.split(":", 1)[0].strip() or "RuntimeError"
        self._log_ui_run_event(event_type, **payload)
        self.event_emitted.emit(StreamEvent("summary_notice", {"message": notice_message, "kind": "session_fallback"}))
        return fallback_session

    def _activate_session_with_workdir_or_fallback(
        self,
        target: SessionSnapshot,
        *,
        fallback_event_type: str,
        notice_message: str,
    ) -> SessionSnapshot:
        target_project_path = normalize_project_path(target.project_path)
        if not self._project_path_is_valid(target_project_path):
            return self._fallback_to_current_project_session(
                event_type=fallback_event_type,
                reason="invalid_project_path",
                notice_message=notice_message,
                target_session_id=target.session_id,
                target_project_path=target_project_path,
            )

        changed, error = self._try_change_workdir(target_project_path)
        if not changed:
            return self._fallback_to_current_project_session(
                event_type=fallback_event_type,
                reason="chdir_failed",
                notice_message=notice_message,
                target_session_id=target.session_id,
                target_project_path=target_project_path,
                error=error,
            )

        self._sync_tool_registry_workdir(target_project_path)
        self._set_current_session_active(target, touch=False)
        return self.current_session

    def _select_session_for_project(self, *, force_new_session: bool = False) -> SessionSnapshot:
        project_path = self._current_project_path()
        if force_new_session:
            return self._create_new_session_for_project(project_path, with_project_label=True)

        seen_ids: set[str] = set()
        candidates: list[SessionSnapshot] = []

        last_active = self.store.get_last_active_session()
        if last_active is not None and last_active.session_id not in seen_ids:
            candidates.append(last_active)
            seen_ids.add(last_active.session_id)

        project_active = self.store.get_active_session_for_project(project_path)
        if project_active is not None and project_active.session_id not in seen_ids:
            candidates.append(project_active)
            seen_ids.add(project_active.session_id)

        all_sessions = self.store.list_sessions()
        if all_sessions:
            newest = self.store.get_session(all_sessions[0].session_id)
            if newest is not None and newest.session_id not in seen_ids:
                candidates.append(newest)
                seen_ids.add(newest.session_id)

        for candidate in candidates:
            if self._project_path_is_valid(candidate.project_path):
                return candidate

        return self._create_new_session_for_project(project_path, with_project_label=True)

    def _maybe_set_session_title(self, user_text: str) -> bool:
        if not self.current_session:
            return False
        current_title = str(self.current_session.title or "").strip()
        project_default_title = append_project_label(CHAT_TITLE_FALLBACK, self.current_session.project_path)
        if current_title and current_title not in {CHAT_TITLE_FALLBACK, project_default_title}:
            return False
        title = generate_chat_title(user_text)
        if current_title == project_default_title:
            title = append_project_label(title, self.current_session.project_path)
        if title == self.current_session.title:
            return False
        self.current_session.title = title
        self.store.save_active_session(self.current_session, touch=False, set_active=True)
        return True

    async def _emit_session_payload(self, *, include_transcript: bool) -> dict[str, Any]:
        payload = await build_ui_payload(
            self.config,
            self.tool_registry,
            self.store,
            self.current_session,
            model_profiles=self.model_profiles,
            model_capabilities=self.model_capabilities,
            agent_app=self.agent_app,
            include_transcript=include_transcript,
        )
        self.session_changed.emit(payload)
        return payload

    async def _repair_current_session_if_needed(self) -> list[str]:
        notices = await repair_session_if_needed(
            self.agent_app,
            self.current_session.thread_id,
            notifier=lambda message: self.event_emitted.emit(
                StreamEvent("summary_notice", {"message": message, "kind": "session_repair"})
            ),
            event_logger=lambda event_type, payload: self._log_ui_run_event(event_type, **payload),
        )
        return notices

    def _log_ui_run_event(self, event_type: str, **payload: Any) -> None:
        if not self.ui_run_logger or not self.current_session:
            return
        normalized_payload = dict(payload)
        normalized_payload.pop("session_id", None)
        try:
            self.ui_run_logger.log_event(self.current_session.session_id, event_type, **normalized_payload)
        except Exception:
            logger.debug("Failed to write UI run event '%s'", event_type, exc_info=True)

    def _emit_cli_output_event(self, payload: dict[str, Any]) -> None:
        tool_id = str((payload or {}).get("tool_id", "")).strip()
        data = str((payload or {}).get("data", ""))
        if not tool_id or not data:
            return
        stream = str((payload or {}).get("stream", "stdout") or "stdout")
        self._emit_stream_event(
            StreamEvent(
                "cli_output",
                {
                    "tool_id": tool_id,
                    "data": data,
                    "stream": stream,
                },
            )
        )

    def _configure_cli_output_bridge(self) -> None:
        try:
            from tools import local_shell

            local_shell.set_cli_output_emitter(self._emit_cli_output_event)
        except Exception:
            logger.debug("Failed to set cli_output bridge.", exc_info=True)

    @staticmethod
    def _clear_cli_output_bridge() -> None:
        try:
            from tools import local_shell

            local_shell.set_cli_output_emitter(None)
        except Exception:
            logger.debug("Failed to clear cli_output bridge.", exc_info=True)

    def _profile_config_path(self) -> Path:
        return BASE_DIR / ".agent_state" / "config.json"

    @staticmethod
    def _profile_bootstrap_env_from_config(config: AgentConfig) -> dict[str, str]:
        provider = str(config.provider or "").strip().lower()
        openai_key = config.openai_api_key.get_secret_value() if config.openai_api_key else ""
        gemini_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else ""
        model = config.openai_model if provider == "openai" else config.gemini_model
        api_key = openai_key if provider == "openai" else gemini_key

        return {
            "PROVIDER": provider,
            "MODEL": str(model or ""),
            "API_KEY": str(api_key or ""),
            "BASE_URL": str(config.openai_base_url or ""),
            # Legacy-compatible keys, still supported by bootstrap logic.
            "OPENAI_MODEL": str(config.openai_model or ""),
            "OPENAI_API_KEY": str(openai_key or ""),
            "OPENAI_BASE_URL": str(config.openai_base_url or ""),
            "GEMINI_MODEL": str(config.gemini_model or ""),
            "GEMINI_API_KEY": str(gemini_key or ""),
        }

    @staticmethod
    def _config_overrides_for_profile(profile: dict[str, str]) -> dict[str, Any]:
        provider = str(profile.get("provider") or "").strip().lower()
        model_name = str(profile.get("model") or "").strip()
        api_key = str(profile.get("api_key") or "").strip()
        base_url = str(profile.get("base_url") or "").strip()

        overrides: dict[str, Any] = {"provider": provider}
        if provider == "openai":
            overrides["openai_model"] = model_name
            overrides["openai_api_key"] = SecretStr(api_key) if api_key else None
            overrides["openai_base_url"] = base_url or None
        else:
            overrides["gemini_model"] = model_name
            overrides["gemini_api_key"] = SecretStr(api_key) if api_key else None
        return overrides

    def _build_config_for_active_profile(self, model_profiles: dict[str, Any]) -> AgentConfig:
        base = self.base_config or self.config or AgentConfig()
        base_values = base.model_dump(mode="python")
        active_profile = find_active_profile(model_profiles)
        if active_profile is None:
            return base
        overrides = self._config_overrides_for_profile(active_profile)
        base_values.update(overrides)
        return AgentConfig(**base_values)

    @staticmethod
    def _selected_profile_id(model_profiles: dict[str, Any]) -> str:
        active = find_active_profile(model_profiles)
        return str(active.get("id") or "").strip() if active else ""

    def _set_effective_model_capabilities(
        self,
        runtime_capabilities: Any,
        *,
        model_profiles: dict[str, Any] | None = None,
    ) -> None:
        self.runtime_model_capabilities = normalize_model_capabilities(runtime_capabilities)
        active_profile = find_active_profile(model_profiles or self.model_profiles or {})
        self.model_capabilities = resolve_model_capabilities(active_profile, self.runtime_model_capabilities)

    async def _rebuild_runtime_for_active_profile(self, model_profiles: dict[str, Any]) -> None:
        active_profile = find_active_profile(model_profiles)
        if active_profile is None:
            return

        new_config = self._build_config_for_active_profile(model_profiles)
        new_agent_app, new_tool_registry = await build_agent_app(new_config)

        old_tool_registry = self.tool_registry
        self._clear_cli_output_bridge()
        self.config = new_config
        self.agent_app = new_agent_app
        self.tool_registry = new_tool_registry
        self._set_effective_model_capabilities(
            getattr(new_tool_registry, "model_capabilities", None)
        )
        self._configure_cli_output_bridge()

        if self.current_session is not None:
            checkpoint_info = getattr(self.tool_registry, "checkpoint_info", {}) or {}
            self.current_session.checkpoint_backend = checkpoint_info.get(
                "resolved_backend",
                self.current_session.checkpoint_backend,
            )
            self.current_session.checkpoint_target = checkpoint_info.get(
                "target",
                self.current_session.checkpoint_target,
            )
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            self._sync_tool_registry_workdir(self.current_session.project_path)

        if old_tool_registry is not None:
            await close_runtime_resources(old_tool_registry)

    async def _apply_model_profiles(
        self,
        candidate_payload: dict[str, Any],
        *,
        success_notice_kind: str,
        success_notice_message: str,
        sync_runtime: bool = False,
        runtime_failure_kind: str = "model_switch_failed",
        runtime_failure_message_prefix: str = "Не удалось применить выбранную модель",
    ) -> bool:
        if self.profile_store is None:
            raise RuntimeError("Model profile store is not initialized.")

        normalized_target = self.profile_store.save(candidate_payload)
        self.model_profiles = normalized_target
        self._set_effective_model_capabilities(
            self.runtime_model_capabilities,
            model_profiles=normalized_target,
        )
        if sync_runtime and self._selected_profile_id(normalized_target):
            try:
                await self._rebuild_runtime_for_active_profile(normalized_target)
            except Exception as exc:
                self.event_emitted.emit(
                    StreamEvent(
                        "summary_notice",
                        {
                            "message": f"{runtime_failure_message_prefix}: {exc}",
                            "kind": runtime_failure_kind,
                        },
                    )
                )
                await self._emit_session_payload(include_transcript=False)
                return False
            self._runtime_profile_id = self._selected_profile_id(normalized_target)
        self.event_emitted.emit(
            StreamEvent(
                "summary_notice",
                {
                    "message": success_notice_message,
                    "kind": success_notice_kind,
                },
            )
        )
        await self._emit_session_payload(include_transcript=False)
        return True

    async def _ensure_runtime_matches_selected_profile(self) -> bool:
        target_profile_id = self._selected_profile_id(self.model_profiles)
        if not target_profile_id:
            return False
        if target_profile_id == self._runtime_profile_id:
            return True

        try:
            await self._rebuild_runtime_for_active_profile(self.model_profiles)
        except Exception as exc:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": f"Не удалось применить выбранную модель: {exc}",
                        "kind": "model_switch_failed",
                    },
                )
            )
            return False

        self._runtime_profile_id = target_profile_id
        await self._emit_session_payload(include_transcript=False)
        return True

    def _emit_stream_event(self, event: StreamEvent) -> None:
        self.event_emitted.emit(event)
        if event.type in {"tool_args_missing"}:
            self._log_ui_run_event(
                event.type,
                thread_id=getattr(self.current_session, "thread_id", ""),
                **dict(event.payload or {}),
            )

    @Slot()
    def initialize(self) -> None:
        try:
            self._run(self._initialize_async())
        except Exception as exc:
            self.initialization_failed.emit(str(exc))

    @Slot(bool)
    def reinitialize(self, force_new_session: bool = False) -> None:
        if self._loop is None:
            try:
                self._run(self._initialize_async(force_new_session=force_new_session))
            except Exception as exc:
                self.initialization_failed.emit(str(exc))
            return

        try:
            self._run(self._shutdown_async())
            self._run(self._initialize_async(force_new_session=force_new_session))
            self.event_emitted.emit(StreamEvent("chat_reset", {}))
        except Exception as exc:
            logger.exception("Reinitialization failed:")
            self.initialization_failed.emit(str(exc))

    async def _initialize_async(self, force_new_session: bool = False) -> None:
        self.base_config = setup_runtime()
        self.config = self.base_config
        self.profile_store = ModelProfileStore(self._profile_config_path())
        self.model_profiles = self.profile_store.load_or_initialize(
            self._profile_bootstrap_env_from_config(self.base_config)
        )
        try:
            self.config = self._build_config_for_active_profile(self.model_profiles)
        except Exception:
            sanitized = dict(self.model_profiles)
            sanitized["active_profile"] = None
            self.model_profiles = self.profile_store.save(sanitized)
            self.config = self.base_config
        self._runtime_profile_id = self._selected_profile_id(self.model_profiles)
        self.ui_run_logger = JsonlRunLogger(self.config.run_log_dir)
        self.store = SessionStore(self.config.session_state_path)
        self.agent_app, self.tool_registry = await build_agent_app(self.config)
        self._set_effective_model_capabilities(
            getattr(self.tool_registry, "model_capabilities", None)
        )
        self._configure_cli_output_bridge()
        selected_session = self._select_session_for_project(force_new_session=force_new_session)
        self.current_session = self._activate_session_with_workdir_or_fallback(
            selected_session,
            fallback_event_type="session_restore_fallback",
            notice_message=(
                "Не удалось открыть рабочую папку восстановленного чата. "
                "Создан новый чат в текущем проекте."
            ),
        )

        await self._repair_current_session_if_needed()

        payload = await build_ui_payload(
            self.config,
            self.tool_registry,
            self.store,
            self.current_session,
            model_profiles=self.model_profiles,
            model_capabilities=self.model_capabilities,
            agent_app=self.agent_app,
            include_transcript=True,
        )
        self.initialized.emit(payload)
        self.session_changed.emit(payload)

    @Slot(object)
    def start_run(self, user_text: object) -> None:
        request_payload = normalize_request_payload(user_text)
        if not request_has_content(request_payload) or self._is_busy or self._awaiting_approval or not self.current_session:
            return
        if find_active_profile(self.model_profiles) is None:
            profiles_exist = bool((self.model_profiles or {}).get("profiles"))
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": (
                            "No enabled models available. Open Settings and enable a profile."
                            if profiles_exist
                            else "No models configured. Open Settings and add a profile."
                        ),
                        "kind": "model_missing",
                    },
                )
            )
            return
        if request_payload["attachments"] and not bool(self.model_capabilities.get("image_input_supported")):
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Current model does not accept image input. Switch to an image-capable model or remove the images.",
                        "kind": "image_input_unsupported",
                        "level": "warning",
                    },
                )
            )
            return
        if not self._run(self._ensure_runtime_matches_selected_profile()):
            return
        if self._maybe_set_session_title(request_user_text(request_payload)):
            self._run(self._emit_session_payload(include_transcript=False))
        self._set_busy(True)
        try:
            self._run(self._start_run_async(request_payload))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _start_run_async(self, user_text: object) -> None:
        request_payload = normalize_request_payload(user_text)
        self._active_run_elapsed_seconds = 0.0
        self._active_request_has_images = bool(request_payload["attachments"])
        repair_notices = await self._repair_current_session_if_needed()
        if repair_notices:
            self._log_ui_run_event(
                "pre_run_session_repair",
                repaired_count=len(repair_notices),
            )
        self.event_emitted.emit(
            StreamEvent(
                "run_started",
                {
                    "text": request_payload["text"],
                    "attachments": request_payload["attachments"],
                },
            )
        )
        await self._run_graph_payload(build_initial_state(request_payload, session_id=self.current_session.session_id))

    async def _run_graph_payload(self, payload: dict | Command) -> None:
        try:
            while True:
                stream = self.agent_app.astream(
                    payload,
                    config=build_graph_config(self.current_session.thread_id, self.config.max_loops),
                    stream_mode=["messages", "updates"],
                    version="v2",  # required for stable StreamPart dict format and __interrupt__ behaviour
                )
                processor = StreamProcessor(
                    self._emit_stream_event,
                    text_max_chars=self.config.stream_text_max_chars,
                    events_max=self.config.stream_events_max,
                    tool_buffer_max=self.config.stream_tool_buffer_max,
                    base_elapsed_seconds=self._active_run_elapsed_seconds,
                )
                result = await processor.process_stream(stream)
                self._active_run_elapsed_seconds = result.elapsed_seconds

                if result.cancelled:
                    self._log_ui_run_event(
                        "run_cancelled",
                        interrupted_tool_count=len(result.cancelled_tools),
                    )
                    await self._repair_current_session_if_needed()
                    self._active_run_elapsed_seconds = 0.0
                    self._active_request_has_images = False
                    self._set_busy(False)
                    return

                if result.interrupt is None:
                    self.store.save_active_session(self.current_session, touch=True, set_active=True)
                    await self._emit_session_payload(include_transcript=False)
                    self._active_run_elapsed_seconds = 0.0
                    self._active_request_has_images = False
                    self._set_busy(False)
                    return

                interrupt_kind = str((result.interrupt or {}).get("kind", "") or "")
                if interrupt_kind == "user_choice":
                    self._awaiting_approval = True
                    self._awaiting_interrupt_kind = "user_choice"
                    self.user_choice_requested.emit(build_user_choice_payload(result.interrupt))
                    self._set_busy(False)
                    return

                approval_payload = build_approval_payload(result.interrupt, self.current_session)
                if self.current_session.approval_mode == APPROVAL_MODE_ALWAYS:
                    self.event_emitted.emit(
                        StreamEvent("approval_resolved", {"approved": True, "always": True, "auto": True})
                    )
                    payload = Command(resume={"approved": True})
                    continue

                self._awaiting_approval = True
                self._awaiting_interrupt_kind = interrupt_kind or "tool_approval"
                self.approval_requested.emit(approval_payload)
                self._set_busy(False)
                return
        except Exception as exc:
            self._active_run_elapsed_seconds = 0.0
            if self._active_request_has_images:
                self.event_emitted.emit(
                    StreamEvent(
                        "summary_notice",
                        {
                            "message": (
                                "Модель не приняла прикреплённое изображение. "
                                "Проверьте чекбокс поддержки изображений у профиля "
                                "или отправьте запрос без картинки."
                            ),
                            "kind": "image_input_failed",
                            "level": "warning",
                        },
                    )
                )
            self._active_request_has_images = False
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    @Slot(bool, bool)
    def resume_approval(self, approved: bool, always: bool = False) -> None:
        if (
            not self.current_session
            or not self._awaiting_approval
            or self._awaiting_interrupt_kind == "user_choice"
        ):
            return
        self._set_busy(True)
        try:
            self._run(self._resume_approval_async(approved, always))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _resume_approval_async(self, approved: bool, always: bool) -> None:
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        if always:
            self.current_session.approval_mode = APPROVAL_MODE_ALWAYS
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            await self._emit_session_payload(include_transcript=False)

        self.event_emitted.emit(
            StreamEvent("approval_resolved", {"approved": approved, "always": always, "auto": False})
        )
        await self._run_graph_payload(Command(resume={"approved": approved}))

    @Slot(str)
    def resume_user_choice(self, chosen: str) -> None:
        if (
            not self.current_session
            or not self._awaiting_approval
            or self._awaiting_interrupt_kind != "user_choice"
        ):
            return
        chosen_text = str(chosen or "").strip()
        if not chosen_text:
            return
        self._set_busy(True)
        try:
            self._run(self._resume_user_choice_async(chosen_text))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _resume_user_choice_async(self, chosen: str) -> None:
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        self.event_emitted.emit(StreamEvent("user_choice_resolved", {"chosen": chosen}))
        await self._run_graph_payload(Command(resume=chosen))

    @Slot()
    def stop_run(self) -> None:
        """Cancel the current run from the GUI thread via asyncio task cancellation."""
        if self._loop and self._current_task and not self._current_task.done():
            self._loop.call_soon_threadsafe(self._current_task.cancel)

    @Slot()
    def new_session(self) -> None:
        if not self.current_session or self._is_busy or self._awaiting_approval:
            return
        checkpoint_info = self.tool_registry.checkpoint_info
        self.current_session = self.store.new_session(
            checkpoint_backend=checkpoint_info.get("resolved_backend", self.config.checkpoint_backend),
            checkpoint_target=checkpoint_info.get("target", "unknown"),
            project_path=self._current_project_path(),
        )
        self.current_session.approval_mode = APPROVAL_MODE_PROMPT
        self.store.save_active_session(self.current_session, touch=False, set_active=True)
        self._sync_tool_registry_workdir(self.current_session.project_path)
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        self._run(self._emit_session_payload(include_transcript=True))

    @Slot(str)
    def switch_session(self, session_id: str) -> None:
        if not session_id or self._is_busy or self._awaiting_approval:
            return
        target = self.store.get_session(session_id)
        if target is None or target.session_id == getattr(self.current_session, "session_id", ""):
            return
        self._run(self._switch_session_async(target))

    async def _switch_session_async(self, target: SessionSnapshot) -> None:
        self.current_session = self._activate_session_with_workdir_or_fallback(
            target,
            fallback_event_type="session_switch_fallback",
            notice_message=(
                "Не удалось открыть рабочую папку выбранного чата. "
                "Создан новый чат в текущем проекте."
            ),
        )
        await self._repair_current_session_if_needed()
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        await self._emit_session_payload(include_transcript=True)

    @Slot(str)
    def delete_session(self, session_id: str) -> None:
        if not session_id or self._is_busy or self._awaiting_approval:
            return
        self._run(self._delete_session_async(session_id))

    async def _delete_session_async(self, session_id: str) -> None:
        if not self.store or not self.current_session:
            return

        deleted = self.store.delete_session(session_id)
        if not deleted:
            self.event_emitted.emit(
                StreamEvent("summary_notice", {"message": "Чат не найден в истории.", "kind": "session_delete"})
            )
            return

        active_deleted = self.current_session.session_id == session_id
        self.event_emitted.emit(
            StreamEvent("summary_notice", {"message": "Чат удалён из истории.", "kind": "session_delete"})
        )

        if not active_deleted:
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            await self._emit_session_payload(include_transcript=False)
            return

        replacement = self.store.get_last_active_session()
        if replacement is None or not self._project_path_is_valid(replacement.project_path):
            fallback_project = self._current_project_path()
            replacement = self._create_new_session_for_project(fallback_project, with_project_label=True)

        self.current_session = self._activate_session_with_workdir_or_fallback(
            replacement,
            fallback_event_type="session_delete_fallback",
            notice_message=(
                "Не удалось открыть рабочую папку следующего чата. "
                "Создан новый чат в текущем проекте."
            ),
        )

        await self._repair_current_session_if_needed()
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        await self._emit_session_payload(include_transcript=True)

    @Slot(str)
    def set_active_profile(self, profile_id: str) -> None:
        if self._is_busy or self._awaiting_approval:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Нельзя переключить модель во время выполнения.",
                        "kind": "model_switch_blocked",
                    },
                )
            )
            return
        if self.profile_store is None:
            return
        try:
            self._run(self._set_active_profile_async(profile_id))
        except Exception as exc:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": f"Не удалось переключить модель: {exc}",
                        "kind": "model_switch_failed",
                    },
                )
            )

    async def _set_active_profile_async(self, profile_id: str) -> None:
        target_id = str(profile_id or "").strip()
        current = normalize_profiles_payload(self.model_profiles)
        known_ids = {
            str(profile.get("id") or "").strip()
            for profile in current.get("profiles", []) or []
            if isinstance(profile, dict)
        }
        if target_id and target_id not in known_ids:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Выбранный профиль не найден.",
                        "kind": "model_switch_failed",
                    },
                )
            )
            return
        if target_id == str(current.get("active_profile") or ""):
            return
        candidate = dict(current)
        candidate["active_profile"] = target_id or None
        target_profile = find_active_profile(candidate)
        target_model = str((target_profile or {}).get("model") or "").strip()
        success_notice_message = (
            f"Модель переключена на {target_model}."
            if target_model
            else "Модель переключена."
        )
        await self._apply_model_profiles(
            candidate,
            success_notice_kind="model_switched",
            success_notice_message=success_notice_message,
            sync_runtime=True,
            runtime_failure_kind="model_switch_failed",
            runtime_failure_message_prefix="Не удалось применить выбранную модель",
        )

    @Slot(object)
    def save_profiles(self, config_payload: object) -> None:
        if self._is_busy or self._awaiting_approval:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Нельзя сохранить профили во время выполнения.",
                        "kind": "profiles_save_blocked",
                    },
                )
            )
            return
        if self.profile_store is None:
            return
        try:
            payload = config_payload if isinstance(config_payload, dict) else {}
            self._run(self._save_profiles_async(payload))
        except Exception as exc:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": f"Не удалось сохранить профили: {exc}",
                        "kind": "profiles_save_failed",
                    },
                )
            )

    async def _save_profiles_async(self, config_payload: dict[str, Any]) -> None:
        await self._apply_model_profiles(
            config_payload,
            success_notice_kind="profiles_saved",
            success_notice_message="Профили моделей сохранены.",
            sync_runtime=True,
            runtime_failure_kind="profiles_apply_failed",
            runtime_failure_message_prefix="Профили сохранены, но не удалось применить активную модель",
        )

    @Slot()
    def shutdown(self) -> None:
        try:
            self._run(self._shutdown_async())
        finally:
            if self._loop is not None:
                self._loop.close()
                self._loop = None
            self.shutdown_complete.emit()

    async def _shutdown_async(self) -> None:
        self._clear_cli_output_bridge()
        await close_runtime_resources(self.tool_registry)
        self.ui_run_logger = None


class AgentRuntimeController(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    user_choice_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)

    _initialize_requested = Signal()
    _start_run_requested = Signal(object)
    _stop_run_requested = Signal()
    _resume_requested = Signal(bool, bool)
    _resume_user_choice_requested = Signal(str)
    _new_session_requested = Signal()
    _switch_session_requested = Signal(str)
    _delete_session_requested = Signal(str)
    _set_active_profile_requested = Signal(str)
    _save_profiles_requested = Signal(object)
    _reinitialize_requested = Signal(bool)
    _shutdown_requested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = AgentRunWorker()
        self._worker.moveToThread(self._thread)

        self._initialize_requested.connect(self._worker.initialize)
        self._reinitialize_requested.connect(self._worker.reinitialize)
        self._start_run_requested.connect(self._worker.start_run)
        self._stop_run_requested.connect(self._worker.stop_run, Qt.DirectConnection)
        self._resume_requested.connect(self._worker.resume_approval)
        self._resume_user_choice_requested.connect(self._worker.resume_user_choice)
        self._new_session_requested.connect(self._worker.new_session)
        self._switch_session_requested.connect(self._worker.switch_session)
        self._delete_session_requested.connect(self._worker.delete_session)
        self._set_active_profile_requested.connect(self._worker.set_active_profile)
        self._save_profiles_requested.connect(self._worker.save_profiles)
        self._shutdown_requested.connect(self._worker.shutdown)

        self._worker.initialized.connect(self.initialized)
        self._worker.initialization_failed.connect(self.initialization_failed)
        self._worker.event_emitted.connect(self.event_emitted)
        self._worker.approval_requested.connect(self.approval_requested)
        self._worker.user_choice_requested.connect(self.user_choice_requested)
        self._worker.session_changed.connect(self.session_changed)
        self._worker.busy_changed.connect(self.busy_changed)
        self._worker.shutdown_complete.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)

        self._thread.start()

    def initialize(self) -> None:
        self._initialize_requested.emit()

    def reinitialize(self, force_new_session: bool = False) -> None:
        self._reinitialize_requested.emit(force_new_session)

    def start_run(self, user_text: object) -> None:
        self._start_run_requested.emit(user_text)

    def stop_run(self) -> None:
        self._stop_run_requested.emit()

    def resume_approval(self, approved: bool, always: bool = False) -> None:
        self._resume_requested.emit(approved, always)

    def resume_user_choice(self, chosen: str) -> None:
        self._resume_user_choice_requested.emit(chosen)

    def new_session(self) -> None:
        self._new_session_requested.emit()

    def switch_session(self, session_id: str) -> None:
        self._switch_session_requested.emit(session_id)

    def delete_session(self, session_id: str) -> None:
        self._delete_session_requested.emit(session_id)

    def set_active_profile(self, profile_id: str) -> None:
        self._set_active_profile_requested.emit(profile_id)

    def save_profiles(self, config_payload: dict[str, Any]) -> None:
        self._save_profiles_requested.emit(config_payload)

    def shutdown(self) -> None:
        if self._thread.isRunning():
            self._shutdown_requested.emit()
            # Give the worker time to run its shutdown slot
            import time
            time.sleep(0.1)
            # Wait for thread to finish cleanup
            self._thread.quit()
            self._thread.wait()
