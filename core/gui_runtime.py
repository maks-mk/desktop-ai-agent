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

from agent import build_agent_app
from core.config import AgentConfig
from core.constants import AGENT_VERSION
from core.logging_config import setup_logging
from core.message_utils import is_tool_message_error, stringify_content
from core.run_logger import JsonlRunLogger
from core.session_store import (
    DEFAULT_CHAT_TITLE,
    SessionListEntry,
    SessionSnapshot,
    SessionStore,
    normalize_project_path,
)
from core.session_utils import repair_session_if_needed
from core.stream_processor import StreamEvent, StreamProcessor
from core.text_utils import format_tool_display, format_tool_output, parse_thought, prepare_markdown_for_render
from core.tool_policy import ToolMetadata

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
    log_level = logging.DEBUG if config.debug else logging.WARNING
    setup_logging(level=log_level)
    return config


def build_initial_state(user_input: str, session_id: str, safety_mode: str = "default") -> dict:
    return {
        "messages": [("user", user_input)],
        "steps": 0,
        "token_usage": {},
        "current_task": user_input,
        "critic_status": "",
        "critic_source": "",
        "critic_feedback": "",
        "session_id": session_id,
        "run_id": uuid.uuid4().hex,
        "turn_id": 1,
        "pending_approval": None,
        "open_tool_issue": None,
        "has_protocol_error": False,
        "critic_retry_count": 0,
        "critic_last_retry_fingerprint": "",
        "last_tool_error": "",
        "last_tool_result": "",
        "safety_mode": safety_mode,
    }


def build_graph_config(thread_id: str, max_loops: int) -> dict:
    recursion_limit = max(12, max_loops * 6)
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
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return stringify_content(content)


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
            text = stringify_content(message.content).strip()
            if not text:
                continue
            current_turn = {"user_text": text, "blocks": []}
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
            text = _extract_ai_text(message)
            if text.strip():
                _thought, clean_text, _has_thought = parse_thought(text)
                markdown = prepare_markdown_for_render(clean_text.strip())
                if markdown:
                    current_turn["blocks"].append({"type": "assistant", "markdown": markdown})
            continue

        if isinstance(message, ToolMessage):
            tool_meta = pending_tool_calls.get(message.tool_call_id, {})
            tool_name = tool_meta.get("name") or message.name or "tool"
            tool_args = tool_meta.get("args") or {}
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
    *,
    agent_app=None,
    include_transcript: bool = False,
) -> dict[str, Any]:
    runtime_snapshot = build_runtime_snapshot(config, tool_registry, snapshot)
    payload = {
        "snapshot": runtime_snapshot,
        "tools": runtime_snapshot["tools"],
        "help_markdown": build_help_markdown(),
        "sessions": serialize_session_entries(store.list_sessions()),
        "active_session_id": snapshot.session_id,
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
    session_changed = Signal(object)
    busy_changed = Signal(bool)
    shutdown_complete = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_task: asyncio.Task | None = None
        self.config: AgentConfig | None = None
        self.store: SessionStore | None = None
        self.agent_app = None
        self.tool_registry = None
        self.current_session: SessionSnapshot | None = None
        self.ui_run_logger: JsonlRunLogger | None = None
        self._is_busy = False
        self._awaiting_approval = False

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
        self.config = setup_runtime()
        self.ui_run_logger = JsonlRunLogger(self.config.run_log_dir)
        self.store = SessionStore(self.config.session_state_path)
        self.agent_app, self.tool_registry = await build_agent_app(self.config)
        self.current_session = self._select_session_for_project(force_new_session=force_new_session)
        if not self._project_path_is_valid(self.current_session.project_path):
            fallback_project = self._current_project_path()
            self._log_ui_run_event(
                "session_restore_fallback",
                reason="invalid_project_path",
                invalid_project_path=str(self.current_session.project_path),
                fallback_project_path=fallback_project,
            )
            self.current_session = self._create_new_session_for_project(
                fallback_project,
                with_project_label=True,
            )
        target_project_path = normalize_project_path(self.current_session.project_path)
        if normalize_project_path(Path.cwd()) != target_project_path:
            os.chdir(target_project_path)
        self._sync_tool_registry_workdir(target_project_path)
        self.current_session.approval_mode = normalize_approval_mode(
            getattr(self.current_session, "approval_mode", APPROVAL_MODE_PROMPT)
        )
        self.store.save_active_session(self.current_session, touch=False, set_active=True)

        await self._repair_current_session_if_needed()

        payload = await build_ui_payload(
            self.config,
            self.tool_registry,
            self.store,
            self.current_session,
            agent_app=self.agent_app,
            include_transcript=True,
        )
        self.initialized.emit(payload)
        self.session_changed.emit(payload)

    @Slot(str)
    def start_run(self, user_text: str) -> None:
        if not user_text.strip() or self._is_busy or self._awaiting_approval or not self.current_session:
            return
        if self._maybe_set_session_title(user_text):
            self._run(self._emit_session_payload(include_transcript=False))
        self._set_busy(True)
        try:
            self._run(self._start_run_async(user_text))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _start_run_async(self, user_text: str) -> None:
        repair_notices = await self._repair_current_session_if_needed()
        if repair_notices:
            self._log_ui_run_event(
                "pre_run_session_repair",
                repaired_count=len(repair_notices),
            )
        self.event_emitted.emit(StreamEvent("run_started", {"text": user_text}))
        await self._run_graph_payload(build_initial_state(user_text, session_id=self.current_session.session_id))

    async def _run_graph_payload(self, payload: dict | Command) -> None:
        try:
            while True:
                stream = self.agent_app.astream(
                    payload,
                    config=build_graph_config(self.current_session.thread_id, self.config.max_loops),
                    stream_mode=["messages", "updates"],
                    version="v2",  # required for stable StreamPart dict format and __interrupt__ behaviour
                )
                processor = StreamProcessor(self._emit_stream_event)
                result = await processor.process_stream(stream)

                if result.interrupt is None:
                    self.store.save_active_session(self.current_session, touch=True, set_active=True)
                    await self._emit_session_payload(include_transcript=False)
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
                self.approval_requested.emit(approval_payload)
                self._set_busy(False)
                return
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    @Slot(bool, bool)
    def resume_approval(self, approved: bool, always: bool = False) -> None:
        if not self.current_session or not self._awaiting_approval:
            return
        self._set_busy(True)
        try:
            self._run(self._resume_approval_async(approved, always))
        except Exception as exc:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": str(exc)}))
            self._set_busy(False)

    async def _resume_approval_async(self, approved: bool, always: bool) -> None:
        self._awaiting_approval = False
        if always:
            self.current_session.approval_mode = APPROVAL_MODE_ALWAYS
            self.store.save_active_session(self.current_session, touch=False, set_active=True)
            await self._emit_session_payload(include_transcript=False)

        self.event_emitted.emit(
            StreamEvent("approval_resolved", {"approved": approved, "always": always, "auto": False})
        )
        await self._run_graph_payload(Command(resume={"approved": approved}))

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
        if not self._project_path_is_valid(target.project_path):
            self._log_ui_run_event(
                "session_switch_fallback",
                reason="invalid_project_path",
                invalid_project_path=str(target.project_path),
                target_session_id=target.session_id,
            )
            fallback_project = self._current_project_path()
            target = self._create_new_session_for_project(fallback_project, with_project_label=True)

        target_project_path = normalize_project_path(target.project_path)
        if normalize_project_path(Path.cwd()) != target_project_path:
            os.chdir(target_project_path)
        self._sync_tool_registry_workdir(target_project_path)
        self.current_session = target
        self.current_session.approval_mode = normalize_approval_mode(self.current_session.approval_mode)
        self.store.save_active_session(self.current_session, touch=False, set_active=True)
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

        target_project_path = normalize_project_path(replacement.project_path)
        if normalize_project_path(Path.cwd()) != target_project_path:
            os.chdir(target_project_path)
        self._sync_tool_registry_workdir(target_project_path)

        self.current_session = replacement
        self.current_session.approval_mode = normalize_approval_mode(self.current_session.approval_mode)
        self.store.save_active_session(self.current_session, touch=False, set_active=True)

        await self._repair_current_session_if_needed()
        self.event_emitted.emit(StreamEvent("chat_reset", {}))
        await self._emit_session_payload(include_transcript=True)

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
        await close_runtime_resources(self.tool_registry)
        self.ui_run_logger = None


class AgentRuntimeController(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)

    _initialize_requested = Signal()
    _start_run_requested = Signal(str)
    _stop_run_requested = Signal()
    _resume_requested = Signal(bool, bool)
    _new_session_requested = Signal()
    _switch_session_requested = Signal(str)
    _delete_session_requested = Signal(str)
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
        self._new_session_requested.connect(self._worker.new_session)
        self._switch_session_requested.connect(self._worker.switch_session)
        self._delete_session_requested.connect(self._worker.delete_session)
        self._shutdown_requested.connect(self._worker.shutdown)

        self._worker.initialized.connect(self.initialized)
        self._worker.initialization_failed.connect(self.initialization_failed)
        self._worker.event_emitted.connect(self.event_emitted)
        self._worker.approval_requested.connect(self.approval_requested)
        self._worker.session_changed.connect(self.session_changed)
        self._worker.busy_changed.connect(self.busy_changed)
        self._worker.shutdown_complete.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)

        self._thread.start()

    def initialize(self) -> None:
        self._initialize_requested.emit()

    def reinitialize(self, force_new_session: bool = False) -> None:
        self._reinitialize_requested.emit(force_new_session)

    def start_run(self, user_text: str) -> None:
        self._start_run_requested.emit(user_text)

    def stop_run(self) -> None:
        self._stop_run_requested.emit()

    def resume_approval(self, approved: bool, always: bool = False) -> None:
        self._resume_requested.emit(approved, always)

    def new_session(self) -> None:
        self._new_session_requested.emit()

    def switch_session(self, session_id: str) -> None:
        self._switch_session_requested.emit(session_id)

    def delete_session(self, session_id: str) -> None:
        self._delete_session_requested.emit(session_id)

    def shutdown(self) -> None:
        if self._thread.isRunning():
            self._shutdown_requested.emit()
            # Give the worker time to run its shutdown slot
            import time
            time.sleep(0.1)
            # Wait for thread to finish cleanup
            self._thread.quit()
            self._thread.wait()
