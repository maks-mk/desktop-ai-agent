from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.config import AgentConfig
from core.logging_config import setup_logging
from core.model_profiles import ModelProfileStore, find_active_profile, normalize_profiles_payload
from core.multimodal import (
    DEFAULT_MODEL_CAPABILITIES,
    build_user_message_content,
    normalize_request_payload,
    request_has_content,
    request_task_text,
    request_user_text,
)
from core.run_logger import JsonlRunLogger
from core.session_store import SessionSnapshot, SessionStore
from ui.runtime_payloads import (
    APPROVAL_MODE_ALWAYS,
    APPROVAL_MODE_PROMPT,
    build_approval_payload,
    build_ui_payload,
    build_user_choice_payload,
    close_runtime_resources,
    normalize_approval_mode,
)
from ui.runtime_session import RuntimeSessionCoordinator
from ui.streaming import StreamEvent, StreamProcessor

logger = logging.getLogger("agent")


def _runtime_module():
    from ui import runtime as runtime_module

    return runtime_module


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
        "turn_mode": "chat",
        "requires_evidence": False,
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
    graph_supersteps_per_logical_step = 6
    graph_superstep_overhead = 8
    recursion_limit = max(
        16,
        int(max_loops or 0) * graph_supersteps_per_logical_step + graph_superstep_overhead,
    )
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


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
        self.checkpoint_runtime = None
        self.current_session: SessionSnapshot | None = None
        self.ui_run_logger: JsonlRunLogger | None = None
        self._is_busy = False
        self._awaiting_approval = False
        self._awaiting_interrupt_kind = ""
        self._active_run_elapsed_seconds = 0.0
        self._active_request_has_images = False
        self._coordinator = RuntimeSessionCoordinator(self)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def _run(self, coro):
        loop = self._ensure_loop()

        async def _wrapper():
            self._current_task = asyncio.current_task()
            try:
                return await coro
            finally:
                self._current_task = None

        try:
            return loop.run_until_complete(_wrapper())
        except asyncio.CancelledError:
            self.event_emitted.emit(StreamEvent("run_failed", {"message": "Stopped by user."}))
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if self._is_busy == busy:
            return
        self._is_busy = busy
        self.busy_changed.emit(busy)

    def _current_project_path(self) -> str:
        return self._coordinator.current_project_path()

    @staticmethod
    def _project_path_is_valid(project_path) -> bool:
        return RuntimeSessionCoordinator.project_path_is_valid(project_path)

    def _create_new_session_for_project(self, project_path: str, *, with_project_label: bool = False):
        return self._coordinator.create_new_session_for_project(project_path, with_project_label=with_project_label)

    def _sync_tool_registry_workdir(self, target_project_path: str) -> None:
        self._coordinator.sync_tool_registry_workdir(target_project_path)

    def _set_current_session_active(self, session, *, touch: bool = False) -> None:
        self._coordinator.set_current_session_active(session, touch=touch)

    def _ensure_current_session_persisted(self) -> bool:
        return self._coordinator.ensure_current_session_persisted()

    def _try_change_workdir(self, target_project_path: str) -> tuple[bool, str]:
        return self._coordinator.try_change_workdir(target_project_path)

    def _fallback_to_current_project_session(self, **kwargs):
        return self._coordinator.fallback_to_current_project_session(**kwargs)

    def _activate_session_with_workdir_or_fallback(self, target, **kwargs):
        return self._coordinator.activate_session_with_workdir_or_fallback(target, **kwargs)

    def _select_session_for_project(self, *, force_new_session: bool = False):
        return self._coordinator.select_session_for_project(force_new_session=force_new_session)

    def _maybe_set_session_title(self, user_text: str) -> bool:
        return self._coordinator.maybe_set_session_title(user_text)

    async def _emit_session_payload(self, *, include_transcript: bool) -> dict[str, Any]:
        return await self._coordinator.emit_session_payload(include_transcript=include_transcript)

    async def _repair_current_session_if_needed(self) -> list[str]:
        return await self._coordinator.repair_current_session_if_needed()

    @staticmethod
    def _profile_config_path():
        return RuntimeSessionCoordinator.profile_config_path()

    @staticmethod
    def _profile_bootstrap_env_from_config(config: AgentConfig) -> dict[str, str]:
        return RuntimeSessionCoordinator.profile_bootstrap_env_from_config(config)

    @staticmethod
    def _config_overrides_for_profile(profile: dict[str, str]) -> dict[str, Any]:
        return RuntimeSessionCoordinator.config_overrides_for_profile(profile)

    def _build_config_for_active_profile(self, model_profiles: dict[str, Any]):
        return self._coordinator.build_config_for_active_profile(model_profiles)

    @staticmethod
    def _selected_profile_id(model_profiles: dict[str, Any]) -> str:
        return RuntimeSessionCoordinator.selected_profile_id(model_profiles)

    def _set_effective_model_capabilities(self, runtime_capabilities: Any, *, model_profiles: dict[str, Any] | None = None) -> None:
        self._coordinator.set_effective_model_capabilities(runtime_capabilities, model_profiles=model_profiles)

    async def _rebuild_runtime_for_active_profile(self, model_profiles: dict[str, Any]) -> None:
        await self._coordinator.rebuild_runtime_for_active_profile(model_profiles)

    async def _apply_model_profiles(self, candidate_payload: dict[str, Any], **kwargs) -> bool:
        return await self._coordinator.apply_model_profiles(candidate_payload, **kwargs)

    async def _ensure_runtime_matches_selected_profile(self) -> bool:
        return await self._coordinator.ensure_runtime_matches_selected_profile()

    def _normalize_approval_mode(self, value: str | None) -> str:
        return normalize_approval_mode(value)

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
        self._emit_stream_event(StreamEvent("cli_output", {"tool_id": tool_id, "data": data, "stream": stream}))

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
        self.base_config = _runtime_module().setup_runtime()
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
        self.agent_app, self.tool_registry = await _runtime_module().build_agent_app(self.config)
        self.checkpoint_runtime = getattr(self.tool_registry, "checkpoint_runtime", None)
        self._set_effective_model_capabilities(getattr(self.tool_registry, "model_capabilities", None))
        self._configure_cli_output_bridge()
        selected_session = self._select_session_for_project(force_new_session=force_new_session)
        self.current_session = self._activate_session_with_workdir_or_fallback(
            selected_session,
            fallback_event_type="session_restore_fallback",
            notice_message=(
                "Could not open the restored chat workspace. "
                "Created a new chat in the current project."
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
        persisted_now = self._ensure_current_session_persisted()
        title_changed = self._maybe_set_session_title(request_user_text(request_payload))
        if persisted_now or title_changed:
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
            self._log_ui_run_event("pre_run_session_repair", repaired_count=len(repair_notices))
        self.event_emitted.emit(
            StreamEvent(
                "run_started",
                {"text": request_payload["text"], "attachments": request_payload["attachments"]},
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
                    version="v2",
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
                    self._log_ui_run_event("run_cancelled", interrupted_tool_count=len(result.cancelled_tools))
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
                                "The model rejected the attached image. "
                                "Check the profile image-input setting or send the request without an image."
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
        if not self.current_session or not self._awaiting_approval or self._awaiting_interrupt_kind == "user_choice":
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
        if not self.current_session or not self._awaiting_approval or self._awaiting_interrupt_kind != "user_choice":
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
            persisted=False,
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
                "Could not open the selected chat workspace. "
                "Created a new chat in the current project."
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
                StreamEvent("summary_notice", {"message": "Chat not found in history.", "kind": "session_delete"})
            )
            return

        active_deleted = self.current_session.session_id == session_id
        self.event_emitted.emit(
            StreamEvent("summary_notice", {"message": "Chat deleted from history.", "kind": "session_delete"})
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
                "Could not open the next chat workspace. "
                "Created a new chat in the current project."
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
                        "message": "Cannot switch models while a run is active.",
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
                        "message": f"Failed to switch models: {exc}",
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
                        "message": "The selected profile was not found.",
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
            runtime_failure_message_prefix="Failed to apply the selected model",
        )

    @Slot(object)
    def save_profiles(self, config_payload: object) -> None:
        if self._is_busy or self._awaiting_approval:
            self.event_emitted.emit(
                StreamEvent(
                    "summary_notice",
                    {
                        "message": "Cannot save profiles while a run is active.",
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
                        "message": f"Failed to save profiles: {exc}",
                        "kind": "profiles_save_failed",
                    },
                )
            )

    async def _save_profiles_async(self, config_payload: dict[str, Any]) -> None:
        await self._apply_model_profiles(
            config_payload,
            success_notice_kind="profiles_saved",
            success_notice_message="Model profiles saved.",
            sync_runtime=True,
            runtime_failure_kind="profiles_apply_failed",
            runtime_failure_message_prefix="Profiles were saved, but the active model could not be applied",
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
        self.checkpoint_runtime = None
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
            import time

            time.sleep(0.1)
            self._thread.quit()
            self._thread.wait()
