from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMenuBar, QMessageBox, QVBoxLayout, QWidget

from core.constants import AGENT_VERSION
from core.input_sanitizer import build_user_input_notice, sanitize_user_text
from core.multimodal import DEFAULT_MODEL_CAPABILITIES, can_read_image_file, resolve_model_capabilities
from core.model_profiles import normalize_profiles_payload
from ui.main_window_state import ComposerStateController, RunStatusController, StreamEventRouter
from ui.runtime import AgentRuntimeController
from ui.theme import ACCENT_BLUE, build_stylesheet
from ui.widgets import ModelSettingsDialog, _fa_icon
from ui.window_components.inspector_controller import InspectorController
from ui.window_components.menu_builder import MenuBuilder
from ui.window_components.sidebar_controller import SidebarController
from ui.window_components.status_bar_manager import StatusBarManager
from ui.window_components.workspace_builder import WorkspaceBuilder

logger = logging.getLogger("agent")


def _configure_qt_logging() -> None:
    rule = "qt.text.font.db=false"
    current = str(os.environ.get("QT_LOGGING_RULES") or "").strip()
    if rule in {item.strip() for item in current.split(";") if item.strip()}:
        return
    os.environ["QT_LOGGING_RULES"] = f"{current};{rule}" if current else rule


class MainWindow(QMainWindow):
    def __init__(self, controller: AgentRuntimeController | None = None, *, auto_initialize: bool = True) -> None:
        super().__init__()
        self.controller = controller or AgentRuntimeController(self)
        self.current_turn = None
        self.current_snapshot: dict | None = None
        self.active_session_id = ""
        self._model_settings_window = None
        self.awaiting_approval = False
        self.awaiting_user_choice = False
        self.is_busy = False
        self.sidebar_collapsed = False
        self._sidebar_width = 330
        self.inspector_collapsed = False
        self._inspector_width = 400
        self._custom_choice_armed = False
        self._tools_hash: int = 0
        self._summarize_in_progress = False
        self.model_profiles_payload: dict = {"active_profile": None, "profiles": []}
        self.model_capabilities: dict = dict(DEFAULT_MODEL_CAPABILITIES)
        self.draft_image_attachments: list[dict] = []
        self._has_active_model = False
        self._primary_status_label = "Initializing runtime…"
        self._status_message_ticket = 0
        self._composer_min_height = 56
        self._composer_max_height = 56
        self._composer_height_padding = 16
        self._composer_growth_lines = 5
        self._composer_height_sync_pending = False
        self._run_start_time: float | None = None
        self._current_status_label = ""
        self._current_status_phase = "working"
        self._last_rendered_elapsed_text = ""
        self._event_router: StreamEventRouter | None = None

        self._menu_builder = MenuBuilder(self)
        self._workspace_builder = WorkspaceBuilder(self)
        self._status_bar_manager = StatusBarManager(self)
        self._sidebar_controller = SidebarController(self)
        self._inspector_controller = InspectorController(self)
        self._realtime_timer = self._status_bar_manager.realtime_timer
        self._composer_state = ComposerStateController(self)
        self._status_controller = RunStatusController(self)

        self._build_ui()
        self._connect_signals()
        self._build_event_dispatch()
        if auto_initialize:
            self.controller.initialize()

    def _update_realtime_elapsed(self) -> None:
        self._status_controller.update_realtime_elapsed()

    def _build_ui(self) -> None:
        self.setWindowTitle(f"AI Agent {AGENT_VERSION}")
        self.resize(1300, 670)
        self.setMinimumSize(900, 600)
        self.setWindowIcon(_fa_icon("fa5s.robot", color=ACCENT_BLUE, size=16))

        status_refs = self._status_bar_manager.build()
        self.runtime_meta_label = status_refs.runtime_meta_label
        self.setStatusBar(status_refs.status_bar)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._build_menu_bar()
        self.workspace = self._build_workspace()
        root.addWidget(self.workspace, 1)

        self.setCentralWidget(central)
        self.setStyleSheet(build_stylesheet())
        self.inspector_container.hide()
        self.inspector_collapsed = True
        self._set_input_enabled(False)

    def _build_menu_bar(self) -> None:
        refs = self._menu_builder.build()
        self.toggle_sidebar_action = refs.toggle_sidebar_action
        self.new_session_action = refs.new_session_action
        self.settings_action = refs.settings_action
        self.info_action = refs.info_action
        self.quit_action = refs.quit_action
        self.status_icon = refs.status_icon
        self.status_text = refs.status_text
        self.status_line_label = refs.status_text
        self.status_meta = refs.status_meta
        self.top_status_chip = refs.top_status_chip
        self.new_project_button = refs.new_project_button
        self.sidebar_toggle_button = refs.sidebar_toggle_button
        self.new_session_button = refs.new_session_button
        self.settings_button = refs.settings_button
        self.info_button = refs.info_button
        self.setMenuWidget(refs.menu_widget)
        self._set_status_visual("Initializing runtime…", busy=True)

    def _build_workspace(self) -> QWidget:
        refs = self._workspace_builder.build()
        self.splitter = refs.splitter
        self.sidebar_container = refs.sidebar_container
        self.sidebar = refs.sidebar
        self.transcript = refs.transcript
        self.composer_shell = refs.composer_shell
        self.composer_container = refs.composer_container
        self.approval_card = refs.approval_card
        self.user_choice_card = refs.user_choice_card
        self.composer_pill = refs.composer_pill
        self.composer_attachments_strip = refs.composer_attachments_strip
        self.composer_notice_label = refs.composer_notice_label
        self.composer = refs.composer
        self.attach_button = refs.attach_button
        self.attach_menu = refs.attach_menu
        self.add_image_action = refs.add_image_action
        self.insert_file_path_action = refs.insert_file_path_action
        self.model_chip = refs.model_chip
        self.model_chip_menu = refs.model_chip_menu
        self.model_chip_group = refs.model_chip_group
        self.model_image_badge = refs.model_image_badge
        self.no_models_label = refs.no_models_label
        self.open_settings_inline_button = refs.open_settings_inline_button
        self.summary_progress_ring = refs.summary_progress_ring
        self.send_button = refs.send_button
        self.stop_action_button = refs.stop_action_button
        self.inspector_container = refs.inspector_container
        self.inspector_panel = refs.inspector_panel
        self.overview_panel = refs.overview_panel
        self.tools_panel = refs.tools_panel
        self.help_text = refs.help_text
        return refs.workspace

    def _connect_signals(self) -> None:
        self.toggle_sidebar_action.triggered.connect(lambda _checked=False: self._toggle_sidebar())
        self.new_session_action.triggered.connect(lambda _checked=False: self._new_session())
        self.settings_action.triggered.connect(lambda _checked=False: self._open_settings_dialog())
        self.info_action.triggered.connect(lambda _checked=False: self._toggle_info_popup())
        self.open_settings_inline_button.clicked.connect(lambda _checked=False: self._open_settings_dialog())
        self.quit_action.triggered.connect(lambda _checked=False: self.close())
        self.stop_action_button.clicked.connect(lambda _checked=False: self.controller.stop_run())
        self.new_project_button.clicked.connect(lambda _checked=False: self._open_new_project())
        self.sidebar_toggle_button.clicked.connect(lambda _checked=False: self.toggle_sidebar_action.trigger())
        self.new_session_button.clicked.connect(lambda _checked=False: self.new_session_action.trigger())
        self.settings_button.clicked.connect(lambda _checked=False: self.settings_action.trigger())
        self.info_button.clicked.connect(lambda _checked=False: self.info_action.trigger())
        self.sidebar.session_activated.connect(self._switch_session)
        self.sidebar.session_delete_requested.connect(self._request_delete_session)

        self.send_button.clicked.connect(self._submit_request)
        self.add_image_action.triggered.connect(self._attach_images)
        self.insert_file_path_action.triggered.connect(self._insert_file_paths)
        self.model_chip_menu.triggered.connect(self._on_model_action_triggered)
        self.composer.submit_requested.connect(self._submit_request)
        self.composer.image_pasted.connect(self._handle_pasted_image)
        self.composer.image_files_pasted.connect(self._handle_pasted_image_files)
        self.composer.textChanged.connect(self._queue_composer_height_sync)
        self.composer.textChanged.connect(self._refresh_submit_controls)
        self.composer.document().documentLayout().documentSizeChanged.connect(self._queue_composer_height_sync)
        self.composer_attachments_strip.attachment_remove_requested.connect(self._remove_draft_attachment)
        self.approval_card.decision_made.connect(self._handle_inline_approval_decision)
        self.user_choice_card.option_selected.connect(self._handle_user_choice_selected)
        self.user_choice_card.custom_option_requested.connect(self._handle_custom_choice_requested)

        self.controller.initialized.connect(self._handle_initialized)
        self.controller.initialization_failed.connect(self._handle_init_failed)
        self.controller.event_emitted.connect(self._handle_event)
        self.controller.approval_requested.connect(self._handle_approval_request)
        self.controller.user_choice_requested.connect(self._handle_user_choice_request)
        self.controller.session_changed.connect(self._handle_session_changed)
        self.controller.busy_changed.connect(self._handle_busy_changed)

    def _build_event_dispatch(self) -> None:
        self._event_router = StreamEventRouter(
            {
                "run_started": self._on_run_started,
                "status_changed": self._on_status_changed,
                "assistant_delta": self._on_assistant_delta,
                "tool_started": self._on_tool_started,
                "cli_output": self._on_cli_output,
                "tool_finished": self._on_tool_finished,
                "tool_args_missing": self._on_tool_args_missing,
                "summary_notice": self._on_summary_notice,
                "approval_resolved": self._on_approval_resolved,
                "run_finished": self._on_run_finished,
                "run_failed": self._on_run_failed,
                "chat_reset": self._on_chat_reset,
            }
        )
        self._event_handlers = self._event_router.handlers

    def _queue_composer_height_sync(self, *_args) -> None:
        self._composer_state.queue_height_sync(*_args)

    def _flush_composer_height_sync(self) -> None:
        self._composer_state.flush_height_sync()

    def _composer_visual_line_count(self) -> int:
        return self._composer_state.composer_visual_line_count()

    def _update_composer_height(self, *_args) -> None:
        self._composer_state.update_height(*_args)

    def _active_model_supports_images(self) -> bool:
        capabilities = resolve_model_capabilities(self._active_model_profile(), self.model_capabilities)
        return bool(capabilities.get("image_input_supported"))

    def _composer_has_request_content(self) -> bool:
        return self._composer_state.composer_has_request_content()

    def _request_blocked_by_image_capability(self) -> bool:
        return self._composer_state.request_blocked_by_image_capability()

    def _show_composer_notice(self, message: str, *, level: str = "warning") -> None:
        self._composer_state.show_composer_notice(message, level=level)

    def _clear_composer_notice(self) -> None:
        self._composer_state.clear_composer_notice()

    def _refresh_draft_attachments(self) -> None:
        self._composer_state.refresh_draft_attachments()

    def _clear_draft_image_attachments(self) -> None:
        self._composer_state.clear_draft_image_attachments()

    def _append_draft_image_attachments(self, attachments: list[dict] | None) -> None:
        self._composer_state.append_draft_image_attachments(attachments)

    def _remove_draft_attachment(self, attachment_id: str) -> None:
        self._composer_state.remove_draft_attachment(attachment_id)

    def _import_image_files(self, file_paths: list[str]) -> None:
        self._composer_state.import_image_files(file_paths)

    def _handle_pasted_image(self, image: object) -> None:
        self._composer_state.handle_pasted_image(image)

    def _handle_pasted_image_files(self, file_paths: object) -> None:
        self._composer_state.handle_pasted_image_files(file_paths)

    def _refresh_submit_controls(self, *_args) -> None:
        self._set_input_enabled(True)

    def _update_send_button_visual(self, can_send: bool) -> None:
        icon_color = "#08090B" if can_send else "#8A857E"
        self.send_button.setIcon(_fa_icon("fa5s.arrow-up", color=icon_color, size=14))

    def _on_run_started(self, payload: dict) -> None:
        self._status_controller.on_run_started(payload)

    def _on_status_changed(self, payload: dict) -> None:
        self._status_controller.on_status_changed(payload)

    def _on_assistant_delta(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        full_text = str(payload.get("full_text", "") or "")
        logger.debug(
            "Stream main_window_assistant_delta full_len=%s",
            len(full_text),
        )
        self.current_turn.set_assistant_markdown(
            full_text,
        )
        self.transcript.notify_content_changed()

    def _on_tool_started(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        self.current_turn.start_tool(payload)
        self.transcript.notify_content_changed()

    def _on_tool_finished(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        self.current_turn.finish_tool(payload)
        self.transcript.notify_content_changed()

    def _on_cli_output(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        self.current_turn.append_tool_output(payload)
        self.transcript.notify_content_changed()

    def _on_summary_notice(self, payload: dict) -> None:
        kind = str(payload.get("kind", "") or "")
        level = str(payload.get("level", "info") or "info")
        if kind == "auto_summary":
            self._summarize_in_progress = False
            if self.current_turn is not None:
                count = int(payload.get("count", 0) or 0)
                if count > 0:
                    self.current_turn.set_summary_notice(
                        f"Context compressed automatically ({count} message(s)).",
                        level="success",
                    )
                else:
                    self.current_turn.set_summary_notice("Context compressed", level="success")
                self.transcript.notify_content_changed()
            return
        if self.current_turn is not None:
            self.current_turn.add_notice(payload.get("message", ""), level=level)
            self.transcript.notify_content_changed()
        else:
            self.transcript.add_global_notice(payload.get("message", ""), level=level)

    def _on_tool_args_missing(self, payload: dict) -> None:
        _ = payload
        return

    def _on_approval_resolved(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        auto_resolved = bool(payload.get("auto"))
        if payload.get("approved"):
            if auto_resolved:
                self.transcript.notify_content_changed()
                return
            message = "Protected action approved."
            if payload.get("always"):
                message = "Protected action approved for this and future actions in the current session."
            self.current_turn.add_notice(message, level="success")
        else:
            self.current_turn.add_notice("Protected action denied.", level="warning")
        self.transcript.notify_content_changed()

    def _on_run_finished(self, payload: dict) -> None:
        self._status_controller.on_run_finished(payload)

    def _on_run_failed(self, payload: dict) -> None:
        self._status_controller.on_run_failed(payload)

    def _on_chat_reset(self, _payload: dict) -> None:
        self._status_controller.on_chat_reset()

    def _set_primary_status_message(self, label: str) -> None:
        self._status_bar_manager.set_primary_status_message(label)

    def _show_transient_status_message(self, label: str, timeout_ms: int = 1800) -> None:
        self._status_bar_manager.show_transient_status_message(label, timeout_ms=timeout_ms)

    def _set_status_visual(self, label: str, *, busy: bool = False, success: bool = False, error: bool = False) -> None:
        self._status_bar_manager.set_status_visual(label, busy=busy, success=success, error=error)

    def _toggle_info_popup(self) -> None:
        self._inspector_controller.toggle_info_popup()

    def _set_input_enabled(self, enabled: bool) -> None:
        has_pending_interrupt = self.awaiting_approval or self.awaiting_user_choice
        can_edit = (
            enabled
            and not self.awaiting_approval
            and not self.is_busy
            and (not self.awaiting_user_choice or self._custom_choice_armed)
        )
        can_send = (
            can_edit
            and self._has_active_model
            and self._composer_has_request_content()
            and not self._request_blocked_by_image_capability()
        )
        attach_enabled = enabled and not has_pending_interrupt and not self.is_busy
        self.composer.setEnabled(can_edit)
        self.send_button.setEnabled(can_send)
        self._update_send_button_visual(can_send)
        self.attach_button.setEnabled(attach_enabled)
        self.add_image_action.setEnabled(attach_enabled and self._has_active_model and self._active_model_supports_images())
        self.insert_file_path_action.setEnabled(attach_enabled)
        self.model_chip.setEnabled(can_edit and self._has_active_model)
        self.open_settings_inline_button.setEnabled(enabled and not has_pending_interrupt and not self.is_busy)
        self.settings_button.setEnabled(enabled and not has_pending_interrupt and not self.is_busy)
        self.info_button.setEnabled(enabled and not self.is_busy)
        self.sidebar.setEnabled(enabled and not has_pending_interrupt and not self.is_busy)
        self.approval_card.set_actions_enabled(self.awaiting_approval and not self.is_busy)
        self.user_choice_card.set_actions_enabled(
            enabled and self.awaiting_user_choice and not self.awaiting_approval and not self.is_busy
        )

    def _handle_initialized(self, payload: dict) -> None:
        self._apply_runtime_payload(payload, restore_transcript=True)
        self._set_status_visual("Ready", success=True)
        self.status_meta.setText("")
        self._set_input_enabled(True)

    def _update_env_info(self, snapshot: dict) -> None:
        self._status_bar_manager.update_env_info(snapshot)

    def _active_model_profile(self) -> dict | None:
        payload = self.model_profiles_payload if isinstance(self.model_profiles_payload, dict) else {}
        active_id = str(payload.get("active_profile") or "").strip()
        for profile in payload.get("profiles", []) or []:
            if isinstance(profile, dict) and str(profile.get("id") or "").strip() == active_id:
                return profile
        return None

    def _refresh_model_selector(self) -> None:
        payload = self.model_profiles_payload if isinstance(self.model_profiles_payload, dict) else {}
        profiles = [item for item in payload.get("profiles", []) or [] if isinstance(item, dict)]
        enabled_profiles = [item for item in profiles if bool(item.get("enabled", True))]
        active_id = str(payload.get("active_profile") or "").strip()

        self.model_chip_menu.clear()
        for action in self.model_chip_group.actions():
            self.model_chip_group.removeAction(action)

        if not profiles:
            self.model_chip.setVisible(False)
            self.no_models_label.setText("No models configured")
            self.no_models_label.setVisible(True)
            self.open_settings_inline_button.setVisible(True)
            self._has_active_model = False
            return

        if not enabled_profiles:
            self.model_chip.setVisible(False)
            self.no_models_label.setText("All models disabled")
            self.no_models_label.setVisible(True)
            self.open_settings_inline_button.setVisible(True)
            self._has_active_model = False
            self.model_image_badge.setVisible(False)
            self._refresh_submit_controls()
            return

        self.model_chip.setVisible(True)
        self.no_models_label.setVisible(False)
        self.open_settings_inline_button.setVisible(False)

        for profile in enabled_profiles:
            profile_id = str(profile.get("id") or "").strip()
            if not profile_id:
                continue
            action = self.model_chip_menu.addAction(profile_id)
            action.setCheckable(True)
            action.setData(profile_id)
            provider = str(profile.get("provider") or "").strip()
            model = str(profile.get("model") or "").strip()
            action.setToolTip(f"Provider: {provider}\nModel: {model}")
            if profile_id == active_id:
                action.setChecked(True)
                self.model_chip.setText(profile_id)
                self.model_chip.setToolTip(f"Provider: {provider}\nModel: {model}")
            self.model_chip_group.addAction(action)

        self._has_active_model = any(str(item.get("id") or "").strip() == active_id for item in enabled_profiles)
        if not self._has_active_model:
            self.model_chip.setText("No models")
            self.model_chip.setToolTip("No enabled active model selected")
        self.model_image_badge.setVisible(self._has_active_model and not self._active_model_supports_images())
        self._refresh_submit_controls()

    def _apply_model_profiles_payload(self, payload: dict | None) -> None:
        self.model_profiles_payload = payload if isinstance(payload, dict) else {"active_profile": None, "profiles": []}
        self._refresh_model_selector()
        if self._request_blocked_by_image_capability():
            self._show_composer_notice(
                "Current model does not support image input. Remove the images or switch to an image-capable model.",
                level="warning",
            )
        else:
            self._clear_composer_notice()
        self._set_input_enabled(True)

    def _apply_model_capabilities_payload(self, payload: dict | None) -> None:
        capabilities = payload if isinstance(payload, dict) else {}
        self.model_capabilities = dict(DEFAULT_MODEL_CAPABILITIES)
        self.model_capabilities.update({"image_input_supported": bool(capabilities.get("image_input_supported"))})
        self.model_image_badge.setVisible(self._has_active_model and not self._active_model_supports_images())
        if self._request_blocked_by_image_capability():
            self._show_composer_notice(
                "Current model does not support image input. Remove the images or switch to an image-capable model.",
                level="warning",
            )
        else:
            self._clear_composer_notice()
        self._refresh_submit_controls()

    def _on_model_action_triggered(self, action: QAction) -> None:
        profile_id = str(action.data() or "").strip()
        if not profile_id:
            return
        self.controller.set_active_profile(profile_id)

    @staticmethod
    def _resolve_model_settings_dialog_class():
        main_module = sys.modules.get("main")
        if main_module is not None:
            patched_dialog = getattr(main_module, "ModelSettingsDialog", None)
            if patched_dialog is not None:
                return patched_dialog
        public_module = sys.modules.get("ui.main_window")
        if public_module is not None:
            patched_dialog = getattr(public_module, "ModelSettingsDialog", None)
            if patched_dialog is not None:
                return patched_dialog
        return ModelSettingsDialog

    def _open_settings_dialog(self) -> None:
        if self.is_busy or self.awaiting_approval or self.awaiting_user_choice:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish before changing settings.")
            return
        if self._model_settings_window is not None and self._model_settings_window.isVisible():
            self._model_settings_window.raise_()
            self._model_settings_window.activateWindow()
            return
        dialog_class = self._resolve_model_settings_dialog_class()
        dialog = dialog_class(self.model_profiles_payload, self)
        self._model_settings_window = dialog
        dialog.profiles_saved.connect(self._save_model_profiles_from_dialog)
        dialog.destroyed.connect(lambda *_args: setattr(self, "_model_settings_window", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _save_model_profiles_from_dialog(self, payload: dict | None) -> None:
        normalized = normalize_profiles_payload(payload or {})
        self._apply_model_profiles_payload(normalized)
        self.controller.save_profiles(normalized)

    def _handle_init_failed(self, message: str) -> None:
        self._set_status_visual("Initialization failed", error=True)
        QMessageBox.critical(self, "Initialization Failed", message)

    def _handle_session_changed(self, snapshot: dict) -> None:
        self._apply_runtime_payload(snapshot, restore_transcript="transcript" in snapshot)

    def _apply_runtime_payload(self, payload: dict, *, restore_transcript: bool) -> None:
        self.current_snapshot = payload.get("snapshot", {})
        self.active_session_id = payload.get("active_session_id", "")
        self._apply_model_profiles_payload(payload.get("model_profiles"))
        self._apply_model_capabilities_payload(payload.get("model_capabilities"))
        if restore_transcript:
            self.composer.sync_session_history_from_transcript(
                self.active_session_id,
                payload.get("transcript"),
            )
        self.composer.set_history_session(self.active_session_id)
        self.overview_panel.set_snapshot(self.current_snapshot)
        self._update_env_info(self.current_snapshot)
        self.summary_progress_ring.set_summary_progress(payload.get("summary_progress"))
        tools = payload.get("tools", self.current_snapshot.get("tools", []))
        new_hash = hash(tuple(t.get("name", "") for t in tools))
        if new_hash != self._tools_hash:
            self._tools_hash = new_hash
            self.tools_panel.set_tools(tools)
        if payload.get("help_markdown"):
            self.help_text.setMarkdown(payload.get("help_markdown", ""))
        self.sidebar.set_sessions(payload.get("sessions", []), self.active_session_id)
        if restore_transcript:
            self.current_turn = None
            self.transcript.load_transcript(payload.get("transcript"))
            self._clear_draft_image_attachments()
            self._clear_composer_notice()
            self._clear_user_choice_request()
            self._clear_approval_request()

    def _handle_busy_changed(self, busy: bool) -> None:
        self._status_controller.handle_busy_changed(busy)

    def _handle_event(self, event) -> None:
        if self._event_router is not None:
            self._event_router.dispatch(event)

    def _handle_approval_request(self, payload: dict) -> None:
        self.awaiting_approval = True
        self.awaiting_user_choice = False
        self._custom_choice_armed = False
        self._clear_user_choice_request()
        if self.current_turn is not None:
            self.current_turn.clear_status()
        self.approval_card.set_request(payload)
        self._set_input_enabled(False)
        self._set_status_visual("Waiting for approval", busy=False)
        self.approval_card.setFocus()
        self.transcript.notify_content_changed()

    def _handle_inline_approval_decision(self, approved: bool, always: bool) -> None:
        if not self.awaiting_approval:
            return
        self.awaiting_approval = False
        self._clear_approval_request()
        self.controller.resume_approval(approved, always)
        self._set_input_enabled(False)

    def _handle_user_choice_request(self, payload: dict) -> None:
        self.awaiting_user_choice = True
        self._custom_choice_armed = False
        if self.current_turn is not None:
            self.current_turn.clear_status()
        self._set_user_choice_request(payload)
        self._set_status_visual("Waiting for your choice", busy=False)
        self._set_input_enabled(True)

    def _sanitize_composer_text(self) -> tuple[str, str]:
        result = sanitize_user_text(self.composer.toPlainText())
        return result.text, build_user_input_notice(result)

    def _submit_request(self) -> None:
        text, sanitize_notice = self._sanitize_composer_text()
        attachments = list(self.draft_image_attachments)
        if not text and not attachments:
            if sanitize_notice:
                self._show_composer_notice(sanitize_notice, level="warning")
            return
        if self.awaiting_user_choice:
            if self.is_busy or self.awaiting_approval or not self._custom_choice_armed:
                return
            self._clear_user_choice_request()
            self.composer.clear()
            self._clear_draft_image_attachments()
            if sanitize_notice:
                self._show_composer_notice(sanitize_notice, level="warning")
            else:
                self._clear_composer_notice()
            self.controller.resume_user_choice(text)
            self._set_input_enabled(False)
            return
        if attachments and not self._active_model_supports_images():
            self._show_composer_notice(
                "Current model does not support image input. Switch models or remove the images.",
                level="warning",
            )
            return
        self._clear_user_choice_request()
        if text:
            self.composer.append_submitted_message(text)
        self.composer.clear()
        if sanitize_notice:
            self._show_composer_notice(sanitize_notice, level="warning")
        else:
            self._clear_composer_notice()
        request_payload = {"text": text, "attachments": attachments}
        self._clear_draft_image_attachments()
        self.controller.start_run(request_payload)
        self._set_input_enabled(False)

    def _set_user_choice_request(self, payload: dict | None) -> None:
        if not isinstance(payload, dict) or not payload:
            self._clear_user_choice_request()
            return
        self.user_choice_card.set_request(payload)
        self.user_choice_card.set_actions_enabled(
            self.awaiting_user_choice and not self.awaiting_approval and not self.is_busy
        )

    def _clear_user_choice_request(self) -> None:
        self.awaiting_user_choice = False
        self._custom_choice_armed = False
        self.user_choice_card.clear_request()

    def _clear_approval_request(self) -> None:
        self.approval_card.clear_request()

    def _handle_user_choice_selected(self, submit_text: str) -> None:
        text = str(submit_text or "").strip()
        if not text or self.is_busy or self.awaiting_approval or not self.awaiting_user_choice:
            return
        self._clear_user_choice_request()
        self.controller.resume_user_choice(text)
        self._set_input_enabled(False)

    def _handle_custom_choice_requested(self) -> None:
        if self.is_busy or self.awaiting_approval or not self.awaiting_user_choice:
            return
        self._custom_choice_armed = True
        self._set_input_enabled(True)
        self.composer.setFocus()
        self.composer.selectAll()

    def _attach_images(self) -> None:
        if self.is_busy:
            return
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images to Attach",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*)",
        )
        if not file_paths:
            return
        self._import_image_files([str(path) for path in file_paths])

    def _insert_file_paths(self) -> None:
        if self.is_busy:
            return
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files to Insert")
        if not file_paths:
            return

        image_paths: list[str] = []
        text_paths: list[str] = []
        for raw_path in file_paths:
            path = str(raw_path or "").strip()
            if not path:
                continue
            if can_read_image_file(path):
                image_paths.append(path)
            else:
                text_paths.append(path)

        if image_paths and self._active_model_supports_images():
            self._import_image_files(image_paths)
        elif image_paths:
            text_paths.extend(image_paths)
            self._show_composer_notice(
                "Current model does not support image input. Inserted image file paths as text references instead.",
                level="warning",
            )

        current_text = self.composer.toPlainText()
        if text_paths:
            if current_text and not current_text.endswith(" "):
                current_text += " "

            for path in text_paths:
                current_text += f"{self.composer.format_file_reference(path)} "

            self.composer.setPlainText(current_text)
            from PySide6.QtGui import QTextCursor

            cursor = self.composer.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.composer.setTextCursor(cursor)
        self.composer.setFocus()

    def _toggle_sidebar(self) -> None:
        self._sidebar_controller.toggle_sidebar()

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        self._sidebar_controller.set_sidebar_collapsed(collapsed)

    def _set_inspector_collapsed(self, collapsed: bool) -> None:
        self._inspector_controller.set_inspector_collapsed(collapsed)

    def _switch_session(self, session_id: str) -> None:
        self._sidebar_controller.switch_session(session_id)

    def _request_delete_session(self, session_id: str) -> None:
        self._sidebar_controller.request_delete_session(session_id)

    def _new_session(self) -> None:
        self._sidebar_controller.new_session()

    def _open_new_project(self) -> None:
        self._sidebar_controller.open_new_project()

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self.controller.shutdown()
        finally:
            super().closeEvent(event)


def main() -> int:
    _configure_qt_logging()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(_fa_icon("fa5s.robot", color=ACCENT_BLUE, size=32))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
