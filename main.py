import sys

import qtawesome as qta
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.constants import AGENT_VERSION
from core.gui_runtime import AgentRuntimeController
from core.gui_widgets import (
    ApprovalDialog,
    ChatTranscriptWidget,
    ComposerTextEdit,
    InfoPopupDialog,
    SessionSidebarWidget,
    TRANSCRIPT_MAX_WIDTH,
)
from core.ui_theme import ACCENT_BLUE, ERROR_RED, SUCCESS_GREEN, TEXT_MUTED, build_stylesheet


class MainWindow(QMainWindow):
    def __init__(self, controller: AgentRuntimeController | None = None, *, auto_initialize: bool = True) -> None:
        super().__init__()
        self.controller = controller or AgentRuntimeController(self)
        self.current_turn = None
        self.current_snapshot: dict | None = None
        self.active_session_id = ""
        self.awaiting_approval = False
        self.is_busy = False
        self.sidebar_collapsed = False
        self._sidebar_width = 280
        self._tools_hash: int = 0  # cache: skip set_tools rebuild when tools haven’t changed
        self._primary_status_label = "Initializing runtime…"
        self._status_message_ticket = 0
        self._build_ui()
        self._connect_signals()
        self._build_event_dispatch()  # must come after _build_ui
        if auto_initialize:
            self.controller.initialize()

    def _build_ui(self) -> None:
        self.setWindowTitle(f"AI Agent {AGENT_VERSION}")
        self.resize(1300, 670)
        self.setMinimumSize(900, 600)
        self.setWindowIcon(qta.icon("fa5s.robot", color=ACCENT_BLUE))
        self.setStatusBar(QStatusBar())
        self.statusBar().setSizeGripEnabled(False)
        self.status_line_label = QLabel("")
        self.status_line_label.setObjectName("StatusBarState")
        self.status_line_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.statusBar().addWidget(self.status_line_label, 0)
        self.runtime_meta_label = QLabel("")
        self.runtime_meta_label.setObjectName("StatusBarMeta")
        self.runtime_meta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.statusBar().addPermanentWidget(self.runtime_meta_label, 1)
        self._set_primary_status_message("Initializing runtime…")

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._build_menu_bar()
        self.workspace = self._build_workspace()
        root.addWidget(self.workspace, 1)

        self.setCentralWidget(central)
        self.setStyleSheet(build_stylesheet())
        self._set_input_enabled(False)

    def _build_menu_bar(self) -> None:
        self.toggle_sidebar_action = QAction(qta.icon("fa5s.columns", color=ACCENT_BLUE), "Toggle Sidebar", self)
        self.new_session_action = QAction(qta.icon("fa5s.plus", color=ACCENT_BLUE), "New Session", self)
        self.info_action = QAction(qta.icon("fa5s.info-circle", color=ACCENT_BLUE), "Information", self)
        self.quit_action = QAction(qta.icon("fa5s.sign-out-alt", color=ERROR_RED), "Quit", self)

        # Keyboard shortcuts
        self.toggle_sidebar_action.setShortcut("Ctrl+B")
        self.new_session_action.setShortcut("Ctrl+N")
        self.info_action.setShortcut("Ctrl+I")

        for action, tooltip in (
            (self.toggle_sidebar_action, "Show or hide chat history (Ctrl+B)"),
            (self.new_session_action, "Start a new session (Ctrl+N)"),
            (self.info_action, "Show session information, tools, and help (Ctrl+I)"),
            (self.quit_action, "Quit"),
        ):
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)

        # --- Actual QMenuBar embedded in custom top bar ---
        actual_menu = QMenuBar()
        actual_menu.setNativeMenuBar(False)
        actual_menu.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        menu = actual_menu.addMenu("File")
        menu.addAction(self.new_session_action)
        menu.addSeparator()
        menu.addAction(self.quit_action)

        view_menu = actual_menu.addMenu("View")
        view_menu.addAction(self.toggle_sidebar_action)
        view_menu.addAction(self.info_action)

        # Keep runtime status labels as lightweight state holders; the visible
        # status now lives in the bottom status bar and inside the transcript.
        self.status_icon = QLabel()
        self.status_text = QLabel("Initializing runtime…")
        self.status_text.setObjectName("StatusLabel")
        self.status_meta = QLabel("")
        self.status_meta.setObjectName("MetaText")

        # --- Right-side action buttons ---
        self.new_project_button = QToolButton()
        self.new_project_button.setIcon(qta.icon("fa5s.folder-open", color=ACCENT_BLUE))
        self.new_project_button.setIconSize(QSize(14, 14))
        self.new_project_button.setAutoRaise(False)
        self.new_project_button.setToolTip("Open new project folder")
        self.new_project_button.setStatusTip("Select a new working directory for the agent")
        self.new_project_button.clicked.connect(self._open_new_project)

        self.sidebar_toggle_button = QToolButton()
        self.sidebar_toggle_button.setIcon(self.toggle_sidebar_action.icon())
        self.sidebar_toggle_button.setIconSize(QSize(14, 14))
        self.sidebar_toggle_button.setAutoRaise(False)
        self.sidebar_toggle_button.setToolTip(self.toggle_sidebar_action.toolTip())
        self.sidebar_toggle_button.setStatusTip(self.toggle_sidebar_action.statusTip())
        self.sidebar_toggle_button.clicked.connect(lambda _checked=False: self.toggle_sidebar_action.trigger())

        self.new_session_button = QToolButton()
        self.new_session_button.setIcon(self.new_session_action.icon())
        self.new_session_button.setIconSize(QSize(14, 14))
        self.new_session_button.setAutoRaise(False)
        self.new_session_button.setToolTip(self.new_session_action.toolTip())
        self.new_session_button.setStatusTip(self.new_session_action.statusTip())
        self.new_session_button.clicked.connect(lambda _checked=False: self.new_session_action.trigger())

        self.info_button = QToolButton()
        self.info_button.setIcon(self.info_action.icon())
        self.info_button.setIconSize(QSize(14, 14))
        self.info_button.setAutoRaise(False)
        self.info_button.setToolTip(self.info_action.toolTip())
        self.info_button.setStatusTip(self.info_action.statusTip())

        right_buttons = QWidget()
        right_layout = QHBoxLayout(right_buttons)
        right_layout.setContentsMargins(0, 2, 4, 2)
        right_layout.setSpacing(4)
        #right_layout.addWidget(self.stop_button)
        right_layout.addWidget(self.new_project_button)
        right_layout.addWidget(self.new_session_button)
        right_layout.addWidget(self.info_button)

        left_controls = QWidget()
        left_layout = QHBoxLayout(left_controls)
        left_layout.setContentsMargins(0, 2, 4, 2)
        left_layout.setSpacing(4)
        left_layout.addWidget(self.sidebar_toggle_button)
        left_layout.addWidget(actual_menu)

        # --- Unified top bar: [menus] [stretch] [buttons] ---
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(left_controls, 0, Qt.AlignVCenter)
        top_layout.addStretch(1)
        top_layout.addWidget(right_buttons, 0, Qt.AlignVCenter)

        self.setMenuWidget(top_bar)

        self.info_popup = InfoPopupDialog(self)
        self.overview_panel = self.info_popup.overview_panel
        self.tools_panel = self.info_popup.tools_panel
        self.help_text = self.info_popup.help_text

        self._set_status_visual("Initializing runtime…", busy=True)

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        self.sidebar_container = QFrame()
        self.sidebar_container.setObjectName("SidebarCard")
        self.sidebar_container.setMinimumWidth(220)
        self.sidebar_container.setMaximumWidth(360)
        sidebar_layout = QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(0)

        self.sidebar = SessionSidebarWidget()
        sidebar_layout.addWidget(self.sidebar, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.transcript = ChatTranscriptWidget()
        right_layout.addWidget(self.transcript, 1)

        # ---- Bottom composer bar ----
        composer_shell = QHBoxLayout()
        composer_shell.setContentsMargins(0, 8, 0, 12)
        composer_shell.setSpacing(0)
        composer_shell.addStretch(1)

        self.composer_container = QWidget()
        self.composer_container.setObjectName("CenteredComposerRow")
        self.composer_container.setMaximumWidth(TRANSCRIPT_MAX_WIDTH)
        self.composer_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QVBoxLayout(self.composer_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Pill frame: [attach] [text] [send]
        self.composer_pill = QFrame()
        self.composer_pill.setObjectName("ComposerPill")
        pill_row = QHBoxLayout(self.composer_pill)
        pill_row.setContentsMargins(10, 8, 8, 8)
        pill_row.setSpacing(6)

        self.attach_button = QToolButton()
        self.attach_button.setIcon(qta.icon("fa5s.plus", color=TEXT_MUTED))
        self.attach_button.setIconSize(QSize(12, 12))
        self.attach_button.setFixedSize(28, 28)
        self.attach_button.setObjectName("ComposerAttachButton")
        self.attach_button.setToolTip("Attach file")
        pill_row.addWidget(self.attach_button, 0, Qt.AlignBottom)

        self.composer = ComposerTextEdit()
        self.composer.setPlaceholderText("Ask the agent…")
        self.composer.setMinimumHeight(50)
        self.composer.setMaximumHeight(200)
        self.composer.setFixedHeight(50)  # auto-resizes via _update_composer_height
        pill_row.addWidget(self.composer, 1)

        self.send_button = QPushButton(qta.icon("fa5s.arrow-up", color="#08090B"), "")
        self.send_button.setObjectName("ComposerSendButton")
        self.send_button.setToolTip("Send (Enter)")
        self.send_button.setFixedSize(30, 30)
        pill_row.addWidget(self.send_button, 0, Qt.AlignBottom)
        
        self.stop_action_button = QPushButton(qta.icon("fa5s.stop", color="#FFFFFF"), "")
        self.stop_action_button.setStyleSheet("QPushButton { background: #F26D6D; border: none; border-radius: 15px; } QPushButton:hover { background: #ff8b8b; }")
        self.stop_action_button.setToolTip("Stop")
        self.stop_action_button.setFixedSize(30, 30)
        self.stop_action_button.setVisible(False)
        pill_row.addWidget(self.stop_action_button, 0, Qt.AlignBottom)

        outer_layout.addWidget(self.composer_pill)
        composer_shell.addWidget(self.composer_container, 3)
        composer_shell.addStretch(1)
        right_layout.addLayout(composer_shell)

        self.splitter.addWidget(self.sidebar_container)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self._sidebar_width, 1000])
        layout.addWidget(self.splitter, 1)
        return workspace

    def _connect_signals(self) -> None:
        self.toggle_sidebar_action.triggered.connect(lambda _checked=False: self._toggle_sidebar())
        self.new_session_action.triggered.connect(lambda _checked=False: self._new_session())
        self.info_action.triggered.connect(lambda _checked=False: self._toggle_info_popup())
        self.info_button.clicked.connect(lambda _checked=False: self._toggle_info_popup())
        self.quit_action.triggered.connect(lambda _checked=False: self.close())
        self.stop_action_button.clicked.connect(lambda _checked=False: self.controller.stop_run())
        self.sidebar.session_activated.connect(self._switch_session)
        self.sidebar.session_delete_requested.connect(self._request_delete_session)

        self.send_button.clicked.connect(self._submit_request)
        self.attach_button.clicked.connect(self._attach_file)
        self.composer.submit_requested.connect(self._submit_request)
        self.composer.document().documentLayout().documentSizeChanged.connect(
            self._update_composer_height
        )

        self.controller.initialized.connect(self._handle_initialized)
        self.controller.initialization_failed.connect(self._handle_init_failed)
        self.controller.event_emitted.connect(self._handle_event)
        self.controller.approval_requested.connect(self._handle_approval_request)
        self.controller.session_changed.connect(self._handle_session_changed)
        self.controller.busy_changed.connect(self._handle_busy_changed)

    def _build_event_dispatch(self) -> None:
        """Build a dispatch table for stream events, replacing the long if/elif chain.
        Handlers are called with (payload: dict) when the matching event arrives."""
        self._event_handlers: dict = {
            "run_started": self._on_run_started,
            "status_changed": self._on_status_changed,
            "assistant_delta": self._on_assistant_delta,
            "tool_started": self._on_tool_started,
            "tool_finished": self._on_tool_finished,
            "tool_args_missing": self._on_tool_args_missing,
            "summary_notice": self._on_summary_notice,
            "approval_resolved": self._on_approval_resolved,
            "run_finished": self._on_run_finished,
            "run_failed": self._on_run_failed,
            "chat_reset": self._on_chat_reset,
        }

    def _update_composer_height(self, *_args) -> None:
        """Auto-resize the composer text field as the user types."""
        doc_height = int(self.composer.document().size().height())
        # Увеличили минимум до 50 и запас до +18, чтобы текст не прокручивался
        new_height = max(50, min(doc_height + 18, 200))
        if self.composer.height() != new_height:
            self.composer.setFixedHeight(new_height)
            
    # --- Individual event handlers (called from dispatch table) ---

    def _on_run_started(self, payload: dict) -> None:
        self.current_turn = self.transcript.start_turn(payload.get("text", ""))
        if self.current_turn is not None:
            self.current_turn.set_status("Thinking")
        self._set_status_visual("Thinking…", busy=True)

    def _on_status_changed(self, payload: dict) -> None:
        if payload.get("label"):
            self._set_status_visual(payload["label"], busy=payload.get("node") != "approval")
            if self.current_turn is not None:
                self.current_turn.set_status(payload["label"])

    def _on_assistant_delta(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        self.current_turn.set_assistant_markdown(payload.get("full_text", ""))
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

    def _on_summary_notice(self, payload: dict) -> None:
        if self.current_turn is not None:
            self.current_turn.add_notice(payload.get("message", ""), level="info")
            self.transcript.notify_content_changed()
        else:
            self.transcript.add_global_notice(payload.get("message", ""), level="info")

    def _on_tool_args_missing(self, payload: dict) -> None:
        tool_name = payload.get("name", "tool")
        message = f"Не удалось восстановить параметры вызова для инструмента `{tool_name}`."
        if self.current_turn is not None:
            self.current_turn.add_notice(message, level="warning")
            self.transcript.notify_content_changed()
        else:
            self.transcript.add_global_notice(message, level="warning")

    def _on_approval_resolved(self, payload: dict) -> None:
        if self.current_turn is None:
            return
        if payload.get("approved"):
            message = "Protected action approved."
            if payload.get("always"):
                message = "Protected action approved for this and future actions in the current session."
            self.current_turn.add_notice(message, level="success")
        else:
            self.current_turn.add_notice("Protected action denied.", level="warning")
        self.transcript.notify_content_changed()

    def _on_run_finished(self, payload: dict) -> None:
        if self.current_turn is not None:
            self.current_turn.clear_status()
            self.current_turn.complete(payload.get("stats", ""))
            self.transcript.notify_content_changed()
        self.status_meta.setText("")
        self._set_status_visual("Ready", success=True)

    def _on_run_failed(self, payload: dict) -> None:
        self.status_meta.setText("")
        msg = payload.get("message", "Run failed")
        if self.current_turn is not None:
            self.current_turn.clear_status()
            self.current_turn.add_notice(msg, level="error")
            self.transcript.notify_content_changed()
        else:
            self.transcript.add_global_notice(msg, level="error")
        self._set_status_visual("Run failed", error=True)

    def _on_chat_reset(self, _payload: dict) -> None:
        self.current_turn = None
        self.transcript.clear_transcript()
        self._show_transient_status_message("Started a new session")

    # --------------------------------------------------------------------------

    def _set_primary_status_message(self, label: str) -> None:
        self._primary_status_label = label
        self._status_message_ticket += 1
        self.status_line_label.setText(label)

    def _show_transient_status_message(self, label: str, timeout_ms: int = 1800) -> None:
        self._status_message_ticket += 1
        ticket = self._status_message_ticket
        self.status_line_label.setText(label)

        def _restore() -> None:
            if ticket != self._status_message_ticket:
                return
            self.status_line_label.setText(self._primary_status_label)

        QTimer.singleShot(timeout_ms + 30, _restore)

    def _set_status_visual(self, label: str, *, busy: bool = False, success: bool = False, error: bool = False) -> None:
        color = ACCENT_BLUE if busy else SUCCESS_GREEN if success else ERROR_RED if error else ACCENT_BLUE
        icon_name = "fa5s.spinner" if busy else "fa5s.check-circle" if success else "fa5s.times-circle" if error else "fa5s.circle"
        self.status_text.setText(label)
        self.status_icon.setPixmap(qta.icon(icon_name, color=color).pixmap(14, 14))
        self._set_primary_status_message(label)

    def _toggle_info_popup(self) -> None:
        if self.info_popup.isVisible():
            self.info_popup.hide()
            return
        self.info_popup.tabs.setCurrentIndex(0)
        self._position_info_popup()
        self.info_popup.show()
        self.info_popup.raise_()
        self.info_popup.activateWindow()

    def _position_info_popup(self) -> None:
        anchor = self.info_button if hasattr(self, "info_button") else self.menuBar()
        global_pos = anchor.mapToGlobal(QPoint(anchor.width() - self.info_popup.width(), anchor.height() + 8))
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            bounds = screen.availableGeometry()
            max_x = max(bounds.left() + 8, bounds.right() - self.info_popup.width() - 8)
            max_y = max(bounds.top() + 8, bounds.bottom() - self.info_popup.height() - 8)
            global_pos.setX(max(bounds.left() + 8, min(global_pos.x(), max_x)))
            global_pos.setY(max(bounds.top() + 8, min(global_pos.y(), max_y)))
        self.info_popup.move(global_pos)

    def _set_input_enabled(self, enabled: bool) -> None:
        can_edit = enabled and not self.awaiting_approval and not self.is_busy
        self.composer.setEnabled(can_edit)
        self.send_button.setEnabled(can_edit)
        self.attach_button.setEnabled(enabled and not self.awaiting_approval)
        self.sidebar.setEnabled(enabled and not self.awaiting_approval and not self.is_busy)

    def _handle_initialized(self, payload: dict) -> None:
        self._apply_runtime_payload(payload, restore_transcript=True)
        self._set_status_visual("Ready", success=True)
        self.status_meta.setText("")
        self._set_input_enabled(True)

    def _update_env_info(self, snapshot: dict) -> None:
        import os
        cwd = os.getcwd()
        if len(cwd) > 56:
            cwd_display = cwd[:18] + "..." + cwd[-34:]
        else:
            cwd_display = cwd
            
        model = snapshot.get("model", "unknown")
        tools_count = snapshot.get("tools_count", 0)
        self.runtime_meta_label.setText(
            f"Workdir: {cwd_display}   |   Model: {model}   |   Tools: {tools_count}"
        )
        self.runtime_meta_label.setToolTip(
            f"Workdir: {cwd}\nModel: {model}\nTools: {tools_count}"
        )

    def _handle_init_failed(self, message: str) -> None:
        self._set_status_visual("Initialization failed", error=True)
        QMessageBox.critical(self, "Initialization Failed", message)

    def _handle_session_changed(self, snapshot: dict) -> None:
        self._apply_runtime_payload(snapshot, restore_transcript="transcript" in snapshot)

    def _apply_runtime_payload(self, payload: dict, *, restore_transcript: bool) -> None:
        self.current_snapshot = payload.get("snapshot", {})
        self.active_session_id = payload.get("active_session_id", "")
        self.overview_panel.set_snapshot(self.current_snapshot)
        self._update_env_info(self.current_snapshot)
        tools = payload.get("tools", self.current_snapshot.get("tools", []))
        # Avoid rebuilding all tool widgets on every session_changed (fires on every tool_finished).
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

    def _handle_busy_changed(self, busy: bool) -> None:
        self.is_busy = busy
        
        # Переключаем видимость кнопок
        self.send_button.setVisible(not busy)
        self.stop_action_button.setVisible(busy)

        if busy:
            self._set_status_visual("Working…", busy=True)
        else:
            if not self.awaiting_approval:
                self._set_status_visual("Ready", success=True)
                self.status_meta.setText("")
        self._set_input_enabled(True)
        
    def _handle_event(self, event) -> None:
        event_type = getattr(event, "type", "")
        payload = getattr(event, "payload", {})
        handler = self._event_handlers.get(event_type)
        if handler:
            handler(payload)

    def _handle_approval_request(self, payload: dict) -> None:
        self.awaiting_approval = True
        self._set_input_enabled(False)
        self._set_status_visual("Waiting for approval", busy=False)

        dialog = ApprovalDialog(payload, self)
        dialog.exec()
        approved, always = dialog.choice
        self.awaiting_approval = False
        self.controller.resume_approval(approved, always)
        self._set_input_enabled(False)

    def _submit_request(self) -> None:
        text = self.composer.toPlainText().strip()
        if not text:
            return
        self.composer.clear()
        self.controller.start_run(text)
        self._set_input_enabled(False)

    def _attach_file(self) -> None:
        if self.is_busy:
            return
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files to Attach")
        if not file_paths:
            return
        
        # Append selected file paths to composer
        current_text = self.composer.toPlainText()
        if current_text and not current_text.endswith(" "):
            current_text += " "
            
        import os
        from pathlib import Path
        cwd = Path.cwd()
        
        for path in file_paths:
            try:
                # Try to make path relative to cwd to avoid "absolute path" virtual mode errors
                rel_path = Path(path).relative_to(cwd)
                path_str = str(rel_path).replace("\\", "/")
            except ValueError:
                # If not relative to cwd, keep absolute path but format it with forward slashes
                path_str = str(Path(path)).replace("\\", "/")
                
            # Wrap in quotes if there's a space
            if " " in path_str:
                current_text += f'"{path_str}" '
            else:
                current_text += f"{path_str} "
                
        self.composer.setPlainText(current_text)
        # Move cursor to end
        from PySide6.QtGui import QTextCursor
        cursor = self.composer.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.composer.setTextCursor(cursor)
        self.composer.setFocus()

    def _toggle_sidebar(self) -> None:
        self._set_sidebar_collapsed(not self.sidebar_collapsed)

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        if self.sidebar_collapsed == collapsed:
            return
        self.sidebar_collapsed = collapsed
        if collapsed:
            self._sidebar_width = max(220, self.sidebar_container.width())
            self.sidebar_container.hide()
            self.splitter.setSizes([0, 1000])
            self.toggle_sidebar_action.setToolTip("Show chat history (Ctrl+B)")
            self.sidebar_toggle_button.setToolTip(self.toggle_sidebar_action.toolTip())
        else:
            self.sidebar_container.show()
            self.splitter.setSizes([self._sidebar_width, 1000])
            self.toggle_sidebar_action.setToolTip("Hide chat history (Ctrl+B)")
            self.sidebar_toggle_button.setToolTip(self.toggle_sidebar_action.toolTip())

    def _switch_session(self, session_id: str) -> None:
        if self.is_busy or self.awaiting_approval or not session_id or session_id == self.active_session_id:
            return
        self.current_turn = None
        self.transcript.clear_transcript()
        self._show_transient_status_message("Switching chat…")
        self.controller.switch_session(session_id)

    def _request_delete_session(self, session_id: str) -> None:
        if not session_id or self.is_busy or self.awaiting_approval:
            return
        title = self.sidebar.title_for_session(session_id) or "Этот чат"
        message = f"Удалить чат «{title}» из истории?"
        answer = QMessageBox.question(
            self,
            "Удаление чата",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._show_transient_status_message("Удаляю чат из истории…")
        self.controller.delete_session(session_id)

    def _new_session(self) -> None:
        if self.is_busy:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish before starting a new session.")
            return
        self.current_turn = None
        self.transcript.clear_transcript()
        self.controller.new_session()
        self._show_transient_status_message("Created a new session")

    def _open_new_project(self) -> None:
        if self.is_busy:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish before changing the project directory.")
            return

        import os

        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Project Directory",
            os.getcwd()
        )

        if not dir_path:
            return

        self._show_transient_status_message(f"Switching project to: {dir_path}")
        os.chdir(dir_path)

        self.current_turn = None
        self.transcript.clear_transcript()
        self._set_status_visual("Creating a new chat for the selected folder…", busy=True)
        self.controller.reinitialize(force_new_session=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self.controller.shutdown()
        finally:
            super().closeEvent(event)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(qta.icon("fa5s.robot", color=ACCENT_BLUE).pixmap(32, 32)))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
