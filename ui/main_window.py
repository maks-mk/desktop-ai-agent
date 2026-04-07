import os
import sys

from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.constants import AGENT_VERSION
from core.multimodal import (
    DEFAULT_MODEL_CAPABILITIES,
    import_image_attachment_from_file,
    import_image_attachment_from_qimage,
    normalize_image_attachments,
    resolve_model_capabilities,
)
from core.model_profiles import normalize_profiles_payload
from ui.runtime import AgentRuntimeController
from ui.theme import ACCENT_BLUE, ERROR_RED, SUCCESS_GREEN, TEXT_MUTED, build_stylesheet
from ui.widgets import (
    ApprovalDialog,
    ChatTranscriptWidget,
    ComposerTextEdit,
    ImageAttachmentStripWidget,
    InfoPopupDialog,
    ModelSettingsDialog,
    SessionSidebarWidget,
    TRANSCRIPT_MAX_WIDTH,
    UserChoiceCardWidget,
    _fa_icon,
)


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
        self.awaiting_approval = False
        self.awaiting_user_choice = False
        self.is_busy = False
        self.sidebar_collapsed = False
        self._sidebar_width = 280
        self._custom_choice_armed = False
        self._tools_hash: int = 0  # cache: skip set_tools rebuild when tools haven’t changed
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
        self._composer_growth_lines = 2
        self._composer_height_sync_pending = False
        self._build_ui()
        self._connect_signals()
        self._build_event_dispatch()  # must come after _build_ui
        if auto_initialize:
            self.controller.initialize()

    def _build_ui(self) -> None:
        self.setWindowTitle(f"AI Agent {AGENT_VERSION}")
        self.resize(1300, 670)
        self.setMinimumSize(900, 600)
        self.setWindowIcon(_fa_icon("fa5s.robot", color=ACCENT_BLUE, size=16))
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
        self.toggle_sidebar_action = QAction(_fa_icon("fa5s.columns", color=ACCENT_BLUE, size=14), "Toggle Sidebar", self)
        self.new_session_action = QAction(_fa_icon("fa5s.plus", color=ACCENT_BLUE, size=14), "New Session", self)
        self.settings_action = QAction(_fa_icon("fa5s.cog", color=ACCENT_BLUE, size=14), "Settings", self)
        self.info_action = QAction(_fa_icon("fa5s.info-circle", color=ACCENT_BLUE, size=14), "Information", self)
        self.quit_action = QAction(_fa_icon("fa5s.sign-out-alt", color=ERROR_RED, size=14), "Quit", self)

        # Keyboard shortcuts
        self.toggle_sidebar_action.setShortcut("Ctrl+B")
        self.new_session_action.setShortcut("Ctrl+N")
        self.info_action.setShortcut("Ctrl+I")

        for action, tooltip in (
            (self.toggle_sidebar_action, "Show or hide chat history (Ctrl+B)"),
            (self.new_session_action, "Start a new session (Ctrl+N)"),
            (self.settings_action, "Manage model profiles"),
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
        view_menu.addAction(self.settings_action)
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
        self.new_project_button.setIcon(_fa_icon("fa5s.folder-open", color=ACCENT_BLUE, size=14))
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

        self.settings_button = QToolButton()
        self.settings_button.setIcon(self.settings_action.icon())
        self.settings_button.setIconSize(QSize(14, 14))
        self.settings_button.setAutoRaise(False)
        self.settings_button.setToolTip(self.settings_action.toolTip())
        self.settings_button.setStatusTip(self.settings_action.statusTip())
        self.settings_button.clicked.connect(lambda _checked=False: self.settings_action.trigger())

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
        right_layout.addWidget(self.settings_button)
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
        self.composer_shell = QHBoxLayout()
        self.composer_shell.setContentsMargins(0, 18, 0, 12)
        self.composer_shell.setSpacing(0)
        self.composer_shell.addStretch(1)

        self.composer_container = QWidget()
        self.composer_container.setObjectName("CenteredComposerRow")
        self.composer_container.setMaximumWidth(TRANSCRIPT_MAX_WIDTH)
        self.composer_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QVBoxLayout(self.composer_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        self.user_choice_card = UserChoiceCardWidget()
        outer_layout.addWidget(self.user_choice_card)

        # Pill frame: [attach] [text] [send]
        self.composer_pill = QFrame()
        self.composer_pill.setObjectName("ComposerPill")
        pill_layout = QVBoxLayout(self.composer_pill)
        pill_layout.setContentsMargins(10, 8, 8, 8)
        pill_layout.setSpacing(6)

        self.composer_attachments_strip = ImageAttachmentStripWidget(thumb_size=48, removable=True)
        pill_layout.addWidget(self.composer_attachments_strip)

        self.composer_notice_label = QLabel("")
        self.composer_notice_label.setObjectName("ComposerNoticeLabel")
        self.composer_notice_label.setWordWrap(True)
        self.composer_notice_label.setVisible(False)
        pill_layout.addWidget(self.composer_notice_label)

        self.composer = ComposerTextEdit()
        self.composer.setPlaceholderText("Ask the agent…")
        line_spacing = max(14, self.composer.fontMetrics().lineSpacing())
        self._composer_max_height = self._composer_min_height + (self._composer_growth_lines * line_spacing)
        self.composer.setMinimumHeight(self._composer_min_height)
        self.composer.setMaximumHeight(self._composer_max_height)
        self.composer.setFixedHeight(self._composer_min_height)  # auto-resizes via _update_composer_height
        self.composer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.composer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.composer.set_history_session(self.active_session_id)
        pill_layout.addWidget(self.composer, 1)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(6)

        self.attach_button = QToolButton()
        self.attach_button.setIcon(_fa_icon("fa5s.plus", color=TEXT_MUTED, size=12))
        self.attach_button.setIconSize(QSize(12, 12))
        self.attach_button.setFixedSize(28, 28)
        self.attach_button.setObjectName("ComposerAttachButton")
        self.attach_button.setToolTip("Add image or Add files")
        self.attach_menu = QMenu(self.attach_button)
        self.add_image_action = self.attach_menu.addAction("Add image…")
        self.insert_file_path_action = self.attach_menu.addAction("Add files…")
        self.attach_button.setMenu(self.attach_menu)
        self.attach_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        control_row.addWidget(self.attach_button, 0, Qt.AlignVCenter)

        self.model_chip = QToolButton()
        self.model_chip.setObjectName("ComposerMetaChipButton")
        self.model_chip.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.model_chip.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.model_chip.setText("Model")
        self.model_chip.setCursor(Qt.PointingHandCursor)
        self.model_chip.setFocusPolicy(Qt.NoFocus)
        self.model_chip_menu = QMenu(self.model_chip)
        self.model_chip.setMenu(self.model_chip_menu)
        self.model_chip_group = QActionGroup(self.model_chip_menu)
        self.model_chip_group.setExclusive(True)
        control_row.addWidget(self.model_chip, 0, Qt.AlignVCenter)

        self.model_image_badge = QLabel("no img")
        self.model_image_badge.setObjectName("ComposerCapabilityBadge")
        self.model_image_badge.setVisible(False)
        control_row.addWidget(self.model_image_badge, 0, Qt.AlignVCenter)

        self.no_models_label = QLabel("No models configured")
        self.no_models_label.setObjectName("ComposerNoModelText")
        self.no_models_label.setVisible(False)
        control_row.addWidget(self.no_models_label, 0, Qt.AlignVCenter)

        self.open_settings_inline_button = QPushButton("Open Settings")
        self.open_settings_inline_button.setObjectName("ComposerOpenSettingsButton")
        self.open_settings_inline_button.setVisible(False)
        self.open_settings_inline_button.setFocusPolicy(Qt.NoFocus)
        control_row.addWidget(self.open_settings_inline_button, 0, Qt.AlignVCenter)

        self.effort_chip = QPushButton("Высокий")
        self.effort_chip.setObjectName("ComposerMetaChip")
        self.effort_chip.setEnabled(False)
        self.effort_chip.setCursor(Qt.ArrowCursor)
        self.effort_chip.setFocusPolicy(Qt.NoFocus)
        self.effort_chip.setVisible(False)
        # control_row.addWidget(self.effort_chip, 0, Qt.AlignVCenter)

        control_row.addStretch(1)

        self.voice_button = QToolButton()
        self.voice_button.setObjectName("ComposerGhostButton")
        self.voice_button.setIcon(_fa_icon("fa5s.microphone", color=TEXT_MUTED, size=11))
        self.voice_button.setIconSize(QSize(11, 11))
        self.voice_button.setToolTip("Voice input (soon)")
        self.voice_button.setEnabled(False)
        self.voice_button.setFixedSize(24, 24)
        self.voice_button.setVisible(False)
        # control_row.addWidget(self.voice_button, 0, Qt.AlignVCenter)

        self.send_button = QPushButton(_fa_icon("fa5s.arrow-up", color="#08090B", size=14), "")
        self.send_button.setObjectName("ComposerSendButton")
        self.send_button.setToolTip("Send (Enter)")
        self.send_button.setFixedSize(32, 32)
        control_row.addWidget(self.send_button, 0, Qt.AlignVCenter)
        
        self.stop_action_button = QPushButton(_fa_icon("fa5s.stop", color="#FFFFFF", size=14), "")
        self.stop_action_button.setStyleSheet("QPushButton { background: #F26D6D; border: none; border-radius: 15px; } QPushButton:hover { background: #ff8b8b; }")
        self.stop_action_button.setToolTip("Stop")
        self.stop_action_button.setFixedSize(32, 32)
        self.stop_action_button.setVisible(False)
        control_row.addWidget(self.stop_action_button, 0, Qt.AlignVCenter)
        pill_layout.addLayout(control_row)

        outer_layout.addWidget(self.composer_pill)
        self.composer_shell.addWidget(self.composer_container, 3)
        self.composer_shell.addStretch(1)
        right_layout.addLayout(self.composer_shell)

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
        self.settings_action.triggered.connect(lambda _checked=False: self._open_settings_dialog())
        self.info_action.triggered.connect(lambda _checked=False: self._toggle_info_popup())
        self.info_button.clicked.connect(lambda _checked=False: self._toggle_info_popup())
        self.open_settings_inline_button.clicked.connect(lambda _checked=False: self._open_settings_dialog())
        self.quit_action.triggered.connect(lambda _checked=False: self.close())
        self.stop_action_button.clicked.connect(lambda _checked=False: self.controller.stop_run())
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
        self.composer.document().documentLayout().documentSizeChanged.connect(
            self._queue_composer_height_sync
        )
        self.composer_attachments_strip.attachment_remove_requested.connect(self._remove_draft_attachment)
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
        """Build a dispatch table for stream events, replacing the long if/elif chain.
        Handlers are called with (payload: dict) when the matching event arrives."""
        self._event_handlers: dict = {
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

    def _queue_composer_height_sync(self, *_args) -> None:
        if self._composer_height_sync_pending:
            return
        self._composer_height_sync_pending = True
        QTimer.singleShot(0, self._flush_composer_height_sync)

    def _flush_composer_height_sync(self) -> None:
        self._composer_height_sync_pending = False
        self._update_composer_height()

    def _composer_visual_line_count(self) -> int:
        """Count visual lines (including soft-wrap) in the composer document."""
        block = self.composer.document().firstBlock()
        total_lines = 0
        while block.isValid():
            layout = block.layout()
            line_count = int(layout.lineCount()) if layout is not None else 0
            total_lines += max(1, line_count)
            block = block.next()
        return max(1, total_lines)

    def _update_composer_height(self, *_args) -> None:
        """Auto-resize the composer text field as the user types."""
        line_spacing = max(14, self.composer.fontMetrics().lineSpacing())
        visual_lines = self._composer_visual_line_count()
        doc_height = (visual_lines * line_spacing) + self._composer_height_padding
        # Grow only for a couple of additional lines, then keep fixed height and let the scrollbar handle overflow.
        new_height = max(self._composer_min_height, min(doc_height, self._composer_max_height))
        if self.composer.height() != new_height:
            self.composer.setFixedHeight(new_height)

    def _active_model_supports_images(self) -> bool:
        capabilities = resolve_model_capabilities(self._active_model_profile(), self.model_capabilities)
        return bool(capabilities.get("image_input_supported"))

    def _composer_has_request_content(self) -> bool:
        return bool(self.composer.toPlainText().strip() or self.draft_image_attachments)

    def _request_blocked_by_image_capability(self) -> bool:
        return bool(self.draft_image_attachments) and not self._active_model_supports_images()

    def _show_composer_notice(self, message: str, *, level: str = "warning") -> None:
        self.composer_notice_label.setText(str(message or "").strip())
        self.composer_notice_label.setProperty("severity", level)
        self.composer_notice_label.style().unpolish(self.composer_notice_label)
        self.composer_notice_label.style().polish(self.composer_notice_label)
        self.composer_notice_label.setVisible(bool(message))

    def _clear_composer_notice(self) -> None:
        self.composer_notice_label.clear()
        self.composer_notice_label.setProperty("severity", "")
        self.composer_notice_label.style().unpolish(self.composer_notice_label)
        self.composer_notice_label.style().polish(self.composer_notice_label)
        self.composer_notice_label.setVisible(False)

    def _refresh_draft_attachments(self) -> None:
        self.composer_attachments_strip.set_attachments(self.draft_image_attachments)
        self._refresh_submit_controls()

    def _clear_draft_image_attachments(self) -> None:
        self.draft_image_attachments = []
        self._refresh_draft_attachments()

    def _append_draft_image_attachments(self, attachments: list[dict] | None) -> None:
        existing_paths = {str(item.get("path") or "").strip() for item in self.draft_image_attachments}
        for attachment in normalize_image_attachments(attachments):
            path = str(attachment.get("path") or "").strip()
            if path and path not in existing_paths:
                self.draft_image_attachments.append(attachment)
                existing_paths.add(path)
        self._refresh_draft_attachments()

    def _remove_draft_attachment(self, attachment_id: str) -> None:
        target_id = str(attachment_id or "").strip()
        if not target_id:
            return
        self.draft_image_attachments = [
            item for item in self.draft_image_attachments if str(item.get("id") or "").strip() != target_id
        ]
        if not self.draft_image_attachments:
            self._clear_composer_notice()
        self._refresh_draft_attachments()

    def _import_image_files(self, file_paths: list[str]) -> None:
        if not file_paths:
            return
        if not self._active_model_supports_images():
            self._show_composer_notice(
                "Current model does not support image input. Switch models or use Add files instead.",
                level="warning",
            )
            return
        imported: list[dict] = []
        for path in file_paths:
            try:
                imported.append(import_image_attachment_from_file(path, session_id=self.active_session_id))
            except ValueError as exc:
                self._show_composer_notice(str(exc), level="warning")
        if imported:
            self._append_draft_image_attachments(imported)
            self._clear_composer_notice()

    def _handle_pasted_image(self, image: object) -> None:
        if not self._active_model_supports_images():
            self._show_composer_notice(
                "Current model does not support image input. Switch models or remove the pasted image.",
                level="warning",
            )
            return
        try:
            attachment = import_image_attachment_from_qimage(image, session_id=self.active_session_id)
        except ValueError as exc:
            self._show_composer_notice(str(exc), level="warning")
            return
        self._append_draft_image_attachments([attachment])
        self._clear_composer_notice()

    def _handle_pasted_image_files(self, file_paths: object) -> None:
        self._import_image_files([str(path) for path in list(file_paths or []) if str(path or "").strip()])

    def _refresh_submit_controls(self, *_args) -> None:
        self._set_input_enabled(True)

    def _update_send_button_visual(self, can_send: bool) -> None:
        icon_color = "#08090B" if can_send else "#8A857E"
        self.send_button.setIcon(_fa_icon("fa5s.arrow-up", color=icon_color, size=14))

    # --- Individual event handlers (called from dispatch table) ---

    def _on_run_started(self, payload: dict) -> None:
        self._clear_user_choice_request()
        self.current_turn = self.transcript.start_turn(
            payload.get("text", ""),
            attachments=list(payload.get("attachments", []) or []),
        )
        self._summarize_in_progress = False
        if self.current_turn is not None:
            self.current_turn.set_status("Thinking")
            self.transcript.notify_content_changed(force=True)
        self._set_status_visual("Thinking…", busy=True)

    def _on_status_changed(self, payload: dict) -> None:
        label = payload.get("label")
        node = str(payload.get("node", "") or "")
        if not label:
            return

        self._set_status_visual(label, busy=node != "approval")
        transcript_changed = False
        if self.current_turn is not None:
            self.current_turn.set_status(label)
            transcript_changed = True
            if node == "summarize":
                self._summarize_in_progress = True
                self.current_turn.set_summary_notice("Контекст автоматически сжимается…", level="info")
            elif self._summarize_in_progress:
                self._summarize_in_progress = False
                self.current_turn.set_summary_notice("Контекст сжат", level="success")
            if transcript_changed:
                self.transcript.notify_content_changed()

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
                        f"Контекст автоматически сжат ({count} сообщ.).",
                        level="success",
                    )
                else:
                    self.current_turn.set_summary_notice("Контекст сжат", level="success")
                self.transcript.notify_content_changed()
            return
        if self.current_turn is not None:
            self.current_turn.add_notice(payload.get("message", ""), level=level)
            self.transcript.notify_content_changed()
        else:
            self.transcript.add_global_notice(payload.get("message", ""), level=level)

    def _on_tool_args_missing(self, payload: dict) -> None:
        _ = payload
        # Keep this diagnostic in logs only. Users can still see the tool card/result,
        # and missing canonical args are an internal stream-quality issue.
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
        self.composer.reset_history_navigation()
        self.current_turn = None
        self.transcript.clear_transcript()
        self._clear_draft_image_attachments()
        self._clear_composer_notice()
        self._clear_user_choice_request()
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
        self.status_icon.setPixmap(_fa_icon(icon_name, color=color, size=14).pixmap(14, 14))
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
        self.sidebar.setEnabled(enabled and not has_pending_interrupt and not self.is_busy)
        self.user_choice_card.set_actions_enabled(
            enabled and self.awaiting_user_choice and not self.awaiting_approval and not self.is_busy
        )

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

        active_profile = self._active_model_profile()
        if active_profile is not None:
            model = str(active_profile.get("model") or snapshot.get("model", "unknown"))
            model_id = str(active_profile.get("id") or model).strip() or "Model"
            provider = str(active_profile.get("provider") or "").strip() or snapshot.get("provider", "")
            self.model_chip.setText(model_id)
            capability_text = "yes" if self._active_model_supports_images() else "no"
            self.model_chip.setToolTip(f"Provider: {provider}\nModel: {model}\nImage input: {capability_text}")
        else:
            model = snapshot.get("model", "unknown")
            self.model_chip.setText("No models")
            self.model_chip.setToolTip("No models configured")

        tools_count = snapshot.get("tools_count", 0)
        self.runtime_meta_label.setText(
            f"Workdir: {cwd_display}   |   Model: {model}   |   Tools: {tools_count}"
        )
        self.runtime_meta_label.setToolTip(
            f"Workdir: {cwd}\nModel: {model}\nTools: {tools_count}"
        )

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

    def _open_settings_dialog(self) -> None:
        if self.is_busy or self.awaiting_approval or self.awaiting_user_choice:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish before changing settings.")
            return
        dialog = ModelSettingsDialog(self.model_profiles_payload, self)
        if dialog.exec():
            payload = normalize_profiles_payload(dialog.result_payload())
            # Optimistic local refresh so reopening Settings immediately shows just-saved values.
            self._apply_model_profiles_payload(payload)
            self.controller.save_profiles(payload)

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
            self._clear_draft_image_attachments()
            self._clear_composer_notice()
            self._clear_user_choice_request()

    def _handle_busy_changed(self, busy: bool) -> None:
        self.is_busy = busy
        
        # Переключаем видимость кнопок
        self.send_button.setVisible(not busy)
        self.stop_action_button.setVisible(busy)

        if busy:
            self._set_status_visual("Working…", busy=True)
        else:
            if not self.awaiting_approval and not self.awaiting_user_choice:
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

    def _handle_user_choice_request(self, payload: dict) -> None:
        self.awaiting_user_choice = True
        self._custom_choice_armed = False
        self._set_user_choice_request(payload)
        self._set_status_visual("Waiting for your choice", busy=False)
        self._set_input_enabled(True)

    def _submit_request(self) -> None:
        text = self.composer.toPlainText().strip()
        attachments = list(self.draft_image_attachments)
        if not text and not attachments:
            return
        if self.awaiting_user_choice:
            if self.is_busy or self.awaiting_approval or not self._custom_choice_armed:
                return
            self._clear_user_choice_request()
            self.composer.clear()
            self._clear_draft_image_attachments()
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

        current_text = self.composer.toPlainText()
        if current_text and not current_text.endswith(" "):
            current_text += " "

        for path in file_paths:
            current_text += f"{self.composer.format_file_reference(path)} "

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
        if (
            self.is_busy
            or self.awaiting_approval
            or self.awaiting_user_choice
            or not session_id
            or session_id == self.active_session_id
        ):
            return
        self.composer.set_history_session(session_id)
        self.current_turn = None
        self.transcript.clear_transcript()
        self._clear_draft_image_attachments()
        self._clear_composer_notice()
        self._clear_user_choice_request()
        self._show_transient_status_message("Switching chat…")
        self.controller.switch_session(session_id)

    def _request_delete_session(self, session_id: str) -> None:
        if not session_id or self.is_busy or self.awaiting_approval or self.awaiting_user_choice:
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
        if self.is_busy or self.awaiting_user_choice:
            QMessageBox.information(self, "Busy", "Wait for the current run to finish before starting a new session.")
            return
        pending_key = "__pending_new_session__"
        self.composer.clear_history_for_session(pending_key)
        self.composer.set_history_session(pending_key)
        self.composer.reset_history_navigation()
        self.current_turn = None
        self.transcript.clear_transcript()
        self._clear_draft_image_attachments()
        self._clear_composer_notice()
        self._clear_user_choice_request()
        self.controller.new_session()
        self._show_transient_status_message("Created a new session")

    def _open_new_project(self) -> None:
        if self.is_busy or self.awaiting_user_choice:
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

        pending_key = "__pending_project_switch__"
        self.composer.clear_history_for_session(pending_key)
        self.composer.set_history_session(pending_key)
        self.composer.reset_history_navigation()
        self.current_turn = None
        self.transcript.clear_transcript()
        self._clear_user_choice_request()
        self._set_status_visual("Creating a new chat for the selected folder…", busy=True)
        self.controller.reinitialize(force_new_session=True)

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
