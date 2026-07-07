from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QActionGroup, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.theme import TEXT_MUTED
from ui.widgets import (
    ApprovalRequestCardWidget,
    ChatTranscriptWidget,
    ComposerTextEdit,
    ImageAttachmentStripWidget,
    InspectorPanelWidget,
    PlanProgressPanelWidget,
    SessionSidebarWidget,
    SummaryProgressRing,
    TRANSCRIPT_MAX_WIDTH,
    UserChoiceCardWidget,
    _fa_icon,
)

COMPOSER_ATTACH_TOOLTIP = "Add images or insert file paths"
COMPOSER_ADD_IMAGE_LABEL = "Add image…"
COMPOSER_INSERT_FILE_PATH_LABEL = "Add files…"


class PlanModeSwitch(QCheckBox):
    def sizeHint(self) -> QSize:
        return QSize(74, 30)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        track = QRectF(0.5, 6.5, 32.0, 17.0)
        checked = self.isChecked()
        enabled = self.isEnabled()
        track_color = QColor("#5C5851" if checked else "#3A3732")
        border_color = QColor("#A8A49E" if checked else "#514D46")
        thumb_color = QColor("#ECEAE6" if enabled else "#524F4A")
        text_color = QColor("#C9C5BE" if checked else "#9E9A94")
        if not enabled:
            track_color = QColor("#2F2D2A")
            border_color = QColor("#38352F")
            text_color = QColor("#524F4A")

        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, 8.5, 8.5)

        thumb_x = 17.0 if checked else 3.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(thumb_color)
        painter.drawEllipse(QRectF(thumb_x, 9.0, 12.0, 12.0))

        painter.setPen(text_color)
        painter.drawText(QRectF(39.0, 0.0, 35.0, 30.0), Qt.AlignVCenter | Qt.AlignLeft, self.text())
        painter.end()



@dataclass(frozen=True)
class WorkspaceBuildResult:
    workspace: QWidget
    splitter: QSplitter
    sidebar_container: QFrame
    sidebar: SessionSidebarWidget
    transcript: ChatTranscriptWidget
    composer_shell: QHBoxLayout
    composer_container: QWidget
    approval_card: ApprovalRequestCardWidget
    user_choice_card: UserChoiceCardWidget
    composer_pill: QFrame
    composer_attachments_strip: ImageAttachmentStripWidget
    composer_notice_label: QLabel
    composer: ComposerTextEdit
    attach_button: QToolButton
    attach_menu: QMenu
    add_image_action: object
    insert_file_path_action: object
    model_chip: QToolButton
    model_chip_menu: QMenu
    model_chip_group: QActionGroup
    model_image_badge: QLabel
    no_models_label: QLabel
    open_settings_inline_button: QPushButton
    plan_mode_button: PlanModeSwitch
    summary_progress_ring: SummaryProgressRing
    send_button: QPushButton
    stop_action_button: QPushButton
    inspector_container: QFrame
    inspector_panel: InspectorPanelWidget
    plan_progress_panel: PlanProgressPanelWidget
    overview_panel: QWidget
    tools_panel: QWidget
    help_text: QWidget


class WorkspaceBuilder:
    def __init__(self, window) -> None:
        self.window = window

    def build(self) -> WorkspaceBuildResult:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        sidebar_container = QFrame()
        sidebar_container.setObjectName("SidebarCard")
        sidebar_container.setMinimumWidth(250)
        sidebar_container.setMaximumWidth(360)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(0)

        sidebar = SessionSidebarWidget()
        sidebar_layout.addWidget(sidebar, 1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        transcript = ChatTranscriptWidget()
        transcript.setAccessibleName("Conversation transcript")
        transcript.setAccessibleDescription("Shows user messages, assistant output, tools, and notices")
        center_layout.addWidget(transcript, 1)

        composer_shell = QHBoxLayout()
        composer_shell.setContentsMargins(0, 18, 0, 12)
        composer_shell.setSpacing(0)
        composer_shell.addStretch(1)

        composer_container = QWidget()
        composer_container.setObjectName("CenteredComposerRow")
        composer_container.setMaximumWidth(TRANSCRIPT_MAX_WIDTH)
        composer_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QVBoxLayout(composer_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        approval_card = ApprovalRequestCardWidget()
        outer_layout.addWidget(approval_card)

        user_choice_card = UserChoiceCardWidget()
        outer_layout.addWidget(user_choice_card)

        composer_pill = QFrame()
        composer_pill.setObjectName("ComposerPill")
        pill_layout = QVBoxLayout(composer_pill)
        pill_layout.setContentsMargins(10, 8, 8, 8)
        pill_layout.setSpacing(6)

        composer_attachments_strip = ImageAttachmentStripWidget(thumb_size=48, removable=True)
        pill_layout.addWidget(composer_attachments_strip)

        composer_notice_label = QLabel("")
        composer_notice_label.setObjectName("ComposerNoticeLabel")
        composer_notice_label.setWordWrap(True)
        composer_notice_label.setVisible(False)
        pill_layout.addWidget(composer_notice_label)

        composer = ComposerTextEdit()
        composer.setPlaceholderText("Describe your task, attach context, or type @ to add files…")
        composer.setAccessibleName("Composer")
        composer.setAccessibleDescription("Write a request for the agent")
        line_spacing = max(14, composer.fontMetrics().lineSpacing())
        self.window._composer_max_height = self.window._composer_min_height + (self.window._composer_growth_lines * line_spacing)
        composer.setMinimumHeight(self.window._composer_min_height)
        composer.setMaximumHeight(self.window._composer_max_height)
        composer.setFixedHeight(self.window._composer_min_height)
        composer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        composer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        composer.set_history_session(self.window.active_session_id)
        pill_layout.addWidget(composer, 1)

        control_row = QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(6)

        attach_button = QToolButton()
        attach_button.setIcon(_fa_icon("fa5s.plus", color=TEXT_MUTED, size=12))
        attach_button.setIconSize(QSize(12, 12))
        attach_button.setFixedSize(28, 28)
        attach_button.setObjectName("ComposerAttachButton")
        attach_button.setToolTip(COMPOSER_ATTACH_TOOLTIP)
        attach_button.setAccessibleName("Add attachments")
        attach_button.setAccessibleDescription("Add images or Add files")
        attach_menu = QMenu(attach_button)
        attach_menu.setObjectName("ComposerPopupMenu")
        add_image_action = attach_menu.addAction(COMPOSER_ADD_IMAGE_LABEL)
        insert_file_path_action = attach_menu.addAction(COMPOSER_INSERT_FILE_PATH_LABEL)
        attach_button.setMenu(attach_menu)
        attach_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        apply_popup_shadow(attach_menu)
        control_row.addWidget(attach_button, 0, Qt.AlignVCenter)

        model_chip = QToolButton()
        model_chip.setObjectName("ComposerMetaChipButton")
        model_chip.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        model_chip.setToolButtonStyle(Qt.ToolButtonTextOnly)
        model_chip.setText("Model")
        model_chip.setCursor(Qt.PointingHandCursor)
        model_chip.setFocusPolicy(Qt.NoFocus)
        model_chip.setAccessibleName("Model selector")
        model_chip.setAccessibleDescription("Select the active model profile")
        model_chip_menu = QMenu(model_chip)
        model_chip_menu.setObjectName("ComposerPopupMenu")
        model_chip.setMenu(model_chip_menu)
        model_chip_group = QActionGroup(model_chip_menu)
        model_chip_group.setExclusive(True)
        apply_popup_shadow(model_chip_menu)
        control_row.addWidget(model_chip, 0, Qt.AlignVCenter)

        model_image_badge = QLabel()
        model_image_badge.setObjectName("ComposerCapabilityBadge")
        model_image_badge.setPixmap(_fa_icon("mdi6.image-off-outline", color=TEXT_MUTED, size=14).pixmap(14, 14))
        model_image_badge.setFixedSize(28, 28)
        model_image_badge.setAlignment(Qt.AlignCenter)
        model_image_badge.setToolTip("Image input unavailable for this model")
        model_image_badge.setAccessibleName("Image input unavailable")
        model_image_badge.setAccessibleDescription("The active model profile does not accept image attachments")
        model_image_badge.setVisible(False)
        control_row.addWidget(model_image_badge, 0, Qt.AlignVCenter)

        no_models_label = QLabel("No models configured")
        no_models_label.setObjectName("ComposerNoModelText")
        no_models_label.setVisible(False)
        control_row.addWidget(no_models_label, 0, Qt.AlignVCenter)

        open_settings_inline_button = QPushButton("Open Settings")
        open_settings_inline_button.setObjectName("ComposerOpenSettingsButton")
        open_settings_inline_button.setVisible(False)
        open_settings_inline_button.setFocusPolicy(Qt.NoFocus)
        control_row.addWidget(open_settings_inline_button, 0, Qt.AlignVCenter)

        plan_mode_button = PlanModeSwitch("Plan")
        plan_mode_button.setObjectName("ComposerPlanModeButton")
        plan_mode_button.setChecked(False)
        plan_mode_button.setFixedSize(74, 30)
        plan_mode_button.setToolTip("Plan mode: the agent will analyse and propose a plan without making changes")
        plan_mode_button.setAccessibleName("Plan mode toggle")
        plan_mode_button.setAccessibleDescription("When enabled, the agent creates a plan using read-only tools and does not execute changes")
        plan_mode_button.setCursor(Qt.PointingHandCursor)
        plan_mode_button.setFocusPolicy(Qt.NoFocus)
        control_row.addWidget(plan_mode_button, 0, Qt.AlignVCenter)

        control_row.addStretch(1)

        summary_progress_ring = SummaryProgressRing()
        control_row.addWidget(summary_progress_ring, 0, Qt.AlignVCenter)

        send_button = QPushButton(_fa_icon("fa5s.arrow-up", color="#08090B", size=14), "")
        send_button.setObjectName("ComposerSendButton")
        send_button.setToolTip("Send (Enter)")
        send_button.setFixedSize(32, 32)
        send_button.setAccessibleName("Send request")
        send_button.setAccessibleDescription("Submit the current request")
        control_row.addWidget(send_button, 0, Qt.AlignVCenter)

        stop_action_button = QPushButton(_fa_icon("fa5s.stop", color="#FFFFFF", size=14), "")
        stop_action_button.setObjectName("ComposerStopButton")
        stop_action_button.setToolTip("Stop")
        stop_action_button.setFixedSize(32, 32)
        stop_action_button.setAccessibleName("Stop current run")
        stop_action_button.setVisible(False)
        control_row.addWidget(stop_action_button, 0, Qt.AlignVCenter)
        pill_layout.addLayout(control_row)

        outer_layout.addWidget(composer_pill)
        composer_shell.addWidget(composer_container, 3)
        composer_shell.addStretch(1)
        center_layout.addLayout(composer_shell)

        inspector_container = QFrame()
        inspector_container.setObjectName("SidebarCard")
        inspector_container.setMinimumWidth(320)
        inspector_container.setMaximumWidth(420)
        inspector_layout = QVBoxLayout(inspector_container)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(0)
        inspector_panel = InspectorPanelWidget()
        overview_panel = inspector_panel.overview_panel
        tools_panel = inspector_panel.tools_panel
        help_text = inspector_panel.help_text
        inspector_layout.addWidget(inspector_panel, 1)

        plan_progress_panel = PlanProgressPanelWidget()
        plan_progress_panel.setVisible(False)
        inspector_layout.addWidget(plan_progress_panel, 1)

        splitter.addWidget(sidebar_container)
        splitter.addWidget(center_panel)
        splitter.addWidget(inspector_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([self.window._sidebar_width, 1000, self.window._inspector_width])
        layout.addWidget(splitter, 1)

        return WorkspaceBuildResult(
            workspace=workspace,
            splitter=splitter,
            sidebar_container=sidebar_container,
            sidebar=sidebar,
            transcript=transcript,
            composer_shell=composer_shell,
            composer_container=composer_container,
            approval_card=approval_card,
            user_choice_card=user_choice_card,
            composer_pill=composer_pill,
            composer_attachments_strip=composer_attachments_strip,
            composer_notice_label=composer_notice_label,
            composer=composer,
            attach_button=attach_button,
            attach_menu=attach_menu,
            add_image_action=add_image_action,
            insert_file_path_action=insert_file_path_action,
            model_chip=model_chip,
            model_chip_menu=model_chip_menu,
            model_chip_group=model_chip_group,
            model_image_badge=model_image_badge,
            no_models_label=no_models_label,
            open_settings_inline_button=open_settings_inline_button,
            plan_mode_button=plan_mode_button,
            summary_progress_ring=summary_progress_ring,
            send_button=send_button,
            stop_action_button=stop_action_button,
            inspector_container=inspector_container,
            inspector_panel=inspector_panel,
            plan_progress_panel=plan_progress_panel,
            overview_panel=overview_panel,
            tools_panel=tools_panel,
            help_text=help_text,
        )


def apply_popup_shadow(popup: QWidget) -> None:
    shadow = QGraphicsDropShadowEffect(popup)
    shadow.setBlurRadius(28)
    shadow.setOffset(0, 10)
    shadow.setColor(QColor(0, 0, 0, 160))
    popup.setGraphicsEffect(shadow)
