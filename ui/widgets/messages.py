from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from core.text_utils import build_tool_ui_labels, split_markdown_segments
from ui.theme import ACCENT_BLUE, AMBER_WARNING, ERROR_RED, SUCCESS_GREEN, TEXT_MUTED
from .attachments import ImageAttachmentStripWidget
from .foundation import (
    AutoTextBrowser,
    CodeBlockWidget,
    CollapsibleSection,
    CopySafePlainTextEdit,
    _sync_plain_text_height,
    _collapsed_user_message_text,
    _fa_icon,
    format_approval_detail_text,
)

class NoticeWidget(QFrame):
    def __init__(self, message: str, level: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FlatNoticeRow")
        self.setFrameShape(QFrame.NoFrame)
        self._level = "info"
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(5)

        self.icon_label = QLabel()
        layout.addWidget(self.icon_label, 0, Qt.AlignTop)

        self.text_label = QLabel(message)
        self.text_label.setObjectName("MetaText")
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label, 1)

        self.set_level(level)

    def _icon_for_level(self, level: str) -> tuple[str, str]:
        if level == "warning":
            return "fa5s.exclamation-triangle", AMBER_WARNING
        if level == "error":
            return "fa5s.times-circle", ERROR_RED
        if level == "success":
            return "fa5s.check-circle", SUCCESS_GREEN
        return "fa5s.info-circle", ACCENT_BLUE

    def set_level(self, level: str) -> None:
        normalized = str(level or "info").strip().lower() or "info"
        self._level = normalized
        icon_name, color = self._icon_for_level(normalized)
        self.icon_label.setPixmap(_fa_icon(icon_name, color=color, size=11).pixmap(11, 11))

    def set_message(self, message: str) -> None:
        self.text_label.setText(str(message or ""))


class RunStatsWidget(QWidget):
    def __init__(self, stats: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 60)
        layout.setSpacing(0)
        layout.addStretch(1)

        chip = QFrame()
        chip.setObjectName("TranscriptMetaChip")
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(8, 4, 8, 4)
        chip_layout.setSpacing(5)

        icon = QLabel()
        icon.setPixmap(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=11).pixmap(11, 11))
        chip_layout.addWidget(icon, 0, Qt.AlignVCenter)

        label = QLabel(stats)
        label.setObjectName("MetaText")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        chip_layout.addWidget(label, 0, Qt.AlignVCenter)

        layout.addWidget(chip, 0, Qt.AlignRight)


class StatusIndicatorWidget(QFrame):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InlineStatusRow")
        self.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 6)
        layout.setSpacing(6)

        self.spinner = QToolButton()
        self.spinner.setObjectName("InlineStatusSpinner")
        self.spinner.setEnabled(False)
        self.spinner.setAutoRaise(True)
        self.spinner.setIcon(_fa_icon("fa5s.spinner", color=TEXT_MUTED, size=12))
        self.spinner.setIconSize(QSize(12, 12))
        self.spinner.setFixedSize(14, 14)
        layout.addWidget(self.spinner, 0, Qt.AlignVCenter)

        self.label = QLabel(label)
        self.label.setObjectName("TranscriptMeta")
        layout.addWidget(self.label, 0, Qt.AlignVCenter)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("MetaText")
        self.meta_label.setVisible(False)
        layout.addWidget(self.meta_label, 0, Qt.AlignVCenter)
        layout.addStretch(1)
        self.set_state(label)

    def set_state(self, label: str, meta: str = "", phase: str = "working") -> None:
        self.label.setText(label)
        meta_text = str(meta or "").strip()
        self.meta_label.setText(meta_text)
        self.meta_label.setVisible(bool(meta_text))
        if self.property("phase") != phase:
            self.setProperty("phase", phase)
            style = self.style()
            if style is not None:
                style.unpolish(self)
                style.polish(self)
        icon_name = "fa5s.pause-circle" if phase == "waiting" else "fa5s.spinner"
        icon_color = TEXT_MUTED if phase not in {"active", "reviewing"} else ACCENT_BLUE
        self.spinner.setIcon(_fa_icon(icon_name, color=icon_color, size=12))

    def set_label(self, label: str) -> None:
        self.set_state(label)


class UserMessageWidget(QFrame):
    def __init__(self, text: str, attachments: list[dict] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.full_text = text
        self.preview_text, self.is_expandable = _collapsed_user_message_text(text)
        self.setObjectName("TranscriptRow")
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 16)
        layout.setSpacing(0)

        # Keep user turns visually anchored to the right.
        layout.addStretch(1)

        self.bubble = QFrame(self)
        self.bubble.setObjectName("UserBubble")
        self.bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.bubble.setMaximumWidth(720)

        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(4)

        self.attachments_strip = ImageAttachmentStripWidget(thumb_size=40, removable=False, parent=self.bubble)
        self.attachments_strip.set_attachments(list(attachments or []))
        bubble_layout.addWidget(self.attachments_strip)

        self.body = QLabel(self.preview_text if self.is_expandable else self.full_text, self.bubble)
        self.body.setObjectName("TranscriptBody")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)

        from PySide6.QtGui import QTextDocument
        doc = QTextDocument()
        font = self.body.font()
        font.setPointSize(11)
        doc.setDefaultFont(font)
        doc.setPlainText(text)
        doc.setTextWidth(680)
        ideal_width = min(680, int(doc.idealWidth()) + 8)

        self.body.setMinimumWidth(ideal_width)
        self.body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.body.setMaximumWidth(680)
        self.body.setVisible(bool(text.strip()))
        bubble_layout.addWidget(self.body)

        self.toggle_button = QToolButton(self.bubble)
        self.toggle_button.setObjectName("DisclosureButton")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setVisible(self.is_expandable)
        self.toggle_button.toggled.connect(self._set_expanded)
        bubble_layout.addWidget(self.toggle_button, 0, Qt.AlignRight)
        self._set_expanded(False)

        layout.addWidget(self.bubble)

    def _set_expanded(self, expanded: bool) -> None:
        self.body.setText(self.full_text if expanded else self.preview_text)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_button.setText("Show less" if expanded else "Show more")


class AssistantMessageWidget(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TranscriptRow")
        self.setFrameShape(QFrame.NoFrame)
        self._markdown = ""
        
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 1, 0, 3)
        self._layout.setSpacing(6)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self._layout.addWidget(self.content_widget, 1)

        self.parts_widgets = []
        self.cursor_label = QLabel("▌", self.content_widget)
        self.cursor_label.setObjectName("AssistantStreamCursor")
        self.cursor_label.setVisible(False)
        self.content_layout.addWidget(self.cursor_label, 0, Qt.AlignLeft)

    @staticmethod
    def _split_markdown_parts(markdown: str) -> list[tuple[str, str, str]]:
        parts: list[tuple[str, str, str]] = []
        for segment in split_markdown_segments(markdown):
            if segment.kind == "code":
                parts.append(("code", segment.text, segment.language))
            elif segment.text.strip() or not parts:
                parts.append(("markdown", segment.text, ""))
        return parts

    def _make_part_widget(self, kind: str) -> QWidget:
        if kind == "code":
            return CodeBlockWidget("", "", parent=self.content_widget)
        widget = AutoTextBrowser(self.content_widget)
        widget.setObjectName("AssistantBody")
        return widget

    @staticmethod
    def _part_widget_matches_kind(widget: QWidget, kind: str) -> bool:
        if kind == "code":
            return isinstance(widget, CodeBlockWidget)
        return isinstance(widget, AutoTextBrowser)

    def _replace_part_widget(self, index: int, kind: str) -> QWidget:
        old_widget = self.parts_widgets[index]
        widget = self._make_part_widget(kind)
        self.content_layout.insertWidget(index, widget)
        self.content_layout.removeWidget(old_widget)
        old_widget.deleteLater()
        self.parts_widgets[index] = widget
        return widget

    def set_content(self, markdown: str) -> None:
        self._markdown = markdown
        text = markdown.strip()

        parts = self._split_markdown_parts(text) if text else []

        while len(self.parts_widgets) < len(parts):
            idx = len(self.parts_widgets)
            w = self._make_part_widget(parts[idx][0])
            self.content_layout.insertWidget(len(self.parts_widgets), w)
            self.parts_widgets.append(w)

        while len(self.parts_widgets) > len(parts):
            w = self.parts_widgets.pop()
            self.content_layout.removeWidget(w)
            w.deleteLater()

        for idx, part in enumerate(parts):
            w = self.parts_widgets[idx]
            part_kind, part_text, part_language = part
            is_code = part_kind == "code"
            if not self._part_widget_matches_kind(w, part_kind):
                w = self._replace_part_widget(idx, part_kind)

            if is_code:
                title = part_language.upper() if part_language else "CODE"
                w.set_code(part_text, part_language, title)
                w.setVisible(True)
            else:
                if part_text.strip() or idx == 0:
                    w.setMarkdown(part_text)
                    w.setVisible(True)
                else:
                    w.setVisible(False)

    def set_markdown(self, markdown: str) -> None:
        self.set_content(markdown)

    def set_streaming(self, active: bool) -> None:
        self.cursor_label.setVisible(bool(active))

    def markdown(self) -> str:
        return self._markdown


class UserChoiceCardWidget(QFrame):
    option_selected = Signal(str)
    custom_option_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("UserChoiceCard")
        self.setFrameShape(QFrame.NoFrame)
        self._option_buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.title_label = QLabel("Your input is required")
        self.title_label.setObjectName("UserChoiceCardTitle")
        layout.addWidget(self.title_label)

        self.question_label = QLabel("")
        self.question_label.setObjectName("UserChoiceCardQuestion")
        self.question_label.setWordWrap(True)
        self.question_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByKeyboard)
        layout.addWidget(self.question_label)

        self.hint_label = QLabel("")
        self.hint_label.setObjectName("UserChoiceCardHint")
        self.hint_label.setWordWrap(True)
        self.hint_label.setVisible(False)
        layout.addWidget(self.hint_label)

        self.options_host = QWidget()
        self.options_layout = QVBoxLayout(self.options_host)
        self.options_layout.setContentsMargins(0, 2, 0, 0)
        self.options_layout.setSpacing(6)
        layout.addWidget(self.options_host)

        self.custom_button = QPushButton("Custom response")
        self.custom_button.setObjectName("UserChoiceCustomButton")
        self.custom_button.clicked.connect(self.custom_option_requested.emit)
        layout.addWidget(self.custom_button)

        self.setVisible(False)

    def set_request(self, payload: dict[str, object]) -> None:
        question = str(payload.get("question") or "").strip()
        options = list(payload.get("options") or [])
        recommended_key = str(payload.get("recommended_key") or payload.get("recommended") or "").strip()

        self.question_label.setText(question or "Choose how to continue.")
        if recommended_key:
            self.hint_label.setText(f"Recommended: {recommended_key}")
            self.hint_label.setVisible(True)
        else:
            self.hint_label.clear()
            self.hint_label.setVisible(False)

        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._option_buttons.clear()

        for option_payload in options:
            if not isinstance(option_payload, dict):
                continue
            label = str(option_payload.get("label") or "").strip()
            submit_text = str(option_payload.get("submit_text") or label).strip()
            if not label or not submit_text:
                continue

            button = QPushButton(label)
            button.setObjectName("UserChoiceOptionButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setProperty("recommended", bool(option_payload.get("recommended")))
            button.clicked.connect(lambda _checked=False, value=submit_text: self.option_selected.emit(value))
            self.options_layout.addWidget(button)
            self._option_buttons.append(button)

        self.options_layout.addStretch(1)
        self.setVisible(bool(self._option_buttons) or bool(question))

    def clear_request(self) -> None:
        self.question_label.clear()
        self.hint_label.clear()
        self.hint_label.setVisible(False)
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._option_buttons.clear()
        self.setVisible(False)

    def set_actions_enabled(self, enabled: bool) -> None:
        for button in self._option_buttons:
            button.setEnabled(enabled)
        self.custom_button.setEnabled(enabled)


class ApprovalRequestCardWidget(QFrame):
    decision_made = Signal(bool, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ApprovalRequestCard")
        self.setFrameShape(QFrame.NoFrame)
        self._tool_sections: list[CollapsibleSection] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        self.title_label = QLabel("Approval required")
        self.title_label.setObjectName("ApprovalCardTitle")
        title_row.addWidget(self.title_label)

        self.risk_badge = QLabel("")
        self.risk_badge.setObjectName("ApprovalRiskBadge")
        self.risk_badge.setVisible(False)
        title_row.addWidget(self.risk_badge, 0, Qt.AlignVCenter)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("ApprovalCardSummary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.impacts_label = QLabel("")
        self.impacts_label.setObjectName("ApprovalCardImpacts")
        self.impacts_label.setWordWrap(True)
        self.impacts_label.setVisible(False)
        layout.addWidget(self.impacts_label)

        self.tools_scroll = QScrollArea()
        self.tools_scroll.setWidgetResizable(True)
        self.tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tools_scroll.setMinimumHeight(0)
        self.tools_scroll.setMaximumHeight(280)
        self.tools_scroll.setVisible(False)

        self.tools_container = QWidget()
        self.tools_layout = QVBoxLayout(self.tools_container)
        self.tools_layout.setContentsMargins(0, 0, 0, 0)
        self.tools_layout.setSpacing(8)
        self.tools_layout.addStretch(1)
        self.tools_scroll.setWidget(self.tools_container)
        layout.addWidget(self.tools_scroll)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)

        actions_row.addStretch(1)

        self.deny_button = QPushButton("Deny")
        self.deny_button.setObjectName("DangerButton")
        self.deny_button.clicked.connect(lambda: self.decision_made.emit(False, False))
        actions_row.addWidget(self.deny_button)

        self.always_button = QPushButton("Always allow")
        self.always_button.setObjectName("SecondaryButton")
        self.always_button.clicked.connect(lambda: self.decision_made.emit(True, True))
        actions_row.addWidget(self.always_button)

        self.approve_button = QPushButton("Approve")
        self.approve_button.setObjectName("PrimaryButton")
        self.approve_button.clicked.connect(lambda: self.decision_made.emit(True, False))
        actions_row.addWidget(self.approve_button)

        layout.addLayout(actions_row)

        self.setAccessibleName("Approval request")
        self.setAccessibleDescription("Review and approve protected tool actions")
        self.approve_button.setAccessibleName("Approve protected action")
        self.always_button.setAccessibleName("Always approve in this session")
        self.deny_button.setAccessibleName("Deny protected action")
        self.setVisible(False)

    def set_request(self, payload: dict[str, Any]) -> None:
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        risk_level = str(summary.get("risk_level", "unknown") or "unknown")
        impacts = [str(item).strip() for item in list(summary.get("impacts", []) or []) if str(item).strip()]
        tools = list(payload.get("tools", []) or [])

        self.risk_badge.setText(risk_level.title())
        self.risk_badge.setProperty("riskLevel", risk_level)
        style = self.risk_badge.style()
        if style is not None:
            style.unpolish(self.risk_badge)
            style.polish(self.risk_badge)
        self.risk_badge.setVisible(True)

        tools_count = len(tools)
        noun = "action" if tools_count == 1 else "actions"
        self.summary_label.setText(f"The agent is paused. Review {tools_count} protected {noun}.")
        if impacts:
            self.impacts_label.setText(f"Will affect: {', '.join(impacts)}")
            self.impacts_label.setVisible(True)
        else:
            self.impacts_label.clear()
            self.impacts_label.setVisible(False)

        while self.tools_layout.count() > 1:
            item = self.tools_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tool_sections.clear()

        for tool in tools:
            card = QFrame()
            card.setObjectName("ApprovalToolCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(4)

            tool_name = str(tool.get("name") or tool.get("display") or "tool").strip() or "tool"
            tool_args = dict(tool.get("args") or {})
            labels = build_tool_ui_labels(tool_name, tool_args, phase="finished")

            name_label = QLabel(labels.get("title") or str(tool.get("display") or tool_name))
            name_label.setObjectName("ApprovalToolTitle")
            card_layout.addWidget(name_label)

            subtitle = str(labels.get("subtitle", "") or "").strip()
            if subtitle:
                subtitle_label = QLabel(subtitle)
                subtitle_label.setObjectName("ApprovalToolSubtitle")
                subtitle_label.setWordWrap(True)
                card_layout.addWidget(subtitle_label)

            args_view = CopySafePlainTextEdit()
            args_view.setObjectName("ApprovalDetailView")
            args_view.setReadOnly(True)
            args_view.setPlainText(format_approval_detail_text(tool_args))
            _sync_plain_text_height(args_view, min_lines=6, max_lines=12, extra_padding=18)
            section = CollapsibleSection("Details", args_view, expanded=tools_count == 1)
            card_layout.addWidget(section)
            self._tool_sections.append(section)
            self.tools_layout.insertWidget(self.tools_layout.count() - 1, card)

        self.tools_scroll.setVisible(bool(tools))
        self.setVisible(True)

    def clear_request(self) -> None:
        self.summary_label.clear()
        self.impacts_label.clear()
        self.impacts_label.setVisible(False)
        self.risk_badge.clear()
        self.risk_badge.setVisible(False)
        while self.tools_layout.count() > 1:
            item = self.tools_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tool_sections.clear()
        self.tools_scroll.setVisible(False)
        self.setVisible(False)

    def set_actions_enabled(self, enabled: bool) -> None:
        self.approve_button.setEnabled(enabled)
        self.always_button.setEnabled(enabled)
        self.deny_button.setEnabled(enabled)
