from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from core.text_utils import split_markdown_segments
from ui.theme import ACCENT_BLUE, AMBER_WARNING, ERROR_RED, SUCCESS_GREEN, TEXT_MUTED
from .attachments import ImageAttachmentStripWidget
from .foundation import AutoTextBrowser, CodeBlockWidget, _collapsed_user_message_text, _fa_icon


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

        # Пружина выталкивает пузырь вправо
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

    @staticmethod
    def _split_markdown_parts(markdown: str) -> list[tuple[str, str, str]]:
        parts: list[tuple[str, str, str]] = []
        for segment in split_markdown_segments(markdown):
            if segment.kind == "code":
                parts.append(("code", segment.text, segment.language))
            elif segment.text.strip() or not parts:
                parts.append(("markdown", segment.text, ""))
        return parts

    def set_markdown(self, markdown: str) -> None:
        self._markdown = markdown
        text = markdown.strip() or "*Thinking…*"

        parts = self._split_markdown_parts(text)

        while len(self.parts_widgets) < len(parts):
            idx = len(self.parts_widgets)
            is_code = parts[idx][0] == "code"
            if is_code:
                w = CodeBlockWidget("", "", parent=self.content_widget)
                self.content_layout.addWidget(w)
            else:
                w = AutoTextBrowser(self.content_widget)
                w.setObjectName("AssistantBody")
                self.content_layout.addWidget(w)
            self.parts_widgets.append(w)

        while len(self.parts_widgets) > len(parts):
            w = self.parts_widgets.pop()
            self.content_layout.removeWidget(w)
            w.deleteLater()

        for idx, part in enumerate(parts):
            w = self.parts_widgets[idx]
            part_kind, part_text, part_language = part
            is_code = part_kind == "code"

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

        self.title_label = QLabel("Нужен выбор пользователя")
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

        self.custom_button = QPushButton("Свой вариант")
        self.custom_button.setObjectName("UserChoiceCustomButton")
        self.custom_button.clicked.connect(self.custom_option_requested.emit)
        layout.addWidget(self.custom_button)

        self.setVisible(False)

    def set_request(self, payload: dict[str, object]) -> None:
        question = str(payload.get("question") or "").strip()
        options = list(payload.get("options") or [])
        recommended_key = str(payload.get("recommended_key") or payload.get("recommended") or "").strip()

        self.question_label.setText(question or "Выберите, как продолжить.")
        if recommended_key:
            self.hint_label.setText(f"Рекомендуемый вариант: {recommended_key}")
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



