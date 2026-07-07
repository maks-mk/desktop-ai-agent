from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget

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

_TEXT_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_CODE_LIKE_RE = re.compile(
    r"(^|\s)(def|class|import|from|return|if|else|elif|for|while|try|except|with|function|const|let|var)\b"
    r"|[{};]|=>|==|!=|:=|::|->|\w+\([^)]*\)"
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
    def _looks_like_plain_text_unclosed_fence(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if _CODE_LIKE_RE.search(stripped):
            return False
        letters = len(_TEXT_LETTER_RE.findall(stripped))
        return letters >= 8 and any(char in stripped for char in " .,!?;:—–-«»()")

    @classmethod
    def _split_markdown_parts(cls, markdown: str) -> list[tuple[str, str, str]]:
        parts: list[tuple[str, str, str]] = []
        for segment in split_markdown_segments(markdown):
            if segment.kind == "code":
                if not segment.closed and not segment.language and cls._looks_like_plain_text_unclosed_fence(segment.text):
                    parts.append(("markdown", f"```\n{segment.text}", ""))
                else:
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


class PlanCardWidget(QFrame):
    def __init__(self, parent: QWidget | None = None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self._compact = compact
        self.setObjectName("PlanCard")
        self.setFrameShape(QFrame.NoFrame)
        self._step_labels: list[QLabel] = []
        self._meta_cards: list[QFrame] = []

        layout = QVBoxLayout(self)
        if compact:
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)
        else:
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(7)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self.title_label = QLabel("План")
        self.title_label.setObjectName("PlanCardTitle")
        self.title_label.setWordWrap(True)
        header.addWidget(self.title_label, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("PlanCardStatus")
        self.status_label.setVisible(False)
        header.addWidget(self.status_label, 0, Qt.AlignTop)
        layout.addLayout(header)

        self.summary_card = QFrame(self)
        self.summary_card.setObjectName("PlanSummaryCard")
        self.summary_card.setFrameShape(QFrame.NoFrame)
        summary_layout = QVBoxLayout(self.summary_card)
        summary_layout.setContentsMargins(10, 8, 10, 9)
        summary_layout.setSpacing(5)

        self.summary_title = QLabel("Обзор плана")
        self.summary_title.setObjectName("PlanCardSectionTitle")
        summary_layout.addWidget(self.summary_title)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("PlanCardSummary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByKeyboard)
        summary_layout.addWidget(self.summary_label)
        self.summary_card.setVisible(False)
        layout.addWidget(self.summary_card)

        self.markdown_body = AutoTextBrowser(self)
        self.markdown_body.setObjectName("PlanCardMarkdown")
        self.markdown_body.setVisible(False)
        layout.addWidget(self.markdown_body)

        self.steps_title = QLabel("Шаги")
        self.steps_title.setObjectName("PlanCardSectionTitle")
        self.steps_title.setVisible(False)
        layout.addWidget(self.steps_title)

        self.steps_host = QWidget(self)
        self.steps_host.setObjectName("PlanStepsList")
        self.steps_layout = QVBoxLayout(self.steps_host)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(6)
        self.steps_host.setVisible(False)
        layout.addWidget(self.steps_host)

        self.meta_host = QWidget(self)
        self.meta_host.setObjectName("PlanMetaList")
        self.meta_layout = QVBoxLayout(self.meta_host)
        self.meta_layout.setContentsMargins(0, 0, 0, 0)
        self.meta_layout.setSpacing(6)
        self.meta_host.setVisible(False)
        layout.addWidget(self.meta_host)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("PlanCardMeta")
        self.meta_label.setWordWrap(True)
        self.meta_label.setVisible(False)

        self.setVisible(False)

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)

    @staticmethod
    def _status_text(status: str) -> str:
        normalized = str(status or "").strip().lower()
        labels = {
            "draft": "Черновик",
            "pending_approval": "На проверке",
            "approved": "Одобрен",
            "executing": "Выполняется",
            "completed": "Завершён",
            "failed": "Ошибка",
            "cancelled": "Отменён",
            "canceled": "Отменён",
            "rejected": "Отклонён",
            "needs_changes": "Нужны правки",
            "replan_pending": "Перепланирование",
        }
        return labels.get(normalized, normalized.replace("_", " ").title()) if normalized else ""

    @staticmethod
    def _step_icon(status: str, active: bool) -> str:
        normalized = str(status or "pending").strip().lower()
        if normalized in {"completed", "skipped"}:
            return "✓"
        if active or normalized in {"active", "in_progress", "executing"}:
            return "●"
        if normalized in {"failed", "blocked"}:
            return "!"
        return "○"

    @staticmethod
    def _plain_plan_text(value: object) -> str:
        return " ".join(str(value or "").replace("\r\n", "\n").split()).strip()

    @classmethod
    def _plain_plan_list(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            value = [value] if value else []
        items: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = cls._plain_plan_text(item)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
        return items

    @classmethod
    def _meta_sections(cls, plan: dict[str, object]) -> str:
        sections = cls._meta_section_items(plan)
        lines: list[str] = []
        for title, items in sections:
            if not items:
                continue
            if lines:
                lines.append("")
            lines.append(title)
            lines.extend(f"- {item}" for item in items)
        return "\n".join(lines).strip()

    @classmethod
    def _meta_section_items(cls, plan: dict[str, object]) -> list[tuple[str, list[str]]]:
        return [
            ("Проверка", cls._plain_plan_list(plan.get("verification"))),
            ("Риски", cls._plain_plan_list(plan.get("risks"))),
            ("Предположения", cls._plain_plan_list(plan.get("assumptions"))),
        ]

    @classmethod
    def _review_markdown(cls, plan: dict[str, object], steps: list[object]) -> str:
        plan_markdown = str(plan.get("plan_markdown") or plan.get("markdown") or "").strip()
        if plan_markdown:
            return plan_markdown

        summary = cls._plain_plan_text(plan.get("summary"))
        title = cls._plain_plan_text(plan.get("title") or plan.get("name")) or "План"
        lines = [f"# {title}"]
        if summary:
            lines.extend(["", "## Кратко", "", summary])
        key_changes = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_title = cls._plain_plan_text(step.get("title") or step.get("description"))
            description = cls._plain_plan_text(step.get("description"))
            if not step_title and not description:
                continue
            if description and description != step_title:
                key_changes.append(f"- **{step_title}**: {description}" if step_title else f"- {description}")
            else:
                key_changes.append(f"- {step_title or description}")
        if key_changes:
            lines.extend(["", "## Key Changes", "", *key_changes])
        for section_title, field_name in (
            ("Проверка", "verification"),
            ("Риски", "risks"),
            ("Предположения", "assumptions"),
        ):
            items = cls._plain_plan_list(plan.get(field_name))
            if items:
                lines.extend(["", f"## {section_title}", "", *[f"- {item}" for item in items]])
        return "\n".join(lines).strip()

    @staticmethod
    def _step_text(step: dict[str, object], index: int, active_step_id: str) -> tuple[str, str]:
        step_id = str(step.get("id") or "").strip()
        title = str(step.get("title") or step.get("description") or "").strip()
        description = str(step.get("description") or "").strip()
        status = str(step.get("status") or "pending").strip().lower()
        active = bool(active_step_id and step_id == active_step_id)
        icon = PlanCardWidget._step_icon(status, active)
        primary = title or f"Шаг {index}"

        text = f"{icon} {index}. {primary}"
        if description and description != primary:
            text = f"{text}\n   {description}"
        return text, "active" if active or icon == "●" else status or "pending"

    def _add_step_row(self, *, index: int, step: dict[str, object], active_step_id: str) -> None:
        text, step_status = self._step_text(step, index, active_step_id)
        lines = text.splitlines()
        first_line = lines[0] if lines else f"{index}. Шаг"
        description = "\n".join(line.strip() for line in lines[1:] if line.strip())
        icon, title = ("", first_line)
        if " " in first_line:
            icon, title = first_line.split(" ", 1)

        row = QFrame(self.steps_host)
        row.setObjectName("PlanStepRow")
        row.setFrameShape(QFrame.NoFrame)
        row.setProperty("stepStatus", step_status)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(9, 8, 9, 8)
        row_layout.setSpacing(8)

        icon_label = QLabel(icon, row)
        icon_label.setObjectName("PlanStepIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedWidth(18)
        icon_label.setProperty("stepStatus", step_status)
        row_layout.addWidget(icon_label, 0, Qt.AlignTop)

        text_host = QWidget(row)
        text_host.setObjectName("PlanStepText")
        text_layout = QVBoxLayout(text_host)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)

        title_label = QLabel(title, text_host)
        title_label.setObjectName("PlanStepTitle")
        title_label.setWordWrap(True)
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByKeyboard)
        title_label.setProperty("stepStatus", step_status)
        text_layout.addWidget(title_label)

        description_label = QLabel(description, text_host)
        description_label.setObjectName("PlanStepDescription")
        description_label.setWordWrap(True)
        description_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByKeyboard)
        description_label.setProperty("stepStatus", step_status)
        description_label.setVisible(bool(description))
        text_layout.addWidget(description_label)

        row_layout.addWidget(text_host, 1)
        self.steps_layout.addWidget(row)
        for widget in (row, icon_label, title_label, description_label):
            self._refresh_style(widget)
        self._step_labels.append(title_label)

    def _clear_steps(self) -> None:
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._step_labels.clear()

    def _set_meta_cards(self, plan: dict[str, object]) -> None:
        while self.meta_layout.count():
            item = self.meta_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._meta_cards.clear()

        sections = [(title, items) for title, items in self._meta_section_items(plan) if items]
        for title, items in sections:
            card = QFrame(self.meta_host)
            card.setObjectName("PlanMetaCard")
            card.setFrameShape(QFrame.NoFrame)
            card.setProperty("metaKind", title)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(9, 7, 9, 8)
            card_layout.setSpacing(4)

            title_label = QLabel(title, card)
            title_label.setObjectName("PlanMetaTitle")
            card_layout.addWidget(title_label)

            body_label = QLabel("\n".join(f"- {item}" for item in items), card)
            body_label.setObjectName("PlanMetaBody")
            body_label.setWordWrap(True)
            body_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByKeyboard)
            card_layout.addWidget(body_label)

            self.meta_layout.addWidget(card)
            self._meta_cards.append(card)
            self._refresh_style(card)

        self.meta_host.setVisible(bool(sections))

    def set_plan(self, payload: dict[str, object] | None) -> None:
        if not isinstance(payload, dict) or not payload:
            self.clear_plan()
            return

        plan = payload.get("current_plan") if isinstance(payload.get("current_plan"), dict) else payload
        if not isinstance(plan, dict) or plan.get("format") == "markdown":
            self.clear_plan()
            return

        if not self._compact:
            plan_markdown = str(payload.get("plan_markdown") or "").strip()
            if plan_markdown:
                plan = {**plan, "plan_markdown": plan_markdown}

        summary = self._plain_plan_text(plan.get("summary") or payload.get("summary"))
        title = "План реализации"
        status = str(payload.get("status") or plan.get("status") or "").strip()
        active_step_id = str(payload.get("active_step_id") or plan.get("active_step_id") or "").strip()
        steps = plan.get("steps") or payload.get("steps") or []
        if not isinstance(steps, list):
            steps = []

        visible_steps = [step for step in steps if isinstance(step, dict)]
        step_count = len(visible_steps)
        count_text = f"{step_count} шаг" if step_count == 1 else f"{step_count} шага" if 2 <= step_count <= 4 else f"{step_count} шагов"
        self.title_label.setText(f"{title} - {count_text}" if step_count else title)
        self.title_label.setVisible(True)
        status_text = self._status_text(status)
        self.status_label.setText(status_text)
        self.status_label.setVisible(bool(status_text) and not self._compact)

        self.summary_label.clear()
        self.summary_label.setText(summary)
        self.summary_card.setVisible(bool(summary))
        self._clear_steps()

        fallback_markdown = ""
        if not visible_steps:
            fallback_markdown = str(payload.get("plan_markdown") or plan.get("plan_markdown") or plan.get("markdown") or "").strip()
        self.markdown_body.setMarkdown(fallback_markdown)
        self.markdown_body.setVisible(bool(fallback_markdown))

        self.steps_title.setVisible(bool(visible_steps))
        self.steps_host.setVisible(bool(visible_steps))
        for index, step in enumerate(visible_steps, start=1):
            self._add_step_row(index=index, step=step, active_step_id=active_step_id)

        meta_text = self._meta_sections(plan)
        self.meta_label.setText(meta_text)
        self.meta_label.setVisible(False)
        self._set_meta_cards(plan)
        self.setVisible(bool(visible_steps or summary or fallback_markdown or title))

    def clear_plan(self) -> None:
        self.title_label.setText("План реализации")
        self.title_label.setVisible(True)
        self.status_label.clear()
        self.status_label.setVisible(False)
        self.summary_label.clear()
        self.summary_card.setVisible(False)
        self.markdown_body.setMarkdown("")
        self.markdown_body.setVisible(False)
        self._clear_steps()
        self.steps_title.setVisible(False)
        self.steps_host.setVisible(False)
        self.meta_label.clear()
        self.meta_label.setVisible(False)
        self._set_meta_cards({})
        self.setVisible(False)


class InlinePlanWidget(QFrame):
    option_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InlinePlanWidget")
        self.setFrameShape(QFrame.NoFrame)
        self._payload: dict[str, object] = {}
        self._expanded = True
        self._implemented = False
        self._option_buttons: list[QPushButton] = []

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(10, 8, 10, 10)
        outer_layout.setSpacing(8)

        self.accent_bar = QFrame(self)
        self.accent_bar.setObjectName("InlinePlanAccent")
        self.accent_bar.setFrameShape(QFrame.NoFrame)
        self.accent_bar.setFixedWidth(3)
        outer_layout.addWidget(self.accent_bar)

        self.content_widget = QWidget(self)
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        outer_layout.addWidget(self.content_widget, 1)

        header_frame = QFrame(self.content_widget)
        header_frame.setObjectName("InlinePlanHeader")
        header_frame.setFrameShape(QFrame.NoFrame)
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self.plan_icon = QLabel(self)
        self.plan_icon.setObjectName("InlinePlanIcon")
        self.plan_icon.setPixmap(_fa_icon("fa5s.tasks", color=TEXT_MUTED, size=12).pixmap(12, 12))
        header.addWidget(self.plan_icon, 0, Qt.AlignVCenter)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("InlinePlanToggle")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.toggled.connect(self.set_expanded)
        header.addWidget(self.toggle_button, 1)

        self.count_label = QLabel("")
        self.count_label.setObjectName("InlinePlanCount")
        header.addWidget(self.count_label, 0, Qt.AlignVCenter | Qt.AlignRight)
        layout.addWidget(header_frame)

        self.markdown_body = AutoTextBrowser(self)
        self.markdown_body.setObjectName("PlanCardMarkdown")
        self.markdown_body.setProperty("maxAutoHeight", 360)
        layout.addWidget(self.markdown_body)

        self.actions_host = QWidget(self)
        self.actions_host.setObjectName("InlinePlanActions")
        actions = QGridLayout(self.actions_host)
        actions.setContentsMargins(0, 2, 0, 0)
        actions.setHorizontalSpacing(6)
        actions.setVerticalSpacing(5)
        layout.addWidget(self.actions_host)

        for index, (label, submit_text) in enumerate(
            (
                ("Реализовать", "implement"),
                ("Изменить", "revise"),
                ("Перестроить", "rebuild"),
                ("Отмена", "cancel"),
            )
        ):
            button = QPushButton(label)
            button.setObjectName("UserChoiceOptionButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setProperty("recommended", submit_text == "implement")
            button.setProperty("planAction", submit_text)
            button.clicked.connect(lambda _checked=False, value=submit_text: self._select_option(value))
            actions.addWidget(button, index // 2, index % 2)
            self._option_buttons.append(button)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)

        self.set_payload({})

    @staticmethod
    def _step_count(payload: dict[str, object]) -> int:
        plan = payload.get("current_plan") if isinstance(payload.get("current_plan"), dict) else payload
        steps = plan.get("steps") if isinstance(plan, dict) else []
        return len(steps) if isinstance(steps, list) else 0

    @staticmethod
    def _markdown(payload: dict[str, object]) -> str:
        plan = payload.get("current_plan") if isinstance(payload.get("current_plan"), dict) else payload
        if not isinstance(plan, dict):
            return str(payload.get("plan_markdown") or "").strip()
        plan_markdown = str(payload.get("plan_markdown") or plan.get("plan_markdown") or plan.get("markdown") or "").strip()
        if plan_markdown:
            return plan_markdown
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        return PlanCardWidget._review_markdown(plan, steps)

    def set_payload(self, payload: dict[str, object] | None) -> None:
        self._payload = dict(payload or {})
        step_count = self._step_count(self._payload)
        self.count_label.setText(f"{step_count} steps" if step_count else "Plan")
        self.count_label.setVisible(bool(step_count))
        self.markdown_body.setMarkdown(self._markdown(self._payload))
        self._refresh_header()
        self.setVisible(bool(self._payload))

    def set_actions_visible(self, visible: bool) -> None:
        self.actions_host.setVisible(bool(visible) and not self._implemented and self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        if self.toggle_button.isChecked() != self._expanded:
            self.toggle_button.setChecked(self._expanded)
            return
        self.markdown_body.setVisible(self._expanded)
        self.actions_host.setVisible(self._expanded and not self._implemented)
        self._refresh_header()

    def collapse_after_implement(self) -> None:
        self._implemented = True
        self.actions_host.setVisible(False)
        self.set_expanded(False)

    def _refresh_header(self) -> None:
        step_count = self._step_count(self._payload)
        suffix = f" ({step_count} steps)" if step_count and not self._expanded else ""
        title = "Plan completed" if self._implemented and not self._expanded else "Implementation Plan"
        if self._implemented:
            title = f"✓ {title}"
        self.toggle_button.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self.toggle_button.setText(f"{title}{suffix}")

    def _select_option(self, submit_text: str) -> None:
        if submit_text == "implement":
            self.collapse_after_implement()
        self.option_selected.emit(submit_text)


class PlanProgressPanelWidget(QFrame):
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlanProgressPanel")
        self.setFrameShape(QFrame.NoFrame)
        self._last_payload: dict[str, object] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header_frame = QFrame(self)
        header_frame.setObjectName("PlanProgressHeader")
        header_frame.setFrameShape(QFrame.NoFrame)
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        title = QLabel("Выполнение плана")
        title.setObjectName("PlanProgressTitle")
        header.addWidget(title, 1)

        self.status_label = QLabel("Ожидание")
        self.status_label.setObjectName("PlanCardStatus")
        header.addWidget(self.status_label, 0, Qt.AlignTop)
        layout.addWidget(header_frame)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("PlanProgressScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.plan_card = PlanCardWidget(self.scroll, compact=True)
        self.scroll.setWidget(self.plan_card)
        layout.addWidget(self.scroll, 1)

        self.empty_label = QLabel("Одобрите план, чтобы отслеживать выполнение здесь.")
        self.empty_label.setObjectName("PlanProgressEmpty")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label, 1)

        self.cancel_button = QPushButton("Отменить план")
        self.cancel_button.setObjectName("PlanProgressCancelButton")
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.cancel_button, 0)

        self.clear_plan()

    def set_plan(self, payload: dict[str, object] | None) -> None:
        self._last_payload = dict(payload) if isinstance(payload, dict) else None
        self.plan_card.set_plan(payload)
        visible = not self.plan_card.isHidden()
        status = ""
        if isinstance(payload, dict):
            plan = payload.get("current_plan") if isinstance(payload.get("current_plan"), dict) else payload
            if isinstance(plan, dict):
                status = str(payload.get("status") or plan.get("status") or "").strip()
        completed = payload.get("completed_steps") if isinstance(payload, dict) else None
        total = payload.get("total_steps") if isinstance(payload, dict) else None
        if isinstance(completed, int) and isinstance(total, int) and total:
            self.status_label.setText(f"{completed} / {total}")
        else:
            self.status_label.setText(PlanCardWidget._status_text(status) or "Выполняется")
        self.scroll.setVisible(visible)
        self.empty_label.setVisible(not visible)
        self.cancel_button.setVisible(visible)

    def clear_plan(self) -> None:
        self._last_payload = None
        self.plan_card.clear_plan()
        self.status_label.setText("Ожидание")
        self.scroll.setVisible(False)
        self.empty_label.setVisible(True)
        self.cancel_button.setVisible(False)


class UserChoiceCardWidget(QFrame):
    option_selected = Signal(str)
    custom_option_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("UserChoiceCard")
        self.setFrameShape(QFrame.NoFrame)
        self._option_buttons: list[QPushButton] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(5)

        self.title_label = QLabel("Review")
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

        self.plan_scroll = QScrollArea(self)
        self.plan_scroll.setObjectName("PlanReviewScroll")
        self.plan_scroll.setWidgetResizable(True)
        self.plan_scroll.setFrameShape(QFrame.NoFrame)
        self.plan_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plan_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.plan_scroll.setMinimumHeight(280)
        self.plan_scroll.setMaximumHeight(520)
        self.plan_card = PlanCardWidget(self.plan_scroll)
        self.plan_scroll.setWidget(self.plan_card)
        self.plan_scroll.setVisible(False)
        layout.addWidget(self.plan_scroll, 1)

        self.options_host = QWidget()
        self.options_layout = QGridLayout(self.options_host)
        self.options_layout.setContentsMargins(0, 2, 0, 0)
        self.options_layout.setHorizontalSpacing(6)
        self.options_layout.setVerticalSpacing(5)
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
        choice_type = str(payload.get("choice_type") or "").strip()
        is_plan_review = choice_type == "plan_review"
        custom_label = str(payload.get("custom_label") or "").strip() or "Custom response"
        self.plan_card.set_plan(payload if isinstance(payload.get("current_plan"), dict) else None)
        self.plan_scroll.setVisible(not self.plan_card.isHidden())
        self.plan_scroll.setMaximumHeight(520 if is_plan_review else 360)
        self.plan_scroll.setMinimumHeight(280 if is_plan_review else 180)

        self.title_label.setText("Review" if is_plan_review else "Your input is required")
        self.title_label.setVisible(not is_plan_review)
        self.question_label.setText(question or "Choose how to continue.")
        self.question_label.setVisible(bool(question) and not is_plan_review)
        if recommended_key and not is_plan_review:
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
            style = button.style()
            if style is not None:
                style.unpolish(button)
                style.polish(button)
            button.clicked.connect(lambda _checked=False, value=submit_text: self.option_selected.emit(value))
            index = len(self._option_buttons)
            if is_plan_review:
                self.options_layout.addWidget(button, index // 2, index % 2)
            else:
                self.options_layout.addWidget(button, index, 0)
            self._option_buttons.append(button)

        self.options_layout.setColumnStretch(0, 1)
        self.options_layout.setColumnStretch(1, 1 if is_plan_review else 0)
        self.custom_button.setText(custom_label)
        self.custom_button.setVisible(not is_plan_review)
        self.setVisible(bool(self._option_buttons) or bool(question) or not self.plan_scroll.isHidden())

    def clear_request(self) -> None:
        self.title_label.setText("Review")
        self.title_label.setVisible(True)
        self.question_label.clear()
        self.question_label.setVisible(True)
        self.hint_label.clear()
        self.hint_label.setVisible(False)
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._option_buttons.clear()
        self.custom_button.setText("Custom response")
        self.custom_button.setVisible(True)
        self.plan_card.clear_plan()
        self.plan_scroll.setVisible(False)
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
