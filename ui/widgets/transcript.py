from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from .foundation import TRANSCRIPT_MAX_WIDTH
from .messages import AssistantMessageWidget, NoticeWidget, RunStatsWidget, StatusIndicatorWidget, UserMessageWidget
from .tools import ToolCardWidget


class ConversationTurnWidget(QWidget):
    def __init__(
        self,
        user_text: str,
        attachments: list[dict[str, Any]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._assistant_markdown = ""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._timeline: list[tuple[str, QWidget]] = []
        self.assistant_segments: list[AssistantMessageWidget] = []
        self.tool_cards: dict[str, ToolCardWidget] = {}
        self.status_widget: StatusIndicatorWidget | None = None
        self.summary_notice_widget: NoticeWidget | None = None
        self._append_block("user", UserMessageWidget(user_text, attachments=list(attachments or []), parent=self))

    @staticmethod
    def _common_prefix_length(first: str, second: str) -> int:
        limit = min(len(first), len(second))
        index = 0
        while index < limit and first[index] == second[index]:
            index += 1
        return index

    def _append_block(self, kind: str, widget: QWidget) -> QWidget:
        if kind in {"assistant", "tool", "notice", "stats"}:
            self.clear_summary_notice()
            self.clear_status()
        self._layout.addWidget(widget)
        self._timeline.append((kind, widget))
        return widget

    def has_rendered_output(self) -> bool:
        return len(self._timeline) > 1

    def set_status(self, label: str, *, meta: str = "", phase: str = "working") -> None:
        if self.status_widget is None:
            self.status_widget = StatusIndicatorWidget(label, parent=self)
            self._layout.addWidget(self.status_widget)
        else:
            self._layout.removeWidget(self.status_widget)
            self._layout.addWidget(self.status_widget)
        self.status_widget.set_state(label, meta=meta, phase=phase)

    def clear_status(self) -> None:
        if self.status_widget is None:
            return
        self._layout.removeWidget(self.status_widget)
        self.status_widget.deleteLater()
        self.status_widget = None

    def set_summary_notice(self, message: str, level: str = "info") -> None:
        text = str(message or "").strip()
        if not text:
            return
        if self.summary_notice_widget is None:
            self.summary_notice_widget = NoticeWidget(text, level=level, parent=self)
            self._layout.addWidget(self.summary_notice_widget)
        else:
            self.summary_notice_widget.set_message(text)
            self.summary_notice_widget.set_level(level)
        if self.status_widget is not None:
            self._layout.removeWidget(self.status_widget)
            self._layout.addWidget(self.status_widget)

    def clear_summary_notice(self) -> None:
        if self.summary_notice_widget is None:
            return
        self._layout.removeWidget(self.summary_notice_widget)
        self.summary_notice_widget.deleteLater()
        self.summary_notice_widget = None

    def _ensure_assistant_segment(self) -> AssistantMessageWidget:
        if self._timeline and self._timeline[-1][0] == "assistant":
            return self._timeline[-1][1]  # type: ignore[return-value]
        segment = AssistantMessageWidget(parent=self)
        self.assistant_segments.append(segment)
        self._append_block("assistant", segment)
        return segment

    def set_assistant_markdown(self, markdown: str) -> None:
        if markdown == self._assistant_markdown and self.assistant_segments:
            return
        if not markdown:
            if self._timeline and self._timeline[-1][0] == "assistant":
                _kind, widget = self._timeline.pop()
                self._layout.removeWidget(widget)
                if widget in self.assistant_segments:
                    self.assistant_segments.remove(widget)  # type: ignore[arg-type]
                widget.deleteLater()
            self._assistant_markdown = ""
            return

        segment = self._ensure_assistant_segment()
        if not self._assistant_markdown:
            segment.set_markdown(markdown)
        elif markdown.startswith(self._assistant_markdown):
            segment_text = markdown[len(self._assistant_markdown):]
            if segment_text:
                segment.set_markdown(segment.markdown() + segment_text)
        else:
            segment.set_markdown(markdown)

        self._assistant_markdown = markdown

    def add_notice(self, message: str, level: str = "info") -> None:
        self._append_block("notice", NoticeWidget(message, level=level, parent=self))

    def add_assistant_message(self, markdown: str) -> AssistantMessageWidget:
        segment = AssistantMessageWidget(parent=self)
        segment.set_markdown(markdown)
        self.assistant_segments.append(segment)
        self._append_block("assistant", segment)
        self._assistant_markdown = markdown
        return segment

    def start_tool(self, payload: dict[str, Any]) -> ToolCardWidget:
        tool_id = payload.get("tool_id", "")
        card = self.tool_cards.get(tool_id)
        if card is None:
            card = ToolCardWidget(payload, parent=self)
            self.tool_cards[tool_id] = card
            self._append_block("tool", card)
        card.update_started_payload(payload)
        return card

    def finish_tool(self, payload: dict[str, Any], *, collapse_delay_ms: int | None = None) -> None:
        tool_id = payload.get("tool_id", "")
        card = self.tool_cards.get(tool_id)
        if card is None:
            card = self.start_tool(payload)
        card.finish(payload, collapse_delay_ms=collapse_delay_ms)

    def append_tool_output(self, payload: dict[str, Any]) -> None:
        tool_id = str(payload.get("tool_id", "") or "").strip()
        if not tool_id:
            return
        card = self.tool_cards.get(tool_id)
        if card is None:
            card = self.start_tool(
                {
                    "tool_id": tool_id,
                    "name": "cli_exec",
                    "args": {},
                    "display": "cli_exec",
                }
            )
        card.append_cli_output(
            str(payload.get("data", "") or ""),
            stream=str(payload.get("stream", "stdout") or "stdout"),
        )

    def complete(self, stats: str) -> None:
        self._append_block("stats", RunStatsWidget(stats, parent=self))

    def restore_blocks(self, blocks: list[dict[str, Any]]) -> None:
        for block in blocks:
            block_type = block.get("type")
            if block_type == "assistant":
                markdown = str(block.get("markdown", "") or "").strip()
                if markdown:
                    self.add_assistant_message(markdown)
            elif block_type == "tool":
                payload = dict(block.get("payload") or {})
                if payload:
                    self.finish_tool(payload, collapse_delay_ms=0)
            elif block_type == "notice":
                message = str(block.get("message", "") or "").strip()
                if message:
                    self.add_notice(message, str(block.get("level") or "info"))
            elif block_type == "stats":
                stats = str(block.get("stats", "") or "").strip()
                if stats:
                    self.complete(stats)

    def block_kinds(self) -> list[str]:
        return [kind for kind, _widget in self._timeline]


class ChatTranscriptWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._auto_follow_enabled = True
        self._pending_scroll = False
        self._pending_force_scroll = False
        self._programmatic_scroll = False
        self._range_follow_ticket = 0
        self._range_follow_force = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.container.setObjectName("TranscriptContainer")
        shell = QHBoxLayout(self.container)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addStretch(1)

        self.column = QWidget()
        self.column.setObjectName("TranscriptColumn")
        self.column.setMaximumWidth(TRANSCRIPT_MAX_WIDTH)
        self.column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.layout = QVBoxLayout(self.column)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.layout.addStretch(1)

        shell.addWidget(self.column, 3)
        shell.addStretch(1)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll)
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.valueChanged.connect(self._handle_scrollbar_value_changed)
        scrollbar.rangeChanged.connect(self._handle_scrollbar_range_changed)

    def clear_transcript(self) -> None:
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._auto_follow_enabled = True
        self._pending_scroll = False
        self._pending_force_scroll = False
        self._range_follow_ticket = 0
        self._range_follow_force = False

    def add_global_notice(self, message: str, level: str = "info") -> None:
        self.layout.insertWidget(self.layout.count() - 1, NoticeWidget(message, level=level, parent=self.column))
        self.notify_content_changed()

    def start_turn(self, user_text: str, attachments: list[dict[str, Any]] | None = None) -> ConversationTurnWidget:
        turn = ConversationTurnWidget(user_text, attachments=attachments, parent=self.column)
        self.layout.insertWidget(self.layout.count() - 1, turn)
        self.notify_content_changed(force=True)
        return turn

    def load_transcript(self, payload: dict[str, Any] | None) -> None:
        self.clear_transcript()
        payload = payload or {}
        summary_notice = str(payload.get("summary_notice", "") or "").strip()
        if summary_notice:
            self.add_global_notice(summary_notice, level="info")
        for turn_data in payload.get("turns", []) or []:
            user_text = str(turn_data.get("user_text", "") or "")
            attachments = list(turn_data.get("attachments", []) or [])
            turn = ConversationTurnWidget(user_text, attachments=attachments, parent=self.column)
            turn.restore_blocks(list(turn_data.get("blocks", []) or []))
            self.layout.insertWidget(self.layout.count() - 1, turn)
        self.notify_content_changed(force=True)

    @property
    def auto_follow_enabled(self) -> bool:
        return self._auto_follow_enabled

    def is_near_bottom(self, threshold: int = 28) -> bool:
        scrollbar = self.scroll.verticalScrollBar()
        return (scrollbar.maximum() - scrollbar.value()) <= max(threshold, scrollbar.pageStep() // 8)

    def _handle_scrollbar_value_changed(self, _value: int) -> None:
        if self._programmatic_scroll:
            return
        self._auto_follow_enabled = self.is_near_bottom()
        if not self._auto_follow_enabled:
            self._range_follow_force = False

    def _handle_scrollbar_range_changed(self, _minimum: int, _maximum: int) -> None:
        if not self._range_follow_ticket:
            return
        self._follow_to_bottom(self._range_follow_ticket)

    def notify_content_changed(self, *, force: bool = False) -> None:
        self.queue_scroll_to_bottom(force=force)

    def queue_scroll_to_bottom(self, *, force: bool = False) -> None:
        if force:
            self._pending_force_scroll = True
        if self._pending_scroll:
            return
        self._pending_scroll = True
        QTimer.singleShot(0, self._flush_pending_scroll)

    def _flush_pending_scroll(self) -> None:
        self._pending_scroll = False
        force = self._pending_force_scroll
        self._pending_force_scroll = False
        if not force and not self._auto_follow_enabled:
            self._range_follow_ticket += 1
            self._range_follow_force = False
            return

        self._range_follow_force = force
        self._scrollbar_to_bottom()
        self._auto_follow_enabled = True
        self._schedule_follow_up_scroll(force=force)

    def _scrollbar_to_bottom(self) -> None:
        scrollbar = self.scroll.verticalScrollBar()
        self._programmatic_scroll = True
        scrollbar.setValue(scrollbar.maximum())
        self._programmatic_scroll = False

    def _schedule_follow_up_scroll(self, *, force: bool) -> None:
        self._range_follow_force = force
        self._range_follow_ticket += 1
        ticket = self._range_follow_ticket
        follow_delays = (0, 20, 80)
        for delay in follow_delays:
            QTimer.singleShot(delay, lambda current=ticket: self._follow_to_bottom(current))
        QTimer.singleShot(max(follow_delays) + 12, lambda current=ticket: self._finish_follow_up(current))

    def _follow_to_bottom(self, ticket: int) -> None:
        if ticket != self._range_follow_ticket:
            return
        if not self._range_follow_force and not self._auto_follow_enabled:
            return
        self._scrollbar_to_bottom()
        self._auto_follow_enabled = True

    def _finish_follow_up(self, ticket: int) -> None:
        if ticket != self._range_follow_ticket:
            return
        self._range_follow_force = False

    def scroll_to_bottom(self) -> None:
        self._pending_scroll = False
        self._pending_force_scroll = False
        self._scrollbar_to_bottom()
        self._auto_follow_enabled = True
        self._schedule_follow_up_scroll(force=True)



