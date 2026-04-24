from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QMimeData, Qt, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import QApplication, QFrame, QListWidget, QListWidgetItem, QPlainTextEdit, QVBoxLayout, QWidget

from core.multimodal import can_read_image_file
from .foundation import (
    COMPOSER_MENTION_EXCLUDED_DIRS,
    COMPOSER_MENTION_MAX_ITEMS,
    COMPOSER_MENTION_POPUP_MAX_WIDTH,
    COMPOSER_MENTION_POPUP_MIN_WIDTH,
)


class ComposerTextEdit(QPlainTextEdit):
    submit_requested = Signal()
    image_pasted = Signal(object)
    image_files_pasted = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerEdit")
        self._history_by_session: dict[str, list[str]] = {}
        self._history_session_id = "__default__"
        self._history_nav_index = 0
        self._file_index: list[dict[str, Any]] = []
        self._file_index_root = ""
        self._file_index_static = False
        self._mention_popup = _ComposerMentionPopup(self)
        self._mention_popup.file_selected.connect(self._insert_selected_mention)
        self.textChanged.connect(self._refresh_mention_popup)
        self.cursorPositionChanged.connect(self._refresh_mention_popup)
        self.set_history_session("")

    def set_history_session(self, session_id: str) -> None:
        self._history_session_id = session_id or "__default__"
        self._history_by_session.setdefault(self._history_session_id, [])
        self._reset_history_navigation()
        self._close_mention_popup()

    def sync_session_history_from_transcript(self, session_id: str, transcript_payload: dict[str, Any] | None) -> None:
        key = session_id or "__default__"
        payload = transcript_payload if isinstance(transcript_payload, dict) else {}
        turns = payload.get("turns", []) or []
        history: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            user_text = str(turn.get("user_text", "") or "").strip()
            if not user_text:
                continue
            if not history or history[-1] != user_text:
                history.append(user_text)
        self._history_by_session[key] = history
        if key == self._history_session_id:
            self._reset_history_navigation()

    def append_submitted_message(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        history = self._history_for_session()
        if not history or history[-1] != value:
            history.append(value)
        self._reset_history_navigation()

    def clear_history_for_session(self, session_id: str | None = None) -> None:
        key = (session_id or self._history_session_id) or "__default__"
        self._history_by_session[key] = []
        if key == self._history_session_id:
            self._reset_history_navigation()

    def reset_history_navigation(self) -> None:
        self._reset_history_navigation()

    def format_file_reference(self, path_value: str | Path) -> str:
        path = Path(path_value)
        cwd = Path.cwd()
        try:
            normalized = str(path.relative_to(cwd).as_posix())
        except ValueError:
            normalized = str(path.as_posix())
        if " " in normalized:
            return f'"{normalized}"'
        return normalized

    def set_file_index_for_testing(self, rel_paths: list[str]) -> None:
        root = Path.cwd()
        self._file_index_root = str(root)
        self._file_index_static = True
        rows: dict[str, dict[str, Any]] = {}
        for value in rel_paths:
            relative_path = Path(value)
            self._add_index_row(rows, self._build_file_index_row(root, root / relative_path))
            for parent in relative_path.parents:
                if str(parent) in {"", "."}:
                    continue
                self._add_index_row(rows, self._build_file_index_row(root, root / parent, is_dir=True))
        self._file_index = list(rows.values())
        self._file_index.sort(key=lambda row: row["relative"])

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._mention_popup.isVisible():
            key = event.key()
            if key == Qt.Key_Up:
                event.accept()
                self._mention_popup.move_selection(-1)
                return
            if key == Qt.Key_Down:
                event.accept()
                self._mention_popup.move_selection(1)
                return
            if key in {Qt.Key_Return, Qt.Key_Enter} and not (event.modifiers() & Qt.ShiftModifier):
                event.accept()
                self._accept_current_mention()
                return
            if key == Qt.Key_Escape:
                event.accept()
                self._close_mention_popup()
                return

        history = self._history_for_session()
        is_browsing_history = 0 <= self._history_nav_index < len(history)
        if event.key() in {Qt.Key_Up, Qt.Key_Down} and (not self.toPlainText() or is_browsing_history):
            if self._navigate_history(-1 if event.key() == Qt.Key_Up else 1):
                event.accept()
                return

        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.submit_requested.emit()
            return

        if self._should_reset_history_nav(event):
            self._reset_history_navigation()

        super().keyPressEvent(event)
        self._refresh_mention_popup()

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source.hasImage():
            image = source.imageData()
            if image is not None:
                self.image_pasted.emit(image)
                self._refresh_mention_popup()
                return
        if source.hasUrls():
            parts = []
            image_paths: list[str] = []
            handled_local_urls = False
            for url in source.urls():
                if url.isLocalFile():
                    handled_local_urls = True
                    local_path = url.toLocalFile()
                    if can_read_image_file(local_path):
                        image_paths.append(local_path)
                    else:
                        parts.append(self.format_file_reference(local_path))
            if image_paths:
                self.image_files_pasted.emit(image_paths)
            if parts:
                self.insertPlainText(" ".join(parts))
            if handled_local_urls:
                self._refresh_mention_popup()
                return
        if source.hasText():
            self.insertPlainText(self._normalize_pasted_text(source.text()))
        else:
            super().insertFromMimeData(source)
        self._refresh_mention_popup()

    @staticmethod
    def _normalize_pasted_text(raw_text: str) -> str:
        text = str(raw_text or "").replace("\u2028", "\n").replace("\u2029", "\n")
        if not text:
            return text
        trimmed = text.strip("\r\n")
        if trimmed and "\n" not in trimmed and "\r" not in trimmed:
            # Clipboard providers often append a trailing line break for single-line values.
            return trimmed
        return text
    
    def focusOutEvent(self, event) -> None:  # type: ignore[override]
        self._close_mention_popup()
        super().focusOutEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        super().mouseReleaseEvent(event)
        self._refresh_mention_popup()

    def _history_for_session(self) -> list[str]:
        return self._history_by_session.setdefault(self._history_session_id, [])

    def _should_reset_history_nav(self, event: QKeyEvent) -> bool:
        if event.text():
            return True
        return event.key() in {
            Qt.Key_Backspace,
            Qt.Key_Delete,
        }

    def _reset_history_navigation(self) -> None:
        self._history_nav_index = len(self._history_for_session())

    def _set_plain_text_and_move_cursor_to_end(self, value: str) -> None:
        self.setPlainText(value)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def _navigate_history(self, direction: int) -> bool:
        history = self._history_for_session()
        if not history:
            return False

        if direction < 0:
            if self._history_nav_index > 0:
                self._history_nav_index -= 1
            else:
                self._history_nav_index = 0
        else:
            if self._history_nav_index < len(history) - 1:
                self._history_nav_index += 1
            else:
                self._history_nav_index = len(history)
                self._set_plain_text_and_move_cursor_to_end("")
                return True

        if 0 <= self._history_nav_index < len(history):
            self._set_plain_text_and_move_cursor_to_end(history[self._history_nav_index])
            return True
        return False

    def _refresh_mention_popup(self) -> None:
        mention = self._current_mention_token()
        if mention is None:
            self._close_mention_popup()
            return

        query = mention["query"].lower()
        matches = self._filter_mention_candidates(query)
        if not matches:
            self._close_mention_popup()
            return

        self._mention_popup.set_items(matches)
        self._position_mention_popup()
        if not self._mention_popup.isVisible():
            self._mention_popup.show()
        self._mention_popup.raise_()

    def _current_mention_token(self) -> dict[str, int | str] | None:
        text = self.toPlainText()
        cursor = self.textCursor()
        pos = cursor.position()
        if pos < 0:
            return None

        before_cursor = text[:pos]
        match = re.search(r"(?:^|\s)@([^\s@]*)$", before_cursor)
        if not match:
            return None

        token_start = match.start(1) - 1
        if token_start < 0:
            return None

        return {
            "start": token_start,
            "end": pos,
            "query": match.group(1),
        }

    def _ensure_file_index(self, *, force_refresh: bool = False) -> None:
        if self._file_index_static:
            return
        root = Path.cwd()
        root_str = str(root)
        if not force_refresh and self._file_index_root == root_str:
            return

        rows: dict[str, dict[str, Any]] = {}
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in COMPOSER_MENTION_EXCLUDED_DIRS]
            for dirname in dirnames:
                full_path = Path(current_root) / dirname
                self._add_index_row(rows, self._build_file_index_row(root, full_path, is_dir=True))
            for filename in filenames:
                full_path = Path(current_root) / filename
                self._add_index_row(rows, self._build_file_index_row(root, full_path))
        self._file_index = sorted(rows.values(), key=lambda row: row["relative"])
        self._file_index_root = root_str

    @staticmethod
    def _add_index_row(rows: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
        relative = str(row.get("relative") or "")
        if relative:
            rows[relative] = row

    def _build_file_index_row(self, root: Path, full_path: Path, *, is_dir: bool = False) -> dict[str, Any]:
        try:
            relative = full_path.relative_to(root).as_posix()
        except ValueError:
            relative = full_path.as_posix()
        folder = Path(relative).parent.as_posix()
        if folder == ".":
            folder = ""
        display = f"{relative}/" if is_dir and not relative.endswith("/") else relative
        return {
            "name": full_path.name,
            "name_lower": full_path.name.lower(),
            "relative": relative,
            "relative_lower": relative.lower(),
            "folder": folder,
            "depth": relative.count("/"),
            "is_dir": is_dir,
            "display": display,
        }

    def _filter_mention_candidates(self, query_lower: str) -> list[dict[str, Any]]:
        self._ensure_file_index(force_refresh=True)
        if not self._file_index:
            return []

        if not query_lower:
            return sorted(
                self._file_index,
                key=lambda row: (
                    int(row.get("depth", 0)),
                    1 if row.get("is_dir") else 0,
                    len(str(row["relative"])),
                    str(row["relative"]),
                ),
            )[:COMPOSER_MENTION_MAX_ITEMS]

        ranked: list[tuple[int, int, int, dict[str, str]]] = []
        for row in self._file_index:
            name_lower = row["name_lower"]
            relative_lower = row["relative_lower"]
            if query_lower not in name_lower and query_lower not in relative_lower:
                continue

            if name_lower.startswith(query_lower):
                rank = 0
            elif query_lower in name_lower:
                rank = 1
            elif relative_lower.startswith(query_lower):
                rank = 2
            else:
                rank = 3
            ranked.append(
                (
                    rank,
                    int(row.get("depth", 0)),
                    1 if row.get("is_dir") else 0,
                    len(row["relative"]),
                    row,
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]["relative"]))
        return [item[4] for item in ranked[:COMPOSER_MENTION_MAX_ITEMS]]

    def _position_mention_popup(self) -> None:
        anchor = self.cursorRect().bottomLeft() + QPoint(0, 6)
        global_pos = self.mapToGlobal(anchor)
        popup_size = self._mention_popup.sizeHint()
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            bounds = screen.availableGeometry()
            max_x = max(bounds.left() + 8, bounds.right() - popup_size.width() - 8)
            max_y = max(bounds.top() + 8, bounds.bottom() - popup_size.height() - 8)
            global_pos.setX(max(bounds.left() + 8, min(global_pos.x(), max_x)))
            global_pos.setY(max(bounds.top() + 8, min(global_pos.y(), max_y)))
        self._mention_popup.move(global_pos)

    def _accept_current_mention(self) -> None:
        selected = self._mention_popup.current_relative_path()
        if not selected:
            self._close_mention_popup()
            return
        self._insert_selected_mention(selected)

    def _insert_selected_mention(self, relative_path: str) -> None:
        mention = self._current_mention_token()
        if mention is None:
            self._close_mention_popup()
            return

        replacement = self.format_file_reference(relative_path)
        cursor = self.textCursor()
        cursor.setPosition(int(mention["start"]))
        cursor.setPosition(int(mention["end"]), QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(replacement)
        self.setTextCursor(cursor)
        self.setFocus()
        self._close_mention_popup()

    def _close_mention_popup(self) -> None:
        self._mention_popup.hide()


class _ComposerMentionPopup(QFrame):
    file_selected = Signal(str)

    def __init__(self, owner: QWidget) -> None:
        window_flags = Qt.Tool | Qt.FramelessWindowHint
        if hasattr(Qt, "WindowDoesNotAcceptFocus"):
            window_flags |= Qt.WindowDoesNotAcceptFocus
        super().__init__(None, window_flags)
        self._owner = owner
        self.setObjectName("ComposerMentionPopup")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumWidth(COMPOSER_MENTION_POPUP_MIN_WIDTH)
        self.setMaximumWidth(COMPOSER_MENTION_POPUP_MAX_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("ComposerMentionList")
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setTextElideMode(Qt.ElideMiddle)
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.setFocusPolicy(Qt.NoFocus)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemActivated.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def set_items(self, rows: list[dict[str, Any]]) -> None:
        self.list_widget.clear()
        max_text_width = 0
        metrics = self.list_widget.fontMetrics()
        for row in rows:
            relative = str(row["relative"])
            text = str(row.get("display") or relative)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, relative)
            item.setToolTip(relative)
            self.list_widget.addItem(item)
            max_text_width = max(max_text_width, metrics.horizontalAdvance(text))

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        row_height = self.list_widget.sizeHintForRow(0) if self.list_widget.count() > 0 else 22
        visible_count = min(8, max(1, self.list_widget.count()))
        total_height = visible_count * max(20, row_height) + 8
        self.list_widget.setFixedHeight(total_height)
        self.setFixedHeight(total_height + 8)
        target_width = max_text_width + 70
        popup_width = max(COMPOSER_MENTION_POPUP_MIN_WIDTH, min(COMPOSER_MENTION_POPUP_MAX_WIDTH, target_width))
        self.setFixedWidth(popup_width)

    def move_selection(self, delta: int) -> None:
        count = self.list_widget.count()
        if count <= 0:
            return
        current = self.list_widget.currentRow()
        if current < 0:
            current = 0
        target = max(0, min(count - 1, current + delta))
        self.list_widget.setCurrentRow(target)

    def current_relative_path(self) -> str:
        item = self.list_widget.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        relative = str(item.data(Qt.UserRole) or "")
        if relative:
            self.file_selected.emit(relative)


