from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qtawesome as qta
from PySide6.QtCore import QMimeData, QRegularExpression, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QKeySequence, QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextFormat
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QTextBrowser, QTextEdit, QToolButton, QVBoxLayout, QWidget

from ui.theme import (
    ACCENT_BLUE,
    AMBER_WARNING,
    BORDER,
    ERROR_RED,
    MONO_FONT_FAMILY,
    SURFACE_ALT,
    SURFACE_BG,
    SUCCESS_GREEN,
    SURFACE_CARD,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

FENCED_BLOCK_RE = re.compile(r"```([\w+-]*)\r?\n(.*?)```", re.DOTALL)
DIFF_HUNK_HEADER_RE = re.compile(r"^@@ -(?P<old>\d+)(?:,\d+)? \+(?P<new>\d+)(?:,\d+)? @@")
RENDERED_DIFF_LINE_RE = re.compile(r"^\s*\d*\s+\d*\s(?P<marker>[+\- ])\s")
# Keep the transcript/composer column comfortably readable on wide displays.
TRANSCRIPT_MAX_WIDTH = 900
USER_MESSAGE_COLLAPSE_CHAR_LIMIT = 420
USER_MESSAGE_COLLAPSE_LINE_LIMIT = 8
COMPOSER_MENTION_MAX_ITEMS = 50
COMPOSER_MENTION_EXCLUDED_DIRS = {".git", "venv", "__pycache__", ".agent_state", "dist"}
COMPOSER_MENTION_POPUP_MIN_WIDTH = 560
COMPOSER_MENTION_POPUP_MAX_WIDTH = 700
CLI_EXEC_MIN_VISIBLE_MS = 800
CLI_EXEC_SUCCESS_FLASH_MS = 250
DIFF_ADD_LINE_BG = QColor("#1E3425")
DIFF_REMOVE_LINE_BG = QColor("#472B2B")


def _make_mono_font() -> QFont:
    font = QFont(MONO_FONT_FAMILY)
    if not font.exactMatch():
        font = QFont("Consolas")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(10)
    return font


def _normalize_copied_text(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace("\u2029", "\n").replace("\u2028", "\n")
    normalized = normalized.replace("\u00A0", " ")
    stripped = normalized.strip("\r\n")
    if stripped and "\n" not in stripped and "\r" not in stripped:
        return stripped
    return normalized


class CopySafePlainTextEdit(QPlainTextEdit):
    def createMimeDataFromSelection(self) -> QMimeData:  # type: ignore[override]
        mime = QMimeData()
        cursor = self.textCursor()
        if cursor.hasSelection():
            mime.setText(_normalize_copied_text(cursor.selectedText()))
        return mime

    def copy(self) -> None:  # type: ignore[override]
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        QApplication.clipboard().setText(_normalize_copied_text(cursor.selectedText()))

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy()
            event.accept()
            return
        super().keyPressEvent(event)


def _fa_icon(name: str, *, color: str = TEXT_MUTED, size: int = 14, **kwargs: Any) -> QIcon:
    safe_size = max(8, int(size))
    icon = qta.icon(name, color=color, **kwargs)
    pixmap = icon.pixmap(safe_size, safe_size)
    if pixmap.isNull():
        return icon
    return QIcon(pixmap)


def _collapsed_user_message_text(text: str) -> tuple[str, bool]:
    raw_text = str(text)
    preview_lines = raw_text.splitlines()
    line_limited = len(preview_lines) > USER_MESSAGE_COLLAPSE_LINE_LIMIT
    preview = "\n".join(preview_lines[:USER_MESSAGE_COLLAPSE_LINE_LIMIT]) if line_limited else raw_text
    char_limited = len(preview) > USER_MESSAGE_COLLAPSE_CHAR_LIMIT
    if char_limited:
        preview = preview[: USER_MESSAGE_COLLAPSE_CHAR_LIMIT].rstrip()
    is_collapsed = line_limited or char_limited or len(preview) < len(raw_text)
    if is_collapsed:
        preview = preview.rstrip()
        if preview:
            preview += "…"
    return preview or raw_text, is_collapsed


def _sync_plain_text_height(
    editor: QPlainTextEdit,
    *,
    min_lines: int = 2,
    max_lines: int = 12,
    extra_padding: int = 16,
) -> None:
    metrics = QFontMetrics(editor.font())
    line_count = max(min_lines, min(editor.blockCount(), max_lines))
    editor.setFixedHeight(line_count * metrics.lineSpacing() + extra_padding)


def _normalize_display_path(path_value: str) -> str:
    value = str(path_value or "").strip()
    if not value:
        return ""
    try:
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.relative_to(Path.cwd()).as_posix()
            except ValueError:
                return candidate.as_posix()
        return candidate.as_posix()
    except Exception:
        return value.replace("\\", "/")


def _extract_diff_path(diff_text: str, fallback_path: str = "") -> str:
    fallback = _normalize_display_path(fallback_path)
    if fallback:
        return fallback
    for raw_line in str(diff_text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                right = parts[3].removeprefix("b/")
                if right:
                    return right
        if line.startswith("+++ "):
            candidate = line[4:].strip().removeprefix("b/")
            if candidate and candidate != "/dev/null":
                return candidate
    return "diff"


def _render_diff_with_line_numbers(diff_text: str) -> tuple[str, int, int]:
    lines = str(diff_text or "").splitlines()
    if not lines:
        return "", 0, 0

    rendered_lines: list[str] = []
    added = 0
    removed = 0
    old_line: int | None = None
    new_line: int | None = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        hunk_match = DIFF_HUNK_HEADER_RE.match(line)
        if hunk_match:
            old_line = int(hunk_match.group("old"))
            new_line = int(hunk_match.group("new"))
            continue

        first_char = line[:1]
        is_add = first_char == "+"
        is_del = first_char == "-"
        is_meta = (
            line.startswith("@@")
            or line.startswith("diff --git")
            or line.startswith("index ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("\\ No newline")
        )

        if is_meta:
            continue

        marker = " "
        payload = line
        if is_add:
            marker = "+"
            payload = line[1:]
        elif is_del:
            marker = "-"
            payload = line[1:]
        elif line.startswith(" "):
            payload = line[1:]

        old_number = ""
        new_number = ""
        if is_add:
            added += 1
            new_number = str(new_line) if new_line is not None else ""
            rendered_lines.append(f"{old_number:>6} {new_number:>6} {marker} {payload}")
            if new_line is not None:
                new_line += 1
            continue

        if is_del:
            removed += 1
            old_number = str(old_line) if old_line is not None else ""
            rendered_lines.append(f"{old_number:>6} {new_number:>6} {marker} {payload}")
            if old_line is not None:
                old_line += 1
            continue

        old_number = str(old_line) if old_line is not None else ""
        new_number = str(new_line) if new_line is not None else ""
        rendered_lines.append(f"{old_number:>6} {new_number:>6} {marker} {payload}")
        if old_line is not None:
            old_line += 1
        if new_line is not None:
            new_line += 1

    return "\n".join(rendered_lines), added, removed


def _build_full_width_diff_selections(editor: QPlainTextEdit) -> list[QTextEdit.ExtraSelection]:
    selections: list[QTextEdit.ExtraSelection] = []
    block = editor.document().firstBlock()
    while block.isValid():
        marker_match = RENDERED_DIFF_LINE_RE.match(block.text())
        marker = marker_match.group("marker") if marker_match else ""
        if marker == "+":
            background = DIFF_ADD_LINE_BG
        elif marker == "-":
            background = DIFF_REMOVE_LINE_BG
        else:
            background = None

        if background is not None:
            selection = QTextEdit.ExtraSelection()
            selection.cursor = QTextCursor(block)
            selection.cursor.clearSelection()
            selection.format.setBackground(background)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selections.append(selection)
        block = block.next()
    return selections


class AutoTextBrowser(QTextBrowser):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._height_sync_pending = False
        self._last_markdown = ""
        self._last_height = 0
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.document().setDocumentMargin(0)
        self.document().documentLayout().documentSizeChanged.connect(self._queue_height_sync)

    def setMarkdown(self, markdown: str) -> None:  # type: ignore[override]
        if markdown == self._last_markdown:
            return
        self._last_markdown = markdown
        super().setMarkdown(markdown)
        self._queue_height_sync()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._queue_height_sync()

    def createMimeDataFromSelection(self) -> QMimeData:  # type: ignore[override]
        mime = QMimeData()
        cursor = self.textCursor()
        if cursor.hasSelection():
            mime.setText(_normalize_copied_text(cursor.selectedText()))
        return mime

    def copy(self) -> None:  # type: ignore[override]
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        QApplication.clipboard().setText(_normalize_copied_text(cursor.selectedText()))

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy()
            event.accept()
            return
        super().keyPressEvent(event)

    def _queue_height_sync(self, *_args) -> None:
        if self._height_sync_pending:
            return
        self._height_sync_pending = True
        QTimer.singleShot(0, self._sync_height)

    def _sync_height(self) -> None:
        self._height_sync_pending = False
        try:
            document = self.document()
            layout = document.documentLayout() if document is not None else None
            if layout is None:
                return
            doc_height = int(layout.documentSize().height())
        except RuntimeError:
            # A queued resize sync can fire after the Qt object was already deleted.
            return
        target_height = max(28, doc_height + 8)
        if target_height == self._last_height:
            return
        self._last_height = target_height
        try:
            self.setFixedHeight(target_height)
            self.updateGeometry()
        except RuntimeError:
            return


class ElidedLabel(QLabel):
    def __init__(self, parent: QWidget | None = None, *, elide_mode: Qt.TextElideMode = Qt.ElideRight) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = elide_mode
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(0)

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text or "")
        self._update_elided_text()

    def full_text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        text = self._full_text
        if not text:
            super().setText("")
            self.setToolTip("")
            return
        available = max(0, self.contentsRect().width())
        if available <= 0:
            super().setText("")
            self.setToolTip(text)
            return
        elided = self.fontMetrics().elidedText(text, self._elide_mode, available)
        super().setText(elided)
        self.setToolTip(text if elided != text else "")


class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, document, language: str = "") -> None:
        super().__init__(document)
        self.language = (language or "").lower()
        self.rules: list[tuple[QRegularExpression, QTextCharFormat]] = []
        self._build_rules()

    def _build_rules(self) -> None:
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#7CC7FF"))
        keyword_format.setFontWeight(QFont.Bold)

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#A8E6A2"))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#8093A7"))
        comment_format.setFontItalic(True)

        if self.language == "diff":
            return

        for pattern in (
            r"\bclass\b",
            r"\bdef\b",
            r"\breturn\b",
            r"\bif\b",
            r"\belse\b",
            r"\belif\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bimport\b",
            r"\bfrom\b",
            r"\basync\b",
            r"\bawait\b",
            r"\btry\b",
            r"\bexcept\b",
            r"\bconst\b",
            r"\blet\b",
            r"\bfunction\b",
        ):
            self.rules.append((QRegularExpression(pattern), keyword_format))
        self.rules.append((QRegularExpression(r"\".*?\""), string_format))
        self.rules.append((QRegularExpression(r"'.*?'"), string_format))
        self.rules.append((QRegularExpression(r"#.*$"), comment_format))
        self.rules.append((QRegularExpression(r"//.*$"), comment_format))

    def highlightBlock(self, text: str) -> None:
        if self.language == "diff":
            text_format = QTextCharFormat()
            line_bg = None
            marker_match = RENDERED_DIFF_LINE_RE.match(text)
            marker = marker_match.group("marker") if marker_match else ""

            if marker == "+":
                text_format.setForeground(QColor("#8FE388"))
                line_bg = DIFF_ADD_LINE_BG
                text_format.setBackground(line_bg)
            elif marker == "-":
                text_format.setForeground(QColor("#FF8B8B"))
                line_bg = DIFF_REMOVE_LINE_BG
                text_format.setBackground(line_bg)
            elif (
                text.lstrip().startswith("@@")
                or text.lstrip().startswith("diff --git")
                or text.lstrip().startswith("index ")
                or text.lstrip().startswith("--- ")
                or text.lstrip().startswith("+++ ")
            ):
                text_format.setForeground(QColor("#7CC7FF"))
                text_format.setFontWeight(QFont.Bold)
            if text_format.foreground().color().isValid():
                self.setFormat(0, len(text), text_format)

            if marker_match:
                number_format = QTextCharFormat()
                number_format.setForeground(QColor("#747C89"))
                if line_bg is not None:
                    number_format.setBackground(line_bg)
                self.setFormat(
                    0,
                    marker_match.start("marker"),
                    number_format,
                )
            return

        for expression, text_format in self.rules:
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), text_format)


class CodeBlockWidget(QWidget):
    def __init__(self, code: str, language: str = "", title: str = "") -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("TranscriptMeta")
        self.title_label.setVisible(bool(title))
        title_row.addWidget(self.title_label, 0, Qt.AlignVCenter)
        title_row.addStretch(1)

        self.copy_button = QToolButton()
        self.copy_button.setObjectName("CodeCopyButton")
        self.copy_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.setToolTip("Copy")
        self.copy_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_MUTED, size=13))
        self.copy_button.clicked.connect(self._copy_code)
        title_row.addWidget(self.copy_button, 0, Qt.AlignVCenter)

        layout.addLayout(title_row)

        self.editor = CopySafePlainTextEdit()
        self.editor.setObjectName("CodeView")
        self.editor.setReadOnly(True)
        self.editor.setFont(_make_mono_font())
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        
        self.highlighter = CodeHighlighter(self.editor.document(), language)
        layout.addWidget(self.editor)
        
        self.set_code(code, language, title)

    def set_code(self, code: str, language: str = "", title: str = "") -> None:
        if title:
            self.title_label.setText(title)
            self.title_label.setVisible(True)
        else:
            self.title_label.setVisible(False)

        if self.highlighter.language != language.lower():
            self.highlighter.setDocument(None)
            self.highlighter = CodeHighlighter(self.editor.document(), language)

        if self.editor.toPlainText() != code:
            self.editor.setPlainText(code)
        self._sync_height()

    def _sync_height(self) -> None:
        _sync_plain_text_height(self.editor, min_lines=2, max_lines=30, extra_padding=18)

    def _copy_code(self) -> None:
        QApplication.clipboard().setText(self.editor.toPlainText())
        self.copy_button.setText("Copied")
        self.copy_button.setIcon(_fa_icon("fa5s.check", color=SUCCESS_GREEN, size=13))
        QTimer.singleShot(1200, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        self.copy_button.setText("Copy")
        self.copy_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_MUTED, size=13))


class DiffBlockWidget(QFrame):
    def __init__(self, diff_text: str, source_path: str = "") -> None:
        super().__init__()
        self.setObjectName("DiffPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._raw_diff = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(10, 7, 10, 7)
        header_row.setSpacing(8)

        self.path_label = QLabel("diff")
        self.path_label.setObjectName("DiffHeaderPath")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header_row.addWidget(self.path_label, 1)

        self.added_label = QLabel("+0")
        self.added_label.setObjectName("DiffStatAdded")
        header_row.addWidget(self.added_label, 0, Qt.AlignVCenter)

        self.removed_label = QLabel("-0")
        self.removed_label.setObjectName("DiffStatRemoved")
        header_row.addWidget(self.removed_label, 0, Qt.AlignVCenter)

        self.copy_button = QToolButton()
        self.copy_button.setObjectName("CodeCopyButton")
        self.copy_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.setToolTip("Copy diff")
        self.copy_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_MUTED, size=13))
        self.copy_button.clicked.connect(self._copy_diff)
        header_row.addWidget(self.copy_button, 0, Qt.AlignVCenter)
        layout.addLayout(header_row)

        self.editor = CopySafePlainTextEdit()
        self.editor.setObjectName("DiffCodeView")
        self.editor.setReadOnly(True)
        self.editor.setFont(_make_mono_font())
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.highlighter = CodeHighlighter(self.editor.document(), "diff")
        layout.addWidget(self.editor)

        self.set_diff(diff_text, source_path=source_path)

    def set_diff(self, diff_text: str, source_path: str = "") -> None:
        self._raw_diff = str(diff_text or "")
        rendered, added, removed = _render_diff_with_line_numbers(self._raw_diff)
        self.path_label.setText(_extract_diff_path(self._raw_diff, source_path))
        self.added_label.setText(f"+{added}")
        self.removed_label.setText(f"-{removed}")
        display_text = rendered if rendered else self._raw_diff
        if self.editor.toPlainText() != display_text:
            self.editor.setPlainText(display_text)
        self.editor.setExtraSelections(_build_full_width_diff_selections(self.editor))
        _sync_plain_text_height(self.editor, min_lines=3, max_lines=26, extra_padding=18)

    def _copy_diff(self) -> None:
        QApplication.clipboard().setText(self._raw_diff)
        self.copy_button.setText("Copied")
        self.copy_button.setIcon(_fa_icon("fa5s.check", color=SUCCESS_GREEN, size=13))
        QTimer.singleShot(1200, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        self.copy_button.setText("")
        self.copy_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_MUTED, size=13))


class CollapsibleSection(QFrame):
    def __init__(
        self,
        title: str,
        content: QWidget,
        expanded: bool = False,
        indent: int = 0,
        content_margins: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("ToolExpandablePanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("ToolExpandableToggle")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setFlat(True)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setMinimumHeight(20)
        self.toggle_button.setIconSize(QSize(8, 8))
        self._set_toggle_icon(expanded)

        self.content = content
        self.content_container = QWidget()
        self.content_container.setObjectName("ToolExpandableContent")
        self.content_container.setAttribute(Qt.WA_StyledBackground, True)
        content_layout = QHBoxLayout(self.content_container)
        if content_margins is None:
            content_margins = (8 + indent, 0, 8, 8)
        content_layout.setContentsMargins(*content_margins)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.content, 1)
        self.content_container.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_container)
        self.toggle_button.toggled.connect(self.set_expanded)

    def _set_toggle_icon(self, expanded: bool) -> None:
        icon_color = TEXT_PRIMARY if expanded else TEXT_MUTED
        self.toggle_button.setIcon(
            _fa_icon("fa5s.caret-down" if expanded else "fa5s.caret-right", color=icon_color, size=10)
        )

    def set_expanded(self, expanded: bool) -> None:
        if self.toggle_button.isChecked() != expanded:
            self.toggle_button.blockSignals(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.blockSignals(False)
        self._set_toggle_icon(expanded)
        self.content_container.setVisible(expanded)



