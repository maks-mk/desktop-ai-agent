from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import qtawesome as qta
from PySide6.QtCore import QAbstractListModel, QMimeData, QModelIndex, QPoint, QRect, QRegularExpression, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeyEvent, QPainter, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTextBrowser,
    QToolButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.ui_theme import (
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
TRANSCRIPT_MAX_WIDTH = 1180
USER_MESSAGE_COLLAPSE_CHAR_LIMIT = 420
USER_MESSAGE_COLLAPSE_LINE_LIMIT = 8


def _make_mono_font() -> QFont:
    font = QFont(MONO_FONT_FAMILY)
    if not font.exactMatch():
        font = QFont("Consolas")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(10)
    return font

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
            if text.startswith("+"):
                text_format.setForeground(QColor("#8FE388"))
            elif text.startswith("-"):
                text_format.setForeground(QColor("#FF8B8B"))
            elif text.startswith("@@"):
                text_format.setForeground(QColor("#7CC7FF"))
                text_format.setFontWeight(QFont.Bold)
            if text_format.foreground().color().isValid():
                self.setFormat(0, len(text), text_format)
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
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("TranscriptMeta")
        self.title_label.setVisible(bool(title))
        layout.addWidget(self.title_label)

        self.editor = QPlainTextEdit()
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
        

class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = False, indent: int = 16) -> None:
        super().__init__()
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("DisclosureButton")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setFlat(True)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setMinimumHeight(18)
        self.toggle_button.setIconSize(QSize(8, 8))
        self._set_toggle_icon(expanded)

        self.content = content
        self.content_container = QWidget()
        content_layout = QHBoxLayout(self.content_container)
        content_layout.setContentsMargins(indent, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.content, 1)
        self.content_container.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_container)
        self.toggle_button.toggled.connect(self.set_expanded)

    def _set_toggle_icon(self, expanded: bool) -> None:
        self.toggle_button.setIcon(
            qta.icon("fa5s.caret-down" if expanded else "fa5s.caret-right", color=TEXT_MUTED)
        )

    def set_expanded(self, expanded: bool) -> None:
        if self.toggle_button.isChecked() != expanded:
            self.toggle_button.blockSignals(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.blockSignals(False)
        self._set_toggle_icon(expanded)
        self.content_container.setVisible(expanded)


class ComposerTextEdit(QPlainTextEdit):
    submit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerEdit")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and not (event.modifiers() & Qt.ShiftModifier):
            event.accept()
            self.submit_requested.emit()
            return
        super().keyPressEvent(event)

    def insertFromMimeData(self, source: QMimeData) -> None:  # type: ignore[override]
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super().insertFromMimeData(source)


def _format_sidebar_time(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    local_dt = dt.astimezone()
    now = datetime.now(local_dt.tzinfo or timezone.utc)
    delta = now - local_dt
    if delta.total_seconds() < 0:
        return "сейчас"
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "сейчас"
    if minutes < 60:
        return f"{minutes}м"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}ч"
    days = hours // 24
    if days < 7:
        return f"{days}д"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks}н"
    return local_dt.strftime("%d %b")


def _sidebar_project_name(project_path: str) -> str:
    text = str(project_path or "").replace("\\", "/").rstrip("/")
    if not text:
        return "project"
    return text.split("/")[-1] or text


def _sidebar_dt(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SessionListModel(QAbstractListModel):
    KindRole = Qt.UserRole + 1
    SessionIdRole = Qt.UserRole + 2
    TitleRole = Qt.UserRole + 3
    UpdatedAtRole = Qt.UserRole + 4
    ProjectPathRole = Qt.UserRole + 5
    ProjectTitleRole = Qt.UserRole + 6

    def __init__(self) -> None:
        super().__init__()
        self._items: list[dict[str, str]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # type: ignore[override]
        if parent.isValid():
            return 0
        return len(self._items)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # type: ignore[override]
        if not index.isValid():
            return Qt.NoItemFlags
        kind = str(self.data(index, self.KindRole) or "session")
        if kind == "group":
            return Qt.ItemIsEnabled
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        item = self._items[index.row()]
        if role == self.KindRole:
            return item.get("kind", "session")
        if role == Qt.DisplayRole:
            return item.get("title", "")
        if role == self.SessionIdRole:
            return item.get("session_id", "")
        if role == self.TitleRole:
            return item.get("title", "")
        if role == self.UpdatedAtRole:
            return item.get("updated_at", "")
        if role == self.ProjectPathRole:
            return item.get("project_path", "")
        if role == self.ProjectTitleRole:
            return item.get("project_title", "")
        return None

    def set_sessions(self, sessions: list[dict[str, str]]) -> None:
        grouped: dict[str, list[dict[str, str]]] = {}
        for raw in sessions:
            row = dict(raw)
            project_key = str(row.get("project_path", "")).strip()
            grouped.setdefault(project_key, []).append(row)

        project_rows: list[tuple[str, list[dict[str, str]]]] = sorted(
            grouped.items(),
            key=lambda pair: max((_sidebar_dt(item.get("updated_at", "")) for item in pair[1]), default=datetime.fromtimestamp(0, tz=timezone.utc)),
            reverse=True,
        )

        items: list[dict[str, str]] = []
        for project_path, rows in project_rows:
            items.append(
                {
                    "kind": "group",
                    "project_path": project_path,
                    "project_title": _sidebar_project_name(project_path),
                    "title": _sidebar_project_name(project_path),
                    "session_id": "",
                    "updated_at": "",
                }
            )
            sorted_rows = sorted(
                rows,
                key=lambda item: _sidebar_dt(item.get("updated_at", "")),
                reverse=True,
            )
            for row in sorted_rows:
                entry = dict(row)
                entry["kind"] = "session"
                entry["project_title"] = _sidebar_project_name(project_path)
                items.append(entry)

        self.beginResetModel()
        self._items = items
        self.endResetModel()

    def session_id_at(self, index: QModelIndex) -> str:
        if not index.isValid():
            return ""
        if str(self.data(index, self.KindRole) or "session") != "session":
            return ""
        return str(self.data(index, self.SessionIdRole) or "")

    def index_for_session(self, session_id: str) -> QModelIndex:
        if not session_id:
            return QModelIndex()
        for row, item in enumerate(self._items):
            if item.get("kind") == "session" and item.get("session_id") == session_id:
                return self.index(row, 0)
        return QModelIndex()

    def session_row_count(self) -> int:
        return sum(1 for item in self._items if item.get("kind") == "session")

    def title_for_session(self, session_id: str) -> str:
        if not session_id:
            return ""
        for item in self._items:
            if item.get("kind") == "session" and item.get("session_id") == session_id:
                return str(item.get("title", "") or "")
        return ""


class SessionItemDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # type: ignore[override]
        painter.save()
        item_kind = str(index.data(SessionListModel.KindRole) or "session")
        rect = option.rect.adjusted(6, 2, -6, -2)
        is_selected = bool(option.state & QStyle.State_Selected)
        is_hovered = bool(option.state & QStyle.State_MouseOver)

        if item_kind == "group":
            icon_rect = QRect(rect.left() + 2, rect.top() + 4, 14, 14)
            painter.drawPixmap(icon_rect, qta.icon("fa5.folder-open", color=TEXT_MUTED).pixmap(12, 12))
            group_text = str(index.data(SessionListModel.ProjectTitleRole) or index.data(SessionListModel.TitleRole) or "")
            title_rect = QRect(rect.left() + 20, rect.top(), rect.width() - 24, rect.height())
            title_font = option.font
            title_font.setPointSize(10)
            title_font.setWeight(QFont.DemiBold)
            painter.setFont(title_font)
            painter.setPen(QColor(TEXT_MUTED))
            painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, group_text)
            painter.restore()
            return

        background = QColor(0, 0, 0, 0)
        if is_selected:
            background = QColor(SURFACE_ALT)
        elif is_hovered:
            background = QColor(SURFACE_ALT)
            background.setAlpha(120)
        if background.alpha() > 0:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, 9, 9)

        title = str(index.data(SessionListModel.TitleRole) or "")
        updated_at = _format_sidebar_time(str(index.data(SessionListModel.UpdatedAtRole) or ""))
        title_font = option.font
        title_font.setPointSize(10.5)
        title_font.setWeight(QFont.DemiBold if is_selected else QFont.Medium)
        painter.setFont(title_font)
        painter.setPen(QColor(TEXT_PRIMARY))

        time_font = option.font
        time_font.setPointSize(9)
        time_font.setWeight(QFont.Medium)
        time_metrics = QFontMetrics(time_font)
        time_width = max(36, time_metrics.horizontalAdvance(updated_at) + 6)

        title_rect = QRect(rect.left() + 12, rect.top() + 1, rect.width() - 20 - time_width, rect.height() - 2)
        time_rect = QRect(rect.right() - time_width - 8, rect.top() + 1, time_width, rect.height() - 2)
        metrics = QFontMetrics(title_font)
        painter.drawText(
            title_rect,
            Qt.AlignLeft | Qt.AlignVCenter,
            metrics.elidedText(title, Qt.ElideRight, max(10, title_rect.width())),
        )

        painter.setFont(time_font)
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(time_rect, Qt.AlignRight | Qt.AlignVCenter, updated_at)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # type: ignore[override]
        _ = option
        kind = str(index.data(SessionListModel.KindRole) or "session")
        return QSize(240, 26 if kind == "group" else 36)


class SessionSidebarWidget(QWidget):
    session_activated = Signal(str)
    session_delete_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        title = QLabel("Беседы")
        title.setObjectName("SidebarSectionTitle")
        root.addWidget(title)

        self.list_view = QListView()
        self.list_view.setObjectName("SessionListView")
        self.list_view.setMouseTracking(True)
        self.list_view.setUniformItemSizes(False)
        self.list_view.setEditTriggers(QListView.NoEditTriggers)
        self.list_view.setSelectionMode(QListView.SingleSelection)
        self.list_view.setSelectionBehavior(QListView.SelectRows)
        self.list_view.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_view.setContextMenuPolicy(Qt.CustomContextMenu)

        self.model = SessionListModel()
        self.delegate = SessionItemDelegate()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)
        self.list_view.clicked.connect(self._emit_clicked_session)
        self.list_view.customContextMenuRequested.connect(self._show_context_menu)
        root.addWidget(self.list_view, 1)

    def set_sessions(self, sessions: list[dict[str, str]], active_session_id: str) -> None:
        self.model.set_sessions(sessions)
        if not active_session_id:
            self.list_view.clearSelection()
            return
        index = self.model.index_for_session(active_session_id)
        if index.isValid():
            self.list_view.setCurrentIndex(index)
            self.list_view.scrollTo(index)
            return
        self.list_view.clearSelection()

    def _emit_clicked_session(self, index: QModelIndex) -> None:
        session_id = self.model.session_id_at(index)
        if session_id:
            self.session_activated.emit(session_id)

    def title_for_session(self, session_id: str) -> str:
        return self.model.title_for_session(session_id)

    def _show_context_menu(self, pos: QPoint) -> None:
        index = self.list_view.indexAt(pos)
        if not index.isValid():
            return
        if str(index.data(SessionListModel.KindRole) or "session") != "session":
            return

        session_id = self.model.session_id_at(index)
        if not session_id:
            return

        menu = QMenu(self.list_view)
        delete_action = menu.addAction("Удалить чат")
        selected = menu.exec(self.list_view.viewport().mapToGlobal(pos))
        if selected is delete_action:
            self.session_delete_requested.emit(session_id)


class OverviewPanelWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        frame = QFrame()
        frame.setObjectName("SidebarCard")
        form = QFormLayout(frame)
        form.setContentsMargins(12, 12, 12, 12)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        self._labels: dict[str, QLabel] = {}
        for key in (
            "Provider",
            "Model",
            "Backend",
            "Tools",
            "Session",
            "Thread",
            "Approvals",
            "MCP",
            "Status",
            "Config",
        ):
            label = QLabel("—")
            label.setWordWrap(True)
            self._labels[key] = label
            form.addRow(key, label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(frame)
        layout.addStretch(1)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        mapping = {
            "Provider": snapshot.get("provider", "—"),
            "Model": snapshot.get("model", "—"),
            "Backend": snapshot.get("backend", "—"),
            "Tools": str(snapshot.get("tools_count", "—")),
            "Session": snapshot.get("session_short", "—"),
            "Thread": snapshot.get("thread_short", "—"),
            "Approvals": snapshot.get("approvals", "—"),
            "MCP": snapshot.get("mcp_text", "—"),
            "Status": snapshot.get("status", "—"),
            "Config": snapshot.get("config_mode", "—"),
        }
        for key, value in mapping.items():
            self._labels[key].setText(str(value))


class ToolsPanelWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container.setObjectName("ToolsContainer")
        self._inner = QVBoxLayout(self._container)
        self._inner.setContentsMargins(6, 6, 6, 6)
        self._inner.setSpacing(2)
        self._inner.addStretch(1)

        self.scroll.setWidget(self._container)
        root.addWidget(self.scroll)

    def set_tools(self, tools: list[dict[str, str]]) -> None:
        while self._inner.count() > 1:
            item = self._inner.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        grouped: dict[str, list[dict[str, str]]] = {"Read-only": [], "Protected": [], "MCP": []}
        for row in tools:
            grouped.setdefault(row["group"], []).append(row)

        group_colors = {
            "Read-only": TEXT_MUTED,
            "Protected": AMBER_WARNING,
            "MCP": ACCENT_BLUE,
        }

        insert_pos = 0
        for group_name in ("Read-only", "Protected", "MCP"):
            items = grouped.get(group_name, [])
            if not items:
                continue

            header = QLabel(group_name.upper())
            header.setStyleSheet(
                f"color: {group_colors[group_name]}; font-size: 7.2pt; "
                f"font-weight: 700; letter-spacing: 0.8px; "
                f"padding: 8px 4px 3px 4px;"
            )
            self._inner.insertWidget(insert_pos, header)
            insert_pos += 1

            for row in items:
                card = QFrame()
                card.setObjectName("ToolCard")
                card.setStyleSheet(
                    f"QFrame#ToolCard {{ background: {SURFACE_CARD}; "
                    f"border: 1px solid {BORDER}; border-radius: 6px; "
                    f"margin: 1px 0px; }}"
                )
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(8, 6, 8, 6)
                card_layout.setSpacing(3)

                top_row = QHBoxLayout()
                top_row.setSpacing(6)

                name_label = QLabel(row["name"])
                name_label.setStyleSheet(
                    f"color: {TEXT_PRIMARY}; font-weight: 600; "
                    f"font-size: 8.8pt; font-family: 'Cascadia Mono';"
                )
                top_row.addWidget(name_label, 1)

                flags = row.get("flags", "")
                if flags:
                    for flag in flags.split(", "):
                        flag = flag.strip()
                        if not flag:
                            continue
                        flag_color = (
                            AMBER_WARNING if flag in ("mutating", "destructive", "approval")
                            else ACCENT_BLUE if flag in ("mcp", "network")
                            else TEXT_MUTED
                        )
                        chip = QLabel(flag)
                        chip.setStyleSheet(
                            f"color: {flag_color}; font-size: 7pt; "
                            f"border: 1px solid {flag_color}33; "
                            f"border-radius: 3px; padding: 0px 4px;"
                        )
                        top_row.addWidget(chip, 0)

                card_layout.addLayout(top_row)

                desc = row.get("description", "")
                if desc:
                    desc_label = QLabel(desc)
                    desc_label.setWordWrap(True)
                    desc_label.setStyleSheet(
                        f"color: {TEXT_MUTED}; font-size: 8pt;"
                    )
                    card_layout.addWidget(desc_label)

                self._inner.insertWidget(insert_pos, card)
                insert_pos += 1

            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setStyleSheet(f"background: {BORDER}; margin: 4px 0px;")
            self._inner.insertWidget(insert_pos, sep)
            insert_pos += 1


class InfoPopupDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InfoPopup")
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(470, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        title = QLabel("Session information")
        title.setObjectName("SectionTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)

        hint = QLabel("Esc or click outside to close")
        hint.setObjectName("MetaText")
        title_row.addWidget(hint, 0, Qt.AlignRight)
        layout.addLayout(title_row)

        self.tabs = QTabWidget()
        self.overview_panel = OverviewPanelWidget()
        self.tools_panel = ToolsPanelWidget()

        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_layout.setSpacing(0)
        self.help_text = QTextBrowser()
        self.help_text.setOpenLinks(False)
        self.help_text.setOpenExternalLinks(False)
        self.help_text.setReadOnly(True)
        help_layout.addWidget(self.help_text)

        self.tabs.addTab(self.overview_panel, qta.icon("fa5s.info-circle", color=ACCENT_BLUE), "Info")
        self.tabs.addTab(self.tools_panel, qta.icon("fa5s.tools", color=ACCENT_BLUE), "Tools")
        self.tabs.addTab(help_widget, qta.icon("fa5s.question-circle", color=ACCENT_BLUE), "Help")
        layout.addWidget(self.tabs, 1)


class NoticeWidget(QFrame):
    def __init__(self, message: str, level: str = "info") -> None:
        super().__init__()
        self.setObjectName("FlatNoticeRow")
        self.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(5)

        icon_name = "fa5s.info-circle"
        color = ACCENT_BLUE
        if level == "warning":
            icon_name = "fa5s.exclamation-triangle"
            color = AMBER_WARNING
        elif level == "error":
            icon_name = "fa5s.times-circle"
            color = ERROR_RED
        elif level == "success":
            icon_name = "fa5s.check-circle"
            color = SUCCESS_GREEN

        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color=color).pixmap(11, 11))
        layout.addWidget(icon_label, 0, Qt.AlignTop)

        text_label = QLabel(message)
        text_label.setObjectName("MetaText")
        text_label.setWordWrap(True)
        layout.addWidget(text_label, 1)


class RunStatsWidget(QWidget):
    def __init__(self, stats: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(1)

        chip = QFrame()
        chip.setObjectName("TranscriptMetaChip")
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(8, 4, 8, 4)
        chip_layout.setSpacing(5)

        icon = QLabel()
        icon.setPixmap(qta.icon("fa5s.check-circle", color=SUCCESS_GREEN).pixmap(11, 11))
        chip_layout.addWidget(icon, 0, Qt.AlignVCenter)

        label = QLabel(stats)
        label.setObjectName("MetaText")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        chip_layout.addWidget(label, 0, Qt.AlignVCenter)

        layout.addWidget(chip, 0, Qt.AlignRight)


class StatusIndicatorWidget(QFrame):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.setObjectName("InlineStatusRow")
        self.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 6)
        layout.setSpacing(6)

        self.spinner = QToolButton()
        self.spinner.setObjectName("InlineStatusSpinner")
        self.spinner.setEnabled(False)
        self.spinner.setAutoRaise(True)
        self.spinner.setIcon(qta.icon("fa5s.spinner", color=TEXT_MUTED, animation=qta.Spin(self.spinner)))
        self.spinner.setIconSize(QSize(12, 12))
        layout.addWidget(self.spinner, 0, Qt.AlignVCenter)

        self.label = QLabel(label)
        self.label.setObjectName("TranscriptMeta")
        layout.addWidget(self.label, 0, Qt.AlignVCenter)
        layout.addStretch(1)

    def set_label(self, label: str) -> None:
        self.label.setText(label)


class UserMessageWidget(QFrame):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.full_text = text
        self.preview_text, self.is_expandable = _collapsed_user_message_text(text)
        self.setObjectName("TranscriptRow")
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 16)
        layout.setSpacing(0)

        # Пружина выталкивает пузырь вправо
        layout.addStretch(1)

        self.bubble = QFrame()
        self.bubble.setObjectName("UserBubble")
        self.bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.bubble.setMaximumWidth(720)

        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble_layout.setSpacing(4)

        self.body = QLabel(self.preview_text if self.is_expandable else self.full_text)
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
        bubble_layout.addWidget(self.body)

        self.toggle_button = QToolButton()
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
    def __init__(self) -> None:
        super().__init__()
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

    def set_markdown(self, markdown: str) -> None:
        self._markdown = markdown
        text = markdown.strip() or "*Thinking…*"
        
        parts = text.split("```")

        while len(self.parts_widgets) < len(parts):
            idx = len(self.parts_widgets)
            is_code = (idx % 2 == 1)
            if is_code:
                w = CodeBlockWidget("", "")
                self.content_layout.addWidget(w)
            else:
                w = AutoTextBrowser()
                w.setObjectName("AssistantBody")
                self.content_layout.addWidget(w)
            self.parts_widgets.append(w)

        while len(self.parts_widgets) > len(parts):
            w = self.parts_widgets.pop()
            self.content_layout.removeWidget(w)
            w.deleteLater()

        for idx, part in enumerate(parts):
            w = self.parts_widgets[idx]
            is_code = (idx % 2 == 1)
            
            if is_code:
                lines = part.split("\n", 1)
                lang = lines[0].strip() if len(lines) > 0 else ""
                code = lines[1] if len(lines) > 1 else ""
                title = lang.upper() if lang else "CODE"
                w.set_code(code, lang, title)
                w.setVisible(True)
            else:
                if part.strip() or idx == 0:
                    w.setMarkdown(part)
                    w.setVisible(True)
                else:
                    w.setVisible(False)

    def markdown(self) -> str:
        return self._markdown

class ToolCardWidget(QFrame):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self.setObjectName("ToolRow")
        self.setFrameShape(QFrame.NoFrame)
        self.tool_id = payload.get("tool_id", "")
        self.payload = payload
        self.output_section: CollapsibleSection | None = None
        self.output_view: QPlainTextEdit | None = None
        self._args_expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.icon_label = QLabel()
        self.icon_label.setPixmap(qta.icon("fa5s.circle", color=TEXT_MUTED).pixmap(7, 7))
        header.addWidget(self.icon_label, 0, Qt.AlignVCenter)

        self.tool_button = QPushButton()
        self.tool_button.setObjectName("ToolCallButton")
        self.tool_button.setCheckable(True)
        self.tool_button.setFlat(True)
        self.tool_button.setText(payload.get("display", "") or payload.get("name", "tool"))
        self.tool_button.setCursor(Qt.PointingHandCursor)
        self.tool_button.setIcon(qta.icon("fa5s.caret-right", color=TEXT_MUTED))
        self.tool_button.setIconSize(QSize(8, 8))
        self.tool_button.toggled.connect(self._set_args_expanded)
        header.addWidget(self.tool_button, 1)

        self.timing_label = QLabel("")
        self.timing_label.setObjectName("MetaText")
        self.timing_label.setVisible(False)
        header.addWidget(self.timing_label, 0, Qt.AlignVCenter | Qt.AlignRight)
        layout.addLayout(header)

        self.args_view = QPlainTextEdit()
        self.args_view.setObjectName("InlineCodeView")
        self.args_view.setReadOnly(True)
        self.args_view.setFont(_make_mono_font())
        self.args_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._set_args(payload.get("args", {}))

        self.args_container = QWidget()
        args_layout = QHBoxLayout(self.args_container)
        args_layout.setContentsMargins(16, 0, 0, 0)
        args_layout.setSpacing(0)
        args_layout.addWidget(self.args_view, 1)
        self.args_container.setVisible(False)
        layout.addWidget(self.args_container)

        self.diff_section: CollapsibleSection | None = None

    @staticmethod
    def _normalize_args(args: Any) -> dict[str, Any]:
        return dict(args) if isinstance(args, dict) else {}

    def _set_args(self, args: Any) -> None:
        normalized = self._normalize_args(args)
        rendered = json.dumps(normalized, ensure_ascii=False, indent=2)
        if self.args_view.toPlainText() != rendered:
            self.args_view.setPlainText(rendered)
        _sync_plain_text_height(self.args_view, min_lines=2, max_lines=8, extra_padding=14)

    def finish(self, payload: dict[str, Any]) -> None:
        previous_display = self.tool_button.text()
        normalized_args = self._normalize_args(payload.get("args", self.payload.get("args", {})))
        merged_payload = dict(self.payload)
        merged_payload.update(payload)
        merged_payload["args"] = normalized_args
        self.payload = merged_payload
        is_error = payload.get("is_error", False)
        icon_name = "fa5s.circle" if not is_error else "fa5s.times-circle"
        color = SUCCESS_GREEN if not is_error else ERROR_RED
        self.icon_label.setPixmap(qta.icon(icon_name, color=color).pixmap(7, 7))
        self.tool_button.setText(self.payload.get("display", "") or previous_display or self.payload.get("name", "tool"))
        self._set_args(normalized_args)

        duration = self.payload.get("duration")
        if duration is not None:
            self.timing_label.setText(f"{duration:.1f}s")
            self.timing_label.setVisible(True)

        if is_error:
            if self.output_view is None:
                self.output_view = QPlainTextEdit()
                self.output_view.setObjectName("InlineCodeView")
                self.output_view.setReadOnly(True)
                self.output_view.setFont(_make_mono_font())
            if self.output_section is None:
                self.output_section = CollapsibleSection("Output", self.output_view, expanded=True)
                self.layout().insertWidget(1, self.output_section)
            self.output_section.setVisible(True)
            self.output_view.setPlainText(self.payload.get("content", ""))
            _sync_plain_text_height(self.output_view, min_lines=2, max_lines=10, extra_padding=14)
            self.output_section.set_expanded(True)
        elif self.output_section is not None:
            self.output_section.setVisible(False)

        diff_text = self.payload.get("diff", "")
        if diff_text and self.diff_section is None:
            self.diff_section = CollapsibleSection("Diff", CodeBlockWidget(diff_text, "diff"), expanded=False)
            self.layout().addWidget(self.diff_section)
        elif diff_text and self.diff_section is not None:
            if isinstance(self.diff_section.content, CodeBlockWidget):
                self.diff_section.content.set_code(diff_text, "diff")

    def _set_args_expanded(self, expanded: bool) -> None:
        self._args_expanded = expanded
        self.tool_button.setIcon(
            qta.icon("fa5s.caret-down" if expanded else "fa5s.caret-right", color=TEXT_MUTED)
        )
        self.args_container.setVisible(expanded)


class ConversationTurnWidget(QWidget):
    def __init__(self, user_text: str) -> None:
        super().__init__()
        self._assistant_markdown = ""
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._timeline: list[tuple[str, QWidget]] = []
        self.assistant_segments: list[AssistantMessageWidget] = []
        self.tool_cards: dict[str, ToolCardWidget] = {}
        self.status_widget: StatusIndicatorWidget | None = None
        self._append_block("user", UserMessageWidget(user_text))

    @staticmethod
    def _common_prefix_length(first: str, second: str) -> int:
        limit = min(len(first), len(second))
        index = 0
        while index < limit and first[index] == second[index]:
            index += 1
        return index

    def _append_block(self, kind: str, widget: QWidget) -> QWidget:
        if kind in {"assistant", "tool", "notice", "stats"}:
            self.clear_status()
        self._layout.addWidget(widget)
        self._timeline.append((kind, widget))
        return widget

    def has_rendered_output(self) -> bool:
        return len(self._timeline) > 1

    def set_status(self, label: str) -> None:
        if self.status_widget is None:
            self.status_widget = StatusIndicatorWidget(label)
            self._layout.addWidget(self.status_widget)
            return
        self._layout.removeWidget(self.status_widget)
        self._layout.addWidget(self.status_widget)
        self.status_widget.set_label(label)

    def clear_status(self) -> None:
        if self.status_widget is None:
            return
        self._layout.removeWidget(self.status_widget)
        self.status_widget.deleteLater()
        self.status_widget = None

    def _ensure_assistant_segment(self) -> AssistantMessageWidget:
        if self._timeline and self._timeline[-1][0] == "assistant":
            return self._timeline[-1][1]  # type: ignore[return-value]
        segment = AssistantMessageWidget()
        self.assistant_segments.append(segment)
        self._append_block("assistant", segment)
        return segment

    def set_assistant_markdown(self, markdown: str) -> None:
        if markdown == self._assistant_markdown and self.assistant_segments:
            return

        if not self._assistant_markdown:
            segment_text = markdown
        elif markdown.startswith(self._assistant_markdown):
            segment_text = markdown[len(self._assistant_markdown):]
        else:
            prefix_len = self._common_prefix_length(self._assistant_markdown, markdown)
            segment_text = markdown[prefix_len:]

        segment = self._ensure_assistant_segment()
        if markdown and not segment_text and not segment.markdown():
            segment.set_markdown(markdown)
        elif segment_text:
            segment.set_markdown(segment.markdown() + segment_text)
        elif not segment.markdown():
            segment.set_markdown(markdown)

        self._assistant_markdown = markdown

    def add_notice(self, message: str, level: str = "info") -> None:
        self._append_block("notice", NoticeWidget(message, level=level))

    def add_assistant_message(self, markdown: str) -> AssistantMessageWidget:
        segment = AssistantMessageWidget()
        segment.set_markdown(markdown)
        self.assistant_segments.append(segment)
        self._append_block("assistant", segment)
        self._assistant_markdown = markdown
        return segment

    def start_tool(self, payload: dict[str, Any]) -> ToolCardWidget:
        tool_id = payload.get("tool_id", "")
        card = self.tool_cards.get(tool_id)
        if card is None:
            card = ToolCardWidget(payload)
            self.tool_cards[tool_id] = card
            self._append_block("tool", card)
        return card

    def finish_tool(self, payload: dict[str, Any]) -> None:
        self.start_tool(payload).finish(payload)

    def complete(self, stats: str) -> None:
        self._append_block("stats", RunStatsWidget(stats))

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
                    self.finish_tool(payload)
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
        self.scroll.verticalScrollBar().valueChanged.connect(self._handle_scrollbar_value_changed)

    def clear_transcript(self) -> None:
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._auto_follow_enabled = True
        self._pending_scroll = False
        self._pending_force_scroll = False

    def add_global_notice(self, message: str, level: str = "info") -> None:
        self.layout.insertWidget(self.layout.count() - 1, NoticeWidget(message, level=level))
        self.notify_content_changed()

    def start_turn(self, user_text: str) -> ConversationTurnWidget:
        turn = ConversationTurnWidget(user_text)
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
            turn = ConversationTurnWidget(user_text)
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
            return

        scrollbar = self.scroll.verticalScrollBar()
        self._programmatic_scroll = True
        scrollbar.setValue(scrollbar.maximum())
        self._programmatic_scroll = False
        self._auto_follow_enabled = True

    def scroll_to_bottom(self) -> None:
        self._pending_scroll = False
        self._pending_force_scroll = False
        scrollbar = self.scroll.verticalScrollBar()
        self._programmatic_scroll = True
        scrollbar.setValue(scrollbar.maximum())
        self._programmatic_scroll = False
        self._auto_follow_enabled = True


class ApprovalDialog(QDialog):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.choice: tuple[bool, bool] = (False, False)
        self.setWindowTitle("Confirmation Needed")
        self.setModal(True)
        self.resize(470, 320)
        self.setMinimumSize(420, 280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Protected action requires confirmation")
        title.setStyleSheet("font-weight: 600; font-size: 13pt;")
        layout.addWidget(title)

        summary = payload.get("summary", {})
        impacts = ", ".join(summary.get("impacts", [])) or "local state"
        summary_label = QLabel(
            f"Risk: {summary.get('risk_level', 'unknown')} • Impacts: {impacts} • "
            f"Default: {'approve' if summary.get('default_approve') else 'deny'}"
        )
        summary_label.setStyleSheet(f"color: {TEXT_MUTED};")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)
        for tool in payload.get("tools", []):
            card = QFrame()
            card.setObjectName("ApprovalCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(6)
            name_label = QLabel(tool.get("name", "tool"))
            name_label.setStyleSheet("font-weight: 600;")
            card_layout.addWidget(name_label)
            args_view = QPlainTextEdit()
            args_view.setObjectName("CodeView")
            args_view.setReadOnly(True)
            args_view.setFont(_make_mono_font())
            args_view.setPlainText(json.dumps(tool.get("args", {}), ensure_ascii=False, indent=2))
            args_view.setFixedHeight(78)
            card_layout.addWidget(CollapsibleSection("Arguments", args_view, expanded=False))
            container_layout.addWidget(card)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        approve_button = QPushButton(qta.icon("fa5s.check", color="white"), "Approve")
        approve_button.setObjectName("PrimaryButton")
        deny_button = QPushButton(qta.icon("fa5s.times", color="white"), "Deny")
        deny_button.setObjectName("DangerButton")
        always_button = QPushButton("Always for this session")
        buttons.addButton(approve_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(always_button, QDialogButtonBox.ActionRole)
        buttons.addButton(deny_button, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

        approve_button.clicked.connect(self._approve)
        always_button.clicked.connect(self._always)
        deny_button.clicked.connect(self._deny)

    def _approve(self) -> None:
        self.choice = (True, False)
        self.accept()

    def _always(self) -> None:
        self.choice = (True, True)
        self.accept()

    def _deny(self) -> None:
        self.choice = (False, False)
        self.reject()
