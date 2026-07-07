from __future__ import annotations

from collections import Counter

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .foundation import _fa_icon
from .tools import ToolCardWidget
from ui.theme import ERROR_RED, SUCCESS_GREEN, TEXT_MUTED


class ToolGroupWidget(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ToolGroupFrame")
        self.setFrameShape(QFrame.NoFrame)
        self._tools: list[ToolCardWidget] = []
        self._collapsed = False
        self._completed = False
        self._completion_announced = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)

        self.header_row = QWidget(self)
        self.header_row.setObjectName("ToolGroupHeaderRow")
        header_layout = QHBoxLayout(self.header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.header_btn = QPushButton(self)
        self.header_btn.setObjectName("ToolGroupHeaderButton")
        self.header_btn.setCheckable(True)
        self.header_btn.setFlat(True)
        self.header_btn.setChecked(True)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header_btn.setMinimumWidth(0)
        self.header_btn.setIconSize(QSize(9, 9))
        self.header_btn.setAccessibleName("Tool results group")
        self.header_btn.setAccessibleDescription("Expand or collapse the tool results for this turn")
        self.header_btn.clicked.connect(self._toggle)
        header_layout.addWidget(self.header_btn, 1)

        self.error_icon_label = QLabel(self.header_row)
        self.error_icon_label.setObjectName("MetaText")
        self.error_icon_label.setPixmap(_fa_icon("fa5s.times-circle", color=ERROR_RED, size=9).pixmap(9, 9))
        self.error_icon_label.setVisible(False)
        header_layout.addWidget(self.error_icon_label, 0, Qt.AlignVCenter)

        self.error_count_label = QLabel("", self.header_row)
        self.error_count_label.setObjectName("MetaText")
        self.error_count_label.setProperty("severity", "error")
        self.error_count_label.setVisible(False)
        header_layout.addWidget(self.error_count_label, 0, Qt.AlignVCenter)

        layout.addWidget(self.header_row)

        self.container = QWidget(self)
        self.container.setObjectName("ToolGroupContainer")
        self.inner = QVBoxLayout(self.container)
        self.inner.setContentsMargins(10, 1, 0, 0)
        self.inner.setSpacing(5)
        layout.addWidget(self.container)

        self._sync_header()

    @staticmethod
    def _plural_ru(value: int, one: str, few: str, many: str) -> str:
        value = abs(int(value))
        if value % 100 in {11, 12, 13, 14}:
            return many
        if value % 10 == 1:
            return one
        if value % 10 in {2, 3, 4}:
            return few
        return many

    @staticmethod
    def _tool_role(card: ToolCardWidget) -> str:
        name = str(card.payload.get("name", "") or "").strip()
        if name in {"write_file", "edit_file", "Write", "SearchReplace"}:
            return "edit"
        if name in {"read_file", "Read"}:
            return "read"
        if name in {"execute", "RunCommand", "cli_exec"}:
            return "command"
        if name in {"grep", "Grep", "glob", "Glob", "search_in_file", "search_in_directory", "find_file"}:
            return "search"
        if name in {"web_search", "WebSearch", "fetch_url", "WebFetch", "fetch_content", "download_file"}:
            return "network"
        if name in {"ls", "LS", "list_directory"}:
            return "list"
        return "tool"

    def _group_role(self) -> str:
        roles = Counter(self._tool_role(tool) for tool in self._tools)
        if not roles:
            return "tool"
        if len(roles) == 1:
            return next(iter(roles))
        if roles.get("edit", 0) == len(self._tools):
            return "edit"
        if roles.get("command", 0) == len(self._tools):
            return "command"
        return "tool"

    def _header_text(self, *, completed: bool) -> str:
        total = len(self._tools)
        if total <= 0:
            return "Выполняется"
        role = self._group_role()
        if role == "edit":
            noun = self._plural_ru(total, "файл", "файла", "файлов")
            return f"{'Изменён' if total == 1 else 'Изменено'} {total} {noun}" if completed else f"Редактирование {total} {noun}"
        if role == "read":
            noun = self._plural_ru(total, "файл", "файла", "файлов")
            return f"{'Прочитан' if total == 1 else 'Прочитано'} {total} {noun}" if completed else f"Чтение {total} {noun}"
        if role == "command":
            if total == 1:
                return "Команда выполнена" if completed else "Запущена команда"
            noun = self._plural_ru(total, "команда", "команды", "команд")
            return f"Выполнено {total} {noun}" if completed else f"Запущено {total} {noun}"
        if role == "search":
            return "Поиск завершён" if completed else "Поиск по файлам"
        if role == "network":
            return "Запрос выполнен" if completed else "Выполняется запрос"
        if role == "list":
            return "Список получен" if completed else "Просмотр каталога"
        noun = self._plural_ru(total, "инструмент", "инструмента", "инструментов")
        return f"Выполнено {total} {noun}" if completed else f"Выполняется {total} {noun}"

    def _error_header_text(self, errors: int) -> str:
        total = len(self._tools)
        role = self._group_role()
        if total == 1:
            return {
                "edit": "Редактирование не удалось",
                "read": "Чтение не удалось",
                "command": "Команда не выполнена",
                "search": "Поиск не удался",
                "network": "Запрос не выполнен",
                "list": "Просмотр не удался",
            }.get(role, "Инструмент завершился ошибкой")
        noun = self._plural_ru(errors, "ошибка", "ошибки", "ошибок")
        return f"Выполнено с ошибками: {errors} {noun}"

    def _set_header_state(self, *, state: str) -> None:
        if self.header_btn.property("state") == state:
            return
        self.header_btn.setProperty("state", state)
        style = self.header_btn.style()
        if style is not None:
            style.unpolish(self.header_btn)
            style.polish(self.header_btn)

    def add_tool(self, card: ToolCardWidget) -> None:
        if card in self._tools:
            return
        self._tools.append(card)
        self.inner.addWidget(card)
        if self._completed:
            self._completed = False
            self._completion_announced = False
            self.expand()
        else:
            self._sync_header()

    @staticmethod
    def _tool_is_finished(card: ToolCardWidget) -> bool:
        return str(card.payload.get("phase", "running") or "running") == "finished"

    def refresh_completion(self, *, auto_collapse: bool = False) -> None:
        self._completed = bool(self._tools) and all(self._tool_is_finished(tool) for tool in self._tools)
        if not self._completed:
            self._completion_announced = False
        if self._completed and auto_collapse:
            self._completion_announced = True
            self._collapsed = True
            self.container.hide()
            self.header_btn.setChecked(False)
        elif self._completed:
            self._completion_announced = True
        self._sync_header()

    def collapse(self) -> None:
        self._completed = bool(self._tools) and all(self._tool_is_finished(tool) for tool in self._tools)
        if self._completed:
            self._completion_announced = True
        if self._collapsed:
            self._sync_header()
            return
        self._collapsed = True
        self.container.hide()
        self.header_btn.setChecked(False)
        self._sync_header()

    def expand(self) -> None:
        self._collapsed = False
        self.container.show()
        self.header_btn.setChecked(True)
        self._sync_header()

    def _toggle(self, checked: bool = False) -> None:
        self._collapsed = not checked
        self.container.setVisible(checked)
        if self._collapsed and self._completed:
            self._completion_announced = True
        self._sync_header()

    def _sync_header(self) -> None:
        expanded = not self._collapsed
        if self._completion_announced:
            total = len(self._tools)
            errors = sum(1 for tool in self._tools if tool.payload.get("is_error", False))
            self.error_icon_label.setVisible(errors > 0)
            self.error_count_label.setText(str(errors) if errors > 0 else "")
            self.error_count_label.setVisible(errors > 0)
            self._set_header_state(state="error" if errors > 0 else "complete")
            if errors > 0:
                self.header_btn.setIcon(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=9))
                self.header_btn.setText(f"{self._error_header_text(errors)} ·")
            else:
                self.header_btn.setIcon(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=9))
                self.header_btn.setText(self._header_text(completed=True))
            return
        self.error_icon_label.setVisible(False)
        self.error_count_label.setVisible(False)
        self.error_count_label.clear()
        self._set_header_state(state="active")
        self.header_btn.setIcon(_fa_icon("fa5s.caret-down" if expanded else "fa5s.caret-right", color=TEXT_MUTED, size=9))
        self.header_btn.setText(self._header_text(completed=False))
