from __future__ import annotations

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
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

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
        self.inner.setContentsMargins(12, 0, 0, 0)
        self.inner.setSpacing(4)
        layout.addWidget(self.container)

        self._sync_header()

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
            if errors > 0:
                self.header_btn.setIcon(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=9))
                self.header_btn.setText(f"Completed ({total} tools) ·")
            else:
                self.header_btn.setIcon(_fa_icon("fa5s.check-circle", color=SUCCESS_GREEN, size=9))
                self.header_btn.setText(f"Completed ({total} tools)")
            return
        self.error_icon_label.setVisible(False)
        self.error_count_label.setVisible(False)
        self.error_count_label.clear()
        self.header_btn.setIcon(_fa_icon("fa5s.caret-down" if expanded else "fa5s.caret-right", color=TEXT_MUTED, size=9))
        self.header_btn.setText("Running...")
