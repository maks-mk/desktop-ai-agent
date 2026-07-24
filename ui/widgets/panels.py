from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui.theme import ACCENT_BLUE
from .foundation import _fa_icon


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
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
        self._inner.setSpacing(6)
        self._inner.addStretch(1)

        self.scroll.setWidget(self._container)
        root.addWidget(self.scroll)

    def set_tools(self, tools: list[dict[str, str]]) -> None:
        while self._inner.count() > 1:
            item = self._inner.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        grouped: dict[str, list[dict[str, str]]] = {"Read-only": [], "Protected": [], "MCP": []}
        for row in tools:
            grouped.setdefault(row["group"], []).append(row)

        insert_pos = 0
        for group_name in ("Read-only", "Protected", "MCP"):
            items = grouped.get(group_name, [])
            if not items:
                continue

            header = QLabel(group_name.upper())
            header.setObjectName("ToolGroupHeader")
            header.setProperty("toolGroup", group_name.lower().replace("-", "_"))
            header.style().unpolish(header)
            header.style().polish(header)
            self._inner.insertWidget(insert_pos, header)
            insert_pos += 1

            for row in items:
                card = QFrame()
                card.setObjectName("ToolCard")
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(10, 8, 10, 8)
                card_layout.setSpacing(4)

                top_row = QHBoxLayout()
                top_row.setSpacing(6)

                name_label = QLabel(row["name"])
                name_label.setObjectName("ToolCardTitle")
                name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                top_row.addWidget(name_label, 1)

                flags = row.get("flags", "")
                if flags:
                    for flag in flags.split(", "):
                        flag = flag.strip()
                        if not flag:
                            continue
                        chip = QLabel(flag)
                        chip.setObjectName("ToolFlagChip")
                        if flag in ("mutating", "destructive", "approval"):
                            chip.setProperty("flagVariant", "warning")
                        elif flag in ("mcp", "network"):
                            chip.setProperty("flagVariant", "accent")
                        else:
                            chip.setProperty("flagVariant", "muted")
                        chip.style().unpolish(chip)
                        chip.style().polish(chip)
                        top_row.addWidget(chip, 0)

                card_layout.addLayout(top_row)

                desc = row.get("description", "")
                if desc:
                    desc_label = QLabel(desc)
                    desc_label.setWordWrap(True)
                    desc_label.setObjectName("ToolCardDescription")
                    card_layout.addWidget(desc_label)

                self._inner.insertWidget(insert_pos, card)
                insert_pos += 1

            sep = QFrame()
            sep.setFixedHeight(1)
            sep.setObjectName("ToolGroupSeparator")
            self._inner.insertWidget(insert_pos, sep)
            insert_pos += 1


class InspectorPanelWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("InspectorPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        title = QLabel("Inspector")
        title.setObjectName("InspectorSectionTitle")
        header_row.addWidget(title)
        header_row.addStretch(1)

        hint = QLabel("Run details, tools, and help")
        hint.setObjectName("InspectorMetaText")
        header_row.addWidget(hint, 0, Qt.AlignRight)
        layout.addLayout(header_row)

        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Inspector tabs")
        self.tabs.setAccessibleDescription("Switch between run details, tools, and help")
        self.overview_panel = OverviewPanelWidget()
        self.overview_panel.setAccessibleName("Run details")
        self.tools_panel = ToolsPanelWidget()
        self.tools_panel.setAccessibleName("Tools panel")

        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_layout.setSpacing(0)
        self.help_text = QTextBrowser()
        self.help_text.setObjectName("InspectorHelpText")
        self.help_text.setOpenLinks(False)
        self.help_text.setOpenExternalLinks(False)
        self.help_text.setReadOnly(True)
        self.help_text.setAccessibleName("Help content")
        help_layout.addWidget(self.help_text)

        self.tabs.addTab(self.overview_panel, _fa_icon("fa5s.play-circle", color=ACCENT_BLUE, size=14), "Run")
        self.tabs.addTab(self.tools_panel, _fa_icon("fa5s.tools", color=ACCENT_BLUE, size=14), "Tools")
        self.tabs.addTab(help_widget, _fa_icon("fa5s.question-circle", color=ACCENT_BLUE, size=14), "Help")
        layout.addWidget(self.tabs, 1)


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

        inspector = InspectorPanelWidget()
        self.tabs = inspector.tabs
        self.overview_panel = inspector.overview_panel
        self.tools_panel = inspector.tools_panel
        self.help_text = inspector.help_text
        layout.addWidget(inspector)
