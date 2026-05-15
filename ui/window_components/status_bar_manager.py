from __future__ import annotations

import os
import time
from dataclasses import dataclass

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QLabel, QStatusBar

from ui.theme import ACCENT_BLUE, ERROR_RED, SUCCESS_GREEN
from ui.widgets import _fa_icon


@dataclass(frozen=True)
class StatusBarBuildResult:
    status_bar: QStatusBar
    runtime_meta_label: QLabel


class StatusBarManager:
    def __init__(self, window) -> None:
        self.window = window
        self.realtime_timer = QTimer(window)
        self.realtime_timer.timeout.connect(window._update_realtime_elapsed)

    def build(self) -> StatusBarBuildResult:
        status_bar = QStatusBar(self.window)
        status_bar.setSizeGripEnabled(False)

        runtime_meta_label = QLabel("")
        runtime_meta_label.setObjectName("StatusBarMeta")
        runtime_meta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_bar.addPermanentWidget(runtime_meta_label, 1)

        return StatusBarBuildResult(
            status_bar=status_bar,
            runtime_meta_label=runtime_meta_label,
        )

    def set_primary_status_message(self, label: str) -> None:
        self.window._primary_status_label = label
        self.window._status_message_ticket += 1

    def show_transient_status_message(self, label: str, timeout_ms: int = 1800) -> None:
        self.window._status_message_ticket += 1

    def set_status_visual(self, label: str, *, busy: bool = False, success: bool = False, error: bool = False) -> None:
        color = ACCENT_BLUE if busy else SUCCESS_GREEN if success else ERROR_RED if error else ACCENT_BLUE
        icon_name = (
            "fa5s.spinner"
            if busy
            else "fa5s.check-circle"
            if success
            else "fa5s.times-circle"
            if error
            else "fa5s.circle"
        )
        self.window.status_text.setText(label)
        self.window.status_icon.setPixmap(_fa_icon(icon_name, color=color, size=14).pixmap(14, 14))
        self.window.top_status_chip.setText(label)
        self.window.top_status_chip.setProperty(
            "statusState",
            "busy" if busy else "success" if success else "error" if error else "idle",
        )
        style = self.window.top_status_chip.style()
        if style is not None:
            style.unpolish(self.window.top_status_chip)
            style.polish(self.window.top_status_chip)
        self.set_primary_status_message(label)

    def update_env_info(self, snapshot: dict) -> None:
        cwd = os.getcwd()
        if len(cwd) > 56:
            cwd_display = cwd[:18] + "..." + cwd[-34:]
        else:
            cwd_display = cwd

        active_profile = self.window._active_model_profile()
        if active_profile is not None:
            model = str(active_profile.get("model") or snapshot.get("model", "unknown"))
            model_id = str(active_profile.get("id") or model).strip() or "Model"
            provider = str(active_profile.get("provider") or "").strip() or snapshot.get("provider", "")
            self.window.model_chip.setText(model_id)
            capability_text = "yes" if self.window._active_model_supports_images() else "no"
            self.window.model_chip.setToolTip(f"Provider: {provider}\nModel: {model}\nImage input: {capability_text}")
        else:
            model = snapshot.get("model", "unknown")
            self.window.model_chip.setText("No models")
            self.window.model_chip.setToolTip("No models configured")

        tools_count = snapshot.get("tools_count", 0)
        self.window.runtime_meta_label.setText(
            f"Workdir: {cwd_display}   |   Model: {model}   |   Tools: {tools_count}"
        )
        self.window.runtime_meta_label.setToolTip(
            f"Workdir: {cwd}\nModel: {model}\nTools: {tools_count}"
        )
