from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

from .foundation import _fa_icon


def _attachment_tooltip(attachment: dict[str, Any]) -> str:
    file_name = str(attachment.get("file_name") or Path(str(attachment.get("path") or "")).name).strip()
    width = int(attachment.get("width") or 0)
    height = int(attachment.get("height") or 0)
    size_bytes = int(attachment.get("size_bytes") or 0)
    tooltip_parts = [file_name] if file_name else []
    if width > 0 and height > 0:
        tooltip_parts.append(f"{width} x {height}")
    if size_bytes > 0:
        tooltip_parts.append(f"{size_bytes // 1024} KB" if size_bytes >= 1024 else f"{size_bytes} B")
    return "\n".join(tooltip_parts)


class ImageAttachmentChipWidget(QFrame):
    remove_requested = Signal(str)

    def __init__(
        self,
        attachment: dict[str, Any],
        *,
        thumb_size: int,
        removable: bool,
    ) -> None:
        super().__init__()
        self.attachment_id = str(attachment.get("id") or "").strip()
        self.setObjectName("ImageAttachmentCard")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        shell = QGridLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setHorizontalSpacing(0)
        shell.setVerticalSpacing(0)

        self.thumb = QLabel()
        self.thumb.setObjectName("ImageAttachmentThumb")
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedSize(thumb_size, thumb_size)
        self.thumb.setToolTip(_attachment_tooltip(attachment))
        self.thumb.setPixmap(self._load_pixmap(attachment, thumb_size))
        shell.addWidget(self.thumb, 0, 0)

        self.remove_button = QToolButton()
        self.remove_button.setObjectName("ImageAttachmentRemoveButton")
        self.remove_button.setAutoRaise(True)
        self.remove_button.setCursor(Qt.PointingHandCursor)
        self.remove_button.setIcon(_fa_icon("fa5s.times", size=10))
        self.remove_button.setToolTip("Remove image")
        self.remove_button.setVisible(removable)
        self.remove_button.clicked.connect(self._emit_remove)
        shell.addWidget(self.remove_button, 0, 0, Qt.AlignTop | Qt.AlignRight)

    def _emit_remove(self) -> None:
        if self.attachment_id:
            self.remove_requested.emit(self.attachment_id)

    @staticmethod
    def _load_pixmap(attachment: dict[str, Any], thumb_size: int) -> QPixmap:
        path = str(attachment.get("path") or "").strip()
        pixmap = QPixmap(path)
        if pixmap.isNull():
            placeholder = QPixmap(thumb_size, thumb_size)
            placeholder.fill(Qt.transparent)
            return placeholder
        return pixmap.scaled(
            thumb_size,
            thumb_size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )


class ImageAttachmentStripWidget(QWidget):
    attachment_remove_requested = Signal(str)

    def __init__(self, *, thumb_size: int = 52, removable: bool = False) -> None:
        super().__init__()
        self._thumb_size = thumb_size
        self._removable = removable
        self._chips: list[ImageAttachmentChipWidget] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._layout = layout
        self.setVisible(False)

    def set_attachments(self, attachments: list[dict[str, Any]] | None) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._chips.clear()

        normalized = [item for item in attachments or [] if isinstance(item, dict)]
        if not normalized:
            self.setVisible(False)
            return

        for attachment in normalized:
            chip = ImageAttachmentChipWidget(
                attachment,
                thumb_size=self._thumb_size,
                removable=self._removable,
            )
            chip.remove_requested.connect(self.attachment_remove_requested.emit)
            self._layout.addWidget(chip, 0, Qt.AlignLeft | Qt.AlignTop)
            self._chips.append(chip)
        self._layout.addStretch(1)
        self.setVisible(True)
