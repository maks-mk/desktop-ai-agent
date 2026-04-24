from __future__ import annotations

import time

from PySide6.QtCore import QTimer

from core.multimodal import (
    import_image_attachment_from_file,
    import_image_attachment_from_qimage,
    normalize_image_attachments,
)
from ui.theme import ACCENT_BLUE, ERROR_RED, SUCCESS_GREEN
from ui.widgets import _fa_icon


class StreamEventRouter:
    def __init__(self, handlers: dict[str, object]) -> None:
        self._handlers = dict(handlers)

    @property
    def handlers(self) -> dict[str, object]:
        return dict(self._handlers)

    def dispatch(self, event) -> None:
        event_type = getattr(event, "type", "")
        payload = getattr(event, "payload", {})
        handler = self._handlers.get(event_type)
        if handler:
            handler(payload)


class ComposerStateController:
    def __init__(self, window) -> None:
        self.window = window

    def queue_height_sync(self, *_args) -> None:
        if self.window._composer_height_sync_pending:
            return
        self.window._composer_height_sync_pending = True
        QTimer.singleShot(0, self.window._flush_composer_height_sync)

    def flush_height_sync(self) -> None:
        self.window._composer_height_sync_pending = False
        self.update_height()

    def composer_visual_line_count(self) -> int:
        block = self.window.composer.document().firstBlock()
        total_lines = 0
        while block.isValid():
            layout = block.layout()
            line_count = int(layout.lineCount()) if layout is not None else 0
            total_lines += max(1, line_count)
            block = block.next()
        return max(1, total_lines)

    def update_height(self, *_args) -> None:
        line_spacing = max(14, self.window.composer.fontMetrics().lineSpacing())
        visual_lines = self.composer_visual_line_count()
        doc_height = (visual_lines * line_spacing) + self.window._composer_height_padding
        new_height = max(self.window._composer_min_height, min(doc_height, self.window._composer_max_height))
        if self.window.composer.height() != new_height:
            self.window.composer.setFixedHeight(new_height)

    def composer_has_request_content(self) -> bool:
        return bool(self.window.composer.toPlainText().strip() or self.window.draft_image_attachments)

    def request_blocked_by_image_capability(self) -> bool:
        return bool(self.window.draft_image_attachments) and not self.window._active_model_supports_images()

    def show_composer_notice(self, message: str, *, level: str = "warning") -> None:
        self.window.composer_notice_label.setText(str(message or "").strip())
        self.window.composer_notice_label.setProperty("severity", level)
        self.window.composer_notice_label.style().unpolish(self.window.composer_notice_label)
        self.window.composer_notice_label.style().polish(self.window.composer_notice_label)
        self.window.composer_notice_label.setVisible(bool(message))

    def clear_composer_notice(self) -> None:
        self.window.composer_notice_label.clear()
        self.window.composer_notice_label.setProperty("severity", "")
        self.window.composer_notice_label.style().unpolish(self.window.composer_notice_label)
        self.window.composer_notice_label.style().polish(self.window.composer_notice_label)
        self.window.composer_notice_label.setVisible(False)

    def refresh_draft_attachments(self) -> None:
        self.window.composer_attachments_strip.set_attachments(self.window.draft_image_attachments)
        self.window._refresh_submit_controls()

    def clear_draft_image_attachments(self) -> None:
        self.window.draft_image_attachments = []
        self.refresh_draft_attachments()

    def append_draft_image_attachments(self, attachments: list[dict] | None) -> None:
        existing_paths = {str(item.get("path") or "").strip() for item in self.window.draft_image_attachments}
        for attachment in normalize_image_attachments(attachments):
            path = str(attachment.get("path") or "").strip()
            if path and path not in existing_paths:
                self.window.draft_image_attachments.append(attachment)
                existing_paths.add(path)
        self.refresh_draft_attachments()

    def remove_draft_attachment(self, attachment_id: str) -> None:
        target_id = str(attachment_id or "").strip()
        if not target_id:
            return
        self.window.draft_image_attachments = [
            item
            for item in self.window.draft_image_attachments
            if str(item.get("id") or "").strip() != target_id
        ]
        if not self.window.draft_image_attachments:
            self.clear_composer_notice()
        self.refresh_draft_attachments()

    def import_image_files(self, file_paths: list[str]) -> None:
        if not file_paths:
            return
        if not self.window._active_model_supports_images():
            self.show_composer_notice(
                "Current model does not support image input. Switch models or use Add files instead.",
                level="warning",
            )
            return
        imported: list[dict] = []
        for path in file_paths:
            try:
                imported.append(import_image_attachment_from_file(path, session_id=self.window.active_session_id))
            except ValueError as exc:
                self.show_composer_notice(str(exc), level="warning")
        if imported:
            self.append_draft_image_attachments(imported)
            self.clear_composer_notice()

    def handle_pasted_image(self, image: object) -> None:
        if not self.window._active_model_supports_images():
            self.show_composer_notice(
                "Current model does not support image input. Switch models or remove the pasted image.",
                level="warning",
            )
            return
        try:
            attachment = import_image_attachment_from_qimage(image, session_id=self.window.active_session_id)
        except ValueError as exc:
            self.show_composer_notice(str(exc), level="warning")
            return
        self.append_draft_image_attachments([attachment])
        self.clear_composer_notice()

    def handle_pasted_image_files(self, file_paths: object) -> None:
        self.import_image_files([str(path) for path in list(file_paths or []) if str(path or "").strip()])


class RunStatusController:
    def __init__(self, window) -> None:
        self.window = window

    def update_realtime_elapsed(self) -> None:
        if self.window.current_turn is None or self.window._run_start_time is None or not self.window.is_busy:
            return

        elapsed = time.time() - self.window._run_start_time
        elapsed_text = f"{int(elapsed)}s"
        if self.window._last_rendered_elapsed_text == elapsed_text:
            return

        self.window._last_rendered_elapsed_text = elapsed_text
        self.window.current_turn.set_status(
            self.window._current_status_label,
            meta=elapsed_text,
            phase=self.window._current_status_phase,
        )
        self.window.transcript.notify_content_changed()

    def on_run_started(self, payload: dict) -> None:
        self.window._clear_user_choice_request()
        self.window._clear_approval_request()
        self.window.current_turn = self.window.transcript.start_turn(
            payload.get("text", ""),
            attachments=list(payload.get("attachments", []) or []),
        )
        self.window._summarize_in_progress = False
        self.window._run_start_time = time.time()
        self.window._current_status_label = "Analyzing request"
        self.window._current_status_phase = "working"

        if self.window.current_turn is not None:
            self.window.current_turn.set_status(
                self.window._current_status_label,
                phase=self.window._current_status_phase,
            )
            self.window.transcript.notify_content_changed(force=True)

        self.set_status_visual("Analyzing request…", busy=True)
        self.window._realtime_timer.start(100)

    def on_status_changed(self, payload: dict) -> None:
        label = payload.get("label")
        node = str(payload.get("node", "") or "")
        elapsed_text = str(payload.get("elapsed_text", "") or "")
        phase = str(payload.get("phase", "working") or "working")
        if not label:
            return

        self.window._current_status_label = label
        self.window._current_status_phase = phase

        try:
            if elapsed_text.endswith("s"):
                server_elapsed = float(elapsed_text[:-1])
                self.window._run_start_time = time.time() - server_elapsed
        except ValueError:
            pass

        self.set_status_visual(label, busy=node != "approval")

        if self.window.current_turn is not None:
            current_elapsed = time.time() - self.window._run_start_time if self.window._run_start_time else 0.0
            self.window._last_rendered_elapsed_text = f"{int(current_elapsed)}s"
            self.window.current_turn.set_status(label, meta=self.window._last_rendered_elapsed_text, phase=phase)

            if node == "summarize":
                self.window._summarize_in_progress = True
                self.window.current_turn.set_summary_notice("Context is being compressed automatically…", level="info")
            elif self.window._summarize_in_progress:
                self.window._summarize_in_progress = False
                self.window.current_turn.set_summary_notice("Context compressed", level="success")
            self.window.transcript.notify_content_changed()

    def on_run_finished(self, payload: dict) -> None:
        self.window._realtime_timer.stop()
        if self.window.current_turn is not None:
            self.window.current_turn.clear_status()
            self.window.current_turn.complete(payload.get("stats", ""))
            self.window.transcript.notify_content_changed()
        self.window.status_meta.setText("")
        self.set_status_visual("Ready", success=True)

    def on_run_failed(self, payload: dict) -> None:
        self.window._realtime_timer.stop()
        self.window.status_meta.setText("")
        message = payload.get("message", "Run failed")
        if self.window.current_turn is not None:
            self.window.current_turn.clear_status()
            self.window.current_turn.add_notice(message, level="error")
            self.window.transcript.notify_content_changed()
        else:
            self.window.transcript.add_global_notice(message, level="error")
        self.set_status_visual("Run failed", error=True)

    def on_chat_reset(self) -> None:
        self.window._realtime_timer.stop()
        self.window.composer.reset_history_navigation()
        self.window.current_turn = None
        self.window.transcript.clear_transcript()
        self.window._clear_draft_image_attachments()
        self.window._clear_composer_notice()
        self.window._clear_user_choice_request()
        self.window._clear_approval_request()
        self.show_transient_status_message("Started a new session")

    def set_primary_status_message(self, label: str) -> None:
        self.window._primary_status_label = label
        self.window._status_message_ticket += 1
        self.window.status_line_label.setText(label)

    def show_transient_status_message(self, label: str, timeout_ms: int = 1800) -> None:
        self.window._status_message_ticket += 1
        ticket = self.window._status_message_ticket
        self.window.status_line_label.setText(label)

        def _restore() -> None:
            if ticket != self.window._status_message_ticket:
                return
            self.window.status_line_label.setText(self.window._primary_status_label)

        QTimer.singleShot(timeout_ms + 30, _restore)

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

    def handle_busy_changed(self, busy: bool) -> None:
        self.window.is_busy = busy

        if not busy:
            self.window._realtime_timer.stop()
        elif self.window.current_turn is not None and not self.window._realtime_timer.isActive():
            self.window._realtime_timer.start(100)

        self.window.send_button.setVisible(not busy)
        self.window.stop_action_button.setVisible(busy)

        if busy:
            self.set_status_visual("Working…", busy=True)
        elif not self.window.awaiting_approval and not self.window.awaiting_user_choice:
            self.set_status_visual("Ready", success=True)
            self.window.status_meta.setText("")
        self.window._set_input_enabled(True)
