import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QLabel, QFrame, QToolBar

import main as agent_cli
from core.gui_runtime import build_runtime_snapshot, summarize_approval_request
from core.stream_processor import StreamEvent
from core.tool_policy import ToolMetadata
from core.ui_theme import AMBER_WARNING, BORDER, ERROR_RED, SURFACE_BG, SURFACE_CARD, TEXT_MUTED


class FakeTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class FakeToolRegistry:
    def __init__(self):
        self.tools = [
            FakeTool("read_file", "Read a file"),
            FakeTool("edit_file", "Edit a file in place"),
            FakeTool("context7:resolve-library-id", "Resolve a Context7 library id"),
        ]
        self.tool_metadata = {
            "read_file": ToolMetadata(name="read_file", read_only=True),
            "edit_file": ToolMetadata(name="edit_file", mutating=True, requires_approval=True),
            "context7:resolve-library-id": ToolMetadata(
                name="context7:resolve-library-id",
                read_only=True,
                networked=True,
                source="mcp",
            ),
        }
        self.checkpoint_info = {
            "backend": "sqlite",
            "resolved_backend": "sqlite",
            "target": ".agent_state/checkpoints.sqlite",
            "warnings": [],
        }
        self.mcp_server_status = [
            {"server": "context7", "loaded_tools": ["resolve-library-id"], "error": ""},
        ]
        self.loader_status = []

    def get_runtime_status_lines(self):
        return [
            "Checkpoint: requested=sqlite active=sqlite target=.agent_state/checkpoints.sqlite",
            "MCP context7: loaded 1 tool(s)",
        ]


class FakeController(QObject):
    initialized = Signal(object)
    initialization_failed = Signal(str)
    event_emitted = Signal(object)
    approval_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.start_calls: list[str] = []
        self.resume_calls: list[tuple[bool, bool]] = []
        self.new_session_calls = 0
        self.switch_session_calls: list[str] = []
        self.delete_session_calls: list[str] = []
        self.reinitialize_calls: list[bool] = []
        self.shutdown_calls = 0
        self.initialize_calls = 0

    def initialize(self):
        self.initialize_calls += 1

    def start_run(self, text: str):
        self.start_calls.append(text)

    def resume_approval(self, approved: bool, always: bool = False):
        self.resume_calls.append((approved, always))

    def new_session(self):
        self.new_session_calls += 1

    def switch_session(self, session_id: str):
        self.switch_session_calls.append(session_id)

    def delete_session(self, session_id: str):
        self.delete_session_calls.append(session_id)

    def reinitialize(self, force_new_session: bool = False):
        self.reinitialize_calls.append(force_new_session)

    def shutdown(self):
        self.shutdown_calls += 1


class GuiUxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = FakeController()
        self.window = agent_cli.MainWindow(controller=self.controller, auto_initialize=False)

    def tearDown(self):
        self.window.close()

    def _process_events(self):
        self.app.processEvents()

    def _snapshot_payload(self):
        config = type(
            "Config",
            (),
            {
                "provider": "openai",
                "openai_model": "gpt-4o",
                "gemini_model": "gemini-1.5-flash",
                "checkpoint_backend": "sqlite",
                "enable_approvals": True,
                "debug": False,
            },
        )()
        session = type(
            "Session",
            (),
            {
                "session_id": "session-1234567890abcdef",
                "thread_id": "thread-abcdef1234567890",
                "approval_mode": "prompt",
                "project_path": "D:/demo/workspace",
                "title": "Current chat",
            },
        )()
        snapshot = build_runtime_snapshot(config, FakeToolRegistry(), session)
        return {
            "snapshot": snapshot,
            "tools": snapshot["tools"],
            "help_markdown": "## Help\n- test",
            "sessions": [
                {
                    "session_id": "session-1234567890abcdef",
                    "thread_id": "thread-abcdef1234567890",
                    "project_path": "D:/demo/workspace",
                    "title": "Current chat",
                    "created_at": "2026-03-31T10:00:00+00:00",
                    "updated_at": "2026-03-31T12:00:00+00:00",
                },
                {
                    "session_id": "session-older",
                    "thread_id": "thread-older",
                    "project_path": "D:/demo/other-project",
                    "title": "Older chat [demo/other-project]",
                    "created_at": "2026-03-30T10:00:00+00:00",
                    "updated_at": "2026-03-30T12:00:00+00:00",
                },
            ],
            "active_session_id": "session-1234567890abcdef",
            "transcript": {"summary_notice": "", "turns": []},
        }

    def test_main_window_populates_runtime_panels_on_initialize(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        self.assertEqual(self.window.overview_panel._labels["Provider"].text(), "OpenAI")
        self.assertEqual(self.window.overview_panel._labels["Model"].text(), "gpt-4o")
        self.assertEqual(self.window.overview_panel._labels["MCP"].text(), "context7")
        tool_cards = self.window.tools_panel.findChildren(QFrame, "ToolCard")
        self.assertEqual(len(tool_cards), 3)
        self.assertIn("Help", self.window.help_text.toPlainText())
        self.assertEqual(self.window.info_popup.tabs.tabText(0), "Info")
        self.assertEqual(self.window.info_popup.tabs.tabText(1), "Tools")
        self.assertEqual(self.window.info_popup.tabs.tabText(2), "Help")
        self.assertIn("Workdir:", self.window.runtime_meta_label.text())
        self.assertIn("Model: gpt-4o", self.window.runtime_meta_label.text())
        self.assertIn("Tools: 3", self.window.runtime_meta_label.text())
        self.assertEqual(self.window.sidebar.model.session_row_count(), 2)
        self.assertGreaterEqual(self.window.sidebar.model.rowCount(), 4)
        self.assertEqual(self.window.active_session_id, "session-1234567890abcdef")
        self.assertEqual(self.window.status_line_label.text(), "Ready")

    def test_submit_request_uses_controller_and_clears_editor(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Собери summary")

        self.window._submit_request()

        self.assertEqual(self.controller.start_calls, ["Собери summary"])
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_run_started_shows_inline_status_before_output(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Сводка"}))
        self.window._handle_event(StreamEvent("status_changed", {"label": "Reviewing", "node": "critic"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Reviewing")

    def test_status_is_rendered_below_existing_output_when_agent_keeps_thinking(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-1",
                    "name": "read_file",
                    "args": {"path": "demo.txt"},
                    "display": "read_file(demo.txt)",
                },
            )
        )
        self.window._handle_event(StreamEvent("status_changed", {"label": "Reviewing", "node": "critic"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Reviewing")

    def test_stream_events_render_transcript_and_compact_tool_sections(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Покажи diff"}))
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "partial",
                    "full_text": "Ответ\n\n```python\nprint('hi')\n```",
                    "has_thought": False,
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-1",
                    "name": "edit_file",
                    "args": {"path": "demo.txt"},
                    "display": "edit_file(demo.txt)",
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "tail",
                    "full_text": "Ответ\n\n```python\nprint('hi')\n```\n\nГотово",
                    "has_thought": False,
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-1",
                    "name": "edit_file",
                    "content": "Success\n```diff\n-foo\n+bar\n```",
                    "summary": "File edited successfully",
                    "is_error": False,
                    "duration": 1.4,
                    "diff": "-foo\n+bar",
                },
            )
        )

        self.assertIsNotNone(self.window.current_turn)
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "assistant", "tool", "assistant"])
        self.assertEqual(len(self.window.current_turn.assistant_segments), 2)
        self.assertIn("Ответ", self.window.current_turn.assistant_segments[0].markdown())
        self.assertIn("Готово", self.window.current_turn.assistant_segments[1].markdown())
        self.assertEqual(self.window.current_turn.assistant_segments[0].frameShape(), QFrame.NoFrame)
        transcript_text_labels = [
            label.text()
            for label in self.window.current_turn.findChildren(QLabel)
            if label.text() in {"Agent", "You"}
        ]
        self.assertEqual(transcript_text_labels, [])
        tool_card = self.window.current_turn.tool_cards["call-1"]
        self.assertEqual(tool_card.frameShape(), QFrame.NoFrame)
        self.assertEqual(tool_card.tool_button.text(), "edit_file(demo.txt)")
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.args_container.isHidden())
        self.assertIsNone(tool_card.output_section)
        self.assertIsNotNone(tool_card.diff_section)
        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.args_container.isHidden())

    def test_tool_error_expands_output_by_default(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Почини"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-err",
                    "name": "edit_file",
                    "content": "error[access_denied]: blocked",
                    "summary": "Skipped",
                    "is_error": True,
                    "duration": 0.4,
                    "diff": "",
                },
            )
        )

        tool_card = self.window.current_turn.tool_cards["call-err"]
        self.assertIsNotNone(tool_card.output_section)
        self.assertTrue(tool_card.output_section.toggle_button.isChecked())

    def test_run_finished_renders_plain_stats_chip(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Сводка"}))
        self.window._handle_event(StreamEvent("status_changed", {"label": "Reviewing", "node": "critic"}))
        self.window._handle_event(StreamEvent("run_finished", {"stats": "3.1s   In: 5328   Out: 106"}))

        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "stats"])
        self.assertEqual(self.window.status_meta.text(), "")
        stats_widget = self.window.current_turn.layout().itemAt(1).widget()
        labels = [label.text() for label in stats_widget.findChildren(QLabel)]
        self.assertTrue(any("3.1s" in text and "Out: 106" in text for text in labels))
        self.assertFalse(any("[dim]" in text for text in labels))

    def test_approval_dialog_resumes_controller_with_selected_choice(self):
        self.window._handle_initialized(self._snapshot_payload())
        payload = {
            "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
            "summary": {"risk_level": "medium", "impacts": ["files"], "default_approve": True},
        }

        dialog_instance = mock.Mock()
        dialog_instance.choice = (True, True)
        dialog_instance.exec.return_value = 0

        with mock.patch.object(agent_cli, "ApprovalDialog", return_value=dialog_instance):
            self.window._handle_approval_request(payload)

        self.assertEqual(self.controller.resume_calls, [(True, True)])

    def test_new_session_clears_transcript_and_calls_controller(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "hello"}))
        self.assertIsNotNone(self.window.current_turn)

        self.window._new_session()

        self.assertEqual(self.controller.new_session_calls, 1)
        self.assertIsNone(self.window.current_turn)

    def test_open_new_project_creates_fresh_session_without_hiding_history(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "history"}))
        self.assertIsNotNone(self.window.current_turn)

        with (
            mock.patch.object(agent_cli.QFileDialog, "getExistingDirectory", return_value="D:/demo/workspace"),
            mock.patch("os.chdir") as chdir_mock,
        ):
            self.window._open_new_project()

        chdir_mock.assert_called_once_with("D:/demo/workspace")
        self.assertEqual(self.controller.reinitialize_calls, [True])
        self.assertIsNone(self.window.current_turn)

    def test_user_message_uses_chat_style_bubble(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Короткий запрос"}))

        user_widget = self.window.current_turn.layout().itemAt(0).widget()

        self.assertEqual(user_widget.bubble.styleSheet(), "")
        self.assertIn("QFrame#UserBubble", self.window.styleSheet())
        self.assertIn("border-radius: 12px", self.window.styleSheet())

    def test_long_user_message_is_collapsible(self):
        self.window._handle_initialized(self._snapshot_payload())
        long_text = "Очень длинный запрос " * 80
        self.window._handle_event(StreamEvent("run_started", {"text": long_text}))

        user_widget = self.window.current_turn.layout().itemAt(0).widget()

        self.assertFalse(user_widget.toggle_button.isHidden())
        self.assertEqual(user_widget.toggle_button.text(), "Show more")
        self.assertNotEqual(user_widget.body.text(), long_text)
        self.assertTrue(user_widget.body.text().endswith("…"))

        user_widget.toggle_button.click()
        self._process_events()
        self.assertEqual(user_widget.toggle_button.text(), "Show less")
        self.assertEqual(user_widget.body.text(), long_text)

    def test_approval_summary_matches_expected_defaults(self):
        destructive = summarize_approval_request(
            [{"name": "safe_delete_file", "policy": {"destructive": True, "mutating": True}}]
        )
        mixed = summarize_approval_request(
            [{"name": "download_file", "policy": {"mutating": True, "networked": True}}]
        )
        regular = summarize_approval_request([{"name": "edit_file", "policy": {"mutating": True}}])

        self.assertFalse(destructive.default_approve)
        self.assertEqual(destructive.risk_level, "high")
        self.assertFalse(mixed.default_approve)
        self.assertIn("network", mixed.impacts)
        self.assertTrue(regular.default_approve)
        self.assertEqual(regular.risk_level, "medium")

    def test_stylesheet_keeps_expected_accent_colors(self):
        stylesheet = self.window.styleSheet()
        self.assertIn(AMBER_WARNING, stylesheet)
        self.assertIn(ERROR_RED, stylesheet)
        self.assertIn(BORDER, stylesheet)
        self.assertIn(SURFACE_BG, stylesheet)
        self.assertIn(SURFACE_CARD, stylesheet)
        self.assertIn(TEXT_MUTED, stylesheet)

    def test_info_popup_coexists_with_sidebar_and_closes_on_toggle(self):
        self.window.show()
        self._process_events()
        self.assertTrue(self.window.sidebar_container.isVisible())
        self.assertFalse(self.window.info_popup.isVisible())
        self.assertTrue(bool(self.window.info_popup.windowFlags() & Qt.Popup))

        self.window._toggle_info_popup()
        self._process_events()
        self.assertTrue(self.window.info_popup.isVisible())
        self.assertEqual(self.window.info_popup.tabs.currentIndex(), 0)

        self.window._toggle_info_popup()
        self._process_events()
        self.assertFalse(self.window.info_popup.isVisible())

    def test_sidebar_toggle_hides_and_restores_chat_list(self):
        self.window.show()
        self._process_events()

        self.assertTrue(self.window.sidebar_container.isVisible())
        self.window._toggle_sidebar()
        self._process_events()
        self.assertFalse(self.window.sidebar_container.isVisible())

        self.window._toggle_sidebar()
        self._process_events()
        self.assertTrue(self.window.sidebar_container.isVisible())

    def test_clicking_sidebar_session_switches_controller_session(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        index = self.window.sidebar.model.index_for_session("session-older")
        self.window.sidebar._emit_clicked_session(index)

        self.assertEqual(self.controller.switch_session_calls, ["session-older"])

    def test_delete_session_requests_controller_after_confirmation(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        with mock.patch.object(agent_cli.QMessageBox, "question", return_value=agent_cli.QMessageBox.Yes):
            self.window._request_delete_session("session-older")

        self.assertEqual(self.controller.delete_session_calls, ["session-older"])

    def test_delete_session_is_cancelled_without_confirmation(self):
        payload = self._snapshot_payload()
        self.window._handle_initialized(payload)

        with mock.patch.object(agent_cli.QMessageBox, "question", return_value=agent_cli.QMessageBox.No):
            self.window._request_delete_session("session-older")

        self.assertEqual(self.controller.delete_session_calls, [])

    def test_menu_bar_uses_corner_buttons_and_compact_composer(self):
        self.assertEqual(self.window.send_button.text(), "")
        self.assertLessEqual(self.window.composer.height(), 72)
        self.assertLessEqual(self.window.send_button.size().width(), 38)
        self.assertIsNone(self.window.findChild(QToolBar))
        self.assertIsNotNone(self.window.menuWidget())
        embedded_menu = self.window.menuWidget().findChild(agent_cli.QMenuBar)
        self.assertIsNotNone(embedded_menu)
        self.assertEqual(embedded_menu.actions()[0].text(), "File")
        self.assertEqual(embedded_menu.actions()[1].text(), "View")
        self.assertEqual(self.window.new_session_button.iconSize().width(), 14)
        self.assertEqual(self.window.new_project_button.iconSize().width(), 14)
        self.assertEqual(self.window.info_button.iconSize().width(), 14)

    def test_approval_dialog_uses_smaller_default_size(self):
        dialog = agent_cli.ApprovalDialog(
            {
                "tools": [{"name": "edit_file", "args": {"path": "demo.txt"}, "policy": {"mutating": True}}],
                "summary": {"risk_level": "medium", "impacts": ["files"], "default_approve": True},
            }
        )
        self.assertLessEqual(dialog.width(), 470)
        self.assertLessEqual(dialog.height(), 320)
        dialog.close()

    def test_transcript_sticky_autofollow_respects_manual_scroll(self):
        scrollbar = self.window.transcript.scroll.verticalScrollBar()
        scrollbar.setRange(0, 200)
        scrollbar.setValue(200)
        self.window.transcript.scroll_to_bottom()
        self.assertTrue(self.window.transcript.auto_follow_enabled)

        scrollbar.setValue(80)
        self._process_events()
        self.assertFalse(self.window.transcript.auto_follow_enabled)
        previous_value = scrollbar.value()

        self.window.transcript.notify_content_changed()
        self._process_events()
        self.assertEqual(scrollbar.value(), previous_value)

        scrollbar.setValue(scrollbar.maximum())
        self._process_events()
        self.assertTrue(self.window.transcript.auto_follow_enabled)

        scrollbar.setValue(120)
        self.window.transcript.notify_content_changed(force=True)
        self._process_events()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_transcript_uses_centered_column_instead_of_full_width_feed(self):
        self.assertEqual(self.window.transcript.column.maximumWidth(), 1180)
        self.assertEqual(self.window.transcript.column.objectName(), "TranscriptColumn")
        self.assertEqual(self.window.composer_container.maximumWidth(), 1180)
        self.assertEqual(self.window.composer_container.objectName(), "CenteredComposerRow")

    def test_composer_buttons_have_correct_tooltips(self):
        self.assertEqual(self.window.attach_button.toolTip(), "Attach file")
        self.assertEqual(self.window.send_button.toolTip(), "Send (Enter)")


if __name__ == "__main__":
    unittest.main()
