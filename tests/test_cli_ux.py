import os
import shutil
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QMimeData, QObject, Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QTextCursor, QTextFormat
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QFrame, QPushButton, QSizePolicy, QToolBar

import main as agent_cli
from core.model_profiles import normalize_profiles_payload
from core.tool_policy import ToolMetadata
from ui.runtime import build_runtime_snapshot, summarize_approval_request
from ui.streaming import StreamEvent
from ui.theme import AMBER_WARNING, BORDER, ERROR_RED, SURFACE_BG, SURFACE_CARD, TEXT_MUTED
from ui.widgets.foundation import AutoTextBrowser, CodeBlockWidget, CopySafePlainTextEdit, TRANSCRIPT_MAX_WIDTH
from ui.widgets.messages import AssistantMessageWidget


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
    user_choice_requested = Signal(object)
    session_changed = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.start_calls: list[object] = []
        self.resume_calls: list[tuple[bool, bool]] = []
        self.resume_choice_calls: list[str] = []
        self.new_session_calls = 0
        self.switch_session_calls: list[str] = []
        self.delete_session_calls: list[str] = []
        self.set_active_profile_calls: list[str] = []
        self.save_profiles_calls: list[dict] = []
        self.reinitialize_calls: list[bool] = []
        self.shutdown_calls = 0
        self.initialize_calls = 0

    def initialize(self):
        self.initialize_calls += 1

    def start_run(self, text: object):
        self.start_calls.append(text)

    def resume_approval(self, approved: bool, always: bool = False):
        self.resume_calls.append((approved, always))

    def resume_user_choice(self, chosen: str):
        self.resume_choice_calls.append(chosen)

    def new_session(self):
        self.new_session_calls += 1

    def switch_session(self, session_id: str):
        self.switch_session_calls.append(session_id)

    def delete_session(self, session_id: str):
        self.delete_session_calls.append(session_id)

    def set_active_profile(self, profile_id: str):
        self.set_active_profile_calls.append(profile_id)

    def save_profiles(self, config_payload: dict):
        self.save_profiles_calls.append(config_payload)

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

    def _wait_for_gui(self, ms: int):
        QTest.qWait(ms)
        self._process_events()

    def _press_composer_key(self, key: int, text: str = "", modifiers: Qt.KeyboardModifier = Qt.NoModifier):
        self.window.composer.setFocus()
        event = QKeyEvent(QEvent.KeyPress, key, modifiers, text)
        QApplication.sendEvent(self.window.composer, event)
        self._process_events()

    def _submit_text(self, text: str):
        self.window._set_input_enabled(True)
        self.window.composer.setPlainText(text)
        self.window._submit_request()

    def _make_test_image_file(self, name: str = "sample.png") -> str:
        temp_root = Path.cwd() / ".tmp_tests" / f"cli-ux-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))
        image_path = temp_root / name
        image = QImage(18, 12, QImage.Format_ARGB32)
        image.fill(0xFF3A7AFE)
        self.assertTrue(image.save(str(image_path), "PNG"))
        return str(image_path)

    def _make_test_file(self, name: str = "notes.txt", content: str = "demo") -> str:
        temp_root = Path.cwd() / ".tmp_tests" / f"cli-ux-file-{time.time_ns()}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: temp_root.exists() and shutil.rmtree(temp_root, ignore_errors=True))
        file_path = temp_root / name
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

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
            "model_capabilities": {"image_input_supported": True},
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
            "model_profiles": {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "supports_image_input": True,
                        "enabled": True,
                    },
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "supports_image_input": False,
                        "enabled": True,
                    },
                ],
            },
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

    def test_configure_qt_logging_adds_font_db_rule_once(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            agent_cli._configure_qt_logging()
            self.assertEqual(os.environ.get("QT_LOGGING_RULES"), "qt.text.font.db=false")
            agent_cli._configure_qt_logging()
            self.assertEqual(os.environ.get("QT_LOGGING_RULES"), "qt.text.font.db=false")

    def test_configure_qt_logging_preserves_existing_rules(self):
        with mock.patch.dict(os.environ, {"QT_LOGGING_RULES": "qt.network.ssl.warning=true"}, clear=False):
            agent_cli._configure_qt_logging()
            self.assertEqual(
                os.environ.get("QT_LOGGING_RULES"),
                "qt.network.ssl.warning=true;qt.text.font.db=false",
            )

    def test_submit_request_uses_controller_and_clears_editor(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Собери summary")

        self.window._submit_request()

        self.assertEqual(self.controller.start_calls, [{"text": "Собери summary", "attachments": []}])
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_send_button_keeps_visible_disabled_icon_until_input_exists(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._set_input_enabled(True)

        disabled_icon_key = self.window.send_button.icon().cacheKey()
        self.assertFalse(self.window.send_button.isEnabled())

        self.window.composer.setPlainText("go")
        self.window._refresh_submit_controls()
        self._process_events()

        enabled_icon_key = self.window.send_button.icon().cacheKey()
        self.assertTrue(self.window.send_button.isEnabled())
        self.assertNotEqual(disabled_icon_key, enabled_icon_key)

    def test_assistant_message_widget_renders_unclosed_fenced_block_as_code_widget(self):
        widget = AssistantMessageWidget()
        self.addCleanup(widget.deleteLater)

        widget.set_markdown("Вот код:\n```python\nprint('hi')\n")
        self._process_events()

        self.assertEqual(len(widget.parts_widgets), 2)
        self.assertIsInstance(widget.parts_widgets[0], AutoTextBrowser)
        self.assertIsInstance(widget.parts_widgets[1], CodeBlockWidget)
        self.assertEqual(widget.parts_widgets[1].editor.toPlainText(), "print('hi')")
        self.assertEqual(widget.parts_widgets[1].title_label.text(), "PYTHON")

    def test_user_choice_card_renders_above_composer_and_resumes_selected_option(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Выбери режим"}))
        self.controller.user_choice_requested.emit(
            {
                "question": "Какой режим выбираем?",
                "recommended_key": "direct_api",
                "options": [
                    {
                        "key": "direct_api",
                        "label": "direct_api: убрать MCP и проверить только API",
                        "submit_text": "direct_api",
                        "recommended": True,
                    },
                    {
                        "key": "keep_mcp",
                        "label": "keep_mcp: оставить MCP и настроить сервер",
                        "submit_text": "keep_mcp",
                        "recommended": False,
                    },
                ],
            }
        )
        self._process_events()

        self.assertFalse(self.window.user_choice_card.isHidden())
        self.assertEqual(self.window.user_choice_card.title_label.text(), "Нужен выбор пользователя")
        self.assertEqual(self.window.user_choice_card.question_label.text(), "Какой режим выбираем?")
        option_buttons = self.window.user_choice_card.findChildren(QPushButton, "UserChoiceOptionButton")
        self.assertEqual(len(option_buttons), 2)
        self.assertTrue(option_buttons[0].property("recommended"))

        QTest.mouseClick(option_buttons[0], Qt.LeftButton)
        self._process_events()

        self.assertEqual(self.controller.start_calls, [])
        self.assertEqual(self.controller.resume_choice_calls, ["direct_api"])
        self.assertTrue(self.window.user_choice_card.isHidden())

    def test_user_choice_card_custom_option_arms_composer_and_resumes_on_submit(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.controller.user_choice_requested.emit(
            {
                "question": "Как продолжаем?",
                "recommended_key": "",
                "options": [
                    {
                        "key": "direct_api",
                        "label": "direct_api: тестируем только API",
                        "submit_text": "direct_api",
                        "recommended": False,
                    }
                ],
            }
        )
        self.window.composer.setPlainText("Мой вариант")
        self._process_events()

        QTest.mouseClick(self.window.user_choice_card.custom_button, Qt.LeftButton)
        self._process_events()

        self.assertFalse(self.window.user_choice_card.isHidden())
        self.assertEqual(self.window.composer.toPlainText(), "Мой вариант")
        self.assertEqual(self.window.composer.textCursor().selectedText(), "Мой вариант")

        self.window._submit_request()

        self.assertEqual(self.controller.start_calls, [])
        self.assertEqual(self.controller.resume_choice_calls, ["Мой вариант"])
        self.assertTrue(self.window.user_choice_card.isHidden())

    def test_composer_enter_submits_while_shift_enter_adds_newline(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("line1")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        self._press_composer_key(Qt.Key_Return, "\n", Qt.ShiftModifier)
        self.assertEqual(self.controller.start_calls, [])
        self.assertIn("\n", self.window.composer.toPlainText())

        self.window.composer.setPlainText("Сделай задачу")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self._press_composer_key(Qt.Key_Return, "\r")

        self.assertEqual(self.controller.start_calls, [{"text": "Сделай задачу", "attachments": []}])
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_paste_image_creates_draft_attachment_chip(self):
        self.window._handle_initialized(self._snapshot_payload())
        mime = QMimeData()
        image = QImage(14, 10, QImage.Format_ARGB32)
        image.fill(0xFF44AA66)
        mime.setImageData(image)

        self.window.composer.insertFromMimeData(mime)
        self._process_events()

        self.assertEqual(len(self.window.draft_image_attachments), 1)
        self.assertFalse(self.window.composer_attachments_strip.isHidden())
        self.assertEqual(len(self.window.composer_attachments_strip._chips), 1)
        self.assertTrue(self.window.send_button.isEnabled())
        self.assertFalse(self.window.composer_notice_label.isVisible())

    def test_run_started_with_attachments_renders_user_preview_row(self):
        self.window._handle_initialized(self._snapshot_payload())
        attachment = {
            "id": "img-1",
            "path": self._make_test_image_file(),
            "mime_type": "image/png",
            "file_name": "sample.png",
            "width": 18,
            "height": 12,
            "size_bytes": 128,
        }

        self.window._handle_event(
            StreamEvent("run_started", {"text": "Опиши изображение", "attachments": [attachment]})
        )
        self._process_events()

        self.assertIsNotNone(self.window.current_turn)
        user_widget = self.window.current_turn._timeline[0][1]
        self.assertFalse(user_widget.attachments_strip.isHidden())
        self.assertEqual(len(user_widget.attachments_strip._chips), 1)

    def test_no_img_badge_is_visible_and_image_paste_shows_notice(self):
        payload = self._snapshot_payload()
        payload["model_capabilities"] = {"image_input_supported": False}
        payload["model_profiles"]["profiles"][0]["supports_image_input"] = False
        self.window._handle_initialized(payload)
        mime = QMimeData()
        image = QImage(14, 10, QImage.Format_ARGB32)
        image.fill(0xFFAA6644)
        mime.setImageData(image)

        self.window.composer.insertFromMimeData(mime)
        self._process_events()

        self.assertFalse(self.window.model_image_badge.isHidden())
        self.assertEqual(self.window.model_image_badge.text(), "no img")
        self.assertEqual(self.window.draft_image_attachments, [])
        self.assertFalse(self.window.composer_notice_label.isHidden())
        self.assertIn("does not support image input", self.window.composer_notice_label.text())

    def test_profile_checkbox_can_override_runtime_no_img_badge(self):
        payload = self._snapshot_payload()
        payload["model_capabilities"] = {"image_input_supported": False}
        payload["model_profiles"]["profiles"][0]["supports_image_input"] = True

        self.window._handle_initialized(payload)
        self.window._set_input_enabled(True)

        self.assertTrue(self.window.model_image_badge.isHidden())
        self.assertTrue(self.window.add_image_action.isEnabled())

    def test_insert_file_paths_keeps_existing_text_reference_flow(self):
        self.window._handle_initialized(self._snapshot_payload())
        file_path = self._make_test_file()

        with mock.patch.object(agent_cli.QFileDialog, "getOpenFileNames", return_value=([file_path], "")):
            self.window._insert_file_paths()

        self.assertIn(self.window.composer.format_file_reference(file_path), self.window.composer.toPlainText())
        self.assertEqual(self.window.draft_image_attachments, [])

    def test_composer_paste_single_line_text_drops_trailing_newline(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Открой ")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("core/nodes.py\r\n")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(self.window.composer.toPlainText(), "Открой core/nodes.py")

    def test_composer_paste_single_line_text_drops_leading_and_trailing_newlines(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Открой ")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("\r\ncore/nodes.py\r\n")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(self.window.composer.toPlainText(), "Открой core/nodes.py")

    def test_composer_paste_multiline_text_keeps_newlines(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.setPlainText("Список:\n")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        mime = QMimeData()
        mime.setText("first.py\nsecond.py\n")
        self.window.composer.insertFromMimeData(mime)

        self.assertEqual(self.window.composer.toPlainText(), "Список:\nfirst.py\nsecond.py\n")

    def test_copy_safe_plain_text_edit_copies_plain_text_without_hidden_paragraph_breaks(self):
        editor = CopySafePlainTextEdit()
        editor.setPlainText("alpha.py")
        editor.selectAll()
        editor.copy()

        self.assertEqual(QApplication.clipboard().text(), "alpha.py")

    def test_copy_safe_plain_text_edit_strips_spurious_edge_newlines_for_single_line_selection(self):
        editor = CopySafePlainTextEdit()
        editor.setPlainText("\nalpha.py\n")
        cursor = editor.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        editor.setTextCursor(cursor)

        mime = editor.createMimeDataFromSelection()

        self.assertEqual(mime.text(), "alpha.py")

    def test_auto_text_browser_copy_normalizes_qt_paragraph_separators(self):
        browser = AutoTextBrowser()
        browser.setMarkdown("first line\n\nsecond line")
        cursor = browser.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        browser.setTextCursor(cursor)

        browser.copy()

        self.assertEqual(QApplication.clipboard().text(), "first line\nsecond line")

    def test_auto_text_browser_copy_strips_spurious_edge_newlines_for_single_line_selection(self):
        browser = AutoTextBrowser()
        browser.setMarkdown("alpha.py")
        cursor = browser.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        browser.setTextCursor(cursor)

        mime = browser.createMimeDataFromSelection()

        self.assertEqual(mime.text(), "alpha.py")

    def test_composer_history_navigation_works_when_empty_and_dedupes_adjacent_entries(self):
        self.window._handle_initialized(self._snapshot_payload())
        self._submit_text("cmd alpha")
        self._submit_text("cmd alpha")
        self._submit_text("cmd beta")
        self.window._set_input_enabled(True)
        self.window.composer.clear()

        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "cmd beta")
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "cmd alpha")
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "cmd alpha")
        self._press_composer_key(Qt.Key_Down)
        self.assertEqual(self.window.composer.toPlainText(), "cmd beta")
        self._press_composer_key(Qt.Key_Down)
        self.assertEqual(self.window.composer.toPlainText(), "")

    def test_composer_up_down_do_not_override_text_when_editor_not_empty(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.append_submitted_message("history item")
        self.window.composer.setPlainText("typed")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        self._press_composer_key(Qt.Key_Up)

        self.assertEqual(self.window.composer.toPlainText(), "typed")

    def test_composer_history_is_restored_from_transcript_payload(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {"user_text": "first", "blocks": []},
                {"user_text": "first", "blocks": []},
                {"user_text": "second", "blocks": []},
            ],
        }
        self.window._handle_initialized(payload)
        self.window.composer.clear()

        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "second")
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "first")

    def test_composer_history_is_session_scoped(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_history_session("session-a")
        self.window.composer.append_submitted_message("alpha")
        self.window.composer.set_history_session("session-b")
        self.window.composer.append_submitted_message("beta")

        self.window.composer.set_history_session("session-a")
        self.window.composer.clear()
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "alpha")

        self.window.composer.set_history_session("session-b")
        self.window.composer.clear()
        self._press_composer_key(Qt.Key_Up)
        self.assertEqual(self.window.composer.toPlainText(), "beta")

    def test_mention_popup_selects_file_and_has_priority_over_submit(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_file_index_for_testing(
            ["main.py", "manual/main_notes.md", "core/gui_widgets.py"]
        )
        self.window.composer.setPlainText("@ma")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)

        self._press_composer_key(Qt.Key_I, "i")
        self.assertTrue(self.window.composer._mention_popup.isVisible())

        self._press_composer_key(Qt.Key_Down)
        self._press_composer_key(Qt.Key_Return, "\r")

        self.assertEqual(self.window.composer.toPlainText(), "manual/main_notes.md")
        self.assertFalse(self.window.composer._mention_popup.isVisible())
        self.assertEqual(self.controller.start_calls, [])

    def test_mention_popup_opens_on_at_character_input(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_file_index_for_testing(["main.py"])
        at_key = Qt.Key_At if hasattr(Qt, "Key_At") else Qt.Key_A

        self._press_composer_key(at_key, "@")

        self.assertTrue(self.window.composer._mention_popup.isVisible())

    def test_mention_popup_shows_root_files_first_and_is_wider(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_file_index_for_testing(
            ["sub/main.py", "root.py", "nested/deep/file.txt"]
        )
        self.window.composer.setPlainText("@")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()

        self.assertTrue(self.window.composer._mention_popup.isVisible())
        self.assertEqual(self.window.composer._mention_popup.current_relative_path(), "root.py")
        self.assertGreaterEqual(self.window.composer._mention_popup.width(), 560)
        self.assertIn("/", self.window.composer._mention_popup.list_widget.item(1).text())

    def test_mention_popup_includes_directories_from_indexed_files(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_file_index_for_testing(
            ["docs/readme.md", "docs/nested/info.txt", "main.py"]
        )
        self.window.composer.setPlainText("@do")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()

        self.assertTrue(self.window.composer._mention_popup.isVisible())
        items = [
            self.window.composer._mention_popup.list_widget.item(index).text()
            for index in range(self.window.composer._mention_popup.list_widget.count())
        ]
        self.assertIn("docs/", items)
        self.assertIn("docs/readme.md", items)

    def test_mention_popup_closes_on_escape_no_matches_and_cursor_change(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window.composer.set_file_index_for_testing(["main.py"])
        self.window.composer.setPlainText("@ma")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()
        self.assertTrue(self.window.composer._mention_popup.isVisible())

        self._press_composer_key(Qt.Key_Escape)
        self.assertFalse(self.window.composer._mention_popup.isVisible())

        self.window.composer.setPlainText("@zz")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()
        self.assertFalse(self.window.composer._mention_popup.isVisible())

        self.window.composer.setPlainText("@ma")
        self.window.composer.moveCursor(QTextCursor.MoveOperation.End)
        self.window.composer._refresh_mention_popup()
        self.assertTrue(self.window.composer._mention_popup.isVisible())
        cursor = self.window.composer.textCursor()
        cursor.setPosition(0)
        self.window.composer.setTextCursor(cursor)
        self.window.composer._refresh_mention_popup()
        self.assertFalse(self.window.composer._mention_popup.isVisible())

    def test_run_started_shows_inline_status_before_output(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.window._handle_event(StreamEvent("run_started", {"text": "Сводка"}))
        self.window._handle_event(StreamEvent("status_changed", {"label": "Self-correcting", "node": "stability_guard"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Self-correcting")

    def test_run_started_requeues_autofollow_after_inserting_thinking_status(self):
        self.window._handle_initialized(self._snapshot_payload())
        with mock.patch.object(
            self.window.transcript,
            "notify_content_changed",
            wraps=self.window.transcript.notify_content_changed,
        ) as notify_mock:
            self.window._handle_event(StreamEvent("run_started", {"text": "Ещё запрос"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Analyzing request")
        self.assertGreaterEqual(notify_mock.call_count, 2)
        self.assertEqual(notify_mock.call_args_list[-1].kwargs.get("force"), True)

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
        self.window._handle_event(StreamEvent("status_changed", {"label": "Self-correcting", "node": "stability_guard"}))

        self.assertIsNotNone(self.window.current_turn.status_widget)
        self.assertEqual(self.window.current_turn.status_widget.label.text(), "Self-correcting")

    def test_auto_summary_notice_shows_progress_then_done_and_hides_after_output(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Большой контекст"}))

        self.window._handle_event(
            StreamEvent("status_changed", {"label": "Compressing context", "node": "summarize"})
        )
        self.assertIsNotNone(self.window.current_turn.summary_notice_widget)
        self.assertIn("сжимается", self.window.current_turn.summary_notice_widget.text_label.text().lower())

        self.window._handle_event(
            StreamEvent("status_changed", {"label": "Thinking", "node": "agent"})
        )
        self.assertIsNotNone(self.window.current_turn.summary_notice_widget)
        self.assertIn("контекст сжат", self.window.current_turn.summary_notice_widget.text_label.text().lower())

        self.window._handle_event(
            StreamEvent(
                "assistant_delta",
                {
                    "text": "готово",
                    "full_text": "Готово",
                    "has_thought": False,
                },
            )
        )
        self._process_events()
        self.assertIsNone(self.window.current_turn.summary_notice_widget)

    def test_auto_summary_notice_event_is_transient_not_persistent_notice_block(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Большой контекст"}))
        self.window._handle_event(
            StreamEvent(
                "summary_notice",
                {"kind": "auto_summary", "count": 3, "message": "Context compressed automatically"},
            )
        )
        self._process_events()

        self.assertIsNotNone(self.window.current_turn.summary_notice_widget)
        self.assertIn("контекст автоматически сжат", self.window.current_turn.summary_notice_widget.text_label.text().lower())
        self.assertNotIn("notice", self.window.current_turn.block_kinds())

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
        self.assertEqual(tool_card.diff_section.content.path_label.text(), "demo.txt")
        self.assertEqual(tool_card.diff_section.content.added_label.text(), "+1")
        self.assertEqual(tool_card.diff_section.content.removed_label.text(), "-1")
        rendered_diff = tool_card.diff_section.content.editor.toPlainText()
        self.assertIn(" + bar", rendered_diff)
        self.assertNotIn("--- ", rendered_diff)
        self.assertNotIn("+++ ", rendered_diff)
        self.assertNotIn("@@ ", rendered_diff)
        full_width_selections = tool_card.diff_section.content.editor.extraSelections()
        self.assertEqual(len(full_width_selections), 2)
        self.assertTrue(all(bool(sel.format.property(QTextFormat.FullWidthSelection)) for sel in full_width_selections))
        selection_colors = {sel.format.background().color().name().lower() for sel in full_width_selections}
        self.assertEqual(selection_colors, {"#1e3425", "#472b2b"})
        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.args_container.isHidden())
        self.assertIn("Success", tool_card.args_view.toPlainText())
        self.assertNotIn('"path"', tool_card.args_view.toPlainText())

    def test_assistant_message_widget_preserves_prose_around_fenced_json_block(self):
        widget = AssistantMessageWidget()
        widget.set_markdown(
            "Ошибка парсинга JSON: модель часто оборачивает JSON в блоки кода (\n"
            "```json\n"
            '{"name":"cli_exec","args":{"command":"python test.py"}}\n'
            "```\n"
            "), которые стандартный json.loads не обрабатывает."
        )
        self._process_events()

        visible_parts = [part for part in widget.parts_widgets if not part.isHidden()]
        self.assertEqual(len(visible_parts), 3)
        self.assertIsInstance(visible_parts[0], AutoTextBrowser)
        self.assertIsInstance(visible_parts[1], CodeBlockWidget)
        self.assertIsInstance(visible_parts[2], AutoTextBrowser)
        self.assertIn("Ошибка парсинга JSON", visible_parts[0].toPlainText())
        self.assertEqual(visible_parts[1].title_label.text(), "JSON")
        self.assertIn('"name":"cli_exec"', visible_parts[1].editor.toPlainText())
        self.assertIn("json.loads", visible_parts[2].toPlainText())

    def test_preview_tool_card_stays_hidden_until_resolved_refresh_arrives(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Сохрани файл"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-preview",
                    "name": "write_file",
                    "args": {},
                    "display": "Preparing file write",
                    "subtitle": "Waiting for arguments…",
                    "raw_display": "write_file",
                    "args_state": "pending",
                    "display_state": "preview",
                    "phase": "preparing",
                    "source_kind": "tool",
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-preview"]
        self.assertTrue(tool_card.isHidden())

        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-preview",
                    "name": "write_file",
                    "args": {"path": "notes.md"},
                    "display": "Writing file",
                    "subtitle": "notes.md",
                    "raw_display": "write_file(notes.md)",
                    "args_state": "complete",
                    "display_state": "resolved",
                    "phase": "running",
                    "source_kind": "tool",
                    "refresh": True,
                },
            )
        )
        self._process_events()

        self.assertFalse(tool_card.isHidden())
        self.assertEqual(tool_card.tool_button.text(), "Writing file")
        self.assertEqual(tool_card.subtitle_label.text(), "notes.md")
        self.assertEqual(tool_card.phase_badge.text(), "Running")
        self.assertNotIn("write_file()", tool_card.tool_button.text())

    def test_cli_exec_renders_live_terminal_panel_and_streams_output(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli",
                    "name": "cli_exec",
                    "args": {"command": "echo hello"},
                    "display": 'cli_exec("echo hello")',
                },
            )
        )
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-cli", "data": "hello\n", "stream": "stdout"})
        )
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-cli", "data": "world\n", "stream": "stdout"})
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli"]
        self.assertFalse(tool_card.header_container.isHidden())
        self.assertTrue(tool_card.args_container.isHidden())
        self.assertTrue(tool_card.tool_button.isEnabled())
        self.assertTrue(tool_card.tool_button.isCheckable())
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertIn("cli_exec", tool_card.tool_button.text())
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertFalse(tool_card.cli_exec_widget.isHidden())
        self.assertEqual(tool_card.cli_exec_widget.command_label.text(), "$ echo hello")
        output_text = tool_card.cli_exec_widget.output_view.toPlainText()
        self.assertIn("hello", output_text)
        self.assertIn("world", output_text)
        self.assertLessEqual(tool_card.cli_exec_widget.output_view.height(), 90)

        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.cli_exec_widget.isHidden())
        tool_card.tool_button.click()
        self._process_events()
        self.assertFalse(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_header_command_is_single_line_for_multiline_command(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти python"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-heredoc",
                    "name": "cli_exec",
                    "args": {"command": "python - <<'PY'\nimport sys\nprint(sys.version)\nPY"},
                    "display": 'cli_exec("python - <<\'PY\' import sys print(sys.version) PY")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-heredoc"]
        header_text = tool_card.cli_exec_widget.command_label.text()
        self.assertTrue(header_text.startswith("$ "))
        self.assertNotIn("\n", header_text)
        self.assertIn("python - <<'PY'", header_text)

    def test_cli_exec_header_uses_eliding_policy_and_does_not_force_wide_layout(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь ширину"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-wide",
                    "name": "cli_exec",
                    "args": {"command": "python -c \"" + ("x" * 800) + "\""},
                    "display": 'cli_exec("python -c ...")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-wide"]
        label = tool_card.cli_exec_widget.command_label
        self.assertEqual(label.sizePolicy().horizontalPolicy(), QSizePolicy.Ignored)
        label.setFixedWidth(180)
        self._process_events()
        self.assertNotIn("\n", label.text())
        self.assertTrue(bool(label.toolTip()))
        self.assertGreater(len(label.toolTip()), len(label.text()))

    def test_cli_exec_card_created_by_output_is_refreshed_when_tool_started_arrives_later(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь race"}))
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-race", "data": "line 1\n", "stream": "stdout"})
        )
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-race",
                    "name": "cli_exec",
                    "args": {"command": "echo race"},
                    "display": 'cli_exec("echo race")',
                    "refresh": True,
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-race"]
        self.assertIn("echo race", tool_card.tool_button.text())
        self.assertEqual(tool_card.cli_exec_widget.command_label.text(), "$ echo race")

    def test_cli_exec_output_autofollow_respects_manual_scroll(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Tail logs"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-tail",
                    "name": "cli_exec",
                    "args": {"command": "tail -f app.log"},
                    "display": 'cli_exec("tail -f app.log")',
                },
            )
        )
        for idx in range(90):
            self.window._handle_event(
                StreamEvent(
                    "cli_output",
                    {"tool_id": "call-tail", "data": f"line {idx}\n", "stream": "stdout"},
                )
            )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-tail"]
        output_view = tool_card.cli_exec_widget.output_view
        scrollbar = output_view.verticalScrollBar()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

        scrollbar.setValue(max(0, scrollbar.maximum() - 20))
        self._process_events()
        previous_value = scrollbar.value()
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-tail", "data": "manual-check\n", "stream": "stdout"})
        )
        self._process_events()
        self.assertEqual(scrollbar.value(), previous_value)

        scrollbar.setValue(scrollbar.maximum())
        self._process_events()
        self.window._handle_event(
            StreamEvent("cli_output", {"tool_id": "call-tail", "data": "follow-bottom\n", "stream": "stdout"})
        )
        self._process_events()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_cli_exec_collapses_after_finish_by_default(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-finish",
                    "name": "cli_exec",
                    "args": {"command": "echo done"},
                    "display": 'cli_exec("echo done")',
                },
            )
        )
        self.window._handle_event(
            StreamEvent("tool_finished", {"tool_id": "call-cli-finish", "name": "cli_exec", "content": "done\n"})
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli-finish"]
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertFalse(tool_card.cli_exec_widget.isHidden())
        self.assertIn("done", tool_card.cli_exec_widget.output_view.toPlainText())
        self._wait_for_gui(950)
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_finish_collapses_even_if_user_kept_card_open(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-finish-open",
                    "name": "cli_exec",
                    "args": {"command": "echo done"},
                    "display": 'cli_exec("echo done")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli-finish-open"]
        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.cli_exec_widget.isHidden())

        # User toggles while running; finish still auto-collapses by policy.
        tool_card.tool_button.click()
        self._process_events()
        self.assertTrue(tool_card.cli_exec_widget.isHidden())
        tool_card.tool_button.click()
        self._process_events()
        self.assertFalse(tool_card.cli_exec_widget.isHidden())

        self.window._handle_event(
            StreamEvent("tool_finished", {"tool_id": "call-cli-finish-open", "name": "cli_exec", "content": "done\n"})
        )
        self._process_events()

        self.assertTrue(tool_card.tool_button.isChecked())
        self.assertFalse(tool_card.cli_exec_widget.isHidden())
        self._wait_for_gui(950)
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_long_running_finishes_without_extra_delay(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Запусти команду"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-long",
                    "name": "cli_exec",
                    "args": {"command": "echo done"},
                    "display": 'cli_exec("echo done")',
                },
            )
        )
        self._process_events()

        tool_card = self.window.current_turn.tool_cards["call-cli-long"]
        tool_card._cli_started_at_monotonic = time.monotonic() - 1.2

        self.window._handle_event(
            StreamEvent("tool_finished", {"tool_id": "call-cli-long", "name": "cli_exec", "content": "done\n"})
        )
        self._process_events()

        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.cli_exec_widget.isHidden())

    def test_cli_exec_restored_from_transcript_is_collapsed(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "run cli",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-cli-restored",
                                "name": "cli_exec",
                                "args": {"command": "python --version"},
                                "display": 'cli_exec("python --version")',
                                "content": "Python 3.12.9\n",
                                "duration": 0.1,
                            },
                        }
                    ],
                }
            ],
        }
        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        tool_card = restored_turn.tool_cards["call-cli-restored"]
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertTrue(tool_card.cli_exec_widget.isHidden())
        self.assertEqual(tool_card.phase_badge.text(), "✓")

    def test_restoring_transcript_does_not_show_transient_top_level_widgets(self):
        class _ShowSpy(QObject):
            def __init__(self, main_window) -> None:
                super().__init__()
                self.main_window = main_window
                self.top_level_widgets: list[tuple[str, str, str]] = []

            def eventFilter(self, obj, event):  # type: ignore[override]
                if event.type() != QEvent.Show:
                    return False
                if not hasattr(obj, "isWindow") or not hasattr(obj, "parentWidget"):
                    return False
                if not obj.isWindow():
                    return False
                if obj is self.main_window:
                    return False
                self.top_level_widgets.append(
                    (
                        type(obj).__name__,
                        getattr(obj, "objectName", lambda: "")(),
                        getattr(obj, "windowTitle", lambda: "")(),
                    )
                )
                return False

        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "прочитай и поправь",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-read",
                                "name": "read_file",
                                "args": {"path": "index.html"},
                                "display": "Reading file",
                                "subtitle": "index.html",
                                "raw_display": "read_file(index.html)",
                                "args_state": "complete",
                                "display_state": "finished",
                                "phase": "finished",
                                "source_kind": "tool",
                                "content": "Read 10 lines.",
                            },
                        },
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-edit",
                                "name": "edit_file",
                                "args": {"path": "index.html"},
                                "display": "Editing file",
                                "subtitle": "index.html",
                                "raw_display": "edit_file(index.html)",
                                "args_state": "complete",
                                "display_state": "finished",
                                "phase": "finished",
                                "source_kind": "tool",
                                "content": "Success: File edited.",
                                "diff": "@@ -1 +1 @@\n-old\n+new",
                            },
                        },
                    ],
                }
            ],
        }
        spy = _ShowSpy(self.window)
        self.app.installEventFilter(spy)
        self.addCleanup(lambda: self.app.removeEventFilter(spy))
        self.window.show()
        self._process_events()

        self.window._handle_initialized(payload)
        self._process_events()

        self.assertEqual(spy.top_level_widgets, [])

    def test_finished_tool_restored_from_transcript_keeps_success_badge(self):
        payload = self._snapshot_payload()
        payload["transcript"] = {
            "summary_notice": "",
            "turns": [
                {
                    "user_text": "прочитай файл",
                    "blocks": [
                        {
                            "type": "tool",
                            "payload": {
                                "tool_id": "call-restored-finished",
                                "name": "read_file",
                                "args": {"path": "index.html"},
                                "display": "Reading file",
                                "subtitle": "index.html",
                                "raw_display": "read_file(index.html)",
                                "args_state": "complete",
                                "display_state": "finished",
                                "phase": "finished",
                                "source_kind": "tool",
                                "summary": "Read 78 lines (3574 chars)",
                                "content": "Read 78 lines.",
                                "is_error": False,
                            },
                        }
                    ],
                }
            ],
        }

        self.window._handle_initialized(payload)
        self._process_events()

        restored_turn = self.window.transcript.layout.itemAt(0).widget()
        tool_card = restored_turn.tool_cards["call-restored-finished"]
        self.assertEqual(tool_card.tool_button.text(), "Reading file")
        self.assertTrue(tool_card.subtitle_label.text().startswith("index.html · Read 78"))
        self.assertEqual(tool_card.subtitle_label.toolTip(), "read_file(index.html)")
        self.assertEqual(tool_card.phase_badge.text(), "✓")
        self.assertNotEqual(tool_card.phase_badge.text(), "Running")

    def test_tool_error_output_is_collapsed_by_default(self):
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
        self.assertFalse(tool_card.tool_button.isChecked())
        self.assertTrue(tool_card.args_container.isHidden())
        tool_card.tool_button.click()
        self._process_events()
        self.assertIn("error[access_denied]", tool_card.args_view.toPlainText().lower())

    def test_cli_exec_error_meta_is_marked_for_error_styling(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Почини"}))
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-cli-err",
                    "name": "cli_exec",
                    "args": {"command": "python bad.py"},
                    "content": "boom",
                    "is_error": True,
                    "duration": 0.4,
                },
            )
        )

        tool_card = self.window.current_turn.tool_cards["call-cli-err"]
        self.assertIsNotNone(tool_card.cli_exec_widget)
        self.assertEqual(tool_card.cli_exec_widget.meta_label.property("severity"), "error")
        self.assertTrue(tool_card.cli_exec_widget.meta_label.text().lower().startswith("error"))

    def test_hidden_internal_notice_event_is_ignored_in_ui(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь остановку"}))
        self.window._handle_event(
            StreamEvent(
                "summary_notice",
                {
                    "kind": "agent_internal_notice",
                    "message": "Автоматическое продолжение остановлено. Нужен новый запрос.",
                    "level": "warning",
                },
            )
        )

        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])

    def test_tool_args_missing_diagnostic_is_not_shown_in_transcript(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь list_directory"}))
        self.window._handle_event(
            StreamEvent(
                "tool_args_missing",
                {
                    "tool_id": "call-dir",
                    "name": "list_directory",
                    "message": "No canonical tool args were available when tool result arrived.",
                },
            )
        )

        self.assertEqual(self.window.current_turn.block_kinds(), ["user"])

    def test_cli_exec_display_strips_ansi_sequences(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Проверь ansi"}))
        self.window._handle_event(
            StreamEvent(
                "tool_started",
                {
                    "tool_id": "call-cli-ansi",
                    "name": "cli_exec",
                    "args": {"command": "bad-command"},
                    "display": 'cli_exec("bad-command")',
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "cli_output",
                {
                    "tool_id": "call-cli-ansi",
                    "stream": "stderr",
                    "data": "\u001b[31mboom\u001b[0m\n",
                },
            )
        )
        self.window._handle_event(
            StreamEvent(
                "tool_finished",
                {
                    "tool_id": "call-cli-ansi",
                    "name": "cli_exec",
                    "args": {"command": "bad-command"},
                    "content": "\u001b[31mboom\u001b[0m\n",
                    "is_error": True,
                    "duration": 0.3,
                },
            )
        )

        tool_card = self.window.current_turn.tool_cards["call-cli-ansi"]
        self.assertIsNotNone(tool_card.cli_exec_widget)
        rendered = tool_card.cli_exec_widget.output_view.toPlainText()
        self.assertEqual(rendered, "boom\n")
        self.assertNotIn("\u001b", rendered)

    def test_run_finished_renders_plain_stats_chip(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Сводка"}))
        self.window._handle_event(StreamEvent("status_changed", {"label": "Self-correcting", "node": "stability_guard"}))
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

    def test_auto_approval_after_always_mode_does_not_repeat_notice(self):
        self.window._handle_initialized(self._snapshot_payload())
        self.window._handle_event(StreamEvent("run_started", {"text": "Исправь файл"}))

        self.window._handle_event(
            StreamEvent("approval_resolved", {"approved": True, "always": True, "auto": False})
        )
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "notice"])

        self.window._handle_event(
            StreamEvent("approval_resolved", {"approved": True, "always": True, "auto": True})
        )
        self.assertEqual(self.window.current_turn.block_kinds(), ["user", "notice"])

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
        self.assertIn("border-radius: 0px", self.window.styleSheet())

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

    def test_transcript_autofollow_handles_range_growth_after_initial_scroll(self):
        scrollbar = self.window.transcript.scroll.verticalScrollBar()
        scrollbar.setRange(0, 200)
        scrollbar.setValue(200)
        self.window.transcript.scroll_to_bottom()
        self.assertTrue(self.window.transcript.auto_follow_enabled)

        self.window.transcript.notify_content_changed()
        self._process_events()
        scrollbar.setRange(0, 280)
        self._process_events()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_transcript_uses_centered_column_instead_of_full_width_feed(self):
        self.assertEqual(self.window.transcript.column.maximumWidth(), TRANSCRIPT_MAX_WIDTH)
        self.assertEqual(self.window.transcript.column.objectName(), "TranscriptColumn")
        self.assertEqual(self.window.composer_container.maximumWidth(), TRANSCRIPT_MAX_WIDTH)
        self.assertEqual(self.window.composer_container.objectName(), "CenteredComposerRow")
        self.assertGreaterEqual(self.window.composer_shell.contentsMargins().top(), 16)

    def test_composer_buttons_have_correct_tooltips(self):
        self.assertEqual(self.window.attach_button.toolTip(), "Add image or insert file path")
        self.assertEqual(self.window.send_button.toolTip(), "Send (Enter)")

    def test_model_selector_renders_active_profile_and_tooltip(self):
        self.window._handle_initialized(self._snapshot_payload())

        self.assertFalse(self.window.model_chip.isHidden())
        self.assertEqual(self.window.model_chip.text(), "gpt-4o")
        self.assertIn("Provider: openai", self.window.model_chip.toolTip())
        self.assertIn("Model: gpt-4o", self.window.model_chip.toolTip())
        self.assertTrue(self.window.no_models_label.isHidden())

    def test_model_selector_action_triggers_controller_switch(self):
        self.window._handle_initialized(self._snapshot_payload())
        actions = self.window.model_chip_menu.actions()
        target_action = next(action for action in actions if action.text() == "gemini-1-5-flash")

        target_action.trigger()

        self.assertEqual(self.controller.set_active_profile_calls, ["gemini-1-5-flash"])

    def test_no_models_cta_is_visible_and_send_is_disabled(self):
        payload = self._snapshot_payload()
        payload["model_profiles"] = {"active_profile": None, "profiles": []}
        self.window._handle_initialized(payload)
        self.window._set_input_enabled(True)

        self.assertTrue(self.window.model_chip.isHidden())
        self.assertFalse(self.window.no_models_label.isHidden())
        self.assertFalse(self.window.open_settings_inline_button.isHidden())
        self.assertFalse(self.window.send_button.isEnabled())

    def test_open_settings_dialog_saves_profiles_via_controller(self):
        self.window._handle_initialized(self._snapshot_payload())
        dialog_instance = mock.Mock()
        dialog_instance.exec.return_value = 1
        dialog_instance.result_payload.return_value = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                }
            ],
        }

        with mock.patch.object(agent_cli, "ModelSettingsDialog", return_value=dialog_instance):
            self.window._open_settings_dialog()

        self.assertEqual(
            self.controller.save_profiles_calls,
            [normalize_profiles_payload(dialog_instance.result_payload.return_value)],
        )
        self.assertEqual(self.window.model_profiles_payload["active_profile"], "gemini-1-5-flash")

    def test_model_settings_dialog_does_not_wipe_profile_on_initial_selection(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "openai/gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "supports_image_input": True,
                    "enabled": True,
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog._profiles[0]["model"], "openai/gpt-4o")
        self.assertEqual(dialog._profiles[0]["id"], "gpt-4o")
        self.assertEqual(dialog.model_edit.text(), "openai/gpt-4o")
        self.assertTrue(dialog.supports_images_checkbox.isChecked())

    def test_model_settings_dialog_opens_with_active_profile_selected(self):
        payload = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog._current_row(), 1)
        self.assertIn("gemini-1-5-flash", dialog.form_hint.text())

    def test_model_settings_dialog_autofills_name_from_model_suffix(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        dialog._add_profile()
        self._process_events()
        dialog.model_edit.setText("openai/gpt-oss-120b")
        self._process_events()

        self.assertEqual(dialog.name_edit.text(), "gpt-oss-120b")
        dialog._save_and_accept()
        result = dialog.result_payload()
        self.assertEqual(result["profiles"][0]["id"], "gpt-oss-120b")

    def test_model_settings_dialog_updates_auto_name_when_model_changes(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "openai/gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                    "supports_image_input": False,
                    "enabled": True,
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        dialog.model_edit.setText("openai/gpt-oss-120b")
        self._process_events()

        self.assertEqual(dialog.name_edit.text(), "gpt-oss-120b")

    def test_model_settings_dialog_saves_manual_image_support_checkbox(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        dialog._add_profile()
        self._process_events()

        dialog.provider_combo.setCurrentText("gemini")
        dialog.model_edit.setText("gemini-2.5-pro")
        dialog.api_key_edit.setText("gm-demo")
        dialog.supports_images_checkbox.setChecked(True)
        self._process_events()

        dialog._save_and_accept()
        result = dialog.result_payload()

        self.assertTrue(result["profiles"][0]["supports_image_input"])

    def test_model_settings_dialog_toggle_disables_profile_without_deleting_it(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        dialog._toggle_profile_enabled(0, False)
        dialog._save_and_accept()
        result = dialog.result_payload()

        self.assertFalse(result["profiles"][0]["enabled"])
        self.assertEqual(result["active_profile"], "gemini-1-5-flash")

    def test_model_settings_dialog_switch_can_disable_and_reenable_profile(self):
        payload = {
            "active_profile": "mistral-medium-latest",
            "profiles": [
                {
                    "id": "gemini-3-1-flash-lite-preview",
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite-preview",
                    "api_key": "gm-demo",
                    "base_url": "",
                    "enabled": True,
                },
                {
                    "id": "mistral-medium-latest",
                    "provider": "openai",
                    "model": "mistral-medium-latest",
                    "api_key": "sk-demo",
                    "base_url": "",
                    "enabled": True,
                },
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertGreaterEqual(dialog.profile_list.minimumWidth(), 280)

        first_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(first_item_widget)
        switch = first_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(switch)
        self.assertEqual(switch.size().width(), 30)
        self.assertTrue(switch.isChecked())

        QTest.mouseClick(switch, Qt.LeftButton)
        self._process_events()

        updated_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(updated_item_widget)
        updated_switch = updated_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(updated_switch)
        self.assertFalse(updated_switch.isChecked())

        QTest.mouseClick(updated_switch, Qt.LeftButton)
        self._process_events()

        reenabled_item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(reenabled_item_widget)
        reenabled_switch = reenabled_item_widget.findChild(QCheckBox, "ModelProfileEnabledSwitch")
        self.assertIsNotNone(reenabled_switch)
        self.assertTrue(reenabled_switch.isChecked())

        dialog._save_and_accept()
        result = dialog.result_payload()
        self.assertTrue(result["profiles"][0]["enabled"])

    def test_all_disabled_profiles_hide_model_picker_and_show_state(self):
        payload = self._snapshot_payload()
        for profile in payload["model_profiles"]["profiles"]:
            profile["enabled"] = False
        payload["model_profiles"]["active_profile"] = None

        self.window._handle_initialized(payload)
        self.window._set_input_enabled(True)

        self.assertTrue(self.window.model_chip.isHidden())
        self.assertFalse(self.window.no_models_label.isHidden())
        self.assertEqual(self.window.no_models_label.text(), "All models disabled")
        self.assertFalse(self.window.send_button.isEnabled())

    def test_model_settings_dialog_disables_base_url_for_gemini(self):
        payload = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-demo",
                    "base_url": "https://should-not-be-used.example",
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertFalse(dialog.base_url_edit.isEnabled())
        self.assertIn("Not used for gemini", dialog.base_url_edit.placeholderText())

    def test_model_settings_dialog_uses_improved_layout_and_save_state(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog.objectName(), "ModelSettingsDialog")
        self.assertIsNotNone(dialog.save_button)
        self.assertFalse(dialog.save_button.isEnabled())
        self.assertIn("Add a profile", dialog.form_hint.text())

        dialog._add_profile()
        self._process_events()
        self.assertTrue(dialog.save_button.isEnabled())

    def test_model_settings_dialog_profile_list_shows_provider_and_model(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "",
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertEqual(dialog.profile_list.count(), 1)
        item_widget = dialog.profile_list.itemWidget(dialog.profile_list.item(0))
        self.assertIsNotNone(item_widget)
        labels = item_widget.findChildren(QLabel)
        combined = "\n".join(label.text() for label in labels)
        self.assertIn("gpt-4o", combined)
        self.assertIn("openai", combined)

    def test_model_settings_dialog_enables_base_url_for_openai(self):
        payload = {
            "active_profile": "gpt-4o",
            "profiles": [
                {
                    "id": "gpt-4o",
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "sk-demo",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        dialog = agent_cli.ModelSettingsDialog(payload, self.window)
        self.addCleanup(dialog.close)
        self._process_events()

        self.assertTrue(dialog.base_url_edit.isEnabled())
        self.assertIn("api.openai.com", dialog.base_url_edit.placeholderText())

    def test_model_settings_dialog_clears_base_url_for_gemini_on_save(self):
        dialog = agent_cli.ModelSettingsDialog({"active_profile": None, "profiles": []}, self.window)
        self.addCleanup(dialog.close)
        dialog._add_profile()
        self._process_events()
        dialog.provider_combo.setCurrentText("gemini")
        dialog.model_edit.setText("gemini-1.5-flash")
        dialog.api_key_edit.setText("gm-demo")
        dialog.base_url_edit.setText("https://ignored.example")
        self._process_events()

        dialog._save_and_accept()
        result = dialog.result_payload()
        self.assertEqual(result["profiles"][0]["provider"], "gemini")
        self.assertEqual(result["profiles"][0]["base_url"], "")

    def test_composer_expands_to_max_height_then_uses_internal_scroll(self):
        self.window._handle_initialized(self._snapshot_payload())
        long_text = "\n".join(f"line-{idx}" for idx in range(80))
        self.window.composer.setPlainText(long_text)
        self.window._update_composer_height()
        self._process_events()

        self.assertEqual(self.window.composer.height(), self.window.composer.maximumHeight())
        self.assertGreater(self.window.composer.verticalScrollBar().maximum(), 0)

    def test_composer_bottom_controls_match_new_layout(self):
        self.assertTrue(hasattr(self.window, "model_chip"))
        self.assertTrue(hasattr(self.window, "effort_chip"))
        self.assertTrue(hasattr(self.window, "voice_button"))
        self.assertEqual(self.window.effort_chip.text(), "Высокий")
        self.assertEqual(self.window.voice_button.toolTip(), "Voice input (soon)")


if __name__ == "__main__":
    unittest.main()
