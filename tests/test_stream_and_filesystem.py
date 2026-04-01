import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import httpx
from langchain_core.messages import RemoveMessage, ToolMessage

from core.text_utils import prepare_markdown_for_render
from core.stream_processor import StreamProcessor
from tools import filesystem
from tools.filesystem import FilesystemManager, _DOWNLOAD_HEADERS, _format_download_http_error


class StreamAndFilesystemTests(unittest.TestCase):
    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_prepare_markdown_wraps_plain_go_code_block(self):
        source = 'Пример (файл main.go):\npackage main\nimport "fmt"\nfunc main() {\n    fmt.Println("hi")\n}'
        rendered = prepare_markdown_for_render(source)
        self.assertIn("```go", rendered)
        self.assertIn("package main", rendered)
        self.assertIn("fmt.Println", rendered)

    def test_download_headers_request_binary_content(self):
        self.assertEqual(_DOWNLOAD_HEADERS["Accept"], "*/*")

    def test_download_http_errors_are_specific(self):
        forbidden = httpx.HTTPStatusError(
            "forbidden",
            request=httpx.Request("GET", "https://example.com/file.mp4"),
            response=httpx.Response(403, request=httpx.Request("GET", "https://example.com/file.mp4")),
        )
        not_found = httpx.HTTPStatusError(
            "not found",
            request=httpx.Request("GET", "https://example.com/file.mp4"),
            response=httpx.Response(404, request=httpx.Request("GET", "https://example.com/file.mp4")),
        )
        self.assertIn("ACCESS_DENIED", _format_download_http_error(forbidden))
        self.assertIn("browser-only access", _format_download_http_error(forbidden))
        self.assertIn("NOT_FOUND", _format_download_http_error(not_found))
        self.assertIn("direct file", _format_download_http_error(not_found))

    def test_stream_processor_emits_tool_error_and_diff_events(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-1"] = {"name": "edit_file", "args": {"path": "demo.txt"}}

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-1",
                name="edit_file",
                content="Success: File edited.\n\nDiff:\n```diff\n-foo\n+bar\n```",
            )
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-2",
                name="read_file",
                content="ERROR[EXECUTION]: boom",
            )
        )

        event_types = [event.type for event in events]
        self.assertIn("tool_finished", event_types)
        self.assertIn("tool_diff", event_types)
        tool_finished_payloads = [event.payload for event in events if event.type == "tool_finished"]
        self.assertTrue(any(payload["name"] == "edit_file" and payload["diff"] for payload in tool_finished_payloads))
        self.assertTrue(any(payload["is_error"] and "boom" in payload["content"] for payload in tool_finished_payloads))

    def test_stream_processor_emits_summarization_notice(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_updates(
            {
                "summarize": {
                    "summary": "compressed summary",
                    "messages": [RemoveMessage(id="1"), RemoveMessage(id="2")],
                }
            }
        )

        notice_events = [event for event in events if event.type == "summary_notice"]
        self.assertEqual(len(notice_events), 1)
        self.assertIn("Context compressed automatically", notice_events[0].payload["message"])
        self.assertEqual(notice_events[0].payload["count"], 2)

    def test_stream_processor_merges_tool_args_and_finishes_with_canonical_payload(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-merge", "name": "edit_file", "args": {}})
        processor._emit_tool_started({"id": "call-merge", "name": "edit_file", "args": {}})
        processor._remember_tool_call(
            {
                "id": "call-merge",
                "name": "edit_file",
                "args": {"path": "demo.txt", "old_string": "old", "new_string": "new"},
            }
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-merge",
                name="edit_file",
                content="Success: File edited.",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(
            finished[0]["args"],
            {"path": "demo.txt", "old_string": "old", "new_string": "new"},
        )
        self.assertIn("demo.txt", finished[0]["display"])

    def test_stream_processor_finish_before_start_keeps_args_when_buffer_has_tool_call(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-late"] = {
            "name": "tail_file",
            "args": {"path": "service.log", "lines": 20},
        }

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-late",
                name="tail_file",
                content="Last 20 line(s)...",
            )
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["args"], {"path": "service.log", "lines": 20})
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["args"], {"path": "service.log", "lines": 20})

    def test_stream_processor_recovers_args_from_tool_message_metadata(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-meta",
                name="edit_file",
                content="Success: File edited.",
                additional_kwargs={
                    "tool_args": {
                        "path": "parse_yandex_forecast_fixed.py",
                        "old_string": "foo",
                        "new_string": "bar",
                    }
                },
            )
        )

        missing = [event for event in events if event.type == "tool_args_missing"]
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(missing, [])
        self.assertEqual(len(finished), 1)
        self.assertEqual(
            finished[0]["args"],
            {
                "path": "parse_yandex_forecast_fixed.py",
                "old_string": "foo",
                "new_string": "bar",
            },
        )
        self.assertIn("parse_yandex_forecast_fixed.py", finished[0]["display"])

    def test_filesystem_delete_uses_virtual_mode_path_guard(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        result = manager.delete_file("..\\outside.txt")
        self.assertIn("ERROR[EXECUTION]", result)
        self.assertIn("ACCESS DENIED", result)

    def test_filesystem_delete_directory_requires_recursive_for_non_empty(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        folder = tmp / "folder"
        folder.mkdir()
        (folder / "child.txt").write_text("data", encoding="utf-8")
        result = manager.delete_directory("folder")
        self.assertIn("recursive=True", result)

    def test_read_file_repairs_trailing_comma_in_existing_path(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        file_path = tmp / "model_info.md"
        file_path.write_text("hello", encoding="utf-8")

        result = manager.read_file("model_info.md, ", show_line_numbers=False)

        self.assertEqual(result, "hello")

    def test_module_filesystem_switches_workspace_after_directory_change(self):
        first = self._workspace_tempdir()
        second = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))

        (first / "from_first.txt").write_text("first", encoding="utf-8")
        (second / "from_second.txt").write_text("second", encoding="utf-8")

        filesystem.set_working_directory(str(first))
        first_result = filesystem.list_directory_tool.invoke({"path": "."})
        self.assertIn("from_first.txt", first_result)
        self.assertNotIn("from_second.txt", first_result)

        filesystem.set_working_directory(str(second))
        second_result = filesystem.list_directory_tool.invoke({"path": "."})
        self.assertIn("from_second.txt", second_result)
        self.assertNotIn("from_first.txt", second_result)

    def test_edit_file_accepts_legacy_aliases_for_old_and_new(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "demo.txt"
        target.write_text("hello old world", encoding="utf-8")

        result = filesystem.edit_file_tool.invoke(
            {
                "path": "demo.txt",
                "old_text": "old",
                "new_text": "new",
            }
        )

        self.assertIn("Success: File edited.", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello new world")

    def test_edit_file_missing_new_string_returns_friendly_validation_error(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "demo.txt"
        target.write_text("hello old world", encoding="utf-8")

        result = filesystem.edit_file_tool.invoke(
            {
                "path": "demo.txt",
                "old_text": "old",
            }
        )

        self.assertIn("ERROR[VALIDATION]", result)
        self.assertIn("new_string", result)

    def test_edit_file_path_sanitizer_strips_browser_user_agent_tail(self):
        tmp = self._workspace_tempdir()
        original_cwd = filesystem.fs_manager.cwd
        self.addCleanup(lambda: setattr(filesystem.fs_manager, "cwd", original_cwd))
        filesystem.set_working_directory(str(tmp))

        target = tmp / "parse_yandex_forecast_fixed.py"
        target.write_text("x = 1\n", encoding="utf-8")
        noisy_path = "parse_yandex_forecast_fixed.py Mozilla/5.0 AppleWebKit/537.36 Safari/537.36"

        result = filesystem.edit_file_tool.invoke(
            {
                "path": noisy_path,
                "old_string": "x = 1",
                "new_string": "x = 2",
            }
        )

        self.assertIn("Success: File edited.", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 2\n")


if __name__ == "__main__":
    unittest.main()
