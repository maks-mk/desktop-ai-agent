import asyncio
import shutil
import unittest
from pathlib import Path
from uuid import uuid4
from unittest import mock

import httpx
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

from core.text_utils import prepare_markdown_for_render
from ui.streaming import StreamProcessor
from tools import filesystem, local_shell
from tools.filesystem import FilesystemManager, _DOWNLOAD_HEADERS, _format_download_http_error


class StreamAndFilesystemTests(unittest.TestCase):
    class _FakeReader:
        def __init__(self, chunks: list[bytes]):
            self._chunks = list(chunks)

        async def read(self, _size: int) -> bytes:
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    class _FakeProcess:
        def __init__(
            self,
            stdout_chunks: list[bytes],
            stderr_chunks: list[bytes],
            returncode: int = 0,
            wait_exception: Exception | None = None,
            pid: int | None = None,
        ):
            self.stdout = StreamAndFilesystemTests._FakeReader(stdout_chunks)
            self.stderr = StreamAndFilesystemTests._FakeReader(stderr_chunks)
            self.returncode = returncode
            self.wait_exception = wait_exception
            self.killed = False
            self.pid = pid

        async def wait(self) -> int:
            if self.wait_exception is not None:
                raise self.wait_exception
            return self.returncode

        def kill(self) -> None:
            self.killed = True

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

    def test_prepare_markdown_normalizes_simple_latex_symbols(self):
        source = (
            'При загрузке страницы звук не инициализируется $\\rightarrow$ браузер не ругается.\n'
            'При нажатии на «СТАРТ» $\\Rightarrow$ звук активируется.'
        )

        rendered = prepare_markdown_for_render(source)

        self.assertIn("→", rendered)
        self.assertIn("⇒", rendered)
        self.assertNotIn("$\\rightarrow$", rendered)
        self.assertNotIn("$\\Rightarrow$", rendered)

    def test_prepare_markdown_keeps_latex_symbols_literal_inside_code(self):
        source = 'Текст `$\\\\rightarrow$` и блок:\n```text\n$\\\\Rightarrow$\n```'

        rendered = prepare_markdown_for_render(source)

        self.assertIn("`$\\\\rightarrow$`", rendered)
        self.assertIn("```text\n$\\\\Rightarrow$\n```", rendered)

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

    def test_stream_processor_treats_free_port_result_as_success(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-port-check"] = {"name": "find_process_by_port", "args": {"port": 8000}}

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-port-check",
                name="find_process_by_port",
                content="No process found listening on port 8000.",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0]["is_error"])
        self.assertIn("No process found listening on port 8000.", finished[0]["content"])

    def test_stream_processor_does_not_inject_preface_before_tool_when_model_text_is_empty(self):
        events = []
        processor = StreamProcessor(events.append)
        processor._remember_tool_call({"id": "call-preface", "name": "read_file", "args": {"path": "demo.txt"}})

        processor._emit_tool_started({"id": "call-preface", "name": "read_file", "args": {"path": "demo.txt"}})

        event_types = [event.type for event in events]
        self.assertIn("tool_started", event_types)
        self.assertNotIn("assistant_delta", event_types)

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
        self.assertEqual(notice_events[0].payload["kind"], "auto_summary")

    def test_stream_processor_stats_use_numeric_input_fallback_when_usage_missing(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "messages",
                "data": (AIMessage(content="Готово"), {"langgraph_node": "agent"}),
            }

        result = asyncio.run(processor.process_stream(_stream()))
        self.assertIsNotNone(result.stats)
        assert result.stats is not None
        self.assertIn("↓ 0", result.stats)
        self.assertNotIn("↓ ?", result.stats)

    def test_stream_processor_reads_token_usage_from_update_payload(self):
        processor = StreamProcessor()
        processor._handle_updates({"agent": {"token_usage": {"prompt_tokens": 321, "completion_tokens": 8}}})
        stats = processor.tracker.render(0.1)
        self.assertIn("↓ 321", stats)
        self.assertIn("↑ 8", stats)

    def test_stream_processor_accumulates_total_elapsed_from_previous_segments(self):
        events = []
        perf_values = iter([100.0, 100.0])

        def _fake_perf_counter():
            try:
                return next(perf_values)
            except StopIteration:
                return 102.7

        processor = None
        with mock.patch("ui.streaming.time.perf_counter", side_effect=_fake_perf_counter):
            processor = StreamProcessor(events.append, base_elapsed_seconds=87.0)

            async def _stream():
                yield {
                    "type": "messages",
                    "data": (AIMessage(content="Готово"), {"langgraph_node": "agent"}),
                }

            result = asyncio.run(processor.process_stream(_stream()))

        self.assertIsNotNone(result.stats)
        self.assertAlmostEqual(result.elapsed_seconds, 89.7, places=1)
        assert result.stats is not None
        self.assertIn("89.7s", result.stats)

    def test_stream_processor_renders_stability_guard_handoff_messages(self):
        events = []
        processor = StreamProcessor(events.append)
        processor._handle_updates(
            {
                "stability_guard": {
                    "messages": [AIMessage(content="Автовыполнение остановлено. Уточните следующий шаг.")],
                }
            }
        )

        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertIn("Автовыполнение остановлено", deltas[0]["full_text"])

    def test_stream_processor_hides_internal_handoff_messages_from_ui_events(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content="internal handoff",
                additional_kwargs={
                    "agent_internal": {
                        "kind": "tool_issue_handoff",
                        "visible_in_ui": False,
                        "ui_notice": "Нужен новый запрос.",
                    }
                },
            )
        )

        self.assertEqual(events, [])
        self.assertEqual(processor.full_text, "")
        self.assertEqual(processor.clean_full, "")

    def test_stream_processor_marks_stability_guard_as_self_correcting(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_messages((AIMessage(content=""), {"langgraph_node": "stability_guard"}))

        statuses = [event.payload for event in events if event.type == "status_changed"]
        self.assertTrue(statuses)
        self.assertEqual(statuses[-1]["node"], "stability_guard")
        self.assertEqual(statuses[-1]["label"], "Self-correcting")

    def test_stream_processor_does_not_parse_choice_requests_from_plain_text(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._handle_agent_message(
            AIMessage(
                content=(
                    "Нужно выбрать один из вариантов.\n\n"
                    "Как продолжаем?\n"
                    "- direct_api: тестируем только API\n"
                    "- keep_mcp: оставляем MCP\n"
                )
            )
        )

        choice_events = [event for event in events if event.type == "user_choice_requested"]
        self.assertEqual(len(choice_events), 0)
        deltas = [event.payload for event in events if event.type == "assistant_delta"]
        self.assertEqual(len(deltas), 1)
        self.assertIn("Как продолжаем?", deltas[0]["full_text"])

    def test_stream_processor_returns_user_choice_interrupt_from_updates(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "updates",
                "data": {
                    "__interrupt__": [
                        {
                            "kind": "user_choice",
                            "question": "Введите ключ API или выберите другой вариант:",
                            "options": [
                                "Ввести ключ API",
                                "Пропустить проверку и вернуть скрипт",
                                "Завершить проверку",
                            ],
                            "recommended": "Ввести ключ API",
                        }
                    ]
                },
            }

        result = asyncio.run(processor.process_stream(_stream()))
        self.assertIsNotNone(result.interrupt)
        assert result.interrupt is not None
        self.assertEqual(result.interrupt["kind"], "user_choice")
        self.assertEqual(result.interrupt["question"], "Введите ключ API или выберите другой вариант:")
        self.assertEqual(
            result.interrupt["options"],
            [
                "Ввести ключ API",
                "Пропустить проверку и вернуть скрипт",
                "Завершить проверку",
            ],
        )

    def test_stream_processor_uses_bounded_memory_caps(self):
        processor = StreamProcessor(
            emit_event=None,
            text_max_chars=24,
            events_max=3,
            tool_buffer_max=2,
        )

        processor._handle_agent_message(AIMessage(content="0123456789" * 6))
        self.assertLessEqual(len(processor.full_text), 24)
        self.assertLessEqual(len(processor.clean_full), 24)

        for idx in range(6):
            processor._emit(f"evt_{idx}", {"i": idx})
        self.assertEqual(len(processor.events), 3)
        self.assertEqual(processor.events[-1].type, "evt_5")

        for idx in range(4):
            processor._remember_tool_call(
                {"id": f"tool-{idx}", "name": "read_file", "args": {"path": f"{idx}.txt"}}
            )
        self.assertLessEqual(len(processor.tool_buffer), 2)
        self.assertLessEqual(len(processor.tool_start_times), 2)

    def test_stream_processor_emit_trims_oldest_events_directly(self):
        processor = StreamProcessor(events_max=2)
        processor._emit("evt_1", {"i": 1})
        processor._emit("evt_2", {"i": 2})
        processor._emit("evt_3", {"i": 3})

        self.assertEqual([event.type for event in processor.events], ["evt_2", "evt_3"])

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

    def test_stream_processor_emits_tool_started_refresh_when_args_arrive_late(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-refresh", "name": "edit_file", "args": {}})
        processor._emit_tool_started({"id": "call-refresh", "name": "edit_file", "args": {}})
        processor._remember_tool_call(
            {
                "id": "call-refresh",
                "name": "edit_file",
                "args": {"path": "late.txt", "old_string": "a", "new_string": "b"},
            }
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertGreaterEqual(len(started), 2)
        self.assertEqual(started[-1]["args"]["path"], "late.txt")
        self.assertTrue(started[-1].get("refresh"))

    def test_stream_processor_tool_display_flattens_multiline_command(self):
        events = []
        processor = StreamProcessor(events.append)
        multiline_command = "python - <<'PY'\nimport sys\nprint(sys.version)\nPY"
        processor._remember_tool_call(
            {
                "id": "call-cmd",
                "name": "cli_exec",
                "args": {"command": multiline_command},
            }
        )

        processor._emit_tool_started(
            {
                "id": "call-cmd",
                "name": "cli_exec",
                "args": {"command": multiline_command},
            }
        )

        started = [event.payload for event in events if event.type == "tool_started"]
        self.assertEqual(len(started), 1)
        self.assertIn("cli_exec", started[0]["display"])
        self.assertNotIn("\n", started[0]["display"])

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

    def test_stream_processor_prefers_tool_execution_duration_from_metadata(self):
        events = []
        processor = StreamProcessor(events.append)
        processor.tool_buffer["call-late-duration"] = {
            "name": "cli_exec",
            "args": {"command": "ping -n 4 google.com"},
        }

        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-late-duration",
                name="cli_exec",
                content="done",
                additional_kwargs={
                    "tool_args": {"command": "ping -n 4 google.com"},
                    "tool_duration_seconds": 1.75,
                },
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["duration"], 1.75)

    def test_stream_processor_deduplicates_duplicate_tool_results(self):
        events = []
        processor = StreamProcessor(events.append)

        processor._remember_tool_call({"id": "call-dir", "name": "list_directory", "args": {"path": "."}})
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-dir",
                name="list_directory",
                content="Directory '.':\n[FILE] demo.py",
            )
        )
        processor._handle_tool_result(
            ToolMessage(
                tool_call_id="call-dir",
                name="list_directory",
                content="Directory '.':\n[FILE] demo.py",
            )
        )

        finished = [event.payload for event in events if event.type == "tool_finished"]
        missing = [event for event in events if event.type == "tool_args_missing"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["args"], {"path": "."})
        self.assertEqual(missing, [])

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

    def test_search_in_file_matches_plain_literal_substrings(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        target = tmp / "script.js"
        target.write_text(
            'function playSound() {}\nresumeBtn.addEventListener("click", playSound)\n',
            encoding="utf-8",
        )

        result = manager.search_in_file("script.js", "resumeBtn.addEventListener")

        self.assertIn("Found 1 match(es)", result)
        self.assertIn('resumeBtn.addEventListener("click", playSound)', result)

    def test_search_in_file_accepts_regex_escaped_literals_in_plain_mode(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        target = tmp / "script.js"
        target.write_text(
            'function playSound() {}\nresumeBtn.addEventListener("click", playSound)\n',
            encoding="utf-8",
        )

        result = manager.search_in_file("script.js", r"playSound\(")
        dotted = manager.search_in_file("script.js", r"resumeBtn\.addEventListener")

        self.assertIn("Found 1 match(es)", result)
        self.assertIn("function playSound() {}", result)
        self.assertIn("Found 1 match(es)", dotted)
        self.assertIn('resumeBtn.addEventListener("click", playSound)', dotted)

    def test_search_in_directory_accepts_regex_escaped_literals_in_plain_mode(self):
        tmp = self._workspace_tempdir()
        manager = FilesystemManager(root_dir=tmp, virtual_mode=True)
        nested = tmp / "src"
        nested.mkdir()
        (nested / "script.js").write_text(
            'function playSound() {}\nresumeBtn.addEventListener("click", playSound)\n',
            encoding="utf-8",
        )

        result = manager.search_in_directory(".", r"resumeBtn\.addEventListener")

        self.assertIn("Found 1 match(es)", result)
        self.assertIn('script.js:2  resumeBtn.addEventListener("click", playSound)', result)

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

    def test_cli_exec_stream_emits_live_chunks_with_tool_id(self):
        process = self._FakeProcess(
            stdout_chunks=[b"line-1\n", b"line-2\n"],
            stderr_chunks=[b"warn-1\n"],
            returncode=0,
        )
        live_events: list[dict[str, str]] = []
        self.addCleanup(lambda: local_shell.set_cli_output_emitter(None))

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            local_shell.set_cli_output_emitter(live_events.append)
            with local_shell.cli_output_context("call-cli-1"):
                result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo"}))

        self.assertIn("line-1", result)
        self.assertIn("line-2", result)
        self.assertIn("[stderr]", result)
        self.assertTrue(live_events)
        self.assertTrue(all(item.get("tool_id") == "call-cli-1" for item in live_events))
        self.assertTrue(any(item.get("stream") == "stdout" for item in live_events))
        self.assertTrue(any(item.get("stream") == "stderr" for item in live_events))

    def test_cli_exec_uses_non_interactive_stdin_and_npm_env(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_kwargs: dict[str, object] = {}

        async def _fake_create_subprocess(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "npm --version"}))

        self.assertIn("ok", result)
        self.assertEqual(captured_kwargs.get("stdin"), asyncio.subprocess.DEVNULL)
        env = dict(captured_kwargs.get("env") or {})
        self.assertEqual(env.get("CI"), "1")
        self.assertEqual(env.get("npm_config_yes"), "true")

    def test_cli_exec_detects_interactive_prompt_and_aborts(self):
        process = self._FakeProcess(
            stdout_chunks=[b"npx@10.2.2\n", b"Ok to proceed? (y)\n"],
            stderr_chunks=[],
            returncode=0,
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "npm install demo-package"}))

        self.assertIn("Interactive prompt detected", result)
        self.assertIn("npm/npx", result)
        self.assertTrue(process.killed)

    def test_cli_exec_stream_preserves_error_result_on_non_zero_exit(self):
        process = self._FakeProcess(
            stdout_chunks=[b"partial-out\n"],
            stderr_chunks=[b"fatal-err\n"],
            returncode=2,
        )
        live_events: list[dict[str, str]] = []
        self.addCleanup(lambda: local_shell.set_cli_output_emitter(None))

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            local_shell.set_cli_output_emitter(live_events.append)
            with local_shell.cli_output_context("call-cli-2"):
                result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "demo --fail"}))

        self.assertIn("Exit Code 2", result)
        self.assertIn("partial-out", result)
        self.assertIn("fatal-err", result)
        self.assertGreaterEqual(len(live_events), 2)

    def test_cli_exec_rejects_foreground_service_commands(self):
        result = asyncio.run(local_shell.cli_exec.ainvoke({"command": "npm exec -- npx http-server -p 8080"}))

        self.assertIn("Foreground service/server commands are not supported", result)
        self.assertIn("run_background_process", result)

    def test_cli_exec_converts_python_heredoc_to_powershell_on_windows(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            if args:
                captured_argv.append(tuple(str(part) for part in args))
            return process

        heredoc_command = "python - <<'PY'\nprint('hello')\nPY"
        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": heredoc_command}))

        self.assertIn("ok", result)
        self.assertTrue(captured_argv)
        self.assertEqual(captured_argv[0][1:3], ("-NoProfile", "-Command"))
        self.assertIn("@'", captured_argv[0][3])
        self.assertIn("'@ | python -", captured_argv[0][3])

    def test_cli_exec_unwraps_nested_powershell_wrapper_on_windows(self):
        process = self._FakeProcess(
            stdout_chunks=[b"ok\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            if args:
                captured_argv.append(tuple(str(part) for part in args))
            return process

        nested = 'powershell -Command "try { $r = 1; Write-Output $r } catch { Write-Output $_ }"'
        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": nested}))

        self.assertIn("ok", result)
        self.assertTrue(captured_argv)
        self.assertEqual(captured_argv[0][1:3], ("-NoProfile", "-Command"))
        self.assertEqual(captured_argv[0][3], "try { $r = 1; Write-Output $r } catch { Write-Output $_ }")

    def test_cli_exec_rewrites_posix_null_device_on_windows(self):
        process = self._FakeProcess(
            stdout_chunks=[b"200\n"],
            stderr_chunks=[],
            returncode=0,
        )
        captured_argv: list[tuple[str, ...]] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            if args:
                captured_argv.append(tuple(str(part) for part in args))
            return process

        command = 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8000'
        with (
            mock.patch.object(local_shell.os, "name", "nt"),
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess_exec),
        ):
            result = asyncio.run(local_shell.cli_exec.ainvoke({"command": command}))

        self.assertIn("200", result)
        self.assertTrue(captured_argv)
        normalized_command = captured_argv[0][3]
        self.assertIn("-o NUL", normalized_command)
        self.assertNotIn("/dev/null", normalized_command)

    def test_cli_exec_cancellation_terminates_running_process(self):
        process = self._FakeProcess(
            stdout_chunks=[],
            stderr_chunks=[],
            wait_exception=asyncio.CancelledError(),
        )

        async def _fake_create_subprocess(*_args, **_kwargs):
            return process

        with (
            mock.patch.object(local_shell.asyncio, "create_subprocess_exec", side_effect=_fake_create_subprocess),
            mock.patch.object(local_shell.asyncio, "create_subprocess_shell", side_effect=_fake_create_subprocess),
        ):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(local_shell.cli_exec.ainvoke({"command": "Get-Process | Select-Object Id"}))

        self.assertTrue(process.killed)

    def test_stream_processor_emits_interrupted_tool_finish_on_cancel(self):
        events = []
        processor = StreamProcessor(events.append)

        async def _stream():
            yield {
                "type": "updates",
                "data": {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[{"id": "tc-cancel", "name": "cli_exec", "args": {"command": "echo 1"}}],
                            )
                        ]
                    }
                },
            }
            raise asyncio.CancelledError()

        result = asyncio.run(processor.process_stream(_stream()))

        self.assertTrue(result.cancelled)
        self.assertEqual(len(result.cancelled_tools), 1)
        finished = [event.payload for event in events if event.type == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertTrue(finished[0]["interrupted"])
        self.assertIn("Execution interrupted", finished[0]["content"])


if __name__ == "__main__":
    unittest.main()
