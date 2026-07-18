import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.config import AgentConfig
from core.context_builder import ContextBuilder
from core.recovery_manager import RecoveryManager
from core.runtime_prompt_policy import RuntimePromptContext, RuntimePromptPolicyBuilder
from core.self_correction_engine import RepairPlan
from core.tool_executor import ToolExecutor
from core.tool_issues import build_tool_issue
from core.tool_policy import ToolMetadata


class RefactorServicesTests(unittest.TestCase):
    def _make_config(self, **overrides) -> AgentConfig:
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MCP_CONFIG_PATH": Path(__file__).resolve().parents[1] / "tests" / "missing_mcp.json",
            "ENABLE_SEARCH_TOOLS": False,
            "ENABLE_SYSTEM_TOOLS": False,
            "ENABLE_PROCESS_TOOLS": False,
            "ENABLE_SHELL_TOOL": False,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def test_context_builder_uses_compact_tool_notice_for_large_catalog(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=True,
            active_tool_names=[f"tool_{i}" for i in range(8)],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_text = "\n".join(
            str(message.content) for message in context if isinstance(message, SystemMessage)
        )
        self.assertIn("Tools are available in this runtime for file, shell, web, or system access.", system_text)
        self.assertNotIn("tool_0, tool_1", system_text)

    def test_context_builder_injects_runtime_contract_from_code(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("RUNTIME CONTRACT:", joined)
        self.assertNotIn("Always respond in Russian.", joined)
        self.assertNotIn("Before using any tool or tool batch", joined)
        self.assertNotIn("After any system change", joined)
        self.assertIn("Current task: Проверь задачу", joined)
        self.assertIn("TOOLS:", joined)
        self.assertIn("TOOL INTENT REQUIREMENT:", joined)
        self.assertIn("Never send an empty assistant message when tool_calls are present.", joined)
        self.assertIn("Execution environment: os=windows;", joined)
        self.assertIn("paths=windows.", joined)
        self.assertIn("Workspace:", joined)
        self.assertNotIn("Working directory:", joined)
        self.assertIn("Local time:", joined)
        self.assertIn("date=", joined)

    def test_prompt_file_controls_default_response_language(self):
        prompt_text = (Path(__file__).resolve().parents[1] / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn("Always respond in Russian.", prompt_text)

    def test_context_builder_keeps_only_workspace_safety_overlay_for_tools(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("SAFETY POLICY: Any write, delete, move, or process-launch working directory must stay inside the active workspace.", joined)
        self.assertNotIn("Before every tool call", joined)

    def test_context_builder_requests_reasoning_summary_by_default(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        joined = "\n".join(str(message.content) for message in context if isinstance(message, SystemMessage))
        self.assertNotIn("THOUGHT VISIBILITY POLICY:", joined)
        self.assertNotIn("<think>", joined)
        self.assertNotIn("舞台上边...dr", joined)

    def test_context_builder_requests_reasoning_summary_even_when_legacy_toggle_is_false(self):
        builder = ContextBuilder(
            config=self._make_config(SHOW_MODEL_THOUGHTS=False),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Проверь задачу",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        joined = "\n".join(str(message.content) for message in context if isinstance(message, SystemMessage))
        self.assertNotIn("THOUGHT VISIBILITY POLICY:", joined)
        self.assertNotIn("<think>", joined)
        self.assertNotIn("舞台上边...dr", joined)

    def test_runtime_prompt_policy_detects_environment_once_per_builder(self):
        with mock.patch("core.runtime_prompt_policy.platform.system", return_value="Windows") as detect_os:
            builder = RuntimePromptPolicyBuilder(config=self._make_config())
            context = RuntimePromptContext(
                current_task="Проверь задачу",
                tools_available=False,
                active_tool_names=(),
            )

            builder.build_messages(context)
            builder.build_messages(context)

        self.assertEqual(detect_os.call_count, 1)

    def test_runtime_prompt_policy_updates_workspace_after_directory_change(self):
        first = Path("C:/projects/first")
        second = Path("C:/projects/second")
        with mock.patch("core.runtime_prompt_policy.Path.cwd", side_effect=[first, first, second]):
            builder = RuntimePromptPolicyBuilder(config=self._make_config())
            context = RuntimePromptContext(
                current_task="Проверь задачу",
                tools_available=False,
                active_tool_names=(),
            )

            first_contract = str(builder.build_messages(context)[0].content)
            second_contract = str(builder.build_messages(context)[0].content)

        self.assertIn(str(first.resolve()), first_contract)
        self.assertIn(str(second.resolve()), second_contract)
        self.assertNotIn(str(first.resolve()), second_contract)

    def test_runtime_prompt_policy_maps_supported_operating_systems(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Windows"),
            mock.patch.dict(
                "core.runtime_prompt_policy.os.environ",
                {"PSModulePath": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\Modules"},
                clear=True,
            ),
        ):
            self.assertEqual(builder._detect_os_family(), "windows")
            self.assertEqual(
                builder._build_execution_environment_line(),
                "Execution environment: os=windows; shell=powershell; paths=windows.",
            )

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Linux"),
            mock.patch.dict(
                "core.runtime_prompt_policy.os.environ",
                {"SHELL": "/bin/bash"},
                clear=True,
            ),
        ):
            self.assertEqual(builder._detect_os_family(), "linux")
            self.assertEqual(
                builder._build_execution_environment_line(),
                "Execution environment: os=linux; shell=bash; paths=unix.",
            )

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Darwin"),
            mock.patch.dict(
                "core.runtime_prompt_policy.os.environ",
                {"SHELL": "/bin/zsh"},
                clear=True,
            ),
        ):
            self.assertEqual(builder._detect_os_family(), "mac")
            self.assertEqual(
                builder._build_execution_environment_line(),
                "Execution environment: os=mac; shell=zsh; paths=unix.",
            )

    def test_runtime_prompt_policy_falls_back_to_reasonable_shell_defaults(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Linux"),
            mock.patch.dict("core.runtime_prompt_policy.os.environ", {}, clear=True),
        ):
            self.assertEqual(builder._detect_shell_family(os_family="linux"), "sh")

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Windows"),
            mock.patch.dict("core.runtime_prompt_policy.os.environ", {}, clear=True),
        ):
            self.assertEqual(builder._detect_shell_family(os_family="windows"), "unknown")

    def test_runtime_prompt_policy_detects_workspace_and_timezone_metadata(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())
        fake_now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone(timedelta(hours=3)))

        with (
            mock.patch("core.runtime_prompt_policy.platform.system", return_value="Linux"),
            mock.patch.dict("core.runtime_prompt_policy.os.environ", {"SHELL": "/bin/bash"}, clear=True),
            mock.patch("core.runtime_prompt_policy.Path.cwd", return_value=Path("/tmp/project")),
            mock.patch("core.runtime_prompt_policy.datetime") as datetime_mock,
        ):
            datetime_mock.now.return_value = fake_now
            environment = builder._detect_execution_environment()

        self.assertEqual(environment.workspace_root, str(Path("/tmp/project").resolve()))
        self.assertEqual(environment.current_working_directory, str(Path("/tmp/project").resolve()))
        self.assertEqual(environment.timezone_name, "UTC+03:00")
        self.assertEqual(environment.utc_offset, "UTC+03:00")

    def test_context_builder_injects_request_user_input_policy_from_code(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Нужен выбор пользователя",
            tools_available=True,
            active_tool_names=["read_file", "request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("REQUEST_USER_INPUT POLICY:", joined)
        self.assertIn("Never batch multiple request_user_input calls.", joined)
        self.assertIn("Do not use request_user_input for approvals of risky actions", joined)
        self.assertIn("Provide 2 to 5 short mutually exclusive options.", joined)
        self.assertIn("Make the request_user_input tool call by itself.", joined)

    def test_context_builder_does_not_inject_request_user_input_demo_policy(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [],
            None,
            summary="",
            current_task="Сделай тест request_user_input для примера",
            tools_available=True,
            active_tool_names=["request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertNotIn("REQUEST_USER_INPUT TEST POLICY:", joined)

        regular_context = builder.build(
            [],
            None,
            summary="",
            current_task="Нужно спросить пользователя, в какую папку сохранить файл",
            tools_available=True,
            active_tool_names=["request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )
        regular_texts = [str(message.content) for message in regular_context if isinstance(message, SystemMessage)]
        self.assertFalse(any("REQUEST_USER_INPUT TEST POLICY:" in text for text in regular_texts))

    def test_context_builder_preserves_tool_then_user_sequence_without_bridge_messages(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                ToolMessage(content="ok", tool_call_id="tool-1", name="read_file"),
                HumanMessage(content="Продолжай"),
            ]
        )

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[0], ToolMessage)
        self.assertIsInstance(sanitized[1], HumanMessage)

    def test_context_builder_does_not_repeat_current_task_after_tool_result(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [ToolMessage(content="ok", tool_call_id="tool-1", name="read_file")],
            None,
            summary="",
            current_task="Проверь list_mistral_models.py",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        self.assertIsInstance(context[-1], ToolMessage)
        self.assertEqual(str(context[-1].content), "ok")

    def test_context_builder_locks_user_choice_after_choice_was_collected(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        context = builder.build(
            [ToolMessage(content="Опция C", tool_call_id="tool-1", name="request_user_input")],
            None,
            summary="",
            current_task="Покажи результат теста",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        self.assertTrue(any("Do not call request_user_input again" in text for text in system_texts))
        self.assertTrue(any("latest request_user_input ToolMessage" in text for text in system_texts))

    def test_context_builder_strips_historical_images_for_text_only_model_and_keeps_text(self):
        builder = ContextBuilder(
            config=self._make_config(),
            model_capabilities={"image_input_supported": False},
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Опиши предыдущее изображение"},
                        {"type": "image", "path": "C:/tmp/demo.png", "mime_type": "image/png"},
                    ]
                )
            ]
        )

        self.assertEqual(len(sanitized), 1)
        self.assertIsInstance(sanitized[0], HumanMessage)
        self.assertEqual(sanitized[0].content, "Опиши предыдущее изображение")

    def test_context_builder_replaces_image_only_history_for_text_only_model_with_placeholder(self):
        builder = ContextBuilder(
            config=self._make_config(),
            model_capabilities={"image_input_supported": False},
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                HumanMessage(
                    content=[
                        {"type": "image", "path": "C:/tmp/demo.png", "mime_type": "image/png"},
                    ]
                )
            ]
        )

        self.assertEqual(len(sanitized), 1)
        self.assertIsInstance(sanitized[0], HumanMessage)
        self.assertIn("Previous image input omitted", sanitized[0].content)

    def test_context_builder_stringifies_openai_assistant_content_lists(self):
        builder = ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        sanitized = builder.sanitize_messages(
            [
                HumanMessage(content="Проверь историю"),
                AIMessage(content=["Первый фрагмент. ", "Второй фрагмент."]),
                ToolMessage(content=[{"type": "text", "text": "ok"}], tool_call_id="tool-1", name="read_file"),
            ]
        )

        self.assertEqual(sanitized[1].content, "Первый фрагмент. Второй фрагмент.")
        self.assertEqual(sanitized[2].content, "ok")

    def test_tool_executor_readonly_error_stays_visible_to_agent_without_issue(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        outcome = executor.handle_result(
            state={"run_id": "run"},
            current_turn_id=1,
            tool_name="read_file",
            tool_args={"path": "README.md"},
            tool_call_id="call-1",
            content="ERROR[EXECUTION]: boom",
            apply_validation=False,
            had_error=True,
        )

        self.assertTrue(outcome.had_error)
        self.assertIsNone(outcome.issue)
        self.assertEqual(outcome.tool_message.status, "error")

    def test_tool_executor_treats_plain_error_like_file_contents_as_success(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, read_only=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        outcome = executor.handle_result(
            state={"run_id": "run"},
            current_turn_id=1,
            tool_name="read_file",
            tool_args={"path": "app.log"},
            tool_call_id="call-plain-error-text",
            content="Error: connection reset by peer\nTraceback follows below as part of the log",
            apply_validation=False,
        )

        self.assertFalse(outcome.had_error)
        self.assertIsNone(outcome.issue)
        self.assertEqual(outcome.tool_message.status, "success")

    def test_tool_executor_promotes_validation_failure_to_error_status(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        with mock.patch("core.tool_executor.validate", return_value="ERROR[VALIDATION]: file was not updated"):
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="edit_file",
                tool_args={"path": "demo.txt"},
                tool_call_id="call-validation",
                content="Success: File edited.",
                apply_validation=True,
            )

        self.assertTrue(outcome.had_error)
        self.assertEqual(outcome.tool_message.status, "error")
        self.assertFalse(outcome.parsed_result.ok)
        self.assertIn("Tool output:", outcome.content)

    def test_tool_executor_isolates_nested_tool_args_from_validation_and_message_state(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )
        tool_args = {
            "path": "demo.txt",
            "options": {"mode": "append"},
            "lines": ["first"],
        }

        def mutate_validation_payload(_content, context):
            context["args"]["options"]["mode"] = "overwrite"
            context["args"]["lines"].append("second")
            return None

        with mock.patch("core.tool_executor.validate", side_effect=mutate_validation_payload):
            outcome = executor.handle_result(
                state={"run_id": "run"},
                current_turn_id=1,
                tool_name="edit_file",
                tool_args=tool_args,
                tool_call_id="call-nested-copy",
                content="Success: File edited.",
                apply_validation=True,
            )

        self.assertEqual(tool_args["options"]["mode"], "append")
        self.assertEqual(tool_args["lines"], ["first"])
        self.assertEqual(
            outcome.tool_message.additional_kwargs["tool_args"],
            {"path": "demo.txt", "options": {"mode": "append"}, "lines": ["first"]},
        )

    def test_tool_executor_merges_multiple_issues(self):
        executor = ToolExecutor(
            config=self._make_config(),
            metadata_for_tool=lambda name: ToolMetadata(name=name, mutating=True),
            log_run_event=lambda *_args, **_kwargs: None,
            workspace_boundary_violated=lambda *_args, **_kwargs: False,
        )

        merged = executor.merge_issues(
            [
                build_tool_issue(
                    current_turn_id=2,
                    kind="tool_error",
                    summary="Missing path",
                    tool_names=["edit_file"],
                    tool_args={"path": "a.txt"},
                    source="tools",
                    error_type="VALIDATION",
                    fingerprint="fp-1",
                    details={"missing_required_fields": ["path"]},
                    progress_fingerprint="fp-1",
                ),
                build_tool_issue(
                    current_turn_id=2,
                    kind="tool_error",
                    summary="Loop detected",
                    tool_names=["edit_file"],
                    tool_args={"path": "a.txt"},
                    source="tools",
                    error_type="LOOP_DETECTED",
                    fingerprint="fp-2",
                    details={"loop_detected": True},
                    progress_fingerprint="fp-2",
                ),
            ],
            current_turn_id=2,
        )

        self.assertIsNotNone(merged)
        self.assertIn("Missing path", merged["summary"])
        self.assertIn("edit_file", merged["tool_names"])
        self.assertTrue(merged["details"]["loop_detected"])

    def test_build_tool_issue_copies_nested_payloads(self):
        tool_args = {"path": "demo.txt", "options": {"mode": "append"}}
        details = {"missing_required_fields": ["path"], "nested": {"retry": ["read_file"]}}

        issue = build_tool_issue(
            current_turn_id=2,
            kind="tool_error",
            summary="Missing path",
            tool_names=["edit_file"],
            tool_args=tool_args,
            source="tools",
            error_type="VALIDATION",
            fingerprint="fp-1",
            details=details,
            progress_fingerprint="fp-1",
        )

        tool_args["options"]["mode"] = "overwrite"
        details["nested"]["retry"].append("find_file")

        self.assertEqual(issue["tool_args"]["options"]["mode"], "append")
        self.assertEqual(issue["details"]["nested"]["retry"], ["read_file"])

    def test_recovery_manager_builds_compact_recovery_message(self):
        manager = RecoveryManager()
        message = manager.build_recovery_system_message(
            {
                "active_issue": {"summary": "Port must be integer"},
                "active_strategy": {
                    "strategy": "normalize_args",
                    "strategy_kind": "fix_args",
                    "llm_guidance": "Retry with normalized arguments.",
                    "suggested_tool_name": "find_process_by_port",
                    "patched_args": {"port": 8080, "extra": "x" * 200},
                    "notes": "Normalize the port before retry.",
                },
            }
        )

        self.assertIsNotNone(message)
        text = str(message.content)
        self.assertIn("RECOVERY MODE:", text)
        self.assertIn("Recovery strategy: fix_args", text)
        self.assertIn("Prepared arguments:", text)
        self.assertNotIn("Structured issue details:", text)
        self.assertNotIn("Do not repeat the exact same failing call unchanged.", text)

    def test_recovery_manager_build_recovery_strategy_copies_nested_payloads(self):
        manager = RecoveryManager()
        repair_plan = RepairPlan(
            strategy="normalize_args",
            reason="validation",
            fingerprint="fp-1",
            tool_name="edit_file",
            suggested_tool_name="edit_file",
            original_args={"path": "demo.txt"},
            patched_args={"path": "demo.txt", "edits": [{"old": "a", "new": "b"}]},
            notes="Retry with normalized arguments.",
        )
        open_tool_issue = {
            "summary": "Need exact old text",
            "details": {"candidates": [{"path": "demo.txt"}]},
        }

        strategy = manager.build_recovery_strategy(
            repair_plan=repair_plan,
            open_tool_issue=open_tool_issue,
            current_task="Обнови файл",
            strategy_id="strategy-1",
        )

        repair_plan.patched_args["edits"][0]["new"] = "c"
        open_tool_issue["details"]["candidates"][0]["path"] = "changed.txt"

        self.assertEqual(strategy["patched_args"]["edits"][0]["new"], "b")
        self.assertEqual(strategy["issue_details"]["candidates"][0]["path"], "demo.txt")

    def test_recovery_manager_handoff_text_hides_internal_recovery_hints(self):
        manager = RecoveryManager()
        text = manager.build_tool_issue_handoff_text(
            {
                "kind": "tool_error",
                "summary": "Command failed with Exit Code 1",
                "tool_names": ["cli_exec"],
                "details": {},
            },
            repair_plan=RepairPlan(
                strategy="llm_replan",
                reason="recovery_stagnated",
                fingerprint="fp-1",
                tool_name="cli_exec",
                suggested_tool_name="cli_exec",
                original_args={"command": "rm bad.txt"},
                patched_args={"command": "rm bad.txt"},
                notes="No deterministic auto-repair available.",
            ),
        )

        self.assertIn("Unable to complete the task", text)
        self.assertIn("stagnation", text.lower())
        self.assertNotIn("Prepared arguments:", text)
        self.assertNotIn("Suggested next tool:", text)
        self.assertNotIn("Hint:", text)

    def test_recovery_manager_builds_soft_internal_ui_notices_by_reason(self):
        manager = RecoveryManager()

        loop_notice = manager.build_internal_ui_notice("loop_budget_exhausted_pending_tool_call")
        stagnation_notice = manager.build_internal_ui_notice("successful_tool_stagnation")
        fallback_notice = manager.build_internal_ui_notice("recovery_stagnated")

        self.assertIn("internal retry limit", loop_notice.lower())
        self.assertIn("loop", stagnation_notice.lower())
        self.assertIn("paused", fallback_notice.lower())

    def test_recovery_manager_resets_retry_budget_when_issue_fingerprint_changes(self):
        manager = RecoveryManager()
        recovery_state = manager.get_recovery_state(
            {
                "turn_id": 1,
                "progress_markers": ["fp-old"],
                "attempts_by_strategy": {"fp-old::llm_replan": 2},
                "llm_replan_attempted_for": ["fp-old"],
            },
            current_turn_id=1,
        )
        issue = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Missing required field(s): path.",
            "tool_names": ["edit_file"],
            "tool_args": {"old_string": "a", "new_string": "b"},
            "source": "tools",
            "error_type": "VALIDATION",
            "fingerprint": "fp-new",
            "progress_fingerprint": "fp-new",
            "details": {"missing_required_fields": ["path"]},
        }

        result = manager.plan_recovery(
            state={
                "self_correction_retry_count": 5,
                "self_correction_fingerprint_history": ["fp-old"],
            },
            messages=[HumanMessage(content="Исправь файл")],
            current_task="Исправь файл",
            current_turn_id=1,
            open_tool_issue=issue,
            recovery_state=recovery_state,
            last_ai=None,
            last_message=None,
            step_count=0,
            max_loops=50,
            hard_loop_ceiling=8,
            max_auto_repairs=8,
            successful_tool_stagnation_limit=3,
        )

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["completion_reason"], "recover_refresh_context")
        self.assertEqual(result["self_correction_retry_count"], 1)
        self.assertEqual(result["recovery_state"]["progress_markers"][-1], "fp-new")
        self.assertEqual(result["recovery_state"]["attempts_by_strategy"]["fp-new::refresh_context"], 1)

    def test_recovery_manager_allows_multiple_llm_replans_for_same_issue(self):
        manager = RecoveryManager()
        recovery_state = manager.get_recovery_state(
            {
                "turn_id": 1,
                "progress_markers": ["fp-protocol"],
                "attempts_by_strategy": {"fp-protocol::llm_replan": 2},
                "llm_replan_attempted_for": ["fp-protocol"],
            },
            current_turn_id=1,
        )
        issue = {
            "turn_id": 1,
            "kind": "protocol_error",
            "summary": "Malformed tool payload.",
            "tool_names": ["read_file"],
            "tool_args": {"path": "README.md"},
            "source": "agent",
            "error_type": "VALIDATION",
            "fingerprint": "fp-protocol",
            "progress_fingerprint": "fp-protocol",
            "details": {"protocol_reason": "tool_protocol_error"},
        }

        result = manager.plan_recovery(
            state={
                "self_correction_retry_count": 3,
                "self_correction_fingerprint_history": ["fp-protocol"],
            },
            messages=[HumanMessage(content="Прочитай README.md")],
            current_task="Прочитай README.md",
            current_turn_id=1,
            open_tool_issue=issue,
            recovery_state=recovery_state,
            last_ai=None,
            last_message=None,
            step_count=0,
            max_loops=50,
            hard_loop_ceiling=8,
            max_auto_repairs=8,
            successful_tool_stagnation_limit=3,
        )

        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["completion_reason"], "recover_llm_replan")
        self.assertEqual(result["self_correction_retry_count"], 4)
        self.assertEqual(result["recovery_state"]["active_strategy"]["strategy"], "llm_replan")
        self.assertEqual(result["recovery_state"]["attempts_by_strategy"]["fp-protocol::llm_replan"], 3)

    def test_sanitize_strips_openai_reasoning_content_blocks_for_gemini(self):
        """OpenAI Responses API reasoning blocks (type=reasoning, summary=[...])
        must be stripped so they don't cause KeyError in langchain-google-genai."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="gemini"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking..."}], "id": "rs_001"},
                {"type": "text", "text": "Ответ модели"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([HumanMessage(content="Вопрос"), ai_msg])

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[1], AIMessage)
        # reasoning block removed, text block preserved
        self.assertEqual(len(sanitized[1].content), 1)
        self.assertEqual(sanitized[1].content[0]["type"], "text")
        self.assertEqual(sanitized[1].content[0]["text"], "Ответ модели")

    def test_sanitize_strips_openai_reasoning_from_additional_kwargs_for_gemini(self):
        """When output_version='v0', langchain-openai moves reasoning into
        additional_kwargs['reasoning'] as a dict with 'summary'. This must be
        stripped for non-OpenAI providers."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="gemini"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[{"type": "text", "text": "Ответ"}],
            additional_kwargs={
                "reasoning": {"summary": [{"type": "summary_text", "text": "Hidden thought"}]},
            },
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        self.assertNotIn("reasoning", sanitized[0].additional_kwargs)

    def test_sanitize_preserves_gemini_native_reasoning_blocks(self):
        """Gemini's own reasoning blocks have a 'reasoning' string key and must
        NOT be stripped — they are needed for multi-turn tool-calling."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="gemini"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "My thought process", "extras": {"signature": "abc123"}},
                {"type": "text", "text": "Ответ"},
            ],
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        # Both blocks preserved — Gemini reasoning has 'reasoning' key
        self.assertEqual(len(sanitized[0].content), 2)
        self.assertEqual(sanitized[0].content[0]["type"], "reasoning")
        self.assertIn("reasoning", sanitized[0].content[0])

    def test_sanitize_strips_reasoning_blocks_for_openai_too(self):
        """Reasoning is ephemeral — strip from history even for OpenAI to keep
        cross-provider replay clean and avoid stale reasoning accumulation."""
        builder = ContextBuilder(
            config=self._make_config(PROVIDER="openai"),
            prompt_loader=lambda: "Base prompt {{current_date}}",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=__import__("re").compile(r"^[A-Za-z0-9]{9}$"),
        )

        ai_msg = AIMessage(
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking..."}]},
                {"type": "text", "text": "Ответ"},
            ],
            additional_kwargs={"reasoning": {"summary": [{"text": "thought"}]}},
            tool_calls=[],
        )

        sanitized = builder.sanitize_messages([ai_msg])

        self.assertEqual(len(sanitized), 1)
        # OpenAI provider stringifies content lists — reasoning block stripped,
        # text block becomes a plain string.
        self.assertEqual(sanitized[0].content, "Ответ")
        self.assertNotIn("reasoning", sanitized[0].additional_kwargs)


if __name__ == "__main__":
    unittest.main()
