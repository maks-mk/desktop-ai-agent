import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.config import AgentConfig
from core.context_builder import ContextBuilder
from core.recovery_manager import RecoveryManager
from core.runtime_prompt_policy import RuntimePromptPolicyBuilder
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
        self.assertIn("use only tools bound in this request", system_text)
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
        self.assertIn("Always respond in Russian.", joined)
        self.assertIn("After any system change", joined)
        self.assertIn("TOOL ACCESS FOR THIS REQUEST:", joined)
        self.assertIn("Execution environment: os=windows;", joined)
        self.assertIn("paths=windows.", joined)
        self.assertIn("Workspace root:", joined)
        self.assertIn("Current working directory:", joined)
        self.assertIn("Local timezone:", joined)

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

    def test_context_builder_injects_request_user_input_demo_policy_only_for_demo_requests(self):
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
        self.assertIn("REQUEST_USER_INPUT TEST POLICY:", joined)
        self.assertIn("Do exactly one request_user_input call.", joined)

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

    def test_context_builder_inserts_short_bridge_after_tool_before_user(self):
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

        self.assertEqual(len(sanitized), 3)
        self.assertIsInstance(sanitized[1], AIMessage)
        self.assertEqual(str(sanitized[1].content), "Continuing.")

    def test_context_builder_repeats_current_task_after_tool_result(self):
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

        self.assertIsInstance(context[-1], HumanMessage)
        self.assertIn("Проверь list_mistral_models.py", str(context[-1].content))

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
            user_choice_locked=True,
        )

        system_texts = [str(message.content) for message in context if isinstance(message, SystemMessage)]
        self.assertTrue(any("Do not call request_user_input again" in text for text in system_texts))

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
        self.assertIn("Recovery strategy: fix_args", text)
        self.assertIn("Prepared arguments:", text)
        self.assertNotIn("Structured issue details:", text)

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

        self.assertIn("Не удалось завершить задачу", text)
        self.assertIn("стагнац", text.lower())
        self.assertNotIn("Prepared arguments:", text)
        self.assertNotIn("Suggested next tool:", text)
        self.assertNotIn("Hint:", text)

    def test_recovery_manager_builds_soft_internal_ui_notices_by_reason(self):
        manager = RecoveryManager()

        loop_notice = manager.build_internal_ui_notice("loop_budget_exhausted_pending_tool_call")
        stagnation_notice = manager.build_internal_ui_notice("successful_tool_stagnation")
        fallback_notice = manager.build_internal_ui_notice("recovery_stagnated")

        self.assertIn("внутренний лимит", loop_notice.lower())
        self.assertIn("по кругу", stagnation_notice.lower())
        self.assertIn("пау", fallback_notice.lower())


if __name__ == "__main__":
    unittest.main()
