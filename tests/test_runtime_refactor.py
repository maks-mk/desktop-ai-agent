import json
import os
import shutil
import sqlite3
import unittest
from unittest import mock
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent import create_agent_workflow
from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core import gui_runtime
from core.gui_runtime import (
    append_project_label,
    build_transcript_payload,
    build_user_choice_payload,
    generate_chat_title,
    short_project_label,
)
from core.model_profiles import ModelProfileStore
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.session_store import SessionSnapshot, SessionStore
from core.stream_processor import StreamProcessor
from core.tool_policy import ToolMetadata
from tools import process_tools
from tools.tool_registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(content="STATUS: FINISHED\nREASON: fallback\nNEXT_STEP: NONE\nCONTROL: FINISH_TURN")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        last_visible = next((message for message in reversed(context) if message.type != "system"), None)
        if isinstance(last_visible, AIMessage):
            raise AssertionError("provider-unsafe assistant-last context")
        return await super().ainvoke(context)


class OpenAIContentSafeFakeLLM(FakeLLM):
    async def ainvoke(self, context):
        for message in context:
            if isinstance(message, AIMessage) and not isinstance(message.content, str):
                raise AssertionError(f"assistant content must be string for OpenAI chat history, got {type(message.content).__name__}")
        return await super().ainvoke(context)


class FakeBindableLLM(FakeLLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.bound_tool_name_batches = []

    def bind_tools(self, tools):
        self.bound_tool_name_batches.append([tool.name for tool in tools])
        return self


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.description = f"Fake tool {name}"
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if callable(self.result):
            return self.result(args)
        return self.result


class RuntimeRefactorTests(unittest.IsolatedAsyncioTestCase):
    def _workspace_tempdir(self) -> Path:
        path = Path.cwd() / ".tmp_tests" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def _require_sqlite_filesystem(self) -> None:
        probe_dir = self._workspace_tempdir()
        probe_path = probe_dir / "probe.sqlite"
        try:
            conn = sqlite3.connect(probe_path)
            conn.execute("create table t(x int)")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError as exc:
            self.skipTest(f"SQLite file backend is unavailable in this environment: {exc}")

    def _make_config(self, **overrides):
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

    def _initial_state(self, task="Проверь задачу", session_id="session-test", run_id="run-test"):
        return {
            "messages": [HumanMessage(content=task)],
            "steps": 0,
            "token_usage": {},
            "current_task": task,
            "turn_outcome": "",
            "self_correction_retry_count": 0,
            "self_correction_retry_turn_id": 0,
            "recovery_state": {
                "turn_id": 1,
                "active_issue": None,
                "active_strategy": None,
                "strategy_queue": [],
                "attempts_by_strategy": {},
                "progress_markers": [],
                "last_successful_evidence": "",
                "external_blocker": None,
                "llm_replan_attempted_for": [],
            },
            "session_id": session_id,
            "run_id": run_id,
            "turn_id": 1,
            "pending_approval": None,
            "open_tool_issue": None,
            "last_tool_error": "",
            "last_tool_result": "",
            "safety_mode": "default",
        }

    async def test_create_checkpoint_runtime_uses_sqlite_backend_when_available(self):
        self._require_sqlite_filesystem()
        tmp = self._workspace_tempdir()
        db_path = tmp / "checkpoints.sqlite"
        runtime = await create_checkpoint_runtime(
            self._make_config(CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=db_path)
        )
        try:
            self.assertEqual(runtime.resolved_backend, "sqlite")
            self.assertEqual(Path(runtime.target), db_path.resolve())
            self.assertTrue(db_path.exists())
        finally:
            await runtime.aclose()

    async def test_sqlite_checkpointer_persists_state_across_recompiled_app(self):
        self._require_sqlite_filesystem()
        tmp = self._workspace_tempdir()
        db_path = tmp / "persist.sqlite"
        config = self._make_config(CHECKPOINT_BACKEND="sqlite", CHECKPOINT_SQLITE_PATH=db_path)
        thread_config = {"configurable": {"thread_id": "persist-thread"}, "recursion_limit": 24}

        runtime1 = await create_checkpoint_runtime(config)
        app1 = create_agent_workflow(
            AgentNodes(
                config=config,
                llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: ok\nNEXT_STEP: NONE")]),
                tools=[],
                llm_with_tools=FakeLLM([AIMessage(content="Первый ответ.")]),
            ),
            config,
            tools_enabled=False,
        ).compile(checkpointer=runtime1.checkpointer)
        await app1.ainvoke(self._initial_state("Первая задача"), config=thread_config)
        await runtime1.aclose()

        runtime2 = await create_checkpoint_runtime(config)
        try:
            app2 = create_agent_workflow(
                AgentNodes(
                    config=config,
                    llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: ok\nNEXT_STEP: NONE")]),
                    tools=[],
                    llm_with_tools=FakeLLM([AIMessage(content="Второй ответ.")]),
                ),
                config,
                tools_enabled=False,
            ).compile(checkpointer=runtime2.checkpointer)
            saved_state = await app2.aget_state({"configurable": {"thread_id": "persist-thread"}})
            saved_messages = saved_state.values["messages"]
            self.assertTrue(any(isinstance(msg, HumanMessage) and msg.content == "Первая задача" for msg in saved_messages))
            self.assertTrue(
                any(
                    isinstance(msg, AIMessage)
                    and "STATUS: FINISHED" in str(msg.content)
                    and "REASON: ok" in str(msg.content)
                    for msg in saved_messages
                )
            )
        finally:
            await runtime2.aclose()

    async def test_approval_interrupt_requires_resume_before_mutating_tool_runs(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-1"}]),
                    AIMessage(content="Готово."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"action": "apply"}])
        self.assertEqual(resumed["messages"][-1].content, "Готово.")

    async def test_destructive_cli_exec_interrupts_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True, ENABLE_SHELL_TOOL=True)
        tool = FakeTool("cli_exec", "Success: command completed.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "cli_exec",
                                "args": {"command": "Remove-Item -LiteralPath demo.txt -Force"},
                                "id": "tc-cli-delete",
                            }
                        ],
                    ),
                    AIMessage(content="Команда подтверждена и выполнена."),
                ]
            ),
            tool_metadata={
                "cli_exec": ToolMetadata(
                    name="cli_exec",
                    mutating=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-cli-thread"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Удалить файл через shell"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        interrupt_entries = interrupted["__interrupt__"]
        interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
        self.assertEqual(interrupt_value["tools"][0]["name"], "cli_exec")
        self.assertTrue(interrupt_value["tools"][0]["policy"]["destructive"])
        self.assertTrue(interrupt_value["tools"][0]["policy"]["requires_approval"])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"command": "Remove-Item -LiteralPath demo.txt -Force"}])
        self.assertEqual(resumed["messages"][-1].content, "Команда подтверждена и выполнена.")

    async def test_safe_delete_file_interrupts_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("safe_delete_file", "Success: file deleted.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "safe_delete_file",
                                "args": {"path": "demo.txt"},
                                "id": "tc-delete-file",
                            }
                        ],
                    ),
                    AIMessage(content="Файл удален после подтверждения."),
                ]
            ),
            tool_metadata={
                "safe_delete_file": ToolMetadata(
                    name="safe_delete_file",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-safe-delete"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Удали файл"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"path": "demo.txt"}])
        self.assertEqual(resumed["messages"][-1].content, "Файл удален после подтверждения.")

    async def test_mcp_execution_tool_interrupts_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("terminal:run_command", "Success: command completed.")
        mcp_metadata = ToolRegistry._infer_mcp_metadata(tool)
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "terminal:run_command",
                                "args": {"command": "git status"},
                                "id": "tc-mcp-run",
                            }
                        ],
                    ),
                    AIMessage(content="MCP команда выполнена после подтверждения."),
                ]
            ),
            tool_metadata={"terminal:run_command": mcp_metadata},
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-mcp-exec"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Запусти MCP команду"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        interrupt_entries = interrupted["__interrupt__"]
        interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
        self.assertEqual(interrupt_value["tools"][0]["name"], "terminal:run_command")
        self.assertTrue(interrupt_value["tools"][0]["policy"]["requires_approval"])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)
        self.assertEqual(tool.calls, [{"command": "git status"}])
        self.assertEqual(resumed["messages"][-1].content, "MCP команда выполнена после подтверждения.")

    async def test_regular_edit_requires_resume_before_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("edit_file", "Success: File edited.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt", "old_string": "a", "new_string": "b"}, "id": "tc-edit-plain"}]),
                    AIMessage(content="Готово без дополнительного approval."),
                ]
            ),
            tool_metadata={
                "edit_file": ToolMetadata(
                    name="edit_file",
                    mutating=True,
                    requires_approval=False,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())

        thread_config = {"configurable": {"thread_id": "plain-edit-approval"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(
            self._initial_state("Исправь файл"),
            config=thread_config,
        )

        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(tool.calls, [])

        resumed = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertEqual(tool.calls, [{"path": "demo.txt", "old_string": "a", "new_string": "b"}])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIn("без дополнительного approval", str(resumed["messages"][-1].content).lower())

    async def test_mutating_non_destructive_tool_error_enters_recovery_after_approval(self):
        config = self._make_config(ENABLE_APPROVALS=True)

        def _edit_result(args):
            if args.get("old_string") == "bad":
                return "ERROR[VALIDATION]: Could not find a match for 'old_string'."
            return "Success: File edited."

        edit_tool = FakeTool("edit_file", _edit_result)
        read_tool = FakeTool("read_file", "import sys\nimport logging")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: done\nNEXT_STEP: NONE")]),
            tools=[edit_tool, read_tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt", "old_string": "bad", "new_string": "good"}, "id": "tc-edit-1"}]),
                    AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "demo.txt"}, "id": "tc-read-1"}]),
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt", "old_string": "import sys", "new_string": "import sys\nimport json"}, "id": "tc-edit-2"}]),
                    AIMessage(content="Исправление завершено после recovery."),
                ]
            ),
            tool_metadata={
                "edit_file": ToolMetadata(name="edit_file", mutating=True, destructive=False, requires_approval=False),
                "read_file": ToolMetadata(name="read_file", read_only=True),
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())

        thread_config = {"configurable": {"thread_id": "mutating-recovery-approval"}, "recursion_limit": 64}

        interrupted = await app.ainvoke(
            self._initial_state("Исправь файл"),
            config=thread_config,
        )

        self.assertIn("__interrupt__", interrupted)
        self.assertEqual(edit_tool.calls, [])
        self.assertEqual(read_tool.calls, [])

        second_interrupt = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertIn("__interrupt__", second_interrupt)
        self.assertEqual(len(edit_tool.calls), 1)
        self.assertEqual(len(read_tool.calls), 1)

        result = await app.ainvoke(Command(resume={"approved": True}), config=thread_config)

        self.assertEqual(len(edit_tool.calls), 2)
        self.assertEqual(len(read_tool.calls), 1)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])

    async def test_invoke_llm_with_retry_normalizes_context_once_per_attempt(self):
        config = self._make_config(MAX_RETRIES=2, RETRY_DELAY=0)
        llm = FakeLLM([RuntimeError("temporary"), AIMessage(content="ok")])
        nodes = AgentNodes(
            config=config,
            llm=llm,
            tools=[],
            llm_with_tools=llm,
        )
        context = [HumanMessage(content="Проверь задачу")]

        with mock.patch.object(
            AgentNodes,
            "_normalize_system_prefix_for_provider",
            autospec=True,
            side_effect=lambda _self, payload: list(payload),
        ) as normalize_mock:
            response = await nodes._invoke_llm_with_retry(llm, context, state=self._initial_state(), node_name="agent")

        self.assertEqual(str(response.content), "ok")
        self.assertEqual(normalize_mock.call_count, 2)

    async def test_approval_rejection_returns_access_denied_without_tool_execution(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: blocker reported\nNEXT_STEP: NONE")]),
            tools=[tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-2"}]),
                    AIMessage(content="Не удалось выполнить действие без подтверждения."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-2"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        tool_messages = [msg for msg in resumed["messages"] if isinstance(msg, ToolMessage)]
        self.assertTrue(tool_messages)
        self.assertIn("ACCESS_DENIED", str(tool_messages[-1].content))
        self.assertIsNone(resumed["open_tool_issue"])

    async def test_approval_rejection_blocks_followup_tool_calls_in_same_turn(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        denied_tool = FakeTool("danger_tool", "Изменение применено.")
        fallback_tool = FakeTool("write_file", "Файл записан.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([AIMessage(content="STATUS: FINISHED\nREASON: blocker reported\nNEXT_STEP: NONE")]),
            tools=[denied_tool, fallback_tool],
            llm_with_tools=FakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-3"}]),
                    AIMessage(
                        content="Сохраню результат в другой файл.",
                        tool_calls=[{"name": "write_file", "args": {"path": "alt.md", "content": "x"}, "id": "tc-4"}],
                    ),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                ),
                "write_file": ToolMetadata(
                    name="write_file",
                    mutating=True,
                    requires_approval=True,
                ),
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-3"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(denied_tool.calls, [])
        self.assertEqual(fallback_tool.calls, [])
        self.assertIsInstance(resumed["messages"][-1], AIMessage)
        self.assertIn("вы отклонили", str(resumed["messages"][-1].content).lower())
        self.assertIsNone(resumed["open_tool_issue"])

    async def test_approval_rejection_resume_keeps_provider_safe_order(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM(
                [
                    AIMessage(
                        content=(
                            "STATUS: INCOMPLETE\n"
                            "REASON: The assistant is waiting for the user after denial.\n"
                            "NEXT_STEP: wait for the next instruction\n"
                            "CONTROL: FINISH_TURN"
                        )
                    )
                ]
            ),
            tools=[tool],
            llm_with_tools=ProviderSafeFakeLLM(
                [
                    AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-safe"}]),
                    AIMessage(content="Действие не выполнено: вы выбрали Нет для необратимой операции."),
                ]
            ),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-safe"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIn("вы отклонили", str(resumed["messages"][-1].content).lower())

    async def test_approval_rejection_finishes_turn_without_secondary_verifier(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        agent_llm = ProviderSafeFakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-mal"}]),
                AIMessage(content="Не сделал, потому что вы выбрали Нет. Жду следующую инструкцию."),
            ]
        )
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )
        app = create_agent_workflow(nodes, config, tools_enabled=True).compile(checkpointer=MemorySaver())
        thread_config = {"configurable": {"thread_id": "approval-thread-malformed"}, "recursion_limit": 36}

        interrupted = await app.ainvoke(self._initial_state("Сделай изменение"), config=thread_config)
        self.assertIn("__interrupt__", interrupted)
        resumed = await app.ainvoke(Command(resume={"approved": False}), config=thread_config)

        self.assertEqual(tool.calls, [])
        self.assertEqual(resumed["turn_outcome"], "finish_turn")
        self.assertIsNone(resumed["open_tool_issue"])
        self.assertEqual(len(agent_llm.invocations), 1)
        final_text = str(resumed["messages"][-1].content).lower()
        self.assertIn("вы отклонили", final_text)

    def test_sanitize_messages_for_model_remaps_non_compliant_tool_call_ids(self):
        config = self._make_config()
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=FakeLLM([]),
        )
        source_id = "chatcmpl-tool-9d8dc0bbe38d6202"
        sanitized = nodes._sanitize_messages_for_model(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": source_id,
                            "name": "read_file",
                            "args": {"path": "README.md"},
                        }
                    ],
                ),
                ToolMessage(
                    tool_call_id=source_id,
                    name="read_file",
                    content="ok",
                ),
            ]
        )

        self.assertEqual(len(sanitized), 2)
        self.assertIsInstance(sanitized[0], AIMessage)
        self.assertIsInstance(sanitized[1], ToolMessage)
        remapped_id = sanitized[0].tool_calls[0]["id"]
        self.assertRegex(remapped_id, r"^[A-Za-z0-9]{9}$")
        self.assertEqual(sanitized[1].tool_call_id, remapped_id)

    async def test_openai_context_stringifies_assistant_content_lists_before_invoke(self):
        config = self._make_config()
        llm = OpenAIContentSafeFakeLLM([AIMessage(content="ok")])
        nodes = AgentNodes(
            config=config,
            llm=llm,
            tools=[],
            llm_with_tools=llm,
        )
        state = self._initial_state("Продолжай анализ")
        context = nodes.context_builder.build(
            [
                HumanMessage(content="Проверь файл"),
                AIMessage(content=["Промежуточный ", "ответ."]),
                HumanMessage(content="Продолжай анализ"),
            ],
            state,
            summary="",
            current_task="Продолжай анализ",
            tools_available=False,
            active_tool_names=[],
            open_tool_issue=None,
            recovery_state=None,
        )

        response = await nodes._invoke_llm_with_retry(llm, context, state=state, node_name="agent")

        self.assertEqual(str(response.content), "ok")
        self.assertTrue(any(isinstance(message, AIMessage) and message.content == "Промежуточный ответ." for message in llm.invocations[0]))

    async def test_tools_node_marks_approval_denied_as_open_issue_before_agent_ack(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        nodes = AgentNodes(
            config=config,
            llm=FakeLLM([]),
            tools=[tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "danger_tool": ToolMetadata(
                    name="danger_tool",
                    mutating=True,
                    destructive=True,
                    requires_approval=True,
                )
            },
        )

        result = await nodes.tools_node(
            {
                **self._initial_state("Сделай изменение"),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-issue"}],
                    )
                ],
                "pending_approval": {
                    "approved": False,
                    "decision": {"approved": False},
                    "tool_call_ids": ["tc-issue"],
                    "tool_names": ["danger_tool"],
                },
            }
        )

        self.assertEqual(result["open_tool_issue"]["kind"], "approval_denied")
        self.assertEqual(result["open_tool_issue"]["turn_id"], 1)

    async def test_run_logger_writes_structured_tool_failure_event(self):
        tmp = self._workspace_tempdir()
        logger = JsonlRunLogger(tmp)
        failing_tool = FakeTool("danger_tool", "ERROR[EXECUTION]: boom")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[failing_tool],
            llm_with_tools=FakeLLM([]),
            run_logger=logger,
        )

        await nodes.tools_node(
            {
                **self._initial_state("Почини ошибку", session_id="session-log", run_id="run-log"),
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "danger_tool", "args": {"action": "x"}, "id": "tc-log"}],
                    )
                ],
            }
        )

        log_path = logger.file_path_for("session-log")
        records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
        end_records = [record for record in records if record["event"] == "tool_call_end"]
        self.assertTrue(end_records)
        self.assertEqual(end_records[-1]["result"]["ok"], False)
        self.assertEqual(end_records[-1]["result"]["error_type"], "EXECUTION")

    def test_session_store_round_trip(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        snapshot = store.new_session(checkpoint_backend="sqlite", checkpoint_target="demo.sqlite")
        snapshot.approval_mode = "always"
        store.save_active_session(snapshot)
        loaded = store.load_active_session()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, snapshot.session_id)
        self.assertEqual(loaded.thread_id, snapshot.thread_id)
        self.assertEqual(loaded.approval_mode, "always")

    def test_session_store_defaults_missing_approval_mode_to_prompt(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        session_path.write_text(
            json.dumps(
                {
                    "session_id": "session-old",
                    "thread_id": "thread-old",
                    "checkpoint_backend": "sqlite",
                    "checkpoint_target": "demo.sqlite",
                    "created_at": "2026-03-25T10:00:00+00:00",
                    "updated_at": "2026-03-25T10:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        loaded = SessionStore(session_path).load_active_session()

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.approval_mode, "prompt")

    def test_session_store_keeps_project_scoped_index(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(first, touch=False)
        store.save_active_session(second, touch=False)

        project_a = store.list_sessions(tmp / "project-a")
        project_b = store.list_sessions(tmp / "project-b")

        self.assertEqual([entry.title for entry in project_a], ["First"])
        self.assertEqual([entry.title for entry in project_b], ["Second"])
        self.assertEqual(store.get_active_session_for_project(tmp / "project-a").session_id, first.session_id)
        self.assertEqual(store.get_active_session_for_project(tmp / "project-b").session_id, second.session_id)
        self.assertCountEqual([entry.title for entry in store.list_sessions()], ["First", "Second"])

    def test_session_store_tracks_global_last_active_session(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        last_active = store.get_last_active_session()

        self.assertIsNotNone(last_active)
        self.assertEqual(last_active.session_id, second.session_id)

    def test_session_store_delete_session_soft_removes_from_index_and_updates_last_active(self):
        tmp = self._workspace_tempdir()
        store = SessionStore(tmp / "session.json")
        first = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-a", title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=tmp / "project-b", title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        deleted = store.delete_session(second.session_id)

        self.assertTrue(deleted)
        self.assertIsNone(store.get_session(second.session_id))
        self.assertEqual(store.get_last_active_session().session_id, first.session_id)
        self.assertEqual(SessionStore(tmp / "session.json").load_active_session().session_id, first.session_id)

    def test_session_store_migrates_legacy_session_into_index(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        session_path.write_text(
            json.dumps(
                {
                    "session_id": "legacy-session",
                    "thread_id": "legacy-thread",
                    "checkpoint_backend": "sqlite",
                    "checkpoint_target": "demo.sqlite",
                    "created_at": "2026-03-25T10:00:00+00:00",
                    "updated_at": "2026-03-25T10:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        store = SessionStore(session_path)
        loaded = store.load_active_session()
        entries = store.list_sessions()

        self.assertIsNotNone(loaded)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].session_id, "legacy-session")

    def test_generate_chat_title_strips_common_prefixes_and_limits_length(self):
        self.assertEqual(
            generate_chat_title("Помоги скачать и настроить Apache на Windows"),
            "Скачать и настроить Apache на Windows",
        )
        self.assertEqual(generate_chat_title("   \n   "), "New Chat")
        self.assertTrue(generate_chat_title("сделай " + ("очень длинный запрос " * 10)).endswith("…"))

    def test_project_title_helpers_append_short_folder_label(self):
        project_path = Path("D:/work/client/demo-app")
        self.assertEqual(short_project_label(project_path), "client/demo-app")
        self.assertEqual(append_project_label("New Chat", project_path), "New Chat [client/demo-app]")

    def test_worker_preserves_project_label_when_first_title_is_generated(self):
        tmp = self._workspace_tempdir()
        worker = gui_runtime.AgentRunWorker()
        worker.store = SessionStore(tmp / "session.json")
        worker.current_session = worker.store.new_session(
            "sqlite",
            "demo.sqlite",
            project_path=tmp / "client" / "demo-app",
            title=append_project_label("New Chat", tmp / "client" / "demo-app"),
        )

        updated = worker._maybe_set_session_title("Сделай сводку по ошибкам")

        self.assertTrue(updated)
        self.assertEqual(worker.current_session.title, "Сводку по ошибкам [client/demo-app]")

    def test_build_transcript_payload_restores_turns_and_summary_notice(self):
        payload = build_transcript_payload(
            {
                "summary": "compressed earlier messages",
                "messages": [
                    HumanMessage(content="Покажи diff"),
                    AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"path": "demo.txt"}, "id": "tc-1"}]),
                    ToolMessage(tool_call_id="tc-1", content="Done\n```diff\n-old\n+new\n```", name="edit_file"),
                    AIMessage(content="Готово."),
                ],
            }
        )

        self.assertTrue(payload["summary_notice"])
        self.assertEqual(len(payload["turns"]), 1)
        self.assertEqual(payload["turns"][0]["user_text"], "Покажи diff")
        block_types = [block["type"] for block in payload["turns"][0]["blocks"]]
        self.assertEqual(block_types, ["tool", "assistant"])
        self.assertEqual(payload["turns"][0]["blocks"][0]["payload"]["name"], "edit_file")
        self.assertIn("Готово", payload["turns"][0]["blocks"][1]["markdown"])

    def test_build_transcript_payload_hides_internal_handoff_and_restores_notice(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Проверь завершение"),
                    AIMessage(
                        content="internal handoff",
                        additional_kwargs={
                            "agent_internal": {
                                "kind": "tool_issue_handoff",
                                "visible_in_ui": False,
                                "ui_notice": "Автопродолжение остановлено. Нужен новый запрос.",
                            }
                        },
                    ),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        self.assertEqual(payload["turns"][0]["user_text"], "Проверь завершение")
        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["notice"])
        self.assertEqual(blocks[0]["message"], "Автопродолжение остановлено. Нужен новый запрос.")
        self.assertEqual(blocks[0]["level"], "warning")

    def test_build_transcript_payload_keeps_assistant_text_without_pending_choice_state(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Нужно выбрать режим"),
                    AIMessage(
                        content=(
                            "Сначала нужно решить, какой путь подтверждаем.\n\n"
                            "Сделаю паузу и запрошу выбор через tool.\n"
                        )
                    ),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["assistant"])
        self.assertIn("Сначала нужно решить", blocks[0]["markdown"])
        legacy_choice_key = "_".join(("pending", "user", "choice"))
        self.assertNotIn(legacy_choice_key, payload)

    def test_build_transcript_payload_restores_assistant_text_from_list_content(self):
        payload = build_transcript_payload(
            {
                "messages": [
                    HumanMessage(content="Подведи итог"),
                    AIMessage(
                        content=[
                            "Я завершил основную часть задачи. ",
                            {"type": "text", "text": "Изменения уже внесены."},
                            {"content": [" Проверка пройдена."]},
                        ]
                    ),
                ],
            }
        )

        self.assertEqual(len(payload["turns"]), 1)
        blocks = payload["turns"][0]["blocks"]
        self.assertEqual([block["type"] for block in blocks], ["assistant"])
        self.assertIn("Я завершил основную часть задачи.", blocks[0]["markdown"])
        self.assertIn("Изменения уже внесены.", blocks[0]["markdown"])
        self.assertIn("Проверка пройдена.", blocks[0]["markdown"])

    def test_build_user_choice_payload_adapts_interrupt_options_for_card(self):
        payload = build_user_choice_payload(
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
        )

        self.assertEqual(payload["kind"], "user_choice")
        self.assertEqual(payload["question"], "Введите ключ API или выберите другой вариант:")
        self.assertEqual(payload["recommended_key"], "Ввести ключ API")
        self.assertEqual(
            [option["submit_text"] for option in payload["options"]],
            [
                "Ввести ключ API",
                "Пропустить проверку и вернуть скрипт",
                "Завершить проверку",
            ],
        )
        recommended = [option for option in payload["options"] if option["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["submit_text"], "Ввести ключ API")

    def test_build_user_choice_payload_matches_recommended_by_synthesized_option_key(self):
        payload = build_user_choice_payload(
            {
                "kind": "user_choice",
                "question": "Как продолжаем?",
                "options": [
                    "Вариант 1: direct_api",
                    "Вариант 2: keep_mcp",
                ],
                "recommended": "option_1",
            }
        )

        recommended = [option for option in payload["options"] if option["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["key"], "option_1")
        self.assertEqual(payload["recommended_key"], "Вариант 1: direct_api")

    async def test_worker_initialize_can_force_new_session_on_reinitialize(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        old_session = store.new_session(checkpoint_backend="sqlite", checkpoint_target="demo.sqlite")
        store.save_active_session(old_session)
        project_path = tmp / "project-folder"
        project_path.mkdir()

        worker = gui_runtime.AgentRunWorker()
        config = self._make_config(SESSION_STATE_PATH=session_path)
        fake_app = type("FakeApp", (), {"get_state": lambda self, _config: type("State", (), {"values": {}})()})()
        tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
            },
        )()

        with (
            mock.patch.object(gui_runtime, "setup_runtime", return_value=config),
            mock.patch.object(gui_runtime, "build_agent_app", new=mock.AsyncMock(return_value=(fake_app, tool_registry))),
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock()),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_path),
        ):
            await worker._initialize_async(force_new_session=True)

        self.assertNotEqual(worker.current_session.session_id, old_session.session_id)
        self.assertNotEqual(worker.current_session.thread_id, old_session.thread_id)
        self.assertEqual(worker.current_session.title, append_project_label("New Chat", project_path))
        persisted = SessionStore(session_path).load_active_session()
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.session_id, worker.current_session.session_id)

    async def test_worker_switch_session_updates_active_snapshot(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False)
        store.save_active_session(second, touch=False)

        worker = gui_runtime.AgentRunWorker()
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()
        worker.current_session = first

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock()),
            mock.patch.object(gui_runtime.os, "chdir") as chdir_mock,
            mock.patch.object(gui_runtime.Path, "cwd", return_value=(tmp / "project-a")),
        ):
            await worker._switch_session_async(second)

        self.assertEqual(worker.current_session.session_id, second.session_id)
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, second.session_id)
        chdir_mock.assert_called_once_with(str((tmp / "project-b").resolve()))

    async def test_worker_initialize_fallbacks_when_chdir_fails(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        blocked_project = tmp / "blocked-project"
        fallback_project = tmp / "fallback-project"
        blocked_project.mkdir()
        fallback_project.mkdir()
        blocked = store.new_session("sqlite", "demo.sqlite", project_path=blocked_project, title="Blocked")
        store.save_active_session(blocked, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        config = self._make_config(SESSION_STATE_PATH=session_path)
        fake_app = type("FakeApp", (), {"get_state": lambda self, _config: type("State", (), {"values": {}})()})()
        tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()

        with (
            mock.patch.object(gui_runtime, "setup_runtime", return_value=config),
            mock.patch.object(gui_runtime, "build_agent_app", new=mock.AsyncMock(return_value=(fake_app, tool_registry))),
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=fallback_project),
            mock.patch.object(gui_runtime.os, "chdir", side_effect=PermissionError("denied")),
            mock.patch.object(worker, "_log_ui_run_event") as log_mock,
        ):
            await worker._initialize_async()

        self.assertNotEqual(worker.current_session.session_id, blocked.session_id)
        self.assertEqual(worker.current_session.project_path, str(fallback_project.resolve()))
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, worker.current_session.session_id)
        fallback_notices = [
            event
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "session_fallback"
        ]
        self.assertTrue(fallback_notices)
        self.assertTrue(any(call.args and call.args[0] == "session_restore_fallback" for call in log_mock.call_args_list))
        self.assertTrue(
            any(call.kwargs.get("reason") == "chdir_failed" and "PermissionError" in call.kwargs.get("error", "") for call in log_mock.call_args_list)
        )

    async def test_worker_switch_session_fallbacks_when_chdir_raises(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()
        worker.current_session = first

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_a),
            mock.patch.object(gui_runtime.os, "chdir", side_effect=OSError("network share denied")),
            mock.patch.object(worker, "_log_ui_run_event") as log_mock,
        ):
            await worker._switch_session_async(second)

        self.assertNotEqual(worker.current_session.session_id, second.session_id)
        self.assertEqual(worker.current_session.project_path, str(project_a.resolve()))
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, worker.current_session.session_id)
        fallback_notices = [
            event
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "session_fallback"
        ]
        self.assertTrue(fallback_notices)
        self.assertTrue(any(call.args and call.args[0] == "session_switch_fallback" for call in log_mock.call_args_list))
        self.assertTrue(
            any(call.kwargs.get("reason") == "chdir_failed" and "OSError" in call.kwargs.get("error", "") for call in log_mock.call_args_list)
        )

    def test_worker_selects_global_last_active_session_on_initialize(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()

        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {"checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"}},
        )()

        with mock.patch.object(gui_runtime.Path, "cwd", return_value=project_a):
            selected = worker._select_session_for_project()

        self.assertEqual(selected.session_id, second.session_id)

    async def test_new_user_turn_ignores_old_open_tool_issue(self):
        agent_llm = FakeLLM([AIMessage(content="Короткая сводка на экране.")])
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[],
            llm_with_tools=agent_llm,
        )
        state = {
            **self._initial_state("Покажи коротко эту инфу на экран"),
            "messages": [
                HumanMessage(content="Сохрани в файл"),
                AIMessage(content="Не сделал, так как вы выбрали Нет. Ожидаю дальнейших инструкций."),
                HumanMessage(content="Покажи коротко эту инфу на экран"),
            ],
            "turn_id": 1,
            "open_tool_issue": {
                "turn_id": 1,
                "kind": "approval_denied",
                "summary": "Execution of 'write_file' was cancelled by approval policy.",
                "tool_names": ["write_file"],
                "source": "approval",
            },
            "current_task": "Покажи коротко эту инфу на экран",
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_id"], 2)
        self.assertIsNone(result["open_tool_issue"])
        unresolved_messages = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if "UNRESOLVED TOOL FAILURE" in str(message.content) or "TOOL EXECUTION DENIED BY USER" in str(message.content)
        ]
        self.assertFalse(unresolved_messages)

    async def test_latest_user_message_overrides_stale_current_task_in_state(self):
        agent_llm = FakeLLM([AIMessage(content="Проверяю только указанный файл.")])
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[],
            llm_with_tools=agent_llm,
        )
        state = {
            **self._initial_state("проверь list_mistral_models.py"),
            "messages": [
                HumanMessage(content="восстанови проверку mistral api"),
                AIMessage(content="Промежуточный результат."),
                HumanMessage(content="проверь list_mistral_models.py"),
            ],
            "current_task": "восстанови проверку mistral api",
            "turn_id": 1,
        }

        result = await nodes.agent_node(state)

        self.assertEqual(result["current_task"], "проверь list_mistral_models.py")
        visible_humans = [
            str(message.content)
            for message in agent_llm.invocations[0]
            if isinstance(message, HumanMessage)
        ]
        self.assertIn("проверь list_mistral_models.py", visible_humans)

    async def test_agent_node_forces_recovery_when_action_turn_ends_with_prose_only(self):
        agent_llm = FakeLLM([AIMessage(content="Сейчас исправлю файл и внесу правки.")])
        edit_tool = FakeTool("edit_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[edit_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "edit_file": ToolMetadata(name="edit_file", mutating=True, requires_approval=False),
            },
        )

        result = await nodes.agent_node(self._initial_state("исправь файл demo.txt"))

        self.assertEqual(result["turn_outcome"], "")
        self.assertTrue(result["has_protocol_error"])
        self.assertIsNotNone(result["open_tool_issue"])
        self.assertEqual(result["open_tool_issue"]["kind"], "protocol_error")
        self.assertEqual(result["open_tool_issue"]["details"]["protocol_reason"], "action_requires_tools")
        self.assertEqual(len(agent_llm.invocations), 1)

        guard_result = await nodes.stability_guard_node({**self._initial_state("исправь файл demo.txt"), **result})
        self.assertEqual(guard_result["turn_outcome"], "recover_agent")

    async def test_agent_node_marks_malformed_tool_payload_as_protocol_error(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content="Открываю файл.",
                    invalid_tool_calls=[
                        {
                            "id": "tc-invalid-1",
                            "name": "read_file",
                            "args": "{path: README.md}",
                            "error": "malformed arguments",
                        }
                    ],
                )
            ]
        )
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )

        result = await nodes.agent_node(self._initial_state("прочитай README.md"))

        self.assertEqual(result["turn_outcome"], "")
        self.assertTrue(result["has_protocol_error"])
        self.assertEqual(result["open_tool_issue"]["kind"], "protocol_error")
        self.assertEqual(result["open_tool_issue"]["details"]["protocol_reason"], "tool_protocol_error")

        guard_result = await nodes.stability_guard_node({**self._initial_state("прочитай README.md"), **result})
        self.assertEqual(guard_result["turn_outcome"], "recover_agent")

    async def test_agent_node_detects_history_tool_mismatch_before_llm_invoke(self):
        agent_llm = FakeLLM([AIMessage(content="Это сообщение не должно быть вызвано.")])
        read_tool = FakeTool("read_file", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={"read_file": ToolMetadata(name="read_file", read_only=True)},
        )
        state = self._initial_state("продолжай")
        state["messages"] = [
            HumanMessage(content="прочитай README.md"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "tc-history-1"}],
            ),
            AIMessage(content="Продолжаю без результата инструмента."),
            HumanMessage(content="продолжай"),
        ]

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "")
        self.assertTrue(result["has_protocol_error"])
        self.assertEqual(result["open_tool_issue"]["details"]["protocol_reason"], "history_tool_mismatch")
        self.assertEqual(agent_llm.invocations, [])

    async def test_agent_node_drops_extra_request_user_input_calls_in_one_response(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content="Запрашиваю выбор пользователя.",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {"question": "Первый выбор", "options": ["A", "B"]},
                            "id": "tc-user-1",
                        },
                        {
                            "name": "request_user_input",
                            "args": {"question": "Второй выбор", "options": ["C", "D"]},
                            "id": "tc-user-2",
                        },
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool],
            llm_with_tools=agent_llm,
        )

        result = await nodes.agent_node(self._initial_state("Проведи тест user input"))

        self.assertEqual(result["turn_outcome"], "run_tools")
        self.assertTrue(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["id"], "tc-user-1")
        self.assertEqual(str(response.content), "Запрашиваю выбор пользователя.")

    async def test_agent_node_blocks_second_request_user_input_in_same_turn(self):
        agent_llm = FakeLLM(
            [
                AIMessage(
                    content="Запрашиваю еще один выбор.",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {"question": "Еще вопрос", "options": ["Да", "Нет"]},
                            "id": "tc-user-repeat",
                        }
                    ],
                ),
                AIMessage(content="Продолжаю работу с уже выбранным вариантом."),
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool],
            llm_with_tools=agent_llm,
        )
        state = self._initial_state("Проведи тест user input")
        state["messages"] = [
            HumanMessage(content="Проведи тест user input"),
            AIMessage(
                content="Запрашиваю выбор.",
                tool_calls=[
                    {
                        "name": "request_user_input",
                        "args": {"question": "Первый вопрос", "options": ["A", "B"]},
                        "id": "tc-user-initial",
                    }
                ],
            ),
            ToolMessage(
                content="A",
                tool_call_id="tc-user-initial",
                name="request_user_input",
            ),
        ]

        result = await nodes.agent_node(state)

        self.assertEqual(result["turn_outcome"], "")
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertFalse(response.tool_calls)
        self.assertEqual(str(response.content), "Продолжаю работу с уже выбранным вариантом.")
        self.assertEqual(len(agent_llm.invocations), 2)

    async def test_agent_node_excludes_request_user_input_from_bound_tools_after_choice(self):
        agent_llm = FakeBindableLLM([AIMessage(content="Продолжаю без нового вопроса.")])
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
        )
        state = self._initial_state("Проведи тест user input")
        state["messages"] = [
            HumanMessage(content="Проведи тест user input"),
            AIMessage(
                content="Запрашиваю выбор.",
                tool_calls=[
                    {
                        "name": "request_user_input",
                        "args": {"question": "Первый вопрос", "options": ["A", "B"]},
                        "id": "tc-user-initial",
                    }
                ],
            ),
            ToolMessage(
                content="A",
                tool_call_id="tc-user-initial",
                name="request_user_input",
            ),
        ]

        await nodes.agent_node(state)

        self.assertTrue(agent_llm.bound_tool_name_batches)
        self.assertNotIn("request_user_input", agent_llm.bound_tool_name_batches[-1])
        self.assertIn("read_file", agent_llm.bound_tool_name_batches[-1])

    async def test_worker_delete_active_session_switches_to_fallback_session(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.current_session = second
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=(tmp / "project-a")),
            mock.patch.object(gui_runtime.os, "chdir"),
        ):
            await worker._delete_session_async(second.session_id)

        self.assertEqual(worker.current_session.session_id, first.session_id)
        self.assertIsNone(SessionStore(session_path).get_session(second.session_id))

    async def test_worker_delete_active_session_fallbacks_when_replacement_chdir_fails(self):
        tmp = self._workspace_tempdir()
        session_path = tmp / "session.json"
        store = SessionStore(session_path)
        project_a = tmp / "project-a"
        project_b = tmp / "project-b"
        project_a.mkdir()
        project_b.mkdir()
        first = store.new_session("sqlite", "demo.sqlite", project_path=project_a, title="First")
        second = store.new_session("sqlite", "demo.sqlite", project_path=project_b, title="Second")
        store.save_active_session(first, touch=False, set_active=True)
        store.save_active_session(second, touch=False, set_active=True)

        worker = gui_runtime.AgentRunWorker()
        events = []
        worker.event_emitted.connect(events.append)
        worker.store = store
        worker.config = self._make_config(SESSION_STATE_PATH=session_path)
        worker.current_session = second
        worker.tool_registry = type(
            "ToolRegistryStub",
            (),
            {
                "checkpoint_info": {"resolved_backend": "sqlite", "target": "demo.sqlite"},
                "tools": [],
                "tool_metadata": {},
                "mcp_server_status": [],
                "get_runtime_status_lines": lambda self: [],
                "sync_working_directory": lambda self, _path: None,
            },
        )()
        worker.agent_app = type(
            "FakeApp",
            (),
            {"get_state": lambda self, _config: type("State", (), {"values": {}})()},
        )()

        with (
            mock.patch.object(gui_runtime, "repair_session_if_needed", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(gui_runtime.Path, "cwd", return_value=project_b),
            mock.patch.object(gui_runtime.os, "chdir", side_effect=OSError("access denied")),
            mock.patch.object(worker, "_log_ui_run_event") as log_mock,
        ):
            await worker._delete_session_async(second.session_id)

        self.assertIsNone(SessionStore(session_path).get_session(second.session_id))
        self.assertEqual(worker.current_session.project_path, str(project_b.resolve()))
        self.assertNotEqual(worker.current_session.session_id, first.session_id)
        self.assertNotEqual(worker.current_session.session_id, second.session_id)
        self.assertEqual(SessionStore(session_path).load_active_session().session_id, worker.current_session.session_id)
        fallback_notices = [
            event
            for event in events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "session_fallback"
        ]
        self.assertTrue(fallback_notices)
        self.assertTrue(any(call.args and call.args[0] == "session_delete_fallback" for call in log_mock.call_args_list))
        self.assertTrue(
            any(call.kwargs.get("reason") == "chdir_failed" and "OSError" in call.kwargs.get("error", "") for call in log_mock.call_args_list)
        )

    async def test_model_profiles_apply_persists_without_runtime_reload(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-old",
                        "base_url": "",
                    }
                ],
            }
        )
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(return_value=None),
        ) as rebuild_mock:
            result = await worker._apply_model_profiles(
                {
                    "active_profile": "gemini-1-5-flash",
                    "profiles": [
                        {
                            "id": "gemini-1-5-flash",
                            "provider": "gemini",
                            "model": "gemini-1.5-flash",
                            "api_key": "gm-new",
                            "base_url": "",
                        }
                    ],
                },
                success_notice_kind="model_switched",
                success_notice_message="ok",
            )

        self.assertTrue(result)
        rebuild_mock.assert_not_called()
        restored = worker.profile_store.load_or_initialize()
        self.assertEqual(restored["active_profile"], "gemini-1-5-flash")

    async def test_model_profiles_apply_success_updates_state(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save({"active_profile": None, "profiles": []})
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        result = await worker._apply_model_profiles(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-new",
                        "base_url": "",
                    }
                ],
            },
            success_notice_kind="profiles_saved",
            success_notice_message="saved",
        )

        self.assertTrue(result)
        self.assertEqual(worker.model_profiles["active_profile"], "gpt-4o")
        self.assertEqual(worker.profile_store.load_or_initialize()["active_profile"], "gpt-4o")

    async def test_set_active_profile_notice_mentions_target_model(self):
        tmp = self._workspace_tempdir()
        profile_path = tmp / "config.json"
        worker = gui_runtime.AgentRunWorker()
        worker.profile_store = ModelProfileStore(profile_path)
        worker.model_profiles = worker.profile_store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-old",
                        "base_url": "",
                    },
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-new",
                        "base_url": "",
                    },
                ],
            }
        )
        emitted_events = []
        worker.event_emitted.connect(emitted_events.append)
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        await worker._set_active_profile_async("gemini-1-5-flash")

        notice_events = [
            event for event in emitted_events
            if getattr(event, "type", "") == "summary_notice" and (event.payload or {}).get("kind") == "model_switched"
        ]
        self.assertEqual(len(notice_events), 1)
        self.assertEqual(
            notice_events[0].payload.get("message"),
            "Модель переключена на gemini-1.5-flash. Новый выбор будет использован в следующем запросе.",
        )

    async def test_runtime_switch_happens_lazy_on_next_request_path(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-key",
                    "base_url": "",
                }
            ],
        }
        worker._runtime_profile_id = "gpt-4o"
        worker._emit_session_payload = mock.AsyncMock(return_value={})

        with mock.patch.object(
            worker,
            "_rebuild_runtime_for_active_profile",
            new=mock.AsyncMock(return_value=None),
        ):
            result = await worker._ensure_runtime_matches_selected_profile()

        self.assertTrue(result)
        self.assertEqual(worker._runtime_profile_id, "gemini-1-5-flash")

    def test_profile_bootstrap_env_comes_from_loaded_config(self):
        config = self._make_config(
            PROVIDER="openai",
            OPENAI_MODEL="openai/gpt-oss-120b",
            OPENAI_API_KEY="sk-config",
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            GEMINI_MODEL="gemini-1.5-flash",
            GEMINI_API_KEY="gm-config",
        )
        env_map = gui_runtime.AgentRunWorker._profile_bootstrap_env_from_config(config)
        self.assertEqual(env_map["PROVIDER"], "openai")
        self.assertEqual(env_map["MODEL"], "openai/gpt-oss-120b")
        self.assertEqual(env_map["API_KEY"], "sk-config")
        self.assertEqual(env_map["BASE_URL"], "https://openrouter.ai/api/v1")
        self.assertEqual(env_map["OPENAI_MODEL"], "openai/gpt-oss-120b")

    def test_worker_start_run_with_missing_model_profile_emits_notice(self):
        worker = gui_runtime.AgentRunWorker()
        worker.model_profiles = {"active_profile": None, "profiles": []}
        worker.current_session = SessionSnapshot(
            session_id="session-1",
            thread_id="thread-1",
            checkpoint_backend="sqlite",
            checkpoint_target="demo.sqlite",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            project_path=str(Path.cwd()),
        )
        events = []
        worker.event_emitted.connect(events.append)

        worker.start_run("hello")

        self.assertTrue(any(event.type == "summary_notice" and event.payload.get("kind") == "model_missing" for event in events))

    def test_stop_background_process_denies_external_pid_by_default(self):
        result = process_tools.stop_background_process.invoke({"pid": os.getpid()})
        self.assertIn("ACCESS_DENIED", result)

    def test_stream_processor_ignores_tool_call_without_id(self):
        processor = StreamProcessor()
        processor._remember_tool_call({"name": "broken_tool", "args": {"x": 1}})
        self.assertEqual(processor.tool_buffer, {})

    def test_stream_processor_emits_notice_for_hidden_internal_message(self):
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

        self.assertEqual([event.type for event in events], ["summary_notice"])
        self.assertEqual(events[0].payload.get("kind"), "agent_internal_notice")
        self.assertEqual(events[0].payload.get("message"), "Нужен новый запрос.")
        self.assertEqual(events[0].payload.get("level"), "warning")
        self.assertEqual(processor.full_text, "")
        self.assertEqual(processor.clean_full, "")

    def test_core_gui_shims_reexport_ui_symbols(self):
        from core.gui_runtime import AgentRuntimeController as ShimController
        from core.gui_widgets import ComposerTextEdit as ShimComposer
        from core.stream_processor import StreamProcessor as ShimStreamProcessor
        from core.ui_theme import build_stylesheet as shim_build_stylesheet
        from ui.runtime import AgentRuntimeController as UiController
        from ui.streaming import StreamProcessor as UiStreamProcessor
        from ui.theme import build_stylesheet as ui_build_stylesheet
        from ui.widgets import ComposerTextEdit as UiComposer

        self.assertIs(ShimController, UiController)
        self.assertIs(ShimStreamProcessor, UiStreamProcessor)
        self.assertIs(ShimComposer, UiComposer)
        self.assertIs(shim_build_stylesheet, ui_build_stylesheet)


if __name__ == "__main__":
    unittest.main()
