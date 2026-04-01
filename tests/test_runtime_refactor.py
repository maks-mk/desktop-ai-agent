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
from core.gui_runtime import append_project_label, build_transcript_payload, generate_chat_title, short_project_label
from core.nodes import AgentNodes
from core.run_logger import JsonlRunLogger
from core.session_store import SessionStore
from core.stream_processor import StreamProcessor
from core.tool_policy import ToolMetadata
from tools import process_tools


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
            "critic_status": "",
            "critic_source": "",
            "critic_feedback": "",
            "turn_outcome": "",
            "retry_instruction": "",
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
            self.assertTrue(any(isinstance(msg, AIMessage) and msg.content == "Первый ответ." for msg in saved_messages))
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
        self.assertIn("you chose No", str(resumed["messages"][-1].content))
        self.assertEqual(resumed["critic_status"], "FINISHED")
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
                    AIMessage(content="Okay, I did not do that because you chose No. Tell me what you want to do instead."),
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
        self.assertIn("you chose No", str(resumed["messages"][-1].content))

    async def test_approval_rejection_finishes_turn_when_critic_verdict_is_malformed(self):
        config = self._make_config(ENABLE_APPROVALS=True)
        tool = FakeTool("danger_tool", "Изменение применено.")
        critic_llm = FakeLLM([AIMessage(content="wait maybe")])
        agent_llm = ProviderSafeFakeLLM(
            [
                AIMessage(content="", tool_calls=[{"name": "danger_tool", "args": {"action": "apply"}, "id": "tc-mal"}]),
                AIMessage(content="Не сделал, потому что вы выбрали Нет. Жду следующую инструкцию."),
            ]
        )
        nodes = AgentNodes(
            config=config,
            llm=critic_llm,
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
        self.assertEqual(len(agent_llm.invocations), 2)
        self.assertEqual(len(critic_llm.invocations), 1)
        self.assertIn("вы выбрали Нет", str(resumed["messages"][-1].content))

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
            llm=FakeLLM([]),
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

    def test_stop_background_process_denies_external_pid_by_default(self):
        result = process_tools.stop_background_process.invoke({"pid": os.getpid()})
        self.assertIn("ACCESS_DENIED", result)

    def test_stream_processor_ignores_tool_call_without_id(self):
        processor = StreamProcessor()
        processor._remember_tool_call({"name": "broken_tool", "args": {"x": 1}})
        self.assertEqual(processor.tool_buffer, {})


if __name__ == "__main__":
    unittest.main()
