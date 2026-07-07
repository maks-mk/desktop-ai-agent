"""Tests for Plan Mode feature.

Covers:
- normalize_request_payload extracts plan_mode
- build_initial_state sets turn_mode="plan" when plan_mode=True
- RuntimePromptPolicyBuilder injects PLAN_MODE_TEXT when plan_mode=True
- ContextBuilder.build passes plan_mode from state["turn_mode"]
- _active_tools_for_turn filters to read-only tools in plan mode
- _sanitize_user_input_tool_calls keeps the one-request-per-turn guard outside auto plan approval
- _build_agent_result passes plan_mode through
"""

from __future__ import annotations

import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.config import AgentConfig
from core.context_builder import ContextBuilder
from core.multimodal import normalize_request_payload
from core.nodes import AgentNodes
from core.runtime_prompt_policy import RuntimePromptContext, RuntimePromptPolicyBuilder
from core.tool_policy import ToolMetadata
from core.turn_outcomes import TURN_OUTCOME_FINISH_TURN, TURN_OUTCOME_RUN_TOOLS, TURN_OUTCOME_RECOVER_AGENT
from ui.runtime import build_initial_state


class PlanModePayloadTests(unittest.TestCase):
    """Tests for normalize_request_payload and build_initial_state with plan_mode."""

    def test_normalize_request_payload_extracts_plan_mode_true(self):
        payload = {"text": "Plan a refactor", "attachments": [], "plan_mode": True}
        result = normalize_request_payload(payload)
        self.assertTrue(result["plan_mode"])
        self.assertEqual(result["text"], "Plan a refactor")

    def test_normalize_request_payload_extracts_plan_mode_false(self):
        payload = {"text": "Do something", "attachments": []}
        result = normalize_request_payload(payload)
        self.assertFalse(result["plan_mode"])

    def test_normalize_request_payload_plan_mode_false_for_non_dict(self):
        result = normalize_request_payload("just text")
        self.assertFalse(result["plan_mode"])
        self.assertEqual(result["text"], "just text")

    def test_build_initial_state_sets_turn_mode_plan(self):
        payload = {"text": "Plan a task", "attachments": [], "plan_mode": True}
        state = build_initial_state(payload, session_id="s1")
        self.assertEqual(state["turn_mode"], "plan")

    def test_build_initial_state_sets_turn_mode_chat_by_default(self):
        payload = {"text": "Do a task", "attachments": []}
        state = build_initial_state(payload, session_id="s1")
        self.assertEqual(state["turn_mode"], "chat")


class PlanModePromptPolicyTests(unittest.TestCase):
    """Tests for RuntimePromptPolicyBuilder with plan_mode."""

    def _make_config(self) -> AgentConfig:
        defaults = {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "safety_mode": "default",
            "max_loops": 10,
        }
        return AgentConfig(**defaults)

    def test_plan_mode_text_injected_when_plan_mode_true(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())
        context = RuntimePromptContext(
            current_task="Plan a refactor",
            tools_available=True,
            active_tool_names=("read_file", "request_user_input"),
            user_choice_locked=False,
            plan_mode=True,
        )
        messages = builder.build_messages(context)
        system_texts = [str(m.content) for m in messages if isinstance(m, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("PLAN MODE IS ACTIVE", joined)
        self.assertIn("request_user_input", joined)
        self.assertIn("decision-complete", joined)
        self.assertIn("never for plan approval", joined)
        self.assertNotIn("Что сделать с этим планом?", joined)
        self.assertNotIn("Да, реализовать", joined)
        self.assertNotIn("Нет, отказаться от реализации", joined)
        self.assertNotIn("Внести правки/дополнения в план", joined)

    def test_plan_mode_text_not_injected_when_plan_mode_false(self):
        builder = RuntimePromptPolicyBuilder(config=self._make_config())
        context = RuntimePromptContext(
            current_task="Do a task",
            tools_available=True,
            active_tool_names=("read_file",),
            user_choice_locked=False,
            plan_mode=False,
        )
        messages = builder.build_messages(context)
        system_texts = [str(m.content) for m in messages if isinstance(m, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertNotIn("PLAN MODE IS ACTIVE", joined)


class PlanModeContextBuilderTests(unittest.TestCase):
    """Tests for ContextBuilder.build with turn_mode=plan in state."""

    def _make_config(self) -> AgentConfig:
        defaults = {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "safety_mode": "default",
            "max_loops": 10,
        }
        return AgentConfig(**defaults)

    def _make_builder(self) -> ContextBuilder:
        import re

        return ContextBuilder(
            config=self._make_config(),
            prompt_loader=lambda: "Editable prompt only",
            is_internal_retry=lambda _msg: False,
            log_run_event=lambda *_args, **_kwargs: None,
            recovery_message_builder=lambda _state: None,
            provider_safe_tool_call_id_re=re.compile(r"^[A-Za-z0-9]{9}$"),
        )

    def test_context_builder_injects_plan_mode_from_state(self):
        builder = self._make_builder()
        state = {
            "messages": [HumanMessage(content="Plan a refactor")],
            "turn_mode": "plan",
        }
        context = builder.build(
            state["messages"],
            state,
            summary="",
            current_task="Plan a refactor",
            tools_available=True,
            active_tool_names=["read_file", "request_user_input"],
            open_tool_issue=None,
            recovery_state=None,
        )
        system_texts = [str(m.content) for m in context if isinstance(m, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertIn("PLAN MODE IS ACTIVE", joined)

    def test_context_builder_no_plan_mode_when_turn_mode_chat(self):
        builder = self._make_builder()
        state = {
            "messages": [HumanMessage(content="Do a task")],
            "turn_mode": "chat",
        }
        context = builder.build(
            state["messages"],
            state,
            summary="",
            current_task="Do a task",
            tools_available=True,
            active_tool_names=["read_file"],
            open_tool_issue=None,
            recovery_state=None,
        )
        system_texts = [str(m.content) for m in context if isinstance(m, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertNotIn("PLAN MODE IS ACTIVE", joined)

    def test_approved_plan_execution_keeps_normal_tool_prompt_contract(self):
        builder = self._make_builder()
        state = {
            "messages": [HumanMessage(content="Implement approved plan")],
            "turn_mode": "chat",
            "plan_status": "executing",
            "active_plan_step_id": "step-1",
            "current_plan": {
                "id": "plan-1",
                "version": 1,
                "summary": "Update the UI",
                "status": "executing",
                "active_step_id": "step-1",
                "risks": [],
                "assumptions": [],
                "estimated_tools": [],
                "estimated_files": [],
                "complexity": "medium",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "Patch UI",
                        "description": "Change the inspector behavior.",
                        "status": "in_progress",
                    }
                ],
            },
        }
        context = builder.build(
            state["messages"],
            state,
            summary="",
            current_task="Implement approved plan",
            tools_available=True,
            active_tool_names=["read_file", "edit_file", "cli_exec"],
            open_tool_issue=None,
            recovery_state=None,
        )
        system_texts = [str(m.content) for m in context if isinstance(m, SystemMessage)]
        joined = "\n".join(system_texts)
        self.assertNotIn("PLAN MODE IS ACTIVE", joined)
        self.assertIn("TOOL INTENT REQUIREMENT", joined)
        self.assertIn("Execute only the active step below", joined)
        self.assertIn("exactly one short line in Russian", joined)
        self.assertIn("Do not write reports", joined)
        self.assertIn("no more than 140 characters", joined)
        self.assertIn("<!--plan-step-complete:step-1-->", joined)
        self.assertIn("Active step title: Patch UI", joined)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def ainvoke(self, context):
        self.invocations.append(context)
        if not self.responses:
            return AIMessage(content="Fallback response.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


class FakeSchemaTool(FakeTool):
    def __init__(self, name, result, schema):
        super().__init__(name, result)
        self._schema = schema

    def get_input_schema(self):
        return self._schema


class PlanModeAgentNodeTests(unittest.IsolatedAsyncioTestCase):
    """Tests for _active_tools_for_turn and _sanitize_user_input_tool_calls in plan mode."""

    def _make_config(self) -> AgentConfig:
        defaults = {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "safety_mode": "default",
            "max_loops": 10,
        }
        return AgentConfig(**defaults)

    def _initial_state(self, task="Plan a refactor", session_id="session-test", run_id="run-test"):
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
        }

    async def test_active_tools_for_turn_filters_to_read_only_in_plan_mode(self):
        """In plan mode, only read-only tools + request_user_input should be active."""
        from core.nodes import AgentNodes

        read_tool = FakeTool("read_file", "ok")
        edit_tool = FakeTool("edit_file", "ok")
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[read_tool, edit_tool, user_input_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "edit_file": ToolMetadata(name="edit_file", read_only=False, mutating=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state()
        state["turn_mode"] = "plan"
        active_tools, active_names = nodes._active_tools_for_turn(state, state["messages"])
        active_set = {nodes._normalize_tool_name(n) for n in active_names}
        self.assertIn("read_file", active_set)
        self.assertIn("request_user_input", active_set)
        self.assertNotIn("edit_file", active_set)

    async def test_active_tools_for_turn_allows_all_in_chat_mode(self):
        """In chat mode, all tools should be active (no plan-mode filtering)."""
        from core.nodes import AgentNodes

        read_tool = FakeTool("read_file", "ok")
        edit_tool = FakeTool("edit_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[read_tool, edit_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "edit_file": ToolMetadata(name="edit_file", read_only=False, mutating=True),
            },
        )
        state = self._initial_state()
        state["turn_mode"] = "chat"
        active_tools, active_names = nodes._active_tools_for_turn(state, state["messages"])
        active_set = {nodes._normalize_tool_name(n) for n in active_names}
        self.assertIn("read_file", active_set)
        self.assertIn("edit_file", active_set)

    async def test_sanitize_user_input_blocks_second_call_in_plan_mode(self):
        """In plan mode, a second request_user_input in the same turn should be blocked."""
        from core.nodes import AgentNodes

        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[user_input_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        messages = [
            HumanMessage(content="Plan a task"),
            AIMessage(
                content="First question.",
                tool_calls=[
                    {
                        "name": "request_user_input",
                        "args": {"question": "Q1", "options": ["A", "B"]},
                        "id": "tc-1",
                    }
                ],
            ),
            ToolMessage(content="A", tool_call_id="tc-1", name="request_user_input"),
        ]
        new_calls = [
            {
                "name": "request_user_input",
                "args": {"question": "Q2", "options": ["C", "D"]},
                "id": "tc-2",
            }
        ]
        sanitized, error, retry = nodes._sanitize_user_input_tool_calls(
            new_calls, messages, plan_mode=True
        )
        self.assertEqual(len(sanitized), 0)
        self.assertIn("at most once per user turn", error)
        self.assertTrue(retry)

    async def test_sanitize_user_input_blocks_second_call_in_chat_mode(self):
        """In chat mode, a second request_user_input in the same turn should be blocked."""
        from core.nodes import AgentNodes

        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[user_input_tool],
            llm_with_tools=FakeLLM([]),
            tool_metadata={
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        messages = [
            HumanMessage(content="Do a task"),
            AIMessage(
                content="First question.",
                tool_calls=[
                    {
                        "name": "request_user_input",
                        "args": {"question": "Q1", "options": ["A", "B"]},
                        "id": "tc-1",
                    }
                ],
            ),
            ToolMessage(content="A", tool_call_id="tc-1", name="request_user_input"),
        ]
        new_calls = [
            {
                "name": "request_user_input",
                "args": {"question": "Q2", "options": ["C", "D"]},
                "id": "tc-2",
            }
        ]
        # In chat mode, the "at most once" guard should trigger.
        sanitized, error, retry = nodes._sanitize_user_input_tool_calls(
            new_calls, messages, plan_mode=False
        )
        self.assertEqual(len(sanitized), 0)
        self.assertIn("at most once per user turn", error)
        self.assertTrue(retry)

    async def test_agent_node_plan_mode_binds_read_only_tools_with_cli_exec_for_rg(self):
        """Full agent_node call in plan mode exposes cli_exec so rg can be used for inspection."""
        from core.nodes import AgentNodes

        agent_llm = FakeBindableLLM([AIMessage(content="I will plan this.")])
        read_tool = FakeTool("read_file", "ok")
        edit_tool = FakeTool("edit_file", "ok")
        cli_tool = FakeTool("cli_exec", "ok")
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[read_tool, edit_tool, cli_tool, user_input_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "edit_file": ToolMetadata(name="edit_file", read_only=False, mutating=True),
                "cli_exec": ToolMetadata(name="cli_exec", read_only=False, mutating=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Plan a refactor")
        state["turn_mode"] = "plan"
        await nodes.agent_node(state)
        self.assertTrue(agent_llm.bound_tool_name_batches)
        bound_names = agent_llm.bound_tool_name_batches[-1]
        bound_set = {nodes._normalize_tool_name(n) for n in bound_names}
        self.assertIn("read_file", bound_set)
        self.assertIn("request_user_input", bound_set)
        self.assertIn("cli_exec", bound_set)
        self.assertNotIn("edit_file", bound_set)

    async def test_agent_node_plan_mode_allows_blocking_clarification(self):
        """In plan mode, the agent may ask one blocking clarification."""
        from core.nodes import AgentNodes

        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="I need one missing decision before planning.",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "Which package manager should be used?",
                                "options": ["npm", "pnpm", "yarn"],
                                "recommended": "npm",
                            },
                            "id": "tc-plan-1",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Plan a refactor")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["name"], "request_user_input")

    async def test_agent_node_plan_mode_adds_plan_approval_request(self):
        """Plan mode appends a fixed approval request after a prose plan."""
        from core.nodes import AgentNodes

        agent_llm = FakeBindableLLM([AIMessage(content="План реализации:\n1. Сделать структуру.")])
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Create a game")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        tool_call = response.tool_calls[0]
        self.assertEqual(tool_call["name"], "request_user_input")
        self.assertEqual(tool_call["args"]["question"], "Что сделать с этим планом?")
        self.assertEqual(
            tool_call["args"]["options"],
            [
                "Да, реализовать",
                "Нет, отказаться от реализации",
                "Внести правки/дополнения в план",
            ],
        )
        self.assertEqual(tool_call["args"]["recommended"], "Да, реализовать")
        self.assertEqual(tool_call["args"]["choice_type"], "plan_approval")
        self.assertNotIn("current_plan", result)
        self.assertNotIn("plan_status", result)




    async def test_build_agent_result_drops_tool_calls_when_tools_disabled(self):
        from core.nodes import AgentNodes

        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=FakeLLM([]),
            tool_metadata={},
        )
        response = AIMessage(
            content="I would inspect files if tools were enabled.",
            tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "tc-read-1"}],
        )

        result = nodes._build_agent_result(
            response,
            current_task="Plan a refactor",
            tools_available=False,
            turn_id=1,
            messages=[HumanMessage(content="Plan a refactor")],
            allowed_tool_names=[],
            plan_mode=True,
            legacy_plan_approval=False,
        )

        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_FINISH_TURN)
        self.assertFalse(result["messages"][-1].tool_calls)

    async def test_agent_node_plan_mode_drops_mutating_tool_calls(self):
        """In plan mode, mutating tool calls are dropped before approval/tools routing."""
        from core.nodes import AgentNodes

        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="Here is the plan. I should not create files yet.",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "index.html", "content": "<html></html>"},
                            "id": "tc-write-1",
                        }
                    ],
                )
            ]
        )
        write_tool = FakeTool("write_file", "ok")
        read_tool = FakeTool("read_file", "ok")
        user_input_tool = FakeTool("request_user_input", "ignored")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[write_tool, read_tool, user_input_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "write_file": ToolMetadata(name="write_file", read_only=False, mutating=True),
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Create a game")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["name"], "request_user_input")
        self.assertEqual(response.tool_calls[0]["args"]["question"], "Что сделать с этим планом?")
        self.assertEqual(response.tool_calls[0]["args"]["choice_type"], "plan_approval")




    async def test_plan_select_step_blocks_unfinished_plan_without_pending_step(self):
        from core.nodes import AgentNodes

        nodes = AgentNodes(
            config=self._make_config(),
            llm=FakeLLM([]),
            tools=[],
            llm_with_tools=FakeLLM([]),
            tool_metadata={},
        )
        state = self._initial_state("Implement approved plan")
        state.update(
            {
                "plan_graph_active": True,
                "plan_status": "executing",
                "current_plan": {
                    "id": "plan-1",
                    "version": 1,
                    "summary": "Do work",
                    "steps": [
                        {
                            "id": "s1",
                            "title": "Patch",
                            "description": "Patch code",
                            "status": "blocked",
                        }
                    ],
                    "risks": [],
                    "assumptions": [],
                    "estimated_tools": [],
                    "estimated_files": [],
                    "complexity": "medium",
                    "status": "executing",
                    "active_step_id": "",
                },
                "active_plan_step_id": "",
            }
        )

        result = await nodes.plan_select_step_node(state)

        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RECOVER_AGENT)
        self.assertEqual(result["plan_status"], "replan_pending")
        self.assertEqual(result["current_plan"]["status"], "replan_pending")
        self.assertTrue(result["has_protocol_error"])
        self.assertIn("unfinished steps", result["last_tool_error"])
        messages = result.get("messages") or []
        self.assertFalse(any(getattr(message, "content", "") == "План выполнен." for message in messages))


class PlanModeApprovalGuardTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the plan-approval guard: approval request without plan text is dropped."""

    def _make_config(self) -> AgentConfig:
        defaults = {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "safety_mode": "default",
            "max_loops": 10,
        }
        return AgentConfig(**defaults)

    def _initial_state(self, task="Plan a refactor", session_id="session-test", run_id="run-test"):
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
        }

    async def test_agent_node_plan_mode_drops_approval_without_plan(self):
        """In plan mode, an approval request with empty/short content is dropped
        and the agent is sent to recovery so it writes the plan first."""
        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "Что сделать с этим планом?",
                                "options": [
                                    "Да, реализовать",
                                    "Нет, отказаться от реализации",
                                    "Внести правки/дополнения в план",
                                ],
                                "recommended": "Да, реализовать",
                            },
                            "id": "tc-approval-1",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Create a game")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RECOVER_AGENT)
        self.assertTrue(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 0)

    async def test_agent_node_plan_mode_keeps_approval_with_plan_text(self):
        """In plan mode, an approval request with sufficient plan text passes through."""
        plan_text = "План реализации:\n1. Создать структуру проекта.\n2. Добавить основные модули.\n3. Настроить тесты."
        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content=plan_text,
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "Что сделать с этим планом?",
                                "options": [
                                    "Да, реализовать",
                                    "Нет, отказаться от реализации",
                                    "Внести правки/дополнения в план",
                                ],
                                "recommended": "Да, реализовать",
                            },
                            "id": "tc-approval-2",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Create a game")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["name"], "request_user_input")

    async def test_agent_node_plan_mode_keeps_clarification_without_plan(self):
        """In plan mode, a non-approval clarification question passes even with short content."""
        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="Need one detail.",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "Which framework?",
                                "options": ["FastAPI", "Flask", "Django"],
                                "recommended": "FastAPI",
                            },
                            "id": "tc-clarify-1",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Create a game")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)

    async def test_agent_node_plan_mode_keeps_read_only_tool_calls(self):
        """In plan mode, read-only tool calls (e.g. read_file) must NOT be dropped.
        Only request_user_input was kept before, causing the model to skip research
        and jump straight to the approval request."""
        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="I will inspect the codebase first.",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "README.md"},
                            "id": "tc-read-1",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        read_tool = FakeTool("read_file", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, read_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "read_file": ToolMetadata(name="read_file", read_only=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Plan a refactor")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(
            nodes._normalize_tool_name(response.tool_calls[0]["name"]),
            "read_file",
        )
    async def test_agent_node_plan_mode_keeps_rg_cli_exec_tool_calls(self):
        """In plan mode, cli_exec rg calls are allowed because they are inspect-only."""
        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="I will search the codebase with rg before planning.",
                    tool_calls=[
                        {
                            "name": "cli_exec",
                            "args": {"command": "rg -n \"with_structured_output\" core tests"},
                            "id": "tc-rg-1",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        cli_tool = FakeTool("cli_exec", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, cli_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "cli_exec": ToolMetadata(name="cli_exec", read_only=False, mutating=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Plan a refactor")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["name"], "cli_exec")
        self.assertEqual(response.tool_calls[0]["args"]["command"], "rg -n \"with_structured_output\" core tests")

    async def test_agent_node_plan_mode_drops_mutating_cli_exec_tool_calls(self):
        """In plan mode, cli_exec remains blocked for mutating shell commands."""
        agent_llm = FakeBindableLLM(
            [
                AIMessage(
                    content="Here is the plan. I should not delete files yet.",
                    tool_calls=[
                        {
                            "name": "cli_exec",
                            "args": {"command": "Remove-Item README.md"},
                            "id": "tc-shell-delete-1",
                        }
                    ],
                )
            ]
        )
        user_input_tool = FakeTool("request_user_input", "ignored")
        cli_tool = FakeTool("cli_exec", "ok")
        nodes = AgentNodes(
            config=self._make_config(),
            llm=agent_llm,
            tools=[user_input_tool, cli_tool],
            llm_with_tools=agent_llm,
            tool_metadata={
                "cli_exec": ToolMetadata(name="cli_exec", read_only=False, mutating=True),
                "request_user_input": ToolMetadata(name="request_user_input", read_only=True),
            },
        )
        state = self._initial_state("Plan a refactor")
        state["turn_mode"] = "plan"
        result = await nodes.agent_node(state)
        self.assertEqual(result["turn_outcome"], TURN_OUTCOME_RUN_TOOLS)
        self.assertFalse(result["has_protocol_error"])
        response = result["messages"][-1]
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0]["name"], "request_user_input")
        self.assertEqual(response.tool_calls[0]["args"]["question"], "Что сделать с этим планом?")
        self.assertEqual(response.tool_calls[0]["args"]["choice_type"], "plan_approval")


class PlanModeApprovalResumeTests(unittest.IsolatedAsyncioTestCase):
    """Tests that resuming a plan approval switches turn_mode to 'chat'."""

    async def test_resume_user_choice_approve_switches_turn_mode_to_chat(self):
        """When the user chooses 'Да, реализовать', the resume Command must
        include update={'turn_mode': 'chat'} so the agent gains mutating tools."""
        from langgraph.types import Command

        from ui.runtime_worker import AgentRunWorker

        worker = AgentRunWorker.__new__(AgentRunWorker)
        worker._awaiting_approval = True
        worker._awaiting_interrupt_kind = "user_choice"
        worker._pending_user_choice_type = "plan_approval"
        captured: list[tuple[Command, dict]] = []

        async def fake_run_graph_payload(payload, **kwargs):
            captured.append((payload, kwargs))

        async def fake_emit(*args, **kwargs):
            pass


        worker._run_graph_payload = fake_run_graph_payload
        worker.event_emitted = mock.MagicMock()
        worker.event_emitted.emit = lambda *a, **k: None

        await worker._resume_user_choice_async("Да, реализовать")

        self.assertEqual(len(captured), 1)
        cmd, kwargs = captured[0]
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.update, {"turn_mode": "chat", "plan_status": "approved"})
        self.assertEqual(cmd.resume, {"choice_type": "plan_approval", "choice": "Да, реализовать"})
        self.assertTrue(kwargs["plan_execution_hint"])


    async def test_resume_user_choice_reject_keeps_resume_only(self):
        """When the user chooses 'Нет, отказаться от реализации', no turn_mode
        update is sent — only the resume value."""
        from langgraph.types import Command

        from ui.runtime_worker import AgentRunWorker

        worker = AgentRunWorker.__new__(AgentRunWorker)
        worker._awaiting_approval = True
        worker._awaiting_interrupt_kind = "user_choice"
        worker._pending_user_choice_type = "plan_approval"
        captured: list[Command] = []

        async def fake_run_graph_payload(payload, **kwargs):
            captured.append(payload)

        worker._run_graph_payload = fake_run_graph_payload
        worker.event_emitted = mock.MagicMock()
        worker.event_emitted.emit = lambda *a, **k: None

        await worker._resume_user_choice_async("Нет, отказаться от реализации")


        self.assertEqual(len(captured), 1)
        cmd = captured[0]
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.update, {"plan_status": "rejected"})
        self.assertEqual(cmd.resume, {"choice_type": "plan_approval", "choice": "Нет, отказаться от реализации"})


    async def test_resume_user_choice_edit_keeps_resume_only(self):
        """When the user chooses 'Внести правки/дополнения в план', no turn_mode
        update is sent — only the resume value."""
        from langgraph.types import Command

        from ui.runtime_worker import AgentRunWorker

        worker = AgentRunWorker.__new__(AgentRunWorker)
        worker._awaiting_approval = True
        worker._awaiting_interrupt_kind = "user_choice"
        worker._pending_user_choice_type = "plan_approval"
        captured: list[Command] = []

        async def fake_run_graph_payload(payload, **kwargs):
            captured.append(payload)

        worker._run_graph_payload = fake_run_graph_payload
        worker.event_emitted = mock.MagicMock()
        worker.event_emitted.emit = lambda *a, **k: None

        await worker._resume_user_choice_async("Внести правки/дополнения в план")


        self.assertEqual(len(captured), 1)
        cmd = captured[0]
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.update, {"plan_status": "needs_changes"})
        self.assertEqual(cmd.resume, {"choice_type": "plan_approval", "choice": "Внести правки/дополнения в план"})


    async def test_resume_plan_replan_resumes_with_rebuild_choice(self):
        from langgraph.types import Command

        from ui.runtime_worker import AgentRunWorker

        worker = AgentRunWorker.__new__(AgentRunWorker)
        worker._pending_user_choice_type = "plan_replan"
        captured: list[tuple[Command, dict]] = []

        async def fake_run_graph_payload(payload, **kwargs):
            captured.append((payload, kwargs))

        worker._run_graph_payload = fake_run_graph_payload
        worker.event_emitted = mock.MagicMock()
        worker.event_emitted.emit = lambda *a, **k: None

        await worker._resume_user_choice_async("rebuild")

        self.assertEqual(len(captured), 1)
        cmd, kwargs = captured[0]
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.resume, {"choice_type": "plan_replan", "choice": "rebuild", "feedback": "rebuild"})
        self.assertTrue(kwargs["plan_execution_hint"])



if __name__ == "__main__":
    unittest.main()
