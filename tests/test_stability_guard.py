import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.config import AgentConfig
from core.nodes import AgentNodes
from core.tool_policy import ToolMetadata
from core.self_correction_engine import build_repair_plan


class _DummyLLM:
    async def ainvoke(self, _context):
        return AIMessage(content="ok")

    def bind_tools(self, _tools):
        return self


class _DummyTool:
    def __init__(self, name: str):
        self.name = name
        self.description = f"Tool {name}"

    async def ainvoke(self, _args):
        return "ok"


class StabilityPolicyTests(unittest.IsolatedAsyncioTestCase):
    def _make_config(self, **overrides) -> AgentConfig:
        defaults = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "PROMPT_PATH": Path(__file__).resolve().parents[1] / "prompt.txt",
            "MODEL_SUPPORTS_TOOLS": True,
            "SELF_CORRECTION_ENABLE_AUTO_REPAIR": True,
            "SELF_CORRECTION_MAX_AUTO_REPAIRS": 2,
            "SELF_CORRECTION_HARD_CEILING": 5,
        }
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def _build_nodes(self, *, tool_metadata=None, **config_overrides) -> AgentNodes:
        config = self._make_config(**config_overrides)
        llm = _DummyLLM()
        tools = [
            _DummyTool("read_file"),
            _DummyTool("find_process_by_port"),
            _DummyTool("edit_file"),
        ]
        return AgentNodes(
            config=config,
            llm=llm,
            tools=tools,
            llm_with_tools=llm,
            tool_metadata=tool_metadata,
        )

    def test_hard_loop_ceiling_matches_max_loops(self):
        nodes = self._build_nodes(MAX_LOOPS=7, SELF_CORRECTION_HARD_CEILING=5)
        self.assertEqual(nodes._hard_loop_ceiling(), 5)

    def test_mutating_tools_require_approval(self):
        nodes = self._build_nodes(
            ENABLE_APPROVALS=True,
            tool_metadata={
                "edit_file": ToolMetadata(name="edit_file", mutating=True),
                "write_file": ToolMetadata(name="write_file", mutating=True),
            },
        )
        self.assertTrue(nodes._tool_requires_approval("edit_file", {"path": "demo.txt"}))
        self.assertTrue(nodes._tool_requires_approval("write_file", {"path": "demo.txt"}))

    def test_cli_exec_approval_depends_on_command_profile(self):
        nodes = self._build_nodes(ENABLE_APPROVALS=True)
        self.assertTrue(nodes._tool_requires_approval("cli_exec", {"command": "Copy-Item a.txt b.txt"}))
        self.assertFalse(nodes._tool_requires_approval("cli_exec", {"command": "Get-ChildItem"}))

    @staticmethod
    def _base_state(task: str) -> dict:
        return {
            "messages": [HumanMessage(content=task), AIMessage(content="Проверяю.")],
            "summary": "",
            "steps": 1,
            "token_usage": {},
            "current_task": task,
            "turn_id": 1,
            "run_id": "run-test",
            "session_id": "session-test",
            "open_tool_issue": None,
            "pending_approval": None,
            "self_correction_retry_count": 0,
            "self_correction_retry_turn_id": 1,
            "self_correction_fingerprint_history": [],
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
        }

    async def test_stability_guard_non_retryable_issue_finishes_with_structured_dead_end(self):
        nodes = self._build_nodes()
        state = self._base_state("исправь файл")
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Unexpected execution failure",
            "tool_names": ["edit_file"],
            "tool_args": {"path": "demo.txt"},
            "source": "tools",
            "error_type": "EXECUTION",
            "fingerprint": "fp-non-retryable",
            "progress_fingerprint": "fp-non-retryable",
            "details": {},
        }

        result = await nodes.stability_guard_node(state)
        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["self_correction_retry_count"], 1)
        self.assertIsNotNone(result["open_tool_issue"])
        self.assertTrue(result["recovery_state"]["strategy_queue"])
        self.assertEqual(result["recovery_state"]["strategy_queue"][0]["strategy"], "llm_replan")

    async def test_stability_guard_retryable_issue_uses_soft_budget_first(self):
        nodes = self._build_nodes()
        state = self._base_state("проверь порт")
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Port must be integer",
            "tool_names": ["find_process_by_port"],
            "tool_args": {"port": "8080"},
            "source": "tools",
            "error_type": "VALIDATION",
            "fingerprint": "fp-retryable",
            "progress_fingerprint": "fp-retryable",
            "details": {},
        }

        first = await nodes.stability_guard_node(state)
        self.assertEqual(first["turn_outcome"], "recover_agent")
        self.assertEqual(first["self_correction_retry_count"], 1)
        self.assertTrue(first["recovery_state"]["strategy_queue"])

        updated_issue = {
            **state["open_tool_issue"],
            "fingerprint": "fp-progress-3",
            "progress_fingerprint": "fp-progress-3",
        }
        repair_plan = build_repair_plan(updated_issue, current_task="проверь порт", max_auto_repairs=2)
        self.assertIsNotNone(repair_plan)
        strategy_id = nodes._repair_plan_strategy_id(repair_plan)  # type: ignore[arg-type]

        state_after_two = dict(state)
        state_after_two["recovery_state"] = {
            "turn_id": 1,
            "active_issue": state["open_tool_issue"],
            "active_strategy": None,
            "strategy_queue": [],
            "attempts_by_strategy": {strategy_id: 1},
            "progress_markers": ["fp-old-1", "fp-old-2"],
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": [],
        }
        state_after_two["open_tool_issue"] = updated_issue
        adaptive = await nodes.stability_guard_node(state_after_two)
        self.assertEqual(adaptive["turn_outcome"], "recover_agent")
        self.assertGreaterEqual(adaptive["self_correction_retry_count"], 1)
        self.assertEqual(adaptive["recovery_state"]["strategy_queue"][0]["strategy"], "llm_replan")

    async def test_stability_guard_retries_edit_file_match_failure(self):
        nodes = self._build_nodes()
        state = self._base_state("исправь файл")
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Could not find a match for 'old_string'.",
            "tool_names": ["edit_file"],
            "tool_args": {"path": "demo.txt", "old_string": "bad", "new_string": "good"},
            "source": "tools",
            "error_type": "VALIDATION",
            "fingerprint": "fp-edit-miss",
            "progress_fingerprint": "fp-edit-miss",
            "details": {},
        }

        result = await nodes.stability_guard_node(state)
        self.assertEqual(result["turn_outcome"], "recover_agent")
        self.assertEqual(result["self_correction_retry_count"], 1)
        self.assertIn("read_file", str(result["recovery_state"]["strategy_queue"][0]["suggested_tool_name"]))

    async def test_stability_guard_retries_on_repeated_fingerprint_before_loop_budget(self):
        nodes = self._build_nodes()
        state = self._base_state("проверь порт")
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Port must be integer",
            "tool_names": ["find_process_by_port"],
            "tool_args": {"port": "8080"},
            "source": "tools",
            "error_type": "VALIDATION",
            "fingerprint": "fp-repeat",
            "progress_fingerprint": "fp-repeat",
            "details": {},
        }
        repair_plan = build_repair_plan(state["open_tool_issue"], current_task="проверь порт", max_auto_repairs=2)
        self.assertIsNotNone(repair_plan)
        strategy_id = nodes._repair_plan_strategy_id(repair_plan)  # type: ignore[arg-type]
        state["recovery_state"] = {
            "turn_id": 1,
            "active_issue": state["open_tool_issue"],
            "active_strategy": None,
            "strategy_queue": [],
            "attempts_by_strategy": {strategy_id: 1},
            "progress_markers": ["fp-repeat"],
            "last_successful_evidence": "",
            "external_blocker": None,
            "llm_replan_attempted_for": ["fp-repeat"],
        }

        result = await nodes.stability_guard_node(state)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "recovery_stagnated")
        self.assertGreaterEqual(result["self_correction_retry_count"], 1)
        self.assertIsNone(result["open_tool_issue"])
        self.assertFalse(result["recovery_state"]["strategy_queue"])
        self.assertIn("стагнац", str(result["messages"][-1].content).lower())

    async def test_stability_guard_finishes_when_self_correction_hard_ceiling_is_reached(self):
        nodes = self._build_nodes(MAX_LOOPS=20, SELF_CORRECTION_HARD_CEILING=2)
        state = self._base_state("исправь файл")
        state["self_correction_retry_count"] = 2
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "protocol_error",
            "summary": "Model kept replying with prose instead of tools.",
            "tool_names": ["edit_file"],
            "tool_args": {"path": "demo.txt"},
            "source": "agent",
            "error_type": "PROTOCOL",
            "fingerprint": "fp-hard-cap",
            "progress_fingerprint": "fp-hard-cap",
            "details": {"protocol_reason": "action_requires_tools"},
        }

        result = await nodes.stability_guard_node(state)

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "recovery_stagnated")
        self.assertIsNone(result["open_tool_issue"])
        self.assertFalse(result["recovery_state"]["strategy_queue"])

    async def test_stability_guard_stops_on_workspace_boundary_violation(self):
        nodes = self._build_nodes()
        state = self._base_state("исправь файл")
        state["open_tool_issue"] = {
            "turn_id": 1,
            "kind": "tool_error",
            "summary": "Access denied outside workspace",
            "tool_names": ["edit_file"],
            "tool_args": {"path": "..\\outside.txt", "old_string": "a", "new_string": "b"},
            "source": "tools",
            "error_type": "ACCESS_DENIED",
            "fingerprint": "fp-boundary",
            "progress_fingerprint": "fp-boundary",
            "details": {"safety_violation": True},
        }

        result = await nodes.stability_guard_node(state)
        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertIsNone(result["open_tool_issue"])
        self.assertIn("рабоч", str(result["messages"][-1].content).lower())

    async def test_stability_guard_continues_on_repeated_identical_successful_tool_results_before_loop_budget(self):
        nodes = self._build_nodes()
        tool_args = {"path": "demo.txt"}
        state = self._base_state("проверь файл")
        state["messages"] = [
            HumanMessage(content="проверь файл"),
            AIMessage(content="", tool_calls=[{"id": "tc-1", "name": "read_file", "args": tool_args}]),
            ToolMessage(content="ok", tool_call_id="tc-1", name="read_file", additional_kwargs={"tool_args": tool_args}),
            AIMessage(content="", tool_calls=[{"id": "tc-2", "name": "read_file", "args": tool_args}]),
            ToolMessage(content="ok", tool_call_id="tc-2", name="read_file", additional_kwargs={"tool_args": tool_args}),
            AIMessage(content="", tool_calls=[{"id": "tc-3", "name": "read_file", "args": tool_args}]),
            ToolMessage(content="ok", tool_call_id="tc-3", name="read_file", additional_kwargs={"tool_args": tool_args}),
        ]

        result = await nodes.stability_guard_node(state)

        self.assertEqual(result["turn_outcome"], "continue_agent")
        self.assertEqual(result["completion_reason"], "successful_tool_stagnation")
        self.assertEqual(result["successful_tool_repeat_count"], 3)
        self.assertEqual(result["successful_tool_name"], "read_file")
        self.assertIsNone(result["open_tool_issue"])

    async def test_stability_guard_resets_retry_state_after_successful_tool_result(self):
        nodes = self._build_nodes()
        tool_args = {"path": "demo.txt"}
        state = self._base_state("проверь файл")
        state["self_correction_retry_count"] = 2
        state["self_correction_fingerprint_history"] = ["fp-1", "fp-2"]
        state["recovery_state"]["attempts_by_strategy"] = {"read_file:fp-2": 2}
        state["recovery_state"]["progress_markers"] = ["fp-2"]
        state["recovery_state"]["llm_replan_attempted_for"] = ["fp-2"]
        state["messages"] = [
            HumanMessage(content="проверь файл"),
            AIMessage(content="", tool_calls=[{"id": "tc-ok", "name": "read_file", "args": tool_args}]),
            ToolMessage(content="ok", tool_call_id="tc-ok", name="read_file", additional_kwargs={"tool_args": tool_args}),
        ]

        result = await nodes.stability_guard_node(state)

        self.assertEqual(result["turn_outcome"], "continue_agent")
        self.assertEqual(result["completion_reason"], "tool_result_ready_for_agent")
        self.assertEqual(result["self_correction_retry_count"], 0)
        self.assertEqual(result["self_correction_fingerprint_history"], [])
        self.assertEqual(result["recovery_state"]["attempts_by_strategy"], {})
        self.assertEqual(result["recovery_state"]["progress_markers"], [])
        self.assertEqual(result["recovery_state"]["llm_replan_attempted_for"], [])
        self.assertIsNone(result["open_tool_issue"])

    async def test_stability_guard_loop_budget_handoff_uses_soft_specific_notice(self):
        nodes = self._build_nodes(MAX_LOOPS=2)
        state = self._base_state("проверь файл")
        state["steps"] = 2
        state["messages"] = [
            HumanMessage(content="проверь файл"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc-loop", "name": "read_file", "args": {"path": "demo.txt"}}],
                id="ai-loop",
            ),
        ]

        result = await nodes.stability_guard_node(state)

        self.assertEqual(result["turn_outcome"], "finish_turn")
        self.assertEqual(result["completion_reason"], "loop_budget_exhausted_pending_tool_call")
        internal = result["messages"][-1].additional_kwargs["agent_internal"]
        self.assertIn("внутренний лимит", str(internal.get("ui_notice", "")).lower())


if __name__ == "__main__":
    unittest.main()
