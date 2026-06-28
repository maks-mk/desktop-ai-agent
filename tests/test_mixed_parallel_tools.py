"""Tests for mixed-mode parallel/sequential tool batch execution (Level 1).

Verifies that ``ToolBatchCoordinator`` correctly partitions tool calls into
parallel-safe and sequential groups, runs them concurrently vs. one-by-one,
and re-assembles ``ToolMessage`` results in the original ``tool_calls`` order.
"""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from langchain_core.messages import AIMessage, ToolMessage

from core.node_orchestrators import ToolBatchCoordinator
from core.nodes import AgentNodes
from core.tool_policy import ToolMetadata


def _tc(name: str, tc_id: str, args: dict | None = None) -> dict[str, Any]:
    return {"name": name, "id": tc_id, "args": args or {}}


def _make_tool_metadata(name: str, *, read_only: bool = True) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        read_only=read_only,
        mutating=not read_only,
        destructive=False,
        requires_approval=False,
        networked=False,
        source="local",
    )


class _FakeOwner:
    """Minimal stand-in for ``AgentNodes`` with just the methods the coordinator calls."""

    PARALLEL_SAFE_TOOL_NAMES = AgentNodes.PARALLEL_SAFE_TOOL_NAMES
    READ_ONLY_LOOP_TOLERANT_TOOL_NAMES = AgentNodes.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES

    def __init__(self, tool_metadata: dict[str, ToolMetadata]) -> None:
        self._metadata = tool_metadata
        self._all_tool_names = tuple(tool_metadata.keys())
        self.config = SimpleNamespace(
            model_supports_tools=True,
            effective_tool_loop_window=0,
            effective_tool_loop_limit_readonly=99,
            effective_tool_loop_limit_mutating=99,
        )
        self.recovery_manager = SimpleNamespace(
            reset_after_success=lambda state, current_turn_id, successful_evidence: state,
        )
        self.tool_executor = SimpleNamespace(
            handle_result=self._fake_handle_result,
        )
        self._process_delay = 0.05
        self._call_log: list[str] = []

    # --- Methods called by ToolBatchCoordinator.run ---

    @staticmethod
    def _fake_handle_result(*, tool_name, tool_call_id, content, had_error, **kwargs):
        from core.tool_executor import ToolExecutionOutcome
        from core.tool_results import parse_tool_execution_result

        parsed = parse_tool_execution_result(content)
        return ToolExecutionOutcome(
            tool_message=ToolMessage(content=content, tool_call_id=tool_call_id),
            parsed_result=parsed,
            had_error=had_error,
            issue=None,
            content=content,
        )

    def _log_node_start(self, state, node, **payload):
        return 0.0

    def _log_node_end(self, state, node, started_at, **payload):
        pass

    def _log_node_error(self, state, node, started_at, error, **payload):
        pass

    def _log_run_event(self, state, event, **payload):
        pass

    def _check_invariants(self, state):
        pass

    def _get_last_pending_ai_with_tool_calls(self, messages):
        return messages[-1] if messages else None

    def _current_turn_id(self, state, messages):
        return 1

    def _active_tools_for_turn(self, state, messages):
        return list(self._metadata.keys()), list(self._metadata.keys())

    def _tool_is_read_only(self, tool_name: str) -> bool:
        meta = self._metadata.get(tool_name)
        return bool(meta and meta.read_only and not meta.mutating and not meta.destructive)

    def _tool_call_is_parallel_safe(self, tool_call: dict[str, Any]) -> bool:
        name = tool_call.get("name") or "unknown_tool"
        return self._tool_is_read_only(name) and name in self.PARALLEL_SAFE_TOOL_NAMES

    def _partition_tool_calls(self, tool_calls):
        parallel = []
        sequential = []
        for tc in tool_calls:
            if self._tool_call_is_parallel_safe(tc):
                parallel.append(tc)
            else:
                sequential.append(tc)
        return parallel, sequential

    def _metadata_for_tool(self, tool_name: str) -> ToolMetadata:
        return self._metadata.get(tool_name, _make_tool_metadata(tool_name))

    def _effective_tool_metadata(self, tool_name, tool_args=None):
        return self._metadata_for_tool(tool_name)

    def _tool_is_allowed_for_turn(self, tool_name, allowed_tool_names=None):
        return True

    def _tool_requires_approval(self, tool_name, tool_args):
        return False

    def _tool_call_is_approved(self, tool_call_id, approval_state):
        return True

    def _missing_required_tool_fields(self, tool_name, tool_args):
        return None

    def _merge_open_tool_issues(self, tool_issues, current_turn_id):
        return None

    async def _process_tool_call(self, tool_call, recent_calls, state, approval_state, current_turn_id, allowed_tool_names):
        name = tool_call.get("name")
        self._call_log.append(name)
        await asyncio.sleep(self._process_delay)
        tool_msg = ToolMessage(
            content=f"result:{name}",
            tool_call_id=tool_call.get("id", ""),
        )
        return tool_msg, False, None


class MixedParallelToolsTests(unittest.IsolatedAsyncioTestCase):
    def _make_owner(self, names_read_only: dict[str, bool]) -> _FakeOwner:
        metadata = {
            name: _make_tool_metadata(name, read_only=ro)
            for name, ro in names_read_only.items()
        }
        return _FakeOwner(metadata)

    def _make_state(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content="", tool_calls=tool_calls)],
            "run_id": "test-run",
        }

    async def test_all_parallel_safe_uses_gather(self):
        """When every call is parallel-safe, all run concurrently."""
        owner = self._make_owner({"read_file": True, "search_in_directory": True})
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("search_in_directory", "tc2"),
        ]
        state = self._make_state(tool_calls)

        with mock.patch.object(
            ToolBatchCoordinator, "_parallel_mode_label", return_value="all"
        ) as mock_label:
            result = await coordinator.run(state)

        # Both calls should have been started within the same concurrency window.
        # If sequential, total time would be ~2*delay; with gather it's ~1*delay.
        mock_label.assert_called_once()
        self.assertEqual(len(result["messages"]), 2)
        # Order preserved
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")
        self.assertEqual(result["messages"][1].tool_call_id, "tc2")

    async def test_all_sequential(self):
        """When no call is parallel-safe, all run sequentially."""
        owner = self._make_owner({"write_file": False, "safe_delete_file": False})
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("write_file", "tc1"),
            _tc("safe_delete_file", "tc2"),
        ]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")
        self.assertEqual(result["messages"][1].tool_call_id, "tc2")

    async def test_mixed_mode_preserves_order(self):
        """Mixed batch: parallel-safe calls run concurrently, mutating calls sequentially,
        and the final ToolMessage order matches the original tool_calls order."""
        # read_file and search_in_directory are parallel-safe; write_file is not.
        owner = self._make_owner({
            "read_file": True,
            "write_file": False,
            "search_in_directory": True,
        })
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("write_file", "tc2"),
            _tc("search_in_directory", "tc3"),
        ]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 3)
        # Original order must be preserved regardless of execution mode.
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")
        self.assertEqual(result["messages"][1].tool_call_id, "tc2")
        self.assertEqual(result["messages"][2].tool_call_id, "tc3")
        # Content matches the tool name.
        self.assertEqual(result["messages"][0].content, "result:read_file")
        self.assertEqual(result["messages"][1].content, "result:write_file")
        self.assertEqual(result["messages"][2].content, "result:search_in_directory")

    async def test_mixed_mode_parallel_runs_concurrently(self):
        """In mixed mode, the parallel-safe calls should overlap in time
        (total time < sum of individual delays)."""
        owner = self._make_owner({
            "read_file": True,
            "search_in_directory": True,
            "write_file": False,
        })
        owner._process_delay = 0.15
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("search_in_directory", "tc2"),
            _tc("write_file", "tc3"),
        ]
        state = self._make_state(tool_calls)

        import time

        start = time.perf_counter()
        result = await coordinator.run(state)
        elapsed = time.perf_counter() - start

        # 2 parallel (0.15s concurrent) + 1 sequential (0.15s) = ~0.30s
        # If all were sequential: ~0.45s
        self.assertLess(elapsed, 0.42, f"Expected concurrent execution, took {elapsed:.3f}s")
        self.assertEqual(len(result["messages"]), 3)

    async def test_single_tool_call(self):
        """A single tool call should work regardless of its parallel-safety."""
        owner = self._make_owner({"write_file": False})
        coordinator = ToolBatchCoordinator(owner)

        tool_calls = [_tc("write_file", "tc1")]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].tool_call_id, "tc1")

    async def test_parallel_exception_does_not_crash_batch(self):
        """If one parallel call raises, the others should still complete."""
        owner = self._make_owner({"read_file": True, "search_in_directory": True})
        coordinator = ToolBatchCoordinator(owner)

        original_process = owner._process_tool_call

        async def flaky_process(tool_call, *args, **kwargs):
            if tool_call.get("id") == "tc1":
                raise RuntimeError("boom")
            return await original_process(tool_call, *args, **kwargs)

        owner._process_tool_call = flaky_process

        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("search_in_directory", "tc2"),
        ]
        state = self._make_state(tool_calls)

        result = await coordinator.run(state)

        self.assertEqual(len(result["messages"]), 2)
        # tc2 should have a valid result.
        self.assertEqual(result["messages"][1].content, "result:search_in_directory")

    # --- _partition_tool_calls unit tests ---

    def test_partition_all_parallel(self):
        owner = self._make_owner({"read_file": True, "search_in_directory": True})
        tool_calls = [_tc("read_file", "tc1"), _tc("search_in_directory", "tc2")]
        parallel, sequential = owner._partition_tool_calls(tool_calls)  # type: ignore[attr-defined]
        self.assertEqual(len(parallel), 2)
        self.assertEqual(len(sequential), 0)

    def test_partition_all_sequential(self):
        owner = self._make_owner({"write_file": False, "safe_delete_file": False})
        tool_calls = [_tc("write_file", "tc1"), _tc("safe_delete_file", "tc2")]
        parallel, sequential = owner._partition_tool_calls(tool_calls)  # type: ignore[attr-defined]
        self.assertEqual(len(parallel), 0)
        self.assertEqual(len(sequential), 2)

    def test_partition_mixed(self):
        owner = self._make_owner({"read_file": True, "write_file": False, "search_in_directory": True})
        tool_calls = [
            _tc("read_file", "tc1"),
            _tc("write_file", "tc2"),
            _tc("search_in_directory", "tc3"),
        ]
        parallel, sequential = owner._partition_tool_calls(tool_calls)  # type: ignore[attr-defined]
        self.assertEqual(len(parallel), 2)
        self.assertEqual(len(sequential), 1)
        self.assertEqual(parallel[0]["name"], "read_file")
        self.assertEqual(parallel[1]["name"], "search_in_directory")
        self.assertEqual(sequential[0]["name"], "write_file")

    # --- _parallel_mode_label unit tests ---

    def test_parallel_mode_label_all(self):
        self.assertEqual(
            ToolBatchCoordinator._parallel_mode_label([{"name": "a"}], []),
            "all",
        )

    def test_parallel_mode_label_mixed(self):
        self.assertEqual(
            ToolBatchCoordinator._parallel_mode_label([{"name": "a"}], [{"name": "b"}]),
            "mixed",
        )

    def test_parallel_mode_label_sequential(self):
        self.assertEqual(
            ToolBatchCoordinator._parallel_mode_label([], [{"name": "b"}]),
            "sequential",
        )


if __name__ == "__main__":
    unittest.main()
