import uuid
import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from core.state import AgentState
from core.config import AgentConfig
from core import constants
from core.run_logger import JsonlRunLogger
from core.tool_policy import ToolMetadata, default_tool_metadata
from core.tool_results import parse_tool_execution_result
from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.message_context import MessageContextHelper
from core.policy_engine import PolicyEngine, classify_shell_command, shell_command_requires_approval, tool_requires_approval
from core.message_utils import compact_text, is_error_text, stringify_content
from core.self_correction_engine import RepairPlan, build_repair_plan, normalize_tool_args, repair_fingerprint
from core.text_utils import format_exception_friendly
from core.context_builder import ContextBuilder
from core.recovery_manager import RecoveryManager
from core.summarize_policy import (
    choose_summary_boundary,
    estimate_tokens,
    format_history_for_summary,
    should_summarize,
)
from core.tool_executor import ToolExecutor
from core.tool_issues import build_tool_issue


class ProviderContextError(RuntimeError):
    """Raised when the agent context violates provider message-ordering constraints."""
    pass

logger = logging.getLogger("agent")


class AgentNodes:
    __slots__ = (
        "config",
        "llm",
        "tools",
        "llm_with_tools",
        "tools_map",
        "_all_tool_names",
        "tool_metadata",
        "run_logger",
        "_cached_base_prompt",
        "message_context",
        "policy_engine",
        "context_builder",
        "recovery_manager",
        "tool_executor",
    )

    # Only these tools are allowed to run in parallel in a single tool-call batch.
    # Any unknown or mutating tool keeps sequential execution for safety.
    PARALLEL_SAFE_TOOL_NAMES = frozenset(
        {
            "file_info",
            "read_file",
            "list_directory",
            "search_in_file",
            "search_in_directory",
            "tail_file",
            "find_file",
            "web_search",
            "fetch_content",
            "batch_web_search",
            "get_public_ip",
            "lookup_ip_info",
            "get_system_info",
            "get_local_network_info",
            "find_process_by_port",
        }
    )
    # Read-only tools can be called repeatedly while an agent verifies edits/results.
    READ_ONLY_LOOP_TOLERANT_TOOL_NAMES = frozenset(
        {
            "file_info",
            "read_file",
            "search_in_file",
            "search_in_directory",
            "tail_file",
            "find_file",
            "list_directory",
            "web_search",
            "fetch_content",
            "batch_web_search",
        }
    )
    # Planning/reasoning tools are helpful, but can easily create oscillation when
    # the model keeps "thinking" instead of switching to concrete actions.
    PLANNING_TOOL_NAMES = frozenset({"sequentialthinking", "sequential-thinking", "sequential_thinking"})
    PLANNING_TOOL_MAX_CALLS_PER_TURN = 4
    PROVIDER_SAFE_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9]{9}$")
    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        tools: List[BaseTool],
        llm_with_tools: Optional[BaseChatModel] = None,
        tool_metadata: Optional[Dict[str, ToolMetadata]] = None,
        run_logger: Optional[JsonlRunLogger] = None,
    ):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm_with_tools or llm

        # Оптимизация: O(1) доступ к инструментам вместо O(N) перебора списка
        self.tools_map = {t.name: t for t in tools}
        self._all_tool_names = tuple(self.tools_map.keys())
        self.tool_metadata = tool_metadata or {}
        self.run_logger = run_logger
        self.message_context = MessageContextHelper()
        self.policy_engine = PolicyEngine()
        self.recovery_manager = RecoveryManager()

        # Оптимизация: кэширование базового промпта (чтобы не читать с диска на каждый шаг)
        self._cached_base_prompt: Optional[str] = None
        self.context_builder = ContextBuilder(
            config=config,
            prompt_loader=self._get_base_prompt,
            is_internal_retry=self._is_internal_retry_message,
            log_run_event=self._log_run_event,
            recovery_message_builder=self.recovery_manager.build_recovery_system_message,
            provider_safe_tool_call_id_re=self.PROVIDER_SAFE_TOOL_CALL_ID_RE,
        )
        self.tool_executor = ToolExecutor(
            config=config,
            metadata_for_tool=self._metadata_for_tool,
            log_run_event=self._log_run_event,
            workspace_boundary_violated=self._workspace_boundary_violated,
        )

    def _log_run_event(self, state: AgentState | None, event_type: str, **payload: Any) -> None:
        if not self.run_logger:
            return
        # Guard against accidental duplicate keyword with the positional session_id
        # argument of JsonlRunLogger.log_event(...).
        payload.pop("session_id", None)
        session_id = None if state is None else state.get("session_id")
        self.run_logger.log_event(session_id, event_type, **payload)

    def _state_log_context(self, state: AgentState | None) -> Dict[str, Any]:
        if not state:
            return {}
        return {
            "run_id": state.get("run_id", ""),
            "step": state.get("steps", 0),
            "turn_id": state.get("turn_id", 0),
            "state_session_id": state.get("session_id", ""),
        }

    def _log_node_start(self, state: AgentState | None, node: str, **payload: Any) -> float:
        event_payload: Dict[str, Any] = {"node": node, **self._state_log_context(state)}
        event_payload.update(payload)
        self._log_run_event(state, "node_start", **event_payload)
        return time.perf_counter()

    def _log_node_end(self, state: AgentState | None, node: str, started_at: float, **payload: Any) -> None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        event_payload: Dict[str, Any] = {
            "node": node,
            "duration_ms": duration_ms,
            **self._state_log_context(state),
        }
        event_payload.update(payload)
        self._log_run_event(state, "node_end", **event_payload)

    def _log_node_error(
        self,
        state: AgentState | None,
        node: str,
        started_at: float,
        error: Exception,
        **payload: Any,
    ) -> None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        event_payload: Dict[str, Any] = {
            "node": node,
            "duration_ms": duration_ms,
            "error_type": type(error).__name__,
            "error": compact_text(str(error), 400),
            **self._state_log_context(state),
        }
        event_payload.update(payload)
        self._log_run_event(state, "node_error", **event_payload)

    def _metadata_for_tool(self, tool_name: str) -> ToolMetadata:
        return self.tool_metadata.get(tool_name, default_tool_metadata(tool_name))

    def _normalize_tool_name(self, tool_name: str) -> str:
        return str(tool_name or "").strip().lower()

    def _is_planning_tool(self, tool_name: str) -> bool:
        normalized = self._normalize_tool_name(tool_name)
        condensed = normalized.replace("-", "").replace("_", "")
        return normalized in self.PLANNING_TOOL_NAMES or condensed == "sequentialthinking"

    def _required_tool_fields(self, tool_name: str) -> List[str]:
        tool = self.tools_map.get(tool_name)
        if not tool:
            return []
        try:
            schema = tool.get_input_schema()
        except Exception:
            return []

        fields = getattr(schema, "model_fields", {}) or {}
        required: List[str] = []
        for field_name, field_info in fields.items():
            try:
                if field_info.is_required():
                    required.append(str(field_name))
            except Exception:
                continue
        return required

    def _missing_required_tool_fields(self, tool_name: str, tool_args: Dict[str, Any]) -> List[str]:
        required = self._required_tool_fields(tool_name)
        if not required:
            return []
        missing: List[str] = []
        for field_name in required:
            value = tool_args.get(field_name)
            if value is None:
                missing.append(field_name)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing

    def _tool_is_read_only(self, tool_name: str) -> bool:
        metadata = self._metadata_for_tool(tool_name)
        return metadata.read_only and not metadata.mutating and not metadata.destructive

    def _effective_tool_metadata(self, tool_name: str, tool_args: Dict[str, Any] | None = None) -> ToolMetadata:
        metadata = self._metadata_for_tool(tool_name)
        if self._normalize_tool_name(tool_name) != "cli_exec":
            return metadata

        command = str(((tool_args or {}).get("command")) or "").strip()
        if not command:
            return metadata

        profile = classify_shell_command(command)
        approval_required = shell_command_requires_approval(command)
        inspect_only = bool(profile.get("inspect_only") and not profile.get("long_running_service"))
        return ToolMetadata(
            name=metadata.name,
            read_only=inspect_only,
            mutating=bool(profile.get("mutating") or profile.get("long_running_service")),
            destructive=bool(profile.get("destructive")),
            requires_approval=approval_required,
            networked=bool(metadata.networked or profile.get("network_diagnostic") or profile.get("http_probe")),
            source=metadata.source,
        )

    def _tool_requires_approval(self, tool_name: str, tool_args: Dict[str, Any] | None = None) -> bool:
        metadata = self._effective_tool_metadata(tool_name, tool_args)
        return tool_requires_approval(
            tool_name,
            tool_args,
            metadata=metadata,
            approvals_enabled=self.config.enable_approvals,
        )

    def _tool_error_requires_recovery_gate(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        parsed_result,
    ) -> bool:
        return self.tool_executor._tool_error_requires_recovery_gate(tool_name, tool_args, parsed_result)

    def _maybe_build_open_tool_issue(
        self,
        *,
        state: AgentState,
        current_turn_id: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        parsed_result,
        content: str,
        issue_details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        return self.tool_executor._build_open_tool_issue(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            parsed_result=parsed_result,
            content=content,
            issue_details=issue_details,
        )

    def tool_calls_require_approval(self, tool_calls: List[Dict[str, Any]]) -> bool:
        return any(
            self._tool_requires_approval(
                (tool_call.get("name") or "unknown_tool"),
                tool_call.get("args") or {},
            )
            for tool_call in tool_calls
            if tool_call.get("name") != "request_user_input"
        )

    # --- NODE: SUMMARIZE ---

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        return estimate_tokens(messages)

    async def summarize_node(self, state: AgentState):
        messages = state["messages"]
        summary = state.get("summary", "")

        estimated_tokens = self._estimate_tokens(messages)
        node_timer = self._log_node_start(
            state,
            "summarize",
            message_count=len(messages),
            estimated_tokens=estimated_tokens,
            threshold=self.config.summary_threshold,
            keep_last=self.config.summary_keep_last,
            has_summary=bool(summary),
        )

        if not should_summarize(messages, threshold=self.config.summary_threshold):
            self._log_node_end(
                state,
                "summarize",
                node_timer,
                outcome="skipped",
                reason="below_threshold",
            )
            return {}

        logger.debug(f"📊 Context size: ~{estimated_tokens} tokens. Summarizing...")

        # Determine cut-off point
        idx = choose_summary_boundary(messages, keep_last=self.config.summary_keep_last)

        to_summarize = messages[:idx]

        # ЗАЩИТА: Если последние N сообщений сами по себе весят больше лимита,
        # мы не можем ничего сжать без потери недавнего контекста.
        if not to_summarize:
            logger.warning(
                f"⚠ Context (~{estimated_tokens} tokens) exceeds threshold, "
                "but cannot summarize further without deleting the most recent active messages. "
                "Expanding context dynamically for this turn."
            )
            self._log_node_end(
                state,
                "summarize",
                node_timer,
                outcome="skipped",
                reason="no_summarizable_messages",
            )
            return {}

        history_text = self._format_history_for_summary(to_summarize)

        prompt = constants.SUMMARY_PROMPT_TEMPLATE.format(summary=summary, history_text=history_text)

        try:
            res = await self.llm.ainvoke(prompt)

            delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
            logger.info(f"🧹 Summary: Removed {len(delete_msgs)} messages. Generated new summary.")
            self._log_run_event(
                state,
                "summary_compacted",
                estimated_tokens=estimated_tokens,
                removed_messages=len(delete_msgs),
                summarized_messages=len(to_summarize),
            )
            self._log_node_end(
                state,
                "summarize",
                node_timer,
                outcome="compacted",
                removed_messages=len(delete_msgs),
                summarized_messages=len(to_summarize),
            )

            return {"summary": res.content, "messages": delete_msgs}
        except Exception as e:
            err_str = str(e)
            if "content_filter" in err_str or "Moderation Block" in err_str:
                logger.warning(
                    "🧹 Summarization skipped due to Content Filter (False Positive). Continuing with full history."
                )
            else:
                logger.error(f"Summarization Error: {format_exception_friendly(e)}")
            self._log_node_error(
                state,
                "summarize",
                node_timer,
                e,
                outcome="failed",
                estimated_tokens=estimated_tokens,
            )
            return {}

    def _format_history_for_summary(self, messages: List[BaseMessage]) -> str:
        return format_history_for_summary(messages, is_internal_retry=self._is_internal_retry_message)

    def _is_internal_retry_message(self, message: BaseMessage) -> bool:
        if not isinstance(message, HumanMessage):
            return False
        metadata = getattr(message, "additional_kwargs", {}) or {}
        internal = metadata.get("agent_internal")
        return isinstance(internal, dict) and internal.get("kind") == "retry_instruction"

    def _current_turn_id(self, state: AgentState, messages: List[BaseMessage]) -> int:
        derived_turn_id = 0
        for message in messages:
            if not isinstance(message, HumanMessage) or self._is_internal_retry_message(message):
                continue
            content = stringify_content(message.content).strip()
            if content and content != constants.REFLECTION_PROMPT:
                derived_turn_id += 1
        return max(int(state.get("turn_id", 0) or 0), derived_turn_id)

    def _get_active_open_tool_issue(
        self,
        state: AgentState,
        messages: List[BaseMessage],
        current_turn_id: int | None = None,
    ) -> Dict[str, Any] | None:
        issue = state.get("open_tool_issue")
        if not isinstance(issue, dict):
            return None

        active_turn_id = current_turn_id if current_turn_id is not None else self._current_turn_id(state, messages)
        issue_turn_id = int(issue.get("turn_id", 0) or 0)
        summary = str(issue.get("summary", "")).strip()
        if issue_turn_id != active_turn_id or not summary:
            return None

        tool_names = [str(name) for name in (issue.get("tool_names") or []) if str(name).strip()]
        return {
            "turn_id": issue_turn_id,
            "kind": str(issue.get("kind") or "tool_error"),
            "summary": summary,
            "tool_names": tool_names,
            "tool_args": dict(issue.get("tool_args") or {}),
            "source": str(issue.get("source") or "tools"),
            "error_type": str(issue.get("error_type") or ""),
            "fingerprint": str(issue.get("fingerprint") or ""),
            "progress_fingerprint": str(issue.get("progress_fingerprint") or issue.get("fingerprint") or ""),
            "details": dict(issue.get("details") or {}),
        }

    def _empty_recovery_state(self, *, turn_id: int) -> Dict[str, Any]:
        return self.recovery_manager.empty_state(turn_id=turn_id)

    def _get_recovery_state(self, state: AgentState, *, current_turn_id: int) -> Dict[str, Any]:
        return self.recovery_manager.get_recovery_state(
            state.get("recovery_state"),
            current_turn_id=current_turn_id,
        )

    def _repair_plan_strategy_id(self, repair_plan: RepairPlan) -> str:
        return self.recovery_manager.repair_plan_strategy_id(repair_plan)

    def _build_recovery_strategy(
        self,
        *,
        repair_plan: RepairPlan,
        open_tool_issue: Dict[str, Any] | None,
        current_task: str,
        strategy_id: str,
    ) -> Dict[str, Any]:
        return self.recovery_manager.build_recovery_strategy(
            repair_plan=repair_plan,
            open_tool_issue=open_tool_issue,
            current_task=current_task,
            strategy_id=strategy_id,
        )

    def _build_recovery_system_message(self, recovery_state: Dict[str, Any] | None) -> SystemMessage | None:
        return self.recovery_manager.build_recovery_system_message(recovery_state)

    def _collect_internal_retry_removals(self, messages: List[BaseMessage]) -> List[RemoveMessage]:
        removals: List[RemoveMessage] = []
        for message in reversed(messages):
            if not self._is_internal_retry_message(message):
                break
            if message.id:
                removals.append(RemoveMessage(id=message.id))
        removals.reverse()
        return removals

    def _build_tool_issue_system_message(self, open_tool_issue: Dict[str, Any] | None) -> SystemMessage | None:
        if not open_tool_issue:
            return None

        issue_summary = open_tool_issue.get("summary", "")
        if open_tool_issue.get("kind") == "approval_denied":
            return SystemMessage(
                content=(
                    "TOOL EXECUTION DENIED BY USER:\n"
                    f"{issue_summary}\n\n"
                    "The user explicitly rejected this tool call. "
                    "Do not simulate the denied tool or describe imaginary results. "
                    "Do not make any more tool calls in this turn. "
                    "Reply briefly and simply: say that you did not do it because the user chose No, "
                    "then wait for the next instruction."
                )
            )

        return SystemMessage(
            content=constants.UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE.format(
                error_summary=issue_summary
            )
        )

    def _build_open_tool_issue(
        self,
        *,
        current_turn_id: int,
        kind: str,
        summary: str,
        tool_names: List[str],
        tool_args: Dict[str, Any] | None,
        source: str,
        error_type: str = "",
        fingerprint: str = "",
        details: Dict[str, Any] | None = None,
        progress_fingerprint: str = "",
    ) -> Dict[str, Any]:
        return {
            "turn_id": current_turn_id,
            "kind": kind,
            "summary": compact_text(summary.strip(), 320),
            "tool_names": [name for name in tool_names if name],
            "tool_args": deepcopy(tool_args) if isinstance(tool_args, dict) else {},
            "source": source,
            "error_type": error_type.strip().upper(),
            "fingerprint": str(fingerprint or "").strip(),
            "progress_fingerprint": str(progress_fingerprint or fingerprint or "").strip(),
            "details": deepcopy(details) if isinstance(details, dict) else {},
        }

    def _iter_path_like_targets(self, tool_args: Dict[str, Any]) -> List[Tuple[str, str]]:
        if not isinstance(tool_args, dict):
            return []
        targets: List[Tuple[str, str]] = []
        for key in ("path", "file_path", "dir_path", "source", "destination", "cwd"):
            raw_value = tool_args.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                targets.append((key, raw_value.strip()))
        return targets

    def _tool_mutates_workspace(self, tool_name: str) -> bool:
        metadata = self._metadata_for_tool(tool_name)
        return metadata.mutating or metadata.destructive

    def _workspace_boundary_violated(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        if not self._tool_mutates_workspace(tool_name):
            return False

        if tool_name == "run_background_process":
            try:
                from tools import process_tools

                process_tools._resolve_cwd(str((tool_args or {}).get("cwd") or "."))
                return False
            except Exception:
                return True

        path_targets = self._iter_path_like_targets(tool_args)
        if not path_targets:
            return False

        try:
            from tools.filesystem import fs_manager
        except Exception:
            return False

        for _, raw_value in path_targets:
            try:
                fs_manager._resolve_path(raw_value)
            except Exception:
                return True
        return False

    def _build_open_tool_issue_details(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        parsed_result,
        issue_details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        details = deepcopy(issue_details) if isinstance(issue_details, dict) else {}
        missing_fields = [
            str(field).strip()
            for field in (details.get("missing_required_fields") or [])
            if str(field).strip()
        ]
        if missing_fields:
            details["missing_required_fields"] = missing_fields
        if str(parsed_result.error_type or "").strip().upper() == "LOOP_DETECTED":
            details["loop_detected"] = True
        details["safety_violation"] = bool(
            details.get("safety_violation") or self._workspace_boundary_violated(tool_name, tool_args)
        )
        return details

    def _merge_open_tool_issues(
        self,
        issues: List[Dict[str, Any]],
        current_turn_id: int,
    ) -> Dict[str, Any] | None:
        return self.tool_executor.merge_issues(issues, current_turn_id=current_turn_id)

    def _current_turn_has_tool_evidence(self, messages: List[BaseMessage]) -> bool:
        return self.message_context.current_turn_has_tool_evidence(
            messages,
            self._is_internal_retry_message,
        )

    def _log_interrupted_tool_result_if_needed(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        parsed_result,
    ) -> None:
        self.tool_executor._log_interrupted_tool_result_if_needed(
            state=state,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            parsed_result=parsed_result,
        )

    def _build_agent_context(
        self,
        messages: List[BaseMessage],
        summary: str,
        current_task: str,
        tools_available: bool,
        active_tool_names: List[str],
        open_tool_issue: Dict[str, Any] | None,
        recovery_state: Dict[str, Any] | None = None,
        state: AgentState | None = None,
        user_choice_locked: bool = False,
    ) -> List[BaseMessage]:
        return self.context_builder.build(
            messages,
            state,
            summary=summary,
            current_task=current_task,
            tools_available=tools_available,
            active_tool_names=active_tool_names,
            open_tool_issue=open_tool_issue,
            recovery_state=recovery_state,
            user_choice_locked=user_choice_locked,
        )

    def _normalize_system_prefix_for_provider(
        self,
        context: List[BaseMessage],
    ) -> List[BaseMessage]:
        return self.context_builder.normalize_system_prefix(context)

    def _message_role_for_provider(self, message: BaseMessage) -> str:
        return self.context_builder._message_role_for_provider(message)

    def _message_is_provider_system(self, message: BaseMessage) -> bool:
        return self.context_builder.message_is_provider_system(message)

    def _is_provider_safe_tool_call_id(self, tool_call_id: str) -> bool:
        return bool(self.PROVIDER_SAFE_TOOL_CALL_ID_RE.fullmatch(str(tool_call_id or "").strip()))

    def _normalize_tool_call_id_for_provider(self, tool_call_id: str, *, used_ids: set[str]) -> str:
        return self.context_builder._normalize_tool_call_id_for_provider(tool_call_id, used_ids=used_ids)

    def _sanitize_messages_for_model(
        self,
        messages: List[BaseMessage],
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        return self.context_builder.sanitize_messages(messages, state=state)

    def _build_protocol_open_tool_issue(
        self,
        *,
        current_turn_id: int,
        summary: str,
        reason: str,
        source: str,
        tool_names: List[str] | None = None,
        tool_args: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
        response_preview: str = "",
    ) -> Dict[str, Any]:
        normalized_tool_names = [
            str(name).strip()
            for name in (tool_names or [])
            if str(name).strip()
        ]
        primary_tool_name = normalized_tool_names[0] if normalized_tool_names else "unknown_tool"
        primary_tool_args = dict(tool_args or {}) if isinstance(tool_args, dict) else {}
        issue_details = dict(details or {}) if isinstance(details, dict) else {}
        if reason:
            issue_details["protocol_reason"] = reason
        if response_preview:
            issue_details["response_preview"] = compact_text(response_preview, 220)
        fingerprint_payload = dict(primary_tool_args)
        if reason:
            fingerprint_payload["__protocol_reason"] = reason
        return build_tool_issue(
            current_turn_id=current_turn_id,
            kind="protocol_error",
            summary=summary,
            tool_names=normalized_tool_names or [primary_tool_name],
            tool_args=primary_tool_args,
            source=source,
            error_type="PROTOCOL",
            fingerprint=repair_fingerprint(primary_tool_name, fingerprint_payload, "PROTOCOL"),
            progress_fingerprint=repair_fingerprint(primary_tool_name, fingerprint_payload, "PROTOCOL"),
            details=issue_details,
        )

    def _summarize_history_tool_mismatch(self, history_issue: Dict[str, Any]) -> str:
        pending_count = len(history_issue.get("pending_tool_calls") or [])
        orphan_count = len(history_issue.get("orphan_tool_results") or [])
        duplicate_count = len(history_issue.get("duplicate_tool_call_ids") or [])
        interleaving = len(history_issue.get("interleaving_markers") or [])

        fragments: List[str] = []
        if pending_count:
            fragments.append(f"{pending_count} незавершенный(ых) tool call")
        if orphan_count:
            fragments.append(f"{orphan_count} orphan tool result")
        if duplicate_count:
            fragments.append(f"{duplicate_count} duplicate tool_call_id")
        if interleaving:
            fragments.append("обнаружены сообщения между tool call и tool result")
        if not fragments:
            fragments.append("история tool call повреждена")
        return "Нарушен внутренний контракт истории инструментов: " + ", ".join(fragments) + "."

    def _build_agent_result(
        self,
        response: AIMessage,
        current_task: str,
        tools_available: bool,
        turn_id: int,
        messages: List[BaseMessage],
        open_tool_issue: Dict[str, Any] | None = None,
        recovery_state: Dict[str, Any] | None = None,
        allowed_tool_names: List[str] | None = None,
        should_force_tools: bool = False,
        current_turn_has_tool_evidence: bool = False,
    ) -> Dict[str, Any]:
        token_usage_update = {}
        if getattr(response, "usage_metadata", None):
            token_usage_update = {"token_usage": response.usage_metadata}

        has_tool_calls = False
        protocol_error = ""
        protocol_issue: Dict[str, Any] | None = None
        outbound_messages: List[BaseMessage] = self._collect_internal_retry_removals(messages)

        if isinstance(response, AIMessage):
            t_calls = list(getattr(response, "tool_calls", []))
            invalid_calls = list(getattr(response, "invalid_tool_calls", []))
            retry_user_input_turn = False

            missing_fields = [
                tc for tc in t_calls
                if not tc.get("id") or not tc.get("name")
            ]
            if missing_fields or invalid_calls:
                protocol_error = self._build_tool_protocol_error(missing_fields, invalid_calls)
                tool_names = [
                    str(tc.get("name") or "").strip()
                    for tc in t_calls
                    if str(tc.get("name") or "").strip()
                ]
                first_args = next(
                    (
                        dict(tc.get("args") or {})
                        for tc in t_calls
                        if isinstance(tc.get("args"), dict)
                    ),
                    {},
                )
                protocol_issue = self._build_protocol_open_tool_issue(
                    current_turn_id=turn_id,
                    summary=protocol_error,
                    reason="tool_protocol_error",
                    source="agent",
                    tool_names=tool_names,
                    tool_args=first_args,
                    details={
                        "invalid_tool_call_count": len(invalid_calls),
                        "missing_field_tool_call_count": len(missing_fields),
                    },
                    response_preview=stringify_content(response.content),
                )
                response = AIMessage(
                    content=self._merge_protocol_error_into_content(response.content, protocol_error),
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                t_calls = []
            else:
                (
                    t_calls,
                    user_input_protocol_error,
                    retry_user_input_turn,
                ) = self._sanitize_user_input_tool_calls(
                    t_calls,
                    messages,
                )
                original_tool_calls = list(t_calls)
                t_calls = self._filter_tool_calls_for_turn(
                    t_calls,
                    allowed_tool_names=allowed_tool_names,
                )
                if user_input_protocol_error:
                    protocol_error = self._merge_protocol_error_text(
                        protocol_error,
                        user_input_protocol_error,
                    )
                    response = response.model_copy(update={"tool_calls": t_calls})
                filtered_out_count = len(original_tool_calls) - len(t_calls)
                if filtered_out_count:
                    dropped_names = [
                        str(tool_call.get("name") or "").strip()
                        for tool_call in original_tool_calls
                        if self._normalize_tool_name(tool_call.get("name") or "") not in {
                            self._normalize_tool_name(name) for name in (allowed_tool_names or [])
                        }
                    ]
                    dropped_error = (
                        "INTERNAL TOOL PROTOCOL ERROR: model requested tool(s) that are not available in the current turn. "
                        "Issue a fresh valid tool call using only allowed tools."
                    )
                    protocol_error = self._merge_protocol_error_text(protocol_error, dropped_error)
                    if not t_calls:
                        protocol_issue = self._build_protocol_open_tool_issue(
                            current_turn_id=turn_id,
                            summary=dropped_error,
                            reason="tool_not_allowed_for_turn",
                            source="agent",
                            tool_names=dropped_names,
                            tool_args={},
                            details={"allowed_tool_names": list(allowed_tool_names or [])},
                            response_preview=stringify_content(response.content),
                        )
                    response = response.model_copy(update={"tool_calls": t_calls})

            has_tool_calls = bool(tools_available and t_calls)
            if has_tool_calls and open_tool_issue and open_tool_issue.get("kind") == "approval_denied":
                response = AIMessage(
                    content="Okay, I did not do that because you chose No. Tell me what you want to do instead.",
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                has_tool_calls = False

            if (
                tools_available
                and should_force_tools
                and not has_tool_calls
                and not current_turn_has_tool_evidence
                and protocol_issue is None
            ):
                response_preview = stringify_content(response.content).strip()
                if response_preview:
                    protocol_issue = self._build_protocol_open_tool_issue(
                        current_turn_id=turn_id,
                        summary=(
                            "Запрос требует проверяемого действия или инструментального подтверждения, "
                            "но модель завершила шаг обычным текстом без валидного tool call."
                        ),
                        reason="action_requires_tools",
                        source="agent",
                        tool_names=list(allowed_tool_names or []),
                        tool_args={},
                        details={"allowed_tool_names": list(allowed_tool_names or [])},
                        response_preview=response_preview,
                    )
                    protocol_error = self._merge_protocol_error_text(
                        protocol_error,
                        "ACTION REQUIRES TOOLS: do not stop at prose. Continue with a valid tool call or a real blocking reason."
                    )

        outbound_messages.append(response)

        return {
            "messages": outbound_messages,
            "turn_id": turn_id,
            "current_task": current_task,
            "turn_outcome": "run_tools" if has_tool_calls else "",
            "recovery_state": recovery_state,
            "pending_approval": None,
            "open_tool_issue": protocol_issue or open_tool_issue,
            "has_protocol_error": bool(protocol_error),
            "last_tool_error": protocol_error,
            "last_tool_result": "",
            "_retry_user_input_turn": retry_user_input_turn,
            **token_usage_update,
        }

    def _filter_tool_calls_for_turn(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        allowed_tool_names: List[str] | None,
    ) -> List[Dict[str, Any]]:
        if not tool_calls or allowed_tool_names is None:
            return tool_calls
        allowed = {self._normalize_tool_name(name) for name in allowed_tool_names}
        return [
            tool_call
            for tool_call in tool_calls
            if self._normalize_tool_name(tool_call.get("name") or "") in allowed
        ]

    def _merge_protocol_error_into_content(self, content: Any, protocol_error: str) -> str:
        base_text = stringify_content(content).strip()
        if not protocol_error:
            return base_text
        if not base_text:
            return protocol_error
        return f"{base_text}\n\n{protocol_error}"

    def _merge_protocol_error_text(self, existing: str, addition: str) -> str:
        left = str(existing or "").strip()
        right = str(addition or "").strip()
        if not left:
            return right
        if not right:
            return left
        return f"{left}\n{right}"

    def _build_tool_protocol_error(
        self,
        missing_fields: List[Dict[str, Any]],
        invalid_calls: List[Dict[str, Any]],
    ) -> str:
        details = []
        if missing_fields:
            details.append(
                f"{len(missing_fields)} tool call(s) were missing required 'name' or 'id' fields"
            )
        if invalid_calls:
            details.append(
                f"{len(invalid_calls)} tool call(s) had invalid arguments and could not be parsed"
            )
        joined = "; ".join(details) if details else "Malformed tool call payload received."
        return (
            "INTERNAL TOOL PROTOCOL ERROR: "
            f"{joined}. Do not invent tool names or IDs. "
            "If tools are still needed, issue a fresh valid tool call."
        )

    def _sanitize_user_input_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        messages: List[BaseMessage],
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        if not tool_calls:
            return tool_calls, "", False

        user_input_calls = [
            tool_call
            for tool_call in tool_calls
            if self._normalize_tool_name(tool_call.get("name") or "") == "request_user_input"
        ]
        if not user_input_calls:
            return tool_calls, "", False

        if self._current_turn_has_user_input_request(messages):
            return [], (
                "INTERNAL TOOL PROTOCOL ERROR: request_user_input may be called at most once per user turn. "
                "You already asked for user input in this turn. Use the answer you already received and continue "
                "instead of asking another question."
            ), True

        if len(user_input_calls) == 1 and len(tool_calls) == 1:
            return tool_calls, "", False

        first_user_input_call = user_input_calls[0]
        if len(user_input_calls) > 1:
            return [first_user_input_call], (
                "INTERNAL TOOL PROTOCOL ERROR: request_user_input may appear at most once in a single assistant "
                "response. Extra user-input requests were dropped. Ask one question, wait for resume, then continue."
            ), False

        return [first_user_input_call], (
            "INTERNAL TOOL PROTOCOL ERROR: request_user_input cannot be combined with other tool calls in the same "
            "assistant response. Non-user-input tool calls were dropped. Ask one question, wait for resume, then continue."
        ), False
    
    # --- NODE: AGENT ---

    async def agent_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "agent",
            message_count=len(state.get("messages") or []),
            has_summary=bool(state.get("summary")),
        )
        messages = state["messages"]
        summary = state.get("summary", "")
        current_task = self._resolve_current_task(state, messages)
        current_turn_id = self._current_turn_id(state, messages)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
        recovery_state = self._get_recovery_state(state, current_turn_id=current_turn_id)
        self._log_run_event(
            state,
            "agent_node_start",
            run_id=state.get("run_id", ""),
            step=state.get("steps", 0),
            current_task=current_task,
        )

        active_tools, active_tool_names = self._active_tools_for_turn(messages)
        if active_tool_names == list(self._all_tool_names):
            llm_for_turn = self.llm_with_tools if active_tool_names else self.llm
        elif active_tools:
            if hasattr(self.llm, "bind_tools"):
                llm_for_turn = self.llm.bind_tools(active_tools)
            else:
                llm_for_turn = self.llm_with_tools
        else:
            llm_for_turn = self.llm
        tools_available = bool(active_tool_names)
        user_choice_locked = self._current_turn_has_completed_user_choice(messages)
        turn_policy = self.policy_engine.evaluate_turn(
            task=current_task,
            messages=messages,
            current_turn_id=current_turn_id,
            is_internal_retry=self._is_internal_retry_message,
        )
        inspect_only_turn = turn_policy.inspect_only
        operational_evidence_required = turn_policy.requires_operational_evidence
        current_turn_has_tool_evidence = self._current_turn_has_tool_evidence(messages)
        self._log_run_event(
            state,
            "intent_decision",
            run_id=state.get("run_id", ""),
            intent=turn_policy.intent,
            inspect_only=inspect_only_turn,
            requires_operational_evidence=operational_evidence_required,
            should_force_tools=turn_policy.should_force_tools,
            has_current_turn_evidence=current_turn_has_tool_evidence,
            task_preview=compact_text(current_task, 180),
        )
        self._log_run_event(
            state,
            "policy_decision",
            run_id=state.get("run_id", ""),
            inspect_only=turn_policy.inspect_only,
            requires_operational_evidence=turn_policy.requires_operational_evidence,
            prefer_read_only_fallback=turn_policy.prefer_read_only_fallback,
        )
        try:
            validation_handoff_reason = ""
            history_issue = self.context_builder.detect_tool_history_mismatch(messages)
            if history_issue:
                validation_handoff_reason = "history_tool_mismatch"
                protocol_issue = self._build_protocol_open_tool_issue(
                    current_turn_id=current_turn_id,
                    summary=self._summarize_history_tool_mismatch(history_issue),
                    reason="history_tool_mismatch",
                    source="history",
                    tool_names=[
                        str(item.get("name") or "").strip()
                        for item in (history_issue.get("pending_tool_calls") or [])
                        if str(item.get("name") or "").strip()
                    ],
                    tool_args=(
                        dict((history_issue.get("pending_tool_calls") or [{}])[0].get("args") or {})
                        if history_issue.get("pending_tool_calls")
                        else {}
                    ),
                    details=history_issue,
                )
                self._log_run_event(
                    state,
                    "history_tool_mismatch_detected",
                    run_id=state.get("run_id", ""),
                    issue=protocol_issue,
                )
                self._log_node_end(
                    state,
                    "agent",
                    node_timer,
                    tool_calls=0,
                    tools_available=tools_available,
                    active_tool_count=len(active_tool_names),
                    validation_handoff_reason=validation_handoff_reason,
                    has_open_tool_issue=True,
                )
                return {
                    "current_task": current_task,
                    "turn_id": current_turn_id,
                    "turn_outcome": "",
                    "recovery_state": recovery_state,
                    "pending_approval": None,
                    "open_tool_issue": protocol_issue,
                    "has_protocol_error": True,
                    "last_tool_error": str(protocol_issue.get("summary") or ""),
                    "last_tool_result": "",
                }

            full_context = self._build_agent_context(
                messages,
                summary,
                current_task,
                tools_available,
                active_tool_names,
                open_tool_issue,
                recovery_state,
                state=state,
                user_choice_locked=user_choice_locked,
            )
            self._assert_provider_safe_agent_context(full_context, state)
            response = await self._invoke_llm_with_retry(
                llm_for_turn,
                full_context,
                state=state,
                node_name="agent",
            )
            result = self._build_agent_result(
                response,
                current_task,
                tools_available,
                current_turn_id,
                messages,
                open_tool_issue=open_tool_issue,
                recovery_state=recovery_state,
                allowed_tool_names=active_tool_names,
                should_force_tools=turn_policy.should_force_tools,
                current_turn_has_tool_evidence=current_turn_has_tool_evidence,
            )
            if result.pop("_retry_user_input_turn", False):
                self._log_run_event(
                    state,
                    "user_input_reask_suppressed",
                    run_id=state.get("run_id", ""),
                    step=state.get("steps", 0),
                    current_task=current_task,
                )
                retry_context = self._normalize_system_prefix_for_provider(
                    [
                        *full_context,
                        SystemMessage(
                            content=(
                                "USER INPUT ALREADY PROVIDED IN THIS TURN. "
                                "Do not call request_user_input again. "
                                "Use the latest request_user_input ToolMessage as the user's final choice and continue."
                            )
                        ),
                    ]
                )
                self._assert_provider_safe_agent_context(retry_context, state)
                response = await self._invoke_llm_with_retry(
                    llm_for_turn,
                    retry_context,
                    state=state,
                    node_name="agent_retry_after_user_choice",
                )
                result = self._build_agent_result(
                    response,
                    current_task,
                    tools_available,
                    current_turn_id,
                    messages,
                    open_tool_issue=open_tool_issue,
                    recovery_state=recovery_state,
                    allowed_tool_names=active_tool_names,
                    should_force_tools=turn_policy.should_force_tools,
                    current_turn_has_tool_evidence=current_turn_has_tool_evidence,
                )
                result.pop("_retry_user_input_turn", None)
            if result.get("open_tool_issue") and result.get("has_protocol_error"):
                validation_handoff_reason = str(
                    ((result.get("open_tool_issue") or {}).get("details") or {}).get("protocol_reason")
                    or "tool_protocol_error"
                )
                self._log_run_event(
                    state,
                    "protocol_recovery_requested",
                    run_id=state.get("run_id", ""),
                    issue=result.get("open_tool_issue"),
                )
            tool_calls_count = len(getattr(response, "tool_calls", []) or [])
            self._log_run_event(
                state,
                "agent_node_end",
                run_id=state.get("run_id", ""),
                tool_calls=tool_calls_count,
                active_tool_names=active_tool_names,
                validation_handoff_reason=validation_handoff_reason,
                content_preview=compact_text(stringify_content(response.content), 240),
            )
            self._log_node_end(
                state,
                "agent",
                node_timer,
                tool_calls=tool_calls_count,
                tools_available=tools_available,
                active_tool_count=len(active_tool_names),
                validation_handoff_reason=validation_handoff_reason,
                has_open_tool_issue=bool(open_tool_issue),
            )
            return result
        except Exception as e:
            self._log_node_error(
                state,
                "agent",
                node_timer,
                e,
                tools_available=tools_available,
                has_open_tool_issue=bool(open_tool_issue),
            )
            raise

    def _hard_loop_ceiling(self) -> int:
        return max(1, int(self.config.max_loops or 1))

    def _handle_pending_tool_budget_exhaustion(
        self,
        *,
        current_task: str,
        last_ai: AIMessage | None,
    ) -> Dict[str, Any]:
        return {
            "completion_reason": "loop_budget_exhausted_pending_tool_call",
            "turn_outcome": "finish_turn",
            "handoff_message": self._build_loop_budget_handoff_text(
                current_task=current_task,
                tool_names=[
                    str(tool_call.get("name") or "").strip()
                    for tool_call in (getattr(last_ai, "tool_calls", []) or [])
                    if str(tool_call.get("name") or "").strip()
                ],
            ),
            "drop_trailing_tool_call": True,
            "next_open_tool_issue": None,
        }

    def _handle_clean_turn(
        self,
        *,
        state: AgentState,
        last_message: BaseMessage | None,
        next_recovery_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        next_recovery_state["active_issue"] = None
        next_recovery_state["active_strategy"] = None
        next_recovery_state["strategy_queue"] = []
        next_recovery_state["external_blocker"] = None
        if isinstance(last_message, ToolMessage):
            next_recovery_state["last_successful_evidence"] = str(
                state.get("last_tool_result") or stringify_content(last_message.content)
            ).strip()
            return {
                "completion_reason": "tool_result_ready_for_agent",
                "turn_outcome": "continue_agent",
                "next_open_tool_issue": None,
            }
        return {
            "completion_reason": "no_open_tool_issue",
            "turn_outcome": "finish_turn",
            "next_open_tool_issue": None,
        }

    def _handle_terminal_tool_issue(
        self,
        *,
        open_tool_issue: Dict[str, Any],
        repair_plan: RepairPlan,
        next_recovery_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        next_recovery_state["active_issue"] = open_tool_issue
        next_recovery_state["active_strategy"] = None
        next_recovery_state["strategy_queue"] = []
        next_recovery_state["external_blocker"] = {
            "reason": repair_plan.terminal_reason or repair_plan.reason,
            "issue_summary": str(open_tool_issue.get("summary", "")),
        }
        return {
            "completion_reason": repair_plan.terminal_reason or "needs_external_input",
            "turn_outcome": "finish_turn",
            "next_open_tool_issue": None,
            "handoff_message": self._build_tool_issue_handoff_text(open_tool_issue, repair_plan=repair_plan),
        }

    def _handle_recoverable_tool_issue(
        self,
        *,
        current_task: str,
        open_tool_issue: Dict[str, Any],
        repair_plan: RepairPlan | None,
        issue_fingerprint: str,
        next_recovery_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        progress_markers = [
            str(item).strip()
            for item in (next_recovery_state.get("progress_markers") or [])
            if str(item).strip()
        ]
        if issue_fingerprint and issue_fingerprint not in progress_markers:
            progress_markers.append(issue_fingerprint)
        next_recovery_state["progress_markers"] = progress_markers

        if not repair_plan:
            repair_plan = RepairPlan(
                strategy="llm_replan",
                reason="missing_repair_plan",
                fingerprint=issue_fingerprint or "missing-plan",
                tool_name=str((open_tool_issue.get("tool_names") or ["unknown_tool"])[0]),
                suggested_tool_name=str((open_tool_issue.get("tool_names") or ["unknown_tool"])[0]),
                original_args=dict(open_tool_issue.get("tool_args") or {}),
                patched_args=dict(open_tool_issue.get("tool_args") or {}),
                notes="Recovery fallback: inspect recent tool output and choose the best next step.",
                llm_guidance="Inspect the failure, gather more context, and continue with a different valid approach.",
            )

        strategy_id = self._repair_plan_strategy_id(repair_plan)
        attempts_by_strategy = dict(next_recovery_state.get("attempts_by_strategy") or {})
        attempt_count = int(attempts_by_strategy.get(strategy_id, 0) or 0) + 1
        attempts_by_strategy[strategy_id] = attempt_count
        next_recovery_state["attempts_by_strategy"] = attempts_by_strategy

        llm_replans = [
            str(item).strip()
            for item in (next_recovery_state.get("llm_replan_attempted_for") or [])
            if str(item).strip()
        ]
        strategy_payload = self._build_recovery_strategy(
            repair_plan=repair_plan,
            open_tool_issue=open_tool_issue,
            current_task=current_task,
            strategy_id=strategy_id,
        )

        if attempt_count == 1:
            next_recovery_state["active_issue"] = open_tool_issue
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = [strategy_payload]
            next_recovery_state["external_blocker"] = None
            return {
                "completion_reason": f"recover_{repair_plan.strategy}",
                "turn_outcome": "recover_agent",
                "next_open_tool_issue": open_tool_issue,
                "next_retry_count": attempt_count,
                "next_fingerprint_history": list(progress_markers),
            }

        if repair_plan.strategy != "llm_replan" and issue_fingerprint and issue_fingerprint not in llm_replans:
            llm_replans.append(issue_fingerprint)
            next_recovery_state["llm_replan_attempted_for"] = llm_replans
            llm_replan = RepairPlan(
                strategy="llm_replan",
                reason=f"{repair_plan.reason}_llm_replan",
                fingerprint=repair_plan.fingerprint,
                tool_name=repair_plan.tool_name,
                suggested_tool_name=repair_plan.suggested_tool_name,
                original_args=repair_plan.original_args,
                patched_args=repair_plan.patched_args,
                notes="Deterministic recovery did not clear the issue. Replan from repository state and recent tool output.",
                llm_guidance="Do not stop. Replan using repository state, recent tool failures, and alternative verification or edit paths until you either succeed or hit a real external blocker.",
                progress_fingerprint=repair_plan.progress_fingerprint,
            )
            llm_strategy_id = self._repair_plan_strategy_id(llm_replan)
            attempts_by_strategy[llm_strategy_id] = int(attempts_by_strategy.get(llm_strategy_id, 0) or 0) + 1
            next_recovery_state["attempts_by_strategy"] = attempts_by_strategy
            next_recovery_state["active_issue"] = open_tool_issue
            next_recovery_state["active_strategy"] = None
            next_recovery_state["strategy_queue"] = [
                self._build_recovery_strategy(
                    repair_plan=llm_replan,
                    open_tool_issue=open_tool_issue,
                    current_task=current_task,
                    strategy_id=llm_strategy_id,
                )
            ]
            next_recovery_state["external_blocker"] = None
            return {
                "completion_reason": "recover_llm_replan",
                "turn_outcome": "recover_agent",
                "next_open_tool_issue": open_tool_issue,
                "next_retry_count": max(int(value or 0) for value in attempts_by_strategy.values()),
                "next_fingerprint_history": list(progress_markers),
            }

        next_recovery_state["active_issue"] = open_tool_issue
        next_recovery_state["active_strategy"] = None
        next_recovery_state["strategy_queue"] = []
        next_recovery_state["external_blocker"] = {
            "reason": "recovery_stagnated",
            "issue_summary": str(open_tool_issue.get("summary", "")),
        }
        return {
            "completion_reason": "recovery_stagnated",
            "turn_outcome": "finish_turn",
            "next_open_tool_issue": None,
            "handoff_message": self._build_tool_issue_handoff_text(open_tool_issue, repair_plan=repair_plan),
            "next_retry_count": max(int(value or 0) for value in attempts_by_strategy.values()),
            "next_fingerprint_history": list(progress_markers),
        }

    # --- NODE: STABILITY GUARD ---

    async def stability_guard_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "stability_guard",
            message_count=len(state.get("messages") or []),
            has_open_tool_issue=bool(state.get("open_tool_issue")),
        )
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        current_task = self._resolve_current_task(state, messages).strip()
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
        last_ai = self._get_last_ai_message(messages)
        last_message = messages[-1] if messages else None
        step_count = int(state.get("steps", 0) or 0)
        recovery_state = self._get_recovery_state(state, current_turn_id=current_turn_id)
        hard_loop_ceiling = self._hard_loop_ceiling()

        try:
            result = self.recovery_manager.plan_recovery(
                state=state,
                messages=messages,
                current_task=current_task,
                current_turn_id=current_turn_id,
                open_tool_issue=open_tool_issue,
                recovery_state=recovery_state,
                last_ai=last_ai,
                last_message=last_message,
                step_count=step_count,
                max_loops=int(self.config.max_loops or 0),
                hard_loop_ceiling=hard_loop_ceiling,
                auto_repair_enabled=bool(self.config.self_correction_enable_auto_repair),
                max_auto_repairs=int(self.config.self_correction_max_auto_repairs or 2),
                successful_tool_stagnation_limit=self._successful_tool_stagnation_limit(
                    str(getattr(last_message, "name", "") or "")
                ),
            )
            outbound_messages: List[BaseMessage] = []
            if (
                result["drop_trailing_tool_call"]
                and last_ai
                and getattr(last_ai, "tool_calls", None)
                and getattr(last_ai, "id", None)
            ):
                outbound_messages.append(RemoveMessage(id=last_ai.id))
            if result["turn_outcome"] == "finish_turn" and result["handoff_message"]:
                handoff_kind = (
                    "loop_budget_handoff"
                    if str(result["completion_reason"]).startswith("loop_budget_exhausted")
                    else "tool_issue_handoff"
                )
                handoff_metadata: Dict[str, Any] = {
                    "agent_internal": {
                        "kind": handoff_kind,
                        "turn_id": current_turn_id,
                        "visible_in_ui": False,
                        "ui_notice": self.recovery_manager.build_internal_ui_notice(
                            str(result["completion_reason"])
                        ),
                    }
                }
                outbound_messages.append(
                    AIMessage(
                        content=result["handoff_message"],
                        additional_kwargs=handoff_metadata,
                    )
                )

            self._log_run_event(
                state,
                "stability_guard_verdict",
                run_id=state.get("run_id", ""),
                outcome=result["turn_outcome"],
                reason=result["completion_reason"],
                retry_count_before=int(state.get("self_correction_retry_count", 0) or 0),
                retry_count_after=result["self_correction_retry_count"],
                has_open_tool_issue=bool(open_tool_issue),
                loop_budget_reached=result["loop_budget_reached"],
                had_pending_tool_calls=result["had_pending_tool_calls"],
                successful_tool_repeat_count=result.get("successful_tool_repeat_count", 0),
                successful_tool_name=result.get("successful_tool_name", ""),
            )
            self._log_node_end(
                state,
                "stability_guard",
                node_timer,
                turn_outcome=result["turn_outcome"],
                reason=result["completion_reason"],
                retry_count=result["self_correction_retry_count"],
                has_open_tool_issue=bool(open_tool_issue),
                loop_budget_reached=result["loop_budget_reached"],
                had_pending_tool_calls=result["had_pending_tool_calls"],
                successful_tool_repeat_count=result.get("successful_tool_repeat_count", 0),
                successful_tool_name=result.get("successful_tool_name", ""),
            )
            if outbound_messages:
                result["messages"] = outbound_messages
            return result
        except Exception as e:
            self._log_node_error(
                state,
                "stability_guard",
                node_timer,
                e,
                has_open_tool_issue=bool(open_tool_issue),
                retry_count=int(state.get("self_correction_retry_count", 0) or 0),
            )
            raise

    async def recovery_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "recovery",
            has_recovery_state=bool(state.get("recovery_state")),
        )
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        recovery_state = self._get_recovery_state(state, current_turn_id=current_turn_id)
        result = self.recovery_manager.apply_recovery(recovery_state, current_turn_id=current_turn_id)
        if result["recovery_status"] == "empty_strategy_queue":
            self._log_node_end(
                state,
                "recovery",
                node_timer,
                outcome="finish_turn",
                reason="empty_strategy_queue",
            )
            return {
                "turn_outcome": "finish_turn",
                "recovery_state": recovery_state,
            }
        self._log_run_event(
            state,
            "recovery_prepared",
            run_id=state.get("run_id", ""),
            turn_id=current_turn_id,
            strategy_id=str((result.get("active_strategy") or {}).get("id") or ""),
            strategy=str((result.get("active_strategy") or {}).get("strategy") or ""),
            suggested_tool=str((result.get("active_strategy") or {}).get("suggested_tool_name") or ""),
        )
        self._log_node_end(
            state,
            "recovery",
            node_timer,
            outcome="recovery_prepared",
            turn_id=current_turn_id,
        )
        return {
            "turn_outcome": "",
            "recovery_state": result["recovery_state"],
        }

    async def approval_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "approval",
            message_count=len(state.get("messages") or []),
        )
        messages = state.get("messages", [])
        if not messages:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="no_messages",
            )
            return {"pending_approval": None}

        pending_ai_with_tools = self._get_last_pending_ai_with_tool_calls(messages)
        if not pending_ai_with_tools:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="no_protected_tool_calls",
            )
            return {"pending_approval": None}

        protected_calls = []
        for tool_call in pending_ai_with_tools.tool_calls:
            tool_name = tool_call.get("name") or "unknown_tool"
            tool_args = tool_call.get("args") or {}
            if not self._tool_requires_approval(tool_name, tool_args):
                continue
            metadata = self._effective_tool_metadata(tool_name, tool_args)
            protected_calls.append(
                {
                    "id": tool_call.get("id") or "",
                    "name": tool_name,
                    "args": tool_args,
                    "policy": metadata.to_dict(),
                }
            )

        if not protected_calls:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="all_tools_readonly",
            )
            return {"pending_approval": None}

        payload = {
            "kind": "tool_approval",
            "message": "Approve protected tool execution?",
            "tools": protected_calls,
            "run_id": state.get("run_id", ""),
            "session_id": state.get("session_id", ""),
        }
        self._log_run_event(
            state,
            "approval_requested",
            run_id=state.get("run_id", ""),
            tool_names=[tool["name"] for tool in protected_calls],
        )
        decision = interrupt(payload)
        approved = self._approval_decision_is_approved(decision)
        approval_state = {
            "approved": approved,
            "decision": decision,
            "tool_call_ids": [tool["id"] for tool in protected_calls if tool["id"]],
            "tool_names": [tool["name"] for tool in protected_calls],
        }
        self._log_run_event(
            state,
            "approval_resolved",
            run_id=state.get("run_id", ""),
            approved=approved,
            tool_names=approval_state["tool_names"],
        )
        self._log_node_end(
            state,
            "approval",
            node_timer,
            outcome="resolved",
            approved=approved,
            protected_count=len(protected_calls),
        )
        return {"pending_approval": approval_state}

    def _approval_decision_is_approved(self, decision: Any) -> bool:
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, dict):
            if "approved" in decision:
                return bool(decision.get("approved"))
            action = str(decision.get("action", "")).strip().lower()
            return action in {"approve", "approved", "yes", "y"}
        return bool(decision)

    # --- NODE: TOOLS ---

    async def tools_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "tools",
            message_count=len(state.get("messages") or []),
            has_pending_approval=bool(state.get("pending_approval")),
        )
        self._check_invariants(state)

        messages = state["messages"]
        last_msg = self._get_last_pending_ai_with_tool_calls(messages)
        current_turn_id = self._current_turn_id(state, messages)

        if not last_msg:
            self._log_node_end(
                state,
                "tools",
                node_timer,
                outcome="skipped",
                reason="no_tool_calls",
            )
            return {}

        final_messages: List[ToolMessage] = []
        has_error = False
        last_error = ""
        last_result = ""
        tool_issues: List[Dict[str, Any]] = []
        approval_state = state.get("pending_approval") or {}
        _, active_tool_names = self._active_tools_for_turn(messages)

        # Оптимизация: собираем историю вызовов один раз, а не для каждого инструмента.
        # ВАЖНО: исключаем последний AI message, чтобы текущий вызов не считался "повтором".
        recent_calls = []
        history_window = self.config.effective_tool_loop_window
        history_slice = messages[-(history_window + 1):-1] if history_window > 0 else messages[:-1]
        for m in reversed(history_slice):
            if isinstance(m, AIMessage) and m.tool_calls:
                recent_calls.extend(m.tool_calls)

        tool_calls = list(last_msg.tool_calls)

        parallel_mode = self._can_parallelize_tool_calls(tool_calls)
        self._log_run_event(
            state,
            "tools_node_start",
            run_id=state.get("run_id", ""),
            tool_call_count=len(tool_calls),
            tool_names=[(tc.get("name") or "unknown_tool") for tc in tool_calls],
            parallel_mode=parallel_mode,
        )
        try:
            if parallel_mode:
                processed = await asyncio.gather(
                    *(
                        self._process_tool_call(
                            tool_call,
                            recent_calls,
                            state,
                            approval_state,
                            current_turn_id,
                            active_tool_names,
                        )
                        for tool_call in tool_calls
                    )
                )
                for tool_msg, had_error, issue in processed:
                    final_messages.append(tool_msg)
                    has_error = has_error or had_error
                    if issue:
                        tool_issues.append(issue)
                    parsed = parse_tool_execution_result(tool_msg.content)
                    if parsed.ok:
                        last_result = parsed.message
                    else:
                        last_error = parsed.message
            else:
                for tool_call in tool_calls:
                    tool_msg, had_error, issue = await self._process_tool_call(
                        tool_call,
                        recent_calls,
                        state,
                        approval_state,
                        current_turn_id,
                        active_tool_names,
                    )
                    final_messages.append(tool_msg)
                    has_error = has_error or had_error
                    if issue:
                        tool_issues.append(issue)
                    parsed = parse_tool_execution_result(tool_msg.content)
                    if parsed.ok:
                        last_result = parsed.message
                    else:
                        last_error = parsed.message

            merged_issue = self._merge_open_tool_issues(tool_issues, current_turn_id)
            self._log_run_event(
                state,
                "tools_node_end",
                run_id=state.get("run_id", ""),
                tool_result_count=len(final_messages),
                has_error=has_error,
                issue_kind="" if not merged_issue else merged_issue.get("kind", ""),
                issue_source="" if not merged_issue else merged_issue.get("source", ""),
            )
            self._log_node_end(
                state,
                "tools",
                node_timer,
                tool_call_count=len(tool_calls),
                tool_result_count=len(final_messages),
                parallel_mode=parallel_mode,
                has_error=has_error,
                has_open_tool_issue=bool(merged_issue),
            )
            return {
                "messages": final_messages,
                "turn_id": current_turn_id,
                "turn_outcome": "run_tools",
                "pending_approval": None,
                "open_tool_issue": merged_issue,
                "has_protocol_error": False,
                "last_tool_error": last_error,
                "last_tool_result": last_result,
            }
        except Exception as e:
            self._log_node_error(
                state,
                "tools",
                node_timer,
                e,
                tool_call_count=len(tool_calls),
                parallel_mode=parallel_mode,
            )
            raise

    def _can_parallelize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> bool:
        if len(tool_calls) < 2:
            return False

        # Both conditions required: metadata says read-only AND name is on the explicit whitelist.
        # The whitelist acts as a second safety gate — unknown / newly-added tools default to sequential.
        return all(
            self._tool_is_read_only(tc.get("name") or "unknown_tool")
            and (tc.get("name") or "") in self.PARALLEL_SAFE_TOOL_NAMES
            for tc in tool_calls
        )

    def _tool_is_allowed_for_turn(
        self,
        tool_name: str,
        allowed_tool_names: List[str] | None = None,
    ) -> bool:
        if allowed_tool_names is not None:
            return self._normalize_tool_name(tool_name) in {
                self._normalize_tool_name(name) for name in allowed_tool_names
            }
        return (
            bool(self.config.model_supports_tools)
            and bool(self._all_tool_names)
            and self._normalize_tool_name(tool_name) in {
                self._normalize_tool_name(name) for name in self._all_tool_names
            }
        )

    async def _process_tool_call(
        self,
        tool_call: Dict[str, Any],
        recent_calls: List[Dict[str, Any]],
        state: AgentState,
        approval_state: Dict[str, Any],
        current_turn_id: int,
        allowed_tool_names: List[str] | None = None,
    ) -> Tuple[ToolMessage, bool, Dict[str, Any] | None]:
        # Безопасное извлечение с фоллбеками
        t_name = tool_call.get("name") or "unknown_tool"
        t_args = tool_call.get("args") or {}
        normalized_args, normalized_changes = normalize_tool_args(
            t_name,
            t_args if isinstance(t_args, dict) else {},
            current_task=str(state.get("current_task") or ""),
        )
        if normalized_changes:
            self._log_run_event(
                state,
                "tool_call_args_repaired",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                original_args=t_args,
                patched_args=normalized_args,
                changes=normalized_changes,
            )
            t_args = normalized_args

        # Генерируем фейковый ID, если LLM забыла его указать, чтобы Pydantic не упал
        t_id = tool_call.get("id")
        if not t_id:
            t_id = f"call_missing_{uuid.uuid4().hex[:8]}"

        had_error = False
        tool_duration_seconds: float | None = None
        metadata = self._effective_tool_metadata(t_name, t_args)
        active_tool_names = (
            list(allowed_tool_names)
            if allowed_tool_names is not None
            else (list(self._all_tool_names) if self.config.model_supports_tools else [])
        )

        if not self._tool_is_allowed_for_turn(t_name, allowed_tool_names):
            outcome = self.tool_executor.build_not_allowed_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=t_name,
                tool_args=t_args,
                tool_call_id=t_id,
                allowed_tool_names=active_tool_names,
            )
            return (
                outcome.tool_message,
                outcome.had_error,
                outcome.issue,
            )

        if self._tool_requires_approval(t_name, t_args) and not self._tool_call_is_approved(t_id, approval_state):
            outcome = self.tool_executor.build_denied_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=t_name,
                tool_args=t_args,
                tool_call_id=t_id,
                policy=metadata.to_dict(),
            )
            return (
                outcome.tool_message,
                outcome.had_error,
                outcome.issue,
            )

        missing_required = self._missing_required_tool_fields(t_name, t_args)
        if missing_required:
            outcome = self.tool_executor.build_missing_required_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=t_name,
                tool_args=t_args,
                tool_call_id=t_id,
                missing_required=missing_required,
            )
            return (
                outcome.tool_message,
                outcome.had_error,
                outcome.issue,
            )

        # Проверка на зацикливание
        loop_count = sum(
            1 for tc in recent_calls if tc.get("name") == t_name and tc.get("args") == t_args
        )
        same_tool_count = sum(
            1 for tc in recent_calls
            if self._normalize_tool_name(tc.get("name") or "") == self._normalize_tool_name(t_name)
        )

        loop_limit = (
            self.config.effective_tool_loop_limit_readonly
            if t_name in self.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES
            else self.config.effective_tool_loop_limit_mutating
        )
        planning_limit_reached = self._is_planning_tool(t_name) and (
            same_tool_count >= self.PLANNING_TOOL_MAX_CALLS_PER_TURN
        )

        if planning_limit_reached:
            content = format_error(
                ErrorType.LOOP_DETECTED,
                (
                    f"Planning tool budget reached for '{t_name}'. "
                    "Stop further planning tool calls in this turn and proceed with concrete action tools."
                ),
            )
            had_error = True
            self._log_run_event(
                state,
                "tool_call_planning_budget_blocked",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                same_tool_count=same_tool_count,
                planning_limit=self.PLANNING_TOOL_MAX_CALLS_PER_TURN,
            )
        elif loop_count >= loop_limit:
            content = format_error(
                ErrorType.LOOP_DETECTED,
                f"Loop detected. You have called '{t_name}' with these exact arguments {loop_limit} times in the recent history. Please try a different approach.",
            )
            had_error = True
            self._log_run_event(
                state,
                "tool_call_loop_blocked",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                loop_count=loop_count,
                loop_limit=loop_limit,
            )
        else:
            self._log_run_event(
                state,
                "tool_call_start",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                policy=metadata.to_dict(),
            )
            started_at = time.perf_counter()
            content = await self._execute_tool(t_name, t_args, state=state, tool_call_id=t_id)
            tool_duration_seconds = max(0.0, time.perf_counter() - started_at)

        outcome = self.tool_executor.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=t_name,
            tool_args=t_args,
            tool_call_id=t_id,
            content=content,
            tool_duration_seconds=tool_duration_seconds,
            had_error=had_error,
            issue_details=(
                {
                    "loop_detected": True,
                    "loop_count": loop_count,
                    "loop_limit": (
                        self.PLANNING_TOOL_MAX_CALLS_PER_TURN if planning_limit_reached else loop_limit
                    ),
                }
                if planning_limit_reached or loop_count >= loop_limit
                else None
            ),
        )

        return (
            outcome.tool_message,
            outcome.had_error,
            outcome.issue,
        )

    def _build_tool_message(
        self,
        *,
        content: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_duration_seconds: float | None = None,
    ) -> ToolMessage:
        parsed_result = parse_tool_execution_result(content)
        additional_kwargs: Dict[str, Any] = {
            "tool_args": deepcopy(tool_args) if isinstance(tool_args, dict) else {}
        }
        if tool_duration_seconds is not None:
            additional_kwargs["tool_duration_seconds"] = float(tool_duration_seconds)
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            additional_kwargs=additional_kwargs,
            status="error" if not parsed_result.ok else "success",
        )

    def _tool_call_is_approved(self, tool_call_id: str, approval_state: Dict[str, Any]) -> bool:
        if not self.config.enable_approvals:
            return True
        if not approval_state:
            return False
        if not approval_state.get("approved"):
            return False
        approved_ids = set(approval_state.get("tool_call_ids") or [])
        return not approved_ids or tool_call_id in approved_ids

    def _check_invariants(self, state: AgentState):
        if not self.config.debug:
            return
        steps = state.get("steps", 0)
        if steps < 0:
            logger.error(f"INVARIANT VIOLATION: steps ({steps}) < 0")

    def _get_last_model_visible_message(self, context: List[BaseMessage]) -> BaseMessage | None:
        return self.context_builder.get_last_model_visible_message(context)

    def _assert_provider_safe_agent_context(
        self,
        context: List[BaseMessage],
        state: AgentState | None = None,
    ) -> None:
        try:
            self.context_builder.assert_provider_safe_context(context, state=state)
        except RuntimeError as exc:
            raise ProviderContextError(str(exc)) from exc

    async def _execute_tool(
        self,
        name: str,
        args: dict,
        state: AgentState | None = None,
        tool_call_id: str = "",
    ) -> str:
        # Быстрый поиск за O(1)
        tool = self.tools_map.get(name)
        if not tool:
            self._log_run_event(
                state,
                "tool_call_missing",
                run_id=None if state is None else state.get("run_id", ""),
                tool_name=name,
                tool_args=args,
            )
            return format_error(ErrorType.NOT_FOUND, f"Tool '{name}' not found.")
        try:
            invoke_scope = nullcontext()
            if name == "cli_exec" and tool_call_id:
                try:
                    from tools.local_shell import cli_output_context

                    invoke_scope = cli_output_context(tool_call_id)
                except Exception:
                    invoke_scope = nullcontext()

            with invoke_scope:
                raw_result = await tool.ainvoke(args)
            content = str(raw_result)
            if not content.strip():
                self._log_run_event(
                    state,
                    "tool_call_empty_result",
                    run_id=None if state is None else state.get("run_id", ""),
                    tool_name=name,
                    tool_args=args,
                )
                return format_error(ErrorType.EXECUTION, "Tool returned empty response.")
            return content
        except asyncio.CancelledError:
            self._log_run_event(
                state,
                "tool_call_cancelled",
                run_id=None if state is None else state.get("run_id", ""),
                tool_name=name,
                tool_args=args,
                tool_call_id=tool_call_id,
                reason="task_cancelled",
            )
            raise
        except GraphInterrupt:
            raise
        except Exception as e:
            self._log_run_event(
                state,
                "tool_call_exception",
                run_id=None if state is None else state.get("run_id", ""),
                tool_name=name,
                tool_args=args,
                error_type=type(e).__name__,
                error=compact_text(str(e), 400),
            )
            return format_error(ErrorType.EXECUTION, str(e))

    # --- HELPERS ---

    def _derive_current_task(self, messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and not self._is_internal_retry_message(message):
                content = stringify_content(message.content).strip()
                if content and content != constants.REFLECTION_PROMPT:
                    return content
        return ""

    def _resolve_current_task(self, state: AgentState | None, messages: List[BaseMessage]) -> str:
        derived_task = self._derive_current_task(messages).strip()
        stored_task = str((state or {}).get("current_task") or "").strip()
        if derived_task:
            if stored_task and stored_task != derived_task:
                self._log_run_event(
                    state,
                    "current_task_overridden_by_latest_user_message",
                    run_id=None if state is None else state.get("run_id", ""),
                    previous_task=compact_text(stored_task, 180),
                    latest_user_message=compact_text(derived_task, 180),
                )
            return derived_task
        return stored_task

    def _active_tools_for_turn(self, messages: List[BaseMessage]) -> Tuple[List[BaseTool], List[str]]:
        if not self.config.model_supports_tools:
            return [], []

        active_tools = list(self.tools)
        if self._current_turn_has_completed_user_choice(messages):
            active_tools = [
                tool
                for tool in active_tools
                if self._normalize_tool_name(tool.name) != "request_user_input"
            ]
        return active_tools, [tool.name for tool in active_tools]

    def _current_turn_has_completed_user_choice(self, messages: List[BaseMessage]) -> bool:
        human_indexes = self.message_context.non_internal_human_indexes(
            messages,
            self._is_internal_retry_message,
        )
        if not human_indexes:
            return False

        start_idx = human_indexes[-1] + 1
        for message in messages[start_idx:]:
            if isinstance(message, ToolMessage):
                if self._normalize_tool_name(message.name or "") == "request_user_input":
                    return True
        return False

    def _current_turn_has_user_input_request(self, messages: List[BaseMessage]) -> bool:
        human_indexes = self.message_context.non_internal_human_indexes(
            messages,
            self._is_internal_retry_message,
        )
        if not human_indexes:
            return False

        start_idx = human_indexes[-1] + 1
        for message in messages[start_idx:]:
            if isinstance(message, ToolMessage):
                if self._normalize_tool_name(message.name or "") == "request_user_input":
                    return True
                continue
            if isinstance(message, (AIMessage, AIMessageChunk)):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    if self._normalize_tool_name(tool_call.get("name") or "") == "request_user_input":
                        return True
        return False

    def _build_tool_issue_handoff_text(
        self,
        open_tool_issue: Dict[str, Any] | None,
        *,
        repair_plan: RepairPlan | None = None,
    ) -> str:
        return self.recovery_manager.build_tool_issue_handoff_text(
            open_tool_issue,
            repair_plan=repair_plan,
        )

    def _build_loop_budget_handoff_text(self, current_task: str, tool_names: List[str]) -> str:
        return self.recovery_manager.build_loop_budget_handoff_text(current_task, tool_names)

    def _successful_tool_stagnation_limit(self, tool_name: str) -> int:
        return 3 if self._tool_is_read_only(tool_name) else 2

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _get_last_pending_ai_with_tool_calls(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        """
        Returns the latest AI message that still has at least one unresolved tool call.
        This is robust to provider id collisions where the newest AI message may replace
        an older message and stop being the last list item.
        """
        resolved_tool_call_ids: set[str] = {
            str(message.tool_call_id or "").strip()
            for message in messages
            if isinstance(message, ToolMessage) and str(message.tool_call_id or "").strip()
        }

        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            tool_calls = list(getattr(message, "tool_calls", []) or [])
            if not tool_calls:
                continue
            unresolved_calls = [
                call for call in tool_calls
                if not str(call.get("id") or "").strip()
                or str(call.get("id") or "").strip() not in resolved_tool_call_ids
            ]
            if unresolved_calls:
                return message
        return None

    def _get_base_prompt(self) -> str:
        """Ленивая загрузка и кэширование промпта для устранения дискового I/O"""
        if self._cached_base_prompt is None:
            prompt_path = self.config.prompt_path.absolute()
            if self.config.prompt_path.exists():
                try:
                    self._cached_base_prompt = self.config.prompt_path.read_text("utf-8")
                    logger.info("Loaded prompt from file: %s", prompt_path)
                except Exception as e:
                    logger.warning(
                        "Failed to read prompt file %s: %s. Using built-in prompt.",
                        prompt_path,
                        e,
                    )
                    self._cached_base_prompt = (
                        "You are an autonomous AI agent.\n"
                        "Reason in English, Reply in Russian.\n"
                        "Date: {{current_date}}"
                    )
            else:
                logger.info("Prompt file not found at %s. Using built-in prompt.", prompt_path)
                self._cached_base_prompt = (
                    "You are an autonomous AI agent.\n"
                    "Reason in English, Reply in Russian.\n"
                    "Date: {{current_date}}"
                )
        return self._cached_base_prompt

    async def _invoke_llm_with_retry(
        self,
        llm,
        context: List[BaseMessage],
        state: AgentState | None = None,
        node_name: str = "",
    ) -> AIMessage:
        current_llm = llm
        context = list(context)
        max_attempts = max(1, self.config.max_retries)
        retry_delay = max(0, self.config.retry_delay)
        self._log_run_event(
            state,
            "llm_invoke_start",
            run_id=None if state is None else state.get("run_id", ""),
            node=node_name,
            max_attempts=max_attempts,
            context_messages=len(context),
        )

        for attempt in range(max_attempts):
            try:
                normalized_context = self._normalize_system_prefix_for_provider(context)
                response = await current_llm.ainvoke(normalized_context)
                invalid_calls = getattr(response, "invalid_tool_calls", None)
                if not response.content and not response.tool_calls and not invalid_calls:
                    raise ValueError("Empty response from LLM")
                self._log_run_event(
                    state,
                    "llm_invoke_success",
                    run_id=None if state is None else state.get("run_id", ""),
                    node=node_name,
                    attempt=attempt + 1,
                    has_content=bool(stringify_content(response.content).strip()),
                    tool_calls=len(getattr(response, "tool_calls", []) or []),
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "auto" in err_str and "tool choice" in err_str and "requires" in err_str:
                    logger.warning(
                        "⚠ Server does not support 'auto' tool choice. Falling back to chat-only mode."
                    )
                    current_llm = self.llm
                    # Безопасное копирование контекста
                    context = list(context)
                    if isinstance(context[0], SystemMessage):
                        context[0] = SystemMessage(
                            content=str(context[0].content)
                            + "\n\nWARNING: Tools are disabled due to server configuration error."
                        )
                    continue

                is_fatal = self._is_fatal_llm_error(e)
                logger.warning(f"LLM Error (Attempt {attempt+1}/{max_attempts}): {e}")
                self._log_run_event(
                    state,
                    "llm_retry",
                    node=node_name,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    fatal=is_fatal,
                    error=str(e),
                )

                if is_fatal:
                    logger.error(f"Fatal LLM error detected. Aborting request: {e}")
                    self._log_run_event(
                        state,
                        "llm_invoke_fatal",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        error=compact_text(str(e), 400),
                    )
                    raise

                if attempt == max_attempts - 1:
                    self._log_run_event(
                        state,
                        "llm_invoke_exhausted",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        error=compact_text(str(e), 400),
                    )
                    raise

                await asyncio.sleep(retry_delay)

        raise RuntimeError("LLM retry loop exited unexpectedly without a response.")

    def _is_fatal_llm_error(self, error: Exception) -> bool:
        err_str = " ".join(str(error).lower().split())
        fatal_markers = (
            "insufficient_balance",
            "insufficient account balance",
            "invalid_api_key",
            "incorrect api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "permission denied",
            "billing",
            "payment required",
            "error code: 401",
            "error code: 402",
            "error code: 403",
        )
        return any(marker in err_str for marker in fatal_markers)
