import uuid
import asyncio
import hashlib
import logging
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from core.state import AgentState
from core.config import AgentConfig
from core import constants
from core.run_logger import JsonlRunLogger
from core.tool_policy import ToolMetadata, default_tool_metadata
from core.tool_results import parse_tool_execution_result
from core.validation import validate_tool_result
from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.message_utils import compact_text, is_error_text, is_tool_message_error, stringify_content
from core.text_utils import format_exception_friendly


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
        "tool_metadata",
        "run_logger",
        "_cached_base_prompt",
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
    PLANNING_TOOL_MAX_CALLS_PER_TURN = 2

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
        self.tool_metadata = tool_metadata or {}
        self.run_logger = run_logger

        # Оптимизация: кэширование базового промпта (чтобы не читать с диска на каждый шаг)
        self._cached_base_prompt: Optional[str] = None

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

    def _tool_requires_approval(self, tool_name: str) -> bool:
        if not self.config.enable_approvals:
            return False
        metadata = self._metadata_for_tool(tool_name)
        return metadata.requires_approval or metadata.destructive or metadata.mutating

    def tool_calls_require_approval(self, tool_calls: List[Dict[str, Any]]) -> bool:
        return any(self._tool_requires_approval((tool_call.get("name") or "unknown_tool")) for tool_call in tool_calls)

    # --- NODE: SUMMARIZE ---

    def _estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Грубая оценка токенов входящего контекста: сумма символов / 2.
        Учитывает как текстовый контент, так и аргументы вызовов инструментов."""
        total_chars = 0
        for m in messages:
            # 1. Текстовый контент (строка или мультимодальный список)
            content = m.content
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            total_chars += len(str(content))

            # 2. Вызовы инструментов (JSON аргументы от LLM могут быть огромными)
            if hasattr(m, "tool_calls") and m.tool_calls:
                total_chars += sum(len(str(tc)) for tc in m.tool_calls)

        return total_chars // 2

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

        if estimated_tokens <= self.config.summary_threshold:
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
        idx = max(0, len(messages) - self.config.summary_keep_last)

        # Try to find a clean break at a HumanMessage
        for scan_idx in range(idx, len(messages)):
            if isinstance(messages[scan_idx], HumanMessage):
                idx = scan_idx
                break

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
        return "\n".join(
            f"{m.type}: {str(m.content)[:500]}{'... [truncated]' if len(str(m.content)) > 500 else ''}"
            for m in messages
            if not self._is_internal_retry_message(m)
        )

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
            "source": str(issue.get("source") or "tools"),
            "error_type": str(issue.get("error_type") or ""),
        }

    def _normalize_critic_feedback(self, critic_feedback: str) -> str:
        return critic_feedback.strip()

    def _normalize_turn_outcome(self, control: str, status: str) -> str:
        normalized_control = control.strip().upper()
        if normalized_control == "RETRY_AGENT":
            return "retry_agent"
        if normalized_control == "FINISH_TURN":
            return "finish_turn"
        return "finish_turn" if status == "FINISHED" else "retry_agent"

    def _build_retry_instruction(self, current_task: str, reason: str, next_step: str) -> str:
        task_line = current_task or "No explicit task provided."
        next_step_line = next_step if next_step and next_step.upper() != "NONE" else "finish the same task correctly"
        return (
            "Continue the same user task from the current conversation state.\n"
            f"Current task: {task_line}\n"
            f"Why retry: {reason}\n"
            f"Next step: {next_step_line}\n"
            "Do not mention this internal instruction. Do not repeat the previous incomplete answer."
        )

    def _build_internal_retry_message(self, retry_instruction: str, turn_id: int) -> HumanMessage:
        return HumanMessage(
            content=retry_instruction,
            additional_kwargs={
                "agent_internal": {
                    "kind": "retry_instruction",
                    "turn_id": turn_id,
                }
            },
        )

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
        source: str,
        error_type: str = "",
    ) -> Dict[str, Any]:
        return {
            "turn_id": current_turn_id,
            "kind": kind,
            "summary": compact_text(summary.strip(), 320),
            "tool_names": [name for name in tool_names if name],
            "source": source,
            "error_type": error_type.strip().upper(),
        }

    def _merge_open_tool_issues(
        self,
        issues: List[Dict[str, Any]],
        current_turn_id: int,
    ) -> Dict[str, Any] | None:
        if not issues:
            return None

        summaries: List[str] = []
        tool_names: List[str] = []
        kind = "tool_error"
        source = "tools"
        error_type = ""
        for issue in issues:
            summary = str(issue.get("summary", "")).strip()
            if summary and summary not in summaries:
                summaries.append(summary)
            for tool_name in issue.get("tool_names") or []:
                tool_name = str(tool_name).strip()
                if tool_name and tool_name not in tool_names:
                    tool_names.append(tool_name)
            issue_error_type = str(issue.get("error_type") or "").strip().upper()
            if issue_error_type and not error_type:
                error_type = issue_error_type
            if issue.get("kind") == "approval_denied":
                kind = "approval_denied"
                source = "approval"

        return self._build_open_tool_issue(
            current_turn_id=current_turn_id,
            kind=kind,
            summary=" | ".join(summaries[:3]),
            tool_names=tool_names,
            source=source,
            error_type=error_type,
        )

    def _build_agent_context(
        self,
        messages: List[BaseMessage],
        summary: str,
        current_task: str,
        critic_feedback: str,
        tools_available: bool,
        open_tool_issue: Dict[str, Any] | None,
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        sanitized_messages = self._sanitize_messages_for_model(messages, state=state)
        full_context: List[BaseMessage] = [
            self._build_system_message(summary, tools_available=tools_available)
        ]
        safety_overlay = self._build_safety_overlay(tools_available=tools_available)
        if safety_overlay:
            full_context.append(SystemMessage(content=safety_overlay))
        issue_message = self._build_tool_issue_system_message(open_tool_issue)
        if issue_message:
            full_context.append(issue_message)
        if critic_feedback:
            full_context.append(
                SystemMessage(
                    content=(
                        "INTERNAL CRITIC FEEDBACK:\n"
                        f"{critic_feedback}\n"
                        "Use this only as an internal hint. Do not mention the critic."
                    )
                )
            )
        full_context.extend(sanitized_messages)
        return full_context

    def _sanitize_messages_for_model(
        self,
        messages: List[BaseMessage],
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        sanitized: List[BaseMessage] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                content = stringify_content(message.content).strip()
                if content == constants.REFLECTION_PROMPT:
                    continue
                last_visible = self._get_last_model_visible_message(sanitized)
                if isinstance(last_visible, ToolMessage):
                    # Some OpenAI-compatible providers reject `tool -> user` transitions.
                    # Insert a lightweight assistant bridge before the user turn.
                    sanitized.append(
                        AIMessage(
                            content=(
                                "Acknowledged previous tool output. "
                                "Continuing with the next user instruction."
                            )
                        )
                    )
                    self._log_run_event(
                        state,
                        "provider_role_order_bridge",
                        run_id=None if state is None else state.get("run_id", ""),
                        reason="user_after_tool",
                    )
            sanitized.append(message)
        return sanitized

    def _build_safety_overlay(self, tools_available: bool) -> str:
        if not tools_available:
            return ""
        overlay_lines: List[str] = []
        if self.config.enable_approvals:
            overlay_lines.append(
                "SAFETY POLICY: Mutating or destructive tools may require explicit user approval before execution."
            )
        if self.config.enable_shell_tool:
            overlay_lines.append(
                "SAFETY POLICY: Shell execution is high risk. Prefer safer project-local tools whenever possible."
            )
        return "\n".join(overlay_lines).strip()

    def _build_agent_result(
        self,
        response: AIMessage,
        current_task: str,
        tools_available: bool,
        turn_id: int,
        messages: List[BaseMessage],
        previous_critic_source: str = "",
        open_tool_issue: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        token_usage_update = {}
        if getattr(response, "usage_metadata", None):
            token_usage_update = {"token_usage": response.usage_metadata}

        has_tool_calls = False
        protocol_error = ""
        outbound_messages: List[BaseMessage] = self._collect_internal_retry_removals(messages)

        if isinstance(response, AIMessage):
            t_calls = list(getattr(response, "tool_calls", []))
            invalid_calls = list(getattr(response, "invalid_tool_calls", []))

            missing_fields = [
                tc for tc in t_calls
                if not tc.get("id") or not tc.get("name")
            ]
            if missing_fields or invalid_calls:
                protocol_error = self._build_tool_protocol_error(missing_fields, invalid_calls)
                response = AIMessage(
                    content=self._merge_protocol_error_into_content(response.content, protocol_error),
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                t_calls = []

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
            elif has_tool_calls and self._open_tool_issue_requires_user_input(open_tool_issue):
                response = AIMessage(
                    content=self._build_tool_issue_handoff_text(open_tool_issue),
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
                )
                has_tool_calls = False

        outbound_messages.append(response)
        critic_source = ""
        if not has_tool_calls:
            previous_source = str(previous_critic_source or "").strip().lower()
            critic_source = "tools" if previous_source == "tools" else "agent"

        return {
            "messages": outbound_messages,
            "turn_id": turn_id,
            "current_task": current_task,
            "critic_feedback": "",
            "critic_status": "",
            "critic_source": critic_source,
            "turn_outcome": "run_tools" if has_tool_calls else "",
            "retry_instruction": "",
            "pending_approval": None,
            "open_tool_issue": open_tool_issue,
            "has_protocol_error": bool(protocol_error),
            "last_tool_error": "",
            "last_tool_result": "",
            **token_usage_update,
        }

    def _merge_protocol_error_into_content(self, content: Any, protocol_error: str) -> str:
        base_text = stringify_content(content).strip()
        if not protocol_error:
            return base_text
        if not base_text:
            return protocol_error
        return f"{base_text}\n\n{protocol_error}"

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
    
    def _build_critic_feedback(
        self,
        status: str,
        reason: str,
        next_step: str,
        critic_source: str,
    ) -> str:
        if status == "FINISHED":
            if critic_source == "tools":
                # Language instruction is intentionally absent here — the system prompt
                # (prompt.txt) is the authoritative place to set the reply language.
                return (
                    "Task appears complete based on the tool results. "
                    "Provide a concise final answer to the user without calling more tools "
                    "unless a final verification is still genuinely required."
                )
            return ""

        next_step_line = next_step if next_step and next_step != "NONE" else "perform an explicit verification step"
        return f"Task incomplete. Reason: {reason}. Next step: {next_step_line}."

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
        critic_feedback = self._normalize_critic_feedback(state.get("critic_feedback") or "")
        current_task = state.get("current_task") or self._derive_current_task(messages)
        current_turn_id = self._current_turn_id(state, messages)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
        self._log_run_event(
            state,
            "agent_node_start",
            run_id=state.get("run_id", ""),
            step=state.get("steps", 0),
            current_task=current_task,
        )

        tools_available = bool(self.tools) and self.config.model_supports_tools
        try:
            full_context = self._build_agent_context(
                messages,
                summary,
                current_task,
                critic_feedback,
                tools_available,
                open_tool_issue,
                state=state,
            )
            self._assert_provider_safe_agent_context(full_context, state)
            response = await self._invoke_llm_with_retry(
                self.llm_with_tools,
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
                previous_critic_source=str(state.get("critic_source", "") or ""),
                open_tool_issue=open_tool_issue,
            )
            tool_calls_count = len(getattr(response, "tool_calls", []) or [])
            self._log_run_event(
                state,
                "agent_node_end",
                run_id=state.get("run_id", ""),
                tool_calls=tool_calls_count,
                content_preview=compact_text(stringify_content(response.content), 240),
            )
            self._log_node_end(
                state,
                "agent",
                node_timer,
                tool_calls=tool_calls_count,
                tools_available=tools_available,
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

    # --- NODE: CRITIC ---

    async def critic_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "critic",
            message_count=len(state.get("messages") or []),
            source=(state.get("critic_source") or "agent"),
        )
        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        current_task = (state.get("current_task") or self._derive_current_task(messages)).strip()
        summary = state.get("summary", "")
        critic_source = (state.get("critic_source") or "agent").strip().lower()
        trace = self._format_trace_for_critic(messages)
        open_tool_issue = self._get_active_open_tool_issue(state, messages, current_turn_id)
        unresolved_tool_error = open_tool_issue.get("summary") if open_tool_issue else "None."

        try:
            prompt = constants.CRITIC_PROMPT_TEMPLATE.format(
                current_task=current_task or "No explicit task provided.",
                summary=summary or "No prior summary.",
                unresolved_tool_error=unresolved_tool_error,
                source=critic_source or "agent",
                trace=trace,
            )

            response = await self._invoke_llm_with_retry(
                self.llm,
                [SystemMessage(content=prompt)],
                state=state,
                node_name="critic",
            )
            parsed = self._parse_critic_response(str(response.content))

            if not parsed:
                status, reason, next_step, control = self._infer_critic_fallback(
                    messages,
                    critic_source,
                    open_tool_issue,
                )
                logger.info(
                    f"Critic returned malformed verdict. Falling back to heuristic {status}: {reason}"
                )
            else:
                status, reason, next_step, control = parsed

            last_ai = self._get_last_ai_message(messages)
            if (
                control == "FINISH_TURN"
                and open_tool_issue
                and not self._assistant_handed_off_open_tool_issue(last_ai, open_tool_issue)
            ):
                status = "INCOMPLETE"
                reason = "A tool issue is still open, so the assistant cannot honestly claim success yet."
                next_step = "resolve the blocker or clearly report it to the user"
                control = "RETRY_AGENT"
            elif control == "RETRY_AGENT" and self._assistant_requests_user_input(last_ai):
                status = "INCOMPLETE"
                reason = "The assistant is waiting for a user decision or clarification."
                next_step = "wait for the next user input"
                control = "FINISH_TURN"

            turn_outcome = self._normalize_turn_outcome(control, status)
            retry_instruction = ""
            next_retry_count = 0
            next_retry_fingerprint = ""
            current_retry_count = int(state.get("critic_retry_count", 0) or 0)
            previous_retry_fingerprint = str(state.get("critic_last_retry_fingerprint", "") or "")
            current_fingerprint = self._assistant_response_fingerprint(last_ai)
            if turn_outcome == "retry_agent":
                retry_limit = max(0, int(getattr(self.config, "critic_max_retries_per_turn", 1)))
                repeated_response = bool(
                    current_fingerprint and current_fingerprint == previous_retry_fingerprint
                )
                retry_limit_reached = current_retry_count >= retry_limit

                if repeated_response or retry_limit_reached:
                    status = "INCOMPLETE"
                    reason = (
                        "Assistant response repeated after a tool issue; stopping auto-retry to avoid oscillation."
                        if repeated_response
                        else "Critic retry budget for this turn is exhausted; waiting for user input."
                    )
                    next_step = "wait for the next user input"
                    control = "FINISH_TURN"
                    turn_outcome = "finish_turn"
                    self._log_run_event(
                        state,
                        "critic_retry_suppressed",
                        run_id=state.get("run_id", ""),
                        repeated_response=repeated_response,
                        retry_limit=retry_limit,
                        retry_count=current_retry_count,
                        has_open_tool_issue=bool(open_tool_issue),
                    )
                else:
                    retry_instruction = self._build_retry_instruction(current_task, reason, next_step)
                    next_retry_count = current_retry_count + 1
                    next_retry_fingerprint = current_fingerprint

            self._log_run_event(
                state,
                "critic_verdict",
                run_id=state.get("run_id", ""),
                status=status,
                reason=reason,
                next_step=next_step,
                source=critic_source,
                control=control,
                turn_outcome=turn_outcome,
            )
            self._log_run_event(
                state,
                "turn_outcome",
                run_id=state.get("run_id", ""),
                outcome=turn_outcome,
                source=critic_source,
            )
            if turn_outcome == "finish_turn":
                self._log_run_event(
                    state,
                    "handoff_to_user",
                    run_id=state.get("run_id", ""),
                    source=critic_source,
                    reason=reason,
                )

            self._log_node_end(
                state,
                "critic",
                node_timer,
                status=status,
                control=control,
                turn_outcome=turn_outcome,
                used_fallback=not bool(parsed),
                has_open_tool_issue=bool(open_tool_issue),
            )
            return {
                "turn_id": current_turn_id,
                "critic_status": status,
                "critic_feedback": self._build_critic_feedback(status, reason, next_step, critic_source),
                "current_task": current_task,
                "turn_outcome": turn_outcome,
                "retry_instruction": retry_instruction,
                "open_tool_issue": None if turn_outcome == "finish_turn" else open_tool_issue,
                "has_protocol_error": False,
                "critic_retry_count": next_retry_count,
                "critic_last_retry_fingerprint": next_retry_fingerprint,
            }
        except Exception as e:
            self._log_node_error(
                state,
                "critic",
                node_timer,
                e,
                source=critic_source,
                has_open_tool_issue=bool(open_tool_issue),
            )
            raise

    async def prepare_retry_node(self, state: AgentState):
        node_timer = self._log_node_start(
            state,
            "prepare_retry",
            has_retry_instruction=bool((state.get("retry_instruction") or "").strip()),
        )
        retry_instruction = (state.get("retry_instruction") or "").strip()
        if not retry_instruction:
            self._log_node_end(
                state,
                "prepare_retry",
                node_timer,
                outcome="finish_turn",
                reason="empty_retry_instruction",
            )
            return {"turn_outcome": "finish_turn"}

        messages = state.get("messages", [])
        current_turn_id = self._current_turn_id(state, messages)
        retry_message = self._build_internal_retry_message(retry_instruction, current_turn_id)
        self._log_run_event(
            state,
            "retry_prepared",
            run_id=state.get("run_id", ""),
            turn_id=current_turn_id,
            instruction_preview=compact_text(retry_instruction, 240),
        )
        self._log_node_end(
            state,
            "prepare_retry",
            node_timer,
            outcome="retry_prepared",
            turn_id=current_turn_id,
        )
        return {
            "messages": [retry_message],
            "turn_outcome": "",
            "retry_instruction": "",
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

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            self._log_node_end(
                state,
                "approval",
                node_timer,
                outcome="skipped",
                reason="no_protected_tool_calls",
            )
            return {"pending_approval": None}

        protected_calls = []
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call.get("name") or "unknown_tool"
            if not self._tool_requires_approval(tool_name):
                continue
            metadata = self._metadata_for_tool(tool_name)
            protected_calls.append(
                {
                    "id": tool_call.get("id") or "",
                    "name": tool_name,
                    "args": tool_call.get("args") or {},
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
        last_msg = messages[-1]
        current_turn_id = self._current_turn_id(state, messages)

        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
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
                        self._process_tool_call(tool_call, recent_calls, state, approval_state, current_turn_id)
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
                "critic_source": "tools",
                "critic_feedback": "",
                "critic_status": "",
                "turn_outcome": "run_tools",
                "retry_instruction": "",
                "pending_approval": None,
                "open_tool_issue": merged_issue,
                "has_protocol_error": False,
                "critic_retry_count": 0,
                "critic_last_retry_fingerprint": "",
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

    async def _process_tool_call(
        self,
        tool_call: Dict[str, Any],
        recent_calls: List[Dict[str, Any]],
        state: AgentState,
        approval_state: Dict[str, Any],
        current_turn_id: int,
    ) -> Tuple[ToolMessage, bool, Dict[str, Any] | None]:
        # Безопасное извлечение с фоллбеками
        t_name = tool_call.get("name") or "unknown_tool"
        t_args = tool_call.get("args") or {}
    
        # Генерируем фейковый ID, если LLM забыла его указать, чтобы Pydantic не упал
        t_id = tool_call.get("id")
        if not t_id:
            t_id = f"call_missing_{uuid.uuid4().hex[:8]}"

        had_error = False
        metadata = self._metadata_for_tool(t_name)

        if self._tool_requires_approval(t_name) and not self._tool_call_is_approved(t_id, approval_state):
            content = format_error(
                ErrorType.ACCESS_DENIED,
                f"Execution of '{t_name}' was cancelled by approval policy.",
            )
            self._log_run_event(
                state,
                "tool_call_denied",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                policy=metadata.to_dict(),
            )
            limit = self.config.safety.max_tool_output
            parsed_result = parse_tool_execution_result(content)
            issue = self._build_open_tool_issue(
                current_turn_id=current_turn_id,
                kind="approval_denied",
                summary=parsed_result.message or content,
                tool_names=[t_name],
                source="approval",
                error_type=parsed_result.error_type,
            )
            return (
                self._build_tool_message(
                    content=truncate_output(content, limit, source=t_name),
                    tool_call_id=t_id,
                    tool_name=t_name,
                    tool_args=t_args,
                ),
                True,
                issue,
            )

        missing_required = self._missing_required_tool_fields(t_name, t_args)
        if missing_required:
            content = format_error(
                ErrorType.VALIDATION,
                f"Missing required field(s): {', '.join(missing_required)}.",
            )
            self._log_run_event(
                state,
                "tool_call_preflight_validation_failed",
                run_id=state.get("run_id", ""),
                tool_name=t_name,
                tool_args=t_args,
                missing_required=missing_required,
            )
            limit = self.config.safety.max_tool_output
            parsed_result = parse_tool_execution_result(content)
            issue = self._build_open_tool_issue(
                current_turn_id=current_turn_id,
                kind="tool_error",
                summary=parsed_result.message or content,
                tool_names=[t_name],
                source="tools",
                error_type=parsed_result.error_type,
            )
            return (
                self._build_tool_message(
                    content=truncate_output(content, limit, source=t_name),
                    tool_call_id=t_id,
                    tool_name=t_name,
                    tool_args=t_args,
                ),
                True,
                issue,
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
            content = await self._execute_tool(t_name, t_args, state=state)

        # Post-Tool Validation Layer
        validation_error = validate_tool_result(t_name, t_args, content)
        if validation_error:
            content = f"{content}\n\n{validation_error}"
            had_error = True

        if is_error_text(content):
            had_error = True

        limit = self.config.safety.max_tool_output
        content = truncate_output(content, limit, source=t_name)
        parsed_result = parse_tool_execution_result(content)
        self._log_run_event(
            state,
            "tool_call_end",
            run_id=state.get("run_id", ""),
            tool_name=t_name,
            tool_args=t_args,
            result=parsed_result.to_event_payload(),
        )

        issue = None
        if not parsed_result.ok:
            issue_kind = "approval_denied" if parsed_result.error_type == "ACCESS_DENIED" else "tool_error"
            issue_source = "approval" if issue_kind == "approval_denied" else "tools"
            issue = self._build_open_tool_issue(
                current_turn_id=current_turn_id,
                kind=issue_kind,
                summary=parsed_result.message or content,
                tool_names=[t_name],
                source=issue_source,
                error_type=parsed_result.error_type,
            )

        return (
            self._build_tool_message(
                content=content,
                tool_call_id=t_id,
                tool_name=t_name,
                tool_args=t_args,
            ),
            had_error,
            issue,
        )

    def _build_tool_message(
        self,
        *,
        content: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> ToolMessage:
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            additional_kwargs={"tool_args": deepcopy(tool_args) if isinstance(tool_args, dict) else {}},
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
        for message in reversed(context):
            if isinstance(message, SystemMessage):
                continue
            return message
        return None

    def _assert_provider_safe_agent_context(
        self,
        context: List[BaseMessage],
        state: AgentState | None = None,
    ) -> None:
        last_visible = self._get_last_model_visible_message(context)
        valid = isinstance(last_visible, (HumanMessage, ToolMessage))
        self._log_run_event(
            state,
            "provider_context_valid",
            run_id=None if state is None else state.get("run_id", ""),
            valid=valid,
            last_visible_type=type(last_visible).__name__ if last_visible else "",
        )
        if valid:
            return

        raise ProviderContextError(
            "Provider-unsafe agent context: the last model-visible message must be HumanMessage or ToolMessage."
        )

    async def _execute_tool(self, name: str, args: dict, state: AgentState | None = None) -> str:
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

    def _format_trace_for_critic(self, messages: List[BaseMessage], max_messages: int = 12) -> str:
        if not messages:
            return "No recent messages."

        trace_messages = messages[-max_messages:]
        formatted = [self._format_message_for_critic(message) for message in trace_messages]
        trace = "\n".join(part for part in formatted if part).strip()
        if not trace:
            return "No recent messages."
        return trace[:12000]

    def _format_message_for_critic(self, message: BaseMessage) -> str:
        if self._is_internal_retry_message(message):
            return ""

        content = stringify_content(message.content)
        content = compact_text(content, 1200)

        if isinstance(message, ToolMessage):
            label = f"tool[{message.name or 'unknown'}]"
            return f"{label}: {content or '[empty tool output]'}"

        if isinstance(message, AIMessage):
            parts: List[str] = []
            if content:
                parts.append(f"assistant: {content}")
            for tool_call in getattr(message, "tool_calls", []) or []:
                name = tool_call.get("name", "unknown")
                args = compact_text(str(tool_call.get("args", {})), 300)
                parts.append(f"assistant_tool_call[{name}]: {args}")
            return "\n".join(parts) if parts else "assistant: [empty response]"

        if isinstance(message, HumanMessage):
            role = "system_hint" if content == constants.REFLECTION_PROMPT else "user"
            return f"{role}: {content or '[empty message]'}"

        return f"{message.type}: {content or '[empty message]'}"


    def _parse_critic_response(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        parsed: Dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            if key in {"STATUS", "REASON", "NEXT_STEP", "CONTROL"} and key not in parsed:
                parsed[key] = value.strip()

        status = parsed.get("STATUS", "").upper().rstrip(".")
        upper_text = text.upper()
        if status not in {"FINISHED", "INCOMPLETE"}:
            if "INCOMPLETE" in upper_text:
                status = "INCOMPLETE"
            elif "FINISHED" in upper_text or "COMPLETE" in upper_text or "DONE" in upper_text:
                status = "FINISHED"
            else:
                return None

        reason = parsed.get("REASON", "").strip()
        if not reason:
            reason = self._extract_reason_from_freeform_text(text, status)

        next_step = parsed.get("NEXT_STEP", "").strip()
        if not next_step:
            next_step = "NONE" if status == "FINISHED" else "continue with the next obvious missing step"

        control = parsed.get("CONTROL", "").strip().upper().rstrip(".")
        if control not in {"FINISH_TURN", "RETRY_AGENT"}:
            control = "FINISH_TURN" if status == "FINISHED" else "RETRY_AGENT"

        normalized_next_step = next_step.upper() if next_step.upper() == "NONE" else next_step
        return status, reason, normalized_next_step, control

    def _open_tool_issue_requires_user_input(self, open_tool_issue: Dict[str, Any] | None) -> bool:
        if not isinstance(open_tool_issue, dict):
            return False
        if str(open_tool_issue.get("kind") or "").strip().lower() != "tool_error":
            return False

        error_type = str(open_tool_issue.get("error_type") or "").strip().upper()
        tool_names = [str(name).strip() for name in (open_tool_issue.get("tool_names") or []) if str(name).strip()]
        only_planning_tools = bool(tool_names) and all(self._is_planning_tool(name) for name in tool_names)
        summary = str(open_tool_issue.get("summary") or "").strip().lower()
        if error_type == "VALIDATION":
            if only_planning_tools:
                return False
            return True

        user_input_markers = (
            "missing required field",
            "required field",
            "accepted aliases",
            "укажите путь",
            "укажите правильный путь",
            "предоставьте путь",
            "provide path",
            "specify the path",
            "provide file",
            "предоставьте файл",
        )
        return any(marker in summary for marker in user_input_markers)

    def _build_tool_issue_handoff_text(self, open_tool_issue: Dict[str, Any] | None) -> str:
        summary = ""
        tool_names: List[str] = []
        if isinstance(open_tool_issue, dict):
            summary = compact_text(str(open_tool_issue.get("summary", "")).strip(), 220)
            tool_names = [str(name).strip() for name in (open_tool_issue.get("tool_names") or []) if str(name).strip()]

        tool_hint = f" (`{tool_names[0]}`)" if tool_names else ""
        summary_line = f"\nДетали ошибки: {summary}" if summary else ""
        return (
            f"Не могу продолжить автоматический вызов инструмента{tool_hint}: не хватает корректных входных данных."
            f"{summary_line}\n"
            "Пожалуйста, укажите путь к файлу и недостающие параметры (что искать и на что заменить), "
            "и я продолжу без повторного цикла."
        )

    def _extract_reason_from_freeform_text(self, text: str, status: str) -> str:
        cleaned = " ".join(text.split()).strip(" -:")
        if cleaned:
            return compact_text(cleaned, 180)
        if status == "FINISHED":
            return "Task appears completed."
        return "Task appears incomplete."

    def _assistant_response_fingerprint(self, last_ai: AIMessage | None) -> str:
        if not last_ai or getattr(last_ai, "tool_calls", None):
            return ""
        content = " ".join(stringify_content(last_ai.content).strip().lower().split())
        if not content:
            return ""
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    def _assistant_handed_off_open_tool_issue(
        self,
        last_ai: AIMessage | None,
        open_tool_issue: Dict[str, Any] | None,
    ) -> bool:
        if not open_tool_issue or not last_ai or getattr(last_ai, "tool_calls", None):
            return False

        content = stringify_content(last_ai.content).strip().lower()
        if not content:
            return False

        if open_tool_issue.get("kind") == "approval_denied":
            return True

        blocker_terms = (
            "не удалось",
            "не найден",
            "не найдено",
            "ошиб",
            "не смог",
            "не получилось",
            "не могу",
            "укажите путь",
            "укажи путь",
            "предоставьте файл",
            "предоставь файл",
            "предоставьте код",
            "предоставь код",
            "cannot",
            "unable",
            "not found",
            "failed",
            "failure",
            "error",
            "blocked",
            "denied",
            "cancelled",
            "provide path",
            "provide the path",
            "specify the path",
            "provide file",
            "provide the file",
            "provide code",
        )
        success_terms = (
            "успеш",
            "готово",
            "сделал",
            "сделано",
            "создан",
            "заверш",
            "completed",
            "success",
            "done",
            "finished",
        )

        if any(term in content for term in blocker_terms):
            return True
        if any(term in content for term in success_terms):
            return False
        return False

    def _assistant_requests_user_input(self, last_ai: AIMessage | None) -> bool:
        if not last_ai or getattr(last_ai, "tool_calls", None):
            return False

        content = stringify_content(last_ai.content).strip().lower()
        if not content:
            return False

        decision_terms = (
            "какой вариант",
            "какой способ",
            "какой из вариантов",
            "выберите",
            "укажите номер",
            "укажите путь",
            "укажите правильный путь",
            "предоставьте путь",
            "предоставьте файл",
            "предоставьте код",
            "подтвердите",
            "как поступить",
            "нужно ваше подтверждение",
            "нужно ваше решение",
            "which option",
            "which approach",
            "choose",
            "select",
            "confirm",
            "please confirm",
            "provide path",
            "provide the path",
            "specify the path",
            "provide file",
            "provide code",
            "let me know your choice",
        )
        waiting_terms = (
            "жду вашего ответа",
            "жду вашего решения",
            "ожидаю вашего ответа",
            "wait for your input",
            "waiting for your input",
            "waiting for your confirmation",
        )

        if any(term in content for term in decision_terms):
            return True
        if any(term in content for term in waiting_terms):
            return True
        return content.endswith("?")

    def _infer_critic_fallback(
        self,
        messages: List[BaseMessage],
        critic_source: str,
        open_tool_issue: Dict[str, Any] | None,
    ) -> Tuple[str, str, str, str]:
        last_ai = self._get_last_ai_message(messages)
        recent_tools = self._get_recent_tool_messages(messages)

        if open_tool_issue:
            if self._assistant_handed_off_open_tool_issue(last_ai, open_tool_issue):
                reason = (
                    "The user denied a protected tool and the assistant already handed control back to the user."
                    if open_tool_issue.get("kind") == "approval_denied"
                    else "A tool failed and the assistant already reported the blocker honestly to the user."
                )
                return (
                    "INCOMPLETE",
                    reason,
                    "wait for the next user instruction",
                    "FINISH_TURN",
                )
            return (
                "INCOMPLETE",
                "An unresolved tool issue remains and critic control was unavailable.",
                "resolve the blocker or clearly report it to the user",
                "RETRY_AGENT",
            )

        if any(is_tool_message_error(msg) for msg in recent_tools):
            return (
                "INCOMPLETE",
                "Recent tool execution reported an explicit error.",
                "fix the failed step",
                "RETRY_AGENT",
            )

        if critic_source == "tools" and recent_tools:
            return (
                "FINISHED",
                "Recent tool results indicate the requested action completed successfully.",
                "NONE",
                "FINISH_TURN",
            )

        if last_ai and not getattr(last_ai, "tool_calls", None):
            content = stringify_content(last_ai.content)
            if content.strip():
                return (
                    "FINISHED",
                    "The latest assistant response appears to deliver the requested result.",
                    "NONE",
                    "FINISH_TURN",
                )

        return (
            "INCOMPLETE",
            "Completion is still unclear from the latest trace.",
            "continue only if something obvious is still missing",
            "RETRY_AGENT",
        )

    def _get_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def _get_recent_tool_messages(self, messages: List[BaseMessage], limit: int = 6) -> List[ToolMessage]:
        recent: List[ToolMessage] = []
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                recent.append(message)
                if len(recent) >= limit:
                    break
            elif recent:
                break
        recent.reverse()
        return recent

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

    def _build_system_message(self, summary: str, tools_available: bool = True) -> SystemMessage:
        raw_prompt = self._get_base_prompt()

        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))

        if self.config.strict_mode:
            prompt += "\nNOTE: STRICT MODE ENABLED. Be precise. No guessing."

        if not tools_available:
            prompt += "\nNOTE: You are in CHAT-ONLY mode. Tools are disabled."

        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"

        return SystemMessage(content=prompt)

    async def _invoke_llm_with_retry(
        self,
        llm,
        context: List[BaseMessage],
        state: AgentState | None = None,
        node_name: str = "",
    ) -> AIMessage:
        current_llm = llm
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
                response = await current_llm.ainvoke(context)
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
