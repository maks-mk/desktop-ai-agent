from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, RemoveMessage, SystemMessage, ToolMessage

from core.errors import ErrorType, format_error
from core.message_utils import compact_text
from core.node_errors import EmptyLLMResponseError
from core.self_correction_engine import normalize_tool_args
from core.tool_args import canonicalize_tool_args, inspect_tool_args_payload
from core.tool_results import parse_tool_execution_result


class AgentTurnOrchestrator:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def run(self, state):
        owner = self.owner
        node_timer = owner._log_node_start(
            state,
            "agent",
            message_count=len(state.get("messages") or []),
            has_summary=bool(state.get("summary")),
        )
        messages = state["messages"]
        summary = state.get("summary", "")
        current_task = owner._resolve_current_task(state, messages)
        current_turn_id = owner._current_turn_id(state, messages)
        open_tool_issue = owner._get_active_open_tool_issue(state, messages, current_turn_id)
        recovery_state = owner._get_recovery_state(state, current_turn_id=current_turn_id)

        active_tools, active_tool_names = owner._active_tools_for_turn(
            state,
            messages,
        )
        llm_for_turn = owner._select_llm_for_active_tools(active_tools, active_tool_names)
        tools_available = bool(active_tool_names)
        user_choice_locked = owner._current_turn_has_completed_user_choice(messages)
        owner._log_run_event(
            state,
            "tool_context_selected",
            run_id=state.get("run_id", ""),
            active_tool_count=len(active_tool_names),
            user_choice_locked=user_choice_locked,
        )
        try:
            validation_handoff_reason = ""
            preflight_loop_issue = owner._preflight_recovery_loop_issue(
                messages,
                current_turn_id=current_turn_id,
                open_tool_issue=open_tool_issue,
                recovery_state=recovery_state,
            )
            if preflight_loop_issue:
                validation_handoff_reason = "recovery_strategy_loop_blocked"
                owner._log_run_event(
                    state,
                    "recovery_strategy_loop_blocked",
                    run_id=state.get("run_id", ""),
                    issue=preflight_loop_issue,
                )
                owner._log_node_end(
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
                    "turn_outcome": "recover_agent",
                    "recovery_state": recovery_state,
                    "pending_approval": None,
                    "open_tool_issue": preflight_loop_issue,
                    "has_protocol_error": False,
                    "last_tool_error": str(preflight_loop_issue.get("summary") or ""),
                    "last_tool_result": "",
                }
            history_issue = owner.context_builder.detect_tool_history_mismatch(messages)
            if history_issue:
                validation_handoff_reason = "history_tool_mismatch"
                protocol_issue = owner._build_protocol_open_tool_issue(
                    current_turn_id=current_turn_id,
                    summary=owner._summarize_history_tool_mismatch(history_issue),
                    reason="history_tool_mismatch",
                    source="history",
                    tool_names=[
                        str(item.get("name") or "").strip()
                        for item in (history_issue.get("pending_tool_calls") or [])
                        if str(item.get("name") or "").strip()
                    ],
                    tool_args=(
                        canonicalize_tool_args((history_issue.get("pending_tool_calls") or [{}])[0].get("args"))
                        if history_issue.get("pending_tool_calls")
                        else {}
                    ),
                    details=history_issue,
                )
                owner._log_run_event(
                    state,
                    "history_tool_mismatch_detected",
                    run_id=state.get("run_id", ""),
                    issue=protocol_issue,
                )
                owner._log_node_end(
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

            full_context = owner._build_agent_context(
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
            owner._assert_provider_safe_agent_context(full_context, state)
            response = await owner._invoke_llm_with_retry(
                llm_for_turn,
                full_context,
                state=state,
                node_name="agent",
            )
            result = owner._build_agent_result(
                response,
                current_task,
                tools_available,
                current_turn_id,
                messages,
                open_tool_issue=open_tool_issue,
                recovery_state=recovery_state,
                allowed_tool_names=active_tool_names,
            )
            if result.pop("_retry_user_input_turn", False):
                owner._log_run_event(
                    state,
                    "user_input_reask_suppressed",
                    run_id=state.get("run_id", ""),
                    step=state.get("steps", 0),
                    current_task=current_task,
                )
                retry_context = owner._normalize_system_prefix_for_provider(
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
                owner._assert_provider_safe_agent_context(retry_context, state)
                response = await owner._invoke_llm_with_retry(
                    llm_for_turn,
                    retry_context,
                    state=state,
                    node_name="agent_retry_after_user_choice",
                )
                result = owner._build_agent_result(
                    response,
                    current_task,
                    tools_available,
                    current_turn_id,
                    messages,
                    open_tool_issue=open_tool_issue,
                    recovery_state=recovery_state,
                    allowed_tool_names=active_tool_names,
                )
                result.pop("_retry_user_input_turn", None)
            if result.get("open_tool_issue") and result.get("has_protocol_error"):
                validation_handoff_reason = str(
                    ((result.get("open_tool_issue") or {}).get("details") or {}).get("protocol_reason")
                    or "tool_protocol_error"
                )
                owner._log_run_event(
                    state,
                    "protocol_recovery_requested",
                    run_id=state.get("run_id", ""),
                    issue=result.get("open_tool_issue"),
                )
            tool_calls_count = len(getattr(response, "tool_calls", []) or [])
            owner._log_node_end(
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
        except EmptyLLMResponseError as exc:
            owner._log_node_error(
                state,
                "agent",
                node_timer,
                exc,
                tools_available=tools_available,
                has_open_tool_issue=bool(open_tool_issue),
                handled=True,
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Модель вернула пустой ответ после повторных попыток. "
                            "Я не выполнял дополнительных действий; повторите запрос или уточните формулировку."
                        )
                    )
                ],
                "current_task": current_task,
                "turn_id": current_turn_id,
                "turn_outcome": "finish_turn",
                "pending_approval": None,
                "open_tool_issue": None,
                "has_protocol_error": False,
                "last_tool_error": str(exc),
                "last_tool_result": "",
            }
        except Exception as exc:
            owner._log_node_error(
                state,
                "agent",
                node_timer,
                exc,
                tools_available=tools_available,
                has_open_tool_issue=bool(open_tool_issue),
            )
            raise


class RecoveryTurnOrchestrator:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def run(self, state):
        owner = self.owner
        node_timer = owner._log_node_start(
            state,
            "recovery",
            has_recovery_state=bool(state.get("recovery_state")),
            has_open_tool_issue=bool(state.get("open_tool_issue")),
        )
        messages = state.get("messages", [])
        current_turn_id = owner._current_turn_id(state, messages)
        current_task = owner._resolve_current_task(state, messages).strip()
        open_tool_issue = owner._get_active_open_tool_issue(state, messages, current_turn_id)
        last_ai = owner._get_last_ai_message(messages)
        last_message = messages[-1] if messages else None
        step_count = int(state.get("steps", 0) or 0)
        recovery_state = owner._get_recovery_state(state, current_turn_id=current_turn_id)
        self_correction_limit = owner._hard_loop_ceiling()
        hard_loop_ceiling = min(2, self_correction_limit) if self_correction_limit > 0 else 0
        max_auto_repairs = min(1, self_correction_limit) if self_correction_limit > 0 else 0

        result = owner.recovery_manager.plan_recovery(
            state=state,
            messages=messages,
            current_task=current_task,
            current_turn_id=current_turn_id,
            open_tool_issue=open_tool_issue,
            recovery_state=recovery_state,
            last_ai=last_ai,
            last_message=last_message,
            step_count=step_count,
            max_loops=int(owner.config.max_loops or 0),
            hard_loop_ceiling=hard_loop_ceiling,
            max_auto_repairs=max_auto_repairs,
            successful_tool_stagnation_limit=owner._successful_tool_stagnation_limit(
                str(getattr(last_message, "name", "") or "")
            ),
        )

        outbound_messages: list[BaseMessage] = []
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
            outbound_messages.append(
                AIMessage(
                    content=result["handoff_message"],
                    additional_kwargs={
                        "agent_internal": {
                            "kind": handoff_kind,
                            "turn_id": current_turn_id,
                            "visible_in_ui": False,
                            "ui_notice": owner.recovery_manager.build_internal_ui_notice(
                                str(result["completion_reason"])
                            ),
                        }
                    },
                )
            )

        turn_outcome = "finish_turn"
        next_recovery_state = result["recovery_state"]
        if result["turn_outcome"] == "recover_agent":
            turn_outcome = "recover_agent"
            active_strategy = next_recovery_state.get("active_strategy") if isinstance(next_recovery_state, dict) else {}
            owner._log_run_event(
                state,
                "recovery_prepared",
                run_id=state.get("run_id", ""),
                turn_id=current_turn_id,
                strategy_id=str((active_strategy or {}).get("id") or ""),
                strategy=str((active_strategy or {}).get("strategy") or ""),
                suggested_tool=str((active_strategy or {}).get("suggested_tool_name") or ""),
            )

        owner._log_run_event(
            state,
            "recovery_verdict",
            run_id=state.get("run_id", ""),
            outcome=turn_outcome,
            reason=result["completion_reason"],
            retry_count=result["self_correction_retry_count"],
            has_open_tool_issue=bool(open_tool_issue),
            loop_budget_reached=result["loop_budget_reached"],
            had_pending_tool_calls=result["had_pending_tool_calls"],
        )
        owner._log_node_end(
            state,
            "recovery",
            node_timer,
            outcome=turn_outcome,
            reason=result["completion_reason"],
            turn_id=current_turn_id,
        )
        payload = {
            "turn_outcome": turn_outcome,
            "recovery_state": next_recovery_state,
            "open_tool_issue": result["open_tool_issue"],
            "has_protocol_error": result["has_protocol_error"],
            "self_correction_retry_count": result["self_correction_retry_count"],
            "self_correction_retry_turn_id": result["self_correction_retry_turn_id"],
            "self_correction_fingerprint_history": result["self_correction_fingerprint_history"],
            "self_correction_last_reason": result["self_correction_last_reason"],
            "last_tool_error": result.get("last_tool_error", state.get("last_tool_error", "")),
            "last_tool_result": result.get("last_tool_result", state.get("last_tool_result", "")),
        }
        if outbound_messages:
            payload["messages"] = outbound_messages
        return payload


class ToolBatchCoordinator:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    async def run(self, state):
        owner = self.owner
        node_timer = owner._log_node_start(
            state,
            "tools",
            message_count=len(state.get("messages") or []),
            has_pending_approval=bool(state.get("pending_approval")),
        )
        owner._check_invariants(state)

        messages = state["messages"]
        last_msg = owner._get_last_pending_ai_with_tool_calls(messages)
        current_turn_id = owner._current_turn_id(state, messages)

        if not last_msg:
            owner._log_node_end(
                state,
                "tools",
                node_timer,
                outcome="skipped",
                reason="no_tool_calls",
            )
            return {}

        final_messages: list[ToolMessage] = []
        has_error = False
        last_error = ""
        last_result = ""
        tool_issues: list[dict[str, Any]] = []
        approval_state = state.get("pending_approval") or {}
        _, active_tool_names = owner._active_tools_for_turn(
            state,
            messages,
        )

        recent_calls = []
        history_window = owner.config.effective_tool_loop_window
        history_slice = messages[-(history_window + 1):-1] if history_window > 0 else messages[:-1]
        for message in reversed(history_slice):
            if isinstance(message, AIMessage) and message.tool_calls:
                recent_calls.extend(message.tool_calls)

        tool_calls = list(last_msg.tool_calls)

        parallel_mode = owner._can_parallelize_tool_calls(tool_calls)
        owner._log_run_event(
            state,
            "tools_node_start",
            run_id=state.get("run_id", ""),
            tool_call_count=len(tool_calls),
            tool_names=[(tool_call.get("name") or "unknown_tool") for tool_call in tool_calls],
            parallel_mode=parallel_mode,
        )
        try:
            if parallel_mode:
                processed = await asyncio.gather(
                    *(
                        owner._process_tool_call(
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
                    tool_msg, had_error, issue = await owner._process_tool_call(
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

            merged_issue = owner._merge_open_tool_issues(tool_issues, current_turn_id)
            owner._log_run_event(
                state,
                "tools_node_end",
                run_id=state.get("run_id", ""),
                tool_result_count=len(final_messages),
                has_error=has_error,
                issue_kind="" if not merged_issue else merged_issue.get("kind", ""),
                issue_source="" if not merged_issue else merged_issue.get("source", ""),
            )
            owner._log_node_end(
                state,
                "tools",
                node_timer,
                tool_call_count=len(tool_calls),
                tool_result_count=len(final_messages),
                parallel_mode=parallel_mode,
                has_error=has_error,
                has_open_tool_issue=bool(merged_issue),
            )
            payload = {
                "messages": final_messages,
                "turn_id": current_turn_id,
                "turn_outcome": "run_tools",
                "pending_approval": None,
                "open_tool_issue": merged_issue,
                "has_protocol_error": False,
                "last_tool_error": last_error,
                "last_tool_result": last_result,
            }
            if not merged_issue:
                payload.update(
                    {
                        "self_correction_retry_count": 0,
                        "self_correction_retry_turn_id": current_turn_id,
                        "self_correction_fingerprint_history": [],
                        "recovery_state": owner.recovery_manager.reset_after_success(
                            state.get("recovery_state"),
                            current_turn_id=current_turn_id,
                            successful_evidence=last_result,
                        ),
                    }
                )
            return payload
        except Exception as exc:
            owner._log_node_error(
                state,
                "tools",
                node_timer,
                exc,
                tool_call_count=len(tool_calls),
                parallel_mode=parallel_mode,
            )
            raise

    async def process_tool_call(
        self,
        tool_call: dict[str, Any],
        recent_calls: list[dict[str, Any]],
        state,
        approval_state: dict[str, Any],
        current_turn_id: int,
        allowed_tool_names: list[str] | None = None,
    ):
        owner = self.owner
        tool_name = tool_call.get("name") or "unknown_tool"
        raw_tool_args = tool_call.get("args")
        tool_args, args_payload_kind = inspect_tool_args_payload(raw_tool_args)
        if args_payload_kind == "json_string":
            owner._log_run_event(
                state,
                "tool_call_args_canonicalized",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_call_id=str(tool_call.get("id") or ""),
                source_kind=args_payload_kind,
                arg_keys=sorted(tool_args.keys()),
                raw_preview=compact_text(str(raw_tool_args), 220),
            )
        elif args_payload_kind not in {"mapping", "missing", "empty_string"}:
            owner._log_run_event(
                state,
                "tool_call_args_unparsed",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_call_id=str(tool_call.get("id") or ""),
                source_kind=args_payload_kind,
                raw_preview=compact_text(str(raw_tool_args), 220),
            )
        normalized_args, normalized_changes = normalize_tool_args(
            tool_name,
            tool_args,
            current_task=str(state.get("current_task") or ""),
        )
        if normalized_changes:
            owner._log_run_event(
                state,
                "tool_call_args_repaired",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                original_args=tool_args,
                patched_args=normalized_args,
                changes=normalized_changes,
            )
            tool_args = normalized_args

        tool_call_id = tool_call.get("id")
        if not tool_call_id:
            tool_call_id = f"call_missing_{uuid.uuid4().hex[:8]}"

        had_error = False
        tool_duration_seconds: float | None = None
        metadata = owner._effective_tool_metadata(tool_name, tool_args)
        active_tool_names = (
            list(allowed_tool_names)
            if allowed_tool_names is not None
            else (list(owner._all_tool_names) if owner.config.model_supports_tools else [])
        )

        if not owner._tool_is_allowed_for_turn(tool_name, allowed_tool_names):
            outcome = owner.tool_executor.build_not_allowed_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                allowed_tool_names=active_tool_names,
            )
            return outcome.tool_message, outcome.had_error, outcome.issue

        if owner._tool_requires_approval(tool_name, tool_args) and not owner._tool_call_is_approved(
            tool_call_id, approval_state
        ):
            outcome = owner.tool_executor.build_denied_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                policy=metadata.to_dict(),
            )
            return outcome.tool_message, outcome.had_error, outcome.issue

        missing_required = owner._missing_required_tool_fields(tool_name, tool_args)
        if missing_required:
            outcome = owner.tool_executor.build_missing_required_result(
                state=state,
                current_turn_id=current_turn_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                missing_required=missing_required,
            )
            return outcome.tool_message, outcome.had_error, outcome.issue

        loop_count = sum(
            1
            for recent_call in recent_calls
            if recent_call.get("name") == tool_name and canonicalize_tool_args(recent_call.get("args")) == tool_args
        )
        loop_limit = (
            owner.config.effective_tool_loop_limit_readonly
            if tool_name in owner.READ_ONLY_LOOP_TOLERANT_TOOL_NAMES
            else owner.config.effective_tool_loop_limit_mutating
        )
        if loop_count >= loop_limit:
            content = format_error(
                ErrorType.LOOP_DETECTED,
                f"Loop detected. You have called '{tool_name}' with these exact arguments {loop_limit} times in the recent history. Please try a different approach.",
            )
            had_error = True
            owner._log_run_event(
                state,
                "tool_call_loop_blocked",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                loop_count=loop_count,
                loop_limit=loop_limit,
            )
        else:
            owner._log_run_event(
                state,
                "tool_call_start",
                run_id=state.get("run_id", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                policy=metadata.to_dict(),
            )
            started_at = time.perf_counter()
            content = await owner._execute_tool(tool_name, tool_args, state=state, tool_call_id=tool_call_id)
            tool_duration_seconds = max(0.0, time.perf_counter() - started_at)

        outcome = owner.tool_executor.handle_result(
            state=state,
            current_turn_id=current_turn_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            content=content,
            tool_duration_seconds=tool_duration_seconds,
            had_error=had_error,
            issue_details=(
                {
                    "loop_detected": True,
                    "loop_count": loop_count,
                    "loop_limit": loop_limit,
                }
                if loop_count >= loop_limit
                else None
            ),
        )
        return outcome.tool_message, outcome.had_error, outcome.issue
