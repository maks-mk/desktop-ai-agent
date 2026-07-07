from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from core.state import AgentState
from core.tool_args import canonicalize_tool_args
from core.constants import TOOL_ISSUE_UI_NOTICE
from core.message_utils import stringify_content
from core.text_tool_calls import extract_text_tool_calls
from core.turn_outcomes import (
    TURN_OUTCOME_FINISH_TURN,
    TURN_OUTCOME_RECOVER_AGENT,
    TURN_OUTCOME_RUN_TOOLS,
)

_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"

_PLAN_APPROVAL_KEYWORDS = frozenset({
    "реализовать", "отказаться", "правки", "дополнения",
    "implement", "reject", "revise", "approve", "execute", "decline",
})
_PLAN_MIN_CONTENT_LENGTH = 50


class AgentMixin:
    """Agent node facade and result-building helpers."""

    async def agent_node(self, state: AgentState):
        return await self.agent_turn.run(state)

    def _normalized_allowed_tool_name_set(self, allowed_tool_names: List[str] | None) -> set[str]:
        return {self._normalize_tool_name(name) for name in (allowed_tool_names or [])}

    def _build_protocol_issue(
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
        return self._build_protocol_open_tool_issue(
            current_turn_id=current_turn_id,
            summary=summary,
            reason=reason,
            source=source,
            tool_names=tool_names,
            tool_args=tool_args,
            details=details,
            response_preview=response_preview,
        )

    def _new_ai_message_with_tool_calls(self, response: AIMessage, tool_calls: List[Dict[str, Any]]) -> AIMessage:
        return response.model_copy(update={"tool_calls": tool_calls})

    def _new_ai_message_without_tool_calls(self, response: AIMessage, content: Any) -> AIMessage:
        return AIMessage(
            content=content,
            additional_kwargs=response.additional_kwargs,
            response_metadata=response.response_metadata,
            usage_metadata=response.usage_metadata,
            id=response.id,
        )

    def _new_ai_message_with_content_and_tool_calls(
        self, response: AIMessage, content: Any, tool_calls: List[Dict[str, Any]]
    ) -> AIMessage:
        return response.model_copy(update={"content": content, "tool_calls": tool_calls})

    def _plan_approval_tool_call(self, turn_id: int) -> Dict[str, Any]:
        return {
            "name": "request_user_input",
            "args": {
                "question": "Что сделать с этим планом?",
                "options": [
                    "Да, реализовать",
                    "Нет, отказаться от реализации",
                    "Внести правки/дополнения в план",
                ],
                "recommended": "Да, реализовать",
                "choice_type": "plan_approval",
            },
            "id": f"planapproval-{max(0, int(turn_id or 0))}",
        }

    def _plan_approval_allowed(self, allowed_tool_names: List[str] | None) -> bool:
        if allowed_tool_names is None:
            return True
        allowed = self._normalized_allowed_tool_name_set(allowed_tool_names)
        return "request_user_input" in allowed

    def _is_plan_approval_request(self, tool_call: Dict[str, Any]) -> bool:
        """Check whether request_user_input is asking to approve the implementation plan."""
        if self._normalize_tool_name(tool_call.get("name") or "") != "request_user_input":
            return False
        args = canonicalize_tool_args(tool_call.get("args"))
        if not isinstance(args, dict):
            return False
        choice_type = str(args.get("choice_type") or "").strip().lower()
        if choice_type:
            return choice_type == "plan_approval"

        # Backward-compatible fallback for older persisted tool calls/prompts.
        options = args.get("options") or []
        if not isinstance(options, list) or not options:
            return False
        options_text = " ".join(str(o).lower() for o in options)
        return any(kw in options_text for kw in _PLAN_APPROVAL_KEYWORDS)

    def _build_agent_protocol_issue(
        self,
        *,
        current_turn_id: int,
        summary: str,
        reason: str,
        tool_names: List[str] | None = None,
        tool_args: Dict[str, Any] | None = None,
        details: Dict[str, Any] | None = None,
        response_preview: str = "",
    ) -> Dict[str, Any]:
        return self._build_protocol_issue(
            current_turn_id=current_turn_id,
            summary=summary,
            reason=reason,
            source="agent",
            tool_names=tool_names,
            tool_args=tool_args,
            details=details,
            response_preview=response_preview,
        )

    def _ensure_gemini_tool_call_signatures(
        self,
        response: AIMessage,
        tool_calls: List[Dict[str, Any]],
    ) -> AIMessage:
        if self.config.provider != "gemini" or not tool_calls:
            return response

        metadata = dict(getattr(response, "additional_kwargs", {}) or {})
        raw_signature_map = metadata.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
        signature_map = dict(raw_signature_map) if isinstance(raw_signature_map, dict) else {}
        normalized_signature_map: Dict[str, str] = {}

        for tool_call in tool_calls:
            tool_call_id = str(tool_call.get("id") or "").strip()
            signature = signature_map.get(tool_call_id)
            if tool_call_id and isinstance(signature, str) and signature:
                normalized_signature_map[tool_call_id] = signature

        if raw_signature_map == normalized_signature_map:
            return response

        if normalized_signature_map:
            metadata[_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = normalized_signature_map
        else:
            metadata.pop(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY, None)
        return response.model_copy(update={"additional_kwargs": metadata})

    def _build_agent_result(
        self,
        response: AIMessage,
        current_task: str,
        tools_available: bool,
        turn_id: int,
        messages: List[Any],
        open_tool_issue: Dict[str, Any] | None = None,
        recovery_state: Dict[str, Any] | None = None,
        allowed_tool_names: List[str] | None = None,
        plan_mode: bool = False,
        legacy_plan_approval: bool = True,
    ) -> Dict[str, Any]:
        token_usage_update = {}
        if getattr(response, "usage_metadata", None):
            token_usage_update = {"token_usage": response.usage_metadata}

        has_tool_calls = False
        protocol_error = ""
        protocol_issue: Dict[str, Any] | None = None
        outbound_messages = self._collect_internal_retry_removals(messages)

        if isinstance(response, AIMessage):
            t_calls = list(getattr(response, "tool_calls", []))
            invalid_calls = list(getattr(response, "invalid_tool_calls", []))
            retry_user_input_turn = False
            if not tools_available and t_calls and not any(
                self._normalize_tool_name(tool_call.get("name") or "") == "request_user_input"
                for tool_call in t_calls
            ):
                t_calls = []
                response = self._new_ai_message_without_tool_calls(response, response.content)


            if tools_available and not t_calls and not invalid_calls:
                recovered_tool_calls = extract_text_tool_calls(
                    stringify_content(response.content),

                    allowed_tool_names=allowed_tool_names or self._all_tool_names,
                    id_prefix=f"txttc{max(0, int(turn_id or 0)) % 100:02d}",
                )
                if recovered_tool_calls.tool_calls:
                    if bool(getattr(self.config, "enable_text_tool_call_recovery", False)):
                        response = self._new_ai_message_with_content_and_tool_calls(
                            response,
                            recovered_tool_calls.cleaned_text or "I will use the requested tool.",
                            recovered_tool_calls.tool_calls,
                        )
                        t_calls = list(recovered_tool_calls.tool_calls)
                    else:
                        tool_names = [
                            str(tool_call.get("name") or "").strip()
                            for tool_call in recovered_tool_calls.tool_calls
                            if str(tool_call.get("name") or "").strip()
                        ]
                        first_args = canonicalize_tool_args(recovered_tool_calls.tool_calls[0].get("args"))
                        protocol_error = (
                            "INTERNAL TOOL PROTOCOL ERROR: the model emitted a textual tool call marker instead of "
                            "a structured tool_calls payload. Textual tool-call recovery is disabled. If a tool is "
                            "still needed, retry with a fresh structured tool call only."
                        )
                        protocol_issue = self._build_agent_protocol_issue(
                            current_turn_id=turn_id,
                            summary=protocol_error,
                            reason="textual_tool_call_disabled",
                            tool_names=tool_names,
                            tool_args=first_args,
                            details={"recovered_textual_tool_call_count": len(recovered_tool_calls.tool_calls)},
                            response_preview=stringify_content(response.content),
                        )

                        response = self._new_ai_message_without_tool_calls(response, protocol_error)

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
                first_args: Dict[str, Any] = {}
                for tc in t_calls:
                    parsed_args = canonicalize_tool_args(tc.get("args"))
                    if parsed_args:
                        first_args = parsed_args
                        break
                protocol_issue = self._build_agent_protocol_issue(
                    current_turn_id=turn_id,
                    summary=protocol_error,
                    reason="tool_protocol_error",
                    tool_names=tool_names,
                    tool_args=first_args,
                    details={
                        "invalid_tool_call_count": len(invalid_calls),
                        "missing_field_tool_call_count": len(missing_fields),
                    },
                    response_preview=str(response.content),
                )

                response = self._new_ai_message_without_tool_calls(
                    response,
                    self._merge_protocol_error_into_content(response.content, protocol_error),
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
                    plan_mode=plan_mode,
                )
                if plan_mode:
                    # Keep request_user_input (at most 1, already enforced by
                    # sanitize) and read-only inspection calls. Mutating tools
                    # are dropped here and again by _filter_tool_calls_for_turn.
                    t_calls = [
                        tool_call
                        for tool_call in t_calls
                        if self._normalize_tool_name(tool_call.get("name") or "") == "request_user_input"
                        or self._tool_call_allowed_in_plan_mode(tool_call)
                    ]

                    response = self._new_ai_message_with_tool_calls(response, t_calls)
                    if t_calls and self._is_plan_approval_request(t_calls[0]):
                        content_text = stringify_content(response.content).strip()
                        if len(content_text) < _PLAN_MIN_CONTENT_LENGTH:
                            plan_guard_error = (
                                "INTERNAL TOOL PROTOCOL ERROR: You called request_user_input for plan approval "
                                "but did not write the plan in your response message content. "
                                "You MUST write the complete plan as visible text BEFORE asking the user to approve it. "
                                "Retry now: output the full plan text, then call request_user_input for approval."
                            )
                            protocol_error = self._merge_protocol_error_text(
                                protocol_error,
                                plan_guard_error,
                            )
                            protocol_issue = self._build_agent_protocol_issue(
                                current_turn_id=turn_id,
                                summary=plan_guard_error,
                                reason="plan_approval_without_plan_text",
                                tool_names=["request_user_input"],
                                tool_args=canonicalize_tool_args(t_calls[0].get("args")),
                                details={"content_length": len(content_text)},
                                response_preview=content_text,
                            )
                            t_calls = []
                            response = self._new_ai_message_without_tool_calls(response, plan_guard_error)
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
                    response = self._new_ai_message_with_tool_calls(response, t_calls)
                filtered_out_count = len(original_tool_calls) - len(t_calls)
                if filtered_out_count:
                    allowed_tool_name_set = self._normalized_allowed_tool_name_set(allowed_tool_names)
                    dropped_names = [
                        str(tool_call.get("name") or "").strip()
                        for tool_call in original_tool_calls
                        if self._normalize_tool_name(tool_call.get("name") or "") not in allowed_tool_name_set
                    ]

                    dropped_error = (
                        "INTERNAL TOOL PROTOCOL ERROR: model requested tool(s) that are not available in the current turn. "
                        "Issue a fresh valid tool call using only allowed tools."
                    )
                    protocol_error = self._merge_protocol_error_text(protocol_error, dropped_error)
                    if not t_calls:
                        protocol_issue = self._build_agent_protocol_issue(
                            current_turn_id=turn_id,
                            summary=dropped_error,
                            reason="tool_not_allowed_for_turn",
                            tool_names=dropped_names,
                            tool_args={},
                            details={"allowed_tool_names": list(allowed_tool_names or [])},
                            response_preview=str(response.content),
                        )

                    response = self._new_ai_message_with_tool_calls(response, t_calls)

            response = self._ensure_gemini_tool_call_signatures(response, t_calls)

            if (
                plan_mode
                and legacy_plan_approval
                and tools_available
                and not t_calls
                and not protocol_issue
                and not retry_user_input_turn
                and self._plan_approval_allowed(allowed_tool_names)
                and not self._current_turn_has_user_input_request(messages)
            ):
                t_calls = [self._plan_approval_tool_call(turn_id)]
                response = self._new_ai_message_with_tool_calls(response, t_calls)

            has_tool_calls = bool(tools_available and t_calls)

            if has_tool_calls and open_tool_issue and open_tool_issue.get("kind") == "approval_denied":
                response = self._new_ai_message_without_tool_calls(

                    response,
                    "Okay, I did not do that because you declined the action. Tell me what you want to do instead.",
                )
                has_tool_calls = False

            if protocol_issue is not None and not has_tool_calls:
                response = self._hide_message_from_ui(
                    response,
                    kind="protocol_invalid_response",
                    ui_notice=TOOL_ISSUE_UI_NOTICE,
                )

        outbound_messages.append(response)

        next_open_tool_issue = protocol_issue or open_tool_issue
        if (
            not has_tool_calls
            and isinstance(next_open_tool_issue, dict)
            and str(next_open_tool_issue.get("kind") or "").strip().lower() == "approval_denied"
        ):
            next_open_tool_issue = None
        elif not has_tool_calls and protocol_issue is None:
            # A normal prose answer after a tool/recovery issue is the model's final response.
            # Clear the stale issue so the graph does not re-enter recovery and ask the model
            # to answer the same turn again.
            if isinstance(next_open_tool_issue, dict) and hasattr(self, "recovery_manager"):
                recovery_state = self.recovery_manager.reset_after_success(
                    recovery_state,
                    current_turn_id=turn_id,
                )
            next_open_tool_issue = None

        if has_tool_calls:
            turn_outcome = TURN_OUTCOME_RUN_TOOLS
        elif protocol_issue is not None:
            turn_outcome = TURN_OUTCOME_RECOVER_AGENT
        else:
            turn_outcome = TURN_OUTCOME_FINISH_TURN

        return {
            "messages": outbound_messages,
            "turn_id": turn_id,
            "current_task": current_task,
            "turn_outcome": turn_outcome,
            "recovery_state": recovery_state,
            "pending_approval": None,
            "open_tool_issue": next_open_tool_issue,
            "has_protocol_error": bool(protocol_error),
            "last_tool_error": protocol_error,
            "last_tool_result": "",
            "_retry_user_input_turn": retry_user_input_turn,
            **token_usage_update,
        }
