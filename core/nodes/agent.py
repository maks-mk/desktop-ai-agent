from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from core.state import AgentState
from core.tool_args import canonicalize_tool_args
from core.constants import TOOL_ISSUE_UI_NOTICE
from core.message_utils import stringify_content
from core.text_tool_calls import extract_text_tool_calls

_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"


class AgentMixin:
    """Agent node facade and result-building helpers."""

    async def agent_node(self, state: AgentState):
        return await self.agent_turn.run(state)

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

            if tools_available and not t_calls and not invalid_calls:
                recovered_tool_calls = extract_text_tool_calls(
                    stringify_content(response.content),
                    allowed_tool_names=allowed_tool_names or self._all_tool_names,
                    id_prefix=f"txttc{max(0, int(turn_id or 0)) % 100:02d}",
                )
                if recovered_tool_calls.tool_calls:
                    response = response.model_copy(
                        update={
                            "content": recovered_tool_calls.cleaned_text or "I will use the requested tool.",
                            "tool_calls": recovered_tool_calls.tool_calls,
                        }
                    )
                    t_calls = list(recovered_tool_calls.tool_calls)

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
                    response_preview=str(response.content),
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
                            response_preview=str(response.content),
                        )
                    response = response.model_copy(update={"tool_calls": t_calls})

            response = self._ensure_gemini_tool_call_signatures(response, t_calls)
            has_tool_calls = bool(tools_available and t_calls)
            if has_tool_calls and open_tool_issue and open_tool_issue.get("kind") == "approval_denied":
                response = AIMessage(
                    content="Okay, I did not do that because you declined the action. Tell me what you want to do instead.",
                    additional_kwargs=response.additional_kwargs,
                    response_metadata=response.response_metadata,
                    usage_metadata=response.usage_metadata,
                    id=response.id,
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

        if has_tool_calls:
            turn_outcome = "run_tools"
        elif protocol_issue is not None:
            turn_outcome = "recover_agent"
        else:
            turn_outcome = "finish_turn"

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
