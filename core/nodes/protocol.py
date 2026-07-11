from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from core.tool_args import canonicalize_tool_args
from core.tool_issues import build_tool_issue
from core.self_correction_engine import repair_fingerprint
from core.message_utils import compact_text, stringify_content
from core.constants import TOOL_ISSUE_UI_NOTICE


class ProtocolMixin:
    """Message protocol helpers for building tool issues, filtering calls, and merging errors."""

    def _hide_message_from_ui(
        self,
        message: AIMessage,
        *,
        kind: str,
        ui_notice: str = "",
    ) -> AIMessage:
        metadata = dict(getattr(message, "additional_kwargs", {}) or {})
        internal = dict(metadata.get("agent_internal") or {})
        internal["kind"] = str(kind or "agent_internal_notice").strip() or "agent_internal_notice"
        internal["visible_in_ui"] = False
        notice_text = str(ui_notice or "").strip()
        if notice_text:
            internal["ui_notice"] = notice_text
        metadata["agent_internal"] = internal
        return message.model_copy(update={"additional_kwargs": metadata})

    def _log_interrupted_tool_result_if_needed(
        self,
        state,
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
            fragments.append(f"{pending_count} unfinished tool call(s)")
        if orphan_count:
            fragments.append(f"{orphan_count} orphan tool result")
        if duplicate_count:
            fragments.append(f"{duplicate_count} duplicate tool_call_id")
        if interleaving:
            fragments.append("messages were found between a tool call and its tool result")
        if not fragments:
            fragments.append("tool call history is corrupted")
        return "Internal tool history contract violated: " + ", ".join(fragments) + "."

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
        messages: List[Any],
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
                "You already asked for user input in this turn. Use the latest request_user_input ToolMessage as the user's final choice and continue "
                "instead of asking another question."
            ), True

        first_user_input_call, request_user_input_error = self._normalize_request_user_input_tool_call(user_input_calls[0])
        if request_user_input_error:
            return [], request_user_input_error, False

        if len(user_input_calls) == 1 and len(tool_calls) == 1:
            return [first_user_input_call], "", False

        if len(user_input_calls) > 1:
            return [first_user_input_call], (
                "INTERNAL TOOL PROTOCOL ERROR: request_user_input may appear at most once in a single assistant "
                "response. Extra user-input requests were dropped. Ask one question, wait for resume, then continue."
            ), False

        return [first_user_input_call], (
            "INTERNAL TOOL PROTOCOL ERROR: request_user_input cannot be combined with other tool calls in the same "
            "assistant response. Non-user-input tool calls were dropped. Ask one question, wait for resume, then continue."
        ), False

    def _normalize_request_user_input_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        normalized_call = dict(tool_call)
        normalized_args = canonicalize_tool_args(tool_call.get("args"))
        request_tool = self.tools_map.get("request_user_input")
        if request_tool is None:
            normalized_call["args"] = normalized_args
            return normalized_call, ""

        try:
            schema = request_tool.get_input_schema()
            validated = schema.model_validate(normalized_args)
            payload = validated.model_dump() if hasattr(validated, "model_dump") else normalized_args
        except ValidationError:
            return {}, (
                "INTERNAL TOOL PROTOCOL ERROR: request_user_input payload must contain one non-empty `question`, "
                "2 to 5 distinct non-empty `options`, and optional `recommended` matching one of those options exactly. "
                "Ask one blocking question, wait for resume, then continue."
            )
        except Exception:
            normalized_call["args"] = normalized_args
            return normalized_call, ""

        normalized_call["args"] = payload if isinstance(payload, dict) else normalized_args
        return normalized_call, ""
