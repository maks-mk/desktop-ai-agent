from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from core import constants
from core.config import AgentConfig
from core.message_utils import stringify_content
from core.runtime_prompt_policy import RuntimePromptContext, RuntimePromptPolicyBuilder
from core.state import AgentState

logger = logging.getLogger("agent")

IsInternalRetry = Callable[[BaseMessage], bool]
PromptLoader = Callable[[], str]
RunLogger = Callable[..., None]
RecoveryMessageBuilder = Callable[[Dict[str, Any] | None], SystemMessage | None]


class ContextBuilder:
    def __init__(
        self,
        *,
        config: AgentConfig,
        prompt_loader: PromptLoader,
        is_internal_retry: IsInternalRetry,
        log_run_event: RunLogger,
        recovery_message_builder: RecoveryMessageBuilder,
        provider_safe_tool_call_id_re: re.Pattern[str],
    ) -> None:
        self.config = config
        self._prompt_loader = prompt_loader
        self._is_internal_retry = is_internal_retry
        self._log_run_event = log_run_event
        self._recovery_message_builder = recovery_message_builder
        self._provider_safe_tool_call_id_re = provider_safe_tool_call_id_re
        self._runtime_policy_builder = RuntimePromptPolicyBuilder(config=config)

    def build(
        self,
        messages: List[BaseMessage],
        state: AgentState | None,
        *,
        summary: str,
        current_task: str,
        tools_available: bool,
        active_tool_names: List[str],
        open_tool_issue: Dict[str, Any] | None,
        recovery_state: Dict[str, Any] | None,
        user_choice_locked: bool = False,
    ) -> List[BaseMessage]:
        sanitized_messages = self.sanitize_messages(messages, state=state)
        full_context: List[BaseMessage] = [
            self._build_system_message(
                summary,
            )
        ]
        full_context.extend(
            self._runtime_policy_builder.build_messages(
                RuntimePromptContext(
                    current_task=current_task,
                    tools_available=tools_available,
                    active_tool_names=tuple(active_tool_names),
                    user_choice_locked=user_choice_locked,
                )
            )
        )
        safety_overlay = self._build_safety_overlay(tools_available=tools_available)
        if safety_overlay:
            full_context.append(SystemMessage(content=safety_overlay))
        issue_message = self._build_tool_issue_system_message(open_tool_issue)
        if issue_message:
            full_context.append(issue_message)
        recovery_message = self._recovery_message_builder(recovery_state)
        if recovery_message:
            full_context.append(recovery_message)
        full_context.extend(sanitized_messages)
        last_visible = self.get_last_model_visible_message(full_context)
        if not isinstance(last_visible, (HumanMessage, ToolMessage)):
            full_context.append(HumanMessage(content=current_task))
        if isinstance(last_visible, ToolMessage) and not recovery_message and not open_tool_issue:
            task_text = current_task.strip() or "Continue the user's latest explicit request."
            full_context.append(
                HumanMessage(
                    content=constants.RECOVERY_CONTINUE_PROMPT_TEMPLATE.format(
                        current_task=task_text
                    )
                )
            )
        return self.normalize_system_prefix(full_context)

    def sanitize_messages(
        self,
        messages: List[BaseMessage],
        *,
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        sanitized: List[BaseMessage] = []
        tool_call_id_map: Dict[str, str] = {}
        used_tool_call_ids: set[str] = set()
        remapped_count = 0
        normalized_content_count = 0

        for message in messages:
            normalized_message: BaseMessage = message

            if isinstance(message, (AIMessage, AIMessageChunk)):
                raw_tool_calls = list(getattr(message, "tool_calls", []) or [])
                if raw_tool_calls:
                    normalized_tool_calls: List[Dict[str, Any]] = []
                    tool_calls_changed = False
                    for tool_call in raw_tool_calls:
                        if not isinstance(tool_call, dict):
                            normalized_tool_calls.append(tool_call)
                            continue
                        cloned_call = dict(tool_call)
                        raw_id = str(cloned_call.get("id") or "").strip()
                        if raw_id:
                            mapped_id = tool_call_id_map.get(raw_id)
                            if not mapped_id:
                                mapped_id = self._normalize_tool_call_id_for_provider(
                                    raw_id,
                                    used_ids=used_tool_call_ids,
                                )
                                tool_call_id_map[raw_id] = mapped_id
                            if mapped_id != raw_id:
                                cloned_call["id"] = mapped_id
                                tool_calls_changed = True
                                remapped_count += 1
                        normalized_tool_calls.append(cloned_call)
                    if tool_calls_changed:
                        normalized_message = message.model_copy(update={"tool_calls": normalized_tool_calls})

            if isinstance(normalized_message, ToolMessage):
                raw_tool_id = str(normalized_message.tool_call_id or "").strip()
                if raw_tool_id:
                    mapped_tool_id = tool_call_id_map.get(raw_tool_id)
                    if not mapped_tool_id:
                        mapped_tool_id = self._normalize_tool_call_id_for_provider(
                            raw_tool_id,
                            used_ids=used_tool_call_ids,
                        )
                        tool_call_id_map[raw_tool_id] = mapped_tool_id
                    if mapped_tool_id != raw_tool_id:
                        normalized_message = normalized_message.model_copy(update={"tool_call_id": mapped_tool_id})
                        remapped_count += 1

            if self.config.provider == "openai":
                raw_content = getattr(normalized_message, "content", None)
                normalized_content = stringify_content(raw_content)
                if normalized_content != raw_content:
                    normalized_message = normalized_message.model_copy(update={"content": normalized_content})
                    normalized_content_count += 1

            if isinstance(normalized_message, HumanMessage):
                content = stringify_content(normalized_message.content).strip()
                if content == constants.REFLECTION_PROMPT:
                    continue
                last_visible = self.get_last_model_visible_message(sanitized)
                if isinstance(last_visible, ToolMessage):
                    sanitized.append(AIMessage(content="Continuing."))
                    self._log_run_event(
                        state,
                        "provider_role_order_bridge",
                        run_id=None if state is None else state.get("run_id", ""),
                        reason="user_after_tool",
                    )
            sanitized.append(normalized_message)

        if remapped_count:
            self._log_run_event(
                state,
                "provider_tool_call_id_remap",
                run_id=None if state is None else state.get("run_id", ""),
                remapped_count=remapped_count,
                distinct_ids=len(tool_call_id_map),
            )
        if normalized_content_count:
            self._log_run_event(
                state,
                "provider_content_normalized",
                run_id=None if state is None else state.get("run_id", ""),
                provider=self.config.provider,
                normalized_count=normalized_content_count,
            )
        return sanitized

    def normalize_system_prefix(self, context: List[BaseMessage]) -> List[BaseMessage]:
        system_messages: List[BaseMessage] = []
        non_system_messages: List[BaseMessage] = []
        for message in context:
            if self.message_is_provider_system(message):
                system_messages.append(message)
            else:
                non_system_messages.append(message)
        return [*system_messages, *non_system_messages]

    def assert_provider_safe_context(
        self,
        context: List[BaseMessage],
        *,
        state: AgentState | None = None,
    ) -> None:
        seen_non_system = False
        system_after_non_system = False
        for message in context:
            if self.message_is_provider_system(message):
                if seen_non_system:
                    system_after_non_system = True
                    break
                continue
            seen_non_system = True

        last_visible = self.get_last_model_visible_message(context)
        valid = isinstance(last_visible, (HumanMessage, ToolMessage)) and not system_after_non_system
        self._log_run_event(
            state,
            "provider_context_valid",
            run_id=None if state is None else state.get("run_id", ""),
            valid=valid,
            last_visible_type=type(last_visible).__name__ if last_visible else "",
            system_after_non_system=system_after_non_system,
        )
        if valid:
            return

        raise RuntimeError(
            "Provider-unsafe agent context: system messages must form a prefix and the last model-visible message must be HumanMessage or ToolMessage."
        )

    def get_last_model_visible_message(self, context: List[BaseMessage]) -> BaseMessage | None:
        for message in reversed(context):
            if self.message_is_provider_system(message):
                continue
            return message
        return None

    def message_is_provider_system(self, message: BaseMessage) -> bool:
        return self._message_role_for_provider(message) in {"system", "developer"}

    def _message_role_for_provider(self, message: BaseMessage) -> str:
        role = ""
        if isinstance(message, SystemMessage):
            return "system"
        raw_role = getattr(message, "role", "")
        if isinstance(raw_role, str):
            role = raw_role.strip().lower()
        if not role:
            raw_type = getattr(message, "type", "")
            if isinstance(raw_type, str):
                role = raw_type.strip().lower()
        return role

    def _normalize_tool_call_id_for_provider(self, raw_id: str, *, used_ids: set[str]) -> str:
        normalized = str(raw_id or "").strip()
        if self._provider_safe_tool_call_id_re.match(normalized) and normalized not in used_ids:
            used_ids.add(normalized)
            return normalized

        seed = normalized or "tool_call"
        suffix = 0
        while True:
            hash_source = seed if suffix == 0 else f"{seed}:{suffix}"
            candidate = hashlib.sha1(hash_source.encode("utf-8")).hexdigest()[:9]
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate
            suffix += 1

    def _build_safety_overlay(self, *, tools_available: bool) -> str:
        if not tools_available:
            return ""
        overlay_lines: List[str] = []
        overlay_lines.append(
            "MANDATORY: Before every tool call, write 1-2 short plain-text sentences describing your immediate action."
        )
        overlay_lines.append(
            "SAFETY POLICY: Any write, delete, move, or process-launch working directory must stay inside the active workspace."
        )
        if self.config.enable_approvals:
            overlay_lines.append(
                "LEGACY POLICY: Some protected tools may still request explicit approval when approvals are enabled."
            )
        if self.config.enable_shell_tool:
            overlay_lines.append(
                "SAFETY POLICY: Shell execution is high risk. Prefer safer project-local tools whenever possible."
            )
        return "\n".join(overlay_lines).strip()

    def _build_tool_issue_system_message(self, open_tool_issue: Dict[str, Any] | None) -> SystemMessage | None:
        if not open_tool_issue:
            return None

        issue_summary = str(open_tool_issue.get("summary", "")).strip()
        if open_tool_issue.get("kind") == "approval_denied":
            return SystemMessage(
                content=(
                    "TOOL EXECUTION DENIED BY USER:\n"
                    f"{issue_summary}\n\n"
                    "The user explicitly rejected this tool call. "
                    "Do not simulate the denied tool or describe imaginary results. "
                    "Do not make any more tool calls in this turn. "
                    "Reply briefly: say that you did not do it because the user chose No, then wait for the next instruction."
                )
            )

        return SystemMessage(
            content=constants.UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE.format(
                error_summary=issue_summary
            )
        )

    def _build_system_message(
        self,
        summary: str,
    ) -> SystemMessage:
        raw_prompt = self._prompt_loader()

        prompt = raw_prompt.replace("{{current_date}}", datetime.now().strftime("%Y-%m-%d"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))

        if summary:
            prompt += f"\n\n<memory>\n{summary}\n</memory>"

        return SystemMessage(content=prompt)
