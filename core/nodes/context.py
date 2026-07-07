from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from core.state import AgentState
from core.node_errors import ProviderContextError


class ContextMixin:
    """Context building and provider-specific normalization helpers."""

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

    def _normalize_tool_call_id_for_provider(self, tool_call_id: str, *, used_ids: set[str]) -> str:
        return self.context_builder._normalize_tool_call_id_for_provider(tool_call_id, used_ids=used_ids)

    def _sanitize_messages_for_model(
        self,
        messages: List[BaseMessage],
        state: AgentState | None = None,
    ) -> List[BaseMessage]:
        return self.context_builder.sanitize_messages(messages, state=state)

    def _assert_provider_safe_agent_context(
        self,
        context: List[BaseMessage],
        state: AgentState | None = None,
    ) -> None:
        try:
            self.context_builder.assert_provider_safe_context(context, state=state)
        except RuntimeError as exc:
            raise ProviderContextError(str(exc)) from exc
