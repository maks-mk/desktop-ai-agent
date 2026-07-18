from __future__ import annotations

from typing import Callable, List

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage


IsInternalRetry = Callable[[BaseMessage], bool]


class MessageContextHelper:
    """History-derived context helpers with no phrase or keyword lexicon dependency."""

    def non_internal_human_indexes(
        self,
        messages: List[BaseMessage],
        is_internal_retry: IsInternalRetry,
    ) -> List[int]:
        indexes: List[int] = []
        for idx, message in enumerate(messages):
            if isinstance(message, HumanMessage) and not is_internal_retry(message):
                indexes.append(idx)
        return indexes

    def _turn_bounds(
        self,
        messages: List[BaseMessage],
        turn_id: int,
        is_internal_retry: IsInternalRetry,
    ) -> tuple[int, int] | None:
        human_indexes = self.non_internal_human_indexes(messages, is_internal_retry)
        if turn_id < 1 or turn_id > len(human_indexes):
            return None
        start = human_indexes[turn_id - 1] + 1
        end = human_indexes[turn_id] if turn_id < len(human_indexes) else len(messages)
        return start, end

    @staticmethod
    def _tool_names_from_messages(messages: List[BaseMessage]) -> List[str]:
        names: List[str] = []
        seen: set[str] = set()
        for message in messages:
            if isinstance(message, ToolMessage):
                name = str(message.name or "").strip()
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
                continue
            if isinstance(message, (AIMessage, AIMessageChunk)):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    if not isinstance(tool_call, dict):
                        continue
                    name = str(tool_call.get("name") or "").strip()
                    if name and name not in seen:
                        names.append(name)
                        seen.add(name)
        return names

    def current_turn_has_tool_evidence(
        self,
        messages: List[BaseMessage],
        is_internal_retry: IsInternalRetry,
    ) -> bool:
        human_indexes = self.non_internal_human_indexes(messages, is_internal_retry)
        if not human_indexes:
            return False
        start = human_indexes[-1] + 1
        return bool(self._tool_names_from_messages(messages[start:]))

    def had_tool_activity_in_previous_turn(
        self,
        messages: List[BaseMessage],
        current_turn_id: int,
        is_internal_retry: IsInternalRetry,
    ) -> bool:
        bounds = self._turn_bounds(messages, current_turn_id - 1, is_internal_retry)
        if bounds is None:
            return False
        start, end = bounds
        return bool(self._tool_names_from_messages(messages[start:end]))

    def recent_tool_context_names(
        self,
        messages: List[BaseMessage],
        is_internal_retry: IsInternalRetry,
    ) -> List[str]:
        human_indexes = self.non_internal_human_indexes(messages, is_internal_retry)
        if len(human_indexes) < 2:
            return []
        start = human_indexes[-2] + 1
        end = human_indexes[-1]
        return self._tool_names_from_messages(messages[start:end])
