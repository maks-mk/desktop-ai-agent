import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage

from core.message_utils import is_tool_message_error, stringify_content
from core.text_utils import (
    TokenTracker,
    format_tool_display,
    format_tool_output,
    parse_thought,
    prepare_markdown_for_render,
)

DIFF_REGEX = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)
logger = logging.getLogger("agent")


@dataclass(frozen=True)
class StreamEvent:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamProcessResult:
    stats: Optional[str]
    interrupt: Optional[Dict[str, Any]] = None
    events: list[StreamEvent] = field(default_factory=list)


class StreamProcessor:
    __slots__ = (
        "emit_event",
        "tracker",
        "full_text",
        "clean_full",
        "has_thought",
        "printed_tool_ids",
        "tool_buffer",
        "tool_start_times",
        "start_time",
        "pending_interrupt",
        "active_node",
        "events",
        "_last_status",
    )

    def __init__(self, emit_event: Callable[[StreamEvent], None] | None = None):
        self.emit_event = emit_event
        self.tracker = TokenTracker()
        self.full_text = ""
        self.clean_full = ""
        self.has_thought = False
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}
        self.tool_start_times: Dict[str, float] = {}
        self.start_time = time.time()
        self.pending_interrupt: Optional[Dict[str, Any]] = None
        self.active_node = "agent"
        self.events: list[StreamEvent] = []
        self._last_status: tuple[str, str] | None = None

    async def process_stream(self, stream) -> StreamProcessResult:
        self._emit_status(force=True)
        try:
            async for chunk in stream:
                # LangGraph v2 streaming yields dicts: {"type": ..., "data": ...}
                mode = chunk["type"]
                payload = chunk["data"]
                self._handle_stream_event(mode, payload)
                if self.pending_interrupt is not None:
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._emit("run_failed", {"message": "Cancelled"})
            return StreamProcessResult(stats=None, events=list(self.events))
        except Exception as exc:
            logger.debug("Stream processing failed: %s", exc)
            self._emit("run_failed", {"message": str(exc)})
            return StreamProcessResult(stats=None, events=list(self.events))

        if self.pending_interrupt is not None:
            interrupt_payload = self.pending_interrupt
            self.pending_interrupt = None
            self._emit("approval_requested", {"interrupt": interrupt_payload})
            return StreamProcessResult(stats=None, interrupt=interrupt_payload, events=list(self.events))

        duration = time.time() - self.start_time
        stats = self.tracker.render(duration)
        self._emit("run_finished", {"stats": stats, "duration": duration})
        return StreamProcessResult(stats=stats, events=list(self.events))

    def _emit(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        event = StreamEvent(event_type, payload or {})
        self.events.append(event)
        if self.emit_event:
            self.emit_event(event)

    def _handle_stream_event(self, mode: str, payload: Any) -> None:
        if mode == "updates":
            self._handle_updates(payload)
        elif mode == "messages":
            self._handle_messages(payload)

    def _handle_updates(self, payload: Dict[str, Any]) -> None:
        if "__interrupt__" in payload:
            self.active_node = "approval"
            self._emit_status(force=True)
            interrupt_entries = payload.get("__interrupt__") or []
            if interrupt_entries:
                interrupt_value = getattr(interrupt_entries[0], "value", interrupt_entries[0])
                if isinstance(interrupt_value, dict):
                    self.pending_interrupt = interrupt_value
                else:
                    self.pending_interrupt = {"value": interrupt_value}
            return

        self.tracker.update_from_node_update(payload)

        summarize_payload = payload.get("summarize") or {}
        if summarize_payload:
            self._handle_summarize_update(summarize_payload)

        agent_payload = payload.get("agent") or {}
        messages = agent_payload.get("messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        last_message = messages[-1] if messages else None
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                self._remember_tool_call(tool_call)
                self._emit_tool_started(tool_call)

    def _handle_summarize_update(self, payload: Dict[str, Any]) -> None:
        summary_text = payload.get("summary")
        removed_messages = payload.get("messages") or []
        remove_count = sum(1 for item in removed_messages if isinstance(item, RemoveMessage))
        if not summary_text:
            return
        rendered_count = remove_count if remove_count > 0 else len(removed_messages)
        self._emit(
            "summary_notice",
            {
                "message": f"Context compressed automatically ({rendered_count} message(s) summarized).",
                "count": rendered_count,
            },
        )

    def _handle_messages(self, payload: tuple[Any, Dict[str, Any]]) -> None:
        message, metadata = payload
        node = metadata.get("langgraph_node")
        self.tracker.update_from_message(message)

        if node:
            self.active_node = node
            self._emit_status()

        if node == "agent" and isinstance(message, (AIMessage, AIMessageChunk)):
            self._handle_agent_message(message)
        elif node == "tools" and isinstance(message, ToolMessage):
            self._handle_tool_result(message)

    def _handle_agent_message(self, message: AIMessage | AIMessageChunk) -> None:
        if message.tool_calls:
            for tool_call in message.tool_calls:
                self._remember_tool_call(tool_call)
                self._emit_tool_started(tool_call)

        chunk = self._extract_text_content(message.content)
        if not chunk:
            return

        self.full_text += chunk
        if "<th" in self.full_text:
            _, self.clean_full, self.has_thought = parse_thought(self.full_text)
        else:
            self.clean_full = self.full_text
            self.has_thought = False

        self._emit_status()
        self._emit(
            "assistant_delta",
            {
                "text": chunk,
                "full_text": prepare_markdown_for_render(self.clean_full),
                "has_thought": self.has_thought,
            },
        )

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return ""

    def _remember_tool_call(self, tool_call: Dict[str, Any]) -> None:
        tool_id = tool_call.get("id")
        if not tool_id:
            return
        existing = self.tool_buffer.get(tool_id, {})
        existing_args = existing.get("args", {})
        incoming_args = tool_call.get("args", {})
        merged_args = self._merge_tool_args(existing_args, incoming_args)
        tool_name = (
            str(tool_call.get("name") or "").strip()
            or str(existing.get("name") or "").strip()
            or "unknown_tool"
        )
        self.tool_buffer[tool_id] = {
            "name": tool_name,
            "args": merged_args,
        }

    def _merge_tool_args(self, current_args: Any, incoming_args: Any) -> Dict[str, Any]:
        current = dict(current_args) if isinstance(current_args, dict) else {}
        incoming = dict(incoming_args) if isinstance(incoming_args, dict) else {}
        if not incoming:
            return current

        merged = dict(current)
        for key, value in incoming.items():
            if value is None:
                continue
            if isinstance(value, str):
                if not value.strip():
                    continue
                merged[key] = value
                continue
            if isinstance(value, dict):
                base = merged.get(key)
                merged[key] = self._merge_tool_args(base if isinstance(base, dict) else {}, value)
                continue
            if isinstance(value, list):
                if not value:
                    continue
                merged[key] = value
                continue
            merged[key] = value

        return merged

    def _emit_tool_started(self, tool_call: Dict[str, Any]) -> None:
        tool_id = tool_call.get("id")
        if not tool_id or tool_id in self.printed_tool_ids:
            return

        tool_info = self.tool_buffer.get(tool_id, {})
        tool_name = (
            str(tool_call.get("name") or "").strip()
            or str(tool_info.get("name") or "").strip()
            or "unknown_tool"
        )
        tool_args = self._merge_tool_args(tool_info.get("args", {}), tool_call.get("args", {}))
        self.tool_buffer[tool_id] = {"name": tool_name, "args": tool_args}

        self.tool_start_times[tool_id] = time.time()
        self.printed_tool_ids.add(tool_id)
        self.active_node = "tools"
        self._emit_status(force=True)
        self._emit(
            "tool_started",
            {
                "tool_id": tool_id,
                "name": tool_name,
                "args": tool_args,
                "display": format_tool_display(
                    tool_name,
                    tool_args,
                ),
            },
        )

    def _handle_tool_result(self, message: ToolMessage) -> None:
        tool_id = str(message.tool_call_id or "")
        message_args = self._extract_tool_args_from_message(message)
        if tool_id and message_args:
            self._remember_tool_call(
                {
                    "id": tool_id,
                    "name": str(message.name or "").strip() or "unknown_tool",
                    "args": message_args,
                }
            )
        if tool_id in self.tool_buffer and tool_id not in self.printed_tool_ids:
            self._emit_tool_started({"id": tool_id, **self.tool_buffer[tool_id]})

        tool_info = self.tool_buffer.get(tool_id, {})
        tool_name = (
            str(tool_info.get("name") or "").strip()
            or str(message.name or "").strip()
            or "unknown_tool"
        )
        tool_args = self._merge_tool_args(tool_info.get("args", {}), message_args)
        if not tool_args:
            self._emit(
                "tool_args_missing",
                {
                    "tool_id": tool_id,
                    "name": tool_name,
                    "message": "No canonical tool args were available when tool result arrived.",
                },
            )

        content_str = stringify_content(message.content)
        is_error = is_tool_message_error(message)
        summary = format_tool_output(tool_name, content_str, is_error)

        elapsed = None
        start_time = self.tool_start_times.pop(tool_id, None)
        if start_time is not None:
            elapsed = time.time() - start_time

        diff_blocks = [match.group(1).strip() for match in DIFF_REGEX.finditer(content_str)]
        payload = {
            "tool_id": tool_id,
            "name": tool_name,
            "args": tool_args,
            "display": format_tool_display(tool_name, tool_args),
            "content": content_str,
            "summary": summary,
            "is_error": is_error,
            "duration": elapsed,
            "diff": diff_blocks[0] if diff_blocks else "",
            "diff_blocks": diff_blocks,
        }
        self._emit("tool_finished", payload)
        if tool_id:
            self.tool_buffer.pop(tool_id, None)
        if diff_blocks:
            self._emit("tool_diff", {"tool_id": tool_id, "diff": diff_blocks[0], "diff_blocks": diff_blocks})

        self.active_node = "agent"
        self._emit_status(force=True)

    @staticmethod
    def _extract_tool_args_from_message(message: ToolMessage) -> Dict[str, Any]:
        metadata = getattr(message, "additional_kwargs", {}) or {}
        candidates = []
        if isinstance(metadata, dict):
            candidates.append(metadata.get("tool_args"))
            candidates.append(metadata.get("args"))
            tool_call_obj = metadata.get("tool_call")
            if isinstance(tool_call_obj, dict):
                candidates.append(tool_call_obj.get("args"))

        for candidate in candidates:
            if isinstance(candidate, dict):
                return dict(candidate)
            if isinstance(candidate, str):
                raw = candidate.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return {}

    def _status_label(self) -> str:
        node_labels = {
            "agent": "Thinking",
            "critic": "Reviewing",
            "tools": "Running tools",
            "summarize": "Compressing context",
            "approval": "Waiting for approval",
        }
        return node_labels.get(self.active_node, "Thinking")

    def _emit_status(self, force: bool = False) -> None:
        label = self._status_label()
        current = (self.active_node, label)
        if not force and current == self._last_status:
            return
        self._last_status = current
        self._emit(
            "status_changed",
            {
                "node": self.active_node,
                "label": label,
                "elapsed": time.time() - self.start_time,
                "has_thought": self.has_thought,
            },
        )
