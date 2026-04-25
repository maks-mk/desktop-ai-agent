import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage

from core.message_utils import is_tool_message_error, stringify_content
from core.text_utils import (
    TokenTracker,
    build_tool_ui_labels,
    classify_tool_args_state,
    format_tool_display,
    format_tool_output,
    parse_thought,
    prepare_markdown_for_render,
)
from core.tool_args import canonicalize_tool_args
from ui.tool_message_utils import extract_tool_args, extract_tool_duration
from ui.visibility import is_hidden_internal_message

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
    cancelled: bool = False
    failed: bool = False
    error_message: str = ""
    cancelled_tools: list[Dict[str, Any]] = field(default_factory=list)
    events: list[StreamEvent] = field(default_factory=list)
    elapsed_seconds: float = 0.0


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
        "_text_max_chars",
        "_events_max",
        "_tool_buffer_max",
        "_completed_tool_ids",
        "_base_elapsed_seconds",
    )

    def __init__(
        self,
        emit_event: Callable[[StreamEvent], None] | None = None,
        *,
        text_max_chars: int = 120000,
        events_max: int = 400,
        tool_buffer_max: int = 128,
        base_elapsed_seconds: float = 0.0,
    ):
        self.emit_event = emit_event
        self.tracker = TokenTracker()
        self.full_text = ""
        self.clean_full = ""
        self.has_thought = False
        self.printed_tool_ids: Set[str] = set()
        self.tool_buffer: Dict[str, Dict[str, Any]] = {}
        self.tool_start_times: Dict[str, float] = {}
        self.start_time = time.perf_counter()
        self.pending_interrupt: Optional[Dict[str, Any]] = None
        self.active_node = "agent"
        self.events: list[StreamEvent] = []
        self._last_status: tuple[str, str] | None = None
        self._text_max_chars = max(1, int(text_max_chars))
        self._events_max = max(1, int(events_max))
        self._tool_buffer_max = max(1, int(tool_buffer_max))
        self._completed_tool_ids: dict[str, None] = {}
        self._base_elapsed_seconds = max(0.0, float(base_elapsed_seconds or 0.0))

    def _elapsed_seconds(self) -> float:
        return self._base_elapsed_seconds + max(0.0, time.perf_counter() - self.start_time)

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
            cancelled_tools = self._emit_interrupted_tool_results(reason="cancelled")
            self._emit("run_failed", {"message": "Cancelled"})
            return StreamProcessResult(
                stats=None,
                cancelled=True,
                cancelled_tools=cancelled_tools,
                events=list(self.events),
                elapsed_seconds=self._elapsed_seconds(),
            )
        except Exception as exc:
            logger.debug("Stream processing failed: %s", exc)
            self._emit("run_failed", {"message": str(exc)})
            return StreamProcessResult(
                stats=None,
                failed=True,
                error_message=str(exc),
                events=list(self.events),
                elapsed_seconds=self._elapsed_seconds(),
            )

        if self.pending_interrupt is not None:
            interrupt_payload = self.pending_interrupt
            self.pending_interrupt = None
            return StreamProcessResult(
                stats=None,
                interrupt=interrupt_payload,
                events=list(self.events),
                elapsed_seconds=self._elapsed_seconds(),
            )

        duration = self._elapsed_seconds()
        stats = self.tracker.render(duration)
        self._emit("run_finished", {"stats": stats, "duration": duration})
        return StreamProcessResult(
            stats=stats,
            events=list(self.events),
            elapsed_seconds=duration,
        )

    def _emit(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        event = StreamEvent(event_type, payload or {})
        self.events.append(event)
        if len(self.events) > self._events_max:
            overflow = len(self.events) - self._events_max
            del self.events[:overflow]
        if self.emit_event:
            self.emit_event(event)

    def _trim_text_buffers(self) -> None:
        if len(self.full_text) > self._text_max_chars:
            self.full_text = "… " + self.full_text[-(self._text_max_chars - 2) :]
        if len(self.clean_full) > self._text_max_chars:
            self.clean_full = "… " + self.clean_full[-(self._text_max_chars - 2) :]

    def _trim_tool_buffers(self) -> None:
        while len(self.tool_buffer) > self._tool_buffer_max:
            oldest_id = next(iter(self.tool_buffer), None)
            if oldest_id is None:
                break
            self.tool_buffer.pop(oldest_id, None)
            self.tool_start_times.pop(oldest_id, None)
            self.printed_tool_ids.discard(oldest_id)
        while len(self.tool_start_times) > self._tool_buffer_max:
            oldest_id = next(iter(self.tool_start_times), None)
            if oldest_id is None:
                break
            self.tool_start_times.pop(oldest_id, None)
            self.printed_tool_ids.discard(oldest_id)

        while len(self._completed_tool_ids) > self._tool_buffer_max:
            oldest_id = next(iter(self._completed_tool_ids), None)
            if oldest_id is None:
                break
            self._completed_tool_ids.pop(oldest_id, None)

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

        for node_name, node_payload in payload.items():
            if node_name in {"agent", "summarize", "__interrupt__"}:
                continue
            if not isinstance(node_payload, dict):
                continue
            node_messages = node_payload.get("messages", [])
            if not isinstance(node_messages, list):
                node_messages = [node_messages]
            for message in node_messages:
                if not isinstance(message, (AIMessage, AIMessageChunk, ToolMessage)):
                    continue
                self.tracker.update_from_message(message)
                if isinstance(message, (AIMessage, AIMessageChunk)):
                    self._handle_agent_message(message)
                elif isinstance(message, ToolMessage):
                    self._handle_tool_result(message)

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
                "kind": "auto_summary",
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

        if is_hidden_internal_message(message):
            return

        chunk = self._extract_text_content(message.content)
        if not chunk:
            return

        self.full_text += chunk
        if "<th" in self.full_text:
            _, self.clean_full, self.has_thought = parse_thought(self.full_text)
        else:
            self.clean_full = self.full_text
            self.has_thought = False

        self._trim_text_buffers()

        self._emit_status()
        rendered_markdown = prepare_markdown_for_render(self.clean_full) if self.clean_full else ""
        self._emit(
            "assistant_delta",
            {
                "text": chunk,
                "full_text": rendered_markdown,
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
        if tool_id in self._completed_tool_ids:
            return
        existing = self.tool_buffer.get(tool_id, {})
        existing_name = str(existing.get("name") or "").strip()
        existing_args = existing.get("args", {})
        incoming_args = tool_call.get("args", {})
        merged_args = self._merge_tool_args(existing_args, incoming_args)
        tool_name = (
            str(tool_call.get("name") or "").strip()
            or existing_name
            or "unknown_tool"
        )
        self.tool_buffer[tool_id] = {
            "name": tool_name,
            "args": merged_args,
        }
        self._trim_tool_buffers()

        # If the card is already visible and we learned richer args later (stream races),
        # push a lightweight refresh so UI updates command/args immediately.
        if tool_id in self.printed_tool_ids and (merged_args != existing_args or tool_name != existing_name):
            payload = self._build_tool_event_payload(tool_id, tool_name, merged_args, phase="preparing")
            payload["refresh"] = True
            self._emit("tool_started", payload)

    def _merge_tool_args(self, current_args: Any, incoming_args: Any) -> Dict[str, Any]:
        current = canonicalize_tool_args(current_args)
        incoming = canonicalize_tool_args(incoming_args)
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
        if not tool_id or tool_id in self.printed_tool_ids or tool_id in self._completed_tool_ids:
            return

        tool_info = self.tool_buffer.get(tool_id, {})
        tool_name = (
            str(tool_call.get("name") or "").strip()
            or str(tool_info.get("name") or "").strip()
            or "unknown_tool"
        )
        tool_args = self._merge_tool_args(tool_info.get("args", {}), tool_call.get("args", {}))
        self.tool_buffer[tool_id] = {"name": tool_name, "args": tool_args}

        self.tool_start_times[tool_id] = time.perf_counter()
        self.printed_tool_ids.add(tool_id)
        self.active_node = "tools"
        self._emit_status(force=True)
        self._emit("tool_started", self._build_tool_event_payload(tool_id, tool_name, tool_args, phase="preparing"))

    def _handle_tool_result(self, message: ToolMessage) -> None:
        tool_id = str(message.tool_call_id or "")
        if tool_id and tool_id in self._completed_tool_ids:
            return

        message_args = extract_tool_args(message)
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

        elapsed = extract_tool_duration(message)
        start_time = self.tool_start_times.pop(tool_id, None)
        if elapsed is None and start_time is not None:
            elapsed = max(0.0, time.perf_counter() - start_time)

        diff_blocks = [match.group(1).strip() for match in DIFF_REGEX.finditer(content_str)]
        payload = {
            "content": content_str,
            "summary": summary,
            "is_error": is_error,
            "duration": elapsed,
            "diff": diff_blocks[0] if diff_blocks else "",
            "diff_blocks": diff_blocks,
        }
        payload.update(self._build_tool_event_payload(tool_id, tool_name, tool_args, phase="finished", is_error=is_error))
        self._emit("tool_finished", payload)
        if tool_id:
            self.tool_buffer.pop(tool_id, None)
            self._completed_tool_ids.pop(tool_id, None)
            self._completed_tool_ids[tool_id] = None
            self._trim_tool_buffers()
        if diff_blocks:
            self._emit("tool_diff", {"tool_id": tool_id, "diff": diff_blocks[0], "diff_blocks": diff_blocks})

        self.active_node = "agent"
        self._emit_status(force=True)

    def _emit_interrupted_tool_results(self, reason: str) -> list[Dict[str, Any]]:
        interrupted_payloads: list[Dict[str, Any]] = []
        active_tool_ids = list(self.tool_start_times.keys()) or list(self.tool_buffer.keys())
        for tool_id in active_tool_ids:
            tool_info = self.tool_buffer.get(tool_id, {})
            tool_name = str(tool_info.get("name") or "unknown_tool")
            tool_args = self._merge_tool_args(tool_info.get("args", {}), {})
            elapsed = None
            start_time = self.tool_start_times.pop(tool_id, None)
            if start_time is not None:
                elapsed = time.perf_counter() - start_time
            content = "Error: Execution interrupted (system limit reached or user stop). Please retry."
            payload = {
                "content": content,
                "summary": format_tool_output(tool_name, content, True),
                "is_error": True,
                "duration": elapsed,
                "diff": "",
                "diff_blocks": [],
                "interrupted": True,
                "interruption_reason": reason,
            }
            payload.update(self._build_tool_event_payload(tool_id, tool_name, tool_args, phase="finished", is_error=True))
            self._emit("tool_finished", payload)
            interrupted_payloads.append(payload)
            self.tool_buffer.pop(tool_id, None)
            self.printed_tool_ids.discard(tool_id)
            if tool_id:
                self._completed_tool_ids.pop(tool_id, None)
                self._completed_tool_ids[tool_id] = None
                self._trim_tool_buffers()

        if interrupted_payloads:
            self.active_node = "agent"
            self._emit_status(force=True)
        return interrupted_payloads

    def _status_label(self) -> str:
        node_labels = {
            "agent": "Analyzing request",
            "recovery": "Reviewing results",
            "tools": "Running tools",
            "summarize": "Compressing context",
            "approval": "Waiting for approval",
        }
        return node_labels.get(self.active_node, "Thinking")

    def _status_phase(self) -> str:
        phase_map = {
            "agent": "working",
            "recovery": "reviewing",
            "tools": "active",
            "summarize": "system",
            "approval": "waiting",
        }
        return phase_map.get(self.active_node, "working")

    def _status_elapsed_text(self) -> str:
        elapsed = self._elapsed_seconds()
        if elapsed < 4.0:
            return ""
        if elapsed < 10.0:
            return f"{elapsed:.1f}s"
        return f"{int(round(elapsed))}s"

    def _build_tool_event_payload(
        self,
        tool_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        phase: str,
        is_error: bool = False,
    ) -> Dict[str, Any]:
        args_state = classify_tool_args_state(tool_name, tool_args)
        labels = build_tool_ui_labels(tool_name, tool_args, phase=phase, is_error=is_error)
        display_state = "finished" if phase == "finished" else ("resolved" if args_state == "complete" else "preview")
        if phase == "preparing" and args_state == "complete":
            phase = "running"
        status_hint = labels.get("subtitle") or ("Preparing tool call…" if display_state == "preview" else "")
        return {
            "tool_id": tool_id,
            "name": tool_name,
            "args": tool_args,
            "display": labels.get("title") or tool_name,
            "subtitle": labels.get("subtitle", ""),
            "raw_display": labels.get("raw_display") or format_tool_display(tool_name, tool_args),
            "args_state": args_state,
            "display_state": display_state,
            "phase": phase,
            "status_hint": status_hint,
            "source_kind": labels.get("source_kind", "tool"),
        }

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
                "elapsed": self._elapsed_seconds(),
                "elapsed_text": self._status_elapsed_text(),
                "phase": self._status_phase(),
                "has_thought": self.has_thought,
            },
        )
