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
    prepare_markdown_for_render,
)
from core.tool_args import canonicalize_tool_args
from ui.tool_message_utils import extract_tool_args, extract_tool_duration
from ui.visibility import get_internal_ui_notice, is_hidden_internal_message

DIFF_REGEX = re.compile(r"```diff\r?\n(.*?)```", re.DOTALL)
_INLINE_THOUGHT_BLOCK_RE = re.compile(r"<(think|thought)>.*?</\1>", re.IGNORECASE | re.DOTALL)
logger = logging.getLogger("agent")


def _clip_debug_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(+{len(text) - limit} chars)"


def _normalize_stream_chunk(chunk: Any) -> tuple[str, Any]:
    if isinstance(chunk, dict):
        return str(chunk["type"]), chunk["data"]
    if isinstance(chunk, tuple) and len(chunk) == 2:
        mode, payload = chunk
        return str(mode), payload
    raise TypeError(f"Unsupported stream chunk format: {type(chunk).__name__}")


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
        "_tool_id_aliases",
        "_tool_index_to_id",
        "_tool_id_to_index",
        "_tool_chunk_accumulators",
        "_synthetic_tool_counter",
        "_base_elapsed_seconds",
        "_last_internal_notice",
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
        self._tool_id_aliases: Dict[str, str] = {}
        self._tool_index_to_id: Dict[int, str] = {}
        self._tool_id_to_index: Dict[str, int] = {}
        self._tool_chunk_accumulators: Dict[str, Dict[int, AIMessageChunk]] = {}
        self._synthetic_tool_counter = 0
        self._base_elapsed_seconds = max(0.0, float(base_elapsed_seconds or 0.0))
        self._last_internal_notice = ""

    def _elapsed_seconds(self) -> float:
        return self._base_elapsed_seconds + max(0.0, time.perf_counter() - self.start_time)

    async def process_stream(self, stream) -> StreamProcessResult:
        self._emit_status(force=True)
        try:
            async for chunk in stream:
                mode, payload = _normalize_stream_chunk(chunk)
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

        active_ids = set(self.tool_buffer) | set(self.tool_start_times)
        for alias_id, target_id in list(self._tool_id_aliases.items()):
            if alias_id in active_ids or target_id in active_ids:
                continue
            self._tool_id_aliases.pop(alias_id, None)
            self._tool_id_to_index.pop(alias_id, None)

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
        if isinstance(last_message, (AIMessage, AIMessageChunk)):
            self._handle_agent_message(last_message, source="updates_agent")

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
                    self._handle_agent_message(message, source=f"updates_{node_name}")
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
            self._handle_agent_message(message, source="messages")
        elif node == "tools" and isinstance(message, ToolMessage):
            self._handle_tool_result(message)

    def _handle_agent_message(self, message: AIMessage | AIMessageChunk, *, source: str = "messages") -> None:
        self._handle_tool_calls_from_message(message, source=source)

        if is_hidden_internal_message(message):
            notice = get_internal_ui_notice(message)
            if notice and notice != self._last_internal_notice:
                self._last_internal_notice = notice
                self._emit(
                    "summary_notice",
                    {
                        "message": notice,
                        "kind": "agent_internal_notice",
                        "level": "warning",
                    },
                )
            return

        chunk = self._extract_text_content(message.content)
        if not chunk:
            chunk = self._extract_text_content(message)
        if not chunk:
            logger.debug(
                "Stream empty assistant chunk content_type=%s additional_kwargs=%s response_metadata=%s content_preview=%s",
                type(message.content).__name__,
                sorted((getattr(message, "additional_kwargs", {}) or {}).keys()),
                sorted((getattr(message, "response_metadata", {}) or {}).keys()),
                _clip_debug_text(message.content),
            )
            return

        merged_text = self._merge_assistant_text(chunk, source=source)
        if merged_text is None:
            return

        self.full_text = merged_text
        self.clean_full = self.full_text
        logger.debug(
            "Stream assistant chunk source=%s chunk_len=%s full_len=%s chunk_preview=%s",
            source,
            len(chunk),
            len(self.full_text),
            _clip_debug_text(chunk),
        )

        self._trim_text_buffers()

        self._emit_status()
        rendered_markdown = prepare_markdown_for_render(self.clean_full) if self.clean_full else ""
        logger.debug(
            "Stream emit_assistant_delta rendered_len=%s rendered_preview=%s",
            len(rendered_markdown),
            _clip_debug_text(rendered_markdown),
        )
        self._emit(
            "assistant_delta",
            {
                "text": chunk,
                "full_text": rendered_markdown,
            },
        )

    def _merge_assistant_text(self, incoming_text: str, *, source: str) -> str | None:
        text = str(incoming_text or "")
        if not text:
            return None

        current = self.full_text
        if not current:
            return text

        if text == current:
            return None

        if source == "messages":
            if text.startswith(current):
                return text
            return current + text

        if source == "updates_agent":
            if current == text:
                return None
            return text

        if text.startswith(current):
            return text
        return current + text

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return _INLINE_THOUGHT_BLOCK_RE.sub("", content)
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(StreamProcessor._extract_text_content(item) for item in content)
        if isinstance(content, dict):
            item_type = str(content.get("type") or "").strip().lower()
            if bool(content.get("thought")) or item_type in {
                "thinking",
                "thought",
                "reasoning",
                "reasoning_content",
                "reasoning_summary",
                "summary_text",
            }:
                return ""
            for key in ("text", "output_text", "content", "answer", "response", "final", "final_text"):
                if key in content:
                    text = StreamProcessor._extract_text_content(content.get(key))
                    if text:
                        return text
            return "".join(
                StreamProcessor._extract_text_content(content.get(key))
                for key in ("parts", "items", "content_blocks", "message", "messages", "data")
                if key in content
            )
        for key in ("content", "content_blocks", "text"):
            try:
                value = getattr(content, key)
            except Exception:
                continue
            if callable(value) or value is None:
                continue
            text = StreamProcessor._extract_text_content(value)
            if text:
                return text
        try:
            additional_kwargs = getattr(content, "additional_kwargs")
        except Exception:
            additional_kwargs = None
        if isinstance(additional_kwargs, dict):
            return StreamProcessor._extract_text_content(additional_kwargs.get("content_blocks"))
        return ""

    def _handle_tool_calls_from_message(self, message: AIMessage | AIMessageChunk, *, source: str) -> None:
        if isinstance(message, AIMessageChunk):
            chunk_calls = list(getattr(message, "tool_call_chunks", []) or [])
            if chunk_calls:
                self._remember_tool_call_chunk_ids(chunk_calls)
                accumulator_key = source or "messages"
                accumulators = self._tool_chunk_accumulators.setdefault(accumulator_key, {})
                processed_any = False
                for chunk in chunk_calls:
                    chunk_name = str(chunk.get("name") or "").strip() if isinstance(chunk, dict) else str(getattr(chunk, "name", "") or "").strip()
                    raw_index = chunk.get("index") if isinstance(chunk, dict) else getattr(chunk, "index", None)
                    try:
                        index = int(raw_index)
                    except (TypeError, ValueError):
                        continue
                    previous = accumulators.get(index)
                    current_chunk = AIMessageChunk(content="", tool_call_chunks=[self._tool_call_chunk_payload(chunk)])
                    if previous is None:
                        gathered = current_chunk
                    else:
                        try:
                            gathered = previous + current_chunk
                        except Exception:
                            gathered = current_chunk
                    accumulators[index] = gathered
                    tool_calls = list(getattr(gathered, "tool_calls", []) or [])
                    if not tool_calls:
                        continue
                    processed_any = True
                    self._process_tool_calls(
                        [
                            dict(tool_call, index=index, name=chunk_name or tool_call.get("name"))
                            for tool_call in tool_calls
                        ],
                        index_order=None,
                    )
                if not processed_any:
                    self._process_tool_calls(list(getattr(message, "tool_calls", []) or []), index_order=None)
                if getattr(message, "chunk_position", None) == "last":
                    self._tool_chunk_accumulators.pop(accumulator_key, None)
                return

        self._process_tool_calls(list(getattr(message, "tool_calls", []) or []), index_order=None)

    @staticmethod
    def _tool_call_chunk_payload(chunk: Any) -> Dict[str, Any]:
        if isinstance(chunk, dict):
            return {
                "name": chunk.get("name"),
                "args": chunk.get("args"),
                "id": chunk.get("id"),
                "index": chunk.get("index"),
            }
        return {
            "name": getattr(chunk, "name", None),
            "args": getattr(chunk, "args", None),
            "id": getattr(chunk, "id", None),
            "index": getattr(chunk, "index", None),
        }

    @staticmethod
    def _tool_indices_from_chunks(chunks: list[Any]) -> list[int]:
        indices: list[int] = []
        seen: set[int] = set()
        for chunk in chunks:
            raw_index = chunk.get("index") if isinstance(chunk, dict) else getattr(chunk, "index", None)
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index in seen:
                continue
            seen.add(index)
            indices.append(index)
        return indices

    def _remember_tool_call_chunk_ids(self, chunks: list[Any]) -> None:
        for chunk in chunks:
            if isinstance(chunk, dict):
                raw_index = chunk.get("index")
                raw_id = chunk.get("id")
                raw_name = chunk.get("name")
            else:
                raw_index = getattr(chunk, "index", None)
                raw_id = getattr(chunk, "id", None)
                raw_name = getattr(chunk, "name", None)
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            tool_id = str(raw_id or "").strip()
            self._tool_id_for_index(index, tool_id, str(raw_name or "").strip())

    def _process_tool_calls(self, tool_calls: list[Dict[str, Any]], *, index_order: list[int] | None) -> None:
        for position, tool_call in enumerate(tool_calls):
            normalized = dict(tool_call)
            if index_order is not None and position < len(index_order):
                normalized["index"] = index_order[position]
            self._remember_tool_call(normalized)
            self._emit_tool_started(normalized)

    def _resolve_tool_id(self, tool_id: Any) -> str:
        current = str(tool_id or "").strip()
        seen: set[str] = set()
        while current and current in self._tool_id_aliases and current not in seen:
            seen.add(current)
            current = str(self._tool_id_aliases.get(current) or "").strip()
        return current

    def _synthetic_tool_id(self, index: int) -> str:
        self._synthetic_tool_counter += 1
        return f"stream-tool-{index}-{self._synthetic_tool_counter}"

    def _tool_id_for_index(self, index: int, incoming_id: str, tool_name: str = "") -> str:
        incoming = self._resolve_tool_id(incoming_id)
        existing = self._tool_index_to_id.get(index)
        if existing and existing in self._completed_tool_ids:
            self._tool_index_to_id.pop(index, None)
            self._tool_id_to_index.pop(existing, None)
            existing = ""
        if existing:
            if incoming and incoming != existing:
                self._tool_id_aliases[incoming] = existing
                self._tool_id_to_index[incoming] = index
            return existing
        tool_id = incoming or self._synthetic_tool_id(index)
        self._tool_index_to_id[index] = tool_id
        self._tool_id_to_index[tool_id] = index
        return tool_id

    def _normalize_tool_call_id(self, tool_call: Dict[str, Any]) -> str:
        raw_id = str(tool_call.get("id") or "").strip()
        raw_index = tool_call.get("index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return self._resolve_tool_id(raw_id)
        return self._tool_id_for_index(index, raw_id, str(tool_call.get("name") or "").strip())

    def _mark_tool_completed(self, tool_id: str, raw_tool_id: str = "") -> None:
        completed_indexes: set[int] = set()
        for completed_id in {tool_id, raw_tool_id, self._resolve_tool_id(raw_tool_id)}:
            if completed_id:
                self._completed_tool_ids.pop(completed_id, None)
                self._completed_tool_ids[completed_id] = None
                index = self._tool_id_to_index.get(completed_id)
                if index is not None:
                    completed_indexes.add(index)
        for completed_id in {tool_id, raw_tool_id}:
            index = self._tool_id_to_index.pop(completed_id, None)
            if index is not None and self._tool_index_to_id.get(index) == tool_id:
                self._tool_index_to_id.pop(index, None)
                completed_indexes.add(index)
        for index in completed_indexes:
            self._clear_tool_chunk_accumulators_for_index(index)

    def _clear_tool_chunk_accumulators_for_index(self, index: int) -> None:
        for accumulator_key, accumulators in list(self._tool_chunk_accumulators.items()):
            accumulators.pop(index, None)
            if not accumulators:
                self._tool_chunk_accumulators.pop(accumulator_key, None)

    def _is_active_tool_id(self, tool_id: str) -> bool:
        return bool(tool_id) and (
            tool_id in self.tool_buffer
            or tool_id in self.tool_start_times
            or tool_id in self.printed_tool_ids
        )

    @staticmethod
    def _tool_args_compatible(candidate_args: Any, result_args: Any) -> bool:
        candidate = canonicalize_tool_args(candidate_args)
        result = canonicalize_tool_args(result_args)
        if not candidate or not result:
            return True
        matched = False
        for key, value in candidate.items():
            if key not in result:
                return False
            matched = True
            if result.get(key) != value:
                return False
        return matched

    def _match_active_tool_result(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        normalized_name = str(tool_name or "").strip()
        ordered_ids = list(dict.fromkeys([*self.tool_start_times.keys(), *self.tool_buffer.keys()]))
        candidates: list[str] = []
        for candidate_id in ordered_ids:
            if candidate_id in self._completed_tool_ids:
                continue
            candidate = self.tool_buffer.get(candidate_id, {})
            candidate_name = str(candidate.get("name") or "").strip()
            if normalized_name and candidate_name and normalized_name != candidate_name:
                continue
            if not normalized_name and len(ordered_ids) != 1:
                continue
            if not self._tool_args_compatible(candidate.get("args", {}), tool_args):
                continue
            candidates.append(candidate_id)
        return candidates[0] if candidates else ""

    def _resolve_tool_result_id(self, raw_tool_id: str, tool_name: str, tool_args: Dict[str, Any]) -> str:
        tool_id = self._resolve_tool_id(raw_tool_id)
        if self._is_active_tool_id(tool_id):
            return tool_id
        matched_id = self._match_active_tool_result(tool_name, tool_args)
        if matched_id and raw_tool_id:
            self._tool_id_aliases[raw_tool_id] = matched_id
            index = self._tool_id_to_index.get(matched_id)
            if index is not None:
                self._tool_id_to_index[raw_tool_id] = index
        return matched_id or tool_id

    def _remember_tool_call(self, tool_call: Dict[str, Any]) -> None:
        tool_id = self._normalize_tool_call_id(tool_call)
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
        tool_id = self._normalize_tool_call_id(tool_call)
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
        raw_tool_id = str(message.tool_call_id or "").strip()
        message_args = extract_tool_args(message)
        message_name = str(message.name or "").strip() or "unknown_tool"
        tool_id = self._resolve_tool_result_id(raw_tool_id, message_name, message_args)
        if tool_id and tool_id in self._completed_tool_ids:
            return
        if raw_tool_id and raw_tool_id in self._completed_tool_ids:
            return

        if tool_id and message_args:
            self._remember_tool_call(
                {
                    "id": tool_id,
                    "name": message_name,
                    "args": message_args,
                }
            )
        if tool_id in self.tool_buffer and tool_id not in self.printed_tool_ids:
            self._emit_tool_started({"id": tool_id, **self.tool_buffer[tool_id]})

        tool_info = self.tool_buffer.get(tool_id, {})
        tool_name = (
            str(tool_info.get("name") or "").strip()
            or message_name
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
            self._mark_tool_completed(tool_id, raw_tool_id)
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
        return node_labels.get(self.active_node, "")

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
        if not label:
            return
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
            },
        )
