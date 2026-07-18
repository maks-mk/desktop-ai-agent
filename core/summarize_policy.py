from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from core.message_utils import stringify_content


IsInternalRetry = Callable[[BaseMessage], bool]

logger = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Tiktoken — lazy initialization.
# The encoder is created once and reused for the lifetime of the process.
# cl100k_base is the GPT-4/GPT-3.5 encoding. It is not ideal for Gemini, but
# its accuracy is within about ±15%, which is far better than a char heuristic.
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODER = None
_TIKTOKEN_AVAILABLE: Optional[bool] = None  # None = not checked yet


def _get_encoder():
    global _TIKTOKEN_ENCODER, _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is True:
        return _TIKTOKEN_ENCODER
    if _TIKTOKEN_AVAILABLE is False:
        return None
    # First call — try to initialize it
    try:
        import tiktoken
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        _TIKTOKEN_AVAILABLE = True
        logger.debug("tiktoken encoder initialised (cl100k_base).")
    except Exception as exc:
        _TIKTOKEN_AVAILABLE = False
        logger.warning(
            "tiktoken unavailable, falling back to char-based token estimate: %s", exc
        )
    return _TIKTOKEN_ENCODER


# Overhead for each message in the Chat Completions API:
# role token + separators = ~4 tokens (per the OpenAI tiktoken spec).
_MESSAGE_OVERHEAD_TOKENS = 4


def _count_tokens_tiktoken(text: str) -> int:
    """Count tokens with tiktoken. Called only when the encoder is available."""
    enc = _get_encoder()
    if enc is None:
        return 0
    try:
        return len(enc.encode(text))
    except Exception:
        return 0


def _count_tokens_fallback(text: str) -> int:
    """Character heuristic: about 3 characters per token.
    3 is more accurate than 2: a compromise between ru (~2 chars/token) and en (~4 chars/token)."""
    return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# Per-message token cache.
#
# Messages are immutable after creation (LangChain guarantees this for
# AIMessage, HumanMessage, ToolMessage).  The cache key combines the message
# id with a cheap hash of the stringified content + tool_calls, so that even
# if a message is replaced with a new instance bearing the same id (e.g. after
# recovery rewriting), the cache will miss and recompute correctly.
#
# The cache is bounded to avoid unbounded memory growth in very long sessions.
# ---------------------------------------------------------------------------

_MESSAGE_TOKEN_CACHE: dict[tuple[str | None, int], int] = {}
_MESSAGE_TOKEN_CACHE_LIMIT = 512


def _message_cache_key(message: BaseMessage) -> tuple[str | None, int]:
    content_str = stringify_content(message.content)
    tool_calls = getattr(message, "tool_calls", None) or []
    tool_calls_str = str(tool_calls) if tool_calls else ""
    combined = content_str + "\x00" + tool_calls_str
    return (getattr(message, "id", None), hash(combined))


def _count_single_message_tokens(message: BaseMessage, *, use_tiktoken: bool) -> int:
    key = _message_cache_key(message)
    cached = _MESSAGE_TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    count_fn = _count_tokens_tiktoken if use_tiktoken else _count_tokens_fallback
    total = 0
    content = stringify_content(message.content)
    total += count_fn(content)

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        total += count_fn(str(tool_calls))

    if use_tiktoken:
        total += _MESSAGE_OVERHEAD_TOKENS

    if len(_MESSAGE_TOKEN_CACHE) >= _MESSAGE_TOKEN_CACHE_LIMIT:
        # Evict oldest entry (dict preserves insertion order in Python 3.7+).
        _MESSAGE_TOKEN_CACHE.pop(next(iter(_MESSAGE_TOKEN_CACHE)))
    _MESSAGE_TOKEN_CACHE[key] = total
    return total


def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Estimate the total token count for a list of messages.

    Algorithm:
    - If tiktoken is available, use cl100k_base + per-message overhead.
    - Otherwise, use a character heuristic with a divisor of 3.
    - Per-message results are cached to avoid recomputation for unchanged
      messages across turns.
    """
    use_tiktoken = _get_encoder() is not None

    total = 0
    for message in messages:
        total += _count_single_message_tokens(message, use_tiktoken=use_tiktoken)

    return total


def estimate_summary_tokens(summary: str) -> int:
    summary_text = str(summary or "").strip()
    if not summary_text:
        return 0
    return estimate_tokens([SystemMessage(content=f"<memory>\n{summary_text}\n</memory>")])


def estimate_context_tokens(messages: List[BaseMessage], *, reserved_tokens: int = 0) -> int:
    """Estimate the model context budget used by message history plus fixed runtime overhead.

    The message list does not include system/developer prompts, tool schemas, and provider
    wrapper fields. The reserve keeps auto-summary progress closer to provider-reported
    prompt/input tokens without pretending to know every provider tokenizer exactly.
    """
    try:
        reserve = max(0, int(reserved_tokens or 0))
    except (TypeError, ValueError):
        reserve = 0
    if not messages:
        return 0
    return estimate_tokens(messages) + reserve


# ---------------------------------------------------------------------------
# The rest is unchanged
# ---------------------------------------------------------------------------

def _soft_summary_margin(threshold: int, *, has_summary: bool) -> int:
    base = max(800, int(threshold * 0.15))
    if has_summary:
        return max(base, int(threshold * 0.35))
    return base


def should_summarize(
    messages: List[BaseMessage],
    *,
    threshold: int,
    keep_last: int,
    has_summary: bool = False,
    reserved_tokens: int = 0,
) -> bool:
    threshold = int(threshold or 0)
    if threshold <= 0:
        return False

    estimated = estimate_context_tokens(messages, reserved_tokens=reserved_tokens)
    if estimated <= threshold:
        return False

    boundary = choose_summary_boundary(messages, keep_last=keep_last)
    summarizable = messages[:boundary]
    if not summarizable:
        return False

    summarizable_human_turns = sum(1 for message in summarizable if isinstance(message, HumanMessage))
    soft_threshold = threshold + _soft_summary_margin(threshold, has_summary=has_summary)
    min_summarizable_messages = max(6, int(keep_last or 0) + 2)

    if estimated < soft_threshold:
        if len(summarizable) < min_summarizable_messages:
            return False
        if summarizable_human_turns < 2:
            return False

    return True


def choose_summary_boundary(messages: List[BaseMessage], *, keep_last: int) -> int:
    idx = max(0, len(messages) - int(keep_last or 0))
    for scan_idx in range(idx, len(messages)):
        if isinstance(messages[scan_idx], HumanMessage):
            return scan_idx
    return idx


def _compact_for_summary(text: str, *, limit: int = 500) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _format_tool_calls_for_summary(message: BaseMessage) -> str:
    if not isinstance(message, (AIMessage, AIMessageChunk)):
        return ""
    raw_tool_calls = list(getattr(message, "tool_calls", []) or [])
    if not raw_tool_calls:
        return ""

    parts: List[str] = []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_name = str(tool_call.get("name") or "tool").strip() or "tool"
        tool_args = tool_call.get("args")
        try:
            rendered_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
        except TypeError:
            rendered_args = str(tool_args)
        parts.append(f"{tool_name}({rendered_args})")
    return _compact_for_summary("; ".join(parts), limit=320)


def format_history_for_summary(
    messages: List[BaseMessage],
    *,
    is_internal_retry: IsInternalRetry,
) -> str:
    parts: List[str] = []
    for message in messages:
        if isinstance(message, HumanMessage) and is_internal_retry(message):
            continue
        rendered = _compact_for_summary(stringify_content(message.content), limit=500)
        tool_call_text = _format_tool_calls_for_summary(message)
        if isinstance(message, ToolMessage):
            tool_name = str(getattr(message, "name", "") or "tool").strip() or "tool"
            header = f"{message.type}({tool_name})"
        else:
            header = message.type

        segments: List[str] = []
        if tool_call_text:
            segments.append(f"tool_calls={tool_call_text}")
        if rendered:
            segments.append(f"content={rendered}")
        if not segments:
            segments.append("<empty>")
        parts.append(f"{header}: {' | '.join(segments)}")
    return "\n".join(parts)
