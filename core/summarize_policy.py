from __future__ import annotations

import json
import logging
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage

from core.message_utils import stringify_content


IsInternalRetry = Callable[[BaseMessage], bool]

logger = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Tiktoken — ленивая инициализация.
# Энкодер создаётся один раз и переиспользуется на протяжении жизни процесса.
# cl100k_base — кодировка GPT-4/GPT-3.5. Для Gemini не идеальна, но даёт
# точность ±15%, что кратно лучше символьной эвристики.
# ---------------------------------------------------------------------------

_TIKTOKEN_ENCODER = None
_TIKTOKEN_AVAILABLE: Optional[bool] = None  # None = не проверялось


def _get_encoder():
    global _TIKTOKEN_ENCODER, _TIKTOKEN_AVAILABLE
    if _TIKTOKEN_AVAILABLE is True:
        return _TIKTOKEN_ENCODER
    if _TIKTOKEN_AVAILABLE is False:
        return None
    # Первый вызов — пробуем инициализировать
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


# Overhead на каждое сообщение в Chat Completions API:
# role-токен + разделители = ~4 токена (по спецификации OpenAI tiktoken).
_MESSAGE_OVERHEAD_TOKENS = 4


def _count_tokens_tiktoken(text: str) -> int:
    """Подсчитать токены через tiktoken. Вызывается только когда энкодер доступен."""
    enc = _get_encoder()
    if enc is None:
        return 0
    try:
        return len(enc.encode(text))
    except Exception:
        return 0


def _count_tokens_fallback(text: str) -> int:
    """Символьная эвристика: ~3 символа на токен.
    3 точнее чем 2: компромисс между ru (~2 sym/tok) и en (~4 sym/tok)."""
    return max(1, len(text) // 3)


def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Оценить суммарное количество токенов для списка сообщений.

    Алгоритм:
    - Если tiktoken доступен — используем cl100k_base + overhead per message.
    - Иначе — символьная эвристика с делителем 3.
    """
    use_tiktoken = _get_encoder() is not None
    count_fn = _count_tokens_tiktoken if use_tiktoken else _count_tokens_fallback

    total = 0
    for message in messages:
        content = stringify_content(message.content)
        total += count_fn(content)

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            tool_calls_text = str(tool_calls)
            total += count_fn(tool_calls_text)

        if use_tiktoken:
            total += _MESSAGE_OVERHEAD_TOKENS

    return total


# ---------------------------------------------------------------------------
# Остальное не изменилось
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
) -> bool:
    threshold = int(threshold or 0)
    if threshold <= 0:
        return False

    estimated = estimate_tokens(messages)
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
