import ast
import asyncio
import json
import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config import AgentConfig
from core.errors import ErrorType, format_error
from core.safety_policy import SafetyPolicy
from core.utils import truncate_output

logger = logging.getLogger(__name__)

try:
    from tavily import AsyncTavilyClient
except ImportError:
    AsyncTavilyClient = None
    logger.warning("Tavily SDK not installed. Search tools will be disabled.")

try:
    from tavily import errors as tavily_errors
except ImportError:
    tavily_errors = None

_TAVILY_MAX_RESULTS = 20
_TAVILY_MAX_EXTRACT_URLS = 20
_MAX_BATCH_QUERIES = 5
_ALLOWED_SEARCH_DEPTHS = {"basic", "advanced", "fast", "ultra-fast"}
_ALLOWED_TOPICS = {"general", "news", "finance"}
_ALLOWED_FORMATS = {"markdown", "text"}
_CONTENT_FORMAT_ALIASES = {
    "md": "markdown",
    "plain": "text",
    "plaintext": "text",
    # Tavily extract endpoint does not support raw json payload mode.
    # Keep agent resilient by mapping common json request to text extraction.
    "json": "text",
}
_SEARCH_DEPTH_ALIASES = {
    "deep": "advanced",
    "deep-search": "advanced",
    "shallow": "basic",
}
_TOPIC_ALIASES = {
    "model specs": "general",
    "specs": "general",
    "documentation": "general",
    "docs": "general",
    "models": "general",
}


class SearchRuntime:
    def __init__(self):
        self.client: Optional[Any] = None
        self.client_initialized = False
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.safety_policy: Optional[SafetyPolicy] = None
        self.config: Optional[AgentConfig] = None
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.max_cache_size = 50

    def get_config(self) -> AgentConfig:
        return self.config or AgentConfig()

    def get_semaphore(self) -> asyncio.Semaphore:
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(5)
        return self.semaphore

    def normalize_for_cache(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self.normalize_for_cache(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple, set)):
            return [self.normalize_for_cache(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def cache_key(self, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        payload = {
            "func": func_name,
            "args": self.normalize_for_cache(args),
            "kwargs": self.normalize_for_cache(kwargs),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def cleanup_cache(self, ttl: int) -> None:
        if not self.cache:
            return

        now = time.monotonic()
        expired = [key for key, (_, ts) in self.cache.items() if now - ts >= ttl]
        for key in expired:
            self.cache.pop(key, None)

        if len(self.cache) < self.max_cache_size:
            return

        items_by_age = sorted(self.cache.items(), key=lambda item: item[1][1])
        target_size = int(self.max_cache_size * 0.85)
        remove_count = max(1, len(self.cache) - target_size)
        for key, _ in items_by_age[:remove_count]:
            self.cache.pop(key, None)

    def with_cache(self, ttl: int):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                key = self.cache_key(func.__name__, args, kwargs)
                now = time.monotonic()
                cached = self.cache.get(key)
                if cached:
                    result, ts = cached
                    if now - ts < ttl:
                        logger.debug("Cache hit for %s", func.__name__)
                        return result
                    self.cache.pop(key, None)

                result = await func(*args, **kwargs)
                if isinstance(result, str) and not result.lower().startswith(("error:", "ошибка:", "error[")):
                    if len(self.cache) >= self.max_cache_size:
                        self.cleanup_cache(ttl)
                    self.cache[key] = (result, now)
                return result

            return wrapper

        return decorator

    def get_client(self) -> Optional[Any]:
        if self.client_initialized:
            return self.client

        self.client_initialized = True
        try:
            config = self.get_config()
        except Exception as e:
            logger.error("Config load failed: %s", e)
            return None

        if not config.enable_search_tools or AsyncTavilyClient is None:
            return None
        if not config.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not set. Web search tools will return errors.")
            return None

        try:
            self.client = AsyncTavilyClient(api_key=config.tavily_api_key.get_secret_value())
            return self.client
        except Exception as e:
            logger.error("Failed to initialize Tavily client: %s", e)
            return None

    async def execute_with_retry(self, coroutine_func, *args, **kwargs):
        try:
            config = self.get_config()
            max_retries = _clamp_int(config.max_retries, 1, 8, 3)
            retry_delay = max(0.1, float(config.retry_delay))
        except Exception:
            max_retries = 3
            retry_delay = 2.0

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                async with self.get_semaphore():
                    return await coroutine_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(retry_delay * (2 ** attempt), 10.0))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unknown retry failure")


_RUNTIME = SearchRuntime()


def set_safety_policy(policy: SafetyPolicy):
    _RUNTIME.safety_policy = policy


def set_runtime_config(config: AgentConfig):
    _RUNTIME.config = config


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _normalize_query(query: str) -> str:
    return " ".join((query or "").strip().split())


def _normalize_search_depth(search_depth: str) -> str:
    value = (search_depth or "basic").strip().lower()
    return _SEARCH_DEPTH_ALIASES.get(value, value)


def _normalize_topic(topic: str) -> str:
    value = (topic or "general").strip().lower()
    return _TOPIC_ALIASES.get(value, value)


def _is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class FetchContentInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    urls: List[str] = Field(...)
    advanced: bool = False
    content_format: str = "markdown"
    query: Optional[str] = None
    chunks_per_source: int = 3

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_urls = data.get("urls")
        if isinstance(raw_urls, str):
            data["urls"] = [raw_urls]
        elif isinstance(raw_urls, tuple):
            data["urls"] = list(raw_urls)
        return data


def _format_tavily_error(exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()

    if tavily_errors is not None:
        if isinstance(exc, getattr(tavily_errors, "MissingAPIKeyError", tuple())):
            return format_error(ErrorType.CONFIG, "TAVILY_API_KEY is missing.")
        if isinstance(exc, getattr(tavily_errors, "InvalidAPIKeyError", tuple())):
            return format_error(ErrorType.ACCESS_DENIED, "Invalid Tavily API key.")
        if isinstance(exc, getattr(tavily_errors, "UsageLimitExceededError", tuple())):
            return format_error(ErrorType.LIMIT_EXCEEDED, "Tavily usage limit exceeded.")
        if isinstance(exc, getattr(tavily_errors, "BadRequestError", tuple())):
            return format_error(ErrorType.VALIDATION, f"Bad Tavily request: {msg}")
        if isinstance(exc, getattr(tavily_errors, "ForbiddenError", tuple())):
            return format_error(ErrorType.ACCESS_DENIED, f"Tavily access forbidden: {msg}")
        if isinstance(exc, getattr(tavily_errors, "TimeoutError", tuple())):
            return format_error(ErrorType.TIMEOUT, f"Tavily request timed out: {msg}")

    if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
        return format_error(ErrorType.ACCESS_DENIED, "Invalid API credentials (401 Unauthorized).")
    if "timeout" in lower:
        return format_error(ErrorType.TIMEOUT, f"Request timed out: {msg}")
    if "limit" in lower or "quota" in lower:
        return format_error(ErrorType.LIMIT_EXCEEDED, f"Usage limit reached: {msg}")
    return format_error(ErrorType.NETWORK, f"Tavily request failed: {msg}")


@_RUNTIME.with_cache(ttl=600)
async def _web_search_impl(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
) -> str:
    client = _RUNTIME.get_client()
    if not client:
        return format_error(ErrorType.CONFIG, "Search is unavailable due to missing configuration (TAVILY_API_KEY or SDK).")

    query = _normalize_query(query)
    if not query:
        return format_error(ErrorType.VALIDATION, "Empty query provided.")

    max_results = _clamp_int(max_results, 1, _TAVILY_MAX_RESULTS, 5)
    search_depth = _normalize_search_depth(search_depth)
    topic = _normalize_topic(topic)
    if search_depth not in _ALLOWED_SEARCH_DEPTHS:
        return format_error(ErrorType.VALIDATION, f"Invalid search_depth '{search_depth}'. Allowed: {sorted(_ALLOWED_SEARCH_DEPTHS)}")
    if topic not in _ALLOWED_TOPICS:
        return format_error(ErrorType.VALIDATION, f"Invalid topic '{topic}'. Allowed: {sorted(_ALLOWED_TOPICS)}")

    try:
        response = await _RUNTIME.execute_with_retry(
            client.search,
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic=topic,
            include_answer=True,
            include_usage=True,
        )
    except Exception as e:
        return _format_tavily_error(e)

    parts: List[str] = []
    answer = response.get("answer")
    if answer:
        parts.append(f"AI Overview:\n{answer}\n{'=' * 40}")

    items = response.get("results", [])
    if not items:
        return "\n".join(parts) if parts else format_error(ErrorType.NOT_FOUND, "No results found.")

    max_chars = _RUNTIME.safety_policy.max_search_chars if _RUNTIME.safety_policy else 10000
    total_chars = 0
    for item in items:
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        score = item.get("score", 0)
        content = item.get("content") or ""
        if not content:
            continue

        block = f"Source: {title} (score: {float(score):.2f})\nURL: {url}\n{content}"
        if total_chars + len(block) > max_chars:
            break
        parts.append(block)
        parts.append("-" * 40)
        total_chars += len(block) + 40

    usage = response.get("usage")
    if usage:
        parts.append(f"Usage: {usage}")
    parts.append("[Search completed. Use the context above.]")
    return "\n".join(parts)


@tool("web_search")
async def web_search(query: str, max_results: int = 5, search_depth: str = "basic", topic: str = "general") -> str:
    """Search web snippets with an AI summary. search_depth: basic/advanced/fast/ultra-fast; topic: general/news/finance."""
    return await _web_search_impl(query, max_results, search_depth, topic)


@tool("fetch_content", args_schema=FetchContentInput)
@_RUNTIME.with_cache(ttl=1800)
async def fetch_content(
    urls: List[str],
    advanced: bool = False,
    content_format: str = "markdown",
    query: Optional[str] = None,
    chunks_per_source: int = 3,
) -> str:
    """Extract text from one or multiple URLs."""
    client = _RUNTIME.get_client()
    if not client:
        return format_error(ErrorType.CONFIG, "Fetch unavailable due to missing configuration.")

    clean_urls = _parse_urls_input(urls)
    if not clean_urls:
        return format_error(ErrorType.VALIDATION, "No valid HTTP/HTTPS URLs provided.")

    content_format = (content_format or "markdown").strip().lower()
    content_format = _CONTENT_FORMAT_ALIASES.get(content_format, content_format)
    if content_format not in _ALLOWED_FORMATS:
        return format_error(ErrorType.VALIDATION, f"Invalid content_format '{content_format}'. Allowed: {sorted(_ALLOWED_FORMATS)}")

    request_kwargs: Dict[str, Any] = {
        "urls": clean_urls,
        "extract_depth": "advanced" if advanced else "basic",
        "format": content_format,
    }
    if query:
        request_kwargs["query"] = _normalize_query(query)
        request_kwargs["chunks_per_source"] = _clamp_int(chunks_per_source, 1, 20, 3)

    try:
        response = await _RUNTIME.execute_with_retry(client.extract, **request_kwargs)
    except Exception as e:
        return _format_tavily_error(e)

    output_parts: List[str] = []
    max_chars_limit = _RUNTIME.safety_policy.max_search_chars if _RUNTIME.safety_policy else 15000
    max_chars_per_url = max(1000, int(max_chars_limit / max(1, len(clean_urls))))
    for item in response.get("results", []):
        url = item.get("url", "Unknown")
        content = item.get("raw_content") or item.get("content") or ""
        output_parts.append(
            f"=== SOURCE: {url} ===\n{truncate_output(content, max_chars_per_url, source=url) or '[Empty]'}\n{'=' * 30}"
        )

    for failed_item in response.get("failed_results", []):
        output_parts.append(
            f"FAILED: {failed_item.get('url', 'unknown')} - {failed_item.get('error', 'Unknown error')}"
        )

    usage = response.get("usage")
    if usage:
        output_parts.append(f"Usage: {usage}")
    return "\n".join(output_parts) or format_error(ErrorType.EXECUTION, "No content extracted.")


@tool("batch_web_search")
@_RUNTIME.with_cache(ttl=600)
async def batch_web_search(
    queries: List[str],
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
) -> str:
    """Run up to 5 web searches in parallel. search_depth: basic/advanced/fast/ultra-fast; topic: general/news/finance."""
    if not queries:
        return format_error(ErrorType.VALIDATION, "No queries provided.")

    normalized_queries: List[str] = []
    seen = set()
    for query in queries:
        clean_query = _normalize_query(str(query))
        if clean_query and clean_query not in seen:
            seen.add(clean_query)
            normalized_queries.append(clean_query)

    if not normalized_queries:
        return format_error(ErrorType.VALIDATION, "No valid queries provided.")

    selected = normalized_queries[:_MAX_BATCH_QUERIES]
    output: List[str] = []
    if len(normalized_queries) > _MAX_BATCH_QUERIES:
        output.append(
            f"WARNING: Only first {_MAX_BATCH_QUERIES} unique queries were executed to prevent API limits.\n{'=' * 40}"
        )

    results = await asyncio.gather(
        *(_web_search_impl(query, max_results=max_results, search_depth=search_depth, topic=topic) for query in selected),
        return_exceptions=True,
    )
    for query, result in zip(selected, results):
        rendered = (
            format_error(ErrorType.EXECUTION, f"Unhandled batch search error: {result}")
            if isinstance(result, Exception)
            else str(result)
        )
        output.append(f"Query: {query}\n{rendered}\n{'=' * 50}")
    return "\n".join(output)



