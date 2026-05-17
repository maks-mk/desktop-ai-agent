import asyncio
import base64
import importlib
import logging
from typing import Any, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.graph import END, START, StateGraph

from core.api_key_rotation import RotatingChatModel
from core.checkpointing import create_checkpoint_runtime
from core.config import AgentConfig
from core.logging_config import setup_logging
from core.model_profiles import ModelProfileStore, find_active_profile, find_profile_by_id
from core.multimodal import extract_model_capabilities, resolve_model_capabilities
from core.nodes import AgentNodes
from core.provider_registry import ProviderRegistry, build_reasoning_kwargs
from core.reasoning_debug import debug_event, elapsed_since, log_unknown_fields, now, preview_value
from core.run_logger import JsonlRunLogger
from core.state import AgentState
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("agent")
reasoning_logger = logging.getLogger("agent.reasoning_debug")

_GOOGLE_RETRY_CONTROL_KWARGS = frozenset(
    {
        "max_retries",
        "wait_exponential_multiplier",
        "wait_exponential_min",
        "wait_exponential_max",
    }
)
_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"


# --- Factories ---


def _normalized_model_name(model_name: str | None) -> str:
    return str(model_name or "").strip().lower()


def _normalized_gemini_model_name(model_name: str | None) -> str:
    normalized = _normalized_model_name(model_name)
    if normalized.startswith("models/"):
        normalized = normalized[len("models/") :]
    return normalized


def _gemini_model_supports_thinking_budget(model_name: str | None) -> bool:
    normalized = _normalized_gemini_model_name(model_name)
    return normalized.startswith("gemini-2.5") or normalized in {
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
    }


def _gemini_model_supports_thinking_level(model_name: str | None) -> bool:
    return _normalized_gemini_model_name(model_name).startswith("gemini-3")


def _chat_model_accepts_kwarg(model_cls: type, name: str) -> bool:
    fields = getattr(model_cls, "model_fields", None)
    if isinstance(fields, dict):
        return name in fields
    return True


def _openai_model_supports_reasoning_controls(model_name: str | None) -> bool:
    normalized = _normalized_model_name(model_name)
    return normalized.startswith(("gpt-5", "o1", "o3", "o4")) or any(
        marker in normalized
        for marker in (
            "reason",
            "thinking",
            "deepseek-r1",
            "gpt-oss",
            "gemma",
            "qwen3",
            "qwq",
            "grok-4",
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-medium-3-5",
        )
    )


def _normalized_reasoning_effort(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        return normalized
    return "medium"


def _normalized_gemini_thinking_level(value: str | None) -> str:
    normalized = _normalized_reasoning_effort(value)
    if normalized in {"none", "minimal"}:
        return "minimal"
    if normalized == "xhigh":
        return "high"
    return normalized


def _encode_gemini_thought_signature(signature: Any) -> str:
    if isinstance(signature, str):
        return signature
    if isinstance(signature, memoryview):
        signature = signature.tobytes()
    if isinstance(signature, bytearray):
        signature = bytes(signature)
    if isinstance(signature, bytes):
        return base64.b64encode(signature).decode("ascii")
    return ""


def _decode_gemini_thought_signature(signature: Any) -> bytes:
    if isinstance(signature, bytes):
        return signature
    if isinstance(signature, bytearray):
        return bytes(signature)
    if isinstance(signature, memoryview):
        return signature.tobytes()
    if not isinstance(signature, str) or not signature:
        return b""
    try:
        return base64.b64decode(signature, validate=True)
    except Exception:
        return signature.encode("utf-8")


def _attach_gemini_thought_signatures_to_result(response: Any, result: ChatResult) -> ChatResult:
    if not result.generations:
        return result
    message = result.generations[0].message
    if not isinstance(message, AIMessage) or not message.tool_calls:
        return result

    candidates = list(getattr(response, "candidates", []) or [])
    if not candidates:
        return result
    content = getattr(candidates[0], "content", None)
    raw_parts = list(getattr(content, "parts", []) or [])
    raw_signatures: list[str] = []
    for part in raw_parts:
        function_call = getattr(part, "function_call", None)
        if not str(getattr(function_call, "name", "") or "").strip():
            continue
        encoded_signature = _encode_gemini_thought_signature(getattr(part, "thought_signature", b""))
        raw_signatures.append(encoded_signature)

    signature_map = {
        str(tool_call.get("id") or "").strip(): signature
        for tool_call, signature in zip(message.tool_calls, raw_signatures)
        if str(tool_call.get("id") or "").strip() and signature
    }
    if not signature_map:
        return result

    metadata = dict(getattr(message, "additional_kwargs", {}) or {})
    existing = metadata.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(signature_map)
    metadata[_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY] = merged
    result.generations[0].message = message.model_copy(update={"additional_kwargs": metadata})
    return result


def _inject_gemini_thought_signatures_into_request(request: Any, messages: list[BaseMessage]) -> None:
    signature_bytes: list[bytes] = []
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        metadata = dict(getattr(message, "additional_kwargs", {}) or {})
        signature_map = metadata.get(_GEMINI_FUNCTION_CALL_THOUGHT_SIGNATURES_KEY)
        if not isinstance(signature_map, dict):
            continue
        for tool_call in message.tool_calls:
            tool_call_id = str(tool_call.get("id") or "").strip()
            signature = _decode_gemini_thought_signature(signature_map.get(tool_call_id))
            if signature:
                signature_bytes.append(signature)

    if not signature_bytes:
        return

    next_signature = iter(signature_bytes)
    for content in list(getattr(request, "contents", []) or []):
        if str(getattr(content, "role", "") or "") != "model":
            continue
        for part in list(getattr(content, "parts", []) or []):
            function_call = getattr(part, "function_call", None)
            if not str(getattr(function_call, "name", "") or "").strip():
                continue
            try:
                part.thought_signature = next(next_signature)
            except StopIteration:
                return


def _gemini_finish_reason(response: Any) -> str:
    candidates = list(getattr(response, "candidates", []) or [])
    if not candidates:
        return ""
    return str(getattr(candidates[0], "finish_reason", "") or "")


def _extract_openai_reasoning_delta(chunk: Any) -> Any:
    if not isinstance(chunk, dict) and hasattr(chunk, "model_dump"):
        try:
            chunk = chunk.model_dump()
        except Exception:
            return None
    if not isinstance(chunk, dict):
        return None

    candidates: list[Any] = []
    for choice in chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        candidates.extend(
            delta.get(key)
            for key in ("reasoning", "reasoning_content", "thinking", "thinking_content", "analysis", "analysis_content")
            if delta.get(key) not in (None, "")
        )
    for value in candidates:
        if value not in (None, ""):
            return value
    return None


def _build_gemini_thought_signature_adapter(base_cls: type, chat_models_module: Any) -> type:
    if not all(
        callable(getattr(chat_models_module, name, None))
        for name in ("_chat_with_retry", "_achat_with_retry", "_response_to_result")
    ):
        return base_cls

    class GeminiThoughtSignatureChatModel(base_cls):
        def _prepare_request(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            request = super()._prepare_request(messages, *args, **kwargs)
            _inject_gemini_thought_signatures_into_request(request, messages)
            debug_event(
                "final_payload",
                provider="gemini",
                model=getattr(self, "model", None),
                payload_preview=preview_value(request),
            )
            log_unknown_fields("gemini_final_payload", request)
            return request

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager=None,
            *,
            tools=None,
            functions=None,
            safety_settings=None,
            tool_config=None,
            generation_config=None,
            cached_content=None,
            tool_choice=None,
            **kwargs: Any,
        ) -> ChatResult:
            request = self._prepare_request(
                messages,
                stop=stop,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content or self.cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )
            if self.timeout is not None and "timeout" not in kwargs:
                kwargs["timeout"] = self.timeout
            if "max_retries" not in kwargs:
                kwargs["max_retries"] = self.max_retries
            response = chat_models_module._chat_with_retry(
                request=request,
                **kwargs,
                generation_method=self.client.generate_content,
                metadata=self.default_metadata,
            )
            debug_event(
                "final_response_object",
                provider="gemini",
                model=getattr(self, "model", None),
                response_preview=preview_value(response),
                finish_reason=_gemini_finish_reason(response),
            )
            log_unknown_fields("gemini_final_response", response)
            result = chat_models_module._response_to_result(response)
            return _attach_gemini_thought_signatures_to_result(response, result)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager=None,
            *,
            tools=None,
            functions=None,
            safety_settings=None,
            tool_config=None,
            generation_config=None,
            cached_content=None,
            tool_choice=None,
            **kwargs: Any,
        ) -> ChatResult:
            if not self.async_client:
                updated_kwargs = {
                    **kwargs,
                    "tools": tools,
                    "functions": functions,
                    "safety_settings": safety_settings,
                    "tool_config": tool_config,
                    "generation_config": generation_config,
                }
                return await super()._agenerate(messages, stop, run_manager, **updated_kwargs)

            request = self._prepare_request(
                messages,
                stop=stop,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content or self.cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )
            if self.timeout is not None and "timeout" not in kwargs:
                kwargs["timeout"] = self.timeout
            if "max_retries" not in kwargs:
                kwargs["max_retries"] = self.max_retries
            response = await chat_models_module._achat_with_retry(
                request=request,
                **kwargs,
                generation_method=self.async_client.generate_content,
                metadata=self.default_metadata,
            )
            debug_event(
                "final_response_object",
                provider="gemini",
                model=getattr(self, "model", None),
                response_preview=preview_value(response),
                finish_reason=_gemini_finish_reason(response),
            )
            log_unknown_fields("gemini_final_response", response)
            result = chat_models_module._response_to_result(response)
            return _attach_gemini_thought_signatures_to_result(response, result)

    return GeminiThoughtSignatureChatModel


def _patch_langchain_google_genai_retry_kwargs() -> None:
    try:
        chat_models = importlib.import_module("langchain_google_genai.chat_models")
    except ImportError:
        return

    if bool(getattr(chat_models, "_agent_retry_patch_applied", False)):
        return

    original_chat_with_retry = getattr(chat_models, "_chat_with_retry", None)
    original_achat_with_retry = getattr(chat_models, "_achat_with_retry", None)
    if not callable(original_chat_with_retry) or not callable(original_achat_with_retry):
        return

    def _strip_retry_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in dict(call_kwargs).items()
            if key not in _GOOGLE_RETRY_CONTROL_KWARGS
        }

    def _patched_chat_with_retry(generation_method, **kwargs):
        def _wrapped_generation_method(**call_kwargs):
            return generation_method(**_strip_retry_kwargs(call_kwargs))

        return original_chat_with_retry(generation_method=_wrapped_generation_method, **kwargs)

    async def _patched_achat_with_retry(generation_method, **kwargs):
        async def _wrapped_generation_method(**call_kwargs):
            return await generation_method(**_strip_retry_kwargs(call_kwargs))

        return await original_achat_with_retry(generation_method=_wrapped_generation_method, **kwargs)

    chat_models._chat_with_retry = _patched_chat_with_retry
    chat_models._achat_with_retry = _patched_achat_with_retry
    chat_models._agent_retry_patch_applied = True


def prepare_llm_with_tools(
    llm: BaseChatModel,
    tools: list[Any],
) -> tuple[BaseChatModel, bool, str]:
    """Bind tools once and report whether structured tool calling is actually available."""
    if not tools:
        return llm, False, ""

    binder = getattr(llm, "bind_tools", None)
    if not callable(binder):
        return llm, False, "LLM backend does not implement bind_tools()."

    try:
        return binder(tools), True, ""
    except Exception as exc:
        return llm, False, str(exc)


def create_llm(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    """Initializes LLM based on configuration."""
    if config.provider == "gemini":
        _patch_langchain_google_genai_retry_kwargs()
        # Lazy import to avoid loading both providers on startup.
        from langchain_google_genai import ChatGoogleGenerativeAI

        try:
            chat_models_module = importlib.import_module("langchain_google_genai.chat_models")
        except ImportError:
            chat_models_module = None
        GeminiChatModel = _build_gemini_thought_signature_adapter(ChatGoogleGenerativeAI, chat_models_module)

        # Безопасное извлечение ключа (защита от краша, если ключ None)
        if api_key_override is None:
            api_key = config.gemini_api_key.get_secret_value() if config.gemini_api_key else None
        else:
            api_key = str(api_key_override or "")
        gemini_kwargs: dict[str, Any] = {
            "model": config.gemini_model,
            "temperature": config.temperature,
            "google_api_key": api_key,
        }
        gemini_thinking_enabled = False
        gemini_reasoning_mode = "disabled"
        if bool(getattr(config, "enable_model_reasoning", True)):
            if _gemini_model_supports_thinking_budget(config.gemini_model) and _chat_model_accepts_kwarg(
                GeminiChatModel,
                "thinking_budget",
            ):
                gemini_kwargs["thinking_budget"] = int(getattr(config, "gemini_thinking_budget", 4096))
                gemini_thinking_enabled = True
                gemini_reasoning_mode = "thinking_budget"
            elif _gemini_model_supports_thinking_level(config.gemini_model) and _chat_model_accepts_kwarg(
                GeminiChatModel,
                "thinking_level",
            ):
                gemini_kwargs["thinking_level"] = _normalized_gemini_thinking_level(
                    getattr(config, "model_reasoning_effort", "medium")
                )
                gemini_thinking_enabled = True
                gemini_reasoning_mode = "thinking_level"
            if gemini_thinking_enabled and _chat_model_accepts_kwarg(GeminiChatModel, "include_thoughts"):
                gemini_kwargs["include_thoughts"] = True
        reasoning_logger.debug(
            "gemini reasoning config model=%s reasoning_enabled=%s mode=%s include_thoughts=%s thinking_budget=%s thinking_level=%s",
            config.gemini_model,
            bool(getattr(config, "enable_model_reasoning", True)),
            gemini_reasoning_mode,
            bool(gemini_kwargs.get("include_thoughts")),
            gemini_kwargs.get("thinking_budget"),
            gemini_kwargs.get("thinking_level"),
        )
        debug_event(
            "reasoning_request",
            provider="gemini",
            base_url=None,
            model=config.gemini_model,
            reasoning_enabled=bool(getattr(config, "enable_model_reasoning", True)),
            reasoning_effort=getattr(config, "model_reasoning_effort", "medium"),
            mode=gemini_reasoning_mode,
        )
        return GeminiChatModel(
            **gemini_kwargs,
        )
    if config.provider == "openai":
        # Lazy import to avoid loading both providers on startup.
        from langchain_openai import ChatOpenAI as BaseChatOpenAI

        class ReasoningDebugChatOpenAI(BaseChatOpenAI):
            def _get_request_payload(self, *args: Any, **kwargs: Any) -> dict:
                start = now()
                payload = super()._get_request_payload(*args, **kwargs)
                debug_event(
                    "final_payload",
                    provider="openai",
                    model=getattr(self, "model_name", None) or getattr(self, "model", None),
                    base_url=str(getattr(self, "openai_api_base", "") or getattr(self, "base_url", "") or ""),
                    payload=payload,
                    elapsed=elapsed_since(start),
                )
                log_unknown_fields("openai_final_payload", payload)
                return payload

            def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
                start = now()
                result = super()._generate(*args, **kwargs)
                debug_event(
                    "final_response_object",
                    provider="openai",
                    model=getattr(self, "model_name", None) or getattr(self, "model", None),
                    response_preview=preview_value(result),
                    elapsed=elapsed_since(start),
                )
                log_unknown_fields("openai_final_response", result)
                return result

            def _log_raw_provider_chunk(self, source: str, chunk: Any) -> None:
                fields_source = chunk
                if not isinstance(fields_source, dict) and hasattr(fields_source, "model_dump"):
                    try:
                        fields_source = fields_source.model_dump()
                    except Exception:
                        fields_source = chunk
                debug_event(
                    "raw_stream_chunk",
                    source=source,
                    provider="openai",
                    model=getattr(self, "model_name", None) or getattr(self, "model", None),
                    chunk_preview=preview_value(chunk),
                )
                log_unknown_fields(source, fields_source)

            def _attach_raw_reasoning_delta(self, generation_chunk: Any, chunk: Any) -> None:
                reasoning_delta = _extract_openai_reasoning_delta(chunk)
                if reasoning_delta in (None, ""):
                    return
                message = getattr(generation_chunk, "message", None)
                if not isinstance(message, AIMessage):
                    return
                existing = dict(getattr(message, "additional_kwargs", {}) or {})
                existing["reasoning_content"] = reasoning_delta
                message.additional_kwargs = existing

            def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any):
                from langchain_core.messages import AIMessageChunk, BaseMessageChunk
                from langchain_openai.chat_models.base import (
                    _convert_responses_chunk_to_generation_chunk,
                    _handle_openai_api_error,
                    _handle_openai_bad_request,
                )
                import openai
                import warnings

                self._ensure_sync_client_available()
                kwargs["stream"] = True

                if self._use_responses_api({**kwargs, **self.model_kwargs}):
                    payload = self._get_request_payload(messages, stop=stop, **kwargs)
                    try:
                        if self.include_response_headers:
                            raw_context_manager = self.root_client.with_raw_response.responses.create(**payload)
                            context_manager = raw_context_manager.parse()
                            headers = {"headers": dict(raw_context_manager.headers)}
                        else:
                            context_manager = self.root_client.responses.create(**payload)
                            headers = {}
                        original_schema_obj = kwargs.get("response_format")

                        with context_manager as response:
                            is_first_chunk = True
                            current_index = -1
                            current_output_index = -1
                            current_sub_index = -1
                            has_reasoning = False
                            for chunk in response:
                                self._log_raw_provider_chunk("openai_responses_stream", chunk)
                                metadata = headers if is_first_chunk else {}
                                (
                                    current_index,
                                    current_output_index,
                                    current_sub_index,
                                    generation_chunk,
                                ) = _convert_responses_chunk_to_generation_chunk(
                                    chunk,
                                    current_index,
                                    current_output_index,
                                    current_sub_index,
                                    schema=original_schema_obj,
                                    metadata=metadata,
                                    has_reasoning=has_reasoning,
                                    output_version=self.output_version,
                                )
                                if generation_chunk:
                                    if run_manager:
                                        run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                                    is_first_chunk = False
                                    if "reasoning" in generation_chunk.message.additional_kwargs:
                                        has_reasoning = True
                                    yield generation_chunk
                    except openai.BadRequestError as exc:
                        _handle_openai_bad_request(exc)
                    except openai.APIError as exc:
                        _handle_openai_api_error(exc)
                    return

                from langchain_openai.chat_models.base import _handle_openai_api_error, _handle_openai_bad_request

                stream_usage = self._should_stream_usage(kwargs.pop("stream_usage", None), **kwargs)
                if stream_usage:
                    kwargs["stream_options"] = {"include_usage": stream_usage}
                payload = self._get_request_payload(messages, stop=stop, **kwargs)
                default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
                base_generation_info = {}
                response = None

                try:
                    if "response_format" in payload:
                        if self.include_response_headers:
                            warnings.warn(
                                "Cannot currently include response headers when response_format is specified."
                            )
                        payload.pop("stream")
                        context_manager = self.root_client.beta.chat.completions.stream(**payload)
                    else:
                        if self.include_response_headers:
                            raw_response = self.client.with_raw_response.create(**payload)
                            response = raw_response.parse()
                            base_generation_info = {"headers": dict(raw_response.headers)}
                        else:
                            response = self.client.create(**payload)
                        context_manager = response
                    with context_manager as response:
                        is_first_chunk = True
                        for chunk in response:
                            self._log_raw_provider_chunk("openai_chat_completions_stream", chunk)
                            if not isinstance(chunk, dict):
                                chunk = chunk.model_dump()
                            generation_chunk = self._convert_chunk_to_generation_chunk(
                                chunk,
                                default_chunk_class,
                                base_generation_info if is_first_chunk else {},
                            )
                            if generation_chunk is None:
                                continue
                            self._attach_raw_reasoning_delta(generation_chunk, chunk)
                            default_chunk_class = generation_chunk.message.__class__
                            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    generation_chunk.text,
                                    chunk=generation_chunk,
                                    logprobs=logprobs,
                                )
                            is_first_chunk = False
                            yield generation_chunk
                except openai.BadRequestError as exc:
                    _handle_openai_bad_request(exc)
                except openai.APIError as exc:
                    _handle_openai_api_error(exc)
                if response is not None and hasattr(response, "get_final_completion") and "response_format" in payload:
                    final_completion = response.get_final_completion()
                    debug_event(
                        "final_response_object",
                        provider="openai",
                        source="openai_chat_completions_stream_final_completion",
                        response_preview=preview_value(final_completion),
                    )
                    generation_chunk = self._get_generation_chunk_from_completion(final_completion)
                    if run_manager:
                        run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                    yield generation_chunk

            async def _astream(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ):
                from langchain_core.messages import AIMessageChunk, BaseMessageChunk
                from langchain_openai.chat_models.base import (
                    _astream_with_chunk_timeout,
                    _convert_responses_chunk_to_generation_chunk,
                    _handle_openai_api_error,
                    _handle_openai_bad_request,
                )
                import openai
                import warnings

                kwargs["stream"] = True

                if self._use_responses_api({**kwargs, **self.model_kwargs}):
                    payload = self._get_request_payload(messages, stop=stop, **kwargs)
                    try:
                        if self.include_response_headers:
                            raw_context_manager = await self.root_async_client.with_raw_response.responses.create(
                                **payload
                            )
                            context_manager = raw_context_manager.parse()
                            headers = {"headers": dict(raw_context_manager.headers)}
                        else:
                            context_manager = await self.root_async_client.responses.create(**payload)
                            headers = {}
                        original_schema_obj = kwargs.get("response_format")

                        async with context_manager as response:
                            is_first_chunk = True
                            current_index = -1
                            current_output_index = -1
                            current_sub_index = -1
                            has_reasoning = False
                            async for chunk in _astream_with_chunk_timeout(
                                response,
                                self.stream_chunk_timeout,
                                model_name=self.model_name,
                            ):
                                self._log_raw_provider_chunk("openai_responses_stream", chunk)
                                metadata = headers if is_first_chunk else {}
                                (
                                    current_index,
                                    current_output_index,
                                    current_sub_index,
                                    generation_chunk,
                                ) = _convert_responses_chunk_to_generation_chunk(
                                    chunk,
                                    current_index,
                                    current_output_index,
                                    current_sub_index,
                                    schema=original_schema_obj,
                                    metadata=metadata,
                                    has_reasoning=has_reasoning,
                                    output_version=self.output_version,
                                )
                                if generation_chunk:
                                    if run_manager:
                                        await run_manager.on_llm_new_token(
                                            generation_chunk.text,
                                            chunk=generation_chunk,
                                        )
                                    is_first_chunk = False
                                    if "reasoning" in generation_chunk.message.additional_kwargs:
                                        has_reasoning = True
                                    yield generation_chunk
                    except openai.BadRequestError as exc:
                        _handle_openai_bad_request(exc)
                    except openai.APIError as exc:
                        _handle_openai_api_error(exc)
                    return

                stream_usage = self._should_stream_usage(kwargs.pop("stream_usage", None), **kwargs)
                if stream_usage:
                    kwargs["stream_options"] = {"include_usage": stream_usage}
                payload = self._get_request_payload(messages, stop=stop, **kwargs)
                default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
                base_generation_info = {}
                response = None

                try:
                    if "response_format" in payload:
                        if self.include_response_headers:
                            warnings.warn(
                                "Cannot currently include response headers when response_format is specified."
                            )
                        payload.pop("stream")
                        context_manager = self.root_async_client.beta.chat.completions.stream(**payload)
                    else:
                        if self.include_response_headers:
                            raw_response = await self.async_client.with_raw_response.create(**payload)
                            response = raw_response.parse()
                            base_generation_info = {"headers": dict(raw_response.headers)}
                        else:
                            response = await self.async_client.create(**payload)
                        context_manager = response
                    async with context_manager as response:
                        is_first_chunk = True
                        async for chunk in _astream_with_chunk_timeout(
                            response,
                            self.stream_chunk_timeout,
                            model_name=self.model_name,
                        ):
                            self._log_raw_provider_chunk("openai_chat_completions_stream", chunk)
                            if not isinstance(chunk, dict):
                                chunk = chunk.model_dump()
                            generation_chunk = self._convert_chunk_to_generation_chunk(
                                chunk,
                                default_chunk_class,
                                base_generation_info if is_first_chunk else {},
                            )
                            if generation_chunk is None:
                                continue
                            self._attach_raw_reasoning_delta(generation_chunk, chunk)
                            default_chunk_class = generation_chunk.message.__class__
                            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    generation_chunk.text,
                                    chunk=generation_chunk,
                                    logprobs=logprobs,
                                )
                            is_first_chunk = False
                            yield generation_chunk
                except openai.BadRequestError as exc:
                    _handle_openai_bad_request(exc)
                except openai.APIError as exc:
                    _handle_openai_api_error(exc)
                if response is not None and hasattr(response, "get_final_completion") and "response_format" in payload:
                    final_completion = await response.get_final_completion()
                    debug_event(
                        "final_response_object",
                        provider="openai",
                        source="openai_chat_completions_stream_final_completion",
                        response_preview=preview_value(final_completion),
                    )
                    generation_chunk = self._get_generation_chunk_from_completion(final_completion)
                    if run_manager:
                        await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                    yield generation_chunk

        ChatOpenAI = ReasoningDebugChatOpenAI

        if api_key_override is None:
            api_key = config.openai_api_key.get_secret_value() if config.openai_api_key else None
        else:
            api_key = str(api_key_override or "")
        openai_kwargs: dict[str, Any] = {
            "model": config.openai_model,
            "temperature": config.temperature,
            "api_key": api_key,
            "base_url": config.openai_base_url,
            "max_retries": 0,
            "stream_usage": True,
        }
        if bool(getattr(config, "enable_model_reasoning", True)) and _openai_model_supports_reasoning_controls(
            config.openai_model
        ):
            registry = ProviderRegistry.from_path(config.provider_registry_path)
            provider_config = registry.match(config.openai_base_url)
            debug_event(
                "reasoning_request",
                provider=provider_config.get("id") if isinstance(provider_config, dict) else None,
                base_url=config.openai_base_url,
                model=config.openai_model,
                reasoning_enabled=bool(getattr(config, "enable_model_reasoning", True)),
                reasoning_effort=_normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
            )
            reasoning_logger.debug(
                "openai reasoning registry match model=%s base_url=%s provider_id=%s supports_reasoning=%s validation=%s path=%s effort=%s",
                config.openai_model,
                config.openai_base_url,
                provider_config.get("id") if isinstance(provider_config, dict) else None,
                provider_config.get("supports_reasoning") if isinstance(provider_config, dict) else None,
                provider_config.get("validation") if isinstance(provider_config, dict) else None,
                (provider_config.get("reasoning") or {}).get("path") if isinstance(provider_config, dict) else None,
                _normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
            )
            build_reasoning_kwargs(
                openai_kwargs,
                provider_config,
                _normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
            )
            reasoning_logger.debug(
                "openai reasoning kwargs applied model=%s has_reasoning=%s has_extra_body=%s reasoning_effort=%s extra_body_keys=%s",
                config.openai_model,
                "reasoning" in openai_kwargs,
                "extra_body" in openai_kwargs,
                openai_kwargs.get("reasoning_effort"),
                sorted((openai_kwargs.get("extra_body") or {}).keys())
                if isinstance(openai_kwargs.get("extra_body"), dict)
                else [],
            )
        else:
            reasoning_logger.debug(
                "openai reasoning skipped model=%s reasoning_enabled=%s model_supports_reasoning=%s",
                config.openai_model,
                bool(getattr(config, "enable_model_reasoning", True)),
                _openai_model_supports_reasoning_controls(config.openai_model),
            )
        return ChatOpenAI(**openai_kwargs)
    raise ValueError(f"Unknown provider: {config.provider}")


def create_runtime_llm(config: AgentConfig) -> BaseChatModel | RotatingChatModel:
    profile_id = str(config.active_model_profile_id or "").strip()
    if not profile_id:
        return create_llm(config)
    return RotatingChatModel(
        config=config,
        profile_id=profile_id,
        profile_store_path=config.model_profile_config_path,
        llm_factory=create_llm,
    )


def _register_llm_cleanup_callback(tool_registry: ToolRegistry, llm: Any) -> bool:
    close_method = getattr(llm, "aclose", None) or getattr(llm, "close", None)
    if callable(close_method):
        tool_registry.register_cleanup_callback(close_method)
        return True
    for target in (getattr(llm, "async_client", None), getattr(llm, "client", None)):
        if target is None:
            continue
        target_close = getattr(target, "aclose", None) or getattr(target, "close", None)
        if callable(target_close):
            tool_registry.register_cleanup_callback(target_close)
            return True
    return False


def _resolve_effective_model_capabilities(config: AgentConfig, runtime_capabilities: dict[str, Any]) -> dict[str, Any]:
    profiles_payload = ModelProfileStore(config.model_profile_config_path).load()
    selected_profile_id = str(config.active_model_profile_id or "").strip()
    active_profile = (
        find_profile_by_id(profiles_payload, selected_profile_id)
        if selected_profile_id
        else find_active_profile(profiles_payload)
    )
    return resolve_model_capabilities(active_profile, runtime_capabilities)


# --- Builder ---


def create_agent_workflow(
    nodes: AgentNodes,
    config: AgentConfig,
    tools_enabled: Optional[bool] = None,
) -> StateGraph:
    """Builds the LangGraph workflow with tool-call based routing and bounded recovery."""
    tools_enabled = bool(nodes.tools) and config.model_supports_tools if tools_enabled is None else tools_enabled
    approval_enabled = bool(tools_enabled and config.enable_approvals)

    workflow = StateGraph(AgentState)

    workflow.add_node("summarize", nodes.summarize_node)
    workflow.add_node("agent", nodes.agent_node)
    workflow.add_node("recovery", nodes.recovery_node)
    workflow.add_node("update_step", lambda state: {"steps": state.get("steps", 0) + 1})

    workflow.add_edge(START, "summarize")
    workflow.add_edge("summarize", "update_step")
    workflow.add_edge("update_step", "agent")

    if tools_enabled:
        workflow.add_node("tools", nodes.tools_node)
        if approval_enabled:
            workflow.add_node("approval", nodes.approval_node)

    def route_after_agent(state: AgentState):
        steps = state.get("steps", 0)
        messages = state.get("messages") or []

        if not messages:
            logger.warning("Agent node returned no messages; ending turn safely.")
            return END

        turn_outcome = str(state.get("turn_outcome") or "").strip().lower()
        has_open_tool_issue = bool(state.get("open_tool_issue"))
        has_protocol_error = bool(state.get("has_protocol_error"))

        if tools_enabled and turn_outcome == "run_tools":
            if steps >= config.max_loops:
                logger.warning(
                    "Loop guard reached at step %s/%s with pending tool calls. Routing to recovery.",
                    steps,
                    config.max_loops,
                )
                return "recovery"
            pending_ai_with_tools = nodes._get_last_pending_ai_with_tool_calls(messages)
            if isinstance(pending_ai_with_tools, AIMessage) and pending_ai_with_tools.tool_calls:
                if approval_enabled and nodes.tool_calls_require_approval(pending_ai_with_tools.tool_calls):
                    return "approval"
                return "tools"
            logger.warning(
                "Agent reported run_tools outcome without a valid tool call payload. "
                "Routing to recovery."
            )
            return "recovery"

        if turn_outcome == "recover_agent" or has_open_tool_issue or has_protocol_error:
            return "recovery"

        return END

    def route_after_tools(state: AgentState):
        if state.get("open_tool_issue"):
            return "recovery"
        return "update_step"

    def route_after_recovery(state: AgentState):
        if state.get("turn_outcome") == "recover_agent":
            return "update_step"
        return END

    if tools_enabled:
        agent_routes = ["tools", "recovery", END]
        if approval_enabled:
            agent_routes.insert(0, "approval")
            workflow.add_edge("approval", "tools")
        workflow.add_conditional_edges("agent", route_after_agent, agent_routes)
        workflow.add_conditional_edges("tools", route_after_tools, ["recovery", "update_step"])
    else:
        workflow.add_conditional_edges("agent", route_after_agent, ["recovery", END])

    workflow.add_conditional_edges("recovery", route_after_recovery, ["update_step", END])

    return workflow


def build_compiled_agent(
    config: AgentConfig,
    tool_registry: ToolRegistry,
    checkpoint_runtime: Any,
    *,
    run_logger: Optional[JsonlRunLogger] = None,
) -> Tuple[Any, ToolRegistry]:
    """Compile an agent app using already-loaded tools and an existing checkpointer."""
    llm = create_runtime_llm(config)
    tool_registry.config = config
    tool_registry.model_capabilities = extract_model_capabilities(llm)
    effective_model_capabilities = _resolve_effective_model_capabilities(
        config,
        tool_registry.model_capabilities,
    )
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.checkpoint_runtime = checkpoint_runtime

    tools = tool_registry.tools
    tool_calling_enabled = bool(tools) and config.model_supports_tools
    llm_with_tools = llm
    if tool_calling_enabled:
        llm_with_tools, tool_calling_enabled, bind_error = prepare_llm_with_tools(llm, tools)
        if tool_calling_enabled:
            logger.info("🛠️ Tools bound to LLM successfully.")
        else:
            tool_registry.loader_status.append(
                {
                    "loader": "llm_tool_binding",
                    "module": config.provider,
                    "loaded_tools": [],
                    "error": bind_error,
                }
            )
            logger.error("Tool calling disabled for this runtime because tool binding failed: %s", bind_error)
    elif not config.model_supports_tools:
        logger.debug("⚠️ Tools disabled: Model does not support tool calling.")

    registered_cleanup_ids: set[int] = set()
    for cleanup_target in (llm, llm_with_tools):
        marker = id(cleanup_target)
        if marker in registered_cleanup_ids:
            continue
        if _register_llm_cleanup_callback(tool_registry, cleanup_target):
            registered_cleanup_ids.add(marker)

    active_tools = tools if tool_calling_enabled else []
    active_tool_metadata = tool_registry.tool_metadata if tool_calling_enabled else {}
    nodes = AgentNodes(
        config=config,
        llm=llm,
        tools=active_tools,
        llm_with_tools=llm_with_tools,
        tool_metadata=active_tool_metadata,
        model_capabilities=effective_model_capabilities,
        run_logger=run_logger,
    )
    workflow = create_agent_workflow(nodes, config, tools_enabled=tool_calling_enabled)
    return workflow.compile(checkpointer=checkpoint_runtime.checkpointer), tool_registry


async def build_agent_app(config: Optional[AgentConfig] = None) -> Tuple[Any, ToolRegistry]:
    """
    Builds the LangGraph application and returns it along with the tool registry.
    """
    # Pydantic AgentConfig автоматически загружает .env.
    config = config or AgentConfig()
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        reasoning_debug_enabled=config.debug_reasoning_stream,
    )

    logger.info(f"Initializing agent: [bold cyan]{config.provider}[/]", extra={"markup": True})

    # 1. Initialize Resources
    tool_registry = ToolRegistry(config)
    await tool_registry.load_all()
    checkpoint_runtime = await create_checkpoint_runtime(config)
    run_logger = JsonlRunLogger(config.run_log_dir)
    tool_registry.checkpoint_info = checkpoint_runtime.to_dict()
    tool_registry.checkpoint_runtime = checkpoint_runtime
    tool_registry.register_cleanup_callback(checkpoint_runtime.aclose)
    return build_compiled_agent(
        config,
        tool_registry,
        checkpoint_runtime,
        run_logger=run_logger,
    )


if __name__ == "__main__":

    async def main():
        app, registry = await build_agent_app()
        print(f"✔ Agent Ready. Tools: {len(registry.tools)}")
        await registry.cleanup()

    asyncio.run(main())
