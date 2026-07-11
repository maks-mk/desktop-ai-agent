"""OpenAI-compatible provider adapter.

Provides a :class:`ReasoningDebugChatOpenAI` subclass that overrides private
streaming methods (``_stream``, ``_astream``) to:

* log raw provider chunks for reasoning-debug diagnostics;
* attach ``reasoning_content`` from non-standard delta fields
  (``reasoning``, ``thinking``, ``analysis``, …) so that downstream code can
  surface thinking tokens from OpenAI-compatible aggregators that don't follow
  the official ``reasoning_content`` key.

The factory :func:`create_openai_chat_model` wires up sampling controls,
disables SDK-level retries (the agent has its own retry/recovery layer), and
applies reasoning kwargs from the provider registry.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from core.config import AgentConfig
from core.providers.base import normalized_reasoning_effort
from core.provider_registry import (
    ProviderRegistry,
    build_reasoning_kwargs,
    provider_supports_reasoning_for_model,
)
from core.reasoning_debug import (
    STRUCTURED_OUTPUT_PARSED_EXCLUDE,
    debug_event,
    elapsed_since,
    log_unknown_fields,
    now,
    preview_value,
)

logger = logging.getLogger("agent")
reasoning_logger = logging.getLogger("agent.reasoning_debug")


def _safe_openai_model_dump(value: Any) -> dict[str, Any]:
    """Dump OpenAI SDK models without serializing structured-output ``parsed``.

    Some OpenAI/LangChain typed response objects carry the parsed Pydantic
    structured output in ``choices[].message.parsed``.  Recent Pydantic versions
    warn when that runtime value is serialized through a field schema that
    expects ``None``.  The parsed object is not needed for stream chunk
    conversion/debug field discovery; LangChain copies it from the typed final
    response separately when required.
    """
    if not hasattr(value, "model_dump"):
        return value
    try:
        return value.model_dump(exclude=STRUCTURED_OUTPUT_PARSED_EXCLUDE)
    except TypeError:
        return value.model_dump()


# ---------------------------------------------------------------------------
# Reasoning-delta extraction
# ---------------------------------------------------------------------------


def extract_openai_reasoning_delta(chunk: Any) -> Any:
    """Extract a reasoning/thinking value from an OpenAI-compatible stream chunk.

    Checks multiple non-standard delta keys used by different aggregators.
    Returns ``None`` if no reasoning content is present.
    """
    if not isinstance(chunk, dict) and hasattr(chunk, "model_dump"):
        try:
            chunk = _safe_openai_model_dump(chunk)
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


# ---------------------------------------------------------------------------
# Reasoning-debug chat model
# ---------------------------------------------------------------------------


def _build_reasoning_debug_chat_openai(base_cls: type) -> type:
    """Return a subclass of ``ChatOpenAI`` with reasoning-debug instrumentation."""

    class ReasoningDebugChatOpenAI(base_cls):
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
                    fields_source = _safe_openai_model_dump(fields_source)
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
            reasoning_delta = extract_openai_reasoning_delta(chunk)
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
                            chunk = _safe_openai_model_dump(chunk)
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
                            chunk = _safe_openai_model_dump(chunk)
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

    return ReasoningDebugChatOpenAI


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_openai_chat_model(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    """Build an OpenAI-compatible chat model with reasoning-debug instrumentation."""
    # Lazy import to avoid loading both providers on startup.
    from langchain_openai import ChatOpenAI as BaseChatOpenAI

    ReasoningDebugChatOpenAI = _build_reasoning_debug_chat_openai(BaseChatOpenAI)
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
    registry = ProviderRegistry.from_path(config.provider_registry_path)
    provider_config = registry.match(config.openai_base_url)
    reasoning_enabled = bool(getattr(config, "enable_model_reasoning", True))
    provider_model_supports_reasoning = provider_supports_reasoning_for_model(provider_config, config.openai_model)
    if reasoning_enabled and provider_model_supports_reasoning:
        debug_event(
            "reasoning_request",
            provider=provider_config.get("id") if isinstance(provider_config, dict) else None,
            base_url=config.openai_base_url,
            model=config.openai_model,
            reasoning_enabled=bool(getattr(config, "enable_model_reasoning", True)),
            reasoning_effort=normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
        )
        reasoning_logger.debug(
            "openai reasoning registry match model=%s base_url=%s provider_id=%s supports_reasoning=%s model_supported=%s validation=%s path=%s effort=%s",
            config.openai_model,
            config.openai_base_url,
            provider_config.get("id") if isinstance(provider_config, dict) else None,
            provider_config.get("supports_reasoning") if isinstance(provider_config, dict) else None,
            provider_model_supports_reasoning,
            provider_config.get("validation") if isinstance(provider_config, dict) else None,
            (provider_config.get("reasoning") or {}).get("path") if isinstance(provider_config, dict) else None,
            normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
        )
        build_reasoning_kwargs(
            openai_kwargs,
            provider_config,
            normalized_reasoning_effort(getattr(config, "model_reasoning_effort", "medium")),
        )
        reasoning_logger.debug(
            "openai reasoning kwargs applied model=%s provider_id=%s reasoning_path=%s has_reasoning_key=%s has_extra_body=%s reasoning_effort=%s extra_body_keys=%s applied_keys=%s",
            config.openai_model,
            provider_config.get("id") if isinstance(provider_config, dict) else None,
            (provider_config.get("reasoning") or {}).get("path") if isinstance(provider_config, dict) else None,
            "reasoning" in openai_kwargs,
            "extra_body" in openai_kwargs,
            openai_kwargs.get("reasoning_effort"),
            sorted((openai_kwargs.get("extra_body") or {}).keys())
            if isinstance(openai_kwargs.get("extra_body"), dict)
            else [],
            sorted(k for k in openai_kwargs if k not in ("model", "temperature", "api_key", "base_url", "max_retries", "stream_usage")),
        )
    else:
        reasoning_logger.debug(
            "openai reasoning skipped model=%s base_url=%s provider_id=%s reasoning_enabled=%s provider_supports_reasoning=%s model_supported=%s",
            config.openai_model,
            config.openai_base_url,
            provider_config.get("id") if isinstance(provider_config, dict) else None,
            reasoning_enabled,
            provider_config.get("supports_reasoning") if isinstance(provider_config, dict) else None,
            provider_model_supports_reasoning,
        )
    return ChatOpenAI(**openai_kwargs)
