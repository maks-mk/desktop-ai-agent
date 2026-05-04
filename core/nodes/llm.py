from __future__ import annotations

import asyncio
import logging
from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from core.api_key_rotation import ApiKeyRotationExhaustedError, classify_api_key_error
from core.state import AgentState
from core.node_errors import EmptyLLMResponseError
from core.message_utils import compact_text, stringify_content

logger = logging.getLogger("agent")


class LLMMixin:
    """LLM selection, invocation with retry, and fatal-error classification."""

    def _select_llm_for_active_tools(
        self,
        active_tools: List[Any],
        active_tool_names: List[str],
    ) -> BaseChatModel:
        if not active_tool_names:
            return self.llm

        if active_tool_names == list(self._all_tool_names):
            return self.llm_with_tools

        binder = getattr(self.llm, "bind_tools", None)
        if not callable(binder):
            return self.llm_with_tools

        try:
            return binder(active_tools)
        except Exception as exc:
            logger.warning(
                "Failed to bind active tool subset; falling back to pre-bound tool model: %s",
                exc,
            )
            return self.llm_with_tools

    async def _invoke_llm_with_retry(
        self,
        llm,
        context: List[Any],
        state: AgentState | None = None,
        node_name: str = "",
    ):
        current_llm = llm
        context = list(context)
        max_attempts = max(1, self.config.max_retries)
        retry_delay = max(0, self.config.retry_delay)
        auto_tool_choice_fallback_used = False
        auto_tool_choice_warning = "WARNING: Tools are disabled due to server configuration error."
        self._log_run_event(
            state,
            "llm_invoke_start",
            run_id=None if state is None else state.get("run_id", ""),
            node=node_name,
            max_attempts=max_attempts,
            context_messages=len(context),
        )

        for attempt in range(max_attempts):
            try:
                normalized_context = self._normalize_system_prefix_for_provider(context)
                response = await current_llm.ainvoke(normalized_context)
                invalid_calls = getattr(response, "invalid_tool_calls", None)
                if not response.content and not response.tool_calls and not invalid_calls:
                    raise EmptyLLMResponseError("Empty response from LLM")
                self._log_run_event(
                    state,
                    "llm_invoke_success",
                    run_id=None if state is None else state.get("run_id", ""),
                    node=node_name,
                    attempt=attempt + 1,
                    has_content=bool(stringify_content(response.content).strip()),
                    tool_calls=len(getattr(response, "tool_calls", []) or []),
                )
                return response
            except Exception as e:
                err_str = str(e)
                if (
                    "auto" in err_str
                    and "tool choice" in err_str
                    and "requires" in err_str
                    and not auto_tool_choice_fallback_used
                ):
                    logger.warning(
                        "⚠ Server does not support 'auto' tool choice. Falling back to chat-only mode."
                    )
                    auto_tool_choice_fallback_used = True
                    current_llm = self.llm
                    context = list(context)
                    if context and isinstance(context[0], SystemMessage):
                        system_content = str(context[0].content)
                        if auto_tool_choice_warning not in system_content:
                            system_content = f"{system_content}\n\n{auto_tool_choice_warning}"
                        context[0] = SystemMessage(
                            content=system_content
                        )
                    continue

                is_fatal = self._is_fatal_llm_error(e)
                logger.warning(f"LLM Error (Attempt {attempt+1}/{max_attempts}): {e}")
                self._log_run_event(
                    state,
                    "llm_retry",
                    node=node_name,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    fatal=is_fatal,
                    error=str(e),
                )

                if is_fatal:
                    logger.error(f"Fatal LLM error detected. Aborting request: {e}")
                    self._log_run_event(
                        state,
                        "llm_invoke_fatal",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        error=compact_text(str(e), 400),
                    )
                    raise

                if attempt == max_attempts - 1:
                    self._log_run_event(
                        state,
                        "llm_invoke_exhausted",
                        run_id=None if state is None else state.get("run_id", ""),
                        node=node_name,
                        attempt=attempt + 1,
                        error_type=type(e).__name__,
                        error=compact_text(str(e), 400),
                    )
                    raise

                await asyncio.sleep(retry_delay)

        raise RuntimeError("LLM retry loop exited unexpectedly without a response.")

    def _is_fatal_llm_error(self, error: Exception) -> bool:
        if isinstance(error, ApiKeyRotationExhaustedError):
            return True
        error_kind = classify_api_key_error(error)
        if error_kind in {"auth", "billing"}:
            return True
        if error_kind == "rate_limit":
            return False
        err_str = " ".join(str(error).lower().split())
        fatal_markers = (
            "insufficient_balance",
            "insufficient account balance",
            "invalid_api_key",
            "incorrect api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "permission denied",
            "billing",
            "payment required",
            "error code: 401",
            "error code: 402",
            "error code: 403",
        )
        return any(marker in err_str for marker in fatal_markers)
