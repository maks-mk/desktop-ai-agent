"""Provider adapter package.

Public API:
    create_llm              — provider-dispatching LLM factory
    create_runtime_llm      — LLM with optional API-key rotation
    prepare_llm_with_tools  — bind tools to an LLM
    extract_openai_reasoning_delta — parse reasoning content from stream chunks
    gemini_model_supports_thinking_budget — check Gemini thinking-budget support
    patch_langchain_google_genai_retry_kwargs — retry-kwargs monkey-patch
"""

from core.providers.base import (
    chat_model_accepts_kwarg,
    gemini_model_supports_thinking_budget,
    gemini_model_supports_thinking_level,
    normalized_gemini_model_name,
    normalized_gemini_thinking_level,
    normalized_model_name,
    normalized_reasoning_effort,
)
from core.providers.factory import (
    create_llm,
    create_runtime_llm,
    prepare_llm_with_tools,
)
from core.providers.gemini import (
    patch_langchain_google_genai_retry_kwargs,
)
from core.providers.openai_reasoning import (
    extract_openai_reasoning_delta,
)

__all__ = [
    "create_llm",
    "create_runtime_llm",
    "prepare_llm_with_tools",
    "extract_openai_reasoning_delta",
    "gemini_model_supports_thinking_budget",
    "gemini_model_supports_thinking_level",
    "normalized_model_name",
    "normalized_gemini_model_name",
    "normalized_reasoning_effort",
    "normalized_gemini_thinking_level",
    "chat_model_accepts_kwarg",
    "patch_langchain_google_genai_retry_kwargs",
]
