"""Provider adapter regression tests.

These tests verify that the provider-specific subclasses and monkey-patches
moved to ``core/providers/`` preserve their structural contract — i.e. that
the private-method overrides and module-level patches are still in place
after the refactor.  They do **not** make real API calls.

If a LangChain/OpenAI/Google SDK update renames or removes a private method,
these tests will fail early instead of silently breaking streaming or
thought-signature round-tripping at runtime.
"""

import unittest
import warnings
from unittest import mock

from pydantic import BaseModel, ConfigDict

from core.providers import (
    chat_model_accepts_kwarg,
    create_llm,
    extract_openai_reasoning_delta,
    gemini_model_supports_thinking_budget,
    gemini_model_supports_thinking_level,
    normalized_gemini_model_name,
    normalized_gemini_thinking_level,
    normalized_model_name,
    normalized_reasoning_effort,
    patch_langchain_google_genai_retry_kwargs,
    prepare_llm_with_tools,
)
from core.providers.base import (
    chat_model_accepts_kwarg as base_chat_model_accepts_kwarg,
)
from core.providers.gemini import (
    _build_gemini_thought_signature_adapter,
    _encode_gemini_thought_signature,
    _decode_gemini_thought_signature,
    create_gemini_chat_model,
)
from core.reasoning_debug import preview_value
from core.providers.openai_reasoning import (
    _build_reasoning_debug_chat_openai,
    _safe_openai_model_dump,
    create_openai_chat_model,
)


class ProviderPackageExportsTests(unittest.TestCase):
    """Verify that all public symbols are re-exported from ``core.providers``."""

    def test_public_api_callable(self):
        for fn in (
            create_llm,
            prepare_llm_with_tools,
            extract_openai_reasoning_delta,
            patch_langchain_google_genai_retry_kwargs,
            gemini_model_supports_thinking_budget,
            gemini_model_supports_thinking_level,
            normalized_model_name,
            normalized_gemini_model_name,
            normalized_reasoning_effort,
            normalized_gemini_thinking_level,
            chat_model_accepts_kwarg,
        ):
            self.assertTrue(callable(fn), f"{fn!r} is not callable")

    def test_base_helpers_are_same_object(self):
        """Re-exported helpers must be identical to the base-module originals."""
        self.assertIs(chat_model_accepts_kwarg, base_chat_model_accepts_kwarg)


class GeminiThoughtSignatureHelperTests(unittest.TestCase):
    """Unit tests for thought-signature encode/decode round-trip."""

    def test_encode_decode_bytes_roundtrip(self):
        original = b"\x00\x01\xff\xfe"
        encoded = _encode_gemini_thought_signature(original)
        decoded = _decode_gemini_thought_signature(encoded)
        self.assertEqual(decoded, original)

    def test_encode_str_passthrough(self):
        self.assertEqual(_encode_gemini_thought_signature("abc"), "abc")

    def test_decode_empty_returns_empty_bytes(self):
        self.assertEqual(_decode_gemini_thought_signature(""), b"")
        self.assertEqual(_decode_gemini_thought_signature(None), b"")

    def test_decode_invalid_base64_falls_back_to_utf8(self):
        # Non-base64 string should fall back to utf-8 encoding, not raise.
        result = _decode_gemini_thought_signature("not!base64?")
        self.assertEqual(result, "not!base64?".encode("utf-8"))

    def test_encode_memoryview(self):
        mv = memoryview(b"hello")
        encoded = _encode_gemini_thought_signature(mv)
        self.assertEqual(_decode_gemini_thought_signature(encoded), b"hello")

    def test_encode_bytearray(self):
        ba = bytearray(b"world")
        encoded = _encode_gemini_thought_signature(ba)
        self.assertEqual(_decode_gemini_thought_signature(encoded), b"world")


class GeminiModelDetectionTests(unittest.TestCase):
    def test_thinking_budget_for_2_5_models(self):
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-2.5-flash"))
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-2.5-pro"))

    def test_thinking_budget_for_latest_aliases(self):
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-flash-latest"))
        self.assertTrue(gemini_model_supports_thinking_budget("gemini-pro-latest"))

    def test_no_thinking_budget_for_older_models(self):
        self.assertFalse(gemini_model_supports_thinking_budget("gemini-1.5-pro"))
        self.assertFalse(gemini_model_supports_thinking_budget("gemini-1.0-pro"))

    def test_thinking_level_for_gemini3(self):
        self.assertTrue(gemini_model_supports_thinking_level("gemini-3-pro"))
        self.assertFalse(gemini_model_supports_thinking_level("gemini-2.5-flash"))

    def test_normalized_gemini_model_name_strips_prefix(self):
        self.assertEqual(normalized_gemini_model_name("models/gemini-2.5-flash"), "gemini-2.5-flash")
        self.assertEqual(normalized_gemini_model_name("Gemini-2.5-Flash"), "gemini-2.5-flash")


class ReasoningEffortNormalizationTests(unittest.TestCase):
    def test_valid_values(self):
        self.assertEqual(normalized_reasoning_effort("high"), "high")
        self.assertEqual(normalized_reasoning_effort("none"), "none")
        self.assertEqual(normalized_reasoning_effort("xhigh"), "xhigh")

    def test_invalid_defaults_to_medium(self):
        self.assertEqual(normalized_reasoning_effort("ultra"), "medium")
        self.assertEqual(normalized_reasoning_effort(None), "medium")
        self.assertEqual(normalized_reasoning_effort(""), "medium")

    def test_gemini_thinking_level_mapping(self):
        self.assertEqual(normalized_gemini_thinking_level("none"), "minimal")
        self.assertEqual(normalized_gemini_thinking_level("minimal"), "minimal")
        self.assertEqual(normalized_gemini_thinking_level("xhigh"), "high")
        self.assertEqual(normalized_gemini_thinking_level("medium"), "medium")


class OpenAIReasoningDeltaTests(unittest.TestCase):
    """Tests for ``extract_openai_reasoning_delta`` — non-standard reasoning keys."""

    def test_standard_reasoning_content(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        self.assertEqual(extract_openai_reasoning_delta(chunk), "thinking...")

    def test_non_standard_thinking_key(self):
        chunk = {"choices": [{"delta": {"thinking": "analyzing..."}}]}
        self.assertEqual(extract_openai_reasoning_delta(chunk), "analyzing...")

    def test_analysis_key(self):
        chunk = {"choices": [{"delta": {"analysis": "evaluating..."}}]}
        self.assertEqual(extract_openai_reasoning_delta(chunk), "evaluating...")

    def test_no_reasoning_returns_none(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        self.assertIsNone(extract_openai_reasoning_delta(chunk))

    def test_empty_delta_returns_none(self):
        chunk = {"choices": [{"delta": {}}]}
        self.assertIsNone(extract_openai_reasoning_delta(chunk))

    def test_no_choices_returns_none(self):
        self.assertIsNone(extract_openai_reasoning_delta({}))

    def test_object_with_model_dump(self):
        class FakeChunk:
            def model_dump(self):
                return {"choices": [{"delta": {"reasoning": "from_obj"}}]}

        self.assertEqual(extract_openai_reasoning_delta(FakeChunk()), "from_obj")

    def test_non_dict_non_object_returns_none(self):
        self.assertIsNone(extract_openai_reasoning_delta(42))
        self.assertIsNone(extract_openai_reasoning_delta("string"))


class ChatModelAcceptsKwargTests(unittest.TestCase):
    def test_pydantic_model_with_field(self):
        class FakeModel:
            model_fields = {"temperature": ..., "model": ...}

        self.assertTrue(chat_model_accepts_kwarg(FakeModel, "temperature"))
        self.assertTrue(chat_model_accepts_kwarg(FakeModel, "model"))

    def test_pydantic_model_without_field(self):
        class FakeModel:
            model_fields = {"temperature": ...}

        self.assertFalse(chat_model_accepts_kwarg(FakeModel, "top_p"))

    def test_non_pydantic_model_defaults_true(self):
        class PlainModel:
            pass

        self.assertTrue(chat_model_accepts_kwarg(PlainModel, "anything"))


class GeminiAdapterStructureTests(unittest.TestCase):
    """Verify that the Gemini thought-signature adapter overrides the right methods."""

    def test_adapter_returns_base_when_module_missing_helpers(self):
        """If chat_models_module lacks required functions, base class is returned."""
        class FakeBase:
            pass

        fake_module = mock.MagicMock()
        # Remove required callables
        del fake_module._chat_with_retry
        result = _build_gemini_thought_signature_adapter(FakeBase, fake_module)
        self.assertIs(result, FakeBase)

    def test_adapter_subclass_has_overrides(self):
        """The adapter subclass must define _prepare_request, _generate, _agenerate."""
        class FakeBase:
            def _prepare_request(self, messages, *args, **kwargs):
                return mock.MagicMock()

        fake_module = mock.MagicMock()
        # All required callables present
        fake_module._chat_with_retry = mock.MagicMock()
        fake_module._achat_with_retry = mock.MagicMock()
        fake_module._response_to_result = mock.MagicMock()

        adapter_cls = _build_gemini_thought_signature_adapter(FakeBase, fake_module)
        self.assertIsNot(adapter_cls, FakeBase)
        self.assertTrue(hasattr(adapter_cls, "_prepare_request"))
        self.assertTrue(hasattr(adapter_cls, "_generate"))
        self.assertTrue(hasattr(adapter_cls, "_agenerate"))


class OpenAIAdapterStructureTests(unittest.TestCase):
    """Verify that the ReasoningDebugChatOpenAI subclass overrides the right methods."""

    def test_subclass_has_stream_overrides(self):
        class FakeBase:
            pass

        cls = _build_reasoning_debug_chat_openai(FakeBase)
        self.assertTrue(hasattr(cls, "_get_request_payload"))
        self.assertTrue(hasattr(cls, "_generate"))
        self.assertTrue(hasattr(cls, "_stream"))
        self.assertTrue(hasattr(cls, "_astream"))
        self.assertTrue(hasattr(cls, "_log_raw_provider_chunk"))
        self.assertTrue(hasattr(cls, "_attach_raw_reasoning_delta"))

    def test_safe_openai_model_dump_excludes_structured_parsed_without_warning(self):
        class FakeParsedPayload(BaseModel):
            summary: str

        class FakeOpenAIMessage(BaseModel):
            model_config = ConfigDict(extra="allow")
            parsed: None = None

        class FakeOpenAIChoice(BaseModel):
            message: FakeOpenAIMessage

        class FakeOpenAICompletion(BaseModel):
            choices: list[FakeOpenAIChoice]

        draft = FakeParsedPayload(summary="Implement a fix")
        message = FakeOpenAIMessage.model_construct(parsed=draft)
        completion = FakeOpenAICompletion(choices=[FakeOpenAIChoice(message=message)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            dumped = _safe_openai_model_dump(completion)

        self.assertEqual(captured, [])
        self.assertNotIn("parsed", dumped["choices"][0]["message"])

    def test_reasoning_debug_preview_excludes_structured_parsed_without_warning(self):
        class FakeParsedPayload(BaseModel):
            summary: str

        class FakeOpenAIMessage(BaseModel):
            model_config = ConfigDict(extra="allow")
            parsed: None = None

        class FakeOpenAIChoice(BaseModel):
            message: FakeOpenAIMessage

        class FakeOpenAICompletion(BaseModel):
            choices: list[FakeOpenAIChoice]

        draft = FakeParsedPayload(summary="Implement a fix")
        message = FakeOpenAIMessage.model_construct(parsed=draft)
        completion = FakeOpenAICompletion(choices=[FakeOpenAIChoice(message=message)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rendered = preview_value(completion)

        self.assertEqual(captured, [])
        self.assertNotIn("parsed", rendered)

    def test_reasoning_debug_preview_excludes_top_level_parsed_without_warning(self):
        class FakeParsedPayload(BaseModel):
            summary: str

        class FakeParsedCompletion(BaseModel):
            model_config = ConfigDict(extra="allow")
            parsed: None = None

        draft = FakeParsedPayload(summary="Implement a fix")
        completion = FakeParsedCompletion.model_construct(parsed=draft)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rendered = preview_value(completion)

        self.assertEqual(captured, [])
        self.assertNotIn("parsed", rendered)


class RetryPatchIdempotencyTests(unittest.TestCase):
    """The retry-kwargs monkey-patch must be idempotent."""

    def test_patch_is_idempotent(self):
        with mock.patch("importlib.import_module") as mock_import:
            fake_module = mock.MagicMock()
            fake_module._chat_with_retry = mock.MagicMock()
            fake_module._achat_with_retry = mock.MagicMock()
            fake_module._agent_retry_patch_applied = False
            mock_import.return_value = fake_module

            patch_langchain_google_genai_retry_kwargs()
            first_chat = fake_module._chat_with_retry
            first_achat = fake_module._achat_with_retry

            # Second call should be a no-op
            patch_langchain_google_genai_retry_kwargs()
            self.assertIs(fake_module._chat_with_retry, first_chat)
            self.assertIs(fake_module._achat_with_retry, first_achat)
            self.assertTrue(fake_module._agent_retry_patch_applied)


class PrepareLLMWithToolsTests(unittest.TestCase):
    def test_no_tools_returns_false(self):
        llm = mock.MagicMock()
        result_llm, enabled, error = prepare_llm_with_tools(llm, [])
        self.assertIs(result_llm, llm)
        self.assertFalse(enabled)
        self.assertEqual(error, "")

    def test_no_bind_tools_method(self):
        llm = mock.MagicMock()
        del llm.bind_tools
        result_llm, enabled, error = prepare_llm_with_tools(llm, [mock.MagicMock()])
        self.assertIs(result_llm, llm)
        self.assertFalse(enabled)
        self.assertIn("bind_tools", error)

    def test_successful_binding(self):
        llm = mock.MagicMock()
        bound = mock.MagicMock()
        llm.bind_tools.return_value = bound
        result_llm, enabled, error = prepare_llm_with_tools(llm, [mock.MagicMock()])
        self.assertIs(result_llm, bound)
        self.assertTrue(enabled)
        self.assertEqual(error, "")

    def test_binding_exception(self):
        llm = mock.MagicMock()
        llm.bind_tools.side_effect = ValueError("unsupported tool schema")
        result_llm, enabled, error = prepare_llm_with_tools(llm, [mock.MagicMock()])
        self.assertIs(result_llm, llm)
        self.assertFalse(enabled)
        self.assertIn("unsupported tool schema", error)


class FactoryUnknownProviderTests(unittest.TestCase):
    def test_unknown_provider_raises(self):
        from core.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        config.provider = "unknown_provider"
        with self.assertRaises(ValueError) as ctx:
            create_llm(config)
        self.assertIn("unknown_provider", str(ctx.exception))


class LlmApiModeTests(unittest.TestCase):
    """Tests for LLM_API_MODE config field and its effect on the OpenAI factory."""

    def _make_config(self, api_mode: str | None) -> "AgentConfig":
        import os
        from core.config import AgentConfig

        env = {
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-5",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "MODEL_REASONING_EFFORT": "medium",
        }
        if api_mode is not None:
            env["LLM_API_MODE"] = api_mode
        with mock.patch.dict(os.environ, env, clear=False):
            # Remove LLM_API_MODE if explicitly None to test default
            if api_mode is None:
                os.environ.pop("LLM_API_MODE", None)
            return AgentConfig()

    def test_default_is_chat(self):
        cfg = self._make_config(None)
        self.assertEqual(cfg.llm_api_mode, "chat")

    def test_responses_mode(self):
        cfg = self._make_config("responses")
        self.assertEqual(cfg.llm_api_mode, "responses")

    def test_invalid_falls_back_to_chat(self):
        cfg = self._make_config("bogus")
        self.assertEqual(cfg.llm_api_mode, "chat")

    def test_case_insensitive(self):
        cfg = self._make_config("RESPONSES")
        self.assertEqual(cfg.llm_api_mode, "responses")

    def test_responses_mode_sets_use_responses_api(self):
        cfg = self._make_config("responses")
        model = create_openai_chat_model(cfg)
        self.assertTrue(model.use_responses_api)

    def test_chat_mode_does_not_set_use_responses_api(self):
        cfg = self._make_config("chat")
        model = create_openai_chat_model(cfg)
        self.assertFalse(model.use_responses_api)

    def test_chat_mode_flattens_reasoning_dict_to_effort_string(self):
        """In chat mode, the registry's reasoning.effort path must not create a
        top-level 'reasoning' dict that would auto-trigger the Responses API."""
        cfg = self._make_config("chat")
        model = create_openai_chat_model(cfg)
        # reasoning_effort should be set as a plain string
        self.assertEqual(model.reasoning_effort, "medium")
        # reasoning dict should NOT be set (it would auto-switch to Responses API)
        self.assertIsNone(model.reasoning)

    def test_responses_mode_keeps_reasoning_dict(self):
        """In responses mode, the registry's reasoning.effort path should produce
        a 'reasoning' dict with effort and summary fields."""
        cfg = self._make_config("responses")
        model = create_openai_chat_model(cfg)
        self.assertIsInstance(model.reasoning, dict)
        self.assertEqual(model.reasoning.get("effort"), "medium")
        self.assertEqual(model.reasoning.get("summary"), "auto")

    def test_chat_mode_use_responses_api_returns_false_with_reasoning(self):
        """Even with reasoning enabled, chat mode must not route to Responses API."""
        cfg = self._make_config("chat")
        model = create_openai_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertFalse(model._use_responses_api(payload))

    def test_responses_mode_use_responses_api_returns_true(self):
        """In responses mode, _use_responses_api must return True."""
        cfg = self._make_config("responses")
        model = create_openai_chat_model(cfg)
        payload = model._get_request_payload([{"role": "user", "content": "hi"}])
        self.assertTrue(model._use_responses_api(payload))


if __name__ == "__main__":
    unittest.main()
