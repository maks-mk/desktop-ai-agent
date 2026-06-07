import asyncio
import unittest
from unittest.mock import patch

import httpx

from core.model_fetcher import (
    AuthError,
    EmptyResultError,
    GeminiModelFetcher,
    InvalidResponseError,
    ModelEntry,
    NetworkError,
    OpenAICompatibleModelFetcher,
    RateLimitError,
)
from ui.widgets.dialogs import ModelFetchWorker


class _FakeAsyncClient:
    def __init__(self, *, response: httpx.Response | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _url: str, **_kwargs):
        if self._error is not None:
            raise self._error
        return self._response


class _FailingFetcher:
    async def fetch(self, _api_key: str, _base_url: str = ""):
        raise RuntimeError("boom")


def _gemini_response(*models: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"models": list(models)},
        request=httpx.Request("GET", "https://generativelanguage.googleapis.com/v1beta/models"),
    )


def _openai_response(*models: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"data": list(models)},
        request=httpx.Request("GET", "https://example.test/v1/models"),
    )


class ModelFetcherTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_gemini_filter_excludes_embed(self):
        response = _gemini_response(
            {"name": "models/gemini-1.5-pro", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual(result, [ModelEntry(id="gemini-1.5-pro", family="gemini", supports_image_input=True)])

    def test_gemini_filter_excludes_audio(self):
        response = _gemini_response(
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-flash-tts", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-audio-preview", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-speech-live", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual([entry.id for entry in result], ["gemini-2.5-flash"])

    def test_gemini_filter_excludes_imagen(self):
        response = _gemini_response(
            {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/imagen-3.0-generate-002", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-image-preview", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual([entry.id for entry in result], ["gemini-2.0-flash"])

    def test_gemini_filter_excludes_aqa(self):
        response = _gemini_response(
            {"name": "models/gemini-1.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-1.5-pro-aqa", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-retrieval-preview", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual([entry.id for entry in result], ["gemini-1.5-flash"])

    def test_gemini_filter_requires_generate_content(self):
        response = _gemini_response(
            {"name": "models/gemini-1.5-pro", "supportedGenerationMethods": ["countTokens"]},
            {"name": "models/gemini-1.5-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual([entry.id for entry in result], ["gemini-1.5-flash"])

    def test_gemini_whitelist_blocks_unknown_family(self):
        response = _gemini_response(
            {"name": "models/claude-3-7-sonnet", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual([entry.id for entry in result], ["gemini-2.5-pro"])

    def test_gemini_allows_gemma(self):
        response = _gemini_response(
            {"name": "models/gemma-3-27b-it", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual(result, [ModelEntry(id="gemma-3-27b-it", family="gemma", supports_image_input=True)])

    def test_gemini_normalizes_prefix(self):
        response = _gemini_response(
            {"name": "models/gemini-1.5-pro", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(GeminiModelFetcher().fetch("gm-key"))
        self.assertEqual(result[0].id, "gemini-1.5-pro")

    def test_gemini_empty_after_filter(self):
        response = _gemini_response(
            {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["generateContent"]},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaises(EmptyResultError):
                self._run(GeminiModelFetcher().fetch("gm-key"))

    def test_openai_no_whitelist(self):
        response = _openai_response(
            {"id": "custom-local-model"},
            {"id": "gpt-4o"},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(OpenAICompatibleModelFetcher().fetch("sk-key", "https://example.test/v1"))
        self.assertEqual([entry.id for entry in result], ["custom-local-model", "gpt-4o"])

    def test_openai_fallback_on_empty(self):
        response = _openai_response(
            {"id": "text-embedding-3-large"},
            {"id": "gpt-4o-audio-preview"},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(OpenAICompatibleModelFetcher().fetch("sk-key", "https://example.test/v1"))
        self.assertEqual([entry.id for entry in result], ["text-embedding-3-large", "gpt-4o-audio-preview"])

    def test_auth_error_on_401(self):
        response = _gemini_response(status_code=401)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaises(AuthError):
                self._run(GeminiModelFetcher().fetch("bad-key"))

    def test_rate_limit_on_429(self):
        response = _gemini_response(status_code=429)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaises(RateLimitError):
                self._run(GeminiModelFetcher().fetch("gm-key"))

    def test_network_error_on_timeout(self):
        timeout_error = httpx.ReadTimeout("timed out")
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(error=timeout_error)):
            with self.assertRaises(NetworkError):
                self._run(GeminiModelFetcher().fetch("gm-key"))

    def test_openai_invalid_json_raises_fetch_error(self):
        response = httpx.Response(
            200,
            text="<html>not json</html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", "https://example.test/v1/models"),
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaises(InvalidResponseError) as context:
                self._run(OpenAICompatibleModelFetcher().fetch("sk-key", "https://example.test/v1"))
        self.assertIn("invalid JSON", str(context.exception))

    def test_model_fetch_worker_reports_unexpected_error_with_request_id(self):
        worker = ModelFetchWorker(42, _FailingFetcher(), "sk-key", "https://example.test/v1")
        captured = []
        worker.failed.connect(lambda request_id, message: captured.append((request_id, message)))

        worker.run()

        self.assertEqual(captured, [(42, "Не удалось загрузить модели.")])


if __name__ == "__main__":
    unittest.main()
