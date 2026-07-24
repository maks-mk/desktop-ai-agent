import asyncio
import unittest
from unittest.mock import patch

import httpx

from core.model_fetcher import (
    AnthropicModelFetcher,
    AuthError,
    EmptyResultError,
    FetchError,
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
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _url: str, **_kwargs):
        self.requests.append((_url, _kwargs))
        if self._error is not None:
            raise self._error
        return self._response


class _PagedFakeAsyncClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _url: str, **_kwargs):
        self.requests.append((_url, _kwargs))
        return self._responses.pop(0)


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

    def test_openai_models_request_uses_configured_headers(self):
        response = _openai_response({"id": "gpt-4o"})
        client = _FakeAsyncClient(response=response)
        headers = {"User-Agent": "CustomAgent/1.0", "x-provider-header": "enabled"}
        with (
            patch("core.model_fetcher.httpx.AsyncClient", return_value=client),
            patch("core.model_fetcher.load_openai_headers", return_value=headers),
        ):
            self._run(OpenAICompatibleModelFetcher().fetch("sk-key", "https://example.test/v1"))

        self.assertEqual(client.requests[0][1]["headers"], {**headers, "Authorization": "Bearer sk-key"})

    def test_gemini_models_request_does_not_use_openai_headers(self):
        response = _gemini_response(
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        )
        client = _FakeAsyncClient(response=response)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            self._run(GeminiModelFetcher().fetch("gm-key"))

        self.assertNotIn("headers", client.requests[0][1])

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

    def test_anthropic_returns_models_from_data(self):
        response = _openai_response(
            {"id": "claude-sonnet-4-5-20250929"},
            {"id": "claude-opus-4-20250514"},
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            result = self._run(AnthropicModelFetcher().fetch("sk-ant-key"))
        self.assertEqual([entry.id for entry in result], ["claude-sonnet-4-5-20250929", "claude-opus-4-20250514"])
        self.assertTrue(all(entry.supports_image_input for entry in result))

    def test_anthropic_fetches_all_pages_with_after_id_cursor(self):
        responses = [
            httpx.Response(
                200,
                json={
                    "data": [{"id": "claude-sonnet-5"}, {"id": "claude-opus-4-8"}],
                    "has_more": True,
                    "last_id": "claude-opus-4-8",
                },
                request=httpx.Request("GET", "https://example.test/v1/models"),
            ),
            httpx.Response(
                200,
                json={"data": [{"id": "claude-haiku-4-5"}], "has_more": False},
                request=httpx.Request("GET", "https://example.test/v1/models"),
            ),
        ]
        client = _PagedFakeAsyncClient(responses)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            result = self._run(AnthropicModelFetcher().fetch("sk-ant-key", "https://proxy.example"))

        self.assertEqual([entry.id for entry in result], ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"])
        self.assertEqual(client.requests[0][0], "https://proxy.example/v1/models")
        self.assertNotIn("params", client.requests[0][1])
        self.assertEqual(client.requests[1][1]["params"], {"after_id": "claude-opus-4-8"})

    def test_anthropic_pagination_deduplicates_model_ids(self):
        responses = [
            httpx.Response(
                200,
                json={"data": [{"id": "claude-sonnet-5"}], "has_more": True, "last_id": "cursor-1"},
                request=httpx.Request("GET", "https://example.test/v1/models"),
            ),
            httpx.Response(
                200,
                json={"data": [{"id": "claude-sonnet-5"}, {"id": "claude-opus-4-8"}], "has_more": False},
                request=httpx.Request("GET", "https://example.test/v1/models"),
            ),
        ]
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_PagedFakeAsyncClient(responses)):
            result = self._run(AnthropicModelFetcher().fetch("sk-ant-key"))
        self.assertEqual([entry.id for entry in result], ["claude-sonnet-5", "claude-opus-4-8"])

    def test_anthropic_pagination_requires_last_id(self):
        response = httpx.Response(
            200,
            json={"data": [{"id": "claude-sonnet-5"}], "has_more": True},
            request=httpx.Request("GET", "https://example.test/v1/models"),
        )
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaisesRegex(FetchError, "without last_id"):
                self._run(AnthropicModelFetcher().fetch("sk-ant-key"))

    def test_anthropic_pagination_rejects_repeated_last_id(self):
        responses = [
            httpx.Response(
                200,
                json={"data": [{"id": "claude-sonnet-5"}], "has_more": True, "last_id": "cursor-1"},
                request=httpx.Request("GET", "https://example.test/v1/models"),
            ),
            httpx.Response(
                200,
                json={"data": [{"id": "claude-opus-4-8"}], "has_more": True, "last_id": "cursor-1"},
                request=httpx.Request("GET", "https://example.test/v1/models"),
            ),
        ]
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_PagedFakeAsyncClient(responses)):
            with self.assertRaisesRegex(FetchError, "repeated last_id"):
                self._run(AnthropicModelFetcher().fetch("sk-ant-key"))

    def test_anthropic_uses_x_api_key_header(self):
        response = _openai_response({"id": "claude-sonnet-4-5-20250929"})
        client = _FakeAsyncClient(response=response)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            self._run(AnthropicModelFetcher().fetch("sk-ant-key"))
        headers = client.requests[0][1]["headers"]
        self.assertEqual(headers["x-api-key"], "sk-ant-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", headers)

    def test_anthropic_default_endpoint_without_base_url(self):
        response = _openai_response({"id": "claude-sonnet-4-5-20250929"})
        client = _FakeAsyncClient(response=response)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            self._run(AnthropicModelFetcher().fetch("sk-ant-key"))
        self.assertEqual(client.requests[0][0], "https://api.anthropic.com/v1/models")

    def test_anthropic_custom_base_url(self):
        response = _openai_response({"id": "claude-sonnet-4-5-20250929"})
        client = _FakeAsyncClient(response=response)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            self._run(AnthropicModelFetcher().fetch("sk-ant-key", "https://proxy.example/v1"))
        self.assertEqual(client.requests[0][0], "https://proxy.example/v1/models")

    def test_anthropic_custom_base_url_without_v1(self):
        response = _openai_response({"id": "claude-sonnet-4-5-20250929"})
        client = _FakeAsyncClient(response=response)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            self._run(AnthropicModelFetcher().fetch("sk-ant-key", "https://proxy.example"))
        self.assertEqual(client.requests[0][0], "https://proxy.example/v1/models")

    def test_anthropic_custom_base_url_with_trailing_slash(self):
        response = _openai_response({"id": "claude-sonnet-4-5-20250929"})
        client = _FakeAsyncClient(response=response)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=client):
            self._run(AnthropicModelFetcher().fetch("sk-ant-key", "https://proxy.example/"))
        self.assertEqual(client.requests[0][0], "https://proxy.example/v1/models")

    def test_anthropic_empty_raises(self):
        response = _openai_response()
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaises(EmptyResultError):
                self._run(AnthropicModelFetcher().fetch("sk-ant-key"))

    def test_anthropic_auth_error_on_401(self):
        response = _openai_response(status_code=401)
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(response=response)):
            with self.assertRaises(AuthError):
                self._run(AnthropicModelFetcher().fetch("bad-key"))

    def test_anthropic_network_error_on_timeout(self):
        timeout_error = httpx.ReadTimeout("timed out")
        with patch("core.model_fetcher.httpx.AsyncClient", return_value=_FakeAsyncClient(error=timeout_error)):
            with self.assertRaises(NetworkError):
                self._run(AnthropicModelFetcher().fetch("sk-ant-key"))


if __name__ == "__main__":
    unittest.main()
