import asyncio
import shutil
import unittest
from pathlib import Path

from core.api_key_rotation import (
    ApiKeyRotationExhaustedError,
    RotatingChatModel,
)
from core.config import AgentConfig
from core.model_profiles import ModelProfileStore


class _FakeStatusResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, response_status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeStatusResponse(response_status_code) if response_status_code is not None else None


class _FakeAuthenticationError(_FakeProviderError):
    pass


class _FakeResourceExhaustedError(_FakeProviderError):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = lambda: "RESOURCE_EXHAUSTED"


class _FakePermissionDeniedError(_FakeProviderError):
    def __init__(self, message: str):
        super().__init__(message)
        self.code = lambda: "PERMISSION_DENIED"


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []
        self.invalid_tool_calls = []


class _FakeModel:
    def __init__(self, api_key: str, outcomes: dict[str, list[object]], calls: list[str]):
        self.api_key = api_key
        self._outcomes = outcomes
        self._calls = calls
        self.profile = {}

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _input, **_kwargs):
        self._calls.append(self.api_key)
        queue = self._outcomes.setdefault(self.api_key, [])
        if not queue:
            return _FakeResponse(f"ok:{self.api_key}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(str(outcome))


class ApiKeyRotationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = Path.cwd() / ".tmp_tests" / f"api_key_rotation_{id(self)}"
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self._tmpdir, ignore_errors=True))

    def _config(self, profile_path: Path) -> AgentConfig:
        return AgentConfig(
            provider="openai",
            openai_model="gpt-4o",
            openai_api_key="sk-seed",
            active_model_profile_id="gpt-4o",
            model_profile_config_path=profile_path,
        )

    def _store(self, profile_path: Path) -> ModelProfileStore:
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2", "sk-3"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        return store

    def test_rotating_model_retries_next_key_on_rate_limit(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_auth_error_without_marking_invalid(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Unauthorized", status_code=401)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        saved_profile = store.load()["profiles"][0]
        self.assertEqual(saved_profile["api_key_index"], 1)
        self.assertEqual(saved_profile["invalid_api_keys"], [])

    def test_rotating_model_rotates_on_billing_error(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Payment Required", status_code=402)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_class_name_based_auth_error(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeAuthenticationError("bad key")],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_provider_code_markers(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeResourceExhaustedError("quota bucket exhausted")],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_rotates_on_response_status_code(self):
        profile_path = self._tmpdir / "config.json"
        store = self._store(profile_path)
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Forbidden", response_status_code=403)],
            "sk-2": ["success-from-sk-2"],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        response = asyncio.run(model.ainvoke("hello"))

        self.assertEqual(response.content, "success-from-sk-2")
        self.assertEqual(calls, ["sk-1", "sk-2"])
        self.assertEqual(store.load()["profiles"][0]["api_key_index"], 1)

    def test_rotating_model_stops_after_pool_is_exhausted(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("429 Too Many Requests", status_code=429)],
            "sk-2": [_FakeProviderError("Quota exceeded", status_code=429)],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        with self.assertRaises(ApiKeyRotationExhaustedError):
            asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1", "sk-2"])

    def test_rotating_model_raises_exhausted_error_when_all_keys_fail_auth(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakeProviderError("Unauthorized", status_code=401)],
            "sk-2": [_FakeProviderError("Forbidden", status_code=403)],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        with self.assertRaises(ApiKeyRotationExhaustedError) as ctx:
            asyncio.run(model.ainvoke("hello"))

        self.assertIn("Все API-ключи", str(ctx.exception))
        self.assertEqual(calls, ["sk-1", "sk-2"])

    def test_rotating_model_exhausts_pool_on_provider_permission_denied_errors(self):
        profile_path = self._tmpdir / "config.json"
        store = ModelProfileStore(profile_path)
        store.save(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_keys": ["sk-1", "sk-2"],
                        "api_key_index": 0,
                        "api_key": "sk-1",
                        "base_url": "",
                    }
                ],
            }
        )
        calls: list[str] = []
        outcomes = {
            "sk-1": [_FakePermissionDeniedError("permission denied")],
            "sk-2": [_FakePermissionDeniedError("permission denied")],
        }

        def factory(config, *, api_key_override=None):
            _ = config
            return _FakeModel(str(api_key_override or ""), outcomes, calls)

        model = RotatingChatModel(
            config=self._config(profile_path),
            profile_id="gpt-4o",
            profile_store_path=profile_path,
            llm_factory=factory,
        )

        with self.assertRaises(ApiKeyRotationExhaustedError):
            asyncio.run(model.ainvoke("hello"))

        self.assertEqual(calls, ["sk-1", "sk-2"])


if __name__ == "__main__":
    unittest.main()
