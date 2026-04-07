import shutil
import unittest
from pathlib import Path

from core.model_profiles import (
    ModelProfileStore,
    bootstrap_profiles_from_env,
    find_active_profile,
    merge_profiles_with_env,
    normalize_profiles_payload,
)


class ModelProfilesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = Path.cwd() / ".tmp_tests" / f"model_profiles_{id(self)}"
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self._tmpdir, ignore_errors=True))

    def test_bootstrap_uses_generic_env_first(self):
        payload = bootstrap_profiles_from_env(
            {
                "PROVIDER": "openai",
                "MODEL": "meta-llama/llama-3-70b",
                "API_KEY": "generic-key",
                "BASE_URL": "http://localhost:8000/v1",
                "OPENAI_MODEL": "gpt-4o",
            }
        )
        self.assertEqual(payload["active_profile"], "llama-3-70b")
        self.assertEqual(len(payload["profiles"]), 1)
        self.assertEqual(payload["profiles"][0]["provider"], "openai")
        self.assertEqual(payload["profiles"][0]["model"], "meta-llama/llama-3-70b")
        self.assertEqual(payload["profiles"][0]["api_key"], "generic-key")
        self.assertEqual(payload["profiles"][0]["base_url"], "http://localhost:8000/v1")

    def test_bootstrap_legacy_openai(self):
        payload = bootstrap_profiles_from_env(
            {
                "PROVIDER": "openai",
                "OPENAI_MODEL": "gpt-4o",
                "OPENAI_API_KEY": "sk-123",
                "OPENAI_BASE_URL": "https://api.openai.com/v1",
            }
        )
        self.assertEqual(payload["active_profile"], "gpt-4o")
        self.assertEqual(payload["profiles"][0]["provider"], "openai")
        self.assertEqual(payload["profiles"][0]["model"], "gpt-4o")

    def test_normalization_skips_invalid_and_uniquifies_ids(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "custom",
                "profiles": [
                    {"id": "custom", "provider": "openai", "model": "gpt-4o", "api_key": "", "base_url": ""},
                    {"id": "custom", "provider": "openai", "model": "gpt-4o-mini", "api_key": "", "base_url": ""},
                    {"id": "", "provider": "gemini", "model": "gemini-1.5-flash", "api_key": "", "base_url": ""},
                    {"id": "x", "provider": "invalid", "model": "x", "api_key": "", "base_url": ""},
                    {"id": "z", "provider": "openai", "model": "", "api_key": "", "base_url": ""},
                ],
            }
        )
        self.assertEqual(len(payload["profiles"]), 3)
        ids = [item["id"] for item in payload["profiles"]]
        self.assertEqual(ids[0], "custom")
        self.assertEqual(ids[1], "custom-1")
        self.assertEqual(ids[2], "gemini-1-5-flash")
        self.assertEqual(payload["active_profile"], "custom")

    def test_normalization_deduplicates_identical_profiles(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "https://api.openai.com/v1",
                    },
                    {
                        "id": "gpt-4o-copy",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "https://api.openai.com/v1",
                    },
                ],
            }
        )
        self.assertEqual(len(payload["profiles"]), 1)
        self.assertEqual(payload["profiles"][0]["id"], "gpt-4o")

    def test_normalization_preserves_manual_image_support_flag(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gemini-1-5-flash",
                "profiles": [
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "supports_image_input": True,
                    }
                ],
            }
        )

        self.assertTrue(payload["profiles"][0]["supports_image_input"])
        active = find_active_profile(payload)
        self.assertIsNotNone(active)
        self.assertTrue(active["supports_image_input"])
        self.assertTrue(active["enabled"])

    def test_normalization_keeps_profiles_with_different_image_support_flags(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "supports_image_input": False,
                    },
                    {
                        "id": "gpt-4o-img",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "supports_image_input": True,
                    },
                ],
            }
        )

        self.assertEqual(len(payload["profiles"]), 2)

    def test_normalization_drops_disabled_profile_from_active_selection(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "enabled": False,
                    },
                    {
                        "id": "gemini-2-5-pro",
                        "provider": "gemini",
                        "model": "gemini-2.5-pro",
                        "api_key": "gm-demo",
                        "base_url": "",
                        "enabled": True,
                    },
                ],
            }
        )

        self.assertEqual(payload["active_profile"], "gemini-2-5-pro")

    def test_normalization_allows_all_profiles_to_be_temporarily_disabled(self):
        payload = normalize_profiles_payload(
            {
                "active_profile": "gpt-4o",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "sk-demo",
                        "base_url": "",
                        "enabled": False,
                    }
                ],
            }
        )

        self.assertIsNone(payload["active_profile"])

    def test_store_load_existing_merges_new_env_model(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        first = store.load_or_initialize(
            {
                "PROVIDER": "openai",
                "MODEL": "gpt-4o",
                "API_KEY": "first",
            }
        )
        second = store.load_or_initialize(
            {
                "PROVIDER": "gemini",
                "MODEL": "gemini-1.5-flash",
                "API_KEY": "second",
            }
        )
        self.assertEqual(len(first["profiles"]), 1)
        self.assertEqual(len(second["profiles"]), 2)
        self.assertEqual(find_active_profile(second)["provider"], "openai")
        self.assertTrue(any(item.get("provider") == "gemini" for item in second["profiles"]))

    def test_store_existing_unconfigured_uses_env_fallback(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        store.save({"active_profile": None, "profiles": []})
        loaded = store.load_or_initialize(
            {
                "PROVIDER": "openai",
                "OPENAI_MODEL": "gpt-4o",
                "OPENAI_API_KEY": "sk-env",
            }
        )
        self.assertEqual(loaded["active_profile"], "gpt-4o")
        self.assertEqual(len(loaded["profiles"]), 1)

    def test_store_first_launch_bootstrap_from_env_does_not_duplicate_on_restart(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        env_payload = {
            "PROVIDER": "openai",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_API_KEY": "sk-env",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }
        first = store.load_or_initialize(env_payload)
        second = store.load_or_initialize(env_payload)

        self.assertEqual(first["active_profile"], "gpt-4o")
        self.assertEqual(len(first["profiles"]), 1)
        self.assertEqual(first, second)

    def test_merge_profiles_with_env_appends_missing_env_model_once(self):
        existing = {
            "active_profile": "gemini-1-5-flash",
            "profiles": [
                {
                    "id": "gemini-1-5-flash",
                    "provider": "gemini",
                    "model": "gemini-1.5-flash",
                    "api_key": "gm-existing",
                    "base_url": "",
                }
            ],
        }
        env_payload = {
            "active_profile": "gpt-oss-120b",
            "profiles": [
                {
                    "id": "gpt-oss-120b",
                    "provider": "openai",
                    "model": "openai/gpt-oss-120b",
                    "api_key": "sk-env",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
        merged_once = merge_profiles_with_env(existing, env_payload)
        merged_twice = merge_profiles_with_env(merged_once, env_payload)

        self.assertEqual(merged_once["active_profile"], "gemini-1-5-flash")
        self.assertEqual(len(merged_once["profiles"]), 2)
        self.assertEqual(len(merged_twice["profiles"]), 2)
        self.assertTrue(
            any(
                item.get("provider") == "openai" and item.get("model") == "openai/gpt-oss-120b"
                for item in merged_twice["profiles"]
            )
        )

    def test_store_load_existing_merges_new_env_model_without_duplication(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        store.save(
            {
                "active_profile": "gemini-1-5-flash",
                "profiles": [
                    {
                        "id": "gemini-1-5-flash",
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "api_key": "gm-existing",
                        "base_url": "",
                    }
                ],
            }
        )
        env_payload = {
            "PROVIDER": "openai",
            "MODEL": "openai/gpt-oss-120b",
            "API_KEY": "sk-env",
            "BASE_URL": "https://api.openai.com/v1",
        }
        first = store.load_or_initialize(env_payload)
        second = store.load_or_initialize(env_payload)

        self.assertEqual(first["active_profile"], "gemini-1-5-flash")
        self.assertEqual(len(first["profiles"]), 2)
        self.assertEqual(len(second["profiles"]), 2)

    def test_store_save_normalizes_active_profile(self):
        store = ModelProfileStore(self._tmpdir / "config.json")
        saved = store.save(
            {
                "active_profile": "missing-id",
                "profiles": [
                    {
                        "id": "gpt-4o",
                        "provider": "openai",
                        "model": "gpt-4o",
                        "api_key": "",
                        "base_url": "",
                    }
                ],
            }
        )
        self.assertEqual(saved["active_profile"], "gpt-4o")


if __name__ == "__main__":
    unittest.main()
