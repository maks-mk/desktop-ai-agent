import unittest

from core.provider_registry import (
    PathConflictError,
    ProviderRegistry,
    ProviderValidationError,
    RegistryValidationError,
    build_reasoning_kwargs,
    provider_supports_reasoning_for_model,
    set_nested,
)


def _registry(*providers):
    return {"schema_version": 1, "data_version": 1, "providers": list(providers)}


def _provider(**overrides):
    payload = {
        "id": "openrouter",
        "enabled": True,
        "priority": 90,
        "match": ["openrouter.ai"],
        "match_type": "suffix",
        "supports_reasoning": True,
        "validation": "map",
        "reasoning": {
            "path": "extra_body.reasoning.effort",
            "allowed_values": ["low", "medium", "high"],
            "value_map": {"minimal": "low", "xhigh": "high"},
        },
    }
    payload.update(overrides)
    return payload


class ProviderRegistryTests(unittest.TestCase):
    def test_match_uses_hostname_only_and_accepts_missing_scheme(self):
        registry = ProviderRegistry(
            _registry(
                _provider(
                    id="openai",
                    match=["api.openai.com"],
                    match_type="exact",
                    reasoning={"path": "reasoning.effort", "allowed_values": ["low"], "value_map": {}},
                )
            )
        )

        self.assertEqual(registry.match("api.openai.com/v1?foo=bar")["id"], "openai")

    def test_suffix_match_requires_label_boundary(self):
        registry = ProviderRegistry(_registry(_provider()))

        self.assertEqual(registry.match("https://api.openrouter.ai/api/v1")["id"], "openrouter")
        self.assertIsNone(registry.match("https://api.evil-openrouter.ai/v1"))

    def test_higher_priority_wins_and_disabled_provider_is_skipped(self):
        registry = ProviderRegistry(
            _registry(
                _provider(id="disabled", priority=1000, enabled=False, match=["openrouter.ai"]),
                _provider(id="low", priority=10, match=["api.openrouter.ai"]),
                _provider(id="high", priority=20, match=["openrouter.ai"]),
            )
        )

        self.assertEqual(registry.match("https://api.openrouter.ai/v1")["id"], "high")

    def test_unknown_provider_returns_none_and_payload_is_unchanged(self):
        payload = {"model": "x"}

        self.assertIsNone(ProviderRegistry(_registry(_provider())).match("https://unknown.example/v1"))
        self.assertIs(build_reasoning_kwargs(payload, None, "high"), payload)
        self.assertEqual(payload, {"model": "x"})

    def test_provider_without_model_match_supports_reasoning_for_any_model(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/api/v1")

        self.assertTrue(provider_supports_reasoning_for_model(config, "any-new-provider-model"))

    def test_provider_model_match_limits_strict_provider_models(self):
        config = ProviderRegistry(
            _registry(
                _provider(
                    id="openai",
                    match=["api.openai.com"],
                    match_type="exact",
                    model_match={"prefix": ["gpt-5", "o1", "o3", "o4"]},
                    reasoning={"path": "reasoning.effort", "allowed_values": ["medium"], "value_map": {}},
                )
            )
        ).match("https://api.openai.com/v1")

        self.assertTrue(provider_supports_reasoning_for_model(config, "gpt-5-mini"))
        self.assertFalse(provider_supports_reasoning_for_model(config, "gpt-4o"))

    def test_build_reasoning_kwargs_maps_value_and_sets_nested_path(self):
        config = ProviderRegistry(_registry(_provider())).match("https://openrouter.ai/api/v1")
        payload = {"model": "x"}

        build_reasoning_kwargs(payload, config, "xhigh")

        self.assertEqual(payload, {"model": "x", "extra_body": {"reasoning": {"effort": "high"}}})

    def test_baai_registry_entry_uses_top_level_reasoning_effort(self):
        config = ProviderRegistry(
            _registry(
                _provider(
                    id="baai",
                    priority=70,
                    match=["api.b.ai"],
                    match_type="exact",
                    reasoning={
                        "path": "reasoning_effort",
                        "allowed_values": ["low", "medium", "high"],
                        "value_map": {"minimal": "low", "xhigh": "high"},
                    },
                )
            )
        ).match("https://api.b.ai/v1")
        payload = {"model": "minimax-m3"}

        build_reasoning_kwargs(payload, config, "xhigh")

        self.assertEqual(payload, {"model": "minimax-m3", "reasoning_effort": "high"})

    def test_build_reasoning_kwargs_supports_extra_fields(self):
        config = ProviderRegistry(
            _registry(
                _provider(
                    reasoning={
                        "path": "reasoning.effort",
                        "allowed_values": ["low", "medium", "high"],
                        "value_map": {"xhigh": "high"},
                        "extra_fields": {"reasoning.summary": "auto"},
                    }
                )
            )
        ).match("https://openrouter.ai/api/v1")
        payload = {}

        build_reasoning_kwargs(payload, config, "medium")

        self.assertEqual(payload, {"reasoning": {"effort": "medium", "summary": "auto"}})

    def test_strict_validation_rejects_invalid_value(self):
        config = _provider(
            validation="strict",
            reasoning={"path": "reasoning_effort", "allowed_values": ["low", "medium", "high"]},
        )

        with self.assertRaises(ProviderValidationError):
            build_reasoning_kwargs({}, config, "xhigh")

    def test_map_validation_rejects_unmapped_invalid_value(self):
        config = _provider()

        with self.assertRaises(ProviderValidationError):
            build_reasoning_kwargs({}, config, "extreme")

    def test_passthrough_validation_accepts_any_value(self):
        config = _provider(
            validation="passthrough",
            reasoning={"path": "reasoning_effort"},
        )
        payload = {}

        build_reasoning_kwargs(payload, config, "extreme")

        self.assertEqual(payload, {"reasoning_effort": "extreme"})

    def test_effort_none_skips_reasoning_payload(self):
        payload = {"model": "x"}

        build_reasoning_kwargs(payload, _provider(), "none")

        self.assertEqual(payload, {"model": "x"})

    def test_set_nested_extends_existing_objects(self):
        payload = {"reasoning": {"tokens": 1000}}

        set_nested(payload, "reasoning.effort", "high")

        self.assertEqual(payload, {"reasoning": {"tokens": 1000, "effort": "high"}})

    def test_set_nested_raises_on_path_conflict(self):
        with self.assertRaises(PathConflictError):
            set_nested({"reasoning": "bad"}, "reasoning.effort", "high")

    def test_registry_validation_requires_versions(self):
        with self.assertRaises(RegistryValidationError):
            ProviderRegistry({"providers": []})

    def test_registry_validation_rejects_duplicate_ids(self):
        with self.assertRaises(RegistryValidationError):
            ProviderRegistry(_registry(_provider(id="same"), _provider(id="same")))

    def test_registry_validation_rejects_invalid_provider_shapes(self):
        invalid_cases = [
            _provider(match=[]),
            _provider(match_type="contains"),
            _provider(supports_reasoning=True, reasoning=None),
            _provider(reasoning={"path": ".bad", "allowed_values": ["low"], "value_map": {}}),
            _provider(validation="bad"),
            _provider(validation="strict", reasoning={"path": "reasoning_effort", "allowed_values": []}),
            _provider(validation="map", reasoning={"path": "reasoning_effort", "allowed_values": ["low"]}),
            _provider(model_match={}),
            _provider(model_match={"regex": ["bad"]}),
            _provider(model_match={"prefix": []}),
        ]

        for provider in invalid_cases:
            with self.subTest(provider=provider):
                with self.assertRaises(RegistryValidationError):
                    ProviderRegistry(_registry(provider))


if __name__ == "__main__":
    unittest.main()
