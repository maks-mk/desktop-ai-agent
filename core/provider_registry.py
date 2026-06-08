from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
import json

from core.reasoning_debug import debug_event

MATCH_TYPES = {"exact", "suffix"}
VALIDATION_MODES = {"strict", "map", "passthrough"}
MODEL_MATCH_FIELDS = {"exact", "prefix", "contains"}


class ProviderValidationError(ValueError):
    def __init__(self, provider_id: str, given_value: str, allowed_values: list[str]):
        self.provider_id = provider_id
        self.given_value = given_value
        self.allowed_values = allowed_values
        super().__init__(
            f'Provider "{provider_id}": invalid reasoning value "{given_value}". '
            f"Allowed: {', '.join(allowed_values)}"
        )


class PathConflictError(ValueError):
    def __init__(self, path: str, conflict_at: str):
        self.path = path
        self.conflict_at = conflict_at
        super().__init__(f'Cannot set nested path "{path}": conflict at "{conflict_at}" (not an object)')


class RegistryValidationError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid provider registry: {reason}")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _ensure_str_list(value: Any, *, field: str, provider_id: str = "") -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        owner = f' for provider "{provider_id}"' if provider_id else ""
        raise RegistryValidationError(f"{field} must be a non-empty string array{owner}")
    return [item.strip().lower() for item in value]


def _hostname_from_base_url(base_url: str) -> str:
    normalized = _clean_text(base_url).lower()
    if not normalized:
        return "api.openai.com"
    if "://" not in normalized:
        normalized = f"https://{normalized}"
    parsed = urlparse(normalized)
    return str(parsed.hostname or "").strip().lower()


def set_nested(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = [_clean_text(part) for part in _clean_text(path).split(".")]
    if not parts or any(not part for part in parts):
        raise RegistryValidationError(f'invalid path "{path}"')

    current = obj
    traversed: list[str] = []
    for part in parts[:-1]:
        traversed.append(part)
        existing = current.get(part)
        if existing is None:
            child: dict[str, Any] = {}
            current[part] = child
            current = child
            continue
        if not isinstance(existing, dict):
            raise PathConflictError(path, ".".join(traversed))
        current = existing
    current[parts[-1]] = value


class ProviderRegistry:
    def __init__(self, registry: Mapping[str, Any]):
        self._registry = dict(registry)
        self._providers = self._validate_registry(self._registry)

    @classmethod
    def from_path(cls, path: str | Path) -> "ProviderRegistry":
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise RegistryValidationError(f"cannot read registry file: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RegistryValidationError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RegistryValidationError("registry root must be an object")
        return cls(payload)

    def match(self, base_url: str | None) -> dict[str, Any] | None:
        hostname = _hostname_from_base_url(base_url or "")
        if not hostname:
            debug_event("provider_registry_match_skipped", base_url=base_url, hostname="")
            return None

        providers = sorted(
            (provider for provider in self._providers if bool(provider.get("enabled", True))),
            key=lambda provider: int(provider.get("priority", 0)),
            reverse=True,
        )
        for provider in providers:
            match_type = provider["match_type"]
            for pattern in provider["match"]:
                if match_type == "exact" and hostname == pattern:
                    debug_event(
                        "provider_registry_matched",
                        base_url=base_url,
                        hostname=hostname,
                        provider_id=provider.get("id"),
                        match_type=match_type,
                        pattern=pattern,
                        priority=provider.get("priority"),
                        supports_reasoning=provider.get("supports_reasoning"),
                    )
                    return dict(provider)
                if match_type == "suffix" and (hostname == pattern or hostname.endswith(f".{pattern}")):
                    debug_event(
                        "provider_registry_matched",
                        base_url=base_url,
                        hostname=hostname,
                        provider_id=provider.get("id"),
                        match_type=match_type,
                        pattern=pattern,
                        priority=provider.get("priority"),
                        supports_reasoning=provider.get("supports_reasoning"),
                    )
                    return dict(provider)
        debug_event("provider_registry_no_match", base_url=base_url, hostname=hostname)
        return None

    @staticmethod
    def _validate_registry(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
        if "schema_version" not in registry:
            raise RegistryValidationError("schema_version is required")
        if "data_version" not in registry:
            raise RegistryValidationError("data_version is required")
        raw_providers = registry.get("providers")
        if not isinstance(raw_providers, list):
            raise RegistryValidationError("providers must be an array")

        seen_ids: set[str] = set()
        providers: list[dict[str, Any]] = []
        for raw in raw_providers:
            if not isinstance(raw, dict):
                raise RegistryValidationError("provider entries must be objects")
            provider = dict(raw)
            provider_id = _clean_text(provider.get("id"))
            if not provider_id:
                raise RegistryValidationError("provider id is required")
            if provider_id in seen_ids:
                raise RegistryValidationError(f'duplicate provider id "{provider_id}"')
            seen_ids.add(provider_id)
            provider["id"] = provider_id
            provider["enabled"] = bool(provider.get("enabled", True))
            try:
                provider["priority"] = int(provider.get("priority", 0))
            except (TypeError, ValueError) as exc:
                raise RegistryValidationError(f'priority must be an integer for provider "{provider_id}"') from exc
            provider["match"] = _ensure_str_list(provider.get("match"), field="match", provider_id=provider_id)
            match_type = _clean_text(provider.get("match_type"))
            if match_type not in MATCH_TYPES:
                raise RegistryValidationError(f'invalid match_type for provider "{provider_id}"')
            provider["match_type"] = match_type
            provider["supports_reasoning"] = bool(provider.get("supports_reasoning", False))
            validation = _clean_text(provider.get("validation"))
            if validation not in VALIDATION_MODES:
                raise RegistryValidationError(f'invalid validation for provider "{provider_id}"')
            provider["validation"] = validation
            if provider["supports_reasoning"]:
                _validate_reasoning_config(provider_id, provider)
                _validate_model_match_config(provider_id, provider)
            providers.append(provider)
        return providers


def _validate_model_match_config(provider_id: str, provider: dict[str, Any]) -> None:
    model_match = provider.get("model_match")
    if model_match is None:
        return
    if not isinstance(model_match, dict):
        raise RegistryValidationError(f'model_match must be an object for provider "{provider_id}"')
    unknown_fields = set(model_match) - MODEL_MATCH_FIELDS
    if unknown_fields:
        raise RegistryValidationError(
            f'unsupported model_match field(s) for provider "{provider_id}": {", ".join(sorted(unknown_fields))}'
        )
    if not any(field in model_match for field in MODEL_MATCH_FIELDS):
        raise RegistryValidationError(f'model_match must define at least one rule for provider "{provider_id}"')
    for field in MODEL_MATCH_FIELDS:
        if field in model_match:
            _ensure_str_list(model_match.get(field), field=f"model_match.{field}", provider_id=provider_id)


def _validate_reasoning_config(provider_id: str, provider: dict[str, Any]) -> None:
    reasoning = provider.get("reasoning")
    if not isinstance(reasoning, dict):
        raise RegistryValidationError(f'reasoning is required for provider "{provider_id}"')
    path = _clean_text(reasoning.get("path"))
    if not path or path.startswith(".") or path.endswith("."):
        raise RegistryValidationError(f'invalid reasoning.path for provider "{provider_id}"')

    validation = provider["validation"]
    if validation in {"strict", "map"}:
        _ensure_str_list(reasoning.get("allowed_values"), field="reasoning.allowed_values", provider_id=provider_id)
    if validation == "map" and not isinstance(reasoning.get("value_map"), dict):
        raise RegistryValidationError(f'reasoning.value_map is required for provider "{provider_id}"')
    extra_fields = reasoning.get("extra_fields", {})
    if extra_fields is not None and not isinstance(extra_fields, dict):
        raise RegistryValidationError(f'reasoning.extra_fields must be an object for provider "{provider_id}"')


def _resolve_reasoning_value(config: Mapping[str, Any], effort_value: str) -> str:
    provider_id = _clean_text(config.get("id"))
    reasoning = config.get("reasoning") if isinstance(config.get("reasoning"), dict) else {}
    validation = _clean_text(config.get("validation"))
    value = _clean_text(effort_value).lower()

    if validation == "map":
        value_map = reasoning.get("value_map") if isinstance(reasoning, dict) else {}
        if isinstance(value_map, dict):
            value = _clean_text(value_map.get(value, value)).lower()
        validation = "strict"
    if validation == "strict":
        allowed_values = _ensure_str_list(reasoning.get("allowed_values"), field="reasoning.allowed_values", provider_id=provider_id)
        if value not in allowed_values:
            raise ProviderValidationError(provider_id, effort_value, allowed_values)
    return value


def provider_supports_reasoning_for_model(config: Mapping[str, Any] | None, model_name: str | None) -> bool:
    if config is None or not bool(config.get("supports_reasoning", False)):
        return False
    model_match = config.get("model_match")
    if model_match is None:
        return True
    if not isinstance(model_match, Mapping):
        return False

    normalized = _clean_text(model_name).lower()
    if not normalized:
        return False
    exact = [_clean_text(value).lower() for value in model_match.get("exact", []) if _clean_text(value)]
    if normalized in exact:
        return True
    prefixes = [_clean_text(value).lower() for value in model_match.get("prefix", []) if _clean_text(value)]
    if any(normalized.startswith(prefix) for prefix in prefixes):
        return True
    markers = [_clean_text(value).lower() for value in model_match.get("contains", []) if _clean_text(value)]
    return any(marker in normalized for marker in markers)


def build_reasoning_kwargs(
    kwargs: dict[str, Any],
    config: Mapping[str, Any] | None,
    effort_value: str,
) -> dict[str, Any]:
    if config is None or not bool(config.get("supports_reasoning", False)):
        debug_event(
            "reasoning_kwargs_skipped",
            provider_id=config.get("id") if isinstance(config, Mapping) else None,
            supports_reasoning=config.get("supports_reasoning") if isinstance(config, Mapping) else None,
            reason="no_provider_or_unsupported",
        )
        return kwargs

    value = _clean_text(effort_value).lower()
    if value == "none":
        debug_event("reasoning_kwargs_skipped", provider_id=config.get("id"), reason="effort_none")
        return kwargs

    reasoning = config.get("reasoning") if isinstance(config.get("reasoning"), dict) else {}
    path = _clean_text(reasoning.get("path"))
    resolved_value = _resolve_reasoning_value(config, value)
    set_nested(kwargs, path, resolved_value)
    applied_paths = [path]
    extra_fields = reasoning.get("extra_fields", {})
    if isinstance(extra_fields, dict):
        for extra_path, extra_value in extra_fields.items():
            set_nested(kwargs, str(extra_path), extra_value)
            applied_paths.append(str(extra_path))
    debug_event(
        "reasoning_kwargs_applied",
        provider_id=config.get("id"),
        input_effort=effort_value,
        resolved_effort=resolved_value,
        paths=applied_paths,
    )
    return kwargs
