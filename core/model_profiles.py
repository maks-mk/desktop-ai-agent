from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

ALLOWED_PROVIDERS = {"openai", "gemini"}
_ID_ALLOWED_RE = re.compile(r"[^a-z0-9_-]+")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_provider(value: Any) -> str:
    normalized = _clean_text(value).lower()
    return normalized if normalized in ALLOWED_PROVIDERS else ""


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _clean_text(value).lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(value)


def sanitize_profile_id(value: Any) -> str:
    raw = _clean_text(value)
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    normalized = _ID_ALLOWED_RE.sub("-", raw.lower().replace(" ", "-"))
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
    return normalized


def _ensure_unique_id(base_id: str, used_ids: set[str]) -> str:
    base = sanitize_profile_id(base_id) or "profile"
    candidate = base
    suffix = 1
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def generate_profile_id(model_name: Any, used_ids: set[str]) -> str:
    model_text = _clean_text(model_name)
    base = model_text.rsplit("/", 1)[-1] if "/" in model_text else model_text
    return _ensure_unique_id(base or "profile", used_ids)


def normalize_profiles_payload(payload: Any) -> dict[str, Any]:
    raw_payload = payload if isinstance(payload, dict) else {}
    raw_profiles = raw_payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raw_profiles = []

    used_ids: set[str] = set()
    seen_profiles: set[tuple[str, str, str, str, bool, bool]] = set()
    profiles: list[dict[str, Any]] = []
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            continue
        provider = _normalize_provider(raw.get("provider"))
        model_name = _clean_text(raw.get("model"))
        if not provider or not model_name:
            continue
        api_key = _clean_text(raw.get("api_key"))
        base_url = _clean_text(raw.get("base_url")) if provider == "openai" else ""
        supports_image_input = _normalize_bool(raw.get("supports_image_input"))
        enabled = _normalize_bool(raw.get("enabled")) if "enabled" in raw else True

        # Deduplicate exact same profile payloads to keep env bootstrap/import idempotent.
        profile_fingerprint = (provider, model_name, api_key, base_url, supports_image_input, enabled)
        if profile_fingerprint in seen_profiles:
            continue
        seen_profiles.add(profile_fingerprint)

        requested_id = sanitize_profile_id(raw.get("id"))
        profile_id = _ensure_unique_id(requested_id or model_name, used_ids)
        profiles.append(
            {
                "id": profile_id,
                "provider": provider,
                "model": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "supports_image_input": supports_image_input,
                "enabled": enabled,
            }
        )

    active_raw = raw_payload.get("active_profile")
    active_profile = _clean_text(active_raw) if active_raw is not None else ""
    enabled_ids = [item["id"] for item in profiles if bool(item.get("enabled", True))]
    if active_profile not in enabled_ids:
        active_profile = enabled_ids[0] if enabled_ids else ""

    return {
        "active_profile": active_profile or None,
        "profiles": profiles,
    }


def bootstrap_profiles_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = env or os.environ
    provider_hint = _normalize_provider(source.get("PROVIDER", ""))
    generic_model = _clean_text(source.get("MODEL"))

    if generic_model and provider_hint:
        profile_payload = {
            "active_profile": None,
            "profiles": [
                {
                    "id": "",
                    "provider": provider_hint,
                    "model": generic_model,
                    "api_key": _clean_text(source.get("API_KEY")),
                    "base_url": _clean_text(source.get("BASE_URL")),
                }
            ],
        }
        return normalize_profiles_payload(profile_payload)

    openai_model = _clean_text(source.get("OPENAI_MODEL"))
    gemini_model = _clean_text(source.get("GEMINI_MODEL"))

    chosen_provider = ""
    chosen_model = ""
    chosen_api_key = ""
    chosen_base_url = ""

    if provider_hint == "openai" and openai_model:
        chosen_provider = "openai"
        chosen_model = openai_model
        chosen_api_key = _clean_text(source.get("OPENAI_API_KEY"))
        chosen_base_url = _clean_text(source.get("OPENAI_BASE_URL"))
    elif provider_hint == "gemini" and gemini_model:
        chosen_provider = "gemini"
        chosen_model = gemini_model
        chosen_api_key = _clean_text(source.get("GEMINI_API_KEY"))
    elif openai_model:
        chosen_provider = "openai"
        chosen_model = openai_model
        chosen_api_key = _clean_text(source.get("OPENAI_API_KEY"))
        chosen_base_url = _clean_text(source.get("OPENAI_BASE_URL"))
    elif gemini_model:
        chosen_provider = "gemini"
        chosen_model = gemini_model
        chosen_api_key = _clean_text(source.get("GEMINI_API_KEY"))

    if chosen_provider and chosen_model:
        return normalize_profiles_payload(
            {
                "active_profile": None,
                "profiles": [
                    {
                        "id": "",
                        "provider": chosen_provider,
                        "model": chosen_model,
                        "api_key": chosen_api_key,
                        "base_url": chosen_base_url,
                    }
                ],
            }
        )

    return {"active_profile": None, "profiles": []}


def find_active_profile(payload: dict[str, Any]) -> dict[str, Any] | None:
    active_id = _clean_text((payload or {}).get("active_profile"))
    for profile in (payload or {}).get("profiles", []) or []:
        if isinstance(profile, dict) and _clean_text(profile.get("id")) == active_id:
            return {
                "id": _clean_text(profile.get("id")),
                "provider": _normalize_provider(profile.get("provider")),
                "model": _clean_text(profile.get("model")),
                "api_key": _clean_text(profile.get("api_key")),
                "base_url": _clean_text(profile.get("base_url")),
                "supports_image_input": _normalize_bool(profile.get("supports_image_input")),
                "enabled": _normalize_bool(profile.get("enabled")) if "enabled" in profile else True,
            }
    return None


def _profile_merge_key(profile: Mapping[str, Any]) -> tuple[str, str, str]:
    provider = _normalize_provider(profile.get("provider"))
    model_name = _clean_text(profile.get("model"))
    base_url = _clean_text(profile.get("base_url")) if provider == "openai" else ""
    return provider, model_name, base_url


def merge_profiles_with_env(existing_payload: Any, env_payload: Any) -> dict[str, Any]:
    current = normalize_profiles_payload(existing_payload)
    env_normalized = normalize_profiles_payload(env_payload)
    env_profiles = [item for item in env_normalized.get("profiles", []) if isinstance(item, dict)]
    if not env_profiles:
        return current

    merged_profiles: list[dict[str, str]] = [dict(item) for item in current.get("profiles", [])]

    for env_profile in env_profiles:
        env_key = _profile_merge_key(env_profile)
        if not all(env_key[:2]):
            continue
        match_index = next(
            (
                index
                for index, existing in enumerate(merged_profiles)
                if _profile_merge_key(existing) == env_key
            ),
            None,
        )
        if match_index is None:
            merged_profiles.append(dict(env_profile))
            continue

        # Keep user-edited values, but backfill missing credentials from env.
        existing_profile = dict(merged_profiles[match_index])
        env_api_key = _clean_text(env_profile.get("api_key"))
        if env_api_key and not _clean_text(existing_profile.get("api_key")):
            existing_profile["api_key"] = env_api_key
        if env_key[0] == "openai":
            env_base_url = _clean_text(env_profile.get("base_url"))
            if env_base_url and not _clean_text(existing_profile.get("base_url")):
                existing_profile["base_url"] = env_base_url
        merged_profiles[match_index] = existing_profile

    active_profile = _clean_text(current.get("active_profile"))
    if not active_profile:
        active_profile = _clean_text(env_normalized.get("active_profile"))

    return normalize_profiles_payload(
        {
            "active_profile": active_profile or None,
            "profiles": merged_profiles,
        }
    )


class ModelProfileStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize(payload: Any) -> dict[str, Any]:
        return normalize_profiles_payload(payload)

    def _read_raw_payload(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return raw if isinstance(raw, dict) else None

    def load_or_initialize(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        env_bootstrap = bootstrap_profiles_from_env(env)
        raw = self._read_raw_payload()
        if raw is None:
            return self.save(env_bootstrap)
        normalized = self.normalize(raw)
        merged = merge_profiles_with_env(normalized, env_bootstrap)
        if merged != normalized:
            return self.save(merged)
        if normalized != raw:
            return self.save(normalized)
        return normalized

    def save(self, payload: Any) -> dict[str, Any]:
        normalized = self.normalize(payload)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        serialized = json.dumps(normalized, ensure_ascii=False, indent=2)
        tmp_path.write_text(serialized, encoding="utf-8")
        try:
            tmp_path.replace(self.path)
        except PermissionError:
            # Some Windows environments temporarily block atomic replace;
            # fall back to direct write instead of failing profile updates.
            self.path.write_text(serialized, encoding="utf-8")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return normalized
