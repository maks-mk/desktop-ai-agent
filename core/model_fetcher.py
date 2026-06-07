from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol
import json

import httpx

ALLOWED_FAMILIES = ("gemini", "gemma")
EXCLUDE_KEYWORDS = (
    "embed",
    "embedding",
    "audio",
    "tts",
    "speech",
    "image",
    "imagen",
    "retrieval",
    "aqa",
)

_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    family: str
    supports_image_input: bool


class ModelFetcher(Protocol):
    async def fetch(self, api_key: str, base_url: str = "") -> list[ModelEntry]:
        ...


class FetchError(Exception):
    pass


class AuthError(FetchError):
    pass


class RateLimitError(FetchError):
    pass


class ServerError(FetchError):
    pass


class NetworkError(FetchError):
    pass


class EmptyResultError(FetchError):
    pass


class InvalidResponseError(FetchError):
    pass


def _normalize_model_id(raw_name: Any) -> str:
    return str(raw_name or "").strip().removeprefix("models/")


def _has_excluded_keyword(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(keyword in lowered for keyword in EXCLUDE_KEYWORDS)


def _raise_for_status(response: httpx.Response) -> None:
    status_code = int(response.status_code)
    if status_code < 400:
        return
    if status_code in {401, 403}:
        raise AuthError(f"Authentication failed with status {status_code}.")
    if status_code == 429:
        raise RateLimitError("Rate limit exceeded.")
    if 500 <= status_code <= 599:
        raise ServerError(f"Server failed with status {status_code}.")
    raise FetchError(f"Request failed with status {status_code}.")


def _json_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        content_type = str(response.headers.get("content-type") or "").strip()
        detail = f" status={response.status_code}"
        if content_type:
            detail += f" content-type={content_type}"
        raise InvalidResponseError(f"Models endpoint returned invalid JSON.{detail}") from exc


def _coerce_methods(raw_value: Any) -> tuple[str, ...]:
    if not isinstance(raw_value, list):
        return ()
    return tuple(str(item or "").strip() for item in raw_value if str(item or "").strip())


def _coerce_items(payload: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get(key)
    if not isinstance(raw_items, list):
        return []
    return [item for item in raw_items if isinstance(item, dict)]


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/")


class GeminiModelFetcher:
    async def fetch(self, api_key: str, _base_url: str = "") -> list[ModelEntry]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(_GEMINI_MODELS_URL, params={"key": str(api_key or "").strip()})
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise NetworkError("Failed to reach Gemini models endpoint.") from exc

        _raise_for_status(response)
        payload = _json_payload(response)
        entries: list[ModelEntry] = []
        for item in _coerce_items(payload, "models"):
            model_id = _normalize_model_id(item.get("name"))
            if not model_id:
                continue
            if not any(model_id.startswith(prefix) for prefix in ALLOWED_FAMILIES):
                continue
            if "generateContent" not in _coerce_methods(item.get("supportedGenerationMethods")):
                continue
            if _has_excluded_keyword(model_id):
                continue
            family = next((prefix for prefix in ALLOWED_FAMILIES if model_id.startswith(prefix)), "")
            entries.append(ModelEntry(id=model_id, family=family, supports_image_input=True))

        if not entries:
            raise EmptyResultError("No Gemini models matched the filter.")
        return entries


class OpenAICompatibleModelFetcher:
    async def fetch(self, api_key: str, base_url: str = "") -> list[ModelEntry]:
        normalized_base_url = _normalize_base_url(base_url)
        if not normalized_base_url:
            raise FetchError("Base URL is required for OpenAI-compatible model discovery.")

        headers = {
            "Authorization": f"Bearer {str(api_key or '').strip()}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{normalized_base_url}/models", headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise NetworkError("Failed to reach OpenAI-compatible models endpoint.") from exc

        _raise_for_status(response)
        payload = _json_payload(response)
        all_entries: list[ModelEntry] = []
        for item in _coerce_items(payload, "data"):
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            all_entries.append(ModelEntry(id=model_id, family="", supports_image_input=False))

        filtered_entries = [entry for entry in all_entries if not _has_excluded_keyword(entry.id)]
        return filtered_entries or all_entries
