from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from core.logging_config import SensitiveDataFilter
from core.model_profiles import ModelProfileStore


class ApiKeyRotationError(RuntimeError):
    """Base error for model-profile API key rotation failures."""


class ApiKeyRotationExhaustedError(ApiKeyRotationError):
    """Raised when every key in the pool has been tried in one full circle without success."""


def _normalized_error_text(error: Exception) -> str:
    return " ".join(str(error).lower().split())


def _extract_status_code(error: Exception) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _extract_provider_code_text(error: Exception) -> str:
    code = getattr(error, "code", None)
    if callable(code):
        try:
            code = code()
        except TypeError:
            code = None
    status = getattr(error, "status", None)
    parts = [type(error).__name__, type(error).__module__, code, status]
    return " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())


def _format_error_for_user(error: Exception | None, error_kind: str | None = None) -> str:
    if error is None:
        return "неизвестная ошибка"
    parts = [type(error).__name__]
    status_code = _extract_status_code(error)
    if error_kind:
        parts.append(f"тип: {error_kind}")
    if status_code is not None:
        parts.append(f"HTTP {status_code}")
    message = " ".join(str(error or "").split())
    if message:
        parts.append(message)
    return " | ".join(parts)


def classify_api_key_error(error: Exception) -> str | None:
    status_code = _extract_status_code(error)
    if status_code in {401, 403}:
        return "auth"
    if status_code == 402:
        return "billing"
    if status_code == 429:
        return "rate_limit"

    text = _normalized_error_text(error)
    provider_code_text = _extract_provider_code_text(error)
    combined = f"{provider_code_text} {text}".strip()
    auth_markers = (
        "authenticationerror",
        "permissiondenied",
        "unauthenticated",
        "forbidden",
        "invalid_api_key",
        "incorrect api key",
        "authentication failed",
        "unauthorized",
        "forbidden",
        "permission denied",
        "permission_denied",
        "invalid api key",
        "error code: 401",
        "error code: 403",
    )
    rate_limit_markers = (
        "ratelimiterror",
        "toomanyrequests",
        "resourceexhausted",
        "resource_exhausted",
        "429",
        "too many requests",
        "rate limit",
        "quota exceeded",
        "insufficient_quota",
        "resource_exhausted",
    )
    billing_markers = (
        "402",
        "payment required",
        "insufficient_balance",
        "insufficient balance",
        "billing",
        "credit balance",
    )
    if any(marker in combined for marker in auth_markers):
        return "auth"
    if any(marker in combined for marker in billing_markers):
        return "billing"
    if any(marker in combined for marker in rate_limit_markers):
        return "rate_limit"
    return None


def _setup_rotation_logger(config: Any) -> logging.Logger:
    """Ensure the api_key_rotation logger writes to a dedicated log file."""
    logger = logging.getLogger("agent.api_key_rotation")
    # Avoid adding duplicate handlers on repeated instantiations
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    log_file = getattr(config, "log_file", None)
    if log_file is None:
        return logger

    rotation_log = Path(log_file).parent / "api_key_rotation.log"
    try:
        rotation_log.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(rotation_log), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
    except Exception:
        pass
    return logger


def _mask_key(key: str, tail: int = 5) -> str:
    """Return key with only the first chars and last *tail* visible, middle replaced with ***.
    Short keys (<= tail+2) are returned as-is with *** appended."""
    if len(key) <= tail + 2:
        return key[:2] + "***"
    return key[: -tail].rstrip() + "***" + key[-tail:]


async def _close_model_clients(model: Any) -> None:
    if model is None:
        return
    for target in (getattr(model, "async_client", None), getattr(model, "client", None)):
        if target is None:
            continue
        close_method = getattr(target, "aclose", None) or getattr(target, "close", None)
        if not callable(close_method):
            continue
        result = close_method()
        if inspect.isawaitable(result):
            await result


class RotatingChatModel:
    def __init__(
        self,
        *,
        config: Any,
        profile_id: str,
        profile_store_path: str | Path,
        llm_factory: Callable[..., Any],
        bound_tools: Sequence[Any] | None = None,
    ) -> None:
        self._config = config
        self._profile_id = str(profile_id or "").strip()
        self._profile_store = ModelProfileStore(profile_store_path)
        self._llm_factory = llm_factory
        self._bound_tools = list(bound_tools or [])
        self._logger = _setup_rotation_logger(config)
        self._model_cache: dict[str, Any] = {}
        self._prototype_model = self._build_model(self._initial_api_key())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._prototype_model, name)

    def bind_tools(self, tools: Sequence[Any]) -> "RotatingChatModel":
        return self.__class__(
            config=self._config,
            profile_id=self._profile_id,
            profile_store_path=self._profile_store.path,
            llm_factory=self._llm_factory,
            bound_tools=list(tools),
        )

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        state = self._profile_store.get_api_key_state(self._profile_id)
        api_keys = list(state.get("api_keys") or [])

        # Пул ключей не задан — работаем с дефолтным ключом из конфига.
        if not api_keys:
            return await self._build_model(self._initial_api_key()).ainvoke(input, **kwargs)

        start_index = int(state.get("api_key_index") or 0)
        total = len(api_keys)
        last_error: Exception | None = None

        for offset in range(total):
            current_index = (start_index + offset) % total
            active_key = api_keys[current_index]

            self._logger.debug(
                f"Attempt {offset + 1}/{total}: key #{current_index} "
                f"for profile '{self._profile_id}'"
            )

            try:
                result = await self._build_model(active_key).ainvoke(input, **kwargs)
                # Успех — индекс уже был сдвинут при предыдущих ошибках,
                # следующий вызов начнёт с текущего рабочего ключа.
                return result

            except Exception as exc:
                error_kind = classify_api_key_error(exc)

                if error_kind is None:
                    # Non-key error (network, timeout, model error, etc.) —
                    # propagate immediately; rotation will not help.
                    self._logger.warning(
                        f"Non-key error for profile '{self._profile_id}': {exc}"
                    )
                    raise

                last_error = exc
                next_index = (current_index + 1) % total
                next_key_masked = _mask_key(api_keys[next_index]) if total > 1 else "(none)"
                self._logger.info(
                    f"Key #{current_index} ({_mask_key(active_key)}) returned "
                    f"{error_kind} error for profile '{self._profile_id}' — "
                    f"switching to key #{next_index} ({next_key_masked})"
                )

                # Advance the index in the store without marking the key as invalid.
                self._profile_store.rotate_api_key(self._profile_id, active_key)

        # Full circle completed — no key succeeded.
        model_label = self._model_label()
        tried_summary = ", ".join(
            f"#{i}({_mask_key(k)})" for i, k in enumerate(api_keys)
        )
        last_error_text = _format_error_for_user(
            last_error,
            classify_api_key_error(last_error) if last_error is not None else None,
        )
        self._logger.error(
            f"All {total} keys exhausted for profile '{self._profile_id}' "
            f"[{tried_summary}]. Last error: {last_error_text}"
        )
        raise ApiKeyRotationExhaustedError(
            f"Все API-ключи для '{model_label}' исчерпаны за один полный цикл. "
            f"Последняя ошибка: {last_error_text}. "
            "Проверьте лимиты и действительность ключей, либо повторите запрос позже."
        ) from last_error

    def _build_model(self, api_key: str) -> Any:
        cache_key = str(api_key or "")
        cached_model = self._model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model
        model = self._llm_factory(self._config, api_key_override=api_key)
        if self._bound_tools:
            model = model.bind_tools(self._bound_tools)
        self._model_cache[cache_key] = model
        return model

    async def aclose(self) -> None:
        seen: set[int] = set()
        for model in list(self._model_cache.values()):
            marker = id(model)
            if marker in seen:
                continue
            seen.add(marker)
            await _close_model_clients(model)
        self._model_cache.clear()

    def _initial_api_key(self) -> str:
        if getattr(self._config, "provider", "") == "gemini":
            secret = getattr(self._config, "gemini_api_key", None)
        else:
            secret = getattr(self._config, "openai_api_key", None)
        return secret.get_secret_value() if secret is not None else ""

    def _model_label(self) -> str:
        if getattr(self._config, "provider", "") == "gemini":
            return str(getattr(self._config, "gemini_model", "") or self._profile_id or "model")
        return str(getattr(self._config, "openai_model", "") or self._profile_id or "model")
