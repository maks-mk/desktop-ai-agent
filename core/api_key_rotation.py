from __future__ import annotations

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


def classify_api_key_error(error: Exception) -> str | None:
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403}:
        return "auth"
    if status_code == 429:
        return "rate_limit"

    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status in {401, 403}:
        return "auth"
    if response_status == 429:
        return "rate_limit"

    text = _normalized_error_text(error)
    auth_markers = (
        "invalid_api_key",
        "incorrect api key",
        "authentication failed",
        "unauthorized",
        "forbidden",
        "permission denied",
        "error code: 401",
        "error code: 403",
    )
    rate_limit_markers = (
        "429",
        "too many requests",
        "rate limit",
        "quota exceeded",
        "insufficient_quota",
        "resource_exhausted",
    )
    if any(marker in text for marker in auth_markers):
        return "auth"
    if any(marker in text for marker in rate_limit_markers):
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
        self._logger.error(
            f"All {total} keys exhausted for profile '{self._profile_id}' "
            f"[{tried_summary}]. Last error: {last_error}"
        )
        raise ApiKeyRotationExhaustedError(
            f"Все API-ключи для '{model_label}' исчерпаны за один полный цикл. "
            "Проверьте лимиты и действительность ключей, либо повторите запрос позже."
        ) from last_error

    def _build_model(self, api_key: str) -> Any:
        model = self._llm_factory(self._config, api_key_override=api_key)
        if self._bound_tools:
            model = model.bind_tools(self._bound_tools)
        return model

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
