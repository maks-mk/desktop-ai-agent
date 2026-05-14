import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional

from core.constants import BASE_DIR


class NoisyLogFilter(logging.Filter):
    BLOCKED_PHRASES = [
        "Key 'additionalProperties' is not supported",
        "Key '$schema' is not supported",
        "AFC is enabled",
        "HTTP Request: POST",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(phrase in message for phrase in self.BLOCKED_PHRASES)


class SensitiveDataFilter(logging.Filter):
    SENSITIVE_FIELD_NAMES = frozenset(
        {
            "api_key",
            "openai_api_key",
            "gemini_api_key",
            "tavily_api_key",
            "authorization",
            "access_token",
            "refresh_token",
            "token",
            "secret",
            "password",
        }
    )
    _STRING_PATTERNS = (
        re.compile(
            r"(?i)\b(?P<label>(?:openai|gemini|tavily)?_?api[_ -]?key|authorization|access[_ -]?token|refresh[_ -]?token|password|secret|token)\b(?P<separator>\s*[:=]\s*)(?P<quote>['\"]?)(?P<value>(?!Bearer\b)[^\s,'\"}\]]+)(?P=quote)"
        ),
        re.compile(r"(?i)(?P<label>\bBearer\s+)(?P<value>[A-Za-z0-9._\-]+)"),
        re.compile(r"(?i)(?P<label>[?&](?:api_key|key|token)=)(?P<value>[^&\s]+)"),
        re.compile(r"(?P<value>sk-[A-Za-z0-9_-]{6,})"),
        re.compile(r"(?P<value>AIza[0-9A-Za-z\-_]{12,})"),
    )
    _RESERVED_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys())

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._sanitize_value(record.msg)
        if record.args:
            record.args = self._sanitize_value(record.args)
        for key, value in list(record.__dict__.items()):
            if key in self._RESERVED_RECORD_ATTRS:
                continue
            record.__dict__[key] = self._sanitize_value(value, key_hint=key)
        return True

    @classmethod
    def _sanitize_value(cls, value, *, key_hint: str = ""):
        normalized_key = str(key_hint or "").strip().lower()
        if isinstance(value, str):
            if normalized_key in cls.SENSITIVE_FIELD_NAMES:
                return cls._mask_secret(value)
            return cls._sanitize_string(value)
        if isinstance(value, dict):
            return {
                item_key: cls._sanitize_value(item_value, key_hint=str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._sanitize_value(item) for item in value)
        if isinstance(value, list):
            return [cls._sanitize_value(item) for item in value]
        if isinstance(value, set):
            return {cls._sanitize_value(item) for item in value}
        return value

    @classmethod
    def _sanitize_string(cls, value: str) -> str:
        sanitized = str(value or "")
        for pattern in cls._STRING_PATTERNS:
            sanitized = pattern.sub(cls._replace_match, sanitized)
        return sanitized

    @classmethod
    def _replace_match(cls, match: re.Match[str]) -> str:
        groups = match.groupdict()
        value = groups.get("value", "")
        masked = cls._mask_secret(value)
        if "label" in groups and "separator" in groups:
            quote = groups.get("quote", "")
            return f"{groups['label']}{groups['separator']}{quote}{masked}{quote}"
        if "label" in groups:
            return f"{groups['label']}{masked}"
        return masked

    @staticmethod
    def _mask_secret(value: str) -> str:
        secret = str(value or "").strip()
        if not secret:
            return secret
        if len(secret) <= 4:
            return "****"
        return f"{secret[:4]}...<redacted>"


def _coerce_log_level(level: int | str | None) -> int:
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    normalized = str(level or "INFO").strip().upper()
    return getattr(logging, normalized, logging.INFO)


def setup_logging(level: int | str | None = None, log_file: str | Path | None = None) -> logging.Logger:
    if level is None:
        level = _coerce_log_level(None)
    else:
        level = _coerce_log_level(level)

    if log_file is None:
        log_file = str(BASE_DIR / os.getenv("LOG_FILE", "logs/agent.log"))
    else:
        log_file = str(log_file)

    handlers: List[logging.Handler] = []

    if level <= logging.DEBUG:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        handlers.append(console_handler)

    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            handlers.append(file_handler)
        except Exception as exc:
            sys.stderr.write(f"Warning: could not create log file: {exc}\n")

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(level=level, handlers=handlers, force=True)

    noise_filter = NoisyLogFilter()
    sensitive_filter = SensitiveDataFilter()
    for handler in handlers:
        handler.addFilter(sensitive_filter)
        handler.addFilter(noise_filter)

    warnings.filterwarnings(
        "ignore",
        message=r".*MALFORMED_RESPONSE is not a valid FinishReason.*",
        category=UserWarning,
        module=r"google\.genai\._common",
    )

    _suppress_library_logs(level)

    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(logging.DEBUG)
    return agent_logger


def _suppress_library_logs(root_level: int) -> None:
    noisy_modules = [
        "langchain_google_genai",
        "google.ai.generativelanguage",
        "google.auth",
        "openai",
        "httpx",
        "httpcore",
        "urllib3",
        "langchain",
        "langchain_core",
        "langgraph",
        "langchain_mcp_adapters",
        "mcp",
        "pydantic",
        "jsonschema",
        "chromadb",
        "hnswlib",
        "sentence_transformers",
        "filelock",
        "grpc",
        "grpc._cython",
        "multipart",
        "markdown_it",
        "markdown_it.rules_block",
        "markdown_it.rules_inline",
    ]

    library_level = logging.WARNING if root_level == logging.DEBUG else logging.ERROR
    for module_name in noisy_modules:
        logging.getLogger(module_name).setLevel(library_level)
