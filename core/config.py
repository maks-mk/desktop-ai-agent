import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import Field, PrivateAttr, SecretStr, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import BASE_DIR

# --- Defaults ---
DEFAULT_MAX_FILE_SIZE = 300 * 1024 * 1024  # 300 MB
DEFAULT_READ_LIMIT = 2000
_SIZE_WITH_UNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)\s*$", re.IGNORECASE)
_SIZE_MULTIPLIERS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000 ** 2,
    "gb": 1000 ** 3,
    "tb": 1000 ** 4,
    "kib": 1024,
    "mib": 1024 ** 2,
    "gib": 1024 ** 3,
    "tib": 1024 ** 4,
}


def _candidate_runtime_dirs() -> list[Path]:
    if getattr(sys, "frozen", False):
        return [BASE_DIR]

    dirs: list[Path] = [BASE_DIR]
    cwd = Path.cwd()
    if cwd not in dirs:
        dirs.append(cwd)
    return dirs


def _existing_path_or_default(filename: str, default_dir: Path = BASE_DIR) -> Path:
    for directory in _candidate_runtime_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return default_dir / filename


def _env_file_candidates() -> tuple[Path, ...]:
    seen: list[Path] = []
    for directory in _candidate_runtime_dirs():
        candidate = directory / ".env"
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def _resolve_runtime_path(value: Union[str, Path], *, base_dir: Path = BASE_DIR) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class AgentConfig(BaseSettings):
    """
    Конфигурация агента, загружаемая из переменных окружения и .env файла.
    """

    model_config = SettingsConfigDict(
        env_file=_env_file_candidates(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Paths
    prompt_path: Path = Field(default_factory=lambda: _existing_path_or_default("prompt.txt"), alias="PROMPT_PATH")
    mcp_config_path: Path = Field(default_factory=lambda: _existing_path_or_default("mcp.json"), alias="MCP_CONFIG_PATH")
    checkpoint_backend: Literal["sqlite", "memory", "postgres"] = Field(
        default="sqlite",
        alias="CHECKPOINT_BACKEND",
    )
    checkpoint_sqlite_path: Path = Field(
        default=BASE_DIR / ".agent_state" / "checkpoints.sqlite",
        alias="CHECKPOINT_SQLITE_PATH",
    )
    checkpoint_postgres_url: Optional[str] = Field(default=None, alias="CHECKPOINT_POSTGRES_URL")
    session_state_path: Path = Field(
        default=BASE_DIR / ".agent_state" / "session.json",
        alias="SESSION_STATE_PATH",
    )
    run_log_dir: Path = Field(default=BASE_DIR / "logs" / "runs", alias="RUN_LOG_DIR")

    # Provider Settings
    provider: Literal["gemini", "openai"] = Field(default="gemini", alias="PROVIDER")

    # Tavily Search
    tavily_api_key: Optional[SecretStr] = Field(default=None, alias="TAVILY_API_KEY")

    # Gemini
    gemini_api_key: Optional[SecretStr] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    # OpenAI / Compatible
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")

    # Common Logic
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_loops: int = Field(default=50, alias="MAX_LOOPS", description="Limit steps per request")
    critic_mode: Literal["hybrid", "always", "off"] = Field(default="hybrid", alias="CRITIC_MODE")
    critic_max_retries_per_turn: int = Field(
        default=1,
        alias="CRITIC_MAX_RETRIES_PER_TURN",
        description="Max critic-triggered retries in the same user turn",
    )
    tool_loop_window: Optional[int] = Field(default=None, alias="TOOL_LOOP_WINDOW")
    tool_loop_limit_mutating: Optional[int] = Field(default=None, alias="TOOL_LOOP_LIMIT_MUTATING")
    tool_loop_limit_readonly: Optional[int] = Field(default=None, alias="TOOL_LOOP_LIMIT_READONLY")

    # Features Toggle
    enable_search_tools: bool = Field(default=True, alias="ENABLE_SEARCH_TOOLS")
    model_supports_tools: bool = Field(default=True, alias="MODEL_SUPPORTS_TOOLS")
    use_system_tools: bool = Field(default=True, alias="ENABLE_SYSTEM_TOOLS")
    enable_filesystem_tools: bool = Field(default=True, alias="ENABLE_FILESYSTEM_TOOLS")
    enable_process_tools: bool = Field(default=False, alias="ENABLE_PROCESS_TOOLS")
    enable_shell_tool: bool = Field(default=False, alias="ENABLE_SHELL_TOOL")
    enable_approvals: bool = Field(default=True, alias="ENABLE_APPROVALS")
    allow_external_process_control: bool = Field(default=False, alias="ALLOW_EXTERNAL_PROCESS_CONTROL")

    # Tools Limits
    max_tool_output_length: int = Field(default=4000, alias="MAX_TOOL_OUTPUT")
    max_file_size: int = Field(
        default=DEFAULT_MAX_FILE_SIZE,
        alias="MAX_FILE_SIZE",
        description="Max file size in bytes",
    )
    max_background_processes: int = Field(default=5, alias="MAX_BACKGROUND_PROCESSES")
    max_search_chars: int = Field(default=15000, alias="MAX_SEARCH_CHARS")
    max_read_lines: int = Field(default=DEFAULT_READ_LIMIT, alias="MAX_READ_LINES")

    # Deterministic Mode
    strict_mode: bool = Field(default=False, alias="STRICT_MODE")

    # Summarization
    summary_threshold: int = Field(
        default=8000,
        alias="SESSION_SIZE",
        description="Estimated input context tokens before summarizing (~chars/2)",
    )
    summary_keep_last: int = Field(default=4, alias="SUMMARY_KEEP_LAST")

    # Network / Retry
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: int = Field(default=2, alias="RETRY_DELAY")
    debug: bool = Field(default=False, alias="DEBUG")

    @field_validator("max_file_size", mode="before")
    @classmethod
    def parse_max_file_size(cls, v: Union[int, float, str]) -> int:
        """
        Parse byte limits strictly.
        Plain numeric values are treated as bytes.
        String values may optionally include explicit units, e.g. 4MB or 300MiB.
        """
        if isinstance(v, (int, float)):
            value = int(v)
        elif isinstance(v, str):
            raw = v.strip()
            if not raw:
                raise ValueError("MAX_FILE_SIZE cannot be empty.")
            if raw.isdigit():
                value = int(raw)
            else:
                match = _SIZE_WITH_UNIT_RE.match(raw)
                if not match:
                    raise ValueError(
                        "Invalid MAX_FILE_SIZE format. Use bytes (e.g. 4096) or explicit units like 4MB / 300MiB."
                    )
                amount = float(match.group(1))
                unit = match.group(2).lower()
                value = int(amount * _SIZE_MULTIPLIERS[unit])
        else:
            raise ValueError("Invalid MAX_FILE_SIZE value.")

        if value < 1:
            raise ValueError("MAX_FILE_SIZE must be greater than 0.")
        return value

    @field_validator(
        "prompt_path",
        "mcp_config_path",
        "checkpoint_sqlite_path",
        "session_state_path",
        "run_log_dir",
        mode="before",
    )
    @classmethod
    def resolve_path_fields(cls, v: Union[str, Path]) -> Path:
        return _resolve_runtime_path(v)

    @field_validator("max_loops", mode="before")
    @classmethod
    def validate_max_loops(cls, v: Union[int, float, str]) -> int:
        """
        Ensure MAX_LOOPS is a positive integer.
        Prevents invalid recursion_limit values and ambiguous loop-guard behavior.
        """
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return 50

        if value < 1:
            return 1
        if value > 10000:
            return 10000
        return value

    @field_validator("critic_mode", mode="before")
    @classmethod
    def normalize_critic_mode(cls, v: str) -> str:
        value = str(v or "hybrid").strip().lower()
        if value not in {"hybrid", "always", "off"}:
            return "hybrid"
        return value

    @field_validator("critic_max_retries_per_turn", mode="before")
    @classmethod
    def parse_critic_max_retries_per_turn(cls, v: Union[int, float, str]) -> int:
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return 1
        if value < 0:
            return 0
        if value > 100:
            return 100
        return value

    @field_validator("checkpoint_backend", mode="before")
    @classmethod
    def normalize_checkpoint_backend(cls, v: str) -> str:
        value = str(v or "sqlite").strip().lower()
        if value not in {"sqlite", "memory", "postgres"}:
            return "sqlite"
        return value

    @field_validator(
        "tool_loop_window",
        "tool_loop_limit_mutating",
        "tool_loop_limit_readonly",
        mode="before",
    )
    @classmethod
    def parse_optional_loop_guard_value(cls, v: Union[int, float, str, None]) -> Optional[int]:
        """Parse optional loop-guard value from env and clamp to safe bounds."""
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        try:
            value = int(float(v))
        except (TypeError, ValueError):
            return None
        if value < 1:
            return 1
        if value > 10000:
            return 10000
        return value

    # Private cache attributes — PrivateAttr is the correct Pydantic v2 way to cache
    # computed values on BaseSettings instances (functools.cached_property doesn't work
    # because Pydantic v2 models don't expose a writable __dict__ slot for it).
    _cache_tool_loop_window: Optional[int] = PrivateAttr(default=None)
    _cache_tool_loop_limit_mutating: Optional[int] = PrivateAttr(default=None)
    _cache_tool_loop_limit_readonly: Optional[int] = PrivateAttr(default=None)
    _cache_safety: Optional[Any] = PrivateAttr(default=None)

    @property
    def effective_tool_loop_window(self) -> int:
        """History window for tool duplicate detection.
        Default is synchronized with MAX_LOOPS, unless overridden in env."""
        if self._cache_tool_loop_window is None:
            value = self.tool_loop_window if self.tool_loop_window is not None else max(10, min(60, self.max_loops))
            self._cache_tool_loop_window = value
        return self._cache_tool_loop_window

    @property
    def effective_tool_loop_limit_mutating(self) -> int:
        """Mutating tools duplicate limit.
        Formula default: max(3, min(8, MAX_LOOPS // 10))."""
        if self._cache_tool_loop_limit_mutating is None:
            value = self.tool_loop_limit_mutating if self.tool_loop_limit_mutating is not None else max(3, min(8, self.max_loops // 10))
            self._cache_tool_loop_limit_mutating = value
        return self._cache_tool_loop_limit_mutating

    @property
    def effective_tool_loop_limit_readonly(self) -> int:
        """Read-only tools duplicate limit.
        Formula default: max(6, min(20, MAX_LOOPS // 4))."""
        if self._cache_tool_loop_limit_readonly is None:
            value = self.tool_loop_limit_readonly if self.tool_loop_limit_readonly is not None else max(6, min(20, self.max_loops // 4))
            self._cache_tool_loop_limit_readonly = value
        return self._cache_tool_loop_limit_readonly

    @property
    def safety(self):
        """Returns SafetyPolicy object. Cached via PrivateAttr to prevent multiple instantiations."""
        if self._cache_safety is None:
            from core.safety_policy import SafetyPolicy
            self._cache_safety = SafetyPolicy(
                max_tool_output=self.max_tool_output_length,
                max_file_size=self.max_file_size,
                max_background_processes=self.max_background_processes,
                max_search_chars=self.max_search_chars,
                max_read_lines=self.max_read_lines,
                allow_shell=self.enable_shell_tool,
                allow_external_process_control=self.allow_external_process_control,
            )
        return self._cache_safety

    @model_validator(mode="after")
    def validate_provider_keys(self) -> "AgentConfig":
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for gemini provider.")

        if self.provider == "openai" and not self.openai_api_key:
            # Bypass API key check if a base_url is provided (common for local models like Ollama/vLLM)
            if not self.openai_base_url:
                raise ValueError("OPENAI_API_KEY required for openai provider.")

        return self
