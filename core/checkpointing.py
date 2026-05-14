import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import MemorySaver

from core.config import AgentConfig

logger = logging.getLogger("agent")


def _ensure_connection_is_alive_compat(conn: Any) -> Any:
    if hasattr(conn, "is_alive"):
        return conn

    def is_alive() -> bool:
        thread = getattr(conn, "_thread", None)
        if thread is not None:
            return bool(thread.is_alive())
        return bool(getattr(conn, "_connection", None) is not None)

    setattr(conn, "is_alive", is_alive)
    return conn


async def _create_async_sqlite_checkpointer(db_path: Path) -> tuple[Any, Callable[[], Any]]:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(db_path))
    checkpointer = AsyncSqliteSaver(_ensure_connection_is_alive_compat(conn))
    return checkpointer, conn.close


@dataclass
class CheckpointRuntime:
    backend: str
    resolved_backend: str
    target: str
    checkpointer: Any
    warnings: list[str] = field(default_factory=list)
    close_callback: Optional[Callable[[], Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "resolved_backend": self.resolved_backend,
            "target": self.target,
            "warnings": list(self.warnings),
        }

    async def aclose(self) -> None:
        if self.close_callback is None:
            return
        result = self.close_callback()
        if inspect.isawaitable(result):
            await result


async def _maybe_setup(checkpointer: Any) -> None:
    setup = getattr(checkpointer, "setup", None)
    if not callable(setup):
        return
    result = setup()
    if inspect.isawaitable(result):
        await result


async def create_checkpoint_runtime(config: AgentConfig) -> CheckpointRuntime:
    backend = config.checkpoint_backend

    if backend == "memory":
        return CheckpointRuntime(
            backend=backend,
            resolved_backend="memory",
            target="in-memory",
            checkpointer=MemorySaver(),
        )

    if backend == "sqlite":
        try:
            import aiosqlite  # noqa: F401
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
        except ImportError:
            warning = (
                "SQLite checkpointer is unavailable because 'langgraph-checkpoint-sqlite' is not installed. "
                "Falling back to MemorySaver."
            )
            logger.warning(warning)
            return CheckpointRuntime(
                backend=backend,
                resolved_backend="memory",
                target="in-memory",
                checkpointer=MemorySaver(),
                warnings=[warning],
            )

        db_path = Path(config.checkpoint_sqlite_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer, close_callback = await _create_async_sqlite_checkpointer(db_path)
        await _maybe_setup(checkpointer)
        return CheckpointRuntime(
            backend=backend,
            resolved_backend="sqlite",
            target=str(db_path),
            checkpointer=checkpointer,
            close_callback=close_callback,
        )

    raise ValueError(f"Unsupported checkpoint backend: {backend}")
