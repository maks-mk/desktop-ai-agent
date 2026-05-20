from __future__ import annotations

from typing import Final, Literal

TurnOutcome = Literal["run_tools", "recover_agent", "finish_turn", "continue_agent"]

TURN_OUTCOME_RUN_TOOLS: Final[TurnOutcome] = "run_tools"
TURN_OUTCOME_RECOVER_AGENT: Final[TurnOutcome] = "recover_agent"
TURN_OUTCOME_FINISH_TURN: Final[TurnOutcome] = "finish_turn"
TURN_OUTCOME_CONTINUE_AGENT: Final[TurnOutcome] = "continue_agent"

_VALID_TURN_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        TURN_OUTCOME_RUN_TOOLS,
        TURN_OUTCOME_RECOVER_AGENT,
        TURN_OUTCOME_FINISH_TURN,
        TURN_OUTCOME_CONTINUE_AGENT,
    }
)


def normalize_turn_outcome(value: object) -> TurnOutcome:
    normalized = str(value or "").strip().lower()
    if normalized in _VALID_TURN_OUTCOMES:
        return normalized  # type: ignore[return-value]
    return TURN_OUTCOME_FINISH_TURN
