from __future__ import annotations

from typing import Any, List
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_core.tools import tool
from langgraph.types import interrupt


def _normalize_choice_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").split()).strip()


class RequestUserInputInput(BaseModel):
    """Strict payload for request_user_input so the model gets one concrete, UI-friendly choice prompt."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(
        description=(
            "Exactly one short blocking question for the user. "
            "Use only when the next step cannot continue without a user decision or missing external input."
        ),
    )
    options: List[str] = Field(
        min_length=2,
        max_length=5,
        description=(
            "2 to 5 short mutually exclusive answer options. "
            "Each option must be concise, self-contained, and directly selectable by the user."
        ),
    )
    recommended: str = Field(
        default="",
        description=(
            "Optional exact option text to recommend. "
            "Leave empty if there is no clear best option."
        ),
    )
    choice_type: str = Field(
        default="clarification",
        description=(
            "Machine-readable reason for the choice prompt: clarification, plan_approval, or plan_revision."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        data["question"] = _normalize_choice_text(data.get("question"))

        raw_options = data.get("options")
        if isinstance(raw_options, (list, tuple)):
            normalized_options: List[str] = []
            seen: set[str] = set()
            for raw_option in raw_options:
                option_text = ""
                if isinstance(raw_option, dict):
                    option_text = _normalize_choice_text(
                        raw_option.get("submit_text")
                        or raw_option.get("value")
                        or raw_option.get("label")
                        or raw_option.get("key")
                    )
                else:
                    option_text = _normalize_choice_text(raw_option)
                if not option_text:
                    continue
                option_key = option_text.casefold()
                if option_key in seen:
                    continue
                seen.add(option_key)
                normalized_options.append(option_text)
            data["options"] = normalized_options

        recommended = _normalize_choice_text(data.get("recommended"))
        if recommended and isinstance(data.get("options"), list):
            option_map = {str(option).casefold(): str(option) for option in data["options"]}
            recommended = option_map.get(recommended.casefold(), recommended)
        data["recommended"] = recommended
        data["choice_type"] = _normalize_choice_text(data.get("choice_type")) or "clarification"
        return data

    @model_validator(mode="after")
    def validate_payload(self) -> "RequestUserInputInput":
        if not self.question:
            raise ValueError("question must be non-empty.")

        if len(self.options) < 2 or len(self.options) > 5:
            raise ValueError("options must contain 2 to 5 distinct entries.")

        if self.recommended and self.recommended not in self.options:
            raise ValueError("recommended must match one of the options exactly.")

        if self.choice_type not in {"clarification", "plan_approval", "plan_revision"}:
            raise ValueError("choice_type must be clarification, plan_approval, or plan_revision.")

        return self


@tool("request_user_input", args_schema=RequestUserInputInput)
def request_user_input(
    question: str,
    options: List[str],
    recommended: str = "",
    choice_type: str = "clarification",
) -> str:
    """Pause the turn and ask the user to choose exactly one option.

    Use this only when the next step is blocked by a real user decision or missing external input
    that cannot be recovered from repository state, current messages, or available tools.

    Rules:
    - Call it at most once per assistant turn.
    - Make this tool call by itself; never combine it with other tool calls.
    - Ask one concrete question, not multiple questions.
    - Provide 2 to 5 short mutually exclusive options.
    - If one option is clearly best, set `recommended` to that exact option text.
    - Do not use it for risky-action approval; the approval flow handles that separately.
    - After resume, treat the returned string as the user's final answer and continue without asking again in the same turn.
    """
    result = interrupt(
        {
            "kind": "user_choice",
            "choice_type": str(choice_type or "clarification"),
            "question": question,
            "options": [str(option) for option in options],
            "recommended": str(recommended or ""),
        }
    )
    return str(result)


request_user_input.metadata = {
    "readOnlyHint": True,
    "humanInTheLoop": True,
    "singleTurnLimit": 1,
}
