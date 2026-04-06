from typing import List

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def request_user_input(
    question: str,
    options: List[str],
    recommended: str = "",
) -> str:
    """
    Request an explicit user choice when the task cannot proceed without it.

    Use only when:
    - A decision has multiple valid paths with different outcomes.
    - External information is required and cannot be inferred from context.

    Do not use for uncertainty you can resolve with available tools.
    Do not call more than once in the same assistant turn.
    Do not batch multiple user-input requests.
    Ask one concrete question, wait for resume, then continue with the answer.
    For demo or test requests, do exactly one call and then finish with a brief confirmation.
    """
    result = interrupt(
        {
            "kind": "user_choice",
            "question": question,
            "options": [str(option) for option in options],
            "recommended": str(recommended or ""),
        }
    )
    return str(result)


request_user_input.metadata = {"readOnlyHint": True}
