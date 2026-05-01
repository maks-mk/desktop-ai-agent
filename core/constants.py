import sys
from pathlib import Path

# Agent version (single source of truth)
AGENT_VERSION = "v0.66.98b"

# Determine the project root directory
if getattr(sys, 'frozen', False):
    # When running as an executable, use the executable directory for config files,
    # but keep the current working directory (cwd) as the process launch location.
    BASE_DIR = Path(sys.executable).parent
else:
    # core/constants.py -> core/ -> root/
    BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROMPTS ---

SUMMARY_PROMPT_TEMPLATE = (
    "Current memory:\n<previous_context>\n{summary}\n</previous_context>\n\n"
    "New events:\n{history_text}\n\n"
    "Update the memory. Rules:\n"
    "- Merge new events, do not repeat existing facts.\n"
    "- Keep: paths, commands, outcomes, errors, decisions, task status.\n"
    "- Drop: greetings, filler, raw tool output already reflected in outcomes.\n"
    "- Be concise. Bullet points preferred.\n"
    "- Return only the updated memory text."
)

REFLECTION_PROMPT = (
    "SYSTEM HINT: The previous tool execution failed. "
    "Fix arguments or choose a different tool, then continue immediately. "
    "Do not claim success until a tool result confirms it."
)

UNRESOLVED_TOOL_ERROR_PROMPT_TEMPLATE = (
    "UNRESOLVED TOOL FAILURE:\n"
    "{error_summary}\n\n"
    "Do not claim success, completion, or verified characteristics unless you have actually resolved this "
    "with later successful tool results. Either retry with corrected arguments, use another tool, "
    "or clearly explain the blocker to the user."
)

RECOVERY_CONTINUE_PROMPT_TEMPLATE = (
    "Continue the current task: {current_task}. "
    "Use tools or another verifiable step instead of stopping."
)

TOOL_ISSUE_NOT_FOUND_TEXT = "Unable to continue: the active tool issue could not be found."

TOOL_ISSUE_APPROVAL_DENIED_TEXT = (
    "Action not completed: you declined an irreversible operation.\n"
    "The next attempt requires a new request or explicit confirmation."
)

TOOL_ISSUE_WORKSPACE_BOUNDARY_TEMPLATE = (
    "Unable to continue: the request goes outside the workspace boundary{tool_hint}.{summary_line}\n"
    "This cannot be fixed automatically without changing the target path."
)

TOOL_ISSUE_MISSING_FIELDS_TEMPLATE = (
    "External data is missing to continue{tool_hint}.{summary_line}\n"
    "Please clarify: {fields_label}."
)

TOOL_ISSUE_STAGNATION_TEMPLATE = (
    "Unable to complete the task after automatic recovery attempts{tool_hint}.{summary_line}\n"
    "Auto-recovery stopped due to stagnation: without new external data or changed conditions, no safe next steps remain."
)

LOOP_BUDGET_HANDOFF_TEMPLATE = (
    "Unable to continue recovery for task {task_hint}{tool_hint}: the internal step limit has been reached.\n"
    "In the current context, automatic strategies have been exhausted."
)

SUCCESSFUL_TOOL_STAGNATION_HANDOFF_TEMPLATE = (
    "Paused the task for {task_hint}{tool_hint}: the same confirmed result repeated {repeat_count} times in a row.\n"
    "Continuing now risks going in circles without making progress."
)

DEFAULT_INTERNAL_UI_NOTICE = "Paused at this step. You can continue with a new message."

LOOP_BUDGET_UI_NOTICE = (
    "Paused at this step: the internal retry limit for this request has been reached. "
    "You can continue with a new message."
)

TOOL_ISSUE_UI_NOTICE = (
    "Paused at this step. To move forward without unnecessary repetition, send a new request or a short clarification."
)

SUCCESSFUL_TOOL_STAGNATION_UI_NOTICE = (
    "Paused at this step: the result has already been confirmed several times in a row, and continuing may just loop. "
    "You can continue with a short message."
)
