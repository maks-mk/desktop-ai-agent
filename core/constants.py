import sys
from pathlib import Path

# Agent version (single source of truth)
AGENT_VERSION = "v0.66.7b"

# Определение корневой директории проекта
if getattr(sys, 'frozen', False):
    # Если запущено как exe, используем директорию исполняемого файла для конфигов,
    # но рабочей директорией оставим текущую (cwd), откуда запущен процесс.
    BASE_DIR = Path(sys.executable).parent
else:
    # core/constants.py -> core/ -> root/
    BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROMPTS ---

SUMMARY_PROMPT_TEMPLATE = (
    "Current memory context:\n<previous_context>\n{summary}\n</previous_context>\n\n"
    "New events:\n{history_text}\n\n"
    "Update <previous_context>. This is a technical log of a software development session. "
    "Keep only key facts, decisions, and results. "
    "Remove chit-chat. Return only the updated context text."
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

TOOL_ISSUE_NOT_FOUND_TEXT = "Не удалось продолжить задачу: активная проблема инструмента не найдена."

TOOL_ISSUE_APPROVAL_DENIED_TEXT = (
    "Действие не выполнено: вы отклонили необратимую операцию.\n"
    "Для следующей попытки нужен новый запрос или явное подтверждение."
)

TOOL_ISSUE_WORKSPACE_BOUNDARY_TEMPLATE = (
    "Не могу продолжить: запрос выходит за пределы рабочей папки{tool_hint}.{summary_line}\n"
    "Автоматически это не исправить без изменения целевого пути."
)

TOOL_ISSUE_MISSING_FIELDS_TEMPLATE = (
    "Для продолжения не хватает внешних данных{tool_hint}.{summary_line}\n"
    "Нужно уточнить: {fields_label}."
)

TOOL_ISSUE_STAGNATION_TEMPLATE = (
    "Не удалось завершить задачу после автоматических попыток исправления{tool_hint}.{summary_line}\n"
    "Автовосстановление остановилось на стагнации: без новых внешних данных или изменения условий безопасных следующих шагов не осталось."
)

LOOP_BUDGET_HANDOFF_TEMPLATE = (
    "Не удалось продолжить восстановление для задачи {task_hint}{tool_hint}: достигнут жёсткий лимит внутренних шагов.\n"
    "В текущем контексте автоматические стратегии уже исчерпаны."
)

SUCCESSFUL_TOOL_STAGNATION_HANDOFF_TEMPLATE = (
    "Поставил задачу на паузу для {task_hint}{tool_hint}: один и тот же подтверждённый результат повторился {repeat_count} раз подряд.\n"
    "Дальше есть риск просто ходить по кругу без нового прогресса."
)

DEFAULT_INTERNAL_UI_NOTICE = "Сделал паузу на этом шаге. Можно продолжить новым сообщением."

LOOP_BUDGET_UI_NOTICE = (
    "Сделал паузу на этом шаге: для текущего запроса уже использован внутренний лимит попыток. "
    "Можно продолжить новым сообщением."
)

TOOL_ISSUE_UI_NOTICE = (
    "Сделал паузу на этом шаге. Чтобы двигаться дальше без лишних повторов, нужен новый запрос или короткое уточнение."
)

SUCCESSFUL_TOOL_STAGNATION_UI_NOTICE = (
    "Сделал паузу на этом шаге: результат уже несколько раз подряд подтвердился, и дальше можно просто ходить по кругу. "
    "Можно продолжить коротким сообщением."
)
