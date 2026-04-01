# Autonomous AI Agent (GUI)

Desktop AI-агент на `LangGraph` + `PySide6` с постоянной историей чатов по проектам, checkpoint persistence и approval-потоком для опасных действий.

## Что есть сейчас

- Главный интерфейс: левый sidebar с чатами, справа transcript + composer.
- История чатов разделена по проектам (рабочим папкам).
- `New Session` создает новый чат в текущем проекте.
- `Open Folder` переключает проект и подгружает его историю (или создает первую сессию).
- Вызовы инструментов показываются компактно: имя + параметр, детали раскрываются по клику.
- Состояние рантайма (`Thinking`, `Reviewing`, `Ready`, ошибки) отображается в статусной строке и в transcript.
- Approval gate для mutating/destructive действий: `Approve`, `Deny`, `Always for this session`.

## Архитектура

- [`agent.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/agent.py): сборка агента, LLM, tool registry, checkpoint backend.
- [`main.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/main.py): главное GUI-окно, toolbar, sidebar, transcript, статус-строка и точка входа приложения.
- [`core/gui_runtime.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/gui_runtime.py): runtime/controller, события, переключение сессий, восстановление transcript.
- [`core/session_store.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/session_store.py): snapshot активной сессии + индекс сессий по проектам.
- [`core/gui_widgets.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/gui_widgets.py): виджеты transcript/sidebar/tool blocks.
- [`core/config.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/config.py): env-конфигурация.
- [`tools/tool_registry.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tools/tool_registry.py): локальные и MCP инструменты, metadata/policy.

## Хранение данных

- Checkpoint state: `CHECKPOINT_SQLITE_PATH` (или другой backend).
- Активная сессия: `SESSION_STATE_PATH` (по умолчанию `.agent_state/session.json`).
- Индекс чатов для sidebar: `.agent_state/session_index.json`.
- Логи запусков: `RUN_LOG_DIR` (JSONL).

При обновлении со старых версий текущая активная сессия мигрируется в индекс автоматически.

## Быстрый старт

1. Создать окружение:
```bash
python -m venv venv
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

3. Создать `.env`:
```powershell
Copy-Item env_example.txt .env
```

4. Запустить GUI:
```bash
python main.py
```

## Основные переменные окружения

### Провайдер
- `PROVIDER=gemini|openai`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`

### Runtime/persistence
- `PROMPT_PATH`
- `MCP_CONFIG_PATH`
- `CHECKPOINT_BACKEND=sqlite|memory|postgres`
- `CHECKPOINT_SQLITE_PATH`
- `CHECKPOINT_POSTGRES_URL`
- `SESSION_STATE_PATH`
- `RUN_LOG_DIR`

### Инструменты и безопасность
- `MODEL_SUPPORTS_TOOLS`
- `ENABLE_FILESYSTEM_TOOLS`
- `ENABLE_SEARCH_TOOLS` (+ `TAVILY_API_KEY`)
- `ENABLE_SYSTEM_TOOLS`
- `ENABLE_PROCESS_TOOLS`
- `ENABLE_SHELL_TOOL`
- `ENABLE_APPROVALS`
- `ALLOW_EXTERNAL_PROCESS_CONTROL`

### Лимиты
- `MAX_TOOL_OUTPUT`
- `MAX_SEARCH_CHARS`
- `MAX_FILE_SIZE`
- `MAX_READ_LINES`
- `MAX_BACKGROUND_PROCESSES`

### Контекст и retry
- `SESSION_SIZE`
- `SUMMARY_KEEP_LAST`
- `MAX_RETRIES`
- `RETRY_DELAY`

### Диагностика
- `DEBUG`
- `STRICT_MODE`
- `LOG_LEVEL`
- `LOG_FILE`

## Тесты

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ключевые наборы:
- [`tests/test_cli_ux.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tests/test_cli_ux.py)
- [`tests/test_runtime_refactor.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tests/test_runtime_refactor.py)
- [`tests/test_stream_and_filesystem.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tests/test_stream_and_filesystem.py)
- [`tests/test_tooling_refactor.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tests/test_tooling_refactor.py)

## Безопасность

- Не храните реальные ключи в репозитории.
- `ENABLE_SHELL_TOOL=true` используйте только при необходимости.
- Для production persistence используйте `postgres` backend.
