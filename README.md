# Autonomous AI Agent (GUI)

Desktop AI-агент с графовым runtime на `LangGraph` и интерфейсом на `PySide6`.
Проект ориентирован на локальную разработку: работа с файлами, запуск инструментов, approvals для рискованных действий, история сессий и интеграция MCP-инструментов.
Позиционирование: **Portable desktop agent** — после сборки в `.exe` не требует установки и может запускаться с флешки на другом ПК.

## Что в проекте сейчас

- Граф агента: `summarize -> classify_turn -> agent -> {approval|tools|recovery|END}`.
- Локальные инструменты: filesystem, search, system, process, shell, user-input.
- MCP-подключение через `mcp.json` (если конфиг существует).
- Session persistence и checkpoint backend (`sqlite`, `memory`, `postgres`).
- Мультимодальный пользовательский ввод (в том числе изображения) с provider-aware маршрутизацией.
- GUI с transcript, sidebar сессий, composer, tool-cards и approvals.

## Структура

```text
.
├─ agent.py
├─ build.bat
├─ env_example.txt
├─ main.py
├─ mcp.json
├─ prompt.txt
├─ requirements.txt
├─ core/
├─ tools/
├─ ui/
├─ tests/
└─ utils/
```

- `core/` — runtime, policy, recovery, state, checkpoints, конфигурация.
- `tools/` — реестр и реализации локальных/MCP инструментов.
- `ui/` — окно, runtime-воркер, стриминг событий, виджеты.
- `tests/` — регрессионные и функциональные тесты.
- `utils/` — вспомогательные скрипты для диагностики и обслуживания проекта.
- `.agent_state/`, `logs/` — служебные директории состояния сессий и логов во время работы.
- `build.bat` — сборка portable `.exe`.
- `env_example.txt` — шаблон env-конфига.
- `mcp.json` — конфигурация MCP-серверов/инструментов.
- `requirements.txt` — зависимости проекта.

## Основные компоненты

- `agent.py` — сборка LLM, загрузка инструментов, создание workflow, compile приложения.
- `core/config.py` — типизированная загрузка env и runtime-лимитов.
- `core/nodes.py` — узлы графа, tool-calling, approvals, recovery.
- `core/context_builder.py` — подготовка и нормализация контекста для провайдера.
- `core/policy_engine.py` — определение intent и политик режима выполнения.
- `core/recovery_manager.py`, `core/self_correction_engine.py` — автоматическое восстановление после tool/runtime-ошибок.
- `core/session_store.py`, `core/session_utils.py`, `core/checkpointing.py` — хранение и восстановление сессий/чекпоинтов.
- `core/model_profiles.py`, `core/multimodal.py` — профили моделей и мультимодальный ввод.
- `tools/tool_registry.py` — загрузка локальных и MCP инструментов, метаданные безопасности.
- `tools/filesystem.py`, `tools/search_tools.py`, `tools/system_tools.py`, `tools/process_tools.py`, `tools/local_shell.py` — основные группы локальных инструментов.
- `ui/runtime.py` — orchestration GUI <-> graph, payloads, resume approval/user-choice.
- `ui/main_window.py` — главное окно и wiring UI-событий.
- `ui/streaming.py` — обработка stream-событий и tool lifecycle в transcript.
- `ui/theme.py` — централизованная тема и стили интерфейса.
- `ui/tool_message_utils.py`, `ui/visibility.py` — нормализация tool payload и фильтрация внутренних сообщений.
- `ui/widgets/composer.py` — composer, history, paste, `@`-mentions.

## Инструменты

Локальные инструменты подключаются через `ToolRegistry` из `tools/tool_registry.py`.

Доступные группы:

- Filesystem: `file_info`, `read_file`, `write_file`, `edit_file`, `list_directory`, `search_in_file`, `search_in_directory`, `tail_file`, `find_file`, `download_file`, delete-tools.
- Search: `web_search`, `fetch_content`, `batch_web_search` (и `crawl_site`, если доступен в модуле).
- System: `get_public_ip`, `lookup_ip_info`, `get_system_info`, `get_local_network_info`.
- Process: `run_background_process`, `stop_background_process`, `find_process_by_port`.
- Shell: `cli_exec`.
- User interaction: `request_user_input` (всегда доступен).

## Быстрый старт

1. Создать виртуальное окружение:

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

## Portable-сборка (exe без установки)

Сценарий использования:

1. Собрать приложение:

```powershell
.\build.bat
```

2. Взять собранный `.exe` (и файлы рядом с ним из output-папки сборки).
3. Скопировать на флешку или в любую папку на другом ПК.
4. Запустить `.exe` — установка не требуется.

Это позволяет использовать агента как переносимый desktop-инструмент: "собрал один раз -> перенес -> запустил".

## Переменные окружения

Ключевые переменные в `core/config.py`:

### Provider

- `PROVIDER=gemini|openai`
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`
- OpenAI-compatible: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`

### Пути и persistence

- `PROMPT_PATH`
- `MCP_CONFIG_PATH`
- `CHECKPOINT_BACKEND=sqlite|memory|postgres`
- `CHECKPOINT_SQLITE_PATH`
- `CHECKPOINT_POSTGRES_URL`
- `SESSION_STATE_PATH`
- `RUN_LOG_DIR`
- `LOG_FILE`

### Фичи и безопасность

- `MODEL_SUPPORTS_TOOLS`
- `ENABLE_FILESYSTEM_TOOLS`
- `ENABLE_SEARCH_TOOLS`
- `ENABLE_SYSTEM_TOOLS`
- `ENABLE_PROCESS_TOOLS`
- `ENABLE_SHELL_TOOL`
- `ENABLE_APPROVALS`
- `ALLOW_EXTERNAL_PROCESS_CONTROL`
- `SELF_CORRECTION_ENABLE_AUTO_REPAIR`
- `SELF_CORRECTION_MAX_AUTO_REPAIRS`

### Лимиты

- `MAX_TOOL_OUTPUT`
- `MAX_FILE_SIZE`
- `MAX_READ_LINES`
- `MAX_BACKGROUND_PROCESSES`
- `MAX_SEARCH_CHARS`
- `STREAM_TEXT_MAX_CHARS`
- `STREAM_EVENTS_MAX`
- `STREAM_TOOL_BUFFER_MAX`
- `SESSION_SIZE`
- `SUMMARY_KEEP_LAST`
- `MAX_RETRIES`
- `RETRY_DELAY`

### Диагностика

- `DEBUG`
- `STRICT_MODE`
- `LOG_LEVEL`

## Тесты

Запуск полного набора:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Актуальные файлы тестов:

- `tests/test_cli_ux.py`
- `tests/test_critic_graph.py`
- `tests/test_intent_engine.py`
- `tests/test_model_profiles.py`
- `tests/test_policy_engine.py`
- `tests/test_refactor_services.py`
- `tests/test_runtime_refactor.py`
- `tests/test_self_correction_engine.py`
- `tests/test_session_utils.py`
- `tests/test_stability_guard.py`
- `tests/test_stream_and_filesystem.py`
- `tests/test_tooling_refactor.py`

## Примечания

- Для `gemini` требуется `GEMINI_API_KEY`.
- Для `openai` без `OPENAI_API_KEY` нужен `OPENAI_BASE_URL` (локальный/совместимый endpoint).
- Для web-search нужен `TAVILY_API_KEY`.
- Для production checkpoint рекомендуется `postgres` backend.
