# Portable Autonomous AI Agent (GUI)

Десктопный AI-агент с runtime на `LangGraph` и графическим интерфейсом на `PySide6` для работы с кодом, файлами, shell-командами, MCP и web-assisted сценариями.

Проект можно запускать из исходников через `python main.py` или собирать в portable-исполняемый файл для Windows через `build.bat`.

![Portable Autonomous AI Agent](./img/01.jpg)

## Что Это За Проект

Этот репозиторий представляет собой локальное десктопное агентное приложение с:

- графовым runtime на `LangGraph`
- GUI с историей чатов, transcript, approvals, настройками и вложениями
- подключаемыми инструментами: filesystem, shell, search, system, process, MCP
- сохранением сессий и durable checkpoints
- approval-паузами перед рискованными действиями
- bounded recovery и self-correction после tool/protocol ошибок
- управлением профилями моделей прямо в GUI
- опциональной поддержкой ввода изображений для активного профиля модели

Текущая точка входа находится в `main.py`, который проксирует запуск в `ui/main_window.py`. Сборка и маршрутизация графа агента находятся в `agent.py`.

После последнего рефакторинга внутренняя структура стала более модульной:

- orchestration узлов LangGraph вынесена в `core/node_orchestrators.py`
- runtime GUI разделён на payload/builders, session coordination и worker lifecycle через `ui/runtime_payloads.py`, `ui/runtime_session.py` и `ui/runtime_worker.py`
- state-heavy логика окна разгружена в `ui/main_window_state.py`

## Runtime Flow

Актуальный граф в `agent.py`:

```text
START
  -> summarize
  -> update_step
  -> agent
     -> approval -> tools
     -> tools
     -> recovery
     -> END

tools -> recovery | update_step
recovery -> update_step | END
```

Ключевые детали:

- `summarize` сжимает старый контекст, когда история сессии становится слишком большой.
- `agent` решает, нужно ли отвечать сразу, вызывать tools или передавать ход в recovery.
- `approval` ставит выполнение на паузу перед mutating или рискованными tool-вызовами.
- `tools` исполняет подтверждённые tool-вызовы и фиксирует проблемы.
- `recovery` применяет ограниченную стратегию retry/self-correction вместо бесконечных циклов.
- recovery-состояние намеренно минимальное: `active_issue`, `active_strategy` и счётчики `self_correction_*` без промежуточных очередей.
- `MAX_LOOPS` и per-tool loop guards ограничивают runaway tool loops.

## GUI

Основные возможности интерфейса:

- боковая панель с историей чатов по проекту
- transcript со streaming-ответами, tool cards, summary-сообщениями и статусными notice
- composer с многострочным вводом, вставкой путей к файлам и image attachments
- диалог approval для mutating и рискованных инструментов
- переключатель активной модели и окно настроек профилей
- info popup с вкладками runtime, tools и help
- переключение проекта, которое создаёт новый чат-контекст для выбранной папки

Особенности composer:

- локальные файлы можно вставлять через `Add files…`, drag-and-drop или clipboard paste
- `@`-mention popup показывает файлы и директории из текущего workspace
- список для `@` пересканируется динамически, поэтому новые файлы и папки, появившиеся после запуска приложения, тоже доступны без перезапуска

Горячие клавиши:

- `Enter` отправить сообщение
- `Shift+Enter` новая строка
- `Ctrl+N` новый чат
- `Ctrl+B` показать или скрыть боковую панель
- `Ctrl+I` открыть info popup
- `Up/Down` в пустом composer листает историю отправленных сообщений

## Возможности Агента

В зависимости от флагов в `.env` и установленных зависимостей агент умеет:

- читать и изменять файлы через filesystem tools
- выполнять локальные shell-команды через shell tool
- смотреть системное состояние и, при включении флага, управлять процессами
- выполнять web search и fetch контента через Tavily
- вызывать MCP tools, описанные в `mcp.json`
- запрашивать у пользователя один явный блокирующий выбор через `request_user_input`
- принимать изображения во вложениях, если активный профиль модели поддерживает image input

## Встроенная Модель Безопасности

В текущем проекте реализован довольно строгий policy-слой runtime:

- approvals по умолчанию включены для mutating и destructive действий
- shell-команды предварительно классифицируются перед выполнением
- MCP tools по умолчанию требуют approval, если явная policy не помечает их как read-only
- tool errors могут переводить выполнение в recovery, а не игнорироваться
- self-correction ограничен параметром `SELF_CORRECTION_RETRY_LIMIT`
- количество фоновых процессов ограничено `MAX_BACKGROUND_PROCESSES`

`request_user_input` обрабатывается отдельно:

- он должен использоваться только для действительно блокирующего выбора
- за один пользовательский ход допускается только один такой запрос
- его нельзя батчить вместе с другими tool calls в одном assistant-ответе

## Быстрый Старт

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
Copy-Item env_example.txt .env
python main.py
```

## Portable Сборка

Сборка portable-версии описана в `build.bat`, который использует `PyInstaller` в режиме `--onefile --windowed`.

```powershell
.\build.bat
```

## Структура Проекта

```text
.
├─ agent.py
├─ main.py
├─ build.bat
├─ core/
├─ tools/
├─ ui/
├─ tests/
├─ utils/
├─ prompt.txt
├─ prompt_dev.txt
├─ env_example.txt
├─ requirements.txt
└─ mcp.json
```

Ключевые каталоги:

- `core/` runtime, конфиг, checkpointing, policy-логика, recovery, состояние сессий и моделей
- `tools/` встроенные инструменты и MCP-интеграция
- `tools/filesystem_impl/` низкоуровневая реализация filesystem-операций
- `ui/` Qt runtime bridge, главное окно, виджеты и streaming-представление
- `tests/` тесты runtime, tooling, policy, model profiles и GUI

Внутри `ui/` сейчас полезно знать такие узлы ответственности:

- `ui/runtime.py` совместимый facade-слой для runtime API
- `ui/widgets/composer.py` история ввода, paste/drag-and-drop и `@`-mentions
- `ui/main_window_state.py` event routing, composer state и статусный таймер

## Конфигурация

Приложение читает настройки из `.env` через `core/config.py`.

### Провайдер И Модели

- `PROVIDER=gemini|openai`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`, по умолчанию `gemini-1.5-flash`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`, по умолчанию `gpt-4o`
- `OPENAI_BASE_URL` опционален для OpenAI-compatible backend

### Управление Runtime

- `TEMPERATURE`
- `MAX_LOOPS`
- `TOOL_LOOP_WINDOW`
- `TOOL_LOOP_LIMIT_MUTATING`
- `TOOL_LOOP_LIMIT_READONLY`
- `SELF_CORRECTION_RETRY_LIMIT`

### Состояние И Персистентность

- `PROMPT_PATH`
- `MCP_CONFIG_PATH`
- `CHECKPOINT_BACKEND=sqlite|memory`
- `CHECKPOINT_SQLITE_PATH`
- `SESSION_STATE_PATH`
- `RUN_LOG_DIR`
- `LOG_FILE`

### Фиче-Флаги

- `MODEL_SUPPORTS_TOOLS`
- `ENABLE_SEARCH_TOOLS`
- `ENABLE_FILESYSTEM_TOOLS`
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
- `STREAM_TEXT_MAX_CHARS`
- `STREAM_EVENTS_MAX`
- `STREAM_TOOL_BUFFER_MAX`

### Суммаризация И Retry

- `SESSION_SIZE`
- `SUMMARY_KEEP_LAST`
- `MAX_RETRIES`
- `RETRY_DELAY`

### Диагностика

- `DEBUG`
- `LOG_LEVEL`
- `STRICT_MODE`

## Профили Моделей

GUI теперь поддерживает несколько сохраняемых профилей моделей через `core/model_profiles.py`.

Профиль хранит:

- провайдера
- имя модели
- API key
- опциональный `base_url` для OpenAI-compatible провайдеров
- флаг поддержки image input
- состояние enabled/disabled

Важно:

- активный профиль выбирается в GUI, а не только через `.env`
- значения из `.env` всё ещё используются для bootstrap начального набора профилей
- legacy-ключи `MODEL`, `API_KEY` и `BASE_URL` по-прежнему поддерживаются для bootstrap/import
- для Gemini-профилей `base_url` игнорируется

## Сессии И Checkpoints

Текущее поведение persistence:

- graph checkpoints могут использовать `sqlite` или `memory`
- локально по умолчанию используется `.agent_state/checkpoints.sqlite`
- активная сессия хранится в `.agent_state/session.json`
- индекс сессий хранится в `.agent_state/session_index.json`
- логи запусков сохраняются в формате JSONL в `logs/runs`

Это позволяет приложению восстанавливать состояние чатов и продолжать работу с тем же durable graph thread metadata.

## MCP

`mcp.json` задаёт опциональные MCP-серверы.

В текущем примере конфигурации есть:

- `filesystem` выключен
- `context7` выключен и помечен как read-only
- `sequential-thinking` включён и помечен как read-only

Поведение policy:

- если `policy.read_only=true`, tools могут считаться read-only
- если `policy.read_only=false`, tools требуют approval
- если блока `policy` нет, MCP tools по умолчанию работают в консервативном approval-режиме

Минимальный пример:

```json
{
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "transport": "http",
    "enabled": true,
    "policy": {
      "read_only": true
    }
  }
}
```

## Prompt Layers

Prompt-слои в текущем коде:

- `prompt.txt` базовый системный prompt
- `prompt_dev.txt` дополнительный prompt-ресурс
- `core/runtime_prompt_policy.py` runtime overlay с tool и user-input policy
- `core/recovery_manager.py` recovery-специфичная логика и подсказки

## Тесты

Запуск тестов:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Если в dev-окружении установлен `pytest`, можно запускать и через него:

```powershell
venv\Scripts\python.exe -m pytest
```

В репозитории уже есть заметное покрытие для:

- поведения runtime graph
- approvals и recovery
- tool metadata и policy engine
- session storage и checkpoints
- model profile management
- GUI-поведения и layout
- composer UX: clipboard paste, drag-and-drop, image attachments и `@`-mentions

## Требования И Примечания

- для `Gemini` нужен `GEMINI_API_KEY`
- для `OpenAI` нужен `OPENAI_API_KEY`, если не используется совместимый backend
- для web search нужен `TAVILY_API_KEY` и `ENABLE_SEARCH_TOOLS=true`
- для desktop UI нужен `PySide6`
- для sqlite-checkpoint нужен пакет `langgraph-checkpoint-sqlite`

## Текущий Tech Stack

- `LangChain`
- `LangGraph`
- `PySide6`
- `pydantic-settings`
- `Tavily`
- `httpx`
- `psutil`
- опциональный MCP через `langchain-mcp-adapters` и `mcp`
