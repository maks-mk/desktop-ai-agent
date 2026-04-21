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

Текущая точка входа находится в [`main.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/main.py), который проксирует запуск в [`ui/main_window.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/ui/main_window.py). Сборка и маршрутизация графа агента находятся в [`agent.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/agent.py).

## Runtime Flow

Актуальный граф в [`agent.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/agent.py):

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
- вызывать MCP tools, описанные в [`mcp.json`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/mcp.json)
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

Сборка portable-версии описана в [`build.bat`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/build.bat), который использует `PyInstaller` в режиме `--onefile --windowed`.

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
├─ prompt_code.txt
├─ prompt_dev.txt
├─ env_example.txt
├─ requirements.txt
└─ mcp.json
```

Ключевые каталоги:

- [`core/`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core) runtime, конфиг, checkpointing, policy-логика, recovery, состояние сессий и моделей
- [`tools/`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tools) встроенные инструменты и MCP-интеграция
- [`tools/filesystem_impl/`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tools/filesystem_impl) низкоуровневая реализация filesystem-операций
- [`ui/`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/ui) Qt runtime bridge, главное окно, виджеты и streaming-представление
- [`tests/`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/tests) тесты runtime, tooling, policy, model profiles и GUI

## Конфигурация

Приложение читает настройки из `.env` через [`core/config.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/config.py).

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
- `CHECKPOINT_BACKEND=sqlite|memory|postgres`
- `CHECKPOINT_SQLITE_PATH`
- `CHECKPOINT_POSTGRES_URL`
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

GUI теперь поддерживает несколько сохраняемых профилей моделей через [`core/model_profiles.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/model_profiles.py).

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

- graph checkpoints могут использовать `sqlite`, `memory` или `postgres`
- локально по умолчанию используется `.agent_state/checkpoints.sqlite`
- активная сессия хранится в `.agent_state/session.json`
- индекс сессий хранится в `.agent_state/session_index.json`
- логи запусков сохраняются в формате JSONL в `logs/runs`

Это позволяет приложению восстанавливать состояние чатов и продолжать работу с тем же durable graph thread metadata.

## MCP

[`mcp.json`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/mcp.json) задаёт опциональные MCP-серверы.

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

- [`prompt.txt`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/prompt.txt) базовый системный prompt
- [`prompt_code.txt`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/prompt_code.txt) coding-ориентированный вариант prompt
- [`prompt_dev.txt`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/prompt_dev.txt) дополнительный prompt-ресурс
- [`core/runtime_prompt_policy.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/runtime_prompt_policy.py) runtime overlay с tool и user-input policy
- [`core/recovery_manager.py`](/D:/py_projects/simple_ai_agent/agent+stategraph/v0.62.3b_gui/core/recovery_manager.py) recovery-специфичная логика и подсказки

## Тесты

Запуск тестов:

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

## Требования И Примечания

- для `Gemini` нужен `GEMINI_API_KEY`
- для `OpenAI` нужен `OPENAI_API_KEY`, если не используется совместимый backend
- для web search нужен `TAVILY_API_KEY` и `ENABLE_SEARCH_TOOLS=true`
- для desktop UI нужен `PySide6`
- для дополнительных checkpoint backend нужны соответствующие пакеты зависимостей

## Текущий Tech Stack

- `LangChain`
- `LangGraph`
- `PySide6`
- `pydantic-settings`
- `Tavily`
- `httpx`
- `psutil`
- опциональный MCP через `langchain-mcp-adapters` и `mcp`
