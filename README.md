# Portable Autonomous AI Agent (GUI)

> *“Created by a SysAdmin for developers. Focus on safety, portability, and zero-nonsense execution. No Docker, no heavy environments, just one binary.”*

Десктопный AI-агент с runtime на `LangGraph` и графическим интерфейсом на `PySide6`.  
Работает с файлами, shell-командами, системой, MCP-серверами и веб-поиском.

Запуск из исходников: `python main.py`.  
Сборка в portable `.exe` для Windows: `build.bat`.

![Portable Autonomous AI Agent](./img/01.jpg)

---

## Возможности

- Графовый runtime на `LangGraph` с bounded recovery и self-correction
- GUI: история чатов, streaming transcript, tool cards, approvals, вложения
- Инструменты: filesystem, shell, web search, system info, process management, MCP
- Approval-паузы перед мутирующими и деструктивными действиями
- Автосуммаризация контекста при длинных сессиях
- Несколько профилей моделей с переключением прямо в GUI
- Durable checkpoints — сессии сохраняются между запусками
- Опциональный image input, если модель его поддерживает

---

## Быстрый Старт

Требования: **Python 3.10+**, API-ключ Gemini или OpenAI.

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
Copy-Item env_example.txt .env
# Открой .env и укажи API-ключ
python main.py
```

---

## Portable Сборка

```powershell
.\build.bat
```

Использует `PyInstaller` в режиме `--onefile --windowed`. Результат — один `.exe` без зависимостей.

---

## Runtime Flow

```text
START
  → summarize        # сжать контекст если сессия стала большой
  → update_step
  → agent            # LLM решает: ответить / вызвать tool / recovery
     → approval      # пауза перед мутирующим действием
        → tools
     → tools         # исполнить tool calls
        → recovery   # если tool вернул ошибку
        → update_step
     → recovery      # если агент вернул protocol error или loop
        → update_step
        → END
     → END
```

- `MAX_LOOPS` и per-tool loop guards предотвращают бесконечные циклы.
- Recovery использует stateful error tracking: `attempts_by_strategy`, `progress_markers`, `llm_replan_attempted_for` — адаптивные повторы с учётом уникальных fingerprints ошибок.
- При смене проблемы (новый fingerprint) retry-бюджет сбрасывается; для одной и той же проблемы разрешены несколько `llm_replan` попыток в рамках `SELF_CORRECTION_RETRY_LIMIT`.

---

## GUI

**Transcript** — streaming-ответы, tool cards с аргументами и результатами, summary-сообщения, статусные notice.

**Composer:**
- Вставка файлов через `Add files…`, drag-and-drop или clipboard paste
- `@`-mention popup с файлами и директориями текущего workspace (обновляется динамически)
- Нормализация текста перед отправкой: `\r\n` → `\n`, удаление control-символов
- Лимит 10 000 символов на запрос с inline-предупреждением при усечении

**Горячие клавиши:**

| Клавиша | Действие |
|---|---|
| `Enter` | Отправить сообщение |
| `Shift+Enter` | Новая строка |
| `Ctrl+N` | Новый чат |
| `Ctrl+B` | Показать / скрыть боковую панель |
| `Ctrl+I` | Info popup |
| `↑` / `↓` в пустом composer | История отправленных сообщений |

---

## Безопасность

- Approvals включены по умолчанию для write, delete, move и process-launch операций
- Shell-команды классифицируются перед выполнением (read-only / mutating / destructive)
- MCP tools требуют approval, если `policy.read_only` явно не выставлен в `true`
- Tool errors переводят выполнение в recovery, не игнорируются
- Workspace boundary check: мутирующие операции не могут выйти за пределы рабочей папки
- API keys, bearer tokens и query tokens редактируются из логов через `SensitiveDataFilter`
- `MAX_BACKGROUND_PROCESSES` ограничивает количество фоновых процессов

`request_user_input` — отдельный инструмент для блокирующего выбора пользователя:
- не более одного вызова за ход
- нельзя батчить с другими tool calls в одном ответе
- использовать только когда следующий шаг реально заблокирован одним конкретным выбором пользователя или отсутствующим внешним значением, которое нельзя получить из контекста или tools
- не использовать для approval рискованных действий: это отдельный flow
- задавать ровно один короткий вопрос
- передавать 2-5 коротких взаимоисключающих опций
- если один вариант явно лучший, указывать его через `recommended`
- после resume продолжать с выбранным ответом, а не спрашивать заново в том же ходе

---

## Профили Моделей

Несколько профилей моделей хранятся через `core/model_profiles.py` и переключаются в GUI.

Каждый профиль содержит: провайдера, имя модели, API key, опциональный `base_url` для OpenAI-compatible бэкендов, флаг image input, статус enabled/disabled.

- Активный профиль выбирается в GUI; `.env` используется только для bootstrap начального набора
- Legacy-ключи `MODEL`, `API_KEY`, `BASE_URL` поддерживаются для import/совместимости
- Для Gemini-профилей `base_url` игнорируется

### Автоматическая загрузка списка моделей

При добавлении или редактировании профиля в `Model Profiles` список моделей загружается автоматически — вводить имя вручную не нужно.

**Gemini:** после ввода API Key список загружается автоматически (debounce 600 мс). Модели фильтруются: остаются только те, что поддерживают `generateContent` и принадлежат семействам `gemini` / `gemma`. Автоматически исключаются embedding-, audio-, image- и служебные модели. Список сгруппирован и отсортирован по убыванию версии.

**OpenAI-compatible:** список загружается при заполнении обоих полей — API Key и Base URL. Применяется базовая фильтрация по ключевым словам. Если загрузка не удалась — поле переходит в режим ручного ввода.

Логика реализована в `core/model_fetcher.py`. При переключении между профилями с одним ключом список берётся из кеша без повторного запроса.

### Ротация API-ключей

Агент поддерживает автоматическую ротацию пула API-ключей для каждого профиля. Это позволяет обходить лимиты бесплатных тарифов (Rate Limits) и обеспечивать бесперебойную работу.

- **Как это работает:** вы можете указать несколько ключей для одной модели. Если текущий ключ исчерпал лимит или вернул ошибку, агент двигается к следующему ключу по кругу и повторяет попытку без прерывания сессии. После одного полного круга, если ни один ключ не сработал, выполнение останавливается с сообщением: *"All API keys have been used without success. Please try again later or check your key limits and validity."* — пользователь сам решает, что делать дальше (подождать, обновить ключи и т.д.).
- **Управление:** настройка доступна в GUI через кнопку «цикличные стрелки» рядом с полем API Key в редакторе профиля.
- **Безопасность:** ключи не помечаются как невалидные и не исключаются из пула. Любой ключ может стать рабочим через некоторое время (например, когда снимется rate-limit), поэтому пул остаётся неизменным.

---

## Сессии и Checkpoints

- Graph checkpoints: `sqlite` (по умолчанию) или `memory`
- `.agent_state/checkpoints.sqlite` — durable checkpoint store
- `.agent_state/session.json` — активная сессия
- `.agent_state/session_index.json` — индекс всех сессий
- `logs/runs/` — JSONL-логи каждого запуска

---

## MCP

`mcp.json` задаёт опциональные MCP-серверы. Все серверы выключены по умолчанию.

Поведение policy:

| `policy.read_only` | Поведение |
|---|---|
| `true` | Tool считается read-only, approval не требуется |
| `false` | Требует approval |
| не указан | Консервативный режим: approval по умолчанию |

Минимальный пример подключения удалённого сервера:

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

---

## Prompt Layers

Промпт собирается из нескольких слоёв при каждом вызове агента:

| Слой | Файл / модуль | Содержимое |
|---|---|---|
| Базовый | `prompt.txt` | Системный промпт агента |
| Runtime | `core/runtime_prompt_policy.py` | OS, shell, workspace, дата, tool policy |
| Safety | `core/context_builder.py` | Workspace boundary, shell warning |
| Recovery | `core/recovery_manager.py` | Инструкции при активной ошибке |
| Memory | state: `summary` | Автосуммаризованный контекст прошлых ходов |

---

## Конфигурация

Все настройки читаются из `.env` через `core/config.py`. Скопируй `env_example.txt` в `.env` и заполни нужные поля.

### Провайдер и модели

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PROVIDER` | `gemini` | `gemini` или `openai` |
| `GEMINI_API_KEY` | — | Обязателен для Gemini |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Имя модели Gemini |
| `OPENAI_API_KEY` | — | Обязателен для OpenAI (если нет `OPENAI_BASE_URL`) |
| `OPENAI_MODEL` | `gpt-4o` | Имя модели OpenAI |
| `OPENAI_BASE_URL` | — | Для OpenAI-compatible бэкендов (Ollama и др.) |
| `ACTIVE_MODEL_PROFILE_ID` | — | ID активного профиля модели (для ротации ключей) |
| `SHOW_MODEL_THOUGHTS` | `false` | Legacy-флаг отображения reasoning (runtime выставляет false) |

### Управление runtime

| Переменная | Описание |
|---|---|
| `TEMPERATURE` | Температура модели |
| `MAX_LOOPS` | Максимум шагов на один запрос (default: 50) |
| `TOOL_LOOP_WINDOW` | Окно истории для детекции дублей tool calls |
| `TOOL_LOOP_LIMIT_MUTATING` | Лимит повторов для мутирующих инструментов |
| `TOOL_LOOP_LIMIT_READONLY` | Лимит повторов для read-only инструментов |
| `SELF_CORRECTION_RETRY_LIMIT` | Потолок попыток self-correction |

### Фиче-флаги

| Переменная | Описание |
|---|---|
| `MODEL_SUPPORTS_TOOLS` | Включить tool calling |
| `ENABLE_FILESYSTEM_TOOLS` | Инструменты для работы с файлами |
| `ENABLE_SHELL_TOOL` | Shell-выполнение команд |
| `ENABLE_SEARCH_TOOLS` | Web search через Tavily |
| `ENABLE_SYSTEM_TOOLS` | Информация о системе |
| `ENABLE_PROCESS_TOOLS` | Управление процессами |
| `ENABLE_APPROVALS` | Approval-паузы перед рискованными действиями |
| `ALLOW_EXTERNAL_PROCESS_CONTROL` | Разрешить управление внешними процессами |
| `TAVILY_API_KEY` | Ключ Tavily для web search |

### Лимиты

| Переменная | Описание |
|---|---|
| `MAX_FILE_SIZE` | Максимальный размер файла (поддерживает `300MB`, `4096`) |
| `MAX_READ_LINES` | Лимит строк при чтении файла |
| `MAX_TOOL_OUTPUT` | Лимит символов в выводе инструмента |
| `MAX_SEARCH_CHARS` | Лимит символов в результатах поиска |
| `MAX_BACKGROUND_PROCESSES` | Лимит фоновых процессов |
| `STREAM_TEXT_MAX_CHARS` | Лимит символов streaming-текста |
| `STREAM_EVENTS_MAX` | Лимит streaming-событий |
| `STREAM_TOOL_BUFFER_MAX` | Буфер streaming tool output |

### Суммаризация и retry

| Переменная | Описание |
|---|---|
| `SESSION_SIZE` | Порог токенов для запуска суммаризации |
| `SUMMARY_KEEP_LAST` | Сколько последних сообщений не трогать при суммаризации |
| `MAX_RETRIES` | Число попыток при ошибке LLM |
| `RETRY_DELAY` | Задержка между попытками (секунды) |

### Персистентность

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CHECKPOINT_BACKEND` | `sqlite` | `sqlite` или `memory` |
| `CHECKPOINT_SQLITE_PATH` | `.agent_state/checkpoints.sqlite` | Путь к БД |
| `SESSION_STATE_PATH` | `.agent_state/session.json` | Активная сессия |
| `RUN_LOG_DIR` | `logs/runs` | Директория JSONL-логов |
| `LOG_FILE` | `logs/agent.log` | Файл лога |
| `PROMPT_PATH` | `prompt.txt` | Путь к системному промпту |
| `MCP_CONFIG_PATH` | `mcp.json` | Путь к конфигу MCP |

### Диагностика

| Переменная | Описание |
|---|---|
| `DEBUG` | Включить debug-режим |
| `LOG_LEVEL` | Уровень логирования (`INFO`, `DEBUG`, `WARNING`) |
| `STRICT_MODE` | Строгий режим: без догадок, точное выполнение |

---

## Структура Проекта

```text
.
├── agent.py              # Сборка и маршрутизация LangGraph графа
├── main.py               # Точка входа
├── build.bat             # Сборка portable .exe
├── prompt.txt            # Системный промпт агента
├── prompt_dev.txt        # Dev-вариант промпта для отладки
├── mcp.json              # Конфиг MCP-серверов
├── env_example.txt       # Шаблон .env
├── requirements.txt
├── core/                 # Ядро runtime (~31 модуль)
│   ├── nodes/            # 11 узлов LangGraph (mixin-архитектура)
│   │   ├── __init__.py   # AgentNodes — сборка всех mixin в единый класс
│   │   ├── base.py       # BaseMixin — логирование, state introspection, turn_id
│   │   ├── llm.py        # LLMMixin — вызов LLM, token streaming, tool binding
│   │   ├── context.py    # ContextMixin — сборка контекста для LLM-промпта
│   │   ├── tool_preflight.py  # ToolPreflightMixin — metadata, JSON-schema, loop detection, approval gate
│   │   ├── protocol.py   # ProtocolMixin — tool issues, message filtering, error merging
│   │   ├── summarize.py  # SummarizeMixin — compact history при превышении токенов
│   │   ├── agent.py      # AgentMixin — фасад agent_node, result building
│   │   ├── approval.py   # ApprovalMixin — interrupt для мутирующих операций
│   │   ├── tools.py      # ToolsMixin — dispatch tool execution, parallel batching
│   │   └── recovery.py   # RecoveryMixin — bounded recovery, self-correction ceiling
│   ├── config.py         # Pydantic-settings конфигурация из .env
│   ├── state.py          # TypedDict состояния графа
│   ├── checkpointing.py  # SQLite / memory checkpoint store
│   ├── context_builder.py # Сборка runtime-промпта (OS, shell, safety, recovery)
│   ├── recovery_manager.py # Recovery strategies и retry logic
│   ├── self_correction_engine.py # Repair fingerprints, strategy dispatch
│   ├── tool_executor.py  # Async execution, parallel coordinator, batch runner
│   ├── tool_policy.py   # ToolMetadata, approval rules
│   ├── policy_engine.py  # Shell command classification (read-only / mutating / destructive)
│   ├── safety_policy.py  # Workspace boundary, safety levels
│   ├── model_fetcher.py  # Автозагрузка и фильтрация моделей из Gemini / OpenAI API
│   ├── model_profiles.py # Профили моделей с переключением в GUI
│   ├── api_key_rotation.py # Circular key-pool rotation without invalidation
│   ├── session_store.py  # Persist сессий и индекса
│   ├── session_utils.py  # Session helpers, ID generation
│   ├── run_logger.py     # JSONL логирование каждого run
│   ├── runtime_prompt_policy.py # Runtime-слой промпта (дата, ОС, политика)
│   ├── text_utils.py     # Форматирование, compact, elide, truncate
│   ├── message_context.py # Message context helper
│   ├── message_utils.py  # Stringify, compact content
│   ├── input_sanitizer.py # Очистка input от control-символов
│   ├── multimodal.py     # Image input, base64, MIME detection
│   ├── errors.py          # Error formatting, ErrorType
│   ├── tool_args.py       # Canonicalize, inspect tool args
│   ├── tool_results.py    # Parse tool execution results
│   ├── tool_issues.py     # Build tool issues, fingerprints
│   ├── node_errors.py     # Provider-specific errors
│   ├── constants.py       # Константы промптов, лимитов
│   ├── validation.py      # Validation helpers
│   ├── fast_copy.py       # Clipboard-safe copy helpers
│   └── utils.py           # Misc utilities
├── tools/                # Встроенные инструменты и MCP-интеграция (7 файлов)
│   ├── filesystem.py      # read_file, write_file, edit_file, list_directory, search, tail…
│   ├── filesystem_impl/   # Низкоуровневая реализация filesystem-операций
│   │   ├── __init__.py
│   │   ├── manager.py     # FilesystemManager
│   │   ├── editing.py     # Diff-based editing helpers
│   │   └── reader.py      # File reading with bounds
│   ├── local_shell.py     # cli_exec, shell streaming, command classification
│   ├── search_tools.py    # web_search, fetch_content, batch_web_search (Tavily)
│   ├── system_tools.py    # get_system_info, get_public_ip, network_info
│   ├── process_tools.py   # run_background_process, process management
│   ├── user_input_tool.py # request_user_input — блокирующий выбор пользователя
│   └── tool_registry.py   # ToolRegistry — загрузка, metadata, MCP clients
├── ui/                    # Qt GUI (~13 модулей)
│   ├── main_window.py     # Entry point re-export
│   ├── main_window_state.py # State machine окна (busy, initialized, error)
│   ├── runtime.py         # Runtime snapshot builder
│   ├── runtime_payloads.py # Payload builders для GUI events
│   ├── runtime_session.py # Session coordination, load/save
│   ├── runtime_worker.py  # Async worker thread, event loop для графа
│   ├── streaming.py       # StreamEvent, event routing
│   ├── theme.py           # QSS stylesheet, палитра, icons
│   ├── tool_message_utils.py # Tool message formatting для GUI
│   ├── visibility.py      # Widget visibility helpers
│   ├── window_components/ # Builders и controllers главного окна (7 файлов)
│   │   ├── __init__.py
│   │   ├── main_window.py      # MainWindow — главное окно, menu, layout
│   │   ├── inspector_controller.py # Inspector panel controller
│   │   ├── sidebar_controller.py   # Sidebar controller
│   │   ├── status_bar_manager.py   # Status bar manager
│   │   ├── menu_builder.py         # Menu builder
│   │   └── workspace_builder.py    # Workspace layout builder
│   └── widgets/           # Виджеты интерфейса (11 файлов)
│       ├── __init__.py    # Экспорт всех public-классов
│       ├── transcript.py  # ChatTranscriptWidget, ConversationTurnWidget
│       ├── tools.py       # ToolCardWidget, CliExecWidget
│       ├── tool_group.py  # ToolGroupWidget — контейнер для группы tool cards
│       ├── composer.py    # ComposerTextEdit — редактор ввода с @-mentions
│       ├── messages.py    # AssistantMessageWidget, UserMessageWidget, NoticeWidget…
│       ├── foundation.py  # Базовые виджеты: CollapsibleSection, CodeBlockWidget, ElidedLabel…
│       ├── attachments.py # ImageAttachmentChipWidget, ImageAttachmentStripWidget
│       ├── dialogs.py     # ApprovalDialog, ModelSettingsDialog, InfoPopupDialog
│       ├── panels.py      # OverviewPanelWidget, InspectorPanelWidget, ToolsPanelWidget
│       └── sidebar.py     # SessionSidebarWidget, SessionListModel
└── tests/                # 17 тестовых файлов (~370K байт)
```

---

## Тесты

18 тестовых файлов:

| Файл | Что покрывает |
|---|---|
| `test_model_fetcher.py` | Фильтрация моделей, нормализация имён, коды ошибок API, fallback-логика |
| `test_api_key_rotation.py` | Circular key-pool rotation, exhaustion handling, auth/rate-limit error classification |
| `test_cli_ux.py` | GUI: composer, transcript, tool cards, streaming, sidebar, attachments, history, mentions, approvals |
| `test_stream_and_filesystem.py` | Streaming events, filesystem tools, tool output, cli_exec |
| `test_runtime_refactor.py` | Runtime payloads, transcript restore, tool group logic, run lifecycle |
| `test_critic_graph.py` | LangGraph workflow, node orchestration, tool batching |
| `test_self_correction_engine.py` | Recovery strategies, fingerprinting, loop detection |
| `test_policy_engine.py` | Shell command classification, tool metadata, approval rules |
| `test_model_profiles.py` | Profile CRUD, switching, validation, serialization |
| `test_session_utils.py` | Session ID generation, index management |
| `test_runtime_session_coordination.py` | Session state coordination, load/save |
| `test_tooling_refactor.py` | Tool registry, MCP loading, tool metadata |
| `test_input_sanitizer.py` | Input sanitization, truncation, control chars |
| `test_logging_config.py` | Log configuration, sensitive data filtering |
| `test_intent_engine.py` | Intent parsing, routing |
| `test_main_window_facade.py` | MainWindow facade behavior |
| `test_ui_helpers.py` | UI helper utilities |
| `test_refactor_services.py` | Service refactoring, internal APIs |
| `test_runtime_payloads.py` | Payload builders, serialization |

Запуск:

```powershell
venv\Scripts\python.exe -m pytest
```

---

## Зависимости

| Пакет | Назначение |
|---|---|
| `langgraph` | Граф агента и state management |
| `langchain` | LLM abstraction, tool calling |
| `langchain-google-genai` | Gemini provider |
| `langchain-openai` | OpenAI / compatible provider |
| `langchain-mcp-adapters` | MCP интеграция |
| `PySide6` | GUI |
| `pydantic-settings` | Конфигурация через `.env` |
| `tiktoken` | Подсчёт токенов для суммаризации |
| `tavily-python` | Web search |
| `psutil` | Системные инструменты и процессы |
| `httpx` | HTTP для MCP и fetch |
| `aiofiles` | Async файловые операции |
| `mcp` | Model Context Protocol |
| `requests` | HTTP-клиент (Google API, Tavily) |
| `sqlite-vec` | Vector-расширение для SQLite checkpoints |
