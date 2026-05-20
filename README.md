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
| `ENABLE_MODEL_REASONING` | `true` | Включает provider-side reasoning/thinking для поддерживаемых моделей |
| `MODEL_REASONING_EFFORT` | `medium` | Усилие reasoning для OpenAI/OpenAI-compatible моделей (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) |
| `GEMINI_THINKING_BUDGET` | `4096` | Thinking budget для `gemini-2.5*` / `gemini-3*`; старые Gemini-модели получают запрос без этого параметра |
| `PROVIDER_REGISTRY_PATH` | `provider_registry.json` | Реестр OpenAI-compatible агрегаторов и их reasoning-параметров |
| `ACTIVE_MODEL_PROFILE_ID` | — | ID активного профиля модели (для ротации ключей) |
| `SHOW_MODEL_THOUGHTS` | `false` | Legacy-флаг отображения reasoning (runtime выставляет false) |

### Добавление OpenAI-compatible агрегаторов

Подробная инструкция по `provider_registry.json`, полям схемы, matching rules и добавлению новых агрегаторов находится в [`provider_registry_guide.md`](provider_registry_guide.md).

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
| `ENABLE_TEXT_TOOL_CALL_RECOVERY` | Диагностический fallback для провайдеров, которые пишут `call:...<tool_call|>` текстом вместо structured `tool_calls`; по умолчанию выключен |
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
| `MODEL_PROFILE_CONFIG_PATH` | `.agent_state/config.json` | Файл профилей моделей и активного профиля |
| `RUN_LOG_DIR` | `logs/runs` | Директория JSONL-логов |
| `LOG_FILE` | `logs/agent.log` | Файл лога |
| `PROMPT_PATH` | `prompt.txt` | Путь к системному промпту |
| `MCP_CONFIG_PATH` | `mcp.json` | Путь к конфигу MCP |

### Диагностика

| Переменная | Описание |
|---|---|
| `DEBUG` | Включить debug-режим |
| `LOG_LEVEL` | Уровень логирования (`INFO`, `DEBUG`, `WARNING`) |
| `DEBUG_REASONING_STREAM` | Отдельный подробный лог reasoning/thinking stream для диагностики провайдеров |
| `STRICT_MODE` | Строгий режим: без догадок, точное выполнение |

---

## Структура Проекта

Подробная карта модулей находится в [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Ниже — краткая рабочая схема.

```text
.
├── main.py                # Точка входа GUI
├── agent.py               # Сборка LLM, tools и LangGraph workflow
├── prompt.txt             # Основной системный промпт
├── prompt_dev.txt         # Дополнительный dev/devops-промпт
├── mcp.json               # Конфигурация MCP-серверов
├── env_example.txt        # Шаблон .env
├── provider_registry.json # Reasoning kwargs для OpenAI-compatible провайдеров
├── provider_registry_guide.md
├── build.bat              # Сборка portable .exe
├── requirements.txt
├── core/                  # Ядро агента: config, state, policies, recovery, provider registry
│   └── nodes/             # Узлы LangGraph: context, llm, agent, tools, approval, recovery
├── tools/                 # Filesystem, shell, search, system, process, user input, MCP registry
│   └── filesystem_impl/   # Низкоуровневые filesystem helpers
├── ui/                    # PySide6 GUI, runtime worker, streaming/status handling
│   ├── window_components/ # Главное окно, sidebar, inspector, menu, status bar
│   └── widgets/           # Transcript, composer, messages, tool cards, dialogs, panels
├── docs/
│   └── PROJECT_STRUCTURE.md
├── tests/                 # Runtime, UI, tools, provider registry, logging, policies
├── .agent_state/          # Локальное состояние, профили, checkpoints
└── logs/                  # JSONL/runtime/debug logs
```

---

## Тесты

20 тестовых файлов:

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
| `test_provider_registry.py` | Matching OpenAI-compatible провайдеров и reasoning kwargs |
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
