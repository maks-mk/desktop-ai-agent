# Конфигурация

Все настройки читаются из `.env` через `core/config.py`. Скопируй `env_example.txt` в `.env` и заполни нужные поля.

---

## Провайдер и модели

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

Подробная инструкция по `provider_registry.json`, полям схемы, matching rules и добавлению новых агрегаторов находится в [`provider_registry_guide.md`](./provider_registry_guide.md).

---

## Управление runtime

| Переменная | Описание |
|---|---|
| `TEMPERATURE` | Температура модели |
| `TOP_P` | Nucleus sampling (`0.95` по умолчанию; `none`/пусто — не отправлять) |
| `TOP_K` | Candidate pool limit для Gemini/поддерживаемых SDK (`40` по умолчанию; для OpenAI-compatible не отправляется) |
| `MAX_LOOPS` | Максимум шагов на один запрос (default: 50) |
| `TOOL_LOOP_WINDOW` | Окно истории для детекции дублей tool calls |
| `TOOL_LOOP_LIMIT_MUTATING` | Лимит повторов для мутирующих инструментов |
| `TOOL_LOOP_LIMIT_READONLY` | Лимит повторов для read-only инструментов |
| `SELF_CORRECTION_RETRY_LIMIT` | Потолок попыток self-correction |

---

## Фиче-флаги

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

---

## Лимиты

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

---

## Суммаризация и retry

| Переменная | Описание |
|---|---|
| `SESSION_SIZE` | Порог токенов для запуска суммаризации |
| `SUMMARY_RESERVED_TOKENS` | Запас на системные инструкции, tool schemas и provider overhead |
| `SUMMARY_KEEP_LAST` | Сколько последних сообщений не трогать при суммаризации |
| `MAX_RETRIES` | Число попыток при ошибке LLM |
| `RETRY_DELAY` | Базовая задержка между попытками (секунды); также используется как base delay для stream-repair backoff |

---

## Персистентность

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

---

## Диагностика

| Переменная | Описание |
|---|---|
| `DEBUG` | Включить debug-режим |
| `LOG_LEVEL` | Уровень логирования (`INFO`, `DEBUG`, `WARNING`) |
| `DEBUG_REASONING_STREAM` | Отдельный подробный лог reasoning/thinking stream для диагностики провайдеров |
| `STRICT_MODE` | Строгий режим: без догадок, точное выполнение |
