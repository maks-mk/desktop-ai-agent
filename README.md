# Autonomous AI Agent (GUI)

Версия: `v0.65.4b`

Desktop AI-агент с графовым runtime (`LangGraph`) и GUI на `PySide6`.
Приложение ориентировано на повседневную разработку: работа с файлами проекта, запуск инструментов, безопасные approvals для рискованных действий, структурированный user-choice flow через `interrupt()` и удобная история чатов по проектам.

## Ключевые возможности

- Чат-интерфейс с persistent-сессиями по проектам.
- Sidebar с историей чатов, быстрым переключением и удалением сессий.
- Transcript с компактным отображением шагов агента и инструментов.
- Inline-карточка выбора для `request_user_input`:
  - tool вызывает `interrupt()` и паузит граф,
  - ответ пользователя резюмирует текущий run,
  - `Свой вариант` отправляет строку обратно в тот же interrupt.
- Composer с автоподстройкой высоты и отправкой `Enter` (`Shift+Enter` — новая строка).
- Быстрое добавление файлов через `@`:
  - popup-список рядом с курсором,
  - фильтрация по имени/пути,
  - навигация `↑/↓`, выбор `Enter`, закрытие `Esc`,
  - приоритет файлов из корня проекта, затем из подпапок.
- История запросов в composer:
  - навигация `↑/↓` как в терминале,
  - работает по текущей сессии,
  - восстановление после перезапуска из transcript.
- Approval-поток для действительно destructive инструментов (`Approve`, `Deny`, `Always for this session`).
- Отдельный user-choice flow для обычных вопросов к пользователю, не требующий approval.
- Поддержка MCP-инструментов (включая Context7 при соответствующей конфигурации).
- Runtime-статусы (`Thinking`, `Self-correcting`, `Ready`, ошибки) в UI.
- Профили моделей в GUI:
  - окно `Settings` (CRUD профилей),
  - селектор активной модели рядом с `+` в composer,
  - безопасное переключение только вне активного выполнения и approval,
  - применение выбранной модели к **следующему** запросу без сброса чата.
- User-choice и approval flow строятся на LangGraph `interrupt()` и `Command(resume=...)`, а не на парсинге текста ответа модели.
- Автоподхват модели из `.env`:
  - при первом запуске env-модель записывается в настройки,
  - при последующих запусках env-модели аккуратно подмешиваются в профильный список,
  - одинаковые профили не дублируются.

## Архитектура

- [`agent.py`](agent.py): сборка агентного графа, LLM, реестр инструментов и checkpoint backend.
- [`main.py`](main.py): тонкая точка входа GUI-приложения.
- [`ui/main_window.py`](ui/main_window.py): главное окно, меню, sidebar, transcript, composer и обработка UI-событий.
- [`ui/runtime.py`](ui/runtime.py): runtime-контроллер, запуск графа, стриминг событий, переключение и восстановление сессий, сборка payload для UI.
- [`ui/streaming.py`](ui/streaming.py): обработка LangGraph stream-событий, tool lifecycle, статусы, notices и summary-сигналы.
- [`ui/widgets/`](ui/widgets): UI-виджеты, разложенные по зонам ответственности.
- [`ui/theme.py`](ui/theme.py): палитра, размеры и stylesheet для PySide6.
- [`ui/visibility.py`](ui/visibility.py): контракт внутренних assistant/recovery сообщений и политика их показа в UI.
- [`core/nodes.py`](core/nodes.py): узлы графа, recovery-loop (`stability_guard` + `recovery`), structured repair strategies, loop-guards, tool-preflight и safety flow.
- [`core/context_builder.py`](core/context_builder.py): сборка и нормализация LLM-контекста с provider-safe ordering.
- [`core/session_store.py`](core/session_store.py): хранение и индекс сессий.
- [`core/config.py`](core/config.py): загрузка env-конфигурации.
- [`tools/tool_registry.py`](tools/tool_registry.py): локальные и MCP инструменты, метаданные, политика безопасности.
- [`tools/user_input_tool.py`](tools/user_input_tool.py): `request_user_input`, который использует `interrupt()` для пользовательского выбора.

## Модульная Карта

### `core/`

- [`core/config.py`](core/config.py): типизированные настройки приложения и `.env`-валидация.
- [`core/constants.py`](core/constants.py): базовые константы и пути проекта.
- [`core/checkpointing.py`](core/checkpointing.py): backend checkpoint-хранилища для LangGraph.
- [`core/errors.py`](core/errors.py): единый формат ошибок.
- [`core/intent_engine.py`](core/intent_engine.py): детерминированная классификация turn intent и follow-up сигналов.
- [`core/logging_config.py`](core/logging_config.py): конфигурация логирования приложения.
- [`core/message_utils.py`](core/message_utils.py): утилиты для работы с сообщениями и текстом.
- [`core/model_profiles.py`](core/model_profiles.py): профили моделей и merge логика для GUI.
- [`core/nodes.py`](core/nodes.py): узлы LangGraph, agent flow, tool execution и self-correction.
- [`core/policy_engine.py`](core/policy_engine.py): turn policy и правила fallback/approval для tool calls.
- [`core/run_logger.py`](core/run_logger.py): JSONL-логирование событий запуска.
- [`core/safety_policy.py`](core/safety_policy.py): ограничения на инструменты, файлы и процессы.
- [`core/self_correction_engine.py`](core/self_correction_engine.py): ремонт tool-call аргументов и repair plan.
- [`core/session_store.py`](core/session_store.py): persistence для чатов и индексов сессий.
- [`core/session_utils.py`](core/session_utils.py): вспомогательная логика по сессиям.
- [`core/state.py`](core/state.py): схема состояния LangGraph.
- [`core/stream_processor.py`](core/stream_processor.py): сборка и буферизация стрим-событий.
- [`core/text_utils.py`](core/text_utils.py): компактное форматирование и текстовые helper-ы.
- [`core/tool_policy.py`](core/tool_policy.py): метаданные инструментов и default policy.
- [`core/tool_results.py`](core/tool_results.py): парсинг результатов инструментов и статусов.
- [`core/ui_theme.py`](core/ui_theme.py): compatibility-shim на `ui/theme.py`.
- [`core/utils.py`](core/utils.py): общие helper-ы.
- [`core/validation.py`](core/validation.py): валидация результатов и входных данных.

### `ui/`

- [`ui/main_window.py`](ui/main_window.py): центральный координатор GUI и wiring сигналов.
- [`ui/runtime.py`](ui/runtime.py): мост между runtime и интерфейсом, snapshot/payload, approvals, user-choice, восстановление transcript.
- [`ui/streaming.py`](ui/streaming.py): интерпретация LangGraph stream, tool start/finish, статусы и уведомления.
- [`ui/theme.py`](ui/theme.py): тема, цвета, размеры и генерация stylesheet.
- [`ui/visibility.py`](ui/visibility.py): внутренние сообщения агента и правила их показа в transcript.
- [`ui/tool_message_utils.py`](ui/tool_message_utils.py): единый разбор `tool_args` и `tool_duration` из `ToolMessage`.
- [`ui/widgets/foundation.py`](ui/widgets/foundation.py): базовые текстовые, code и diff-виджеты, общие helper-ы.
- [`ui/widgets/composer.py`](ui/widgets/composer.py): composer, история запросов, paste-логика, `@`-mentions.
- [`ui/widgets/sidebar.py`](ui/widgets/sidebar.py): список сессий и sidebar-логика.
- [`ui/widgets/panels.py`](ui/widgets/panels.py): info/help панели и вспомогательные карточки.
- [`ui/widgets/dialogs.py`](ui/widgets/dialogs.py): диалоги настроек и approval.
- [`ui/widgets/messages.py`](ui/widgets/messages.py): user/assistant bubbles, notices, статусы и summary.
- [`ui/widgets/tools.py`](ui/widgets/tools.py): tool cards, CLI output, diffs и tool-result presentation.
- [`ui/widgets/transcript.py`](ui/widgets/transcript.py): сборка turn-ов и восстановление transcript из payload.

### `tools/`

- [`tools/tool_registry.py`](tools/tool_registry.py): загрузка локальных и MCP инструментов.
- [`tools/filesystem.py`](tools/filesystem.py): файловые инструменты верхнего уровня.
- [`tools/delete_tools.py`](tools/delete_tools.py): удаление файлов и директорий.
- [`tools/local_shell.py`](tools/local_shell.py): shell/terminal execution tools.
- [`tools/process_tools.py`](tools/process_tools.py): управление процессами.
- [`tools/search_tools.py`](tools/search_tools.py): web/search tools.
- [`tools/system_tools.py`](tools/system_tools.py): системные и диагностические инструменты.

## Tool Routing

- Отдельный `tool selector` удалён: агент получает полный набор доступных инструментов на turn.
- Безопасность исполнения обеспечивают метаданные инструментов, `PolicyEngine`, `stability_guard` и реальные platform/workspace boundaries.
- `request_user_input` всегда доступен и не проходит через approval flow.
- Словарный `intent` больше не участвует в боевом control flow; живая message-context логика вынесена в `MessageContextHelper`, а `IntentEngine` оставлен только для совместимости.
- Если turn-policy требует инструменты, а модель отвечает без tool-calls, включается детерминированный путь self-correction/handoff (reason: `action_requires_tools`).
- Для коротких follow-up (`продолжай`, `еще раз`) при наличии tool-контекста приоритет отдаётся контексту предыдущего turn, а не только словарю ключевых фраз.

## Что Такое `request_user_input`

- `request_user_input` — это отдельный инструмент в общем списке tools, который модель видит так же, как остальные инструменты (по имени tool).
- Он нужен для ситуаций, где требуется осознанный выбор пользователя и нельзя безопасно продолжать автоматически.
- Вызов `request_user_input(...)` не делает внешнего действия (не пишет файл и не запускает процесс): tool вызывает `interrupt(...)` и ставит граф на паузу.
- UI показывает inline-карточку выбора, а затем резюмирует тот же run через `Command(resume=...)`.
- Значение, переданное в `resume`, возвращается обратно как результат вызова `request_user_input`.
- Это штатный HITL-поток LangGraph; обычные текстовые вопросы в ответе модели не заменяют этот механизм.

## Поведение self-correction

- В графе нет отдельного LLM-узла критика: контроль стабильности выполняют `stability_guard` и явный узел `recovery`.
- При ошибке инструмента runtime сначала пробует детерминированные recovery-стратегии (`normalize_args`, `switch_tool`, `verify_side_effect`, `repair_then_rerun`, `resume_after_transient_failure`), затем при необходимости включает LLM-replan.
- Остановка допустима только при подтверждённом успехе, реальном внешнем блокере, platform/workspace boundary или опасной необратимой операции, требующей явного approval.

## Хранение данных

- Checkpoint state: `CHECKPOINT_SQLITE_PATH` (или другой backend).
- Активная сессия: `SESSION_STATE_PATH` (по умолчанию `.agent_state/session.json`).
- Индекс чатов: `.agent_state/session_index.json`.
- Логи запусков: `RUN_LOG_DIR` (JSONL).
- Профили моделей: `.agent_state/config.json`:
  - `active_profile: str | null`
  - `profiles: [{id, provider, model, api_key, base_url}]`

При старте runtime текущая активная сессия и transcript автоматически восстанавливаются.

## Профили моделей и ENV

- Источник правды для профилей в GUI — `.agent_state/config.json`.
- Если `config.json` отсутствует или пустой по профилям, профиль создается из `.env`.
- Если `config.json` уже существует, env-профили добавляются без дублирования существующих.
- `active_profile` сохраняется при merge (если валиден), иначе выбирается первый доступный профиль.
- Для `provider=gemini` поле `base_url` в UI отключено и не используется.
- Для `provider=openai` `base_url` доступен и сохраняется.

## Быстрый старт

1. Создать виртуальное окружение:

```bash
python -m venv venv
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Подготовить `.env`:

```powershell
Copy-Item env_example.txt .env
```

4. Запустить приложение:

```bash
python main.py
```

## Основные переменные окружения

### LLM-провайдер

- `PROVIDER=gemini|openai`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`
- (опционально, generic bootstrap) `MODEL`, `API_KEY`, `BASE_URL`

### Runtime и persistence

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
- `SELF_CORRECTION_ENABLE_AUTO_REPAIR`
- `SELF_CORRECTION_MAX_AUTO_REPAIRS`
- `ENABLE_APPROVALS`
- `ALLOW_EXTERNAL_PROCESS_CONTROL`
- Отдельного флага для `request_user_input` нет, tool всегда включен.

### Лимиты

- `MAX_TOOL_OUTPUT`
- `MAX_SEARCH_CHARS`
- `MAX_FILE_SIZE`
- `MAX_READ_LINES`
- `MAX_BACKGROUND_PROCESSES`
- `STREAM_TEXT_MAX_CHARS`
- `STREAM_EVENTS_MAX`
- `STREAM_TOOL_BUFFER_MAX`

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

## Управление сессиями

- `New Session` создает новый чат в текущем проекте.
- `Open Folder` переключает рабочую папку и контекст сессий.
- Sidebar показывает историю чатов с группировкой по проектам.

## Горячие клавиши в Composer

- `Enter` — отправить запрос.
- `Shift+Enter` — новая строка.
- `@` — открыть popup файлов.
- `↑/↓` — навигация по popup или истории запросов (в зависимости от контекста).
- `Esc` — закрыть popup.

## Тесты

Запуск полного набора:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Примечание: GUI-наборы (`PySide6`) проверяйте через локальный `venv`. Вне `venv` часть UI-тестов может не стартовать из-за отсутствующих desktop-зависимостей.

Ключевые наборы:

- [`tests/test_cli_ux.py`](tests/test_cli_ux.py)
- [`tests/test_critic_graph.py`](tests/test_critic_graph.py)
- [`tests/test_intent_engine.py`](tests/test_intent_engine.py)
- [`tests/test_model_profiles.py`](tests/test_model_profiles.py)
- [`tests/test_runtime_refactor.py`](tests/test_runtime_refactor.py)
- [`tests/test_stream_and_filesystem.py`](tests/test_stream_and_filesystem.py)
- [`tests/test_tooling_refactor.py`](tests/test_tooling_refactor.py)
- [`tests/test_stability_guard.py`](tests/test_stability_guard.py)

## Безопасность

- Не храните реальные API-ключи в репозитории.
- Включайте `ENABLE_SHELL_TOOL=true` только при необходимости; destructive shell-операции всё ещё должны проходить через policy/safety boundaries.
- Для production persistence рекомендуется `postgres` backend.
- Проверяйте политику инструментов перед запуском mutating/destructive действий.
