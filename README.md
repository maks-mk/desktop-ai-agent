# Autonomous AI Agent (GUI)

Версия: `v0.65.7b`

Desktop AI-агент с графовым runtime (`LangGraph`) и GUI на `PySide6`.
Приложение ориентировано на повседневную разработку: работа с файлами проекта, запуск инструментов, безопасные approvals для рискованных действий, структурированный user-choice flow через `interrupt()` и удобная история чатов по проектам.

## Структура проекта

```text
v0.65.7b/
├─ agent.py
├─ main.py
├─ prompt.txt
├─ core/
├─ tools/
├─ ui/
└─ tests/
```

- `core/` — граф, policy/recovery, контекст, состояние и persistence.
- `tools/` — локальные и MCP инструменты, реестр и safety metadata.
- `ui/` — окно, runtime-контроллер, stream-processing и виджеты.
- `tests/` — unit/integration regression наборы.

## Ключевые возможности

- Чат-интерфейс с persistent-сессиями по проектам.
- Надежное восстановление transcript после перезапуска:
  - сохраняются и корректно рендерятся не только tool-блоки, но и текстовые ответы assistant,
  - поддерживаются mixed-форматы assistant content (`str`/`list`/вложенные блоки).
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
- Inline-статус `Thinking` стабильно показывается перед выводом в каждом новом run (включая длинный transcript и повторные запросы).
- Надежный multimodal input для разных провайдеров:
  - для `gemini` используется стандартный LangChain image block,
  - для `openai`/OpenAI-compatible используется provider-native `image_url` (`data:<mime>;base64,...`),
  - для strict OpenAI-compatible API все runtime `SystemMessage` объединяются в один system-prefix, чтобы избежать ошибок формата вида `System message must be at the beginning`.
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
- [`core/nodes.py`](core/nodes.py): узлы графа `summarize -> classify_turn -> agent -> {approval|tools|recovery|END}`, bounded-recovery, loop-guards, tool-preflight и safety flow.
- [`core/context_builder.py`](core/context_builder.py): сборка и нормализация LLM-контекста с provider-safe ordering.
- [`core/runtime_prompt_policy.py`](core/runtime_prompt_policy.py): runtime policy overlays и динамические system-инструкции (OS/shell/path style, workspace/cwd, timezone, tool-access, `request_user_input` rules). Языковая политика по умолчанию задается в `prompt.txt`.
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
- [`core/message_context.py`](core/message_context.py): history-derived message-context helpers для follow-up и tool-context сигналов.
- [`core/logging_config.py`](core/logging_config.py): конфигурация логирования приложения.
- [`core/message_utils.py`](core/message_utils.py): утилиты для работы с сообщениями и текстом.
- [`core/model_profiles.py`](core/model_profiles.py): профили моделей и merge логика для GUI.
- [`core/nodes.py`](core/nodes.py): узлы LangGraph, agent flow, tool execution и self-correction.
- [`core/policy_engine.py`](core/policy_engine.py): turn policy и правила fallback/approval для tool calls.
- [`core/recovery_manager.py`](core/recovery_manager.py): стратегии восстановления, handoff и UI-notices для recovery flow.
- [`core/runtime_prompt_policy.py`](core/runtime_prompt_policy.py): единый builder runtime contract на основе фактического окружения.
- [`core/run_logger.py`](core/run_logger.py): JSONL-логирование событий запуска.
- [`core/safety_policy.py`](core/safety_policy.py): ограничения на инструменты, файлы и процессы.
- [`core/self_correction_engine.py`](core/self_correction_engine.py): ремонт tool-call аргументов и repair plan.
- [`core/session_store.py`](core/session_store.py): persistence для чатов и индексов сессий.
- [`core/session_utils.py`](core/session_utils.py): вспомогательная логика по сессиям.
- [`core/state.py`](core/state.py): схема состояния LangGraph.
- [`core/text_utils.py`](core/text_utils.py): компактное форматирование и текстовые helper-ы.
- [`core/tool_policy.py`](core/tool_policy.py): метаданные инструментов и default policy.
- [`core/tool_results.py`](core/tool_results.py): парсинг результатов инструментов и статусов.
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
- [`tools/filesystem_impl/`](tools/filesystem_impl): внутренняя реализация файлового слоя (`manager`, `pathing`, `editing`).
- [`tools/local_shell.py`](tools/local_shell.py): shell/terminal execution tools.
- [`tools/process_tools.py`](tools/process_tools.py): управление процессами.
- [`tools/search_tools.py`](tools/search_tools.py): web/search tools.
- [`tools/system_tools.py`](tools/system_tools.py): системные и диагностические инструменты.
- [`tools/user_input_tool.py`](tools/user_input_tool.py): HITL-инструмент выбора пользователя через `interrupt()/resume`.

## Tool Routing

- Отдельный `tool selector` удалён: агент получает полный набор доступных инструментов на turn.
- Безопасность исполнения обеспечивают метаданные инструментов, `PolicyEngine`, approval flow и реальные platform/workspace boundaries.
- `request_user_input` всегда доступен и не проходит через approval flow.
- Multimodal routing учитывает провайдера:
  - `openai`/OpenAI-compatible получает картинки в формате `image_url`,
  - остальные провайдеры получают стандартные LangChain image blocks.
- Для tool-mode в системном overlay зафиксировано требование: перед каждым tool-call агент формулирует короткое намерение, после чего отправляет структурированный вызов инструмента.
- `intent` используется в `classify_turn` только для определения режима хода (`chat`/`inspect`/`act`/`recover`) и требования evidence; доступность инструментов от `intent` не зависит.
- Если turn-policy требует evidence, а модель завершает шаг без tool-calls и без подтверждения, включается детерминированный путь self-correction/handoff (reason: `action_requires_tools`).
- Для коротких follow-up (`продолжай`, `еще раз`) при наличии tool-контекста приоритет отдаётся контексту предыдущего turn, а не только словарю ключевых фраз.

## Что Такое `request_user_input`

- `request_user_input` — это отдельный инструмент в общем списке tools, который модель видит так же, как остальные инструменты (по имени tool).
- Он нужен для ситуаций, где требуется осознанный выбор пользователя и нельзя безопасно продолжать автоматически.
- Вызов `request_user_input(...)` не делает внешнего действия (не пишет файл и не запускает процесс): tool вызывает `interrupt(...)` и ставит граф на паузу.
- UI показывает inline-карточку выбора, а затем резюмирует тот же run через `Command(resume=...)`.
- Значение, переданное в `resume`, возвращается обратно как результат вызова `request_user_input`.
- Это штатный HITL-поток LangGraph; обычные текстовые вопросы в ответе модели не заменяют этот механизм.

## Поддержка изображений (вход)

- Агент поддерживает image input в пользовательском сообщении (multimodal turn).
- Поддерживаются несколько изображений в одном запросе.
- Изображения сохраняются в managed storage сессии (`.agent_state/attachments/<session_id>/...`) и восстанавливаются после перезапуска вместе с transcript.

### Как включить

- Откройте `Settings` → `Model Settings`.
- Для нужного профиля включите чекбокс `Поддержка изображений`.
- Если у профиля выключена поддержка изображений, рядом с моделью отображается бейдж `no img`, а image-attach path блокируется.

### Как пользоваться

- Вставка из буфера обмена: `Ctrl+V` (если в буфере изображение, добавляется image attachment с мини-превью).
- Добавление из файла: кнопка `+` в composer → `Add image...`.
- Обычные файлы через `+` → `Insert file path...` сохраняют старое поведение (как текстовые пути), не становятся image-attachments.
- После отправки изображения идут в модель как multimodal content:
  - `openai`/OpenAI-compatible: provider-native `image_url` (`data:<mime>;base64,...`),
  - другие провайдеры: стандартные LangChain image blocks.

### Поведение при ошибках

- Если профиль отключен для image input, запрос с изображением не запускается и показывается понятный notice.
- Если профиль включен, но upstream-модель фактически не приняла изображение, runtime возвращает безопасный notice `image_input_failed` вместо падения приложения.

## Поведение self-correction

- В графе нет отдельного LLM-узла критика и нет глобального post-pass `stability_guard` в happy-path.
- Recovery запускается только при реальной проблеме: `open_tool_issue`, protocol violation, hard loop budget или отсутствие требуемого evidence.
- При ошибке инструмента runtime сначала пробует детерминированные recovery-стратегии (`normalize_args`, `switch_tool`, `verify_side_effect`, `repair_then_rerun`, `resume_after_transient_failure`), затем при необходимости выполняет один `llm_replan`.
- Recovery bounded по issue fingerprint (жесткий ceiling), чтобы исключить бесконечные репланы.
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

- `PROMPT_PATH` (включая языковую политику ответа; например `Always respond in Russian.`)
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
