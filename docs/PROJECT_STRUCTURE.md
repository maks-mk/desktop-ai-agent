# Архитектура и структура проекта

Документ описывает актуальную структуру проекта для разработки и отладки. Временные директории (`venv/`, `__pycache__/`, `.git/`) не перечисляются.

## Корень проекта

- `main.py` - точка входа GUI, создает `QApplication` и открывает главное окно.
- `agent.py` - сборка LangGraph-агента, создание LLM, настройка reasoning/thinking, регистрация tools и компиляция графа.
- `prompt.txt` - общий системный промпт агента.
- `prompt_dev.txt` - дополнительный dev/devops-промпт.
- `mcp.json` - конфигурация MCP-серверов.
- `requirements.txt` - Python-зависимости.
- `env_example.txt` - пример переменных окружения.
- `provider_registry.json` - registry OpenAI-compatible провайдеров и их reasoning kwargs.
- `provider_registry_guide.md` - подробная инструкция по `provider_registry.json`.
- `README.md` - пользовательская и эксплуатационная документация.
- `build.bat` - вспомогательный build-скрипт для Windows.
- `icon.ico` - иконка приложения.

## Runtime-данные

- `.agent_state/` - пользовательское состояние приложения: активные настройки, профили моделей, SQLite-чекпоинты и локальные runtime-файлы.
- `logs/` - runtime/debug логи, включая reasoning/debug трассировку.

Эти директории относятся к локальному состоянию запуска и не являются исходным кодом.

## Директория `core/`

Ядро агента: конфигурация, состояние, политики, выполнение узлов графа, self-correction и сервисные функции.

### Состояние, сессии и checkpointing

- `state.py` - структура `AgentState`, передаваемая между узлами графа.
- `checkpointing.py` - SQLite checkpoint saver и восстановление состояния графа.
- `session_store.py` - хранение и выбор сессий.
- `session_utils.py` - утилиты идентификаторов, метаданных и снимков сессий.
- `context_builder.py` - сборка входного контекста для LLM.
- `message_context.py` - работа с контекстом сообщений.
- `message_utils.py` - нормализация и извлечение текста из сообщений.

### Конфигурация, модели и провайдеры

- `config.py` - загрузка настроек из `.env` и runtime-конфигурация агента.
- `model_profiles.py` - профили моделей, активный профиль, валидация и сохранение.
- `model_fetcher.py` - получение списков моделей от провайдеров.
- `api_key_rotation.py` - ротация API-ключей при rate limit и похожих ошибках.
- `provider_registry.py` - matching OpenAI-compatible агрегаторов и построение reasoning kwargs.
- `reasoning_debug.py` - отдельное debug-логирование reasoning/thinking и status-сигналов.
- `run_logger.py` - JSONL-логирование отдельных запусков агента.
- `logging_config.py` - настройка логирования приложения.

### Узлы и оркестрация графа

- `node_orchestrators.py` - высокоуровневые оркестраторы хода агента и tools.
- `node_errors.py` - классификация ошибок узлов.
- `recovery_manager.py` - управление попытками восстановления после сбоев.
- `self_correction_engine.py` - построение repair-планов для tool/protocol ошибок.
- `summarize_policy.py` - правила сжатия истории.

### Policies, safety и tools-инфраструктура

- `safety_policy.py` - правила безопасности shell/filesystem операций.
- `policy_engine.py` - движок политик разрешений и запретов.
- `tool_policy.py` - метаданные инструментов: mutating, destructive, approval и т.п.
- `tool_executor.py` - выполнение tools и нормализация результатов.
- `tool_args.py` - канонизация аргументов tool calls.
- `tool_issues.py` - описание tool/protocol проблем.
- `tool_results.py` - структуры результатов tools.
- `runtime_prompt_policy.py` - runtime-контракт и динамические инструкции для system prompt.

### Утилиты и валидация

- `constants.py` - общие константы проекта.
- `errors.py` - общие исключения.
- `fast_copy.py` - быстрые операции копирования/сериализации.
- `input_sanitizer.py` - санитизация пользовательского ввода.
- `multimodal.py` - проверка изображений и возможностей мультимодальных моделей.
- `text_tool_calls.py` - восстановление текстовых pseudo-tool-call маркеров от совместимых провайдеров.
- `text_utils.py` - форматирование markdown/text и отображение tool output.
- `utils.py` - общие вспомогательные функции.
- `validation.py` - валидация runtime-данных.

## Директория `core/nodes/`

Реализация отдельных узлов LangGraph.

- `base.py` - базовые mixin/helper функции для узлов.
- `context.py` - подготовка контекста перед LLM-вызовом.
- `llm.py` - LLM turn orchestration.
- `agent.py` - обработка ответа LLM, tool calls, Gemini thought signatures и protocol checks.
- `tool_preflight.py` - предварительная проверка tool calls перед выполнением.
- `tools.py` - выполнение tool calls и сбор `ToolMessage`.
- `approval.py` - interrupt/approval flow для действий, требующих подтверждения.
- `recovery.py` - узел восстановления после tool/protocol ошибок.
- `protocol.py` - protocol guard logic.
- `summarize.py` - сжатие истории при превышении лимитов.

## Директория `tools/`

Инструменты, доступные агенту.

- `tool_registry.py` - центральный реестр tools, metadata и MCP-интеграции.
- `filesystem.py` - filesystem tools: `read_file`, `write_file`, `edit_file`, `list_directory`, `search_*`, `download_file`, safe delete.
- `local_shell.py` - `cli_exec`, streaming stdout/stderr и управление shell-командами.
- `process_tools.py` - фоновые процессы: запуск, остановка, поиск по порту.
- `search_tools.py` - web search/fetch tools и runtime search config.
- `system_tools.py` - системная информация, сеть, IP.
- `user_input_tool.py` - запрос уточняющего выбора у пользователя.

### `tools/filesystem_impl/`

- `manager.py` - backend filesystem-операций.
- `editing.py` - точечное редактирование текстовых файлов.
- `pathing.py` - нормализация и проверка путей.

## Директория `ui/`

GUI на PySide6 и слой выполнения агента в отдельном worker-потоке.

- `main_window.py` - facade/entry для главного окна.
- `runtime.py` - публичный runtime controller facade.
- `runtime_worker.py` - `AgentRunWorker` и `AgentRuntimeController`, запуск графа, interrupts, repair/continue flow.
- `runtime_session.py` - координация активной runtime-сессии.
- `runtime_payloads.py` - payloads для UI: transcript, tools snapshot, approval/user-choice cards.
- `streaming.py` - нормализация streaming events, reasoning/thinking signals, статусы `Working...`/`Thinking...`.
- `main_window_state.py` - routing stream events и состояние composer/run status.
- `theme.py` - цвета, стили и Qt stylesheet.
- `tool_message_utils.py` - форматирование tool messages для UI.
- `visibility.py` - фильтрация внутренних сообщений из transcript.

### `ui/window_components/`

- `main_window.py` - основной `MainWindow`.
- `menu_builder.py` - меню и actions.
- `workspace_builder.py` - layout: sidebar, transcript, composer, inspector.
- `sidebar_controller.py` - логика sidebar и сессий.
- `inspector_controller.py` - управление inspector-панелью.
- `status_bar_manager.py` - статусбар, runtime indicators и meta-информация.

### `ui/widgets/`

- `foundation.py` - базовые виджеты, markdown/code/diff helpers.
- `messages.py` - user/assistant/notice/status/approval/user-choice виджеты.
- `composer.py` - поле ввода, history, mentions, image attachments.
- `transcript.py` - отображение диалога и группировка turn-ов.
- `tool_group.py` - группировка tool cards одного хода.
- `tools.py` - карточки tools и CLI output widget.
- `attachments.py` - image attachment chips.
- `dialogs.py` - настройки моделей, API key rotation, model fetch worker.
- `panels.py` - overview/tools/inspector panels.
- `sidebar.py` - список сессий, модель и delegate.

## Директория `tests/`

Автотесты покрывают runtime, UI helpers, tools, provider registry, reasoning/status parsing и session coordination.

- `test_provider_registry.py` - matching провайдеров и reasoning kwargs.
- `test_runtime_refactor.py`, `test_runtime_payloads.py`, `test_runtime_session_coordination.py` - runtime и graph behavior.
- `test_stream_and_filesystem.py` - streaming, statuses, filesystem tools.
- `test_tooling_refactor.py`, `test_policy_engine.py`, `test_api_key_rotation.py` - tools, policies и ключи.
- `test_model_profiles.py`, `test_model_fetcher.py` - профили и загрузка моделей.
- `test_cli_ux.py`, `test_main_window_facade.py`, `test_ui_helpers.py` - UI и UX.
- `test_refactor_services.py`, `test_self_correction_engine.py`, `test_critic_graph.py`, `test_intent_engine.py`, `test_input_sanitizer.py`, `test_logging_config.py`, `test_session_utils.py` - сервисы ядра и вспомогательная логика.

## Основной поток выполнения

```text
main.py
  -> ui/window_components/main_window.py
  -> ui/runtime_worker.py
  -> agent.py
  -> core/nodes/*.py
  -> tools/*.py
```

Ключевой цикл:

1. Пользователь отправляет запрос через UI.
2. `AgentRuntimeController` запускает `AgentRunWorker`.
3. LangGraph вызывает LLM-узел.
4. Если LLM вернула `tool_calls`, выполняется `tools`-узел.
5. Tool results возвращаются в контекст, после чего LLM продолжает или дает финальный ответ.
6. Streaming/status events отображаются в transcript и status indicator.
7. Состояние сохраняется через checkpoint/session store.

## Стек технологий

- LangGraph - orchestration stateful agent graph.
- LangChain - модели, messages и tool binding.
- PySide6 - GUI.
- Pydantic / pydantic-settings - настройки и валидация.
- SQLite - checkpointing и локальное состояние.
- JSONL / logging - run logs и debug трассировка.
- MCP - подключаемые внешние инструменты.
