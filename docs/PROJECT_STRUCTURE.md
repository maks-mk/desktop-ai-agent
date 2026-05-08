# Архитектура и модули проекта

Этот документ содержит полное описание структуры проекта для разработчиков.

## Корень проекта
- `main.py` — Точка входа. Инициализирует QApplication и MainWindow.
- `agent.py` — Сборка LangGraph-агента: определение узлов, переходов и компиляция графа.
- `prompt.txt` — Системный промпт агента (правила, ограничения, инструкции).
- `mcp.json` — Конфигурация MCP-серверов (Model Context Protocol).
- `requirements.txt` — Зависимости Python (LangChain, PySide6, Pydantic и др.).

---

## Директория `core/` (Ядро системы)

### Управление состоянием и графом
- `state.py` — Класс `AgentState`: структура данных, передаваемая между узлами графа.
- `node_orchestrators.py` — Оркестраторы узлов: высокоуровневая логика выполнения ходов агента.
- `checkpointing.py` — Сохранение/восстановление сессий в SQLite (персистентность чатов).

### Логика узлов (`core/nodes/`)
- `agent.py` — Узел "agent": вызов LLM, парсинг ответа, извлечение tool_calls.
- `tools.py` — Узел "tools": выполнение инструментов, обработка результатов.
- `recovery.py` — Узел "recovery": стратегии самоисправления при ошибках инструментов.
- `summarize.py` — Узел "summarize": сжатие контекста через LLM при превышении лимитов.
- `approval.py` — Узел "approval": пауза графа для подтверждения действий пользователем.

### Безопасность и политики
- `safety_policy.py` — Валидация shell-команд и опасных операций.
- `policy_engine.py` — Движок политик: правила разрешено/запрещено.
- `tool_policy.py` — Метаданные инструментов: flags (mutating, destructive, approval).
- `runtime_prompt_policy.py` — Динамическая генерация системного промпта (ОС, дата, пути).

### Модели и API
- `model_profiles.py` — Управление профилями моделей (CRUD, активный профиль, валидация).
- `api_key_rotation.py` — Ротация API-ключей при Rate Limit ошибках.
- `model_fetcher.py` — Получение списка моделей от провайдеров (Gemini, OpenAI, Ollama).

### Конфигурация и утилиты
- `config.py` — Загрузка настроек из `.env` (Pydantic Settings).
- `context_builder.py` — Сборка контекста для LLM: история, system messages, tool results.
- `run_logger.py` — Логирование запусков в JSONL (отладка, аудит).
- `text_utils.py` — Утилиты для работы с текстом: split_markdown_segments, format_tool_display.
- `input_sanitizer.py` — Санитизация пользовательского ввода (sanitize_user_text).
- `multimodal.py` — Работа с изображениями: проверка файлов, определение возможностей модели.
- `constants.py` — Константы проекта (AGENT_VERSION и др.).

---

## Директория `tools/` (Инструменты агента)

### Реестр и инфраструктура
- `tool_registry.py` — Центральный реестр инструментов: регистрация, метаданные, доступ.
- `filesystem.py` — Инструменты ФС: read_file, write_file, edit_file, search_in_file.
- `local_shell.py` — Выполнение shell-команд (cli_exec), управление процессами.
- `search_tools.py` — Веб-поиск (Tavily), извлечение контента (fetch_content).
- `system_tools.py` — Информация о системе: IP, сеть, ОС, железо.
- `process_tools.py` — Управление фоновыми процессами (старт, стоп, статус).
- `user_input_tool.py` — Запрос уточняющей информации у пользователя.

### Реализация файловой системы (`tools/filesystem_impl/`)
- `__init__.py` — Экспорт функций filesystem_impl.
- `manager.py` — Менеджер операций ФС: координация read/write/edit.
- `editing.py` — Логика редактирования: diff, patch, apply_changes.
- `pathing.py` — Нормализация путей, валидация, разрешение относительных путей.

---

## Директория `ui/` (Графический интерфейс на PySide6)

### Главное окно и контроллеры
- `main_window.py` — Класс MainWindow: сборка UI, обработка событий, связь с runtime.
- `runtime.py` — AgentRuntimeController: запуск агента в потоке, эмиссия событий.
- `streaming.py` — Обработка streaming-ответов от LLM (потоковый вывод в чат).
- `theme.py` — Стилизация: цвета (ACCENT_BLUE, ERROR_RED), шрифты, CSS.
- `main_window_state.py` — Контроллеры состояния: ComposerStateController, RunStatusController, StreamEventRouter.

### Компоненты окна (`ui/window_components/`)
- `__init__.py` — Экспорт компонентов.
- `main_window.py` — (см. выше) Основной класс окна.
- `menu_builder.py` — Построение меню: File, View, тулбар, кнопки, actions.
- `workspace_builder.py` — Построение рабочей области: splitter, sidebar, transcript, composer, inspector.
- `sidebar_controller.py` — Логика sidebar: переключение сессий, удаление, новый проект.
- `inspector_controller.py` — Логика inspector: показать/скрыть панель деталей.
- `status_bar_manager.py` — Управление status bar: статусы, таймеры, мета-информация.

### Виджеты (`ui/widgets/`)
- `__init__.py` — Экспорт всех виджетов.
- `foundation.py` — Базовые классы и утилиты: AutoTextBrowser, CodeBlockWidget, DiffBlockWidget, CollapsibleSection, _fa_icon, _make_mono_font.
- `messages.py` — Виджеты сообщений: UserMessageWidget, AssistantMessageWidget, NoticeWidget, RunStatsWidget, StatusIndicatorWidget, ApprovalRequestCardWidget, UserChoiceCardWidget.
- `composer.py` — ComposerTextEdit: поле ввода с автодополнением @file, история команд, вставка изображений.
- `transcript.py` — ChatTranscriptWidget, ConversationTurnWidget: отображение диалога, группировка инструментов.
- `tools.py` — ToolCardWidget, CliExecWidget: карточка инструмента, вывод CLI с ANSI-очисткой.
- `tool_group.py` — ToolGroupWidget: группировка инструментов одного хода, сворачивание/разворачивание.
- `attachments.py` — ImageAttachmentChipWidget, ImageAttachmentStripWidget: превью изображений, удаление.
- `dialogs.py` — Диалоги: ModelSettingsDialog (профили моделей, inline-ротация ключей), ModelFetchWorker.
- `panels.py` — Панели: OverviewPanelWidget (детали запуска), ToolsPanelWidget (список инструментов), InspectorPanelWidget, InfoPopupDialog.
- `sidebar.py` — SessionSidebarWidget, SessionListModel, SessionItemDelegate: список сессий с группировкой по проектам.
- `foundation.py` — (см. выше) Базовые компоненты.

---

## Директория `utils/` (Вспомогательные утилиты)
- `count_lines_fixed.py` — Точный подсчет строк кода через токенизатор Python (исключая docstrings).
- `log_parser.py` — Парсинг JSONL-логов в pretty-JSON (отладочный скрипт).

---

## Директория `data/` (Данные runtime)
- `agent_state.db` — SQLite-база для хранения сессий и чекпоинтов (создается автоматически).
- `logs/` — JSONL-логи запусков агента (аудит, отладка).

---

## Ключевые взаимосвязи

```
main.py → ui/main_window.py → ui/runtime.py → agent.py → core/nodes/*.py → tools/*.py
                                      ↓
                              core/state.py (общее состояние)
                                      ↓
                          core/checkpointing.py (сохранение в data/)
```

1. **GUI → Runtime → Agent**: MainWindow отправляет запрос в AgentRuntimeController, который запускает LangGraph.
2. **Agent → Tools**: Узел "tools" вызывает инструменты из `tools/`, результаты возвращаются в граф.
3. **Streaming**: События streaming (токены, статусы инструментов) эмитятся через StreamEventRouter в UI.
4. **Персистентность**: После каждого хода состояние сохраняется в `data/agent_state.db` через checkpointing.py.

---

## Стек технологий
- **LangGraph** — Оркестрация агента (StateGraph, узлы, переходы).
- **PySide6** — GUI (Qt для Python).
- **Pydantic** — Валидация настроек и данных.
- **SQLite** — Хранение сессий.
- **JSONL** — Логирование запусков.
