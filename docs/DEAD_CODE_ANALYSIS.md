# Анализ мёртвого кода — Portable Autonomous AI Agent

**Дата анализа:** 2026-06-29
**Инструмент:** `vulture 2.16` (min-confidence 60) + ручная верификация `rg`
**Объём:** 117 Python-файлов (без `venv`, `__pycache__`, `.git`)
**Коммит:** `dbd18e9`

---

## Методология

1. Запуск `vulture` на `agent.py main.py core tools ui` с min-confidence 60.
2. Ручная верификация каждого срабатывания через `rg` по всему репозиторию (включая `tests/`).
3. Классификация: **мёртвый код** / **test-only API** / **false positive** (pydantic validators, Qt overrides, tool registry, back-compat re-exports).

---

## Сводка

| Категория | Кол-во | Действие |
|---|---|---|
| Подтверждённый мёртвый код | 19 | Кандидат на удаление |
| Test-only API (используются только тестами) | 7 | Оставить |
| False positives (pydantic / Qt / registry / re-exports) | ~84 | Оставить |

---

## 1. Подтверждённый мёртвый код

Эти элементы не вызываются нигде в runtime-коде и нигде в тестах.

### core/

| Файл | Строка | Элемент | Тип | Примечание | Статус |
|---|---|---|---|---|---|
| `core/constants.py` | 45 | `RECOVERY_CONTINUE_PROMPT_TEMPLATE` | константа | Не импортируется нигде вне файла | ✅ Удалено |
| `core/recovery_manager.py` | 35 | `_actionable_recovery_hint()` | функция | Не вызывается вне файла | ✅ Удалено |
| `core/recovery_manager.py` | 220 | `build_successful_tool_stagnation_handoff_text()` | метод | Не вызывается вне файла | ✅ Удалено |
| `core/constants.py` | 77 | `SUCCESSFUL_TOOL_STAGNATION_HANDOFF_TEMPLATE` | константа | Использовалась только удалённым методом | ✅ Удалено |
| `core/session_store.py` | 255 | `update_session_title()` | метод | Не вызывается вне файла | ✅ Удалено |
| `core/nodes/context.py` | 47 | `_message_is_provider_system()` | метод | Не вызывается вне файла | ✅ Удалено |
| `core/nodes/context.py` | 50 | `_is_provider_safe_tool_call_id()` | метод | Не вызывается вне файла | ✅ Удалено |
| `core/nodes/context.py` | 63 | `_get_last_model_visible_message()` | метод | Не вызывается вне файла | ✅ Удалено |
| `core/text_utils.py` | 473 | `_append_unique_text()` | функция | Не вызывается вне файла | ✅ Удалено |
| `core/input_sanitizer.py` | 26 | `is_empty` (property) | property | Не читается вне файла | ✅ Удалено |
| `core/logging_config.py` | 215 | `_reasoning_debug_handler` (attr) | атрибут-маркер | Только устанавливается, нигде не читается | ✅ Удалено |

### ui/

| Файл | Строка | Элемент | Тип | Примечание | Статус |
|---|---|---|---|---|---|
| `ui/streaming.py` | 631 | `_is_replay_of_visible_text()` | метод | Не вызывается вне файла | ✅ Удалено |
| `ui/streaming.py` | 1093 | `_tool_indices_from_chunks()` | staticmethod | Не вызывается вне файла | ✅ Удалено |
| `ui/runtime_worker.py` | 189 | `_set_current_session_active()` | делегат | Thin wrapper к coordinator, не вызывается | ✅ Удалено |
| `ui/runtime_worker.py` | 198 | `_fallback_to_current_project_session()` | делегат | Thin wrapper к coordinator, не вызывается | ✅ Удалено |
| `ui/runtime_worker.py` | 225 | `_config_overrides_for_profile()` | staticmethod-делегат | Не вызывается | ✅ Удалено |
| `ui/widgets/foundation.py` | 30 | `FENCED_BLOCK_RE` | константа | Не используется вне файла | ✅ Удалено |
| `ui/runtime_payloads.py` | 30 | `APPROVAL_MODE_DENY` | константа | Не используется вне файла | ✅ Удалено |
| `ui/widgets/messages.py` | 138 | `set_label()` | метод | Не вызывается вне файла (используется `set_level`, не `set_label`) | ✅ Удалено |
| `ui/widgets/transcript.py` | 100 | `has_rendered_output()` | метод | Не вызывается вне файла | ✅ Удалено |

### tools/

| Файл | Строка | Элемент | Тип | Примечание | Статус |
|---|---|---|---|---|---|
| `tools/tool_registry.py` | 603 | `get_runtime_status()` | метод | Не вызывается; используется только `get_runtime_status_lines()` | ✅ Удалено |

### Write-only атрибуты (устанавливаются, но не читаются)

| Файл | Строка | Элемент | Примечание | Статус |
|---|---|---|---|---|
| `ui/window_components/main_window.py` | 82–83 | `_primary_status_label`, `_status_message_ticket` | Инициализируются, пишутся из `main_window_state.py` и `status_bar_manager.py`, но нигде не читаются | ✅ Удалено |
| `ui/window_components/main_window.py` | 259 | `_event_handlers` | Присваивается, но не читается | ✅ Удалено |

### Неиспользуемые theme-константы

| Файл | Строка | Элемент | Примечание | Статус |
|---|---|---|---|---|
| `ui/theme.py` | 19 | `SEPARATOR` | Не используется вне theme.py | ✅ Удалено |
| `ui/theme.py` | 39 | `SOFT_RADIUS_LG` | Не используется вне theme.py | ✅ Удалено |
| `ui/theme.py` | 86–94 | `transcript_panel_border`, `transcript_panel_hover`, `tool_panel_border`, `tool_call_idle` | Вычисляются в palette и присваиваются как атрибуты, но не читаются вне theme.py | ✅ Удалено |

---

## 2. Test-only API (не мёртвый код, но не используется в runtime)

Эти элементы вызываются только из `tests/`. Удаление сломает тесты. Если тесты — часть контракта, оставить как есть.

| Файл | Элемент | Тест-референс |
|---|---|---|
| `core/validation.py` | `validate_tool_result()` | `test_tooling_refactor.py` |
| `core/message_context.py` | `recent_tool_context_names()` | `test_intent_engine.py` |
| `core/message_context.py` | `current_turn_has_tool_evidence()` | `test_intent_engine.py` |
| `core/message_context.py` | `had_tool_activity_in_previous_turn()` | `test_intent_engine.py` |
| `core/nodes/context.py` | `_sanitize_messages_for_model()` | `test_session_utils.py`, `test_runtime_refactor.py` |
| `core/session_store.py` | `load_active_session()` | `test_runtime_refactor.py` (множественные) |
| `ui/runtime_worker.py` | `_try_change_workdir()` | `test_runtime_session_coordination.py` |

---

## 3. False positives (не мёртвый код)

### 3.1 Pydantic validators (вызываются неявно pydantic runtime)

Все методы `core/config.py` (`parse_max_file_size`, `validate_top_p`, `validate_top_k`, `resolve_path_fields`, `normalize_log_level`, `validate_max_loops`, `migrate_legacy_self_correction_settings`, `normalize_checkpoint_backend`, `parse_optional_loop_guard_value`, `validate_positive_runtime_limits`, `validate_self_correction_retry_limit`, `validate_provider_keys`, `parse_optional_sampling_value`) — декорированы `@field_validator` / `@model_validator`, вызываются pydantic при валидации.

Аналогично `normalize_payload` / `validate_payload` в `tools/filesystem.py`, `tools/process_tools.py`, `tools/search_tools.py`, `tools/user_input_tool.py` — pydantic `@model_validator`.

`model_config = ConfigDict(...)` — обязательный атрибут pydantic-модели.

### 3.2 Qt virtual method overrides (вызываются Qt framework)

| Файл | Метод | Qt base class |
|---|---|---|
| `ui/widgets/foundation.py` | `createMimeDataFromSelection()` (×2) | `QPlainTextEdit` / `QTextEdit` |
| `ui/widgets/foundation.py` | `highlightBlock()` | `QSyntaxHighlighter` |
| `ui/widgets/sidebar.py` | `rowCount()` | `QAbstractListModel` |
| `ui/widgets/sidebar.py` | `paint()` | `QStyledItemDelegate` |
| `ui/widgets/dialogs.py` | `showPopup()` | `QComboBox` |

### 3.3 Tool registry (регистрируются через декоратор `@tool(...)`)

Все tool-функции в `tools/` вызываются не напрямую, а через `ToolRegistry` по имени. Vulture не видит этот паттерн:

- `tools/local_shell.py`: `cli_exec`
- `tools/search_tools.py`: `web_search`, `fetch_content`, `batch_web_search`
- `tools/system_tools.py`: `get_public_ip`, `lookup_ip_info`, `get_system_info`, `get_local_network_info`
- `tools/process_tools.py`: `run_background_process`, `stop_background_process`, `find_process_by_port`

### 3.4 Back-compat re-exports в agent.py

`_gemini_model_supports_thinking_budget`, `_patch_langchain_google_genai_retry_kwargs`, `_extract_openai_reasoning_delta` — re-exports для обратной совместимости тестов (`test_runtime_refactor.py` обращается к ним через `agent_module._...`).

### 3.5 State-поля (TypedDict, NotRequired)

Поля `core/state.py` (`token_usage`, `turn_mode`, `requires_evidence`, `self_correction_retry_count`, `self_correction_retry_turn_id`, `self_correction_fingerprint_history`, `self_correction_last_reason`, `run_id`, `pending_approval`, `last_tool_error`, `last_tool_result`) — часть `TypedDict` с `NotRequired`. Используются graph-нодами через dict-access (`state["field"]`), vulture не отслеживает такой доступ.

### 3.6 Pydantic-модели tool-схем

`tools/filesystem.py`: `EditFileInput`, `WriteFileInput`, `DeleteFileInput`, `DeleteDirectoryInput` — pydantic-схемы, передаются в `@tool(args_schema=...)`. Vulture помечает их `model_config` и `normalize_payload` как неиспользуемые, но они валидны.

### 3.7 UI-атрибуты, используемые в тестах

`set_file_index_for_testing`, `set_markdown`, `output_section`, `auto_follow_enabled`, `model_edit`, `result_payload`, `_apply_api_key_rotation_to_profile`, `_model_popup_search`, `_update_composer_height`, `rowCount`, `status_line_label` — вызываются из `test_cli_ux.py`.

### 3.8 `_level`, `_args_expanded`, `_cli_expanded`

`ui/widgets/messages.py:_level` — читается/пишется внутри `set_level()` / `_icon_for_level()`.
`ui/widgets/tools.py:_args_expanded`, `_cli_expanded` — читаются/пишутся внутри `_set_args_expanded()` / `_set_cli_expanded()`.

---

## Рекомендации

### ✅ Безопасно удалить — ВЫПОЛНЕНО (19 элементов + 5 write-only/theme + 1 зависимая константа)

**Дата выполнения:** 2026-06-29
**Проверка:** `venv\Scripts\python.exe -m pytest tests/ -q` — 662 passed, 21 subtests passed (1 предсуществующий сбой `test_create_llm_for_nvidia_reasoning_model_uses_registry_reasoning_effort`, не связан с очисткой).
**Дополнительно удалено:** `SUCCESSFUL_TOOL_STAGNATION_HANDOFF_TEMPLATE` из `core/constants.py` — использовалась только удалённым методом `build_successful_tool_stagnation_handoff_text()`.

1. **core/constants.py**: `RECOVERY_CONTINUE_PROMPT_TEMPLATE` ✅
2. **core/recovery_manager.py**: `_actionable_recovery_hint()`, `build_successful_tool_stagnation_handoff_text()` ✅
3. **core/constants.py**: `SUCCESSFUL_TOOL_STAGNATION_HANDOFF_TEMPLATE` ✅ (зависимая константа)
4. **core/session_store.py**: `update_session_title()` ✅
5. **core/nodes/context.py**: `_message_is_provider_system()`, `_is_provider_safe_tool_call_id()`, `_get_last_model_visible_message()` ✅
6. **core/text_utils.py**: `_append_unique_text()` ✅
7. **core/input_sanitizer.py**: `is_empty` (property) ✅
8. **core/logging_config.py**: `_reasoning_debug_handler` (строка 215 — установка маркера) ✅
9. **ui/streaming.py**: `_is_replay_of_visible_text()`, `_tool_indices_from_chunks()` ✅
10. **ui/runtime_worker.py**: `_set_current_session_active()`, `_fallback_to_current_project_session()`, `_config_overrides_for_profile()` ✅
11. **ui/widgets/foundation.py**: `FENCED_BLOCK_RE` ✅
12. **ui/runtime_payloads.py**: `APPROVAL_MODE_DENY` ✅
13. **ui/widgets/messages.py**: `set_label()` ✅
14. **ui/widgets/transcript.py**: `has_rendered_output()` ✅
15. **tools/tool_registry.py**: `get_runtime_status()` ✅
16. **ui/window_components/main_window.py**: `_primary_status_label`, `_status_message_ticket`, `_event_handlers` (write-only) ✅
    - Также очищены write-сайты в `ui/main_window_state.py` и `ui/window_components/status_bar_manager.py` (методы `set_primary_status_message` и `show_transient_status_message` заменены на `pass`).
17. **ui/theme.py**: `SEPARATOR`, `SOFT_RADIUS_LG`, `transcript_panel_border`, `transcript_panel_hover`, `tool_panel_border`, `tool_call_idle` ✅
    - Проверено: palette-переменные не используются в f-string стилей (`{...}` подстановках), не ссылаются CSS-селекторы в `.qss` файлах.

### Перед удалением проверить — ВЫПОЛНЕНО

- **Write-only атрибуты** (`_primary_status_label`, `_status_message_ticket`, `_event_handlers`): проверен динамический доступ через `getattr` / reflection — не обнаружен. Записи в `main_window_state.py` и `status_bar_manager.py` заменены на `pass` (методы вызываются, но побочные эффекты на неиспользуемые атрибуты удалены). ✅
- **Theme-константы**: `transcript_panel_border` и др. проверены — не используются в QSS-строках, `setProperty`, `.qss` файлах или строках стилей. ✅

### Не удалять

- Все pydantic validators, Qt overrides, tool-функции, back-compat re-exports, state-поля, test-only API.

---

## Раунд 2 — Thin-delegate методы (коммит `fb810ba`)

**Дата анализа:** 2026-06-30
**Инструмент:** `vulture 2.16` (min-confidence 60) + ручная верификация `findstr` / `rg`
**Коммит:** `fb810ba`

После первого раунда vulture (min-confidence 80) не нашёл новых кандидатов в `core/`, `tools/`, `ui/` (0 срабатываний). Оставшиеся срабатывания на `agent.py` / `main.py` — back-compat re-exports для тестов (не подлежат удалению).

Однако при min-confidence 60 обнаружились 7 thin-delegate методов в UI — wrappers к controller/state объектам, оставшиеся после рефакторинга, когда вызовы перенесли напрямую к controllers.

### Подтверждённый мёртвый код (7 методов, 21 строка)

| Файл | Строка | Элемент | Тип | Статус |
|---|---|---|---|---|
| `ui/widgets/tools.py` | 141 | `has_output()` | метод | ✅ Удалено |
| `ui/window_components/main_window.py` | 264 | `_composer_visual_line_count()` | thin delegate → `_composer_state` | ✅ Удалено |
| `ui/window_components/main_window.py` | 286 | `_refresh_draft_attachments()` | thin delegate → `_composer_state` | ✅ Удалено |
| `ui/window_components/main_window.py` | 292 | `_append_draft_image_attachments()` | thin delegate → `_composer_state` | ✅ Удалено |
| `ui/window_components/main_window.py` | 499 | `_set_primary_status_message()` | thin delegate → `_status_bar_manager` | ✅ Удалено |
| `ui/window_components/main_window.py` | 891 | `_set_sidebar_collapsed()` | thin delegate → `_sidebar_controller` | ✅ Удалено |
| `ui/window_components/main_window.py` | 894 | `_set_inspector_collapsed()` | thin delegate → `_inspector_controller` | ✅ Удалено |

### Верификация

Для каждого метода проверено:
- `findstr` по всем `.py` (включая `tests/`) — только определение, нигде не вызывается.
- Нет `getattr(self, "_method_name")` динамического доступа.
- Нет `SLOT("...()")` / `QMetaObject` / `pyqtSlot` строковых подключений в проекте.
- Соседние thin delegates (например `_update_composer_height`, `_clear_draft_image_attachments`) — **используются** в runtime или тестах; паттерн валиден, эти 7 конкретных методов оказались лишними.

### Проверка

`venv\Scripts\python.exe -m pytest tests/ -q` — 662 passed, 21 subtests passed (1 предсуществующий сбой `test_create_llm_for_nvidia_reasoning_model_uses_registry_reasoning_effort`, не связан с очисткой).

### Итог

После двух раундов очистки vulture с min-confidence 80 не находит нового мёртвого кода. Дальнейшие кандидаты — только false positives (pydantic, Qt, tool registry, back-compat re-exports, test-only API, dynamic dispatch через `getattr`).

---

## Команда для воспроизведения

```powershell
python -m vulture agent.py main.py core tools ui --min-confidence 60
```
