# Архитектура

## Runtime Flow

```text
START
  → summarize        # сжать контекст если сессия стала большой
  → update_step
  → agent            # LLM решает: ответить / вызвать tool / recovery
     → approval      # пауза перед мутирующим действием
        → tools
     → tools         # исполнить tool calls (read-only — параллельно, остальные — последовательно)
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
- Stream-interruption recovery: при обрыве потока провайдера история автоматически чинится, ошибка классифицируется (`rate_limit` / `timeout` / `server_error` / `network`), и запуск продолжается после экспоненциального backoff с джиттером (`RETRY_DELAY * 2^attempt + random jitter`, для rate-limit — `RETRY_DELAY * 1.5`). Максимум `MAX_STREAM_REPAIR_RESUMES` попыток.

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
| Plan Mode | `core/runtime_prompt_policy.py` | Инструкция для режима планирования (включается при `turn_mode == "plan"`) |

---

## Plan Mode

Когда `turn_mode == "plan"` (устанавливается через `plan_mode: True` в payload запроса):

- **Prompt**: добавляется короткий `PLAN_MODE_TEXT` контракт. Он требует сначала исследовать repo read-only, затем при необходимости запросить только одно блокирующее решение через `request_user_input`, а финальный ответ вернуть ровно одним блоком `<implementation_plan>...</implementation_plan>`. Approval моделью не запрашивается: runtime сам показывает обязательный interrupt `choice_type='plan_review'`.
- **Tools**: `_active_tools_for_turn` фильтрует набор до read-only инструментов + `request_user_input`. Мутирующие и деструктивные инструменты недоступны.
- **Protocol**: `_sanitize_user_input_tool_calls` сохраняет общий guard "at most once per turn" для legacy flow. В Plan Mode mutating tool calls отбрасываются до маршрутизации; `plan_build` собирает structured plan по `RuntimePlanDraft`, а `plan_review` формирует review interrupt из `current_plan`.
- **State**: structured plan сохраняется в `current_plan`; кроме `summary`, `steps`, `risks` и `assumptions`, схема также хранит `verification`. `plan_status` переходит в `pending_approval` до выбора пользователя. События `plan_progress` несут `current_plan`, `completed_steps`, `total_steps` и active-step context для UI.
- **Resume**: UI продолжает graph через `Command(resume={"choice_type": "plan_review", "choice": <selected option>, "feedback": <selected option>})`. `implement` переводит план к выполнению, `revise` запрашивает правки, `rebuild` перестраивает план, `cancel` отклоняет его. Legacy `plan_approval` resume сохраняется только для обратной совместимости.
- **Execution UI**: во время реализации `plan_progress` управляет отдельной side panel. Она остаётся видимой через retry/follow-up runs и скрывается только в terminal states (`completed`, `rejected`, `cancelled`).

---

## Сессии и Checkpoints

- Graph checkpoints: `sqlite` (по умолчанию) или `memory`
- `.agent_state/checkpoints.sqlite` — durable checkpoint store
- `.agent_state/session.json` — активная сессия
- `.agent_state/session_index.json` — индекс всех сессий
- `logs/runs/` — JSONL-логи каждого запуска
