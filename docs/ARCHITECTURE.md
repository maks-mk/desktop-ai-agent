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
- Stream-interruption recovery: при обрыве потока провайдера история автоматически чинится, ошибка классифицируется (`rate_limit` / `timeout` / `server_error` / `network`), и запуск продолжается после экспоненциального backoff с джиттером (`RETRY_DELAY * 2^attempt + random jitter`, для rate-limit — `RETRY_DELAY * 1.5`). Лимит попыток авто-продолжения — `min(MAX_RETRIES, 2)` (не отдельная env-переменная).

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

## Сессии и Checkpoints

- Graph checkpoints: `sqlite` (по умолчанию) или `memory`
- `.agent_state/checkpoints.sqlite` — durable checkpoint store
- `.agent_state/session.json` — активная сессия
- `.agent_state/session_index.json` — индекс всех сессий
- `logs/runs/` — JSONL-логи каждого запуска
