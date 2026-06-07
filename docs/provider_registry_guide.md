# Provider Registry Guide

`provider_registry.json` описывает, какие reasoning/thinking параметры нужно передавать OpenAI-compatible агрегаторам. Это нужно потому, что единого стандарта нет: один провайдер принимает `reasoning.effort`, другой `extra_body.reasoning.effort`, третий `reasoning_effort`, а некоторые возвращают `400`, если отправить неизвестное поле.

Registry применяется только к профилям с `provider: openai`. Gemini настраивается отдельно через Google SDK-поля `thinking_budget`, `thinking_level` и `include_thoughts`.

## Где находится файл

По умолчанию используется файл:

```text
provider_registry.json
```

Путь можно переопределить через `.env`:

```env
PROVIDER_REGISTRY_PATH=provider_registry.json
```

Runtime загружает registry в `create_llm()` перед созданием `ChatOpenAI`. Если `OPENAI_BASE_URL` не найден в registry или провайдер помечен как `supports_reasoning: false`, reasoning payload не добавляется.

## Общая структура

```json
{
  "schema_version": 1,
  "data_version": 1,
  "providers": []
}
```

| Поле | Для чего |
|---|---|
| `schema_version` | Версия схемы. Меняется при несовместимых изменениях формата |
| `data_version` | Версия данных. Меняется при добавлении или правке провайдеров |
| `providers` | Список provider entries |

## Provider Entry

Минимальный пример агрегатора с `extra_body.reasoning.effort`:

```json
{
  "id": "example_gateway",
  "enabled": true,
  "priority": 80,
  "match": ["api.example.com"],
  "match_type": "suffix",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "extra_body.reasoning.effort",
    "allowed_values": ["low", "medium", "high"],
    "value_map": {
      "minimal": "low",
      "xhigh": "high"
    }
  },
  "notes": "Reasoning object API"
}
```

| Поле | Обязательное | Для чего |
|---|---:|---|
| `id` | да | Уникальный идентификатор провайдера |
| `enabled` | да | `false` временно выключает entry без удаления |
| `priority` | да | Победитель при нескольких совпадениях; больше значит важнее |
| `match` | да | Hostname-паттерны без path, например `openrouter.ai` |
| `match_type` | да | `exact` или `suffix` |
| `supports_reasoning` | да | Если `false`, provider матчится, но reasoning-поля не добавляются |
| `validation` | да | `strict`, `map` или `passthrough` |
| `reasoning` | если `supports_reasoning=true` | Настройка поля effort и дополнительных полей |
| `notes` | нет | Человеческая пометка, на runtime не влияет |

## Matching по `base_url`

Registry матчится по hostname из `OPENAI_BASE_URL`.

Примеры:

| `base_url` | Hostname | Может совпасть с |
|---|---|---|
| `https://openrouter.ai/api/v1` | `openrouter.ai` | `openrouter.ai` |
| `https://api.openrouter.ai/v1` | `api.openrouter.ai` | `openrouter.ai`, если `match_type: suffix` |
| `api.openai.com/v1?foo=bar` | `api.openai.com` | `api.openai.com` |
| `http://localhost:3002/v1` | `localhost` | не добавлять в registry |

`match_type`:

| Значение | Логика |
|---|---|
| `exact` | Hostname должен совпасть полностью |
| `suffix` | Совпадает сам hostname или его поддомен через точку |

Важно: `suffix` не является substring search. Паттерн `openrouter.ai` совпадёт с `api.openrouter.ai`, но не с `evil-openrouter.ai`.

## Reasoning Config

```json
{
  "path": "extra_body.reasoning.effort",
  "allowed_values": ["low", "medium", "high"],
  "value_map": {
    "minimal": "low",
    "xhigh": "high"
  },
  "extra_fields": {
    "extra_body.reasoning.summary": "auto"
  }
}
```

| Поле | Для чего |
|---|---|
| `path` | Dot-path, куда записать значение `MODEL_REASONING_EFFORT` |
| `allowed_values` | Допустимые значения после нормализации |
| `value_map` | Маппинг входного effort в значение, которое принимает провайдер |
| `extra_fields` | Дополнительные постоянные поля, если они документированы провайдером |

`path` пишет в kwargs для `ChatOpenAI`, не напрямую в HTTP JSON. Например:

| `path` | Итоговый kwargs |
|---|---|
| `reasoning.effort` | `{"reasoning": {"effort": "high"}}` |
| `extra_body.reasoning.effort` | `{"extra_body": {"reasoning": {"effort": "high"}}}` |
| `reasoning_effort` | `{"reasoning_effort": "high"}` |

## Validation Modes

| Mode | Поведение |
|---|---|
| `strict` | Значение должно быть в `allowed_values`, иначе ошибка |
| `map` | Сначала применяет `value_map`, потом проверяет `allowed_values` |
| `passthrough` | Пишет значение как есть, без проверки |

Обычно нужен `map`: пользователь может поставить `MODEL_REASONING_EFFORT=xhigh`, а провайдер принимает только `high`.

Особый случай: `MODEL_REASONING_EFFORT=none` выключает добавление reasoning payload на уровне runtime. Это работает даже если provider entry поддерживает reasoning.

## Как выбрать `path`

Смотри документацию конкретного агрегатора.

Типовые варианты:

| Провайдерный формат | `path` | Примеры |
|---|---|---|
| Native OpenAI Responses API | `reasoning.effort` | OpenAI |
| OpenAI SDK `extra_body` | `extra_body.reasoning.effort` | OpenRouter |
| Top-level Chat Completions field | `reasoning_effort` | Ollama Cloud, Fireworks, NVIDIA NIM, Mistral, AIHubMix |

Не угадывай поле по названию модели. Один и тот же model id через разные gateways может требовать разные параметры.

## Как добавить нового агрегатора

1. Найди в документации провайдера точный параметр reasoning/thinking.
2. Проверь, какие effort values поддерживаются: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` и т.д.
3. Добавь entry в `provider_registry.json`.
4. Если провайдер не документирует reasoning API, добавь entry с `supports_reasoning: false`.
5. Выбери `match_type: exact`, если должен матчиться только один hostname.
6. Выбери `match_type: suffix`, если у провайдера могут быть поддомены.
7. Не добавляй локальные endpoints (`localhost`, `127.0.0.1`) в registry.
8. Добавь или обнови тест в `tests/test_runtime_refactor.py` для expected kwargs.
9. Если меняешь validation/matching, добавь тест в `tests/test_provider_registry.py`.
10. Запусти тесты.

Команды:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_provider_registry.py tests\test_runtime_refactor.py -p no:cacheprovider
.\venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

## Примеры

### OpenAI

```json
{
  "id": "openai",
  "enabled": true,
  "priority": 100,
  "match": ["api.openai.com"],
  "match_type": "exact",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "reasoning.effort",
    "allowed_values": ["minimal", "low", "medium", "high", "xhigh"],
    "value_map": {},
    "extra_fields": {
      "reasoning.summary": "auto"
    }
  }
}
```

### OpenRouter

```json
{
  "id": "openrouter",
  "enabled": true,
  "priority": 90,
  "match": ["openrouter.ai"],
  "match_type": "suffix",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "extra_body.reasoning.effort",
    "allowed_values": ["minimal", "low", "medium", "high", "xhigh"],
    "value_map": {}
  }
}
```

### Top-level `reasoning_effort`

```json
{
  "id": "example_reasoning_effort",
  "enabled": true,
  "priority": 80,
  "match": ["api.example.com"],
  "match_type": "exact",
  "supports_reasoning": true,
  "validation": "map",
  "reasoning": {
    "path": "reasoning_effort",
    "allowed_values": ["low", "medium", "high"],
    "value_map": {
      "minimal": "low",
      "xhigh": "high"
    }
  }
}
```

Если провайдер поддерживает не весь общий набор effort values, мапь неподдерживаемые значения в ближайшее документированное. Например, Mistral сейчас документирует `high` и `none`: `none` обрабатывается глобально как выключение payload, а `minimal`/`low`/`medium`/`xhigh` можно свести к `high`.

### Conservative Entry

Используй это, когда провайдер есть в профилях, но reasoning API не подтверждён:

```json
{
  "id": "unknown_gateway",
  "enabled": true,
  "priority": 60,
  "match": ["api.unknown.example"],
  "match_type": "exact",
  "supports_reasoning": false,
  "validation": "passthrough",
  "notes": "Conservative default: do not send undocumented reasoning fields."
}
```

## Частые ошибки

- Отправлять `reasoning_effort` провайдеру, который ждёт `reasoning.effort`.
- Класть OpenRouter reasoning не в `extra_body`, когда используется OpenAI SDK совместимый клиент.
- Добавлять `summary: auto` без подтверждения в документации провайдера.
- Добавлять `localhost` в registry. Локальные OpenAI-compatible серверы слишком разные; лучше не слать им provider-specific поля по умолчанию.
- Делать `suffix` слишком широким, например `ai` или `com`.
- Включать reasoning для провайдера, но забывать добавить модель в `_openai_model_supports_reasoning_controls()` в `agent.py`. Registry выбирает wire format по provider, а model gate решает, нужно ли вообще добавлять reasoning controls.
