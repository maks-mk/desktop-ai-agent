# План: Добавление поддержки `provider=anthropic`

> Версия плана: 1.0  
> Дата: 2026-07-20  
> Проект: `v0.66.996b_test`  
> Источники: локальный код + Context7 (`/anthropics/anthropic-sdk-python`, `/websites/langchain`)

---

## 0. Краткое резюме

Текущая архитектура поддерживает два провайдера — `gemini` и `openai` (OpenAI-compatible). Добавление `anthropic` требует:

1. Новых полей конфигурации (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`).
2. Нового адаптера провайдера (`core/providers/anthropic.py`) на базе `langchain-anthropic.ChatAnthropic`.
3. Расширения `Literal`-типа `provider` и всех диспетчеризаций по провайдеру.
4. Поддержки специфики Anthropic: top-level `system` prompt, extended thinking с `budget_tokens` или через effort-levels (`ANTHROPIC_REASONING`), image input в формате Anthropic, round-trip thinking-блоков с сигнатурами.
5. Обновления UI (диалог профилей, status bar, runtime payloads).
6. Обновления `requirements.txt` и `env_example.txt`.

**Принцип:** минимальные изменения, повторяющие существующие паттерны `gemini`/`openai`. Не рефакторить существующий код.

---

## 1. Зависимости

### 1.1. Установить пакеты

| Пакет | Версия (latest) | Назначение |
|---|---|---|
| `langchain-anthropic` | `1.4.8` | LangChain-интеграция `ChatAnthropic` |
| `anthropic` | (зависимость `langchain-anthropic`) | Низкоуровневый SDK |

### 1.2. Обновить `requirements.txt`

```
langchain-anthropic==1.4.8
```

> `anthropic` подтянется транзитивно. Зафиксировать версию при необходимости после `pip install`.

### 1.3. Установить в venv

```powershell
pip install langchain-anthropic==1.4.8
pip freeze | findstr anthropic
```

---

## 2. Конфигурация (`core/config.py`)

### 2.1. Расширить `provider` Literal

```python
# Было:
provider: Literal["gemini", "openai"] = Field(default="gemini", alias="PROVIDER")
# Стало:
provider: Literal["gemini", "openai", "anthropic"] = Field(default="gemini", alias="PROVIDER")
```

### 2.2. Добавить Anthropic-поля

В блок «Provider Settings» после OpenAI-полей:

```python
# Anthropic
anthropic_api_key: Optional[SecretStr] = Field(default=None, alias="ANTHROPIC_API_KEY")
anthropic_model: str = Field(default="claude-sonnet-4-5-20250929", alias="ANTHROPIC_MODEL")
anthropic_base_url: Optional[str] = Field(default=None, alias="ANTHROPIC_BASE_URL")
```

> `anthropic_base_url` — опциональный, для совместимости с прокси/агрегаторами (Anthropic-compatible endpoints). По умолчанию `None` → SDK использует `https://api.anthropic.com`.

### 2.3. Добавить поле `max_tokens` для Anthropic

Anthropic Messages API **требует** `max_tokens` (обязательный параметр). Добавить:

```python
anthropic_max_tokens: int = Field(default=8192, alias="ANTHROPIC_MAX_TOKENS")
```

### 2.4. Обновить валидацию ключей

В `validate_provider_keys` (model_validator, mode="after"):

```python
if self.provider == "anthropic" and not self.anthropic_api_key:
    if not self.anthropic_base_url:
        raise ValueError("ANTHROPIC_API_KEY required for anthropic provider.")
```

> Аналогично openai: если задан `base_url` (прокси без ключа), проверка обходится.

---

## 3. Провайдер-адаптер (`core/providers/anthropic.py`)

Новый файл по образцу `gemini.py` / `openai_reasoning.py`.

### 3.1. Фабрика `create_anthropic_chat_model`

```python
def create_anthropic_chat_model(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    if api_key_override is None:
        api_key = config.anthropic_api_key.get_secret_value() if config.anthropic_api_key else None
    else:
        api_key = str(api_key_override or "")

    anthropic_kwargs: dict[str, Any] = {
        "model": config.anthropic_model,
        "temperature": config.temperature,
        "api_key": api_key,
        "max_tokens": config.anthropic_max_tokens,
        "max_retries": 0,  # агент имеет свой retry/recovery слой
    }
    if config.anthropic_base_url:
        anthropic_kwargs["base_url"] = config.anthropic_base_url

    # Extended thinking
    if bool(getattr(config, "enable_model_reasoning", True)):
        thinking_budget = int(getattr(config, "anthropic_thinking_budget", 4096))
        # budget_tokens must be >= 1024 and < max_tokens
        thinking_budget = max(1024, min(thinking_budget, config.anthropic_max_tokens - 1))
        anthropic_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    return ChatAnthropic(**anthropic_kwargs)
```

### 3.2. Особенности Anthropic API (из Context7)

| Особенность | Описание | Действие |
|---|---|---|
| **System prompt** | Передаётся top-level параметром `system=`, а не как `SystemMessage` в массиве messages | `ChatAnthropic` (langchain) автоматически извлекает `SystemMessage` из списка и передает как `system`. **Проверить** — если нет, переопределить `_get_request_payload` |
| **Extended thinking** | `thinking={"type":"enabled","budget_tokens":N}`. `budget_tokens` ≥ 1024 и < `max_tokens` | Реализовать в фабрике. Маппить `model_reasoning_effort` → `budget_tokens` |
| **Thinking blocks round-trip** | Thinking-блоки содержат `signature` для multi-turn continuity. При display="omitted" возвращается `redacted_thinking` | Проверить, что `ChatAnthropic` корректно round-trip-ит thinking-блоки через `additional_kwargs`. При необходимости — переопределить `_convert_*` методы |
| **`max_tokens` обязателен** | В отличие от OpenAI/Gemini | Добавить `anthropic_max_tokens` в конфиг |
| **Image input** | Формат: `{"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}` | Обновить `core/multimodal.py` |
| **Tool calling** | `bind_tools()` поддерживается нативно | Работает через существующий `prepare_llm_with_tools` |
| **Streaming** | События `thinking`, `text`, `content_block_stop` | Проверить совместимость с `ui/streaming.py` |

### 3.3. Reasoning-effort → thinking budget mapping

Добавить конфиг-поле:

```python
anthropic_thinking_budget: int = Field(default=4096, alias="ANTHROPIC_THINKING_BUDGET")
```

Альтернативно (если хочется переиспользовать `model_reasoning_effort`):

| `model_reasoning_effort` | `budget_tokens` |
|---|---|
| `minimal` | 1024 |
| `low` | 2048 |
| `medium` | 4096 |
| `high` | 8192 |
| `xhigh` | 16000 |

> **Решение:** использовать явное поле `ANTHROPIC_THINKING_BUDGET` (как `GEMINI_THINKING_BUDGET`), а `model_reasoning_effort` — только для логирования/диагностики. Это даёт пользователю точный контроль и соответствует паттерну Gemini.
>
> **См. также раздел 17** — альтернативный effort-based подход через `ANTHROPIC_REASONING`, который заменяет `budget_tokens` на уровни `low`/`medium`/`high`/`xhigh`/`max`.

### 3.4. Reasoning-debug инструментация (опционально, фаза 2)

По образцу `ReasoningDebugChatOpenAI` — подкласс `ChatAnthropic` с переопределением `_get_request_payload` / `_generate` / `_astream` для логирования raw chunks в `reasoning_debug.log`. Вынести в отдельный подкласс только если `DEBUG_REASONING_STREAM=true`.

---

## 4. Диспетчеризация провайдеров

### 4.1. `core/providers/factory.py`

```python
from core.providers.anthropic import create_anthropic_chat_model

def create_llm(config, *, api_key_override=None):
    if config.provider == "gemini":
        return create_gemini_chat_model(config, api_key_override=api_key_override)
    if config.provider == "openai":
        return create_openai_chat_model(config, api_key_override=api_key_override)
    if config.provider == "anthropic":
        return create_anthropic_chat_model(config, api_key_override=api_key_override)
    raise ValueError(f"Unknown provider: {config.provider}")
```

### 4.2. `core/providers/__init__.py`

Добавить re-export при необходимости (если тесты/агент ссылаются на anthropic-специфичные хелперы).

---

## 5. Model profiles (`core/model_profiles.py`)

### 5.1. Расширить `ALLOWED_PROVIDERS`

```python
ALLOWED_PROVIDERS = {"openai", "gemini", "anthropic"}
```

### 5.2. Обновить нормализацию профилей

В функциях `_normalize_provider`, `profile_from_env`, `resolve_profile_provider_model_base_url`:

- `base_url` relevant only for `openai` and `anthropic` (для прокси).
- Добавить ветку `anthropic` в выбор модели и API-ключа.

### 5.3. Формат профиля в `.agent_state/config.json`

```json
{
  "id": "claude-sonnet",
  "name": "Claude Sonnet 4.5",
  "provider": "anthropic",
  "model": "claude-sonnet-4-5-20250929",
  "api_key": "sk-ant-...",
  "base_url": "",
  "supports_image_input": true
}
```

---

## 6. Context builder (`core/context_builder.py`)

### 6.1. `normalize_system_prefix`

Текущая логика объединяет несколько `SystemMessage` в один только для `openai`. Проверить, нужно ли это для `anthropic`.

> `ChatAnthropic` (langchain) принимает массив `SystemMessage` и объединяет их в top-level `system` параметр. **Проверить** — если SDK требует одну строку, добавить ветку:
> ```python
> if self.config.provider in ("openai", "anthropic") and len(system_messages) > 1:
>     # merge
> ```

### 6.2. `_message_role_for_provider`

Текущая реализация возвращает `"system"` для `SystemMessage`. Anthropic не имеет роли `system` в messages — но `ChatAnthropic` обрабатывает это на уровне LangChain. **Изменений не требуется**, если `ChatAnthropic` корректно фильтрует system messages.

### 6.3. `_strip_cross_provider_reasoning`

Текущая логика удаляет reasoning-блоки с `type: "reasoning"` (OpenAI Responses API) без ключа `reasoning`. Для Anthropic thinking-блоки имеют `type: "thinking"` с `thinking` и `signature` полями.

**Действие:** расширить фильтрацию:
```python
if isinstance(block, dict) and block.get("type") in ("reasoning", "thinking"):
    # Anthropic thinking blocks have "thinking" + "signature" — preserve for multi-turn
    # OpenAI reasoning blocks have "summary" — strip
    if block.get("type") == "thinking" and "thinking" in block:
        # Anthropic native thinking — preserve (has signature for round-trip)
        new_content.append(block)
        continue
    if "reasoning" not in block:
        block_count += 1
        continue
```

> **Важно:** Anthropic thinking-блоки с `signature` **должны** сохраняться в истории для multi-turn continuity. Не удалять их.

### 6.4. `sanitize_messages` — content normalization

Текущая ветка `if self.config.provider == "openai"` stringify-ит content. Проверить, нужно ли это для `anthropic`. Скорее всего — нет, `ChatAnthropic` принимает list-content. **Изменений не требуется** если тесты пройдут.

---

## 7. Multimodal (`core/multimodal.py`)

### 7.1. `materialize_user_message_content_for_model`

Добавить Anthropic-формат изображений:

```python
def _as_anthropic_image_block(base64_data: str, mime_type: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_data,
        },
    }
```

В `materialize_user_message_content_for_model`:

```python
if provider_name == "openai":
    materialized.append(_as_openai_image_url_block(base64_data, mime_type))
elif provider_name == "anthropic":
    materialized.append(_as_anthropic_image_block(base64_data, mime_type))
else:
    # Gemini-style (default)
    materialized.append({"type": "image", "base64": base64_data, "mime_type": mime_type})
```

---

## 8. API key rotation (`core/api_key_rotation.py`)

### 8.1. Обновить `_resolve_api_key` и `_resolve_model_name`

```python
def _resolve_api_key(self):
    if getattr(self._config, "provider", "") == "gemini":
        secret = getattr(self._config, "gemini_api_key", None)
    elif getattr(self._config, "provider", "") == "anthropic":
        secret = getattr(self._config, "anthropic_api_key", None)
    else:
        secret = getattr(self._config, "openai_api_key", None)
    # ...

def _resolve_model_name(self):
    if getattr(self._config, "provider", "") == "gemini":
        return str(getattr(self._config, "gemini_model", "") or self._profile_id or "model")
    if getattr(self._config, "provider", "") == "anthropic":
        return str(getattr(self._config, "anthropic_model", "") or self._profile_id or "model")
    return str(getattr(self._config, "openai_model", "") or self._profile_id or "model")
```

---

## 9. UI

### 9.1. `ui/widgets/dialogs.py` — диалог профилей моделей

- `ALLOWED_PROVIDERS` импортируется из `core.model_profiles` (обновится автоматически).
- `self.provider_combo.addItems(["openai", "gemini"])` → `["openai", "gemini", "anthropic"]`.
- `_on_provider_changed` / `_update_base_url_field_state` — добавить ветку `anthropic` (base_url опционален, как у openai).
- `_resolve_fetcher` — добавить `AnthropicModelFetcher` для загрузки списка моделей.
- `_append_anthropic_model_items` — метод для сортировки/фильтрации моделей Claude (по образцу `_append_gemini_model_items`).

### 9.2. `core/model_fetcher.py` — `AnthropicModelFetcher`

```python
class AnthropicModelFetcher:
    async def fetch(self, api_key: str, base_url: str = "") -> list[ModelEntry]:
        # Anthropic не имеет публичного /models endpoint.
        # Вариант A: вернуть статичный список актуальных моделей Claude.
        # Вариант B: если base_url задан (прокси с /models), запросить его.
        ...
```

> **Решение:** вернуть статичный список моделей Claude (claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5, и т.д.), так как Anthropic API не предоставляет endpoint для листинга моделей. Если `base_url` задан и поддерживает `/v1/models` (OpenAI-compatible прокси), использовать `OpenAICompatibleModelFetcher`.

### 9.3. `ui/runtime_session.py`

#### `profile_bootstrap_env_from_config`

```python
@staticmethod
def profile_bootstrap_env_from_config(config) -> dict[str, str]:
    provider = str(config.provider or "").strip().lower()
    # ...
    anthropic_key = config.anthropic_api_key.get_secret_value() if config.anthropic_api_key else ""
    if provider == "openai":
        model = config.openai_model
        api_key = openai_key
    elif provider == "anthropic":
        model = config.anthropic_model
        api_key = anthropic_key
    else:
        model = config.gemini_model
        api_key = gemini_key

    return {
        "PROVIDER": provider,
        "MODEL": str(model or ""),
        "API_KEY": str(api_key or ""),
        # ...
        "ANTHROPIC_MODEL": str(config.anthropic_model or ""),
        "ANTHROPIC_API_KEY": str(anthropic_key or ""),
        "ANTHROPIC_BASE_URL": str(config.anthropic_base_url or ""),
        # ...
    }
```

#### `config_overrides_for_profile`

```python
@staticmethod
def config_overrides_for_profile(profile: dict[str, str]) -> dict[str, Any]:
    provider = str(profile.get("provider") or "").strip().lower()
    # ...
    if provider == "openai":
        overrides["openai_model"] = model_name
        overrides["openai_api_key"] = SecretStr(api_key) if api_key else None
        overrides["openai_base_url"] = base_url or None
    elif provider == "anthropic":
        overrides["anthropic_model"] = model_name
        overrides["anthropic_api_key"] = SecretStr(api_key) if api_key else None
        overrides["anthropic_base_url"] = base_url or None
    else:
        overrides["gemini_model"] = model_name
        overrides["gemini_api_key"] = SecretStr(api_key) if api_key else None
    return overrides
```

### 9.4. `ui/runtime_payloads.py`

#### `_provider_model`

```python
def _provider_model(config: AgentConfig) -> tuple[str, str]:
    if config.provider == "gemini":
        return "Gemini", config.gemini_model
    if config.provider == "anthropic":
        return "Anthropic", config.anthropic_model
    return "OpenAI", config.openai_model
```

### 9.5. `ui/widgets/dialogs.py` — `_update_base_url_field_state`

Добавить обработку `anthropic` — показать/скрыть поле `base_url` (опциональное, как у openai).

---

## 10. Streaming и reasoning (`ui/streaming.py`)

### 10.1. Проверить обработку thinking-блоков

Текущий `_describe_thinking_signal` уже распознаёт типы `thinking`, `thought`, `reasoning`, `reasoning_content`. Проверить, что `ChatAnthropic` streaming-чанки попадают в эти ветки.

Anthropic streaming events (из Context7):
- `event.type == "thinking"` → `event.thinking` (text delta)
- `event.type == "text"` → `event.text` (text delta)

`ChatAnthropic` (langchain) конвертирует их в `AIMessageChunk` с content blocks. Проверить, что `_has_thinking_content` корректно детектит thinking.

### 10.2. `_reasoning_type_from_signal`

Уже поддерживает `"thinking_block"`. **Изменений скорее всего не требуется**, но проверить интеграцию.

---

## 11. Provider registry (`provider_registry.json`)

### 11.1. Решение: не добавлять Anthropic в registry

`provider_registry.json` предназначен для URL-matching OpenAI-compatible агрегаторов (reasoning kwargs). Anthropic использует **нативный** API, не OpenAI-compatible. Reasoning (extended thinking) настраивается через `thinking` параметр в фабрике, а не через registry.

> **Исключение:** если `anthropic_base_url` указывает на OpenAI-compatible прокси (например, OpenRouter), то пользователь должен использовать `provider=openai` с соответствующим `base_url`. Registry уже покрывает такие случаи.

---

## 12. Документация и примеры

### 12.1. `env_example.txt`

Добавить блок:

```ini
# --- Anthropic ---
# ANTHROPIC_API_KEY=sk-ant-REPLACE_WITH_YOUR_KEY
# ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
# ANTHROPIC_BASE_URL=https://api.anthropic.com
# Extended thinking budget (tokens). Must be >= 1024 and < ANTHROPIC_MAX_TOKENS.
# ANTHROPIC_THINKING_BUDGET=4096
# ANTHROPIC_MAX_TOKENS=8192
```

### 12.2. `README.md` / `README_EN.md`

Добавить `anthropic` в список поддерживаемых провайдеров и инструкции по настройке.

---

## 13. Тестирование

### 13.1. Существующие тесты

Проверить, что тесты с `provider="openai"` / `provider="gemini"` не сломались. Тесты в `tests/` используют `ln` (обфусцированный `provider`) — убедиться, что `Literal`-расширение не влияет.

### 13.2. Новые тесты

| Тест | Файл | Описание |
|---|---|---|
| `test_anthropic_config` | `tests/test_config.py` (новый или существующий) | `provider="anthropic"` валидация, ключи, `max_tokens` |
| `test_anthropic_factory` | `tests/test_providers.py` | `create_llm` с `provider="anthropic"` возвращает `ChatAnthropic` |
| `test_anthropic_thinking_budget` | `tests/test_providers.py` | `budget_tokens` clamping (≥1024, <max_tokens) |
| `test_anthropic_image_block` | `tests/test_multimodal.py` | `materialize_user_message_content_for_model` с `provider="anthropic"` |
| `test_anthropic_profile` | `tests/test_model_profiles.py` | Нормализация профиля с `provider="anthropic"` |
| `test_context_anthropic_system_prefix` | `tests/test_context_builder.py` | System message handling для anthropic |

### 13.3. Ручное тестирование

1. `PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...`, `ANTHROPIC_MODEL=claude-sonnet-4-5-20250929` → запуск агента.
2. Tool calling: модель вызывает инструменты, tool results корректно возвращаются.
3. Extended thinking: `ANTHROPIC_THINKING_BUDGET=4096` → thinking-блоки стримятся.
4. Multi-turn с thinking: thinking-блоки с `signature` корректно round-trip-ятся.
5. Image input: прикрепить изображение → модель получает Anthropic-формат image block.
6. UI: диалог профилей — выбрать `anthropic`, ввести ключ, загрузить модели, сохранить профиль, активировать.
7. API key rotation: несколько ключей в профиле → ротация работает.

---

## 14. Порядок реализации

### Фаза 1: Базовая поддержка (MVP)

1. **Зависимости** — `pip install langchain-anthropic`, обновить `requirements.txt`.
2. **`core/config.py`** — `Literal`, поля, валидация.
3. **`core/providers/anthropic.py`** — фабрика `create_anthropic_chat_model`.
4. **`core/providers/factory.py`** — диспетчеризация.
5. **`core/model_profiles.py`** — `ALLOWED_PROVIDERS`.
6. **`core/api_key_rotation.py`** — ключи и модель.
7. **`core/multimodal.py`** — Anthropic image block.
8. **`env_example.txt`** — пример конфигурации.

**Контрольная точка:** `PROVIDER=anthropic` запускается из CLI, базовый чат работает.

### Фаза 2: UI и профили

9. **`ui/runtime_payloads.py`** — `_provider_model`.
10. **`ui/runtime_session.py`** — `profile_bootstrap_env_from_config`, `config_overrides_for_profile`.
11. **`ui/widgets/dialogs.py`** — `provider_combo`, `_on_provider_changed`, `_update_base_url_field_state`.
12. **`core/model_fetcher.py`** — `AnthropicModelFetcher`.

**Контрольная точка:** UI диалог профилей поддерживает anthropic, модели загружаются, профили сохраняются.

### Фаза 3: Thinking и edge cases

13. **`core/context_builder.py`** — `_strip_cross_provider_reasoning` (thinking blocks), `normalize_system_prefix` (если нужно).
14. **`ui/streaming.py`** — проверить thinking streaming.
15. **Тесты** — новые + существующие.
16. **Документация** — README.

**Контрольная точка:** Extended thinking работает, multi-turn с thinking-блоками стабилен, тесты зелёные.

---

## 15. Риски и открытые вопросы

| Риск | Вероятность | Митигация |
|---|---|---|
| `ChatAnthropic` не извлекает `SystemMessage` в top-level `system` автоматически | Средняя | Проверить на этапе 1. Если нет — переопределить `_get_request_payload` в подклассе |
| Thinking-блоки теряются при cross-provider switch (anthropic → openai) | Низкая | `_strip_cross_provider_reasoning` уже удаляет чужие reasoning-блоки. Добавить `thinking` type |
| `max_tokens` слишком мал для thinking budget | Средняя | Валидация: `budget_tokens < max_tokens`. Дефолт `max_tokens=8192`, `budget=4096` |
| Anthropic не имеет `/models` endpoint | Точно | `AnthropicModelFetcher` возвращает статичный список |
| `langchain-anthropic` версия несовместима с `langchain-core==1.4.9` | Низкая | `langchain-anthropic==1.4.8` совместима с актуальным langchain 1.x |
| Tool call ID format не проходит `PROVIDER_SAFE_TOOL_CALL_ID_RE` (`^[A-Za-z0-9]{9}$`) | Средняя | Проверить формат ID от Anthropic. При необходимости — хеширование через существующий `_normalize_tool_call_id_for_provider` |

---

## 16. Файлы для изменения (чек-лист)

- [ ] `requirements.txt` — добавить `langchain-anthropic`
- [ ] `core/config.py` — `Literal`, поля, валидация
- [ ] `core/providers/anthropic.py` — **новый файл**
- [ ] `core/providers/factory.py` — диспетчеризация
- [ ] `core/providers/__init__.py` — re-exports (если нужно)
- [ ] `core/model_profiles.py` — `ALLOWED_PROVIDERS`, нормализация
- [ ] `core/api_key_rotation.py` — ключи/модель
- [ ] `core/multimodal.py` — Anthropic image block
- [ ] `core/context_builder.py` — thinking blocks в `_strip_cross_provider_reasoning`, `normalize_system_prefix`
- [ ] `core/model_fetcher.py` — `AnthropicModelFetcher`
- [ ] `ui/runtime_payloads.py` — `_provider_model`
- [ ] `ui/runtime_session.py` — `profile_bootstrap_env_from_config`, `config_overrides_for_profile`
- [ ] `ui/widgets/dialogs.py` — `provider_combo`, fetcher, base_url field
- [ ] `env_example.txt` — Anthropic-блок
- [ ] `README.md` / `README_EN.md` — документация
- [ ] `tests/` — новые тесты

---

## 17. Reasoning через env-параметр `ANTHROPIC_REASONING` (effort-based)

> Источники: Context7 `/anthropics/anthropic-sdk-python` — `OutputConfigParam`, `EffortCapability`, `ThinkingConfigAdaptiveParam`.

### 17.1. Контекст: два механизма reasoning в Anthropic API

Anthropic поддерживает два способа включения extended thinking:

| Механизм | Параметр API | Управление | Раздел плана |
|---|---|---|---|
| **Budget-based** (классический) | `thinking={"type":"enabled","budget_tokens":N}` | Точное число токенов | Раздел 3.3 (`ANTHROPIC_THINKING_BUDGET`) |
| **Effort-based** (новый, 2025) | `thinking={"type":"adaptive"}` + `output_config={"effort":"low"\|"medium"\|"high"\|"xhigh"\|"max"}` | Уровень reasoning | **Этот раздел** (`ANTHROPIC_REASONING`) |

Из SDK (`anthropic-sdk-python`):

```python
# OutputConfigParam — effort levels
class OutputConfigParam(TypedDict, total=False):
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]]
    format: Optional[JSONOutputFormatParam]

# ThinkingConfigAdaptiveParam — adaptive thinking mode
class ThinkingConfigAdaptiveParam(TypedDict, total=False):
    type: Required[Literal["adaptive"]]
    display: Optional[Literal["summarized", "omitted"]]

# EffortCapability — supported levels per model
class EffortCapability(BaseModel):
    high: CapabilitySupport
    low: CapabilitySupport
    max: CapabilitySupport
    medium: CapabilitySupport
    supported: bool
    xhigh: Optional[CapabilitySupport] = None  # may not be available on all models
```

Ключевые отличия от budget-based:
- `thinking.type` = `"adaptive"` (не `"enabled"`) — модель сама решает, сколько думать.
- `budget_tokens` **отсутствует** — бюджет определяется уровнем `effort`.
- `effort` передаётся в `output_config`, **не** в `thinking`.
- Уровень `xhigh` опционален — может не поддерживаться некоторыми моделями.
- Уровень `max` — максимальный reasoning (аналог `xhigh` в OpenAI registry).

### 17.2. Env-параметр `ANTHROPIC_REASONING`

```ini
# .env
ANTHROPIC_REASONING=medium
```

**Допустимые значения:** `low`, `medium`, `high`, `xhigh`, `max` (case-insensitive, нормализуются к lower).

**Поведение:**
- Если `ANTHROPIC_REASONING` задано и непусто → используется **effort-based** подход (`thinking.type="adaptive"` + `output_config.effort`).
- Если `ANTHROPIC_REASONING` не задано/пусто → используется **budget-based** подход из раздела 3.3 (`thinking.type="enabled"` + `budget_tokens`).
- `ANTHROPIC_REASONING=off` (или `none`) → thinking полностью отключён (`thinking.type="disabled"`).

> **Важно:** `ANTHROPIC_REASONING` и `ANTHROPIC_THINKING_BUDGET` — взаимоисключающие. Если заданы оба, приоритет у `ANTHROPIC_REASONING` (effort-based), `ANTHROPIC_THINKING_BUDGET` игнорируется с warning-логом.

### 17.3. Поле конфигурации (`core/config.py`)

```python
# Anthropic reasoning level (effort-based). Overrides ANTHROPIC_THINKING_BUDGET when set.
# Allowed: "low", "medium", "high", "xhigh", "max", "off", "none", "" (empty = use budget mode)
anthropic_reasoning: str = Field(default="", alias="ANTHROPIC_REASONING")
```

Добавить валидацию в `model_validator` (mode="after"):

```python
_ANTHROPIC_REASONING_ALLOWED = {"", "off", "none", "low", "medium", "high", "xhigh", "max"}

if self.provider == "anthropic":
    val = (self.anthropic_reasoning or "").strip().lower()
    if val not in _ANTHROPIC_REASONING_ALLOWED:
        raise ValueError(
            f"ANTHROPIC_REASONING='{self.anthropic_reasoning}' is invalid. "
            f"Allowed: {', '.join(sorted(v or '(empty)' for v in _ANTHROPIC_REASONING_ALLOWED))}"
        )
    self.anthropic_reasoning = val
```

### 17.4. Маппинг `ANTHROPIC_REASONING` → API параметры

| `ANTHROPIC_REASONING` | `thinking` | `output_config.effort` | Описание |
|---|---|---|---|
| `""` (пусто) | `{"type":"enabled","budget_tokens":N}` | — | Budget-based (раздел 3.3) |
| `off` / `none` | `{"type":"disabled"}` | — | Thinking отключён |
| `low` | `{"type":"adaptive"}` | `"low"` | Минимальный reasoning |
| `medium` | `{"type":"adaptive"}` | `"medium"` | Средний (default-подобный) |
| `high` | `{"type":"adaptive"}` | `"high"` | Высокий |
| `xhigh` | `{"type":"adaptive"}` | `"xhigh"` | Очень высокий (не все модели) |
| `max` | `{"type":"adaptive"}` | `"max"` | Максимальный |

### 17.5. Обновление фабрики (`core/providers/anthropic.py`)

Расширить `create_anthropic_chat_model` для effort-based режима:

```python
def create_anthropic_chat_model(config: AgentConfig, *, api_key_override: str | None = None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # ... (api_key, model, max_tokens — как в разделе 3.1) ...

    anthropic_kwargs: dict[str, Any] = {
        "model": config.anthropic_model,
        "temperature": config.temperature,
        "api_key": api_key,
        "max_tokens": config.anthropic_max_tokens,
        "max_retries": 0,
    }
    if config.anthropic_base_url:
        anthropic_kwargs["base_url"] = config.anthropic_base_url

    # --- Reasoning configuration ---
    reasoning_level = (getattr(config, "anthropic_reasoning", "") or "").strip().lower()

    if not bool(getattr(config, "enable_model_reasoning", True)):
        # Global reasoning disabled — turn off thinking
        anthropic_kwargs["thinking"] = {"type": "disabled"}
    elif reasoning_level in ("off", "none"):
        anthropic_kwargs["thinking"] = {"type": "disabled"}
    elif reasoning_level == "":
        # Budget-based (classic) — section 3.3
        thinking_budget = int(getattr(config, "anthropic_thinking_budget", 4096))
        thinking_budget = max(1024, min(thinking_budget, config.anthropic_max_tokens - 1))
        anthropic_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    else:
        # Effort-based (adaptive) — this section
        anthropic_kwargs["thinking"] = {"type": "adaptive"}
        anthropic_kwargs["output_config"] = {"effort": reasoning_level}

    return ChatAnthropic(**anthropic_kwargs)
```

> **Проверить:** поддерживает ли `langchain-anthropic.ChatAnthropic` параметр `output_config` в конструкторе. Если нет — передавать через `model_kwargs` или переопределить `_get_request_payload`. Это **открытый вопрос** для фазы реализации (см. раздел 15, риски).

### 17.6. Взаимодействие с `MODEL_REASONING_EFFORT`

Существующее поле `model_reasoning_effort` (env `MODEL_REASONING_EFFORT`, default `"medium"`) используется для OpenAI/Gemini через `provider_registry.json`.

Для Anthropic:
- `ANTHROPIC_REASONING` — **основной** параметр, если задан.
- `MODEL_REASONING_EFFORT` — **fallback**, если `ANTHROPIC_REASONING` пуст.

Логика приоритета в фабрике:

```python
reasoning_level = (getattr(config, "anthropic_reasoning", "") or "").strip().lower()
if not reasoning_level:
    # Fallback: map MODEL_REASONING_EFFORT to Anthropic effort levels
    effort = (getattr(config, "model_reasoning_effort", "medium") or "medium").strip().lower()
    # OpenAI "minimal" → Anthropic "low"; "xhigh" → "xhigh"
    effort_map = {"minimal": "low", "low": "low", "medium": "medium",
                  "high": "high", "xhigh": "xhigh"}
    reasoning_level = effort_map.get(effort, "medium")
```

| `MODEL_REASONING_EFFORT` | → `ANTHROPIC_REASONING` (fallback) |
|---|---|
| `minimal` | `low` |
| `low` | `low` |
| `medium` | `medium` |
| `high` | `high` |
| `xhigh` | `xhigh` |

> Уровень `max` доступен только через явный `ANTHROPIC_REASONING=max` (нет аналога в `MODEL_REASONING_EFFORT`).

### 17.7. Provider registry (`provider_registry.json`)

Anthropic **не добавляется** в `provider_registry.json` (раздел 11). Reasoning для Anthropic настраивается нативно в фабрике, а не через registry `build_reasoning_kwargs()`.

Однако для консистентности диагностики — добавить логирование в `build_reasoning_kwargs` или в фабрику:

```python
debug_event("anthropic_reasoning_configured", mode=mode, effort=effort_level, budget=budget)
```

где `mode` = `"disabled"` | `"budget"` | `"effort"`.

### 17.8. Streaming (`ui/streaming.py`)

Effort-based reasoning использует `thinking.type="adaptive"`. Streaming-события те же, что и для budget-based:
- `thinking` event → `event.thinking` (text delta)
- `text` event → `event.text` (text delta)

`ChatAnthropic` конвертирует их в `AIMessageChunk` с content blocks. Существующий `_describe_thinking_signal` / `_has_thinking_content` уже распознаёт `thinking` type — **изменений не требуется**.

> **Проверить:** что `thinking.type="adaptive"` не меняет формат streaming-событий по сравнению с `type="enabled"`. Ожидается, что формат идентичен — различие только в том, как модель решает, сколько думать.

### 17.9. `env_example.txt`

Добавить в Anthropic-блок (раздел 12.1):

```ini
# --- Anthropic reasoning ---
# Reasoning level (effort-based). Overrides ANTHROPIC_THINKING_BUDGET when set.
# Allowed: low, medium, high, xhigh, max, off, none
# Empty = use ANTHROPIC_THINKING_BUDGET (budget-based mode)
# ANTHROPIC_REASONING=medium
```

### 17.10. Тесты

| Тест | Файл | Описание |
|---|---|---|
| `test_anthropic_reasoning_effort_low` | `tests/test_providers.py` | `ANTHROPIC_REASONING=low` → `thinking.type="adaptive"`, `output_config.effort="low"` |
| `test_anthropic_reasoning_effort_max` | `tests/test_providers.py` | `ANTHROPIC_REASONING=max` → `output_config.effort="max"` |
| `test_anthropic_reasoning_off` | `tests/test_providers.py` | `ANTHROPIC_REASONING=off` → `thinking.type="disabled"` |
| `test_anthropic_reasoning_empty` | `tests/test_providers.py` | `ANTHROPIC_REASONING=""` → budget-based (`thinking.type="enabled"`) |
| `test_anthropic_reasoning_invalid` | `tests/test_config.py` | `ANTHROPIC_REASONING=invalid` → `ValueError` |
| `test_anthropic_reasoning_fallback` | `tests/test_providers.py` | `ANTHROPIC_REASONING=""` + `MODEL_REASONING_EFFORT=high` → `effort="high"` |
| `test_anthropic_reasoning_priority` | `tests/test_providers.py` | `ANTHROPIC_REASONING=high` + `ANTHROPIC_THINKING_BUDGET=8192` → effort-based, budget ignored |

### 17.11. Обновление чек-листа (раздел 16)

Добавить:
- [ ] `core/config.py` — поле `anthropic_reasoning`, валидация
- [ ] `core/providers/anthropic.py` — effort-based ветка в фабрике
- [ ] `env_example.txt` — `ANTHROPIC_REASONING`
- [ ] `tests/` — тесты effort-based reasoning

### 17.12. Риски (дополнение к разделу 15)

| Риск | Вероятность | Митигация |
|---|---|---|
| `ChatAnthropic` не поддерживает `output_config` в конструкторе | Средняя | Проверить сигнатуру `ChatAnthropic.__init__`. Если нет — передать через `model_kwargs={"output_config": {...}}` или переопределить `_get_request_payload` |
| Уровень `xhigh` не поддерживается моделью | Средняя | `EffortCapability.xhigh` — optional. Ловить API-ошибку и fallback на `high` с warning |
| `thinking.type="adaptive"` требует beta-флаг | Низкая | Проверить, нужен ли `betas=["adaptive-thinking-2025-..."]` в SDK. Если да — добавить через `default_headers` или `extra_headers` |
| `output_config.effort` конфликтует с `temperature` | Низкая | Проверить docs: effort-based может требовать `temperature=1.0`. При необходимости — override temperature в effort-режиме |

---

## 18. Ссылки на источники (Context7)

- **Anthropic SDK Python:** `/anthropics/anthropic-sdk-python`
  - `client.messages.create()` — базовый вызов
  - `client.messages.stream()` — стриминг с `thinking` и `text` event types
  - `thinking={"type":"enabled","budget_tokens":N}` — extended thinking
  - `thinking={"type":"adaptive"}` + `output_config={"effort":"low"|"medium"|"high"|"xhigh"|"max"}` — effort-based reasoning (раздел 17)
  - `OutputConfigParam` — `effort: Literal["low","medium","high","xhigh","max"]`
  - `EffortCapability` — supported levels per model (`xhigh` optional)
  - `ThinkingConfigAdaptiveParam` — `type="adaptive"`, `display: "summarized"|"omitted"`
  - `BetaThinkingConfigEnabledParam` — `budget_tokens` ≥ 1024, < `max_tokens`, `display: "summarized"|"omitted"`
  - `BetaRedactedThinkingBlockParam` — round-trip для omitted thinking
  - `ThinkingBlockParam` — `signature`, `thinking`, `type: "thinking"`
  - System prompt — top-level `system` parameter, не message role
  - Tool calling — `tools=[...]` в `messages.create()`, `@beta_tool` / `to_dict()`

- **LangChain:** `/websites/langchain`
  - `ChatAnthropic(model="claude-sonnet-4-6")` — инициализация
  - `bindTools([tool])` — tool calling
  - `thinking: { type: "enabled", budget_tokens: 5000 }` — в конструкторе `ChatAnthropic`
  - `streamEvents()` — `message.reasoning` и `message.text` для streaming thinking
  - Поддерживает: tool calling, structured output, image input, token streaming, async, token usage
  - Не поддерживает: audio/video input, logprobs
