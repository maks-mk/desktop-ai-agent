# Code Review: ядро агента

Дата: 2026-05-20

Область ревью: `agent.py`, `core/state.py`, `core/nodes/*`, `core/node_orchestrators.py`, `core/context_builder.py`, `core/tool_executor.py`, `core/checkpointing.py`, граница стриминга в `ui/runtime_worker.py` / `ui/streaming.py`.

## База LangGraph

Ревью сверено с практиками LangGraph через Context7:

- использовать `MessagesState` / `add_messages` для истории сообщений, чтобы новые сообщения добавлялись reducer-ом, а не перезаписывали список;
- строить agent loop через `StateGraph`, условные переходы после LLM-ноды и отдельную tools-ноду;
- использовать checkpointer для `interrupt()` и human-in-the-loop approval;
- разделять token stream (`messages`) и state updates (`updates`), потому что это разные источники событий;
- для transient failures использовать retry/checkpointing на границах нод, а не размазывать восстановление по UI.

## Краткий вывод

Архитектура ядра в целом зрелая: есть `add_messages`, checkpointer, approval через `interrupt`, preflight tool validation, recovery loop, loop guard, run logs и отдельные сервисы для контекста/инструментов. Это хорошая база для LangGraph-агента.

Главный риск сейчас не в самой topology графа, а в слоях совместимости с провайдерами и tool-calling протоколом. В коде накопились защитные механизмы против разных OpenAI-compatible агрегаторов: private SDK overrides, reasoning stream parsing, textual tool-call recovery, UI dedupe. Они полезны, но часть из них уже стала слишком широкой для core path и может создавать побочные эффекты: дубли комментариев, pseudo tool calls, отличия между провайдерами.

## Implementation Status

- [x] Добавлен безопасный default для textual tool-call recovery: текстовые markers больше не исполняются как tool calls без явного флага.
- [x] Добавлен `ENABLE_TEXT_TOOL_CALL_RECOVERY=false` в config/env/README.
- [x] Добавлены типизированные `turn_outcome` constants и normalization helper.
- [x] Уточнен stream guard для `updates_agent` tool-call preface replay.
- [x] Добавлены unit tests для textual recovery default/legacy, outcome normalization и stream replay suppression.
- [x] Полный тестовый набор пройден: `556 passed`.
- [ ] Deferred: вынести provider adapters из `agent.py` в `core/providers/`.
- [ ] Deferred: заменить multiple inheritance/service owner calls на явный runtime services container.

## Findings

### P1: Provider adapters слишком глубоко встроены в `agent.py`

Места:

- `agent.py:239` - `_build_gemini_thought_signature_adapter`
- `agent.py:369` - `_patch_langchain_google_genai_retry_kwargs`
- `agent.py:494` - вложенный `ReasoningDebugChatOpenAI`
- `agent.py:549` - override private `_stream`
- `agent.py:678` - override private `_astream`

Проблема: `create_llm()` отвечает сразу за слишком многое: выбор провайдера, reasoning kwargs, Gemini thought signatures, OpenAI-compatible stream debug, monkey patches и fallback-поведение. Часть логики опирается на private методы LangChain/OpenAI wrappers. Это хрупко при обновлениях зависимостей и усложняет диагностику багов вроде дублирования финального ответа или пропажи/задержки thinking status.

Рекомендация:

- вынести provider-specific код в `core/providers/`;
- сделать отдельные фабрики вроде `create_openai_chat_model()`, `create_gemini_chat_model()`;
- оставить в `agent.py` только orchestration-level выбор;
- debug stream wrapper включать только через явный флаг логирования;
- private overrides покрыть маленькими provider regression tests, чтобы обновление LangChain/OpenAI SDK не ломало stream contract незаметно.

Это не требует срочной переписи всего ядра, но это самый полезный следующий рефакторинг.

### P1: Textual tool-call recovery выполняет prose как инструментальный вызов

Места:

- `core/nodes/agent.py:75`
- `core/nodes/agent.py:76`
- `core/nodes/agent.py:81`
- `core/text_tool_calls.py:8`
- `core/text_tool_calls.py:22`

Проблема: если модель пишет текст вида `call:read_file{...}<tool_call|>`, core может восстановить это как настоящий `tool_calls`. Это помогает с некоторыми сбойными моделями/агрегаторами, но нарушает обычный контракт structured tool calling: видимый assistant text превращается в действие. Именно такой слой может усиливать проблемы с Ollama/Gemma-подобным поведением, где модель "проговаривает" tool call вместо структурного вызова.

Рекомендация:

- отключить textual recovery по умолчанию для провайдеров, где structured `tool_calls` работают;
- включать только через явный allowlist в конфиге провайдера/модели;
- логировать recovery как protocol repair;
- не показывать marker-текст в UI даже кратковременно;
- для write/edit tools всегда сохранять approval/preflight guard.

Долгосрочно лучше считать textual tool call protocol error/recovery case, а не нормальным способом вызова инструмента.

### P2: Роутинг графа завязан на loose string state

Места:

- `agent.py:953`
- `agent.py:965`
- `agent.py:973`
- `agent.py:989`
- `agent.py:994`
- `agent.py:1004`
- `core/nodes/agent.py:215`
- `core/nodes/agent.py:225`

Проблема: переходы зависят от `turn_outcome` и нескольких полей состояния (`pending_approval`, `open_tool_issues`, `recoverable_tool_errors`, tool calls в последнем сообщении). Это работает, но контракт между `AgentMixin`, `RecoveryManager`, `ToolBatchCoordinator` и route-функциями не типизирован. Ошибка в строке outcome или незачищенное поле состояния может направить граф не туда.

Рекомендация:

- ввести `TurnOutcome = Literal["run_tools", "recover_agent", "finish_turn"]` или enum;
- вынести route helpers в отдельный маленький модуль;
- добавить unit tests на route table;
- для нод, которые уже точно знают следующий шаг, рассмотреть `Command(goto=...)`.

Сама LangGraph-схема близка к best practice, улучшить стоит именно явность контракта.

### P2: `AgentNodes` держится на multiple inheritance и скрытых owner-зависимостях

Места:

- `core/nodes/__init__.py:27`
- `core/node_orchestrators.py`

Проблема: `AgentNodes` собирается через набор mixin-ов, а orchestrator-ы вызывают много методов через `owner._...`. Получается гибко, но тяжело читать: реальная зависимость ноды не видна из конструктора, любой mixin потенциально связан с любым другим.

Рекомендация:

- постепенно перейти к явному контейнеру сервисов, например `AgentRuntimeServices`;
- orchestrator-ам передавать конкретные зависимости: context builder, tool executor, recovery manager, LLM invoker;
- оставить node methods тонкими адаптерами для LangGraph.

Это снизит связность и упростит будущий перенос provider adapters из `agent.py`.

### P2: Streaming contract стоит сделать более строгим

Места:

- `ui/runtime_worker.py:437`
- `ui/streaming.py`

Проблема: граф стримится с `stream_mode=["messages", "updates"]`. По LangGraph это два разных потока: token chunks и state updates. Если UI считает оба источника полноценным assistant text, появляются временные дубли: комментарий уже пришел как chunk, потом появляется в state update, потом dedupe/финализация его убирает.

Рекомендация:

- зафиксировать правило: основной assistant text рендерится из `messages`;
- `updates.agent.messages` использовать для tool metadata, финальной сверки и fallback, когда token stream не дал текста;
- текущий dedupe оставить как provider compatibility guard, но не как основной механизм нормализации;
- добавить тест на порядок событий: assistant comment -> tool running -> tool completed -> next assistant comment без временного дубля.

Это должно помочь кейсам, где некоторые OpenAI-compatible провайдеры присылают события в нестабильном порядке.

### P3: Step counter выглядит как graph loop counter, но по смыслу это LLM iteration counter

Места:

- `agent.py:942`
- `agent.py:965`
- `ui/runtime_worker.py:98`

Проблема: `steps` увеличивается перед `agent`-нодой и используется для `MAX_LOOPS`. Это нормально как guard от бесконечных LLM turns, но название легко спутать с LangGraph supersteps или количеством tool calls.

Рекомендация:

- переименовать внутренне в `agent_iterations` / `llm_turns`, либо добавить комментарий в state;
- отдельно держать `recursion_limit`, как сейчас делает `build_graph_config()`.

### P3: Суммаризация использует основной runtime LLM

Места:

- `core/nodes/summarize.py:95`

Проблема: summary-нода вызывает основной `self.llm`. Если активная модель дорогая, reasoning-enabled или нестабильная у агрегатора, автосуммаризация наследует те же риски. Ошибки summary сейчас безопасно проглатываются, но качество/стоимость может плавать.

Рекомендация:

- добавить опциональный `SUMMARY_MODEL_PROFILE_ID` или lightweight summarizer;
- для summary отключать tool calling и reasoning kwargs;
- логировать summary model, token estimate и факт skip/failure.

## Что сделано хорошо

- `AgentState.messages` использует `add_messages`, это правильный LangGraph-паттерн для истории.
- Approval через `interrupt()` совместим с checkpointer-подходом LangGraph.
- `AsyncSqliteSaver` / `MemorySaver` вынесены в отдельный слой, setup/close обработаны аккуратно.
- Tool execution имеет preflight, approval gates и structured error output.
- Параллельное выполнение ограничено read-only whitelist, это консервативно и правильно.
- `ContextBuilder` проверяет tool history mismatch и нормализует provider-safe tool call ids.
- Recovery state и run logging уже дают хорошую основу для диагностики.

## Рекомендуемый порядок работ

1. Вынести provider adapters из `agent.py` в `core/providers/`.
2. Перевести textual tool-call recovery на явный provider/model allowlist.
3. Типизировать `turn_outcome` и покрыть route-функции unit tests.
4. Уточнить stream contract: рендерить assistant text из одного основного источника.
5. Добавить отдельный summary profile или режим summary без reasoning.

## Итог

Ядро не выглядит сломанным или построенным против LangGraph. Основная topology близка к рекомендованному agent loop: summarize/context -> LLM -> route -> tools/approval/recovery -> next LLM turn. То, что сейчас стоит улучшать, находится вокруг графа: compatibility layers, stream normalization и слишком широкая эвристика восстановления tool calls из текста.

Если двигаться маленькими шагами, первый лучший шаг - отделить provider-specific reasoning/stream adapters от graph runtime. После этого станет проще чинить отдельные провайдеры без риска задеть общую логику агента.
