# Portable Autonomous AI Agent (GUI)

[English README](./README_EN.md)

> *"Created by a SysAdmin for developers. Focus on safety, portability, and zero-nonsense execution. No Docker, no heavy environments, just one binary."*

Десктопный AI-агент с runtime на `LangGraph` и графическим интерфейсом на `PySide6`.  
Работает с файлами, shell-командами, системой, MCP-серверами и веб-поиском.

Запуск из исходников: `python main.py`.  
Сборка в portable `.exe` для Windows: `build.bat`.

![Portable Autonomous AI Agent](./img/01.jpg)

---

## Цель проекта

Это не AI IDE. Цель — предоставить переносимого автономного помощника, которого можно скопировать на другой компьютер и сразу использовать с минимальной настройкой.

Основные приоритеты:

- переносимость;
- безопасность;
- локальные инструменты;
- автоматизация;
- работа с файлами и системой;
- поиск информации в интернете;
- надёжность.

Проект не стремится конкурировать с AI IDE по количеству функций и не пытается заменить специализированные инструменты для разработки. Фокус — практическое выполнение повседневных задач: работа с файлами, shell-командами, системой, веб-поиском, документацией, скриптами и автоматизацией в одном переносимом приложении без сложной инфраструктуры и дополнительных сервисов.

---

## Возможности

- Графовый runtime на `LangGraph` с bounded recovery и self-correction
- Mixed-mode parallel tool batch: read-only инструменты запускаются параллельно через `asyncio.gather`, остальные — последовательно; результаты собираются в исходном порядке
- GUI: история чатов, streaming transcript, tool cards, approvals, user-choice карточки, вложения
- Fuzzy replay suppression: повторный префейс модели после tool-вызова подавляется даже при минимальных расхождениях текста (опечатки, пунктуация)
- Live CLI output streaming: вывод shell-команд отображается в карточке инструмента в реальном времени, а не только после завершения
- Exit-code-neutral команды: `grep`, `rg`, `vulture`, `pytest`, `diff` и др. с ненулевым exit code не помечаются как ошибка — вывод возвращается с префиксом `Exit Code: N`
- Stream-interruption recovery с классификацией ошибок (`rate_limit` / `timeout` / `server_error` / `network`) и экспоненциальным backoff с джиттером перед авто-продолжением
- Инструменты: filesystem, shell, web search, system info, process management, MCP
- Approval-паузы перед мутирующими и деструктивными действиями
- Автосуммаризация контекста при длинных сессиях
- Несколько профилей моделей с переключением прямо в GUI
- Durable checkpoints — сессии сохраняются между запусками
- Опциональный image input, если модель его поддерживает

---

## Быстрый старт

Требования: **Python 3.10+**, API-ключ Gemini или OpenAI.

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
Copy-Item env_example.txt .env
# Открой .env и укажи API-ключ
python main.py
```

---

## Portable сборка

```powershell
.\build.bat
```

Использует `PyInstaller` в режиме `--onefile --windowed`. Результат — один `.exe` без зависимостей.

---

## Архитектура

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

Подробнее: [Runtime Flow, Prompt Layers, Sessions & Checkpoints](./docs/ARCHITECTURE.md)

---

## Структура проекта

```text
.
├── main.py                # Точка входа GUI
├── agent.py               # Сборка графа LangGraph: routing, tool binding, checkpointing
├── prompt.txt             # Основной системный промпт
├── mcp.json               # Конфигурация MCP-серверов
├── env_example.txt        # Шаблон .env
├── provider_registry.json # Reasoning kwargs для OpenAI-compatible провайдеров
├── build.bat              # Сборка portable .exe
├── requirements.txt
├── core/                  # Ядро агента: config, state, policies, recovery, provider registry
│   ├── nodes/             # Узлы LangGraph: context, llm, agent, tools, approval, recovery
│   └── providers/         # Provider-адаптеры (gemini, openai_reasoning, factory)
├── tools/                 # Filesystem, shell, search, system, process, user input, MCP registry
├── ui/                    # PySide6 GUI, runtime worker, streaming/status handling
├── docs/                  # Документация
├── tests/                 # Runtime, UI, tools, provider registry, logging, policies
├── .agent_state/          # Локальное состояние, профили, checkpoints
└── logs/                  # JSONL/runtime/debug logs
```

Полная карта модулей: [`docs/PROJECT_STRUCTURE.md`](./docs/PROJECT_STRUCTURE.md)

---

## Тесты

Полный целевой regression-набор для runtime/GUI/streaming: 495 тестов.

```powershell
venv\Scripts\python.exe -m pytest
```

---

## Зависимости

### ripgrep (`rg`) — рекомендуется

Для более эффективного поиска по файлам, логам, конфигурациям и кодовым базам рекомендуется установить [`ripgrep`](https://github.com/BurntSushi/ripgrep). Готовые сборки для Windows — в [Releases](https://github.com/BurntSushi/ripgrep/releases) (архив `x86_64-pc-windows-msvc.zip`).

Для portable-сборки скопируйте `rg.exe` рядом с исполняемым файлом агента. Если `rg` отсутствует, агент продолжит работать в обычном режиме, используя стандартные инструменты файловой системы.

### Python-пакеты

| Пакет | Назначение |
|---|---|
| `langgraph` | Граф агента и state management |
| `langchain` | LLM abstraction, tool calling |
| `langchain-google-genai` | Gemini provider |
| `langchain-openai` | OpenAI / compatible provider |
| `langchain-mcp-adapters` | MCP интеграция |
| `PySide6` | GUI |
| `pydantic-settings` | Конфигурация через `.env` |
| `tiktoken` | Подсчёт токенов для суммаризации |
| `tavily-python` | Web search |
| `psutil` | Системные инструменты и процессы |
| `httpx` | HTTP для MCP и fetch |
| `aiofiles` | Async файловые операции |
| `mcp` | Model Context Protocol |
| `requests` | HTTP-клиент (Google API, Tavily) |
| `sqlite-vec` | Vector-расширение для SQLite checkpoints |

---

## Документация

| Документ | Содержание |
|---|---|
| [Архитектура](./docs/ARCHITECTURE.md) | Runtime Flow, Prompt Layers, Sessions & Checkpoints |
| [Конфигурация](./docs/CONFIGURATION.md) | Все переменные `.env` (провайдеры, runtime, фиче-флаги, лимиты, retry, персистентность, диагностика) |
| [GUI](./docs/GUI_GUIDE.md) | Transcript, CLI output widget, Composer, горячие клавиши |
| [Безопасность](./docs/SECURITY.md) | Approvals, workspace boundary, `request_user_input` |
| [Профили моделей](./docs/MODEL_PROFILES.md) | Управление профилями, автозагрузка моделей, ротация API-ключей |
| [MCP](./docs/MCP.md) | Конфигурация MCP-серверов, policy, пример |
| [Структура проекта](./docs/PROJECT_STRUCTURE.md) | Полная карта модулей |
| [Provider Registry](./docs/provider_registry_guide.md) | Добавление OpenAI-compatible агрегаторов |
| [Dead Code Analysis](./docs/DEAD_CODE_ANALYSIS.md) | Отчёт по очистке мёртвого кода |
