🚀 Portable Autonomous AI Agent (GUI)

Portable desktop AI agent с графовым runtime ("LangGraph") и GUI на "PySide6".
Запускается как ".exe" без установки — можно перенести на флешке и работать на любом ПК.

![Portable Autonomous AI Agent](./img/01.jpg)

---

✨ Ключевая идея

«Собрал один раз → перенёс → запустил → работаешь»

- ❌ Без установки Python / Node / IDE
- ❌ Без привязки к конкретной машине
- ✔ Работает в ограниченных средах
- ✔ Полный контроль над поведением
- ✔ Локальная работа с файлами и системой

---

⚙️ Что умеет агент

- 📂 Анализ проектов и структуры кода
- ✏️ Редактирование файлов с показом diff
- 🖥️ Выполнение CLI-команд
- 🔍 Поиск по файлам и директориям
- 🌐 Web-поиск и загрузка контента
- 🌐 Пакетный web-поиск (batch_web_search)
- 🔌 Подключение внешних инструментов через MCP
- 🔄 Автоматическое восстановление после ошибок
- ⚠️ Approval для потенциально опасных действий
- 🧠 Мультимодальный ввод (включая изображения)
- 💾 Сохранение сессий и чекпоинтов

---

🧠 Как он работает

Граф выполнения:

summarize → classify_turn → agent → {approval | tools | recovery | END}

---

Основные принципы:

- System-driven execution — поведение задаётся системой
- Bounded recovery — нет бесконечных циклов
- Tool-first подход — агент действует, а не просто отвечает
- Прозрачность — все действия видны

---

👤 Кому это может быть полезно

- DevOps / SRE
- Системные администраторы
- Разработчики (вне IDE)
- Диагностика и отладка
- Ограниченные среды

---

⚠️ Чем это не является

Не замена:

- IDE с AI (Cursor, Windsurf)
- Облачных ассистентов (Claude, ChatGPT)
- CLI-инструментов вендоров

---

Это:

«дополнительный инструмент с фокусом на портативность и контроль»

---

🔌 MCP (Model Context Protocol)

Агент поддерживает подключение внешних инструментов через "mcp.json".

Позволяет:

- подключать сторонние MCP-серверы
- расширять функциональность без изменения кода
- использовать внешние API и сервисы

Если файл "mcp.json" присутствует — инструменты подключаются автоматически.

---

🖥️ Интерфейс (GUI)

- Transcript с шагами выполнения
- Tool-cards
- Diff изменений
- Время выполнения
- Sidebar сессий
- Lazy visibility

---

📦 Portable режим

```powershell
.\build.bat
```

→ скопировать `.exe` и соседние runtime-файлы из папки сборки → запустить на другом ПК

---

💡 Почему portable важно

- Работа на чужих машинах
- Нет зависимостей
- Подходит для ограниченных сред
- Полная автономность

---

📁 Структура проекта

```text
.
├─ agent.py
├─ core/
├─ tools/
├─ ui/
├─ tests/
├─ utils/
├─ env_example.txt
├─ requirements.txt
├─ mcp.json
├─ prompt.txt
└─ build.bat
```

---

🧩 Основные компоненты

- "core/" — runtime, policy, recovery
- "tools/" — локальные и MCP инструменты
- "ui/" — GUI
- "agent.py" — сборка агента

---

🛠️ Инструменты

Filesystem

- read/write/edit file
- search / list directory

Shell

- "cli_exec"

Search

- "web_search"
- "fetch_content"
- "batch_web_search"

System / Process

- system info
- background processes

MCP

- внешние инструменты через "mcp.json"

User

- "request_user_input"

---

🧪 Пример

User: Найди информацию по теме X

Агент:

- выполняет batch_web_search
- агрегирует результаты
- возвращает сводку

---

🚀 Быстрый старт

```bash
python -m venv venv
pip install -r requirements.txt
```

```powershell
Copy-Item env_example.txt .env
```

```bash
python main.py
```

---

⚙️ Конфигурация

Provider

- `PROVIDER=gemini|openai`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`

Пути

- `PROMPT_PATH`
- `MCP_CONFIG_PATH`
- `SESSION_STATE_PATH`
- `RUN_LOG_DIR`

Features

- `ENABLE_*_TOOLS`
- `ENABLE_APPROVALS`
- `MODEL_SUPPORTS_TOOLS`

---

🧪 Тесты

```bash
python -m unittest discover -s tests -v
```

---

🧠 Архитектура

- Graph execution
- Provider-aware routing
- Multimodal support
- MCP extensibility
- Self-correction
- Approval system

---

⚠️ Требования

- Интернет нужен только для web-search/MCP/внешних API.
- API-ключ нужен для выбранного провайдера (`Gemini` или `OpenAI-compatible`).
- Для локальных file/system/process задач агент может работать без веб-доступа.

---

🚀 Итог

«Минимум зависимости → максимум контроля»

Агент — это инструмент выполнения задач,
а не просто интерфейс к LLM.
