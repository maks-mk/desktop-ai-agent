# Portable Autonomous AI Agent (GUI)

[Russian README](./README.md)

> *"Created by a SysAdmin for developers. Focus on safety, portability, and zero-nonsense execution. No Docker, no heavy environments, just one binary."*

A desktop AI agent with a `LangGraph` runtime and a `PySide6` GUI.  
It works with files, shell commands, the operating system, MCP servers, and web search.

Run from source: `python main.py`.  
Build a portable Windows `.exe`: `build.bat`.

![Portable Autonomous AI Agent](./img/01.jpg)

---

## Project Goal

This is not an AI IDE. The goal of the project is to provide a portable autonomous assistant that can be copied to another computer and used immediately with minimal setup.

Main priorities:

- portability;
- safety;
- local tools;
- automation;
- working with files and the operating system;
- web search;
- reliability.

The project does not try to compete with AI IDEs by feature count and does not try to replace specialized development tools. Its focus is practical execution of everyday tasks: working with files, shell commands, the system, web search, documentation, scripts, and automation in one portable application without complex infrastructure or additional services.

---

## Features

- Graph runtime on `LangGraph` with bounded recovery and self-correction
- GUI: chat history, streaming transcript, tool cards, approvals, attachments
- Tools: filesystem, shell, web search, system info, process management, MCP
- Approval pauses before mutating and destructive actions
- Automatic context summarization for long sessions
- Multiple model profiles with switching directly in the GUI
- Durable checkpoints: sessions persist between launches
- Optional image input when the selected model supports vision

---

## Quick Start

Requirements: **Python 3.10+**, plus a Gemini or OpenAI API key.

```powershell
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
Copy-Item env_example.txt .env
# Open .env and add your API key
python main.py
```

---

## Portable Build

```powershell
.\build.bat
```

Uses `PyInstaller` in `--onefile --windowed` mode. The result is a single `.exe` without external runtime dependencies.

---

## Runtime Flow

```text
START
  -> summarize        # compact context if the session has grown large
  -> update_step
  -> agent            # LLM decides: answer / call tool / recover
     -> approval      # pause before a mutating action
        -> tools
     -> tools         # execute tool calls
        -> recovery   # if a tool returned an error
        -> update_step
     -> recovery      # if the agent returned a protocol error or loop
        -> update_step
        -> END
     -> END
```

- `MAX_LOOPS` and per-tool loop guards prevent infinite loops.
- Recovery uses stateful error tracking: `attempts_by_strategy`, `progress_markers`, `llm_replan_attempted_for` - adaptive retries based on unique error fingerprints.
- When the problem changes with a new fingerprint, the retry budget resets. For the same problem, multiple `llm_replan` attempts are allowed within `SELF_CORRECTION_RETRY_LIMIT`.

---

## GUI

**Transcript** - streaming answers, tool cards with arguments and results, summary messages, status notices.

**Composer:**

- Insert files through `Add files...`, drag-and-drop, or clipboard paste
- `@` mention popup for files and directories in the current workspace, refreshed dynamically
- Text normalization before sending: `\r\n` -> `\n`, control characters removed
- 10,000 character request limit with an inline warning when truncated

**Hotkeys:**

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Ctrl+N` | New chat |
| `Ctrl+B` | Show / hide sidebar |
| `Ctrl+I` | Info popup |
| `Up` / `Down` in an empty composer | Sent message history |

---

## Safety

- Approvals are enabled by default for write, delete, move, and process-launch operations
- Shell commands are classified before execution: read-only / mutating / destructive
- MCP tools require approval unless `policy.read_only` is explicitly set to `true`
- Tool errors move execution into recovery and are not ignored
- Workspace boundary checks prevent mutating operations from escaping the workspace folder
- API keys, bearer tokens, and query tokens are redacted from logs through `SensitiveDataFilter`
- `MAX_BACKGROUND_PROCESSES` limits the number of background processes

`request_user_input` is a separate tool for blocking user choices:

- at most one call per turn;
- cannot be batched with other tool calls in the same response;
- use only when the next step is truly blocked by one specific user choice or an external value that cannot be obtained from context or tools;
- do not use it for approval of risky actions: that is a separate flow;
- ask exactly one short question;
- pass 2-5 short mutually exclusive options;
- if one option is clearly best, mark it with `recommended`;
- after resume, continue with the selected answer instead of asking again in the same turn.

---

## Model Profiles

Multiple model profiles are stored through `core/model_profiles.py` and switched in the GUI.

Each profile contains: provider, model name, API key, optional `base_url` for OpenAI-compatible backends, image input flag, and enabled/disabled status.

- The active profile is selected in the GUI; `.env` is used only to bootstrap the initial set
- Legacy keys `MODEL`, `API_KEY`, and `BASE_URL` are supported for import and compatibility
- `base_url` is ignored for Gemini profiles

### Automatic Model List Loading

When adding or editing a profile in `Model Profiles`, the model list is loaded automatically, so the model name does not have to be typed manually.

**Gemini:** after entering an API key, the list loads automatically with a 600 ms debounce. Models are filtered to those that support `generateContent` and belong to the `gemini` / `gemma` families. Embedding, audio, image, and service models are excluded automatically. The list is grouped and sorted by descending version.

**OpenAI-compatible:** the list loads after both fields are filled: API key and Base URL. Basic keyword filtering is applied. If loading fails, the field switches to manual input mode.

The logic lives in `core/model_fetcher.py`. When switching between profiles with the same key, the list is loaded from cache without another request.

### API Key Rotation

The agent supports automatic rotation of an API key pool for each profile. This helps work around free-tier rate limits and keeps runs going.

- **How it works:** you can specify multiple keys for one model. If the current key hits a limit or returns an error, the agent moves to the next key in a circular order and retries without interrupting the session. After one full cycle, if no key worked, execution stops with: *"All API keys have been used without success. Please try again later or check your key limits and validity."* The user then decides what to do next.
- **Management:** available in the GUI through the circular-arrows button next to the API Key field in the profile editor.
- **Safety:** keys are not marked invalid and are not removed from the pool. A key may become usable again later, for example after a rate limit resets, so the pool remains unchanged.

---

## Sessions and Checkpoints

- Graph checkpoints: `sqlite` by default, or `memory`
- `.agent_state/checkpoints.sqlite` - durable checkpoint store
- `.agent_state/session.json` - active session
- `.agent_state/session_index.json` - all sessions index
- `logs/runs/` - JSONL logs for each run

---

## MCP

`mcp.json` defines optional MCP servers. All servers are disabled by default.

Policy behavior:

| `policy.read_only` | Behavior |
|---|---|
| `true` | Tool is considered read-only, approval is not required |
| `false` | Requires approval |
| missing | Conservative mode: approval by default |

Minimal remote server example:

```json
{
  "context7": {
    "type": "remote",
    "url": "https://mcp.context7.com/mcp",
    "transport": "http",
    "enabled": true,
    "policy": {
      "read_only": true
    }
  }
}
```

---

## Prompt Layers

The prompt is assembled from several layers on every agent call:

| Layer | File / module | Contents |
|---|---|---|
| Base | `prompt.txt` | Main system prompt |
| Runtime | `core/runtime_prompt_policy.py` | OS, shell, workspace, date, tool policy |
| Safety | `core/context_builder.py` | Workspace boundary, shell warning |
| Recovery | `core/recovery_manager.py` | Instructions for the active error |
| Memory | state: `summary` | Auto-summarized context from previous turns |

---

## Configuration

All settings are read from `.env` through `core/config.py`. Copy `env_example.txt` to `.env` and fill in the values you need.

### Providers and Models

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_API_KEY` | - | Required for Gemini |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | - | Required for OpenAI unless `OPENAI_BASE_URL` is set |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `OPENAI_BASE_URL` | - | For OpenAI-compatible backends such as Ollama |
| `ENABLE_MODEL_REASONING` | `true` | Enables provider-side reasoning/thinking for supported models |
| `MODEL_REASONING_EFFORT` | `medium` | Reasoning effort for OpenAI/OpenAI-compatible models: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `GEMINI_THINKING_BUDGET` | `4096` | Thinking budget for `gemini-2.5*` / `gemini-3*`; older Gemini models are requested without this parameter |
| `PROVIDER_REGISTRY_PATH` | `provider_registry.json` | Registry of OpenAI-compatible aggregators and their reasoning parameters |
| `ACTIVE_MODEL_PROFILE_ID` | - | Active model profile ID for key rotation |
| `SHOW_MODEL_THOUGHTS` | `false` | Legacy reasoning display flag; runtime keeps it false |

### Adding OpenAI-Compatible Aggregators

Detailed instructions for `provider_registry.json`, schema fields, matching rules, and adding new aggregators are available in [`./docs/provider_registry_guide.md`](./docs/provider_registry_guide.md).

### Runtime Controls

| Variable | Description |
|---|---|
| `TEMPERATURE` | Model temperature |
| `TOP_P` | Nucleus sampling, default `0.95`; set `none` or leave empty to skip sending it |
| `TOP_K` | Candidate pool limit for Gemini/supported SDKs, default `40`; not sent to OpenAI-compatible providers |
| `MAX_LOOPS` | Maximum steps for one request, default: 50 |
| `TOOL_LOOP_WINDOW` | History window for duplicate tool-call detection |
| `TOOL_LOOP_LIMIT_MUTATING` | Repeat limit for mutating tools |
| `TOOL_LOOP_LIMIT_READONLY` | Repeat limit for read-only tools |
| `SELF_CORRECTION_RETRY_LIMIT` | Self-correction retry ceiling |

### Feature Flags

| Variable | Description |
|---|---|
| `MODEL_SUPPORTS_TOOLS` | Enable tool calling |
| `ENABLE_TEXT_TOOL_CALL_RECOVERY` | Diagnostic fallback for providers that write `call:...<tool_call|>` as text instead of structured `tool_calls`; disabled by default |
| `ENABLE_FILESYSTEM_TOOLS` | File tools |
| `ENABLE_SHELL_TOOL` | Shell command execution |
| `ENABLE_SEARCH_TOOLS` | Web search through Tavily |
| `ENABLE_SYSTEM_TOOLS` | System information |
| `ENABLE_PROCESS_TOOLS` | Process management |
| `ENABLE_APPROVALS` | Approval pauses before risky actions |
| `ALLOW_EXTERNAL_PROCESS_CONTROL` | Allow control of external processes |
| `TAVILY_API_KEY` | Tavily key for web search |

### Limits

| Variable | Description |
|---|---|
| `MAX_FILE_SIZE` | Maximum file size, supports `300MB` and `4096` |
| `MAX_READ_LINES` | File read line limit |
| `MAX_TOOL_OUTPUT` | Tool output character limit |
| `MAX_SEARCH_CHARS` | Search result character limit |
| `MAX_BACKGROUND_PROCESSES` | Background process limit |
| `STREAM_TEXT_MAX_CHARS` | Streaming text character limit |
| `STREAM_EVENTS_MAX` | Streaming event limit |
| `STREAM_TOOL_BUFFER_MAX` | Streaming tool-output buffer |

### Summarization and Retry

| Variable | Description |
|---|---|
| `SESSION_SIZE` | Token threshold for triggering summarization |
| `SUMMARY_RESERVED_TOKENS` | Reserve for system instructions, tool schemas, and provider overhead |
| `SUMMARY_KEEP_LAST` | Number of latest messages to keep raw during summarization |
| `MAX_RETRIES` | Number of attempts after an LLM error |
| `RETRY_DELAY` | Delay between attempts, in seconds |

### Persistence

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_BACKEND` | `sqlite` | `sqlite` or `memory` |
| `CHECKPOINT_SQLITE_PATH` | `.agent_state/checkpoints.sqlite` | Database path |
| `SESSION_STATE_PATH` | `.agent_state/session.json` | Active session |
| `MODEL_PROFILE_CONFIG_PATH` | `.agent_state/config.json` | Model profiles and active profile file |
| `RUN_LOG_DIR` | `logs/runs` | JSONL run log directory |
| `LOG_FILE` | `logs/agent.log` | Log file |
| `PROMPT_PATH` | `prompt.txt` | System prompt path |
| `MCP_CONFIG_PATH` | `mcp.json` | MCP config path |

### Diagnostics

| Variable | Description |
|---|---|
| `DEBUG` | Enable debug mode |
| `LOG_LEVEL` | Log level: `INFO`, `DEBUG`, `WARNING` |
| `DEBUG_REASONING_STREAM` | Separate detailed reasoning/thinking stream log for provider diagnostics |
| `STRICT_MODE` | Strict mode: no guessing, exact execution |

---

## Project Structure

A detailed module map is available in [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Below is a short working overview.

```text
.
|-- main.py                # GUI entry point
|-- agent.py               # LLM, tools, and LangGraph workflow assembly
|-- prompt.txt             # Main system prompt
|-- prompt_dev.txt         # Additional dev/devops prompt
|-- mcp.json               # MCP server configuration
|-- env_example.txt        # .env template
|-- provider_registry.json # Reasoning kwargs for OpenAI-compatible providers
|-- docs/provider_registry_guide.md
|-- build.bat              # Portable .exe build
|-- requirements.txt
|-- core/                  # Agent core: config, state, policies, recovery, provider registry
|   `-- nodes/             # LangGraph nodes: context, llm, agent, tools, approval, recovery
|-- tools/                 # Filesystem, shell, search, system, process, user input, MCP registry
|   `-- filesystem_impl/   # Low-level filesystem helpers
|-- ui/                    # PySide6 GUI, runtime worker, streaming/status handling
|   |-- window_components/ # Main window, sidebar, inspector, menu, status bar
|   `-- widgets/           # Transcript, composer, messages, tool cards, dialogs, panels
|-- docs/
|   `-- PROJECT_STRUCTURE.md
|-- tests/                 # Runtime, UI, tools, provider registry, logging, policies
|-- .agent_state/          # Local state, profiles, checkpoints
`-- logs/                  # JSONL/runtime/debug logs
```

---

## Tests

20 test files:

| File | Coverage |
|---|---|
| `test_model_fetcher.py` | Model filtering, name normalization, API error codes, fallback logic |
| `test_api_key_rotation.py` | Circular key-pool rotation, exhaustion handling, auth/rate-limit error classification |
| `test_cli_ux.py` | GUI: composer, transcript, tool cards, streaming, sidebar, attachments, history, mentions, approvals |
| `test_stream_and_filesystem.py` | Streaming events, filesystem tools, tool output, cli_exec |
| `test_runtime_refactor.py` | Runtime payloads, transcript restore, tool group logic, run lifecycle |
| `test_critic_graph.py` | LangGraph workflow, node orchestration, tool batching |
| `test_self_correction_engine.py` | Recovery strategies, fingerprinting, loop detection |
| `test_policy_engine.py` | Shell command classification, tool metadata, approval rules |
| `test_model_profiles.py` | Profile CRUD, switching, validation, serialization |
| `test_session_utils.py` | Session ID generation, index management |
| `test_runtime_session_coordination.py` | Session state coordination, load/save |
| `test_tooling_refactor.py` | Tool registry, MCP loading, tool metadata |
| `test_provider_registry.py` | OpenAI-compatible provider matching and reasoning kwargs |
| `test_input_sanitizer.py` | Input sanitization, truncation, control chars |
| `test_logging_config.py` | Log configuration, sensitive data filtering |
| `test_intent_engine.py` | Intent parsing, routing |
| `test_main_window_facade.py` | MainWindow facade behavior |
| `test_ui_helpers.py` | UI helper utilities |
| `test_refactor_services.py` | Service refactoring, internal APIs |
| `test_runtime_payloads.py` | Payload builders, serialization |

Run:

```powershell
venv\Scripts\python.exe -m pytest
```

---

## Dependencies

### Recommended: ripgrep (`rg`)

For more efficient search through files, logs, configuration, and codebases, installing `ripgrep` (`rg`) is recommended. The agent can use it to quickly find files, symbols, lines, and other data instead of reading many files sequentially.

Official repository: [BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep). Prebuilt Windows binaries are available in [Releases](https://github.com/BurntSushi/ripgrep/releases). Usually you need the `x86_64-pc-windows-msvc.zip` archive.

Installation options:

- Add the folder with `rg.exe` to the system `PATH` and verify it with `rg --version`.
- For a portable agent build, copy `rg.exe` next to the agent executable. If the `rg` command is not found automatically, use `.\rg.exe` or add the agent folder to `PATH`.

Example portable folder:

```text
PortableAgent/
|-- Agent.exe
|-- rg.exe
|-- .env
|-- prompt.txt
`-- ...
```

If `rg` is not available, the agent continues to work normally with the standard filesystem tools.

### Python Packages

| Package | Purpose |
|---|---|
| `langgraph` | Agent graph and state management |
| `langchain` | LLM abstraction, tool calling |
| `langchain-google-genai` | Gemini provider |
| `langchain-openai` | OpenAI / compatible provider |
| `langchain-mcp-adapters` | MCP integration |
| `PySide6` | GUI |
| `pydantic-settings` | Configuration through `.env` |
| `tiktoken` | Token counting for summarization |
| `tavily-python` | Web search |
| `psutil` | System tools and processes |
| `httpx` | HTTP for MCP and fetch |
| `aiofiles` | Async file operations |
| `mcp` | Model Context Protocol |
| `requests` | HTTP client for Google API and Tavily |
| `sqlite-vec` | Vector extension for SQLite checkpoints |
