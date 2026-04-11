import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Iterator, Optional
from langchain_core.tools import tool

from core.utils import truncate_output
from core.errors import format_error, ErrorType
from core.safety_policy import SafetyPolicy
from core.policy_engine import classify_shell_command

# Константы
DEFAULT_TIMEOUT = 120

# Глобальные настройки
_SAFETY_POLICY: Optional[SafetyPolicy] = None
_WORKING_DIRECTORY: str = os.getcwd()  # По умолчанию текущая папка процесса
_CLI_OUTPUT_EMITTER: Optional[Callable[[dict[str, str]], None]] = None
_CLI_TOOL_ID: ContextVar[str] = ContextVar("cli_tool_id", default="")

_WINDOWS_COMMAND_HINTS = {
    "cat": "type <file> (или Get-Content <file>)",
    "ls": "dir (или Get-ChildItem)",
    "pwd": "cd (или Get-Location)",
    "cp": "copy (или Copy-Item)",
    "mv": "move (или Move-Item)",
    "rm": "del (или Remove-Item)",
    "grep": "findstr <pattern> <file> (или Select-String ...)",
    "head": "Get-Content <file> -TotalCount N",
    "tail": "Get-Content <file> -Tail N",
    "which": "where <command> (или Get-Command <name>)",
    "clear": "cls (или Clear-Host)",
}

_WINDOWS_PYTHON_HEREDOC_RE = re.compile(
    r"^\s*(?P<exe>python(?:3(?:\.\d+)?)?)\s*-\s*<<\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\r?\n(?P<body>[\s\S]*?)\r?\n(?P=tag)\s*$",
    re.IGNORECASE,
)
_WINDOWS_POWERSHELL_WRAPPER_RE = re.compile(
    r"^\s*(?:pwsh|powershell(?:\.exe)?)\s+(?:(?:-(?:NoProfile|NoLogo|NonInteractive))\s+)*-Command\s+(?P<body>[\s\S]+?)\s*$",
    re.IGNORECASE,
)
_WINDOWS_NULL_OUTPUT_FLAG_RE = re.compile(
    r"(?i)(?P<flag>--output|-o)\s+(?P<quote>['\"]?)/dev/null(?P=quote)"
)
_WINDOWS_NULL_OUTPUT_FLAG_EQ_RE = re.compile(
    r"(?i)(?P<flag>--output=)(?P<quote>['\"]?)/dev/null(?P=quote)"
)
_WINDOWS_NULL_REDIRECT_RE = re.compile(
    r"(?i)(?P<redir>\d?>)\s*(?P<quote>['\"]?)/dev/null(?P=quote)"
)
_NPM_LIKE_COMMAND_RE = re.compile(r"(^|[;&|()\s])(?:npm|npx)(?=$|[;&|()\s])", re.IGNORECASE)
_INTERACTIVE_PROMPT_PATTERNS = (
    re.compile(r"ok to proceed\?\s*\(y\)", re.IGNORECASE),
    re.compile(r"do you want to continue\?\s*\[(?:y|yes)/(?:n|no)\]", re.IGNORECASE),
    re.compile(r"\[(?:y|yes)/(?:n|no)\]", re.IGNORECASE),
    re.compile(r"\[(?:n|no)/(?:y|yes)\]", re.IGNORECASE),
    re.compile(r"press any key", re.IGNORECASE),
    re.compile(r"hit ctrl-c to stop", re.IGNORECASE),
)
_INSPECT_ONLY_COMMAND_PATTERNS = (
    re.compile(r"\bget-process\b", re.IGNORECASE),
    re.compile(r"\btasklist\b", re.IGNORECASE),
    re.compile(r"\bwhere-object\b", re.IGNORECASE),
    re.compile(r"\bselect-object\b", re.IGNORECASE),
    re.compile(r"\bfindstr\b", re.IGNORECASE),
    re.compile(r"\bget-childitem\b", re.IGNORECASE),
    re.compile(r"\bget-content\b", re.IGNORECASE),
    re.compile(r"\bselect-string\b", re.IGNORECASE),
    re.compile(r"\bdir\b", re.IGNORECASE),
    re.compile(r"\btype\b", re.IGNORECASE),
    re.compile(r"\bwhere\b", re.IGNORECASE),
    re.compile(r"\bnetstat\b", re.IGNORECASE),
    re.compile(r"\bss\b", re.IGNORECASE),
    re.compile(r"\bps\b", re.IGNORECASE),
)
_MUTATING_COMMAND_PATTERNS = (
    re.compile(r"\btaskkill\b", re.IGNORECASE),
    re.compile(r"\bstop-process\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\bdel\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
    re.compile(r"\bmove-item\b", re.IGNORECASE),
    re.compile(r"\brename-item\b", re.IGNORECASE),
    re.compile(r"\bcopy-item\b", re.IGNORECASE),
    re.compile(r"\bset-content\b", re.IGNORECASE),
    re.compile(r"\badd-content\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+uninstall\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\b", re.IGNORECASE),
)
_DESTRUCTIVE_COMMAND_PATTERNS = (
    re.compile(r"\btaskkill\b", re.IGNORECASE),
    re.compile(r"\bstop-process\b", re.IGNORECASE),
    re.compile(r"\bremove-item\b", re.IGNORECASE),
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\bdel\b", re.IGNORECASE),
    re.compile(r"\brmdir\b", re.IGNORECASE),
)
_LONG_RUNNING_SERVICE_PATTERNS = (
    re.compile(r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+http\.server\b", re.IGNORECASE),
    re.compile(r"\bhttp-server\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+start\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+run\s+dev\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+exec\b.*\bhttp-server\b", re.IGNORECASE),
    re.compile(r"\bnpx\b.*\bhttp-server\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bflask\s+run\b", re.IGNORECASE),
    re.compile(r"\bwebpack(?:\.cmd)?\s+serve\b", re.IGNORECASE),
    re.compile(r"\bserve\b", re.IGNORECASE),
)


def _get_windows_command_hint(command: str, stderr: str) -> str:
    """Return a friendly Windows-specific hint for common Unix commands."""
    if os.name != "nt":
        return ""

    lower_stderr = stderr.lower()
    if "is not recognized as an internal or external command" not in lower_stderr:
        return ""

    parts = command.strip().split()
    if not parts:
        return ""

    first_token = parts[0].strip("\"'").lower()
    suggestion = _WINDOWS_COMMAND_HINTS.get(first_token)
    if not suggestion:
        return ""
    return f"\nHint (Windows): команда '{first_token}' не найдена. Попробуйте: {suggestion}."


def _normalize_windows_python_heredoc(command: str) -> str:
    """Converts bash-style python heredoc into a PowerShell-compatible command body on Windows."""
    if os.name != "nt":
        return command

    match = _WINDOWS_PYTHON_HEREDOC_RE.match(command.strip())
    if not match:
        return command

    exe = match.group("exe")
    body = match.group("body").replace("\r\n", "\n")
    # PowerShell single-quoted here-string terminator must be at line start.
    # Indent-breaking sequences are extremely unlikely for generated scripts; if present,
    # fallback to original command so the model sees a direct error and can correct manually.
    if "\n'@\n" in f"\n{body}\n":
        return command

    return f"@'\n{body}\n'@ | {exe} -"


def _strip_nested_windows_powershell_wrapper(command: str) -> str:
    """
    cli_exec already runs PowerShell on Windows.
    If the model wraps the command in `powershell -Command ...`, unwrap it so
    `$vars` are not expanded by an outer shell layer.
    """
    if os.name != "nt":
        return command
    raw = str(command or "")
    match = _WINDOWS_POWERSHELL_WRAPPER_RE.match(raw.strip())
    if not match:
        return command
    body = str(match.group("body") or "").strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in {'"', "'"}:
        body = body[1:-1]
    return body or command


def _normalize_windows_null_device(command: str) -> str:
    """Maps POSIX null sink paths to Windows `NUL` for common CLI patterns."""
    if os.name != "nt":
        return command

    raw = str(command or "")
    if "/dev/null" not in raw.lower():
        return raw

    normalized = _WINDOWS_NULL_OUTPUT_FLAG_RE.sub(
        lambda match: f"{match.group('flag')} {match.group('quote')}NUL{match.group('quote')}",
        raw,
    )
    normalized = _WINDOWS_NULL_OUTPUT_FLAG_EQ_RE.sub(
        lambda match: f"{match.group('flag')}{match.group('quote')}NUL{match.group('quote')}",
        normalized,
    )
    normalized = _WINDOWS_NULL_REDIRECT_RE.sub(
        lambda match: f"{match.group('redir')} {match.group('quote')}NUL{match.group('quote')}",
        normalized,
    )
    return normalized


def _prepare_shell_env(command: str) -> dict[str, str]:
    env = os.environ.copy()
    # Best-effort non-interactive mode for CLI tools that support CI semantics.
    env.setdefault("CI", "1")

    if _NPM_LIKE_COMMAND_RE.search(command or ""):
        # Prevent npm/npx confirmation prompts such as "Ok to proceed? (y)".
        env.setdefault("npm_config_yes", "true")
        env.setdefault("npm_config_audit", "false")
        env.setdefault("npm_config_fund", "false")
    return env


def _detect_interactive_prompt(text: str) -> str | None:
    sample = str(text or "")
    if not sample:
        return None
    for pattern in _INTERACTIVE_PROMPT_PATTERNS:
        match = pattern.search(sample)
        if match:
            return match.group(0)
    return None


def classify_cli_command(command: str) -> dict[str, Any]:
    # Shared classification policy for both routing and execution layers.
    return classify_shell_command(command)


def _powershell_executable() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"


def _windows_subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not create_no_window:
        return {}
    return {"creationflags": create_no_window}


async def _terminate_process_tree(process: Any) -> None:
    pid = getattr(process, "pid", None)
    if os.name == "nt" and pid:
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                **_windows_subprocess_kwargs(),
            )
            await asyncio.wait_for(killer.wait(), timeout=3)
        except Exception:
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            process.kill()
        except OSError:
            pass

    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except Exception:
        return


def set_safety_policy(policy: SafetyPolicy):
    """Sets the global safety policy for shell execution."""
    global _SAFETY_POLICY
    _SAFETY_POLICY = policy

def set_working_directory(cwd: str):
    """
    Syncs the shell's working directory with the FilesystemManager's workspace.
    Call this when initializing the agent to ensure tools look at the same folders.
    """
    global _WORKING_DIRECTORY
    _WORKING_DIRECTORY = cwd


def set_cli_output_emitter(emitter: Optional[Callable[[dict[str, str]], None]]) -> None:
    """Registers callback that receives streaming CLI chunks for UI rendering."""
    global _CLI_OUTPUT_EMITTER
    _CLI_OUTPUT_EMITTER = emitter


@contextmanager
def cli_output_context(tool_id: str) -> Iterator[None]:
    """Binds current tool call id so streaming chunks can be routed to the right widget."""
    token = _CLI_TOOL_ID.set(str(tool_id or "").strip())
    try:
        yield
    finally:
        _CLI_TOOL_ID.reset(token)


def _emit_cli_output(data: str, stream: str) -> None:
    if not data:
        return
    emitter = _CLI_OUTPUT_EMITTER
    tool_id = _CLI_TOOL_ID.get()
    if emitter is None or not tool_id:
        return
    try:
        emitter({"tool_id": tool_id, "data": data, "stream": stream})
    except Exception:
        # Streaming is best-effort and must never fail tool execution.
        return

@tool("cli_exec")
async def cli_exec(command: str) -> str:
    """
    Executes a shell command on the host machine.
    
    IMPORTANT RULES FOR LLM:
    1. STATELESSNESS: Commands are stateless. `cd folder` in one call will NOT affect the next call. 
       If you need to change directories, chain commands: e.g., `cd folder && npm install`.
    2. NO INTERACTIVE COMMANDS: DO NOT run commands that require user input (e.g., `nano`, `vim`, `python` without args, `less`, `top`). 
       They will hang until timeout!
    3. BACKGROUND TASKS: Never use cli_exec to start background processes or services.
       Background runs (dev server, watcher, daemon) are not allowed in cli_exec.
    4. LONG SCRIPTS: For complex logic, write a script file using `write_file` and then execute it.
    
    Supports pipe (|), redirects (>), and chain operators (&&).
    
    Args:
        command: The shell command to execute (e.g., 'ls -la', 'git status').
    """
    if _SAFETY_POLICY and not _SAFETY_POLICY.allow_shell:
        return format_error(ErrorType.ACCESS_DENIED, "Shell execution is disabled by SafetyPolicy.")

    if not command.strip():
        return format_error(ErrorType.VALIDATION, "Command cannot be empty.")

    normalized_command = _normalize_windows_python_heredoc(command)
    normalized_command = _strip_nested_windows_powershell_wrapper(normalized_command)
    normalized_command = _normalize_windows_null_device(normalized_command)
    command_profile = classify_cli_command(normalized_command)
    if command_profile["long_running_service"]:
        return format_error(
            ErrorType.VALIDATION,
            "Foreground service/server commands are not supported in cli_exec. Use run_background_process instead.",
        )
    command_env = _prepare_shell_env(normalized_command)

    try:
        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                _powershell_executable(),
                "-NoProfile",
                "-Command",
                normalized_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=_WORKING_DIRECTORY,
                env=command_env,
                **_windows_subprocess_kwargs(),
            )
        else:
            process = await asyncio.create_subprocess_shell(
                normalized_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=_WORKING_DIRECTORY,
                env=command_env,
            )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        chunk_queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        interactive_prompt: str = ""
        interactive_prompt_sample = ""

        async def _read_stream(stream_name: str, reader: asyncio.StreamReader | None) -> None:
            if reader is None:
                await chunk_queue.put((stream_name, None))
                return
            try:
                while True:
                    chunk = await reader.read(1024)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    if not text:
                        continue
                    await chunk_queue.put((stream_name, text))
            finally:
                await chunk_queue.put((stream_name, None))

        async def _collect_stream_output() -> None:
            nonlocal interactive_prompt, interactive_prompt_sample
            completed_readers = 0
            prompt_window = ""
            while completed_readers < 2:
                stream_name, chunk = await chunk_queue.get()
                if chunk is None:
                    completed_readers += 1
                    continue
                if stream_name == "stdout":
                    stdout_chunks.append(chunk)
                else:
                    stderr_chunks.append(chunk)
                _emit_cli_output(chunk, stream_name)

                if not interactive_prompt:
                    prompt_window = (prompt_window + chunk)[-1200:]
                    detected_prompt = _detect_interactive_prompt(prompt_window)
                    if detected_prompt:
                        interactive_prompt = detected_prompt
                        interactive_prompt_sample = prompt_window[-320:].strip()
                        await _terminate_process_tree(process)

        stdout_reader_task = asyncio.create_task(_read_stream("stdout", process.stdout))
        stderr_reader_task = asyncio.create_task(_read_stream("stderr", process.stderr))
        collector_task = asyncio.create_task(_collect_stream_output())

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=DEFAULT_TIMEOUT)
        except asyncio.TimeoutError:
            timed_out = True
            await _terminate_process_tree(process)
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            raise
        finally:
            await asyncio.gather(stdout_reader_task, stderr_reader_task, return_exceptions=True)
            await collector_task

        stdout = "".join(stdout_chunks).strip()
        stderr = "".join(stderr_chunks).strip()

        output_parts =[]
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")

        output = "\n".join(output_parts)

        if interactive_prompt:
            details = f"\nOutput tail:\n{interactive_prompt_sample}" if interactive_prompt_sample else ""
            return format_error(
                ErrorType.EXECUTION,
                "Interactive prompt detected in cli_exec output "
                f"('{interactive_prompt}'). Run command in non-interactive mode "
                "(for npm/npx add -y/--yes)."
                f"{details}",
            )

        if timed_out:
            details = f"\nPartial output:\n{output}" if output else ""
            return format_error(
                ErrorType.TIMEOUT,
                f"Command timed out after {DEFAULT_TIMEOUT} seconds. Did you run an interactive command (like nano/vim) or a blocking server?{details}"
            )
        
        if process.returncode != 0:
            error_msg = f"Command failed with Exit Code {process.returncode}."
            cmd_hint = _get_windows_command_hint(command, stderr)
            if os.name == "nt" and "<< was unexpected at this time." in stderr:
                cmd_hint += (
                    "\nHint (Windows): bash-style heredoc (`python - <<'PY'`) не поддерживается через cmd.exe. "
                    "Используйте PowerShell here-string: @' ... '@ | python -"
                )
            if output:
                error_msg += f"\nOutput:\n{output}"
            else:
                error_msg += " (No output)"
            if cmd_hint:
                error_msg += cmd_hint
            return format_error(ErrorType.EXECUTION, error_msg)

        if not output:
            output = "Command executed successfully (no output)."
        
        limit = _SAFETY_POLICY.max_tool_output if _SAFETY_POLICY else 5000
        return truncate_output(output, limit, source="shell")

    except Exception as e:
        return format_error(ErrorType.EXECUTION, f"Error executing command: {str(e)}")
