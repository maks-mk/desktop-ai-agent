from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import platform
from typing import Iterable, List, Sequence

from langchain_core.messages import SystemMessage

from core.config import AgentConfig


@dataclass(frozen=True)
class RuntimePromptContext:
    current_task: str
    tools_available: bool
    active_tool_names: Sequence[str]
    user_choice_locked: bool = False


@dataclass(frozen=True)
class RuntimeExecutionEnvironment:
    os_family: str
    shell_family: str
    path_style: str
    workspace_root: str
    current_working_directory: str
    timezone_name: str
    utc_offset: str


class RuntimePromptPolicyBuilder:
    REQUEST_USER_INPUT_TOOL_NAME = "request_user_input"
    REQUEST_USER_INPUT_DEMO_MARKERS = (
        "request_user_input",
        "test",
        "demo",
        "demonstrat",
        "example",
        "тест",
        "демо",
        "пример",
    )

    def __init__(self, *, config: AgentConfig) -> None:
        self.config = config

    def build_messages(self, context: RuntimePromptContext) -> List[SystemMessage]:
        messages: List[SystemMessage] = [
            SystemMessage(content=self._build_runtime_contract(context)),
        ]

        strict_mode_message = self._build_strict_mode_message()
        if strict_mode_message:
            messages.append(SystemMessage(content=strict_mode_message))

        tool_access_message = self._build_tool_access_message(context)
        if tool_access_message:
            messages.append(SystemMessage(content=tool_access_message))

        request_user_input_message = self._build_request_user_input_policy(context)
        if request_user_input_message:
            messages.append(SystemMessage(content=request_user_input_message))

        demo_policy_message = self._build_request_user_input_demo_policy(context)
        if demo_policy_message:
            messages.append(SystemMessage(content=demo_policy_message))

        if context.user_choice_locked:
            messages.append(
                SystemMessage(
                    content=(
                        "USER CHOICE ALREADY COLLECTED IN THIS TURN.\n"
                        "Do not call request_user_input again.\n"
                        "Use the selected value from the latest request_user_input tool result and continue."
                    )
                )
            )

        return messages

    def _build_runtime_contract(self, context: RuntimePromptContext) -> str:
        environment = self._detect_execution_environment()
        lines = [
            "RUNTIME CONTRACT:",
            "You operate in a terminal (CLI) environment without GUI.",
            self._build_execution_environment_line(environment),
            f"Workspace root: {environment.workspace_root}",
            f"Current working directory: {environment.current_working_directory}",
            f"Local timezone: {environment.timezone_name} ({environment.utc_offset})",
            "Always respond in Russian.",
            (
                "Execute user requests exactly as stated. Do not add, assume, or infer extra work beyond the explicit "
                "request."
            ),
            (
                "Ask for clarification only when the request cannot be executed correctly and safely with the available "
                "context and tools."
            ),
            "After any system change (installation, configuration, file edits), verify the result before reporting success.",
            "Be concise. No preamble, no summaries after actions, no unsolicited explanations.",
            f"Current date: {datetime.now().strftime('%Y-%m-%d')}",
        ]
        return "\n".join(lines)

    def _build_execution_environment_line(self, environment: RuntimeExecutionEnvironment | None = None) -> str:
        environment = environment or self._detect_execution_environment()
        return (
            "Execution environment: "
            f"os={environment.os_family}; "
            f"shell={environment.shell_family}; "
            f"paths={environment.path_style}."
        )

    def _build_strict_mode_message(self) -> str:
        if not self.config.strict_mode:
            return ""
        return "STRICT MODE: Be precise. No guessing."

    def _build_tool_access_message(self, context: RuntimePromptContext) -> str:
        if not context.tools_available:
            return (
                "TOOL ACCESS FOR THIS REQUEST:\n"
                "Tools are disabled for this request.\n"
                "ACTIVE_TOOLS_FOR_THIS_REQUEST: none. Do not claim any tool is available."
            )

        names = self._normalized_tool_names(context.active_tool_names)
        if not names:
            return (
                "TOOL ACCESS FOR THIS REQUEST:\n"
                "Use only tools bound in this request."
            )
        if len(names) <= 4:
            return (
                "TOOL ACCESS FOR THIS REQUEST:\n"
                "ACTIVE_TOOLS_FOR_THIS_REQUEST: "
                + ", ".join(names)
                + ". Do not claim other tools are available."
            )
        return (
            "TOOL ACCESS FOR THIS REQUEST:\n"
            "ACTIVE_TOOLS_FOR_THIS_REQUEST: use only tools bound in this request."
        )

    def _build_request_user_input_policy(self, context: RuntimePromptContext) -> str:
        if self.REQUEST_USER_INPUT_TOOL_NAME not in self._normalized_tool_names(context.active_tool_names):
            return ""
        return (
            "REQUEST_USER_INPUT POLICY:\n"
            "Call request_user_input only for a real blocking decision or missing external input that cannot be resolved "
            "from context or tools.\n"
            "Ask exactly one user-choice question per turn. Never batch multiple request_user_input calls.\n"
            "After resume, continue with the chosen answer instead of asking again."
        )

    def _build_request_user_input_demo_policy(self, context: RuntimePromptContext) -> str:
        if self.REQUEST_USER_INPUT_TOOL_NAME not in self._normalized_tool_names(context.active_tool_names):
            return ""
        if not self._looks_like_request_user_input_demo(context.current_task):
            return ""
        return (
            "REQUEST_USER_INPUT TEST POLICY:\n"
            "The current task is a test or demonstration of request_user_input.\n"
            "Do exactly one request_user_input call.\n"
            "After resume, reply briefly with the selected value.\n"
            "Do not add file writes, shell commands, extra verification, or more questions unless the user explicitly asked for them."
        )

    def _looks_like_request_user_input_demo(self, current_task: str) -> bool:
        text = str(current_task or "").casefold()
        if self.REQUEST_USER_INPUT_TOOL_NAME not in text:
            return False
        return any(marker in text for marker in self.REQUEST_USER_INPUT_DEMO_MARKERS[1:])

    def _detect_execution_environment(self) -> RuntimeExecutionEnvironment:
        os_family = self._detect_os_family()
        workspace_root = str(Path.cwd().resolve())
        now = datetime.now().astimezone()
        return RuntimeExecutionEnvironment(
            os_family=os_family,
            shell_family=self._detect_shell_family(os_family=os_family),
            path_style="windows" if os_family == "windows" else "unix" if os_family in {"linux", "mac"} else "unknown",
            workspace_root=workspace_root,
            current_working_directory=workspace_root,
            timezone_name=self._detect_timezone_name(now),
            utc_offset=self._format_utc_offset(now),
        )

    @staticmethod
    def _detect_os_family() -> str:
        raw_name = platform.system().strip().casefold()
        if raw_name == "windows":
            return "windows"
        if raw_name == "linux":
            return "linux"
        if raw_name == "darwin":
            return "mac"
        return "unknown"

    @staticmethod
    def _detect_shell_family(*, os_family: str) -> str:
        shell_candidates = [
            os.environ.get("SHELL", ""),
            os.environ.get("COMSPEC", ""),
            os.environ.get("TERM_SHELL", ""),
        ]
        if "PSModulePath" in os.environ:
            shell_candidates.insert(0, "powershell")

        for candidate in shell_candidates:
            normalized = str(candidate or "").replace("\\", "/").casefold().strip()
            if not normalized:
                continue
            if "pwsh" in normalized or "powershell" in normalized:
                return "powershell"
            if normalized.endswith("cmd.exe") or normalized.endswith("/cmd") or normalized == "cmd":
                return "cmd"
            if normalized.endswith("/bash") or normalized == "bash":
                return "bash"
            if normalized.endswith("/zsh") or normalized == "zsh":
                return "zsh"
            if normalized.endswith("/fish") or normalized == "fish":
                return "fish"
            if normalized.endswith("/sh") or normalized == "sh":
                return "sh"

        if os_family in {"linux", "mac"}:
            return "sh"
        return "unknown"

    @staticmethod
    def _detect_timezone_name(now: datetime) -> str:
        zone_key = str(getattr(now.tzinfo, "key", "") or "").strip()
        if zone_key:
            return zone_key
        tz_name = str(now.tzname() or "").strip()
        if tz_name and all(char.isascii() and (char.isalnum() or char in "/_+-:") for char in tz_name):
            return tz_name
        return RuntimePromptPolicyBuilder._format_utc_offset(now)

    @staticmethod
    def _format_utc_offset(now: datetime) -> str:
        offset = now.utcoffset()
        if offset is None:
            return "UTC?"
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        absolute_minutes = abs(total_minutes)
        hours, minutes = divmod(absolute_minutes, 60)
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    @staticmethod
    def _normalized_tool_names(tool_names: Iterable[str]) -> List[str]:
        return [str(name).strip() for name in tool_names if str(name).strip()]
