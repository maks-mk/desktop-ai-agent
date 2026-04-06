import re
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from langchain_core.messages import AIMessage, AIMessageChunk

_THOUGHT_RE = re.compile(r"<(thought|think)>(.*?)</\1>", re.DOTALL)
_CLEAN_MD_RE = re.compile(r"\n{3,}")
_CRAWL_PAGES_RE = re.compile(r"(\d+) pages processed")
_CRAWL_DEPTH_RE = re.compile(r"max_depth: (\d+)")
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_FILE_EXT_RE = re.compile(r"\.([a-z0-9]+)\b", re.IGNORECASE)
_SIMPLE_LATEX_INLINE_RE = re.compile(r"\$\s*(\\[A-Za-z]+)\s*\$")

# Matches Markdown links whose href is a local file path (not http/https/ftp/mailto).
# Rich URL-encodes non-ASCII hrefs, which breaks Cyrillic filenames in output.
# Capture groups: 1=link text, 2=href
_LOCAL_LINK_RE = re.compile(
    r"\[([^\]]+)\]\((?!https?://|ftp://|mailto:)([^)]+)\)",
    re.IGNORECASE,
)
_SIMPLE_LATEX_SYMBOLS = {
    r"\to": "→",
    r"\rightarrow": "→",
    r"\gets": "←",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\le": "≤",
    r"\leq": "≤",
    r"\neq": "≠",
    r"\pm": "±",
    r"\times": "×",
    r"\cdot": "·",
    r"\approx": "≈",
    r"\infty": "∞",
}


def _rewrite_outside_code(text: str, replacer: Callable[[str], str]) -> str:
    parts: list[str] = []
    last = 0
    for block in _FENCED_BLOCK_RE.finditer(text):
        parts.append(_rewrite_outside_inline_code(text[last:block.start()], replacer))
        parts.append(block.group(0))
        last = block.end()
    parts.append(_rewrite_outside_inline_code(text[last:], replacer))
    return "".join(parts)


def _rewrite_outside_inline_code(text: str, replacer: Callable[[str], str]) -> str:
    parts: list[str] = []
    last = 0
    for block in _INLINE_CODE_RE.finditer(text):
        parts.append(replacer(text[last:block.start()]))
        parts.append(block.group(0))
        last = block.end()
    parts.append(replacer(text[last:]))
    return "".join(parts)


def _normalize_simple_latex_inline(text: str) -> str:
    def _replace_segment(segment: str) -> str:
        def _replace_match(match: re.Match) -> str:
            command = str(match.group(1) or "").strip()
            return _SIMPLE_LATEX_SYMBOLS.get(command, match.group(0))

        return _SIMPLE_LATEX_INLINE_RE.sub(_replace_match, segment)

    return _rewrite_outside_code(text, _replace_segment)


def _rewrite_local_file_links(text: str) -> str:
    """Convert Markdown local-file links to inline code to prevent Rich URL-encoding.

    [имя.md](имя.md)          →  `имя.md`
    [label](path/to/file.py)  →  `path/to/file.py` (label)
    """
    def _replace(m: re.Match) -> str:
        label: str = m.group(1).strip()
        href: str = m.group(2).strip()
        # If the href itself is the label (auto-link), just emit inline code
        if label == href or label.lower() == href.lower():
            return f"`{href}`"
        # Otherwise emit: `href` (label)
        return f"`{href}` ({label})"

    return _rewrite_outside_code(text, lambda segment: _LOCAL_LINK_RE.sub(_replace, segment))

_LANG_BY_EXTENSION = {
    "py": "python",
    "js": "javascript",
    "jsx": "jsx",
    "ts": "typescript",
    "tsx": "tsx",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "cs": "csharp",
    "cpp": "cpp",
    "c": "c",
    "h": "c",
    "hpp": "cpp",
    "json": "json",
    "yml": "yaml",
    "yaml": "yaml",
    "sh": "bash",
    "bash": "bash",
    "ps1": "powershell",
    "sql": "sql",
    "html": "html",
    "css": "css",
    "md": "markdown",
}
_CODE_PREFIXES = (
    "package ", "import ", "func ", "def ", "class ", "const ", "let ", "var ",
    "if ", "elif ", "else:", "for ", "while ", "return ", "public ", "private ",
    "protected ", "interface ", "type ", "struct ", "fn ", "using ", "namespace ",
    "#include", "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ",
)
_CODE_TOKENS = (
    " := ", " => ", " == ", " != ", ".Println(", "fmt.", "console.", "System.out",
    "->", "::", "();", "{", "}", "[", "]",
)


def truncate_value(value: str, max_length: int = 60) -> str:
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def _single_line_preview(value: Any) -> str:
    text = str(value or "")
    # Tool card headers should stay one-line even when command/query contains newlines.
    return " ".join(text.split())


def abbreviate_path(path_str: str, max_length: int = 60) -> str:
    try:
        path = Path(path_str)
        if len(path.parts) == 1:
            return path_str

        try:
            rel_str = str(path.relative_to(Path.cwd()))
            if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                return rel_str
        except (ValueError, OSError):
            pass

        if len(path_str) <= max_length:
            return path_str
    except Exception:
        pass

    return truncate_value(path_str, max_length)


def _format_path_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    path_value = tool_args.get("file_path") or tool_args.get("path")
    if path_value:
        return f"{tool_name}({abbreviate_path(str(path_value))})"
    return None


def _format_query_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    query = tool_args.get("query")
    if query is not None:
        return f'{tool_name}("{truncate_value(_single_line_preview(query), 80)}")'
    return None


def _format_pattern_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    pattern_val = tool_args.get("pattern") or tool_args.get("name_pattern")
    if pattern_val is not None:
        return f'{tool_name}("{truncate_value(_single_line_preview(pattern_val), 70)}")'
    return None


def _format_command_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    command = tool_args.get("command")
    if command is not None:
        return f'{tool_name}("{truncate_value(_single_line_preview(command), 100)}")'
    return None


def _format_list_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    path = tool_args.get("path")
    return f"{tool_name}({abbreviate_path(str(path))})" if path else f"{tool_name}()"


def _format_url_tool(tool_name: str, tool_args: Dict[str, Any]) -> str | None:
    url_val = tool_args.get("url") or tool_args.get("urls")
    if url_val:
        return f'{tool_name}("{truncate_value(_single_line_preview(url_val), 80)}")'
    return None


DISPLAY_RULES: tuple[tuple[set[str], Callable[[str, Dict[str, Any]], str | None]], ...] = (
    ({"read_file", "write_file", "edit_file", "Read", "Write", "SearchReplace", "tail_file"}, _format_path_tool),
    ({"web_search", "WebSearch"}, _format_query_tool),
    ({"grep", "Grep", "glob", "Glob", "search_in_file", "search_in_directory", "find_file"}, _format_pattern_tool),
    ({"execute", "RunCommand", "cli_exec"}, _format_command_tool),
    ({"ls", "LS", "list_directory"}, _format_list_tool),
    ({"fetch_url", "WebFetch", "fetch_content", "download_file"}, _format_url_tool),
)


def format_tool_display(tool_name: str, tool_args: Dict[str, Any]) -> str:
    for names, formatter in DISPLAY_RULES:
        if tool_name in names:
            formatted = formatter(tool_name, tool_args)
            if formatted:
                return formatted
            break

    args_str = ", ".join(f"{k}={truncate_value(_single_line_preview(v), 50)}" for k, v in tool_args.items())
    return f"{tool_name}({args_str})"


def _collapse_non_code_markdown(text: str) -> str:
    parts = []
    last_index = 0
    for match in _FENCED_BLOCK_RE.finditer(text):
        parts.append(_CLEAN_MD_RE.sub("\n\n", text[last_index:match.start()]))
        parts.append(match.group(0))
        last_index = match.end()
    parts.append(_CLEAN_MD_RE.sub("\n\n", text[last_index:]))
    return "".join(parts)


def clean_markdown_text(text: str) -> str:
    if not text:
        return text
    return _collapse_non_code_markdown(text)


def parse_thought(text: str) -> Tuple[str, str, bool]:
    clean_text = _THOUGHT_RE.sub("", text)

    for tag in ("<thought>", "<think>"):
        start_idx = clean_text.find(tag)
        if start_idx != -1:
            close_tag = tag.replace("<", "</")
            if close_tag not in clean_text:
                content_start = start_idx + len(tag)
                return clean_text[content_start:].strip(), clean_text[:start_idx], True

    return "", clean_text.strip(), False


def _code_signal_score(line: str) -> int:
    stripped = line.strip()
    if not stripped:
        return 0
    if stripped.startswith(("```", "- ", "* ", "> ")) or re.match(r"^\d+\.\s", stripped):
        return 0

    score = 0
    if any(stripped.startswith(prefix) for prefix in _CODE_PREFIXES):
        score += 2
    if stripped in {"{", "}", "};"}:
        score += 2
    if stripped.startswith(("//", "/*", "*/", "#include")):
        score += 1
    if any(token in stripped for token in _CODE_TOKENS):
        score += 1
    if stripped.endswith(("{", "}", ";", ")")):
        score += 1
    if re.match(r"^[A-Za-z_][\w.]*\([^)]*\)\s*\{?$", stripped):
        score += 1
    if re.match(r"^[A-Za-z_][\w<>\[\]]+\s+[A-Za-z_][\w<>\[\]]*\s*=", stripped):
        score += 1
    return score


def _guess_code_language(context_line: str, code_lines: list[str]) -> str:
    context = f"{context_line}\n" + "\n".join(code_lines[:3])
    ext_match = _FILE_EXT_RE.search(context)
    if ext_match:
        language = _LANG_BY_EXTENSION.get(ext_match.group(1).lower())
        if language:
            return language

    joined = "\n".join(code_lines[:5]).strip()
    if joined.startswith("package ") or 'fmt."' in joined or ".Println(" in joined:
        return "go"
    if joined.startswith(("def ", "class ", "import ")) and ":" in joined:
        return "python"
    if joined.startswith(("const ", "let ", "function ")) or "console." in joined:
        return "javascript"
    if joined.startswith(("fn ", "let ")) and "println!" in joined:
        return "rust"
    if joined.startswith(("SELECT ", "INSERT ", "UPDATE ", "DELETE ")):
        return "sql"
    return ""


def normalize_markdown_code_blocks(text: str) -> str:
    if not text:
        return text

    lines = text.splitlines()
    output: list[str] = []
    in_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            output.append(line)
            index += 1
            continue

        if in_fence:
            output.append(line)
            index += 1
            continue

        block: list[str] = []
        block_scores = 0
        code_line_count = 0
        scan = index

        while scan < len(lines):
            current = lines[scan]
            current_stripped = current.strip()
            if current_stripped.startswith("```"):
                break
            if not current_stripped:
                if not block:
                    break
                lookahead = scan + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead >= len(lines) or _code_signal_score(lines[lookahead]) == 0:
                    break
                block.append(current)
                scan += 1
                continue

            score = _code_signal_score(current)
            if score == 0:
                break

            block.append(current)
            block_scores += score
            code_line_count += 1
            scan += 1

        if code_line_count >= 2 and block_scores >= 4:
            context_line = next((prev for prev in reversed(output) if prev.strip() and not prev.strip().startswith("```")), "")
            language = _guess_code_language(context_line, [entry for entry in block if entry.strip()])
            if output and output[-1].strip():
                output.append("")
            output.append(f"```{language}" if language else "```")
            output.extend(block)
            output.append("```")
            if scan < len(lines) and lines[scan].strip():
                output.append("")
            index = scan
            continue

        output.append(line)
        index += 1

    return "\n".join(output)


def prepare_markdown_for_render(text: str) -> str:
    text = _normalize_simple_latex_inline(text)
    text = _rewrite_local_file_links(text)
    return normalize_markdown_code_blocks(clean_markdown_text(text))


def _hint_for_error(content: str) -> str:
    lower_content = content.lower()
    if "401" in lower_content or "unauthorized" in lower_content:
        return " (Hint: Check your API keys in .env)"
    if "not found" in lower_content and ("file" in lower_content or "dir" in lower_content):
        return " (Hint: Check path relative to workspace)"
    if "disabled" in lower_content:
        return " (Hint: Check .env configuration)"
    if "connection" in lower_content or "timeout" in lower_content:
        return " (Hint: Network issue, try again)"
    return ""


def _format_web_search_output(content: str) -> str:
    count = content.count("http")
    return f"Found {count} results" if count > 0 else "No results found"


def _format_crawl_output(content: str) -> str:
    pages_match = _CRAWL_PAGES_RE.search(content)
    depth_match = _CRAWL_DEPTH_RE.search(content)
    pages = pages_match.group(1) if pages_match else "?"
    depth = depth_match.group(1) if depth_match else "?"
    if pages != "?" or depth != "?":
        return f"Crawled {pages} pages (depth: {depth})"
    return "Crawl completed"


def _format_cli_output(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "Command executed (no output)"

    first_line = lines[0].replace("[stderr]", "").strip()
    preview = truncate_value(first_line, 60)
    if len(lines) > 1:
        return f"{preview} [dim](+{len(lines) - 1} lines)[/]"
    return preview


def _format_list_output(content: str) -> str:
    lines = content.splitlines()
    count = len(lines)
    preview = ", ".join(line.strip() for line in lines[:3])
    if count > 3:
        return f"Listed {count} items: {preview}, ..."
    return f"Listed {count} items: {preview}"


OUTPUT_RULES: tuple[tuple[Callable[[str], bool], Callable[[str], str]], ...] = (
    (lambda name: "web_search" in name, _format_web_search_output),
    (lambda name: "crawl_site" in name, _format_crawl_output),
    (lambda name: "cli_exec" in name or "shell" in name, _format_cli_output),
    (lambda name: "list" in name and "directory" in name, _format_list_output),
    (lambda name: "read" in name, lambda content: f"Read {len(content.splitlines())} lines ({len(content)} chars)"),
    (lambda name: "write" in name or "save" in name, lambda content: "File saved successfully"),
    (lambda name: "edit_file" in name, lambda content: "File edited successfully"),
    (lambda name: "delete" in name, lambda content: "Deleted successfully"),
    (lambda name: "fetch" in name or "download" in name, lambda content: f"Fetched content ({len(content)} chars)"),
)


def format_tool_output(name: str, content: str, is_error: bool) -> str:
    content = str(content).strip()

    if is_error:
        lower_content = content.lower()
        if "error[access_denied]" in lower_content or "cancelled by approval policy" in lower_content:
            return "Skipped"
        summary = truncate_value(content, 120)
        return f"[red]{summary}[/][yellow italic]{_hint_for_error(content)}[/]"

    name_lower = name.lower()
    for predicate, formatter in OUTPUT_RULES:
        if predicate(name_lower):
            return formatter(content)

    return truncate_value(content, 150)


def format_exception_friendly(e: Exception) -> str:
    err_str = str(e)
    err_type = type(e).__name__

    if "429" in err_str or "RateLimit" in err_type or "QuotaExceeded" in err_type or "ResourceExhausted" in err_type:
        return "Rate Limit Exceeded (429). Please wait a moment or check your API quota."

    if "401" in err_str or "403" in err_str or "Authentication" in err_type:
        return "Authentication Failed. Check your API KEY in .env."

    if "402" in err_str or "insufficient_balance" in err_str or "Insufficient account balance" in err_str:
        return "Insufficient account balance (402). Top up the provider account or switch model/provider."

    if "context_length_exceeded" in err_str or "too many tokens" in err_str.lower():
        return "Context Limit Reached. Use 'reset' to start fresh."

    if "ConnectError" in err_type or "Timeout" in err_type or "ReadTimeout" in err_type:
        return "Network Error. Connection failed or timed out."

    if len(err_str) > 300:
        return f"Error ({err_type}): {err_str[:300]}...[truncated]"

    return f"Error ({err_type}): {err_str}"


class TokenTracker:
    __slots__ = ("max_input", "total_output", "_streaming_len", "_seen_msg_ids")

    def __init__(self):
        self.max_input = 0
        self.total_output = 0
        self._streaming_len = 0
        self._seen_msg_ids: set = set()

    def update_from_message(self, msg: Any):
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            content = msg.content
            chunk_len = 0
            if isinstance(content, str):
                chunk_len = len(content)
            elif isinstance(content, list):
                chunk_len = sum(len(x.get("text", "")) for x in content if isinstance(x, dict))

            if isinstance(msg, AIMessageChunk):
                self._streaming_len += chunk_len
            elif self._streaming_len == 0:
                self._streaming_len = chunk_len

        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            self._apply_metadata(msg.usage_metadata, msg_id=getattr(msg, "id", None))

    def update_from_node_update(self, update: Dict):
        if not isinstance(update, dict):
            return

        for node_payload in update.values():
            if not isinstance(node_payload, dict):
                continue
            usage = node_payload.get("token_usage")
            if isinstance(usage, dict):
                self._apply_metadata(usage)

    @staticmethod
    def _coerce_token_int(value: Any) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, coerced)

    @classmethod
    def _extract_output_tokens(cls, usage: Dict[str, Any]) -> int:
        for key in ("output_tokens", "completion_tokens", "completion_token_count", "output_token_count"):
            if key in usage:
                return cls._coerce_token_int(usage.get(key))
        return 0

    @classmethod
    def _extract_input_tokens(cls, usage: Dict[str, Any], output_tokens: int) -> int:
        for key in ("input_tokens", "prompt_tokens", "prompt_token_count", "input_token_count"):
            if key in usage:
                return cls._coerce_token_int(usage.get(key))
        total_tokens = cls._coerce_token_int(usage.get("total_tokens"))
        if total_tokens > 0 and output_tokens > 0:
            return max(0, total_tokens - output_tokens)
        return 0

    def _apply_metadata(self, usage: Dict, msg_id: str = None):
        if msg_id:
            if msg_id in self._seen_msg_ids:
                return
            self._seen_msg_ids.add(msg_id)

        out_t = self._extract_output_tokens(usage)
        in_t = self._extract_input_tokens(usage, out_t)
        if in_t > self.max_input:
            self.max_input = in_t

        self.total_output += out_t

    def render(self, duration: float) -> str:
        display_out = self.total_output
        if self._streaming_len > 10 and display_out < (self._streaming_len // 10):
            display_out = self._streaming_len // 3

        in_display = str(self.max_input if self.max_input > 0 else 0)
        return f"{duration:.1f}s  ↓ {in_display}  ↑ {display_out}"
