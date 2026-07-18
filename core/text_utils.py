import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage, AIMessageChunk

_CLEAN_MD_RE = re.compile(r"\n{3,}")
_CRAWL_PAGES_RE = re.compile(r"(\d+) pages processed")
_CRAWL_DEPTH_RE = re.compile(r"max_depth: (\d+)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_SIMPLE_LATEX_INLINE_RE = re.compile(r"\$\s*(\\[A-Za-z]+)\s*\$")
_MARKDOWN_FENCE_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^`~\r\n]*)[ \t]*(?:\r?\n)?$")


@dataclass(frozen=True)
class MarkdownSegment:
    kind: str
    text: str
    raw: str
    language: str = ""
    closed: bool = True


def split_markdown_segments(text: str) -> list[MarkdownSegment]:
    if text == "":
        return [MarkdownSegment("markdown", "", "")]

    segments: list[MarkdownSegment] = []
    markdown_lines: list[str] = []
    code_lines: list[str] = []
    open_fence = ""
    fence_marker = ""
    language = ""
    in_fence = False

    def _flush_markdown() -> None:
        markdown_text = "".join(markdown_lines)
        if markdown_text:
            segments.append(MarkdownSegment("markdown", markdown_text, markdown_text))
        markdown_lines.clear()

    def _fence_match(line: str) -> re.Match | None:
        return _MARKDOWN_FENCE_RE.match(line)

    def _is_closing_fence(line: str) -> bool:
        match = _fence_match(line)
        if not match:
            return False
        marker = match.group("fence")
        return bool(
            fence_marker
            and not match.group("info").strip()
            and marker[0] == fence_marker[0]
            and len(marker) >= len(fence_marker)
        )

    for line in text.splitlines(keepends=True):
        fence_match = _fence_match(line)
        if fence_match:
            if in_fence:
                if _is_closing_fence(line):
                    code_text = "".join(code_lines)
                    segments.append(
                        MarkdownSegment(
                            "code",
                            code_text,
                            f"{open_fence}{code_text}{line}",
                            language=language,
                            closed=True,
                        )
                    )
                    code_lines.clear()
                    open_fence = ""
                    fence_marker = ""
                    language = ""
                    in_fence = False
                else:
                    code_lines.append(line)
            else:
                _flush_markdown()
                open_fence = line
                fence_marker = fence_match.group("fence")
                language = fence_match.group("info").strip()
                in_fence = True
            continue

        if in_fence:
            code_lines.append(line)
        else:
            markdown_lines.append(line)

    if in_fence:
        code_text = "".join(code_lines)
        segments.append(
            MarkdownSegment(
                "code",
                code_text,
                f"{open_fence}{code_text}",
                language=language,
                closed=False,
            )
        )
    else:
        _flush_markdown()

    return segments or [MarkdownSegment("markdown", "", "")]

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
    for segment in split_markdown_segments(text):
        if segment.kind == "code":
            parts.append(segment.raw)
        else:
            parts.append(_rewrite_outside_inline_code(segment.text, replacer))
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


_ESCAPED_MARKDOWN_MARKERS_RE = re.compile(r"\\([\*_`#>!|\[\]\(\){}+\-.])")


def _unescape_common_markdown_markers(text: str) -> str:
    """Undo provider-overescaped Markdown markers outside code spans/blocks."""
    return _rewrite_outside_code(text, lambda segment: _ESCAPED_MARKDOWN_MARKERS_RE.sub(r"\1", segment))


def _rewrite_local_file_links(text: str) -> str:
    """Convert Markdown local-file links to inline code to prevent Rich URL-encoding.

    [filename.md](filename.md)          →  `filename.md`
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

def truncate_value(value: str, max_length: int = 60) -> str:
    if len(value) > max_length:
        return value[:max_length] + "..."
    return value


def _single_line_preview(value: Any) -> str:
    text = str(value or "")
    # Tool card headers should stay one-line even when command/query contains newlines.
    return " ".join(text.split())


def _first_non_empty_item(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return next((str(item) for item in value if str(item).strip()), "")
    return str(value or "")


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
    query = tool_args.get("queries") if tool_name == "batch_web_search" else tool_args.get("query")
    query_text = _first_non_empty_item(query)
    if query_text:
        return f'{tool_name}("{truncate_value(_single_line_preview(query_text), 80)}")'
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
    url_text = _first_non_empty_item(url_val)
    if url_text:
        return f'{tool_name}("{truncate_value(_single_line_preview(url_text), 80)}")'
    return None


DISPLAY_RULES: tuple[tuple[set[str], Callable[[str, Dict[str, Any]], str | None]], ...] = (
    ({"read_file", "write_file", "edit_file", "Read", "Write", "SearchReplace"}, _format_path_tool),
    ({"web_search", "WebSearch", "batch_web_search"}, _format_query_tool),
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


def classify_tool_args_state(tool_name: str, tool_args: Dict[str, Any]) -> str:
    args = dict(tool_args or {})
    if not args:
        return "pending"

    anchor_keys: tuple[str, ...] = ()
    if tool_name in {"read_file", "write_file", "edit_file", "Read", "Write", "SearchReplace"}:
        anchor_keys = ("path", "file_path")
    elif tool_name in {"web_search", "WebSearch"}:
        anchor_keys = ("query",)
    elif tool_name == "batch_web_search":
        anchor_keys = ("queries",)
    elif tool_name in {"grep", "Grep", "glob", "Glob", "search_in_file", "search_in_directory", "find_file"}:
        anchor_keys = ("pattern", "name_pattern")
    elif tool_name in {"execute", "RunCommand", "cli_exec"}:
        anchor_keys = ("command",)
    elif tool_name in {"fetch_url", "WebFetch", "fetch_content", "download_file"}:
        anchor_keys = ("url", "urls")

    if anchor_keys and not any(args.get(key) for key in anchor_keys):
        return "partial"
    return "complete"


def tool_source_kind(tool_name: str) -> str:
    normalized = str(tool_name or "").strip().lower()
    if normalized == "cli_exec":
        return "cli"
    if ":" in normalized:
        return "mcp"
    return "tool"


def _humanize_tool_name(tool_name: str) -> str:
    words = str(tool_name or "").replace(":", " ").replace("-", " ").replace("_", " ").strip().split()
    return " ".join(word.upper() if word.casefold() == "id" else word.capitalize() for word in words) or "Tool"


def _mcp_target_summary(tool_args: Dict[str, Any]) -> str:
    preferred_keys = (
        "query", "topic", "question", "library_name", "libraryName", "library_id", "libraryId",
        "path", "file_path", "url", "uri", "name", "id",
    )
    for key in preferred_keys:
        value = tool_args.get(key)
        if value not in (None, "", [], {}):
            return truncate_value(_single_line_preview(value), 100)
    for value in tool_args.values():
        if value not in (None, "", [], {}):
            return truncate_value(_single_line_preview(value), 100)
    return ""


def build_mcp_tool_ui_labels(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    phase: str = "running",
    is_error: bool = False,
    server_name: str = "",
) -> Dict[str, str]:
    human_name = _humanize_tool_name(tool_name)
    server = _humanize_tool_name(server_name) if server_name else "MCP"
    if is_error:
        title = f"{server}: {human_name} failed"
    elif phase == "finished":
        title = f"{server}: {human_name} completed"
    else:
        title = f"{server}: {human_name}"
    args = dict(tool_args or {})
    return {
        "title": title,
        "subtitle": _mcp_target_summary(args),
        "raw_display": format_tool_display(tool_name, args),
        "args_state": "complete" if args else "pending",
        "source_kind": "mcp",
    }


def tool_title_case(tool_name: str) -> str:
    words = str(tool_name or "").replace(":", " ").replace("_", " ").strip().split()
    if not words:
        return "Tool"
    return " ".join(word[:1].upper() + word[1:] for word in words)


def tool_target_summary(tool_name: str, tool_args: Dict[str, Any]) -> str:
    args = dict(tool_args or {})
    normalized_name = str(tool_name or "").strip()

    if normalized_name in {"read_file", "write_file", "edit_file", "Read", "Write", "SearchReplace"}:
        path_value = args.get("file_path") or args.get("path")
        return abbreviate_path(str(path_value)) if path_value else ""
    if normalized_name in {"ls", "LS", "list_directory"}:
        path_value = args.get("path")
        return abbreviate_path(str(path_value)) if path_value else "current directory"
    if normalized_name in {"web_search", "WebSearch"}:
        query = args.get("query")
        return truncate_value(_single_line_preview(query), 80) if query else ""
    if normalized_name == "batch_web_search":
        query = _first_non_empty_item(args.get("queries"))
        return truncate_value(_single_line_preview(query), 80) if query else ""
    if normalized_name in {"grep", "Grep", "glob", "Glob", "search_in_file", "search_in_directory", "find_file"}:
        pattern_val = args.get("pattern") or args.get("name_pattern")
        return truncate_value(_single_line_preview(pattern_val), 70) if pattern_val else ""
    if normalized_name in {"execute", "RunCommand", "cli_exec"}:
        command = args.get("command")
        return truncate_value(_single_line_preview(command), 100) if command else ""
    if normalized_name in {"fetch_url", "WebFetch", "fetch_content", "download_file"}:
        url_text = _first_non_empty_item(args.get("url") or args.get("urls"))
        return truncate_value(_single_line_preview(url_text), 80) if url_text else ""
    return ""


def build_tool_ui_labels(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    phase: str = "running",
    is_error: bool = False,
) -> Dict[str, str]:
    normalized_name = str(tool_name or "").strip() or "unknown_tool"
    args = dict(tool_args or {})
    args_state = classify_tool_args_state(normalized_name, args)
    target = tool_target_summary(normalized_name, args)

    action_map = {
        "read_file": "Reading file",
        "write_file": "Writing file",
        "edit_file": "Editing file",
        "Read": "Reading file",
        "Write": "Writing file",
        "SearchReplace": "Editing file",
        "ls": "Listing directory",
        "LS": "Listing directory",
        "list_directory": "Listing directory",
        "web_search": "Searching web",
        "WebSearch": "Searching web",
        "batch_web_search": "Searching web",
        "grep": "Searching files",
        "Grep": "Searching files",
        "glob": "Finding files",
        "Glob": "Finding files",
        "search_in_file": "Searching file",
        "search_in_directory": "Searching directory",
        "find_file": "Finding file",
        "fetch_url": "Fetching URL",
        "WebFetch": "Fetching URL",
        "fetch_content": "Fetching content",
        "download_file": "Downloading file",
        "execute": "Running command",
        "RunCommand": "Running command",
        "cli_exec": "Running command",
    }
    preparing_map = {
        "read_file": "Preparing file read",
        "write_file": "Preparing file write",
        "edit_file": "Preparing edit",
        "Read": "Preparing file read",
        "Write": "Preparing file write",
        "SearchReplace": "Preparing edit",
        "ls": "Preparing directory listing",
        "LS": "Preparing directory listing",
        "list_directory": "Preparing directory listing",
        "web_search": "Preparing search",
        "WebSearch": "Preparing search",
        "batch_web_search": "Preparing search",
        "grep": "Preparing search",
        "Grep": "Preparing search",
        "glob": "Preparing file search",
        "Glob": "Preparing file search",
        "search_in_file": "Preparing file search",
        "search_in_directory": "Preparing directory search",
        "find_file": "Preparing file search",
        "fetch_url": "Preparing fetch",
        "WebFetch": "Preparing fetch",
        "fetch_content": "Preparing fetch",
        "download_file": "Preparing download",
        "execute": "Preparing command",
        "RunCommand": "Preparing command",
        "cli_exec": "Preparing command",
    }
    base_title = action_map.get(normalized_name, tool_title_case(normalized_name))

    if is_error:
        title = f"{base_title} failed"
    elif phase == "finished":
        title = base_title
    elif phase == "preparing":
        title = preparing_map.get(normalized_name, f"Preparing {base_title.lower()}")
    else:
        title = base_title

    if args_state == "pending":
        subtitle = "Waiting for arguments…"
    elif args_state == "partial":
        subtitle = target or "Resolving arguments…"
    else:
        subtitle = target

    raw_display = format_tool_display(normalized_name, args) if args_state == "complete" else normalized_name
    return {
        "title": title,
        "subtitle": subtitle,
        "raw_display": raw_display,
        "args_state": args_state,
        "source_kind": tool_source_kind(normalized_name),
    }


def _collapse_non_code_markdown(text: str) -> str:
    parts: list[str] = []
    for segment in split_markdown_segments(text):
        if segment.kind == "code":
            parts.append(segment.raw)
        else:
            parts.append(_CLEAN_MD_RE.sub("\n\n", segment.text))
    return "".join(parts)


def clean_markdown_text(text: str) -> str:
    if not text:
        return text
    return _collapse_non_code_markdown(text)


def prepare_markdown_for_render(text: str) -> str:
    text = _normalize_simple_latex_inline(text)
    text = _unescape_common_markdown_markers(text)
    text = _rewrite_local_file_links(text)
    # Do not infer fenced code while text is streaming. The inference can change
    # its mind as a line grows, which makes ordinary prose flash as a code block.
    # Explicit Markdown fences remain fully supported by the renderer.
    return clean_markdown_text(text)


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
        return f"{preview} (+{len(lines) - 1} lines)"
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
        return f"{summary}{_hint_for_error(content)}"

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
