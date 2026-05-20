import difflib
import json
import logging
import re
from pathlib import Path

from core.errors import ErrorType, format_error

logger = logging.getLogger(__name__)

_READ_FILE_LINE_NUMBER_RE = re.compile(r"^\s*\d+\s{2}(.*)$")


def _strip_read_file_line_numbers(text: str) -> tuple[str, bool]:
    """Remove line-number prefixes produced by read_file(show_line_numbers=True)."""
    lines = text.split("\n")
    meaningful_count = 0
    numbered_count = 0
    stripped_lines: list[str] = []

    for line in lines:
        if not line.strip():
            stripped_lines.append(line)
            continue

        meaningful_count += 1
        match = _READ_FILE_LINE_NUMBER_RE.match(line)
        if match:
            numbered_count += 1
            stripped_lines.append(match.group(1))
        else:
            stripped_lines.append(line)

    if not meaningful_count:
        return text, False

    if numbered_count >= max(1, int(meaningful_count * 0.6)):
        return "\n".join(stripped_lines), True

    return text, False


def _realign_indentation(
    file_lines: list[str],
    match_idx: int,
    search_len: int,
    new_lines: list[str],
) -> list[str]:
    """Re-align new_lines indentation to match the indentation of the block being replaced.

    When the model reads a file with show_line_numbers=True (or otherwise loses leading
    whitespace) and passes old_string / new_string without proper indentation, the
    trim/aggressive fallback can find the right block but would write new_lines verbatim,
    corrupting indentation.  This function detects the delta between the base indentation
    of new_lines and the base indentation of the matched file block, then shifts every
    line of new_lines by that delta so the replacement preserves the original indent level.
    """
    # Base indentation = indentation of the first non-empty line in the matched file block.
    file_base = ""
    for line in file_lines[match_idx : match_idx + search_len]:
        if line.strip():
            file_base = line[: len(line) - len(line.lstrip())]
            break

    # Base indentation = indentation of the first non-empty line in new_lines.
    new_base = ""
    for line in new_lines:
        if line.strip():
            new_base = line[: len(line) - len(line.lstrip())]
            break

    # Nothing to adjust.
    if file_base == new_base:
        return new_lines

    result: list[str] = []
    for line in new_lines:
        if not line.strip():
            # Preserve blank / whitespace-only lines as-is.
            result.append(line)
            continue

        current_indent = line[: len(line) - len(line.lstrip())]
        content = line.lstrip()

        if current_indent.startswith(new_base):
            # Strip the model's base indent and prepend the file's base indent,
            # keeping any extra indentation (e.g. nested blocks) intact.
            extra = current_indent[len(new_base):]
            result.append(file_base + extra + content)
        else:
            # The line is *less* indented than new_base (e.g. a dedented except/else).
            # Compute how much less, and apply the same relative reduction to file_base.
            under = len(new_base) - len(current_indent)
            adjusted_base = file_base[: max(0, len(file_base) - under)]
            result.append(adjusted_base + content)

    return result


def edit_text_file(target: Path, path_label: str, old_string: str, new_string: str) -> str:
    try:
        if not target.exists():
            return format_error(ErrorType.NOT_FOUND, f"File '{path_label}' not found.")

        content = target.read_text(encoding="utf-8")
        original_newline = "\r\n" if "\r\n" in content else "\n"
        content_norm = content.replace("\r\n", "\n")
        old_string_norm = old_string.replace("\r\n", "\n")
        new_string_norm = new_string.replace("\r\n", "\n")

        old_string_norm, stripped_old_line_numbers = _strip_read_file_line_numbers(old_string_norm)
        new_string_norm, stripped_new_line_numbers = _strip_read_file_line_numbers(new_string_norm)

        fallback_warning: str = ""
        line_number_warning: str = ""
        if stripped_old_line_numbers or stripped_new_line_numbers:
            logger.info("edit_file: stripped read_file line-number prefixes for '%s'.", path_label)
            line_number_warning = (
                "\n\n⚠ Warning: read_file line-number prefixes were removed before applying the edit. "
                "Verify the diff carefully."
            )

        count = content_norm.count(old_string_norm)
        if count == 1:
            new_content = content_norm.replace(old_string_norm, new_string_norm, 1)
        elif count > 1:
            return format_error(
                ErrorType.VALIDATION,
                f"Found {count} identical occurrences of the exact target text. "
                "Please provide more context (e.g., surrounding lines) to uniquely identify the block.",
            )
        else:
            file_lines = content_norm.split("\n")
            search_lines = old_string_norm.split("\n")

            while search_lines and not search_lines[0].strip():
                search_lines.pop(0)
            while search_lines and not search_lines[-1].strip():
                search_lines.pop()

            if not search_lines:
                return format_error(ErrorType.VALIDATION, "old_string is empty or only contains whitespaces.")

            search_len = len(search_lines)

            def find_matches(normalize_line):
                search_norm = [normalize_line(line) for line in search_lines]
                file_norm = [normalize_line(line) for line in file_lines]
                return [
                    index
                    for index in range(len(file_lines) - search_len + 1)
                    if file_norm[index : index + search_len] == search_norm
                ]

            match_mode = "trim"
            matches = find_matches(lambda value: value.strip())
            if not matches and search_len >= 3:
                match_mode = "aggressive"
                matches = find_matches(lambda value: "".join(value.split()))
            elif not matches and search_len < 3:
                snippet = old_string_norm[:80].replace("\n", "\\n")
                return format_error(
                    ErrorType.VALIDATION,
                    f"Could not find an exact/indentation-safe match for short old_string snippet '{snippet}...'. "
                    "For 1-2 line replacements, provide exact text or include more surrounding lines.",
                )

            if len(matches) == 0:
                snippet = old_string_norm[:50].replace("\n", "\\n")
                return format_error(
                    ErrorType.VALIDATION,
                    f"Could not find a match for 'old_string'.\n"
                    f"Snippet: '{snippet}...'.\n"
                    "Make sure you are replacing existing lines and DID NOT include line numbers in old_string.",
                )
            if len(matches) > 1:
                return format_error(
                    ErrorType.VALIDATION,
                    f"Found {len(matches)} occurrences of the target block. "
                    "Please include more surrounding lines to uniquely identify the block.",
                )

            match_idx = matches[0]

            if match_mode == "aggressive":
                logger.warning("edit_file: aggressive whitespace-insensitive match used for '%s'.", path_label)
                fallback_warning = (
                    "\n\n⚠ Warning: exact match failed — aggressive (whitespace-insensitive) fallback was used. "
                    "Indentation of new_string was realigned to match the replaced block. "
                    "Verify the diff carefully."
                )
            else:
                logger.info("edit_file: trim (indentation-insensitive) fallback used for '%s'.", path_label)
                fallback_warning = (
                    "\n\n⚠ Warning: exact match failed — trim (indentation-insensitive) fallback was used. "
                    "Indentation of new_string was realigned to match the replaced block. "
                    "Verify the diff carefully."
                )

            # FIX: realign new_string indentation to the file block before substitution.
            new_string_lines = _realign_indentation(
                file_lines, match_idx, search_len, new_string_norm.split("\n")
            )

            new_file_lines = (
                file_lines[:match_idx]
                + new_string_lines
                + file_lines[match_idx + search_len :]
            )
            new_content = "\n".join(new_file_lines)

        final_content = new_content.replace("\n", original_newline)
        if target.suffix.lower() == ".json":
            try:
                json.loads(final_content)
            except json.JSONDecodeError as exc:
                return format_error(
                    ErrorType.VALIDATION,
                    f"Edit would produce invalid JSON in '{path_label}' (line {exc.lineno}, column {exc.colno}): {exc.msg}",
                )

        with open(target, "w", encoding="utf-8", newline="") as file_obj:
            file_obj.write(final_content)

        diff = difflib.unified_diff(
            content_norm.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{path_label}",
            tofile=f"b/{path_label}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        return f"Success: File edited.{line_number_warning}{fallback_warning}\n\nDiff:\n```diff\n{diff_text}\n```"
    except Exception as exc:
        return format_error(ErrorType.EXECUTION, f"Error editing file: {exc}")
