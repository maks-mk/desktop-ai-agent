import errno
import fnmatch
import itertools
import os
import re
from pathlib import Path
from typing import Optional, Union

from core.config import DEFAULT_MAX_FILE_SIZE, DEFAULT_READ_LIMIT
from core.errors import ErrorType, format_error
from core.safety_policy import SafetyPolicy
from core.utils import truncate_output

from .editing import edit_text_file
from .pathing import (
    IGNORED_DIRS,
    count_file_lines,
    delete_directory_path,
    is_binary_path,
    resolve_existing_path,
    resolve_path,
)


class FilesystemManager:
    __slots__ = ("cwd", "virtual_mode", "safety_policy")
    _LITERAL_ESCAPE_RE = re.compile(r"\\([.^$*+?{}\[\]()|\\])")

    def __init__(self, root_dir: Union[str, Path] = None, virtual_mode: bool = True):
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.safety_policy: Optional[SafetyPolicy] = None

    def set_policy(self, policy: SafetyPolicy):
        self.safety_policy = policy

    def _is_binary(self, path: Union[str, Path]) -> bool:
        return is_binary_path(str(path))

    def _count_lines(self, path: Path) -> int:
        return count_file_lines(path)

    def _resolve_path(self, path_str: str) -> Path:
        return resolve_path(self.cwd, self.virtual_mode, path_str)

    def _resolve_existing(self, path: str, expected: str) -> Path:
        return resolve_existing_path(self.cwd, self.virtual_mode, path, expected)

    def _tool_limit(self) -> int:
        return self.safety_policy.max_tool_output if self.safety_policy else 5000

    def _truncate(self, output: str, source: str = "filesystem") -> str:
        return truncate_output(output, self._tool_limit(), source=source)

    def delete_file(self, path: str) -> str:
        try:
            target = self._resolve_existing(path, "file")
            target.unlink()
            return f"Success: File {path} deleted."
        except FileNotFoundError:
            return format_error(ErrorType.NOT_FOUND, f"File not found: {path}")
        except IsADirectoryError:
            return format_error(ErrorType.VALIDATION, f"{path} is a directory. Use safe_delete_directory.")
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, str(exc))

    def delete_directory(self, path: str, recursive: bool = False) -> str:
        try:
            target = self._resolve_existing(path, "dir")
            delete_directory_path(target, recursive)
            if recursive:
                return f"Success: Directory {path} deleted recursively."
            return f"Success: Empty directory {path} deleted."
        except FileNotFoundError:
            return format_error(ErrorType.NOT_FOUND, f"Directory not found: {path}")
        except NotADirectoryError:
            return format_error(ErrorType.VALIDATION, f"{path} is a file.")
        except OSError as exc:
            if exc.errno == errno.ENOTEMPTY:
                return format_error(ErrorType.VALIDATION, "Directory is not empty. Set recursive=True to delete it.")
            return format_error(ErrorType.EXECUTION, str(exc))
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, str(exc))

    def read_file(self, path: str, offset: int = 0, limit: int = DEFAULT_READ_LIMIT, show_line_numbers: bool = True) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            stats = target.stat()
            max_size = self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE
            if stats.st_size > max_size:
                return format_error(ErrorType.LIMIT_EXCEEDED, f"File is too large ({stats.st_size} bytes). Max: {max_size}.")
            if self._is_binary(target):
                return format_error(ErrorType.VALIDATION, "File binary or unknown encoding.")
            if stats.st_size == 0:
                return "System reminder: File exists but has empty contents."

            total_lines = self._count_lines(target)
            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            limit = min(limit, policy_limit)
            if offset >= total_lines and total_lines > 0:
                return format_error(ErrorType.VALIDATION, f"Line offset {offset} exceeds file length ({total_lines} lines).")

            char_budget = max(500, self._tool_limit() - 120)
            result: list[str] = []
            chars_used = 0
            lines_read = 0
            try:
                with open(target, "r", encoding="utf-8", errors="replace") as file_obj:
                    for index, line in enumerate(itertools.islice(file_obj, offset, offset + limit)):
                        clean_line = line.rstrip("\n").rstrip("\r")
                        formatted = f"{offset + index + 1:6}  {clean_line}" if show_line_numbers else clean_line
                        line_chars = len(formatted) + 1
                        if chars_used + line_chars > char_budget:
                            break
                        result.append(formatted)
                        chars_used += line_chars
                        lines_read += 1
            except Exception as exc:
                return format_error(ErrorType.EXECUTION, f"Error reading file stream: {exc}")

            output = "\n".join(result)
            end_index = offset + lines_read
            if total_lines > end_index:
                output += (
                    f"\n\n[TRUNCATED] Showing lines {offset + 1}-{end_index} of {total_lines} "
                    f"({stats.st_size} bytes total). "
                    f"To continue: read_file(path='{path}', offset={end_index}, "
                    f"limit={limit}, show_line_numbers={show_line_numbers})"
                )
            else:
                if offset == 0 and not show_line_numbers and end_index >= total_lines:
                    return output
                output += f"\n\n[EOF] Lines {offset + 1}-{end_index} of {total_lines} ({stats.st_size} bytes)."
            return output
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error reading file: {exc}")

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Success: File '{path}' saved ({len(content)} chars)."
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error writing file: {exc}")

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        return edit_text_file(self._resolve_path(path), path, old_string, new_string)

    @classmethod
    def _literal_search_candidates(cls, pattern: str, *, ignore_case: bool) -> list[str]:
        raw_pattern = str(pattern or "")
        candidates: list[str] = []
        for candidate in (
            raw_pattern,
            cls._LITERAL_ESCAPE_RE.sub(r"\1", raw_pattern),
        ):
            normalized = candidate.lower() if ignore_case else candidate
            if normalized not in candidates:
                candidates.append(normalized)
        return candidates

    def _line_matches_literal_pattern(self, line: str, pattern: str, *, ignore_case: bool) -> bool:
        target_line = line.lower() if ignore_case else line
        for candidate in self._literal_search_candidates(pattern, ignore_case=ignore_case):
            if candidate and candidate in target_line:
                return True
        return False

    def search_in_file(self, path: str, pattern: str, use_regex: bool = False, ignore_case: bool = False) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")
            if self._is_binary(target):
                return format_error(ErrorType.VALIDATION, "File is binary or has unknown encoding.")

            flags = re.IGNORECASE if ignore_case else 0
            matches = []
            if use_regex:
                content = target.read_text(encoding="utf-8", errors="replace")
                flags |= re.MULTILINE
                try:
                    compiled = re.compile(pattern, flags)
                except re.error as exc:
                    return format_error(ErrorType.VALIDATION, f"Invalid regex pattern: {exc}")
                for match in compiled.finditer(content):
                    start_pos = match.start()
                    line_no = content.count("\n", 0, start_pos) + 1
                    match_str = match.group(0)
                    display_line = match_str.splitlines()[0] if match_str else ""
                    if not display_line.strip():
                        line_start = content.rfind("\n", 0, start_pos) + 1
                        line_end = content.find("\n", start_pos)
                        if line_end == -1:
                            line_end = len(content)
                        display_line = content[line_start:line_end]
                    matches.append((line_no, display_line))
            else:
                with open(target, "r", encoding="utf-8", errors="replace") as file_obj:
                    for index, line in enumerate(file_obj):
                        if self._line_matches_literal_pattern(line, pattern, ignore_case=ignore_case):
                            matches.append((index + 1, line.rstrip("\n").rstrip("\r")))

            if not matches:
                return f"No matches found for '{pattern}' in '{path}'."
            results = "\n".join(f"{line_no:6}  {line}" for line_no, line in matches)
            return self._truncate(f"Found {len(matches)} match(es) in '{path}':\n\n{results}")
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error searching in file: {exc}")

    def search_in_directory(
        self,
        path: str,
        pattern: str,
        use_regex: bool = False,
        ignore_case: bool = False,
        extensions: Optional[str] = None,
        max_matches: int = 500,
        max_files: int = 200,
        max_depth: Optional[int] = None,
    ) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if not target.is_dir():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a directory.")

            ext_filter = None
            if extensions:
                ext_filter = {
                    ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
                    for ext in extensions.split(",")
                }

            flags = re.IGNORECASE if ignore_case else 0
            compiled_regex = None
            if use_regex:
                flags |= re.MULTILINE
                try:
                    compiled_regex = re.compile(pattern, flags)
                except re.error as exc:
                    return format_error(ErrorType.VALIDATION, f"Invalid regex pattern: {exc}")

            all_results: list[str] = []
            files_scanned = 0
            dirs_skipped_ignored = 0
            files_skipped_binary = 0
            max_size = self.safety_policy.max_file_size if self.safety_policy else DEFAULT_MAX_FILE_SIZE
            truncated = False
            dirs_to_scan = [(target, 0)]

            while dirs_to_scan and not truncated:
                current_dir, depth = dirs_to_scan.pop()
                if max_depth is not None and depth >= max_depth:
                    continue
                try:
                    with os.scandir(current_dir) as entries:
                        for entry in entries:
                            if truncated:
                                break
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name in IGNORED_DIRS:
                                    dirs_skipped_ignored += 1
                                else:
                                    dirs_to_scan.append((Path(entry.path), depth + 1))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            if ext_filter and Path(entry.name).suffix not in ext_filter:
                                continue
                            try:
                                if entry.stat(follow_symlinks=False).st_size > max_size:
                                    continue
                            except OSError:
                                continue
                            if files_scanned >= max_files:
                                truncated = True
                                break

                            file_path = entry.path
                            try:
                                if self._is_binary(file_path):
                                    files_skipped_binary += 1
                                    continue
                            except OSError:
                                files_skipped_binary += 1
                                continue

                            files_scanned += 1
                            try:
                                rel_path = Path(file_path).relative_to(self.cwd)
                            except ValueError:
                                rel_path = file_path

                            try:
                                if use_regex and compiled_regex is not None:
                                    content = Path(file_path).read_text(encoding="utf-8", errors="replace")
                                    for match in compiled_regex.finditer(content):
                                        start_pos = match.start()
                                        line_no = content.count("\n", 0, start_pos) + 1
                                        match_str = match.group(0)
                                        display_line = match_str.splitlines()[0] if match_str else ""
                                        if not display_line.strip():
                                            line_start = content.rfind("\n", 0, start_pos) + 1
                                            line_end = content.find("\n", start_pos)
                                            if line_end == -1:
                                                line_end = len(content)
                                            display_line = content[line_start:line_end]
                                        all_results.append(f"{rel_path}:{line_no}  {display_line}")
                                        if len(all_results) >= max_matches:
                                            truncated = True
                                            break
                                else:
                                    with open(file_path, "r", encoding="utf-8", errors="replace") as file_obj:
                                        for index, line in enumerate(file_obj):
                                            if self._line_matches_literal_pattern(
                                                line,
                                                pattern,
                                                ignore_case=ignore_case,
                                            ):
                                                all_results.append(f"{rel_path}:{index + 1}  {line.rstrip()}")
                                                if len(all_results) >= max_matches:
                                                    truncated = True
                                                    break
                            except Exception:
                                continue
                except PermissionError:
                    continue

            if not all_results:
                return (
                    f"No matches found for '{pattern}' in '{path}'. "
                    f"Scanned {files_scanned} file(s), "
                    f"skipped {dirs_skipped_ignored} ignored dirs, "
                    f"{files_skipped_binary} binary files."
                )

            header_lines = [f"Found {len(all_results)} match(es) across {files_scanned} file(s)."]
            if truncated:
                header_lines.append(f"⚠ Results truncated (max_matches={max_matches}, max_files={max_files}). Narrow search.")
            if dirs_skipped_ignored:
                header_lines.append(f"  Skipped {dirs_skipped_ignored} ignored directory branches.")
            return self._truncate("\n".join(header_lines) + "\n\n" + "\n".join(all_results))
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error searching in directory: {exc}")

    def tail_file(self, path: str, lines: int = 50, show_line_numbers: bool = True) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a file.")

            policy_limit = self.safety_policy.max_read_lines if self.safety_policy else DEFAULT_READ_LIMIT
            lines = min(lines, policy_limit)
            collected: list[bytes] = []
            newlines_found = 0
            with open(target, "rb") as file_obj:
                file_obj.seek(0, 2)
                remaining = file_obj.tell()
                while newlines_found <= lines and remaining > 0:
                    step = min(8192, remaining)
                    remaining -= step
                    file_obj.seek(remaining)
                    chunk = file_obj.read(step)
                    collected.append(chunk)
                    newlines_found += chunk.count(b"\n")

            raw = b"".join(reversed(collected))
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")

            all_lines = text.splitlines()
            if all_lines and all_lines[-1] == "":
                all_lines = all_lines[:-1]
            selected = all_lines[-lines:] if len(all_lines) > lines else all_lines
            if show_line_numbers:
                result = "\n".join(f"tail-{index + 1:02}  {line}" for index, line in enumerate(selected))
            else:
                result = "\n".join(selected)
            return f"Last {len(selected)} line(s) of '{path}':\n\n{result}"
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error reading tail of file: {exc}")

    def find_files(self, path: str, name_pattern: str, max_results: int = 200, max_depth: Optional[int] = None) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if not target.is_dir():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a directory.")

            all_results = []
            files_scanned = 0
            dirs_skipped = 0
            truncated = False
            dirs_to_scan = [(target, 0)]
            while dirs_to_scan and not truncated:
                current_dir, depth = dirs_to_scan.pop()
                if max_depth is not None and depth >= max_depth:
                    continue
                try:
                    with os.scandir(current_dir) as entries:
                        for entry in entries:
                            if truncated:
                                break
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name in IGNORED_DIRS:
                                    dirs_skipped += 1
                                else:
                                    dirs_to_scan.append((Path(entry.path), depth + 1))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            files_scanned += 1
                            if fnmatch.fnmatch(entry.name, name_pattern) or fnmatch.fnmatch(entry.name.lower(), name_pattern.lower()):
                                try:
                                    rel_path = Path(entry.path).relative_to(self.cwd)
                                except ValueError:
                                    rel_path = Path(entry.path)
                                all_results.append(str(rel_path).replace("\\", "/"))
                                if len(all_results) >= max_results:
                                    truncated = True
                                    break
                except PermissionError:
                    continue

            if not all_results:
                return (
                    f"No files matching '{name_pattern}' found in '{path}'. "
                    f"Scanned {files_scanned} files, skipped {dirs_skipped} ignored directories."
                )
            header = f"Found {len(all_results)} match(es) for '{name_pattern}':"
            if truncated:
                header += f" (Truncated to {max_results} max results)"
            return self._truncate(header + "\n\n" + "\n".join(all_results))
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error finding files: {exc}")

    def file_info(self, path: str) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"File '{path}' not found.")
            if not target.is_file():
                return format_error(ErrorType.VALIDATION, f"'{path}' is not a file or directory.")

            stats = target.stat()
            size_bytes = stats.st_size
            is_binary = self._is_binary(target)
            total_lines = self._count_lines(target) if not is_binary else None
            char_budget = max(500, self._tool_limit() - 120)
            avg_chars_per_line = round(size_bytes / total_lines) if total_lines and total_lines > 0 else None
            suggested_limit = max(50, char_budget // max(1, avg_chars_per_line)) if avg_chars_per_line and not is_binary else None

            lines = [
                f"path:          {path}",
                f"size:          {size_bytes} bytes ({size_bytes / 1024:.1f} KB)",
                f"binary:        {'yes' if is_binary else 'no'}",
            ]
            if total_lines is not None:
                lines.append(f"total_lines:   {total_lines}")
            if avg_chars_per_line is not None:
                lines.append(f"avg_line_len:  ~{avg_chars_per_line} chars")
            if suggested_limit is not None:
                lines.append(
                    f"suggested_limit: {suggested_limit} lines per read_file() call "
                    f"(fits within MAX_TOOL_OUTPUT at current settings)"
                )
            import time
            lines.append(f"modified:      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))}")
            return "\n".join(lines)
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error getting file info: {exc}")

    def list_files(self, path: str, include_hidden: bool = False) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return format_error(ErrorType.NOT_FOUND, f"Path '{path}' not found.")
            if target.is_file():
                return f"[FILE] {target.name} ({target.stat().st_size} bytes)"

            results = []
            try:
                with os.scandir(target) as entries:
                    ordered = sorted(list(entries), key=lambda entry: (not entry.is_dir(follow_symlinks=False), entry.name.lower()))
                    for entry in ordered:
                        if entry.name.startswith(".") and not include_hidden:
                            continue
                        if entry.name in {".DS_Store", "__pycache__"}:
                            continue
                        prefix = "[DIR] " if entry.is_dir(follow_symlinks=False) else "[FILE]"
                        results.append(f"{prefix} {entry.name}")
            except PermissionError:
                return format_error(ErrorType.EXECUTION, "Permission denied while accessing directory.")

            output = f"Directory '{path}':\n" + "\n".join(results) + f"\n\n(Total {len(results)} items)"
            return self._truncate(output)
        except Exception as exc:
            return format_error(ErrorType.EXECUTION, f"Error listing directory: {exc}")
