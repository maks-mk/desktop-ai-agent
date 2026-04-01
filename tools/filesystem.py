"""
Compatibility facade for filesystem tools.
Tool names and imports stay stable while the implementation lives in smaller internal modules.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Optional

import aiofiles
import httpx
from langchain_core.tools import tool
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from core.config import DEFAULT_MAX_FILE_SIZE
from core.errors import ErrorType, format_error
from core.safety_policy import SafetyPolicy
from tools.filesystem_impl import FilesystemManager

logger = logging.getLogger(__name__)

fs_manager = FilesystemManager(virtual_mode=True)


def set_safety_policy(policy: SafetyPolicy):
    fs_manager.set_policy(policy)


def set_working_directory(cwd: str):
    fs_manager.cwd = Path(cwd).resolve()


_EDIT_FILE_ALIAS_MAP = {
    "old_string": ("old_text", "search_text", "find_text", "target_text", "old", "before", "from"),
    "new_string": ("new_text", "replace_text", "replacement", "replace_with", "new", "after", "to"),
}


def _cleanup_edit_path(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.strip("\"'")
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(?:path|file_path|filepath)\s*[:=]\s*", "", text, flags=re.IGNORECASE)

    # Defensive cleanup for malformed LLM arguments where browser UA lands in `path`.
    for marker in (" Mozilla/", " AppleWebKit/", " Safari/", " Chrome/", " Firefox/", " Edg/"):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip(" ,;")
            break

    return text or None


class EditFileInput(BaseModel):
    """Tolerant input schema for edit_file to reduce LLM argument-shape failures."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    path: str | None = Field(
        default=None,
        description="Path to the file to edit.",
    )
    old_string: str | None = Field(
        default=None,
        validation_alias=AliasChoices("old_string", *_EDIT_FILE_ALIAS_MAP["old_string"]),
        description="Exact old text block to replace.",
    )
    new_string: str | None = Field(
        default=None,
        validation_alias=AliasChoices("new_string", *_EDIT_FILE_ALIAS_MAP["new_string"]),
        description="Replacement text block.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)

        for field_name, aliases in _EDIT_FILE_ALIAS_MAP.items():
            if data.get(field_name) is not None:
                continue
            for alias in aliases:
                if data.get(alias) is not None:
                    data[field_name] = data[alias]
                    break

        data["path"] = _cleanup_edit_path(data.get("path"))

        for key in ("old_string", "new_string"):
            if data.get(key) is None:
                continue
            text = str(data[key]).replace("\r\n", "\n")
            data[key] = text

        return data


@tool("file_info")
def file_info_tool(path: str) -> str:
    """Returns metadata for a file: size, line count, and suggested chunk size for read_file()."""
    return fs_manager.file_info(path)


@tool("read_file")
def read_file_tool(path: str, offset: int = 0, limit: int = 2000, show_line_numbers: bool = True) -> str:
    """Reads a file from the filesystem with automatic pagination."""
    return fs_manager.read_file(path, offset, limit, show_line_numbers)


@tool("write_file")
def write_file_tool(path: str, content: str) -> str:
    """Writes content to a file. Overwrites existing files completely."""
    return fs_manager.write_file(path, content)


@tool("edit_file", args_schema=EditFileInput)
def edit_file_tool(path: str | None = None, old_string: str | None = None, new_string: str | None = None) -> str:
    """Replaces text in a file with exact-match and safe heuristic fallback modes."""
    if not path:
        return format_error(ErrorType.VALIDATION, "Missing required field: path.")
    if old_string is None or not str(old_string).strip():
        return format_error(
            ErrorType.VALIDATION,
            "Missing required field: old_string. Accepted aliases: old_text, search_text, find_text.",
        )
    if new_string is None:
        return format_error(
            ErrorType.VALIDATION,
            "Missing required field: new_string. Accepted aliases: new_text, replace_text, replacement.",
        )
    return fs_manager.edit_file(path, old_string, new_string)


@tool("list_directory")
def list_directory_tool(path: str = ".", include_hidden: bool = False) -> str:
    """Lists files and directories in a given path."""
    return fs_manager.list_files(path, include_hidden)


@tool("search_in_file")
def search_in_file_tool(path: str, pattern: str, use_regex: bool = False, ignore_case: bool = False) -> str:
    """Searches for a text pattern (or regex) in a single file."""
    return fs_manager.search_in_file(path, pattern, use_regex, ignore_case)


@tool("search_in_directory")
def search_in_directory_tool(
    path: str,
    pattern: str,
    use_regex: bool = False,
    ignore_case: bool = False,
    extensions: Optional[str] = None,
    max_matches: int = 500,
    max_files: int = 200,
    max_depth: Optional[int] = None,
) -> str:
    """Recursively searches for a text pattern (or regex) across all files in a directory."""
    return fs_manager.search_in_directory(
        path,
        pattern,
        use_regex,
        ignore_case,
        extensions,
        max_matches,
        max_files,
        max_depth,
    )


@tool("tail_file")
def tail_file_tool(path: str, lines: int = 50, show_line_numbers: bool = True) -> str:
    """Returns the last N lines of a file (like Unix `tail`)."""
    return fs_manager.tail_file(path, lines, show_line_numbers)


@tool("safe_delete_file")
async def safe_delete_file(file_path: str) -> str:
    """Deletes a file in the working directory."""
    return await asyncio.to_thread(fs_manager.delete_file, file_path)


@tool("safe_delete_directory")
async def safe_delete_directory(dir_path: str, recursive: bool = False) -> str:
    """Deletes a directory in the working directory."""
    return await asyncio.to_thread(fs_manager.delete_directory, dir_path, recursive)


_DOWNLOAD_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


def _cleanup_partial_download(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _format_download_http_error(exc: httpx.HTTPStatusError) -> str:
    status_code = exc.response.status_code
    reason = exc.response.reason_phrase or "Unknown error"
    if status_code == 403:
        return format_error(
            ErrorType.ACCESS_DENIED,
            "HTTP 403 - Forbidden. Remote host blocked the download or requires browser-only access.",
        )
    if status_code == 404:
        return format_error(
            ErrorType.NOT_FOUND,
            "HTTP 404 - Not Found. Check that the URL points to the direct file, not a landing page.",
        )
    return format_error(ErrorType.NETWORK, f"HTTP {status_code} - {reason}")


def _format_download_request_error(exc: httpx.RequestError) -> str:
    detail = str(exc).strip() or type(exc).__name__
    return format_error(ErrorType.NETWORK, f"Network request failed ({type(exc).__name__}): {detail}")


@tool("download_file")
async def download_file(url: str, filename: Optional[str] = None) -> str:
    """Downloads a file from a URL to the current working directory."""
    temp_destination: Optional[Path] = None
    try:
        if not filename:
            filename = url.split("/")[-1] or "downloaded_file"

        try:
            destination = fs_manager._resolve_path(filename)
        except ValueError as exc:
            return format_error(ErrorType.VALIDATION, str(exc))

        temp_destination = destination.with_name(destination.name + ".part")
        _cleanup_partial_download(temp_destination)

        from tools.system_tools import get_net_client

        client = get_net_client()
        logger.info("⬇️ Downloading %s to %s", url, destination)

        try:
            async with client.client.stream(
                "GET",
                url,
                follow_redirects=True,
                headers=_DOWNLOAD_HEADERS,
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                max_size = fs_manager.safety_policy.max_file_size if fs_manager.safety_policy else DEFAULT_MAX_FILE_SIZE
                if content_length:
                    try:
                        if int(content_length) > max_size:
                            return format_error(ErrorType.LIMIT_EXCEEDED, f"File too large (>{max_size} bytes). Download aborted.")
                    except ValueError:
                        pass

                async with aiofiles.open(temp_destination, "wb") as file_obj:
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            _cleanup_partial_download(temp_destination)
                            return format_error(ErrorType.LIMIT_EXCEEDED, f"File exceeded max size {max_size}. Aborted.")
                        await file_obj.write(chunk)

            temp_destination.replace(destination)
            return f"Success: File downloaded to {destination}"
        except httpx.HTTPStatusError as exc:
            _cleanup_partial_download(temp_destination)
            return _format_download_http_error(exc)
        except httpx.RequestError as exc:
            _cleanup_partial_download(temp_destination)
            return _format_download_request_error(exc)
    except Exception as exc:
        if temp_destination is not None:
            _cleanup_partial_download(temp_destination)
        return format_error(ErrorType.EXECUTION, f"Error downloading file: {exc}")


@tool("find_file")
def find_file_tool(name_pattern: str, path: str = ".", max_results: int = 200, max_depth: Optional[int] = None) -> str:
    """Finds files by their name pattern in a directory (recursive)."""
    return fs_manager.find_files(path, name_pattern, max_results, max_depth)


__all__ = [
    "FilesystemManager",
    "fs_manager",
    "set_safety_policy",
    "set_working_directory",
    "file_info_tool",
    "read_file_tool",
    "write_file_tool",
    "edit_file_tool",
    "list_directory_tool",
    "search_in_file_tool",
    "search_in_directory_tool",
    "tail_file_tool",
    "safe_delete_file",
    "safe_delete_directory",
    "download_file",
    "find_file_tool",
    "_DOWNLOAD_HEADERS",
    "_format_download_http_error",
]
