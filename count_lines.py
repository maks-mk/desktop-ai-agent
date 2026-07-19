#!/usr/bin/env python3
from pathlib import Path
from typing import Iterator
import argparse
import ast
import io
import tokenize

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "venv",
    ".venv",
    "env",
    ".env",
    "ENV",
    "build",
    "dist",
    "site-packages",
    "node_modules",
}


def is_excluded(path: Path, root: Path) -> bool:
    """True, если файл лежит внутри одной из EXCLUDED_DIRS относительно root.

    Сам корень сканирования не учитывается, поэтому запуск внутри папки с
    именем из EXCLUDED_DIRS (например dist) не отбрасывает все файлы.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    # parts[:-1] — родительские директории файла относительно root (без имени файла)
    return any(part in EXCLUDED_DIRS for part in rel.parts[:-1])


def _is_standalone_string(node: ast.AST) -> bool:
    """True для выражения-инструкции, состоящего только из строкового литерала
    (docstring или bare string)."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _significant_token_lines(source: bytes) -> set[int]:
    """Возвращает строки, где есть токены кода, а не только пробелы/комментарии."""
    ignored_tokens = {
        tokenize.ENCODING,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    lines: set[int] = set()

    for token in tokenize.tokenize(io.BytesIO(source).readline):
        if token.type in ignored_tokens:
            continue

        start_line = token.start[0]
        end_line = token.end[0]
        for line_no in range(start_line, end_line + 1):
            lines.add(line_no)

    return lines


def count_real_lines(file_path: Path) -> int:
    """
    Считает реальные строки кода:
    - без пустых строк
    - без комментариев
    - без standalone-строк и docstring

    Использует ast, чтобы отличать docstring/standalone-строки от строковых
    литералов в выражениях (например, x = "hello" считается кодом).
    Битые файлы (SyntaxError) не роняют подсчёт — возвращается 0.
    """
    try:
        source = file_path.read_bytes()
    except OSError:
        return 0

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return 0

    try:
        significant_lines = _significant_token_lines(source)
    except tokenize.TokenError:
        return 0

    code_lines: set[int] = set()
    string_lines: set[int] = set()

    for node in ast.walk(tree):
        if _is_standalone_string(node):
            start = node.lineno
            end = node.end_lineno or start
            for line_no in range(start, end + 1):
                string_lines.add(line_no)
        if hasattr(node, "lineno"):
            start = node.lineno
            end = node.end_lineno or start
            for line_no in range(start, end + 1):
                code_lines.add(line_no)

    return len((code_lines - string_lines) & significant_lines)


def iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if not is_excluded(path, root):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count real lines of Python code excluding comments, blank lines, venv, cache dirs, etc."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to scan. Default: current directory.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show per-file line counts.",
    )

    args = parser.parse_args()
    root = Path(args.path).resolve()

    if not root.exists():
        parser.error(f"Path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Path is not a directory: {root}")

    total = 0
    files_count = 0

    for py_file in sorted(iter_python_files(root)):
        count = count_real_lines(py_file)
        total += count
        files_count += 1

        if args.details:
            print(f"{count:6}  {py_file.relative_to(root)}")

    print()
    print(f"Python files: {files_count}")
    print(f"Real code lines: {total}")


if __name__ == "__main__":
    main()
