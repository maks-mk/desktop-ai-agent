#!/usr/bin/env python3
from pathlib import Path
import argparse
import tokenize
from io import BytesIO

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

def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)

def count_real_lines(file_path: Path) -> int:
    """
    Считает реальные строки кода:
    - без пустых строк
    - без комментариев
    - без docstring/string-only строк
    """
    try:
        source = file_path.read_bytes()
    except (OSError, UnicodeDecodeError):
        return 0

    code_lines = set()

    try:
        tokens = tokenize.tokenize(BytesIO(source).readline)
    except tokenize.TokenError:
        return 0

    for token in tokens:
        token_type = token.type
        start_line = token.start[0]
        end_line = token.end[0]
        text = token.string

        if token_type in {
            tokenize.ENCODING,
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        }:
            continue

        # Игнорируем standalone string/docstring.
        # Это не идеально для всех редких случаев, но хорошо работает для подсчёта LOC.
        if token_type == tokenize.STRING:
            continue

        for line_no in range(start_line, end_line + 1):
            if text.strip():
                code_lines.add(line_no)

    return len(code_lines)

def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if not is_excluded(path):
            yield path

def main():
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