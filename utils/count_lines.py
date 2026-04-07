import os

EXCLUDE_DIRS = {"venv", "__pycache__"}


def count_code_lines(filepath):
    code_lines = 0
    in_multiline = False
    multiline_delim = None

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()

                if not stripped:
                    continue

                # обработка многострочных строк
                if in_multiline:
                    if multiline_delim in stripped:
                        in_multiline = False
                        multiline_delim = None
                    continue

                if stripped.startswith(("'''", '"""')):
                    # начало многострочного блока
                    if stripped.count("'''") == 2 or stripped.count('"""') == 2:
                        continue  # строка типа '''text'''
                    in_multiline = True
                    multiline_delim = "'''" if stripped.startswith("'''") else '"""'
                    continue

                # однострочные комментарии
                if stripped.startswith("#"):
                    continue

                code_lines += 1

    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")

    return code_lines


def count_project(root="."):
    total_lines = 0
    total_files = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for file in filenames:
            if file.endswith(".py"):
                filepath = os.path.join(dirpath, file)
                lines = count_code_lines(filepath)
                total_lines += lines
                total_files += 1

    return total_files, total_lines


if __name__ == "__main__":
    files, lines = count_project(".")
    print(f"Файлов: {files}")
    print(f"Строк кода (без мусора): {lines}")