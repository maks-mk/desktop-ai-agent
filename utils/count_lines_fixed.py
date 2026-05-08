import os
import tokenize
import io

EXCLUDE_DIRS = {"venv", "__pycache__", "tests"}


def count_code_lines(filepath):
    """Точный подсчет строк кода с использованием токенизатора Python"""
    try:
        with open(filepath, "rb") as f:
            tokens = tokenize.tokenize(f.readline)
            code_lines = set()
            
            for token in tokens:
                # Считаем только реальный код (не комментарии, не строки, не пустые строки)
                if token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.OP):
                    code_lines.add(token.start[0])
                elif token.type == tokenize.STRING:
                    # Строки считаем только если они не docstrings
                    # (docstring - это STRING токен, который является первым statement)
                    # Для упрощения считаем все STRING как код, т.к. они могут быть значениями
                    code_lines.add(token.start[0])
            
            return len(code_lines)
            
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return 0


def count_code_lines_accurate(filepath):
    """Максимально точный подсчет - только строки с исполняемым кодом"""
    try:
        with open(filepath, "rb") as f:
            tokens = list(tokenize.tokenize(f.readline))
            code_lines = set()
            
            for i, token in enumerate(tokens):
                # Пропускаем комментарии, NEWLINE, NL, INDENT, DEDENT, ENCODING, ENDMARKER
                if token.type in (tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL, 
                                  tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
                                  tokenize.ENDMARKER):
                    continue
                
                # Пропускаем standalone строки (docstrings)
                if token.type == tokenize.STRING:
                    # Проверяем, является ли это docstring (после INDENT или NEWLINE)
                    if i > 0 and tokens[i-1].type in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL):
                        # Проверяем следующий токен - если NEWLINE, то это docstring
                        if i < len(tokens) - 1 and tokens[i+1].type in (tokenize.NEWLINE, tokenize.NL):
                            continue
                
                code_lines.add(token.start[0])
            
            return len(code_lines)
            
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return 0


def count_project(root=".", accurate=False):
    """
    Подсчет строк кода во всем проекте
    
    Args:
        root: корневая директория
        accurate: использовать точный режим (без docstrings)
    """
    total_lines = 0
    total_files = 0
    counter = count_code_lines_accurate if accurate else count_code_lines

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for file in filenames:
            if file.endswith(".py"):
                filepath = os.path.join(dirpath, file)
                lines = counter(filepath)
                total_lines += lines
                total_files += 1

    return total_files, total_lines


if __name__ == "__main__":
    import sys
    
    accurate = "--accurate" in sys.argv
    mode = "точный (без docstrings)" if accurate else "стандартный"
    
    files, lines = count_project(".", accurate=accurate)
    print(f"Режим: {mode}")
    print(f"Файлов: {files}")
    print(f"Строк кода: {lines}")
