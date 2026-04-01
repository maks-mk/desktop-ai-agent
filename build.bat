@echo off
setlocal
chcp 65001 > nul

rem Force UTF-8 output so emoji in log messages don't crash on cp1251 consoles
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] venv Python not found: "%VENV_PYTHON%"
    pause
    exit /b 1
)

call "%VENV_PYTHON%" -m PyInstaller --name ai-agent --onefile --windowed --clean --collect-all tiktoken --collect-all langgraph --collect-all langchain --collect-all langchain_openai --collect-all langchain_google_genai --collect-all PySide6 --collect-all qtawesome --collect-all dotenv --collect-submodules tools --hidden-import=tiktoken_ext --hidden-import=tiktoken_ext.openai_public main.py
set "BUILD_EXIT=%ERRORLEVEL%"

pause
exit /b %BUILD_EXIT%
