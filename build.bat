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

pushd "%SCRIPT_DIR%"

call "%VENV_PYTHON%" -m PyInstaller ^
--name ai-agent ^
--onefile ^
--windowed ^
--clean ^
--paths "%SCRIPT_DIR%" ^
--collect-all tiktoken ^
--collect-all langgraph ^
--collect-all langchain ^
--collect-all langchain_openai ^
--collect-all langchain_google_genai ^
--collect-all qtawesome ^
--collect-submodules tools ^
--collect-submodules ui ^
--hidden-import=PySide6.QtCore ^
--hidden-import=PySide6.QtGui ^
--hidden-import=PySide6.QtWidgets ^
--hidden-import=PySide6.QtSvg ^
--hidden-import=tiktoken_ext ^
--hidden-import=tiktoken_ext.openai_public ^
--icon="%SCRIPT_DIR%icon.ico" ^
"%SCRIPT_DIR%main.py"
set "BUILD_EXIT=%ERRORLEVEL%"

popd

pause
exit /b %BUILD_EXIT%
