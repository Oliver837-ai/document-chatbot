@echo off
rem This script checks for Ollama installation and pulls the required LLM model.

set LOCAL_LLM_MODEL=phi3
set OLLAMA_DOWNLOAD_URL=https://ollama.com/download/windows
set INSTALL_MARKER_FILE=.installed_marker

echo.
echo --- Ollama and LLM Model Setup ---
echo This script will help you get Ollama and the '%LOCAL_LLM_MODEL%' model.
echo.

rem --- NEW: Check for installation marker ---
if not exist "%INSTALL_MARKER_FILE%" (
    echo ERROR: Program not initialized. Please run "setup.py" FIRST.
    echo Exiting.
    pause
    exit /b 1
)
rem -----------------------------------------

rem Check if Ollama is installed by trying to run 'ollama --version'
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Ollama is not found on your system.
    echo Please install Ollama first from: %OLLAMA_DOWNLOAD_URL%
    echo ^(It's a simple executable installer.^)
    echo After installation, please run this batch file again.
    echo.
    pause
    exit /b 1
) else (
    echo Ollama is detected. Version:
    ollama --version
    echo.
)

echo Attempting to pull the LLM model: %LOCAL_LLM_MODEL%
echo This may take some time depending on your internet speed and model size (around 2-3 GB).
echo Do NOT close this window until the download is complete!

ollama pull %LOCAL_LLM_MODEL%
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to pull Ollama model '%LOCAL_LLM_MODEL%'.
    echo Possible reasons:
    echo - Internet connection issues.
    echo - Incorrect model name.
    echo - Insufficient disk space.
    echo - Ollama server not running correctly (check your system tray).
    echo Please try again or check Ollama's logs.
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo SUCCESS: Model '%LOCAL_LLM_MODEL%' downloaded!
)

echo.
echo --- Ollama and LLM Model Setup Complete! ---
echo You can now run the AI Document Assistant.
echo Please run 'run_app.bat' to start the application.
echo.
pause
