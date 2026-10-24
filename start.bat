@echo off
setlocal
cd /d "%~dp0"
if exist "..\..\work\.venv-next\Scripts\python.exe" (
  set "NF_RUNTIME=..\..\work\.venv-next\Scripts\python.exe"
  goto workspace_runtime
)
if exist "..\..\work\.venv\Scripts\python.exe" (
  "..\..\work\.venv\Scripts\python.exe" -c "import torch,fastapi,uvicorn" >nul 2>&1
  if not errorlevel 1 (
    set "NF_RUNTIME=..\..\work\.venv\Scripts\python.exe"
    goto workspace_runtime
  )
)
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
if not exist ".venv\.neuroforge-ready" (
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto failed
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto failed
  echo ready> .venv\.neuroforge-ready
)
echo NeuroForge AI: http://127.0.0.1:8787
echo Keep this window open. Press Ctrl+C to stop the engine.
.venv\Scripts\python.exe run.py
if errorlevel 1 goto failed
exit /b 0
:workspace_runtime
echo NeuroForge AI: http://127.0.0.1:8787
"%NF_RUNTIME%" run.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Setup or startup failed. See the error above. Check Python 3.10+ and network access.
pause
exit /b 1
