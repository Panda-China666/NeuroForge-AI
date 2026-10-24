@echo off
setlocal
cd /d "%~dp0"
if exist "..\..\work\.venv-next\Scripts\python.exe" (
  set "NF_RUNTIME=..\..\work\.venv-next\Scripts\python.exe"
) else (
  set "NF_RUNTIME=.venv\Scripts\python.exe"
)
if not exist "%NF_RUNTIME%" (
  echo Run start.bat once to create the Python environment.
  exit /b 1
)
"%NF_RUNTIME%" -c "import torch,subprocess,sys; tag=str(torch.__version__).partition('+')[2] or ('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu'); subprocess.check_call([sys.executable,'-m','pip','install','torchvision==0.23.0','--index-url','https://download.pytorch.org/whl/'+tag])"
if errorlevel 1 exit /b 1
"%NF_RUNTIME%" -m pip install -r requirements-ecosystem.txt
if errorlevel 1 exit /b 1
echo Model ecosystem adapters installed. Restart NeuroForge.
