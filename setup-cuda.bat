@echo off
setlocal
cd /d "%~dp0"
echo NeuroForge CUDA setup - NVIDIA driver required
nvidia-smi
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --resume-retries 15 --timeout 120
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.ones(16,16,device='cuda'); print(torch.cuda.get_device_name(0), float((x@x).sum()))"
if errorlevel 1 goto failed
echo ready> .venv\.neuroforge-ready
echo CUDA is ready. Run start.bat to launch NeuroForge.
pause
exit /b 0
:failed
echo CUDA setup failed. Check the output above and NVIDIA driver compatibility.
pause
exit /b 1
