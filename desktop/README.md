# Windows portable build

The binary distribution is Windows x64 with a CPython 3.10 runtime. It is a directory application, not a single self-extracting EXE. The small PyInstaller entry point launches the bundled Python/Tk desktop shell; the shell starts an isolated local backend and browser workspace.

Build from a Windows x64 CPython 3.10 virtual environment containing the required CUDA PyTorch, app requirements, torchvision, timm, transformers and ONNX Runtime packages. Install PyInstaller into that environment, or provide the packager environment at work/.venv-packager relative to the workspace root. Run:

```powershell
& 'PATH-TO-VENV/Scripts/python.exe' desktop/build_portable.py 'PATH-TO-OUTPUT/NeuroForge-Windows-0.4.1'
```

The builder copies CPython, the virtual environment dependencies, Tcl/Tk and the app sources; omits user runtime data, caches, C++ headers, static libraries and debug symbols; builds the entry point; and writes BUILD-MANIFEST.json. The docs directory, if present, is copied to the distribution root. Use a fresh output folder for new releases to avoid stale files from previous builds.

Start normally by double-clicking NeuroForge.exe. Optional flags: --port 8877, --data-dir "D:/NeuroForgeData", --no-browser. --smoke-test REPORT.json requires an explicit --data-dir and exercises proxy, training, reconnect and safe shutdown in that workspace. Test data should be isolated from user experiments.

The desktop launcher opens the local browser workspace, reconnects to its backend, and saves user data in the selected workspace folder. CUDA tasks use the installed NVIDIA driver.
