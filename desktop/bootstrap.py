"""Conventional PyInstaller bootstrap; the application owns its separate runtime."""
import sys,subprocess
from pathlib import Path

def main():
    root=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[2]
    runtime=root/'_internal'/'python'/'pythonw.exe';script=root/'_internal'/'app'/'desktop_ui.py'
    if not runtime.exists() or not script.exists():
        import ctypes
        ctypes.windll.user32.MessageBoxW(None,'Extract the entire application folder, including _internal. Do not move the EXE by itself.','NeuroForge startup',0x40);return 1
    child=subprocess.Popen([str(runtime),'-X','utf8',str(script),*sys.argv[1:]],cwd=script.parent,creationflags=subprocess.CREATE_NO_WINDOW)
    return child.wait()
if __name__=='__main__':raise SystemExit(main())
