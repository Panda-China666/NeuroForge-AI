"""Build a relocatable Windows runtime without system Python dependencies."""
from pathlib import Path
import sys,shutil,subprocess,json,os,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
TARGET=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT.parent/'NeuroForge-Windows-0.4.1'
VENV=Path(sys.executable).resolve().parents[1]
BASE=Path(sys.base_prefix)
TARGET.mkdir(parents=True,exist_ok=True)
APP=TARGET/'_internal'/'app';PY=TARGET/'_internal'/'python';PY.mkdir(parents=True,exist_ok=True)

def copytree(src,dst,mode='normal'):
    count=0;total=0
    for folder,dirs,names in os.walk(src):
        relative=Path(folder).relative_to(src)
        dirs[:]=[d for d in dirs if d not in ('__pycache__','.pytest_cache','.git','.venv','tests') and not d.endswith('.egg-info')]
        if mode=='app' and relative==Path('.'):dirs[:]=[d for d in dirs if d!='runtime']
        if mode=='stdlib':dirs[:]=[d for d in dirs if d not in ('site-packages','test','idlelib','tkinter','turtledemo','ensurepip')]
        if mode=='deps' and relative==Path('torch'):dirs[:]=[d for d in dirs if d not in ('include','share')]
        for name in names:
            if name.endswith(('.pyc','.pyo','.pdb','.lib')) or name.startswith('__editable__') or name=='pyvenv.cfg':continue
            if mode=='deps' and ('neuroforge_ai' in name.lower() or 'neuroforge_ai' in str(relative).lower()):continue
            source=Path(folder)/name;target=dst/relative/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or target.stat().st_size!=source.stat().st_size or target.stat().st_mtime<source.stat().st_mtime:shutil.copy2(source,target)
            count+=1;total+=source.stat().st_size
    print(mode,count,round(total/1e9,3),'GB',flush=True);return count,total

copytree(ROOT,APP,'app')
for name in ('python.exe','pythonw.exe','python310.dll','python3.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt'):
    if (BASE/name).exists():shutil.copy2(BASE/name,PY/name)
copytree(BASE/'DLLs',PY/'DLLs')
copytree(BASE/'Lib',PY/'Lib','stdlib')
copytree(VENV/'Lib'/'site-packages',PY/'Lib'/'site-packages','deps')
(PY/'python310._pth').write_text('.\nDLLs\nLib\nLib/site-packages\n../app\nimport site\n',encoding='ascii')
packager=ROOT.parents[1]/'work'/'.venv-packager'/'Scripts'/'python.exe'
if not packager.exists():packager=Path(sys.executable)
build=ROOT.parents[1]/'work'/'portable-bootstrap'
subprocess.run([str(packager),'-m','PyInstaller','--noconfirm','--clean','--onedir','--windowed','--noupx','--name','NeuroForge','--contents-directory','_bootstrap','--distpath',str(build/'dist'),'--workpath',str(build/'work'),'--specpath',str(build/'spec'),str(ROOT/'desktop'/'bootstrap.py')],check=True)
shutil.copy2(build/'dist'/'NeuroForge'/'NeuroForge.exe',TARGET/'NeuroForge.exe')
shutil.copytree(build/'dist'/'NeuroForge'/'_bootstrap',TARGET/'_bootstrap',dirs_exist_ok=True)
shutil.copytree(BASE/'Lib'/'tkinter',PY/'Lib'/'tkinter',dirs_exist_ok=True)
shutil.copytree(BASE/'tcl',PY/'tcl',dirs_exist_ok=True)
if (ROOT/'docs').exists():shutil.copytree(ROOT/'docs',TARGET/'docs',dirs_exist_ok=True)
(TARGET/'Start-NeuroForge.cmd').write_text('@echo off\ncd /d "%~dp0"\nstart "" "%~dp0_internal\\python\\pythonw.exe" -X utf8 "%~dp0_internal\\app\\desktop_ui.py" %*\n',encoding='ascii')
(TARGET/'START-HERE.txt').write_text('NeuroForge AI 0.4.1 - Windows x64 portable edition\n\nExtract the complete folder, then double-click NeuroForge.exe.\nKeep NeuroForge.exe, _bootstrap and _internal together.\nNo external Python installation is required. CPU works without an NVIDIA GPU. CUDA uses the installed NVIDIA driver.\nUser data defaults to %LOCALAPPDATA%\\NeuroForgeAI\\data. Existing source workspaces are not automatically merged. Use Choose data folder after stopping the backend to select an existing runtime folder.\nRead docs/NeuroForge-User-Manual-zh-CN.html for complete Chinese instructions.\nThe bundled runtime omits C++ development headers, static .lib libraries and debug symbols. Python custom operators remain supported.\nUnsigned local development build. Windows 10/11 x64 and the packaged desktop launcher are the target.\n',encoding='utf-8')
report=dict(version='0.4.1',architecture='Windows x64',built=time.strftime('%Y-%m-%d %H:%M:%S'),bundled_python=sys.version,bundled_runtime='CPython + PyTorch CUDA + torchvision/timm/transformers/ONNX',entry='NeuroForge.exe',independent_of_system_python=True,development_components_omitted=['C++ headers','static .lib files','debug .pdb files'],executable_sha256=hashlib.sha256((TARGET/'NeuroForge.exe').read_bytes()).hexdigest())
(TARGET/'BUILD-MANIFEST.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
