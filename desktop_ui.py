"""NeuroForge local desktop shell. No external Python installation is used."""
import argparse,contextlib,json,os,queue,secrets,socket,subprocess,sys,tempfile,threading,time,traceback,urllib.request,urllib.error,uuid,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ACTIVE={'queued','running','pausing','cancelling'}

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def options(argv=None):
    parser=argparse.ArgumentParser(description='NeuroForge Windows portable desktop')
    default=Path(os.environ.get('LOCALAPPDATA',Path.home()))/'NeuroForgeAI'/'data';settings=default.parent/'desktop-settings.json'
    if settings.exists():
        try:default=Path(json.loads(settings.read_text(encoding='utf-8'))['data_root'])
        except (OSError,ValueError,KeyError):pass
    parser.add_argument('--data-dir',type=Path,default=default);parser.add_argument('--port',type=int,default=8787);parser.add_argument('--no-browser',action='store_true');parser.add_argument('--smoke-test',type=Path)
    o=parser.parse_args(argv)
    if not 1024<=o.port<=65535:parser.error('Choose a port between 1024 and 65535')
    o.settings=settings;o.explicit_data='--data-dir' in (argv if argv is not None else sys.argv[1:]);o.data_dir=o.data_dir.resolve();return o

@contextlib.contextmanager
def startup_lock(port):
    import msvcrt
    lock=Path(tempfile.gettempdir())/('NeuroForge-startup-'+str(port)+'.lock')
    with lock.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();until=time.monotonic()+160
        while True:
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if time.monotonic()>until:raise TimeoutError('Another launcher is still starting this port. Wait and retry.')
                time.sleep(.3)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

class Backend:
    def __init__(self,o):self.o=o;self.info=None;self.child=None;self.token=None;self.instance=None;self.started_new=False
    @property
    def url(self):return f'http://127.0.0.1:{self.o.port}'
    @property
    def record(self):return self.o.data_dir/f'desktop-session-{self.o.port}.json'
    @property
    def log(self):return self.o.data_dir/'logs'/'server.log'
    @property
    def can_stop(self):return bool(self.token)
    def occupied(self):
        try:
            with socket.create_connection(('127.0.0.1',self.o.port),.5):return True
        except OSError:return False
    def request(self,path,method='GET',body=None,auth=False,timeout=10):
        headers={'Content-Type':'application/json'}
        if auth:headers['X-NeuroForge-Desktop']=self.token or ''
        req=urllib.request.Request(self.url+'/api'+path,data=json.dumps(body).encode() if body is not None else None,headers=headers,method=method)
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            try:message=json.load(exc).get('detail',str(exc))
            except (ValueError,AttributeError):message='Backend returned HTTP '+str(exc.code)
            raise RuntimeError(str(message)) from None
    def connect(self):
        identity={}
        try:
            identity=self.request('/desktop/identity',timeout=2)
            if identity.get('application')!='NeuroForge AI':return False
        except (OSError,ValueError,RuntimeError):pass
        try:
            info=self.request('/system',timeout=10)
            if not all(k in info for k in ('ops','torch','data_root')):return False
            self.info=info;self.o.data_dir=Path(info['data_root'])
            if identity.get('desktop_instance') and self.record.exists():
                saved=json.loads(self.record.read_text(encoding='utf-8'))
                if saved.get('instance')==identity['desktop_instance']:self.token=saved.get('token');self.instance=saved['instance']
            return True
        except (OSError,ValueError,RuntimeError):self.info=None;return False
    def start(self,report=lambda s:None):
        with startup_lock(self.o.port):
            if self.occupied():
                if not self.connect():raise RuntimeError(f'Port {self.o.port} is occupied by another service. No process was stopped. Use --port 8877 or close the other service yourself.')
                report('Connected to your desktop backend.' if self.can_stop else 'Connected to an existing backend. It will be left running.');return
            python=ROOT/'_internal'/'python'/'python.exe';app=ROOT/'_internal'/'app'
            if not python.exists() or not (app/'run.py').exists():raise RuntimeError('Extract the complete folder, including _internal. Do not run inside a ZIP preview.')
            self.log.parent.mkdir(parents=True,exist_ok=True);self.token=secrets.token_urlsafe(32);self.instance=uuid.uuid4().hex
            env=dict(os.environ)
            for key in ('PYTHONPATH','PYTHONHOME','NEUROFORGE_INLINE_WORKER'):env.pop(key,None)
            env.update(NEUROFORGE_DATA=str(self.o.data_dir),NEUROFORGE_PORT=str(self.o.port),NEUROFORGE_DESKTOP_TOKEN=self.token,NEUROFORGE_DESKTOP_INSTANCE=self.instance,NEUROFORGE_DESKTOP_LOG=str(self.log),CUBLAS_WORKSPACE_CONFIG=':4096:8',PATH=str(python.parent)+';'+str(python.parent/'DLLs')+';'+os.environ.get('SystemRoot',r'C:\Windows')+r'\System32;'+os.environ.get('SystemRoot',r'C:\Windows'))
            self.child=subprocess.Popen([str(python),'-u','-X','utf8',str(app/'run.py'),'--port',str(self.o.port)],cwd=app,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            self.started_new=True;atomic_json(self.record,dict(instance=self.instance,token=self.token,pid=self.child.pid,port=self.o.port))
            report('Loading bundled PyTorch. First start can take a minute...');until=time.monotonic()+150
            while time.monotonic()<until:
                if self.child.poll() is not None:raise RuntimeError(f'Backend exited ({self.child.returncode}). Inspect {self.log}')
                try:
                    identity=self.request('/desktop/identity',timeout=1.2)
                    if identity.get('desktop_instance')==self.instance:self.info=self.request('/system');report('Your workspace is ready.');return
                except (OSError,ValueError,RuntimeError):pass
                time.sleep(.6)
            raise TimeoutError('Startup is taking longer than expected. Backend was not forcibly stopped. Inspect '+str(self.log))
    def stop(self):
        if not self.can_stop:raise RuntimeError('This launcher does not own the connected service. It will not stop it.')
        self.request('/desktop/stop','POST',{},True)
        until=time.monotonic()+15
        while self.occupied() and time.monotonic()<until:time.sleep(.15)
        if self.occupied():raise RuntimeError('Graceful shutdown is still in progress. No process was killed.')
        self.token=None;self.info=None
    def wait_job(self,ident):
        until=time.monotonic()+180
        while time.monotonic()<until:
            j=self.request('/jobs/'+ident,timeout=15)
            if j['status'] in ('completed','failed','cancelled','paused'):
                if j['status']!='completed':raise RuntimeError(j.get('error','Worker did not complete'))
                return j
            time.sleep(.4)
        raise TimeoutError('Worker test timed out')

def smoke(o):
    if not o.explicit_data:raise RuntimeError('Smoke tests require --data-dir to protect the real workspace.')
    b=Backend(o)
    if b.occupied():raise RuntimeError('Smoke-test port is occupied. Choose an unused port.')
    report=dict(application='NeuroForge AI',executable=str(ROOT/'NeuroForge.exe'),started_at=time.strftime('%Y-%m-%d %H:%M:%S'),data_root=str(o.data_dir),embedded_python=sys.executable,sys_path=sys.path,ok=False)
    try:
        b.start();report['system']=b.info;project=b.request('/projects')[0]
        body=dict(project_id=project['id'],dataset_id=project['dataset_id'],ir=project['ir'],device='cuda:0' if b.info['cuda'] else 'cpu',proxy_methods=['gradnorm','snip'],epochs=1,batch_size=8,seed=42)
        report['proxy']=b.wait_job(b.request('/jobs/proxy','POST',body)['id']);report['training']=b.wait_job(b.request('/jobs/train','POST',body)['id'])
        other=Backend(o);other.start();report['reconnect_can_stop']=other.can_stop;other.stop();report['stopped']=not other.occupied();report['ok']=True
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc()
        try:
            if b.can_stop:b.stop()
        except Exception:pass
    atomic_json(o.smoke_test,report);return 0 if report['ok'] else 1

def gui(o):
    import tkinter as tk
    from tkinter import ttk,messagebox,filedialog
    root=tk.Tk();root.title('NeuroForge AI · Desktop');root.geometry('760x470');root.minsize(700,450);root.configure(bg='#f8f9fa')
    style=ttk.Style();style.theme_use('clam');style.configure('TButton',padding=(12,9),font=('Segoe UI',10));style.configure('TFrame',background='#f8f9fa');style.configure('TLabel',background='#f8f9fa',foreground='#526459',font=('Segoe UI',10))
    frame=ttk.Frame(root,padding=30);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='neuroforge AI',font=('Segoe UI',27,'bold'),foreground='#25392c').pack(anchor='w');ttk.Label(frame,text='Design visually. Explore intelligently. Train efficiently.').pack(anchor='w',pady=(5,24))
    status=tk.StringVar(value='Preparing your local workspace...');details=tk.StringVar(value='Local interface: http://127.0.0.1:'+str(o.port))
    ttk.Label(frame,textvariable=status,font=('Segoe UI',12,'bold'),wraplength=680).pack(anchor='w');ttk.Label(frame,textvariable=details,wraplength=680).pack(anchor='w',pady=(14,20))
    row=ttk.Frame(frame);row.pack(fill='x');row2=ttk.Frame(frame);row2.pack(fill='x',pady=(14,0))
    state={'backend':Backend(o),'busy':False,'close_after':False};events=queue.Queue();buttons={}
    def refresh():
        b=state['backend'];busy=state['busy']
        for key,enabled in {'start':not busy and b.info is None,'open':not busy and b.info is not None,'stop':not busy and b.info is not None and b.can_stop,'data':not busy,'choose':not busy and b.info is None}.items():buttons[key].configure(state='normal' if enabled else 'disabled')
    def run(action,kind):
        if state['busy']:return
        state['busy']=True;refresh()
        def work():
            try:action();events.put(('done',kind))
            except Exception as exc:events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()
    def start():run(lambda:state['backend'].start(lambda s:events.put(('status',s))),'start')
    def stop():run(state['backend'].stop,'stop')
    def open_data():
        b=state['backend'];path=Path(b.info['data_root']) if b.info else o.data_dir;path.mkdir(parents=True,exist_ok=True);os.startfile(str(path))
    def choose():
        selected=filedialog.askdirectory(title='Choose an existing runtime folder or an empty workspace folder')
        if selected:o.data_dir=Path(selected);atomic_json(o.settings,dict(data_root=selected));state['backend']=Backend(o);status.set('Data folder selected. Click Start backend.');details.set(selected)
    def manual():
        path=ROOT/'docs'/'NeuroForge-User-Manual-zh-CN.html'
        if path.exists():os.startfile(str(path))
        else:messagebox.showinfo('User manual','The manual is in the docs folder.')
    for key,text,command in [('start','Start backend',start),('open','Open workspace',lambda:webbrowser.open(state['backend'].url)),('stop','Stop backend',stop),('data','Data folder',open_data)]:
        button=ttk.Button(row,text=text,command=command);button.pack(side='left',padx=(0,9));buttons[key]=button
    buttons['choose']=ttk.Button(row2,text='Choose data folder',command=choose);buttons['choose'].pack(side='left',padx=(0,9));ttk.Button(row2,text='User manual',command=manual).pack(side='left')
    ttk.Label(frame,text='Local execution · CPU / CUDA · v0.4.1',foreground='#97a198').pack(side='bottom',anchor='w',pady=(25,0))
    def poll():
        while not events.empty():
            kind,value=events.get();b=state['backend']
            if kind=='status':status.set(value)
            elif kind=='error':state['busy']=False;state['close_after']=False;status.set('The backend was kept safe.');details.set(value);messagebox.showinfo('NeuroForge',value);refresh()
            else:
                state['busy']=False;refresh()
                if value=='start':
                    status.set('Your workspace is ready.' if b.can_stop else 'Connected to an existing workspace.')
                    details.set('Address: '+b.url+'\nDevice: '+str(b.info['device'])+'\nPyTorch: '+str(b.info['torch'])+'    Python: '+str(b.info['python'])+'\nData: '+str(b.info['data_root']))
                    if not o.no_browser:webbrowser.open(b.url)
                else:
                    status.set('Backend stopped safely.');details.set('Projects and checkpoints remain in:\n'+str(o.data_dir))
                    if state['close_after']:root.destroy();return
        root.after(120,poll)
    def close():
        b=state['backend']
        if state['busy']:return
        if not b.info or not b.can_stop:root.destroy();return
        answer=messagebox.askyesnocancel('Close NeuroForge','Stop the backend before closing?\n\nYes: stop safely if no tasks are active.\nNo: keep tasks running in the background.\nCancel: stay here.')
        if answer is None:return
        if answer:state['close_after']=True;stop()
        else:root.destroy()
    root.protocol('WM_DELETE_WINDOW',close);refresh();root.after(120,poll);root.after(200,start);root.mainloop()

if __name__=='__main__':
    try:
        opt=options()
        if opt.smoke_test:raise SystemExit(smoke(opt))
        gui(opt)
    except Exception as exc:
        if '--smoke-test' in sys.argv:
            try:atomic_json(sys.argv[sys.argv.index('--smoke-test')+1],dict(ok=False,error=str(exc),traceback=traceback.format_exc()))
            except Exception:pass
        else:
            import tkinter.messagebox as box;box.showerror('NeuroForge startup',str(exc))
        raise SystemExit(1)
