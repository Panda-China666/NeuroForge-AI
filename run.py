from pathlib import Path
import argparse,sys,os
if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser(description='NeuroForge local workstation')
    parser.add_argument('--port',type=int,default=int(os.environ.get('NEUROFORGE_PORT','8787')))
    args=parser.parse_args()
    if not 1024<=args.port<=65535:
        print('Choose a local port between 1024 and 65535.');sys.exit(2)
    os.environ['NEUROFORGE_PORT']=str(args.port)
    if os.environ.get('NEUROFORGE_DESKTOP_LOG'):
        log=Path(os.environ['NEUROFORGE_DESKTOP_LOG']);log.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=sys.stderr=log.open('a',encoding='utf-8',buffering=1)
    import uvicorn
    from neuroforge import desktop_runtime
    from neuroforge.store import ROOT
    (ROOT/'server.pid').write_text(str(os.getpid()),encoding='utf-8')
    server=uvicorn.Server(uvicorn.Config('neuroforge.app:app',host='127.0.0.1',port=args.port,workers=1,access_log=False))
    desktop_runtime.SERVER=server
    server.run()
