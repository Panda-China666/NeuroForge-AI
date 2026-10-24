"""Private desktop lifecycle controls; never stop an unrelated local service."""
import hmac,os,threading
from fastapi import APIRouter,Request,HTTPException
from . import store,worker
router=APIRouter(prefix='/api/desktop')
SERVER=None
ACTIVE={'queued','running','pausing','cancelling'}

def identity():return dict(application='NeuroForge AI',version='0.4.1',desktop_instance=os.environ.get('NEUROFORGE_DESKTOP_INSTANCE'),data_root=str(store.ROOT))

@router.get('/identity')
def get_identity():return identity()

@router.post('/stop')
def stop(request:Request):
    token=os.environ.get('NEUROFORGE_DESKTOP_TOKEN','')
    if not token or not hmac.compare_digest(request.headers.get('X-NeuroForge-Desktop',''),token):raise HTTPException(403,'Only the owning desktop launcher can stop this runtime')
    if SERVER is None:raise HTTPException(409,'This process was not started by the desktop runtime')
    with worker.CONTROL_LOCK:
        active=[j['id'] for j in store.all_of('job') if j['status'] in ACTIVE]
        if active:raise HTTPException(409,'Active tasks are preserved. Pause or cancel them in Training Studio / Search Lab, then stop the backend.')
        worker.ACCEPTING_JOBS=False
        timer=threading.Timer(.4,lambda:setattr(SERVER,'should_exit',True));timer.daemon=True;timer.start()
    return {'stopping':True}
