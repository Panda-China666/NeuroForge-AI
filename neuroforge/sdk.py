"""Dependency-free client for a running local NeuroForge server."""
import json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError

class NeuroForge:
    def __init__(self,url='http://127.0.0.1:8787',timeout=60):self.url=url.rstrip('/');self.timeout=timeout
    def request(self,path,method='GET',body=None,binary=False):
        req=Request(self.url+'/api'+path,data=json.dumps(body).encode() if body is not None else None,method=method,headers={'Content-Type':'application/json'})
        try:
            with urlopen(req,timeout=self.timeout) as response:raw=response.read()
        except HTTPError as exc:raise RuntimeError(exc.read().decode()) from exc
        return raw if binary else json.loads(raw)
    def system(self):return self.request('/system')
    def projects(self):return self.request('/projects')
    def datasets(self):return self.request('/datasets')
    def create_project(self,name,ir,dataset_id,search_space=None):return self.request('/projects','POST',dict(name=name,ir=ir,dataset_id=dataset_id,search_space=search_space))
    def analyze(self,ir):return self.request('/analyze','POST',{'ir':ir})
    def import_code(self,source,input_shape,**options):return self.request('/code/import','POST',dict(source=source,input_shape=input_shape,**options))
    def diff(self,left,right):return self.request('/architectures/diff','POST',dict(left=left,right=right))
    def submit(self,kind,project_id,**options):
        project=next(p for p in self.projects() if p['id']==project_id)
        payload=dict(project_id=project_id,dataset_id=project['dataset_id'],ir=project['ir'],data_pipeline=project.get('data_pipeline',[]))
        if kind=='search':payload['search_space']=project.get('search_space')
        payload.update(options);return self.request('/jobs/'+kind,'POST',payload)
    def train(self,project_id,**options):return self.submit('train',project_id,**options)
    def search(self,project_id,**options):return self.submit('search',project_id,**options)
    def debug(self,project_id,**options):return self.submit('debug',project_id,**options)
    def profile(self,project_id,**options):return self.submit('profile',project_id,**options)
    def job(self,ident):return self.request('/jobs/'+ident)
    def control(self,ident,action):return self.request('/jobs/'+ident+'/control','POST',{'action':action})
    def wait(self,ident,timeout=3600,poll=.5):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            j=self.job(ident)
            if j['status'] in ('completed','failed','cancelled','paused'):return j
            time.sleep(poll)
        raise TimeoutError(ident)
    def export(self,ident,path):
        path=Path(path);path.write_bytes(self.request('/export/job/'+ident,binary=True));return path

    def model_catalog(self):return self.request('/ecosystem/catalog')
    def model_assets(self):return self.request('/ecosystem/assets')
    def import_model(self,project_id,provider,model,pretrained=False,allow_download=False,**options):return self.request('/ecosystem/import','POST',dict(project_id=project_id,provider=provider,model=model,pretrained=pretrained,allow_download=allow_download,**options))
    def register_operator(self,name,source,input_shape,parameters=None,*,trusted_code=False):return self.request('/operators','POST',dict(name=name,source=source,input_shape=input_shape,parameters=parameters or {},trusted_code=trusted_code))
    def operators(self):return self.request('/operators')
    def preview_plan(self,project_id,plan):
        p=next(p for p in self.projects() if p['id']==project_id)
        return self.request('/copilot/preview','POST',dict(project_id=project_id,ir=p['ir'],search_space=p.get('search_space'),plan=plan))
    def apply_plan(self,plan_id,ir,search_space,*,confirm=False):return self.request('/copilot/plans/'+plan_id+'/apply','POST',dict(confirm=confirm,ir=ir,search_space=search_space))
    def optimization_report(self,baseline_id,variant_id,**options):return self.request('/copilot/optimization-report','POST',dict(baseline_id=baseline_id,variant_id=variant_id,**options))
