"""Product agents and trusted model/operator integrations."""
import asyncio,copy
from typing import Literal
from fastapi import APIRouter,Request,Response
from pydantic import BaseModel,ConfigDict,Field,SecretStr
from . import copilot,store,engine,worker,ecosystem,custom_ops
router=APIRouter(prefix='/api')
class Body(BaseModel):model_config=ConfigDict(extra='forbid')
class Settings(Body):endpoint:str=Field(max_length=500);model:str=Field(max_length=120);api_key:SecretStr;protocol:Literal['responses','chat']='responses'
class Draft(Body):project_id:str;ir:dict|None=None;search_space:dict|None=None
class Prompt(Draft):prompt:str=Field(min_length=1,max_length=6000);mode:Literal['architecture','optimization']='architecture'
class Preview(Draft):plan:copilot.Plan
class Approve(Body):confirm:Literal[True];ir:dict;search_space:dict
class Import(Body):
    project_id:str;provider:Literal['torchvision','timm','huggingface'];model:str=Field(max_length=180);pretrained:bool=False;allow_download:bool=False;seed:int=42;hf_config:dict|None=None
class Operator(Body):
    name:str=Field(min_length=1,max_length=100);description:str=Field('',max_length=1000);source:str=Field(max_length=50000);parameters:dict=Field(default_factory=dict);input_shape:list[int];trusted_code:Literal[True]
class Replace(Body):ir:dict;path:str;width:int=Field(32,ge=1,le=512)
class Scene(Body):project_id:str;document:dict

@router.get('/copilot/config')
def settings(request:Request):return copilot.configuration(request.cookies.get('nf_copilot'))
@router.post('/copilot/config')
def configure(p:Settings,request:Request,response:Response):
    token=copilot.configure(request.cookies.get('nf_copilot'),p.endpoint,p.model,p.api_key.get_secret_value(),p.protocol)
    response.set_cookie('nf_copilot',token,httponly=True,samesite='strict',max_age=8*3600,secure=request.url.scheme=='https')
    return copilot.configuration(token)
@router.delete('/copilot/config')
def forget(request:Request,response:Response):
    copilot.forget(request.cookies.get('nf_copilot'));response.delete_cookie('nf_copilot');return {'configured':False}
@router.post('/copilot/context')
def context(p:Draft):return copilot.context(store.get('project',p.project_id),p.ir,p.search_space)
@router.post('/copilot/plan')
async def generate(p:Prompt,request:Request):return await asyncio.to_thread(copilot.agent,store.get('project',p.project_id),request.cookies.get('nf_copilot'),p.prompt,p.mode,p.ir,p.search_space)
@router.post('/copilot/preview')
def preview(p:Preview):
    project=store.get('project',p.project_id);return copilot.save_preview(project,copilot.compile_plan(project,p.plan.model_dump(),p.ir,p.search_space))
@router.get('/copilot/plans')
def plans(project_id:str):return [p for p in store.all_of('agent_plan') if p['project_id']==project_id][:30]
@router.post('/copilot/plans/{ident}/apply')
def apply(ident:str,p:Approve):return copilot.apply_plan(ident,p.ir,p.search_space)
@router.post('/copilot/plans/{ident}/discard')
def discard(ident:str):
    with copilot.LOCK:
        p=store.get('agent_plan',ident)
        if p['status']!='preview':raise ValueError('Only pending previews can be discarded')
        return store.update('agent_plan',ident,status='discarded')
@router.get('/ecosystem/catalog')
def catalog():return ecosystem.catalog()
@router.get('/ecosystem/assets')
def assets():return store.all_of('model_asset')
@router.post('/ecosystem/import')
def model_import(p:Import):
    project=store.get('project',p.project_id);dataset=store.get('dataset',project['dataset_id'])
    return worker.submit('import_model',dict(**p.model_dump(exclude_none=True),input_shape=dataset['shape'],input_type=dataset.get('input_type','image'),classes=len(dataset['classes'])))
@router.post('/ecosystem/replacement')
def replace(p:Replace):
    ir=ecosystem.replacement(p.ir,p.path,p.width)
    from .graph_tools import diff
    return dict(ir=ir,analysis=engine.analyze(ir),diff=diff(p.ir,ir))
@router.get('/operators')
def operators():return dict(example=custom_ops.EXAMPLE,operators=store.all_of('custom_operator'))
@router.post('/operators')
def register(p:Operator):return worker.submit('register_operator',p.model_dump())
@router.post('/operators/{ident}/disable')
def disable(ident:str):
    import json
    from .asset_ops import metadata
    m=metadata('operator',ident);m['enabled']=False;folder=store.ROOT/'custom_operators'/ident;(folder/'metadata.json').write_text(json.dumps(m),encoding='utf-8')
    return store.update('custom_operator',ident,enabled=False)
@router.post('/cad/scenes')
def scene(p:Scene):
    project=store.get('project',p.project_id)
    if len(str(p.document))>500000:raise ValueError('Scene document exceeds the budget')
    return store.update('project',project['id'],cad_scene=p.document)

class Outcome(Body):
    baseline_id:str;variant_id:str;target_reduction:float=Field(.3,gt=0,lt=1);accuracy_tolerance:float=Field(.01,ge=0,le=.2)
@router.post('/copilot/optimization-report')
def outcome(p:Outcome):
    from .assistant_report import compare
    return compare(p.baseline_id,p.variant_id,p.target_reduction,p.accuracy_tolerance)
