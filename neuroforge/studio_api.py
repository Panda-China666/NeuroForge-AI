"""Studio 0.3 HTTP tools. Compute endpoints only enqueue isolated workers."""
import io,json,base64
from typing import Literal
import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter
from fastapi.responses import Response,FileResponse
from pydantic import BaseModel,ConfigDict,Field
from . import engine,graph_tools,figures,diagrams,pipeline,data,store,proxies
router=APIRouter(prefix='/api')
class Body(BaseModel):model_config=ConfigDict(extra='forbid')
class ImportCode(Body):
    source:str=Field(max_length=500000);input_shape:list[int];class_name:str|None=None;input_type:Literal['image','tokens','sequence']='image'
class Diff(Body):left:dict;right:dict
class Edit(Body):
    ir:dict;action:Literal['insert','connect'];node_id:str;op:str='Identity';config:dict=Field(default_factory=dict);target:str|None=None
class FigureCreate(Body):ir:dict;options:dict=Field(default_factory=dict)
class FigureDoc(Body):
    document:dict;format:Literal['svg','png','pdf','json']='svg';resolution:int=Field(2400,ge=800,le=6000);name:str=Field('Architecture figure',max_length=120);project_id:str|None=None
class Pipeline(Body):
    dataset_id:str;steps:list[dict]=Field(default_factory=list);seed:int=Field(42,ge=0,le=2**31-1);index:int=Field(0,ge=0);name:str=Field('Image pipeline',max_length=120)

@router.get('/algorithms')
def algorithms():return dict(proxies=proxies.REGISTRY,search=['random','evolution','enumerate','bayesian'])
@router.post('/code/import')
def code_import(p:ImportCode):return graph_tools.import_code(p.source,p.input_shape,p.class_name,p.input_type)
@router.post('/architectures/diff')
def architecture_diff(p:Diff):return graph_tools.diff(p.left,p.right)
@router.post('/graph/edit')
def edit(p:Edit):return graph_tools.edit(p.ir,p.action,p.node_id,p.op,p.config,p.target)
@router.post('/figures/create')
def figure_create(p:FigureCreate):
    engine.analyze(p.ir);doc=figures.create(p.ir,p.options);doc.update(source_graph_hash=engine.fingerprint(p.ir),mode=p.options.get('mode','2d'));return doc
@router.post('/figures/render')
def figure_render(p:FigureDoc):
    sc=figures.parse(p.document)
    if p.format=='json':return p.document
    content=sc.svg() if p.format=='svg' else sc.png(p.resolution) if p.format=='png' else sc.pdf()
    return Response(content,media_type={'svg':'image/svg+xml','png':'image/png','pdf':'application/pdf'}[p.format])
@router.post('/figures')
def figure_save(p:FigureDoc):
    figures.parse(p.document)
    if p.project_id:store.get('project',p.project_id)
    return store.put('figure',dict(name=p.name,document=p.document,project_id=p.project_id))
@router.get('/figures')
def figure_list():return store.all_of('figure')
@router.post('/pipelines/preview')
def pipeline_preview(p:Pipeline):
    x,y,m=data.load(store.DATA,p.dataset_id)
    if m.get('input_type'):raise ValueError('Image pipelines do not accept token IDs')
    pipeline.validate(p.steps,m['shape'])
    if p.index>=len(x):raise ValueError('Sample index out of range')
    before=x[p.index:p.index+1];after=pipeline.apply(before,p.steps,p.seed)
    def encode(v):
        image=(v[0].detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8');out=io.BytesIO();Image.fromarray(image.squeeze(-1) if image.shape[-1]==1 else image).save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
    return dict(before=encode(before),after=encode(after),shape=list(after.shape),min=after.min().item(),max=after.max().item(),mean=after.mean().item(),display='PNG preview clamped to 0..1; numeric statistics use the real tensor')
@router.post('/pipelines')
def pipeline_save(p:Pipeline):
    m=store.get('dataset',p.dataset_id);pipeline.validate(p.steps,m['shape']);return store.put('pipeline',p.model_dump())
@router.get('/pipelines')
def pipeline_list():return store.all_of('pipeline')
@router.get('/jobs/{ident}/tensors')
def tensors(ident:str,node:str):
    store.get('job',ident);path=store.ARTIFACTS/ident/'tensors.pt'
    if not path.exists():raise ValueError('No captured tensor file for this run')
    tensors=torch.load(path,map_location='cpu',weights_only=True)
    if node not in tensors:raise ValueError('Tensor is not in the full capture budget')
    out=io.BytesIO();np.savez_compressed(out,**{k:v.numpy() for k,v in tensors[node].items() if v is not None});return Response(out.getvalue(),media_type='application/octet-stream',headers={'Content-Disposition':'attachment; filename="tensor.npz"'})
@router.get('/jobs/{ident}/optimization')
def optimization_artifact(ident:str):
    store.get('job',ident);path=store.ARTIFACTS/ident/'optimization.zip'
    if not path.exists():raise ValueError('Optimization artifact is not ready')
    return FileResponse(path,filename='neuroforge-optimization-'+ident+'.zip')

@router.post('/projects/{ident}/pipeline')
def project_pipeline(ident:str,p:Pipeline):
    project=store.get('project',ident)
    if project.get('dataset_id')!=p.dataset_id:raise ValueError('Apply this dataset to the project first')
    pipeline.validate(p.steps,project['ir']['input_shape'])
    return store.update('project',ident,data_pipeline=p.steps)
