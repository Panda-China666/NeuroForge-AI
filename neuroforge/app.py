import io
import json
import math
import platform
import time
import zipfile
import os
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
import torch
from . import data,engine,store,worker,runtime,search_space,diagrams

app=FastAPI(title='NeuroForge AI',version='0.4.1',description='Visual neural architecture design and search')
STATIC=Path(__file__).resolve().parents[1]/'static'


@app.middleware('http')
async def local_guard(request:Request,call_next):
    # Local workstation service. Reject cross-origin browser mutation and DNS rebinding.
    host=request.headers.get('host','').split(':')[0]
    if host not in ('127.0.0.1','localhost','testserver'):
        return JSONResponse({'detail':'Local workstation access only'},status_code=403)
    origin=request.headers.get('origin')
    if origin and origin not in (f'http://127.0.0.1:{os.environ.get("NEUROFORGE_PORT","8787")}',f'http://localhost:{os.environ.get("NEUROFORGE_PORT","8787")}','http://testserver'):
        return JSONResponse({'detail':'Cross-origin access denied'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Cache-Control']='no-store'
    return response


from fastapi.exceptions import RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
    # Never reflect input values such as API credentials in HTTP validation errors.
    return JSONResponse({'detail':[{k:v for k,v in e.items() if k in ('loc','msg','type')} for e in exc.errors()]},status_code=422)


@app.exception_handler(ValueError)
async def value_error(request,exc):
    return JSONResponse({'detail':str(exc)},status_code=400)


@app.exception_handler(KeyError)
async def key_error(request,exc):
    return JSONResponse({'detail':str(exc)},status_code=404)


class Body(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Project(Body):
    name:str=Field(default='Untitled research',min_length=1,max_length=120)
    ir:dict|None=None
    dataset_id:str|None=None
    layout:dict=Field(default_factory=dict)
    search_space:dict|None=None


class Analysis(Body):
    ir:dict


class Task(Body):
    project_id:str
    dataset_id:str
    ir:dict
    seed:int=Field(42,ge=0,le=2**31-1)
    count:int=Field(12,ge=2,le=256)
    strategy:Literal['random','evolution','bayesian','enumerate']='random'
    channels:list[int]=Field(default_factory=lambda:[8,16,24,32],min_length=1,max_length=8)
    depth:list[int]=Field(default_factory=lambda:[1,2,3],min_length=1,max_length=4)
    residual:list[bool]=Field(default_factory=lambda:[True,False],min_length=1,max_length=2)
    max_params:int=Field(1_000_000,ge=100,le=250_000_000)
    max_macs:int=Field(100_000_000,ge=1000,le=10_000_000_000)
    epochs:int=Field(5,ge=1,le=200)
    batch_size:int=Field(32,ge=2,le=128)
    lr:float=Field(.001,ge=.000001,le=.1)
    weight_decay:float=Field(.0001,ge=0,le=1)
    device:str=Field('auto',max_length=24)
    search_space:dict|None=None
    budget_seconds:float|None=Field(None,ge=1,le=86400)
    proxy_methods:list[str]=Field(default_factory=lambda:['snip','synflow','naswot'],min_length=1,max_length=7)
    inspect_batch:int=Field(4,ge=2,le=16)
    benchmark_repeats:int=Field(10,ge=2,le=30)
    checkpoint_job:str|None=None
    hardware_aware:bool=False
    max_latency_ms:float|None=Field(None,gt=0,le=100000)
    max_measured_memory_mb:float|None=Field(None,gt=0,le=100000)
    min_throughput:float|None=Field(None,gt=0,le=10000000)
    amp:bool=False
    gradient_accumulation:int=Field(1,ge=1,le=32)
    early_stopping_patience:int=Field(0,ge=0,le=50)
    early_stopping_delta:float=Field(0,ge=0,le=1)
    auto_batch:bool=False
    data_pipeline:list[dict]=Field(default_factory=list,max_length=16)
    optimization:Literal['prune','dynamic_int8','onnx']='prune'
    prune_amount:float=Field(.3,ge=.01,le=.95)
    source_candidate:str|None=None
    source_job:str|None=None


class Split(Body):
    seed:int=Field(42,ge=0,le=2**31-1)
    val:float=Field(.15,ge=.05,le=.4)
    test:float=Field(.15,ge=.05,le=.4)


class Action(Body):
    action:Literal['pause','resume','cancel','rerun']


class Space(Body):
    ir:dict
    search_space:dict


class Instantiate(Space):
    genes:dict


class Package(Body):
    ir:dict
    selected:list[str]
    name:str=Field(min_length=1,max_length=80)
    description:str=Field('',max_length=1000)
    category:str=Field('My Blocks',max_length=80)
    bindings:dict=Field(default_factory=dict)
    source_project:str|None=None


class Diagram(Body):
    ir:dict
    mode:Literal['2d','3d']='2d'
    format:Literal['svg','png','pdf']='svg'
    style:Literal['clean','minimal','color','mono','poster']='clean'
    theme:Literal['product','white','dark','cover','investor']='product'
    orientation:Literal['horizontal','vertical']='horizontal'
    resolution:int=Field(2400,ge=800,le=6000)
    show_shapes:bool=True
    show_labels:bool=True
    show_params:bool=True
    color_coding:bool=True
    legend:bool=True
    font_size:int=Field(12,ge=9,le=18)
    line_width:float=Field(1.5,ge=.5,le=4)
    radius:int=Field(8,ge=0,le=20)
    title:str=Field('',max_length=120)
    subtitle:str=Field('',max_length=180)
    paper_size:Literal['auto','a4','letter','poster']='auto'
    camera_angle:int=Field(28,ge=5,le=60)
    depth:int=Field(22,ge=4,le=50)
    shadow:float=Field(.6,ge=0,le=1)
    layout:Literal['compact','cinematic']='compact'


def initialize():
    worker.recover()
    if not store.all_of('dataset'):
        store.put('dataset',data.builtin(store.DATA))
    if not any(d.get('input_type')=='tokens' for d in store.all_of('dataset')):
        store.put('dataset',data.builtin_tokens(store.DATA))
    if not store.all_of('project'):
        dataset=next(d for d in store.all_of('dataset') if d.get('input_type')!='tokens')
        store.put('project',dict(name='Edge Vision Research',ir=engine.template(),dataset_id=dataset['id'],revision=1))
    if not store.all_of('module'):
        image=engine.template(channels=16,depth=1)
        _,node=search_space.package_module(image,['block1'],'Residual Cell',bindings={'channels':[{'node':'block1','parameter':'out_channels'}]})
        enc=dict(id='encoder_cell',op='Module',name='Transformer Cell',inputs=['input'],config={'num_heads':4,'ffn_ratio':4},
                 module=dict(schema_version=1,input_shape=[65,64],input_ports=[{'id':'input','shape':[65,64]}],nodes=[dict(id='encoder',op='TransformerEncoder',inputs=['input'],config={'num_heads':4,'ffn_ratio':4,'dropout':.1})],output='encoder'),
                 bindings={'num_heads':[{'node':'encoder','parameter':'num_heads'}],'ffn_ratio':[{'node':'encoder','parameter':'ffn_ratio'}]})
        for n,category in [(node,'Vision Blocks'),(enc,'Transformer Blocks')]:
            spec=engine.analyze(n['module'],_level=1)
            store.put('module',dict(name=n['name'],node=n,category=category,version=1,description='Built-in reusable block',builtin=True,favorite=False,
                      params=spec['params'],internal_nodes=len(spec['nodes']),input_ports=n['module']['input_ports'],output_shape=spec['output_shape'],parameter_schema=n['bindings'],compatibility='CPU / CUDA'))


initialize()


@app.get('/api/system')
def system():
    return dict(version='0.4.1',torch=str(torch.__version__),python=platform.python_version(),
                cuda=torch.cuda.is_available(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                devices=runtime.devices(),cuda_version=torch.version.cuda,ops=engine.OPS,worker='isolated process per compute task',data_root=str(store.ROOT))


@app.get('/api/projects')
def projects():
    return store.all_of('project')


@app.post('/api/projects')
def create_project(p:Project):
    m=None
    if p.dataset_id:
        m=store.get('dataset',p.dataset_id)
    if p.ir:
        ir=p.ir
    elif m and m.get('input_type')=='tokens':
        ir=engine.transformer_template(vocab_size=m['vocab_size'],length=m['shape'][0],classes=len(m['classes']))
    else:
        ir=engine.template(classes=len(m['classes']) if m else 3,size=m['shape'][1] if m else 32)
    engine.analyze(ir)
    space=p.search_space or {'schema_version':1,'variables':[],'nodes':{},'global_constraints':{},'local_constraints':[]}
    search_space.validate(space,ir)
    return store.put('project',dict(name=p.name,ir=ir,dataset_id=p.dataset_id,layout=check_layout(p.layout),search_space=space,revision=1))


def check_layout(layout):
    if len(layout)>257:
        raise ValueError('Layout exceeds 257 nodes')
    for key,pos in layout.items():
        if not isinstance(pos,dict) or set(pos)!={'x','y'} or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>100000 for v in pos.values()):
            raise ValueError('Layout coordinates must be finite x/y values')
    return layout


@app.put('/api/projects/{ident}')
def save_project(ident:str,p:Project):
    old=store.get('project',ident)
    ir=p.ir or old['ir']
    engine.analyze(ir)
    if p.dataset_id:
        store.get('dataset',p.dataset_id)
    layout=check_layout(p.layout)
    space=p.search_space if p.search_space is not None else old.get('search_space',{'schema_version':1,'variables':[],'nodes':{},'global_constraints':{},'local_constraints':[]})
    search_space.validate(space,ir)
    store.put('version',dict(project_id=ident,revision=old['revision'],ir=old['ir'],layout=old.get('layout',{}),search_space=old.get('search_space',{}),dataset_id=old.get('dataset_id')))
    return store.update('project',ident,name=p.name,ir=ir,dataset_id=p.dataset_id,layout=layout,search_space=space,revision=old['revision']+1)


@app.get('/api/projects/{ident}/versions')
def versions(ident:str):
    return [v for v in store.all_of('version') if v['project_id']==ident]


@app.post('/api/analyze')
def analyze(p:Analysis):
    return engine.analyze(p.ir)


@app.post('/api/code')
def code(p:Analysis):
    engine.analyze(p.ir)
    return {'code':engine.export_code(p.ir)}


@app.get('/api/datasets')
def datasets():
    return store.all_of('dataset')


@app.post('/api/datasets/tokens')
def token_upload(p:dict):
    return store.put('dataset',data.tokens(store.DATA,p.get('records'),str(p.get('name','Token dataset'))[:120],int(p.get('seed',42)),int(p.get('vocab_size',128))))


@app.post('/api/datasets')
async def upload(file:UploadFile=File(...),size:int=Form(32),seed:int=Form(42)):
    if size not in (16,32,64,128):
        raise ValueError('Image size must be 16,32,64 or 128')
    content=await file.read(data.MAX_BYTES+1)
    try:
        return store.put('dataset',data.import_zip(store.DATA,content,(file.filename or 'dataset')[:120],size,seed))
    except zipfile.BadZipFile:
        raise ValueError('Invalid ZIP archive')


@app.post('/api/datasets/{ident}/split')
def resplit(ident:str,p:Split):
    store.get('dataset',ident)
    x,y,m=data.load(store.DATA,ident)
    if m.get('input_type')=='tokens':raise ValueError('Token dataset re-split is not yet available; import with a new seed.')
    arr=(x.permute(0,2,3,1).numpy()*255).round().astype('uint8')
    return store.put('dataset',data.finalize(store.DATA,list(arr),y.tolist(),m['classes'],m['name'],p.seed,
                     m['issues'],m['source'],p.val,p.test,parent=ident))


@app.get('/api/datasets/{ident}/preview/{index}')
def preview(ident:str,index:int):
    store.get('dataset',ident)
    x,y,m=data.load(store.DATA,ident)
    if index<0 or index>=len(y):
        raise HTTPException(404,'Image not found')
    if m.get('input_type')=='tokens':return dict(tokens=x[index].tolist(),label=m['classes'][int(y[index])])
    from PIL import Image
    arr=(x[index].permute(1,2,0).numpy()*255).round().astype('uint8')
    stream=io.BytesIO()
    Image.fromarray(arr).save(stream,format='PNG')
    return Response(stream.getvalue(),media_type='image/png')


@app.post('/api/jobs/{kind}')
def start_job(kind:Literal['proxy','search','train','debug','profile','optimize'],p:Task):
    validate_task(kind,p)
    return worker.submit(kind,p.model_dump())


def validate_task(kind,p):
    from . import pipeline,proxies
    if any(n not in proxies.REGISTRY for n in p.proxy_methods):raise ValueError('Unknown proxy algorithm')
    pipeline.validate(p.data_pipeline,p.ir['input_shape'])
    store.get('project',p.project_id)
    m=store.get('dataset',p.dataset_id)
    spec=engine.analyze(p.ir)
    if p.ir['input_shape'] != m['shape'] or spec['output_shape'] != [len(m['classes'])]:
        raise ValueError('Dataset/model mismatch: apply the dataset to the model')
    execution_batch=(2 if p.auto_batch else p.batch_size) if kind=='train' else p.inspect_batch if kind in ('debug','profile','optimize') else 16
    if spec['activation_memory_mb'] * execution_batch > 256:
        raise ValueError('Local execution activation budget exceeds 256 MB; reduce batch size or model size')
    if any(v not in (4,8,12,16,24,32,48,64) for v in p.channels) or any(v not in (1,2,3,4) for v in p.depth):
        raise ValueError('Unsupported search domain')
    if p.search_space:search_space.validate(p.search_space,p.ir)
    return p


@app.get('/api/jobs')
def jobs():
    return store.all_of('job')


@app.get('/api/jobs/{ident}')
def job(ident:str):
    return store.get('job',ident)


@app.post('/api/jobs/{ident}/control')
def control(ident:str,p:Action):
    if p.action=='rerun':
        old=store.get('job',ident)
        return worker.submit(old['kind'],old['payload'])
    return worker.control(ident,p.action)


@app.get('/api/templates/{kind}')
def get_template(kind:Literal['cnn','residual','vit','transformer'],classes:int=3,size:int=32):
    if classes<2 or classes>100 or size not in (16,32,64,128):raise ValueError('Invalid template dimensions')
    if kind=='vit':return engine.vit_template(classes=classes,size=size)
    if kind=='transformer':return engine.transformer_template(classes=classes)
    return engine.template(classes=classes,size=size,residual=kind=='residual')


@app.post('/api/search-space/summary')
def summarize(p:Space):return search_space.summary(p.search_space,p.ir)


@app.post('/api/search-space/instantiate')
def instantiate(p:Instantiate):
    ir=search_space.instantiate(p.ir,p.search_space,p.genes)
    return dict(ir=ir,analysis=engine.analyze(ir))


@app.post('/api/modules/package')
def package(p:Package):
    ir,node=search_space.package_module(p.ir,p.selected,p.name,bindings=p.bindings)
    spec=engine.analyze(node['module'],_level=1)
    saved=store.put('module',dict(name=p.name,description=p.description,category=p.category,version=1,node=node,
                    params=spec['params'],internal_nodes=spec['expanded_nodes'],input_ports=node['module']['input_ports'],
                    output_shape=spec['output_shape'],parameter_schema=node['bindings'],source_project=p.source_project,
                    compatibility='IR v1 / PyTorch / CPU / CUDA',favorite=False,last_used=None))
    return dict(ir=ir,node=node,module=saved)


@app.get('/api/modules')
def modules():return store.all_of('module')


@app.post('/api/modules/{ident}/favorite')
def favorite(ident:str):
    m=store.get('module',ident)
    return store.update('module',ident,favorite=not m.get('favorite',False))


@app.post('/api/modules/{ident}/used')
def module_used(ident:str):return store.update('module',ident,last_used=time.time())


@app.get('/api/modules/{ident}/export')
def export_module(ident:str):return store.get('module',ident)


@app.post('/api/modules/import')
def import_module(p:dict):
    node=p.get('node',{});spec=engine.analyze(node.get('module',{}),_level=1)
    return store.put('module',dict(name=str(p.get('name','Imported module'))[:80],description=str(p.get('description',''))[:1000],
              category='Imported',version=p.get('version',1),node=node,params=spec['params'],internal_nodes=spec['expanded_nodes'],
              input_ports=node['module'].get('input_ports',[{'id':'input','shape':node['module']['input_shape']}]),
              output_shape=spec['output_shape'],favorite=False,compatibility='IR v1 / PyTorch',parameter_schema=node.get('bindings',{})))


@app.post('/api/diagrams')
def export_diagram(p:Diagram):
    content,mime=diagrams.render(p.ir,p.model_dump(exclude={'ir'}))
    return Response(content,media_type=mime,headers={'Content-Disposition':f'inline; filename="neuroforge-{p.mode}.{p.format}"'})


def torch_requirements(version):
    tag=version.split('+')[-1] if '+' in version else ''
    index='--extra-index-url https://download.pytorch.org/whl/'+tag+'\n' if tag=='cpu' or tag.startswith('cu') else ''
    return index+'torch=='+version+'\n'


from .asset_ops import asset_export_files


def archive_response(files,name):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for path,content in files.items():
            if isinstance(content,Path):
                z.write(content,path)
            else:
                z.writestr(path,content)
    return Response(stream.getvalue(),media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="{name}.zip"'})


@app.get('/api/export/project/{ident}')
def export_project(ident:str):
    p=store.get('project',ident)
    files={'project.json':json.dumps(p,ensure_ascii=False,indent=2),
                             'architecture.json':json.dumps(p['ir'],indent=2),'model.py':engine.export_code(p['ir']),
                             'requirements.txt':torch_requirements(str(torch.__version__))}
    if p.get('dataset_id'):
        files['dataset_manifest.json']=json.dumps(store.get('dataset',p['dataset_id']),ensure_ascii=False,indent=2)
    files.update(asset_export_files(p['ir']));return archive_response(files,'neuroforge-project-'+ident)


@app.get('/api/export/job/{ident}')
def export_job(ident:str):
    j=store.get('job',ident)
    p=j['payload']
    m=store.get('dataset',p['dataset_id'])
    files={'experiment.json':json.dumps(j,ensure_ascii=False,indent=2),
           'architecture.json':json.dumps(p['ir'],indent=2),'model.py':engine.export_code(p['ir']),
           'dataset_manifest.json':json.dumps(m,ensure_ascii=False,indent=2),
           'requirements.txt':torch_requirements(j.get('environment',{}).get('torch',str(torch.__version__))),
           'README.txt':'NeuroForge experiment artifact\nLoad best.pt using model.build_model().load_state_dict(torch.load("best.pt", weights_only=True)).\ncheckpoint.pt includes optimizer/scheduler/RNG and latest epoch.\nDataset pixels are not embedded; dataset_manifest.json identifies the version and split indices.\nSearch exports include all candidate IRs in experiment.json; model.py represents the submitted seed architecture.\n'}
    for name in ('best.pt','checkpoint.pt'):
        path=store.ARTIFACTS/ident/name
        if path.exists():
            files[name]=path
    if j['kind']=='train':
        files['reproduce.py']=Path(__file__).with_name('reproduce_template.py')
        files['pipeline.py']=Path(__file__).with_name('pipeline.py')
        files['requirements.txt']+='numpy>=1.26,<3\n'
        files['README.txt']+='\nReproduce from scratch:\npython reproduce.py --dataset /path/to/runtime/datasets/'+p['dataset_id']+'/images.npz --output rerun\nThe script validates the original dataset hash and reuses split indices and the full training protocol. CPU reruns require matching environment for exact numerical reproducibility.\n'
    files.update(asset_export_files(p['ir']));return archive_response(files,'neuroforge-experiment-'+ident)


@app.get('/')
def index():
    return FileResponse(STATIC/'index.html')


from .desktop_runtime import router as desktop_router
app.include_router(desktop_router)
from .agent_api import router as agent_router
app.include_router(agent_router)
from .studio_api import router as studio_router
app.include_router(studio_router)

app.mount('/static',StaticFiles(directory=STATIC),name='static')
