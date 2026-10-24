"""Explicitly trusted Python operator SDK; execution happens only in workers."""
import hashlib,json,time,uuid
from torch import nn
import torch
from .asset_ops import metadata,asset_root

EXAMPLE='''import torch
from torch import nn

class Operator(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
    def forward(self, x):
        return x * torch.sigmoid(self.gain)
'''


def construct(source,parameters):
    namespace={'__name__':'neuroforge_user_operator'}
    exec(compile(source,'<trusted custom operator>','exec'),namespace)
    cls=namespace.get('Operator')
    if not isinstance(cls,type) or not issubclass(cls,nn.Module):raise ValueError('Define class Operator(torch.nn.Module)')
    return cls(**parameters)


def register(ident,p):
    from . import store
    if not p.get('trusted_code'):raise ValueError('Explicitly acknowledge trusted local code execution first')
    if len(p['source'])>50000:raise ValueError('Operator source exceeds 50KB')
    shape=p['input_shape'];parameters=p.get('parameters',{})
    if len(shape) not in (1,2,3) or any(type(v)!=int or not 1<=v<=1024 for v in shape) or __import__('math').prod(shape)>1000000:raise ValueError('Custom probe shape exceeds the local budget')
    torch.manual_seed(42);model=construct(p['source'],parameters).eval();x=torch.randn(2,*shape,requires_grad=True);out=model(x)
    if not isinstance(out,torch.Tensor) or out.shape[0]!=2 or not torch.isfinite(out).all() or out.numel()>4000000:raise ValueError('Operator must return a finite tensor preserving the batch dimension')
    if not out.requires_grad:raise ValueError('Operator must preserve differentiability')
    out.square().mean().backward();count=sum(v.numel() for v in model.parameters())
    if count>50000000:raise ValueError('Operator exceeds the 50M parameter budget')
    aid=uuid.uuid4().hex[:16];folder=asset_root()/'custom_operators'/aid;folder.mkdir(parents=True)
    version=1+max([m.get('version',1) for m in store.all_of('custom_operator') if m['name']==p['name']] or [0])
    m=dict(id=aid,name=p['name'],version=version,description=p.get('description',''),enabled=True,parameters=parameters,contracts={'':dict(input_shape=shape,output_shape=list(out.shape[1:]),params=count)},source_sha256=hashlib.sha256(p['source'].encode()).hexdigest(),created_at=time.time(),trust='user explicitly enabled local Python code')
    (folder/'operator.py').write_text(p['source'],encoding='utf-8');(folder/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8');store.put('custom_operator',m)
    store.update('job',ident,status='completed',progress=1,result=m,finished_at=time.time())


def load_operator(ident,parameters):
    m=metadata('operator',ident)
    if not m.get('enabled'):raise ValueError('This trusted operator has been disabled')
    if parameters!=m['parameters']:raise ValueError('Revalidate changed operator parameters as a new registry version')
    source=(asset_root()/'custom_operators'/ident/'operator.py').read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest()!=m['source_sha256']:raise ValueError('Operator source changed. Register and acknowledge a new version.')
    return construct(source,parameters)
