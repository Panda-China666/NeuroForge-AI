import copy,io,json
import pytest,torch
from fastapi.testclient import TestClient
from neuroforge import engine,graph_tools,inspection,pipeline,figures,search_algorithms,proxies,store
from neuroforge.app import app
from test_api import payload,wait,submit

CODE='''import torch
from torch import nn
class Net(nn.Module):
    def __init__(self,channels=8):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,channels,3,padding=1,bias=False),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Linear(channels,3,bias=False)
    def forward(self,x):
        x=self.features(x)
        return self.head(x)
'''


def test_code_import_matches_real_torch_and_rejects_semantics():
    result=graph_tools.import_code(CODE,[3,32,32]);model=engine.GraphModel(result['ir'])
    namespace={};exec(CODE,namespace);source=namespace['Net']()
    for name,index in [('features_0',0)]:model.layers[name].load_state_dict(source.features[index].state_dict())
    model.layers['head'].load_state_dict(source.head.state_dict());x=torch.rand(2,3,32,32)
    assert torch.allclose(source(x),model(x))
    assert sum(p.numel() for p in model.parameters())==result['analysis']['params']
    with pytest.raises(ValueError,match='Line'):graph_tools.import_code(CODE.replace('return self.head(x)','if x.sum()>0:\n            return self.head(x)'),[3,32,32])
    with pytest.raises(ValueError,match='dilation'):graph_tools.import_code(CODE.replace('bias=False)','bias=False,dilation=2)',1),[3,32,32])
    assert graph_tools.import_code(engine.export_code(result['ir']),[3,32,32])['ir']==result['ir']
    # Top-level source is parsed as data and is never executed.
    assert graph_tools.import_code('raise RuntimeError("must never run")\n'+CODE,[3,32,32])['ir']


def test_inspection_real_values_and_gradients():
    ir=engine.template(channels=4,depth=1);x=torch.rand(4,3,32,32);y=torch.tensor([0,1,2,0])
    report=inspection.run(ir,x,y,'cpu',capture=True,repeats=2)
    n=next(n for n in report['nodes'] if n['id']=='block1')
    assert n['activation']['shape']==[4,4,32,32] and n['gradient']['l2']>0
    assert sum(n['activation']['histogram'])==4*4*32*32
    assert report['benchmark']['latency_ms']>0
    assert report['weights'][0]['gradient_norm']>0
    prof=inspection.run(ir,x,y,'cpu',capture=False,repeats=2)
    assert all(n['calls']==2 for n in prof['nodes'])


def test_more_proxies_and_finite_search():
    ir=engine.template(channels=4,depth=1);x=torch.rand(6,3,32,32);y=torch.tensor([0,1,2,0,1,2])
    scores=engine.proxy(ir,x,y,1,methods=['gradnorm','fisher','zico','jacov'])
    assert all(m['status']=='completed' for m in scores['methods'].values()),scores
    space={'schema_version':1,'nodes':{},'variables':[{'id':'width','node_id':'block1','target':'config.out_channels','kind':'discrete','candidate_values':[4,8,12,16,24,32]}]}
    assert len(list(search_algorithms.enumerate_genes(space)))==6
    import random
    samples=[]
    for i,g in enumerate(list(search_algorithms.enumerate_genes(space))[:4]):
        model=__import__('neuroforge.search_space',fromlist=['instantiate']).instantiate(ir,space,g)
        samples.append(dict(ir=model,genes=g,proxy={'scores':{'gradnorm':i},'params':engine.analyze(model)['params']}))
    seen={engine.fingerprint(c['ir']) for c in samples};suggestion=search_algorithms.propose(space,ir,random.Random(3),samples,seen)
    assert suggestion['width'] in [24,32]


def test_diff_checked_edit_and_figure_scene():
    ir=engine.template(channels=4,depth=1);after=graph_tools.edit(ir,'insert','flatten','Linear',{'out_features':8})['ir'];delta=graph_tools.diff(ir,after)
    assert len(delta['added'])==1 and delta['edges_added'] and delta['changed']
    doc=figures.create(ir,{'mode':'3d'});doc['objects'].append(['text',100,100,'Edited title',20,'#123456',500,'start'])
    scene=figures.parse(doc);assert b'Edited title' in scene.svg() and scene.png(800).startswith(b'\x89PNG')
    bad=copy.deepcopy(doc);bad['objects'][-1][5]='url(javascript:bad)'
    with pytest.raises(ValueError):figures.parse(bad)


def test_pipeline_determinism_and_scope():
    x=torch.rand(4,3,32,32);steps=[{'op':'noise','params':{'std':.1},'scope':'train'}]
    assert torch.equal(pipeline.apply(x,steps,7),pipeline.apply(x,steps,7))
    assert not torch.equal(x,pipeline.apply(x,steps,7))
    assert torch.equal(x,pipeline.apply(x,steps,7,training=False))
    assert torch.equal(x,x.clone())
    with pytest.raises(ValueError):pipeline.validate([{'op':'resize','params':{'size':16}}],[3,32,32])


def test_studio_api_jobs_optimizer_and_training_options():
    client=TestClient(app);p=payload()
    assert len(client.get('/api/algorithms').json()['proxies'])==7
    assert client.post('/api/code/import',json={'source':CODE,'input_shape':[3,32,32]}).status_code==200
    debug=wait(submit('debug',dict(p,inspect_batch=3,benchmark_repeats=2)))
    assert debug['result']['nodes']
    blob=client.get('/api/jobs/'+debug['id']+'/tensors',params={'node':'block1'})
    assert blob.status_code==200 and blob.content[:2]==b'PK'
    trained=wait(submit('train',dict(p,epochs=4,gradient_accumulation=3,early_stopping_patience=1,early_stopping_delta=1.,auto_batch=True,data_pipeline=[{'op':'flip','params':{'probability':.5},'scope':'train'}])))
    assert trained['early_stopped'] and len(trained['metrics'])==2 and trained['effective_accumulation']==3
    for op in ('prune','dynamic_int8','onnx'):
        j=wait(submit('optimize',dict(p,optimization=op,inspect_batch=3)))
        assert j['result']['after']['latency_ms']>0
        if op=='onnx':assert j['result']['onnx_max_abs_error']<1e-4
        assert client.get('/api/jobs/'+j['id']+'/optimization').status_code==200

def test_auto_batch_reduces_after_real_oom_signal(monkeypatch):
    original=engine.GraphModel.forward
    def constrained(self,*args):
        if len(args[0])>8:raise torch.OutOfMemoryError('Simulated device capacity for batch preflight')
        return original(self,*args)
    monkeypatch.setattr(engine.GraphModel,'forward',constrained)
    j=wait(submit('train',payload(epochs=1,batch_size=32,auto_batch=True)))
    assert j['effective_batch_size']==8
    assert any('OOM' in line['text'] for line in j['logs'])


def test_imported_transformer_matches_torch_layer():
    code='''from torch import nn
class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.TransformerEncoderLayer(16,4,32,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.head=nn.Linear(16,3)
    def forward(self,x):
        x=self.encoder(x)
        x=x.mean(dim=1)
        return self.head(x)
'''
    ir=graph_tools.import_code(code,[5,16],input_type='sequence')['ir'];namespace={};exec(code,namespace)
    source=namespace['EncoderNet']().eval();model=engine.GraphModel(ir).eval();a=source.encoder;b=model.layers['encoder']
    b.norm1.load_state_dict(a.norm1.state_dict());b.norm2.load_state_dict(a.norm2.state_dict())
    b.attn.qkv.weight.data.copy_(a.self_attn.in_proj_weight);b.attn.qkv.bias.data.copy_(a.self_attn.in_proj_bias)
    b.attn.proj.load_state_dict(a.self_attn.out_proj.state_dict());b.ffn.layers[0].load_state_dict(a.linear1.state_dict());b.ffn.layers[3].load_state_dict(a.linear2.state_dict())
    model.layers['head'].load_state_dict(source.head.state_dict());x=torch.rand(3,5,16)
    assert torch.allclose(model(x),source(x),atol=1e-6)
