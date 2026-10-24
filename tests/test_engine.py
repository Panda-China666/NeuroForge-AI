import copy
import io
import random
import zipfile
import numpy as np
import pytest
import torch
from PIL import Image
from neuroforge import engine,data

torch.set_num_threads(2)


@pytest.mark.parametrize('residual',[True,False])
def test_compiler_counts_shapes_export_and_gradients(residual):
    ir=engine.template(channels=8,depth=2,residual=residual)
    spec=engine.analyze(ir)
    model=engine.GraphModel(ir)
    assert sum(p.numel() for p in model.parameters())==spec['params']
    x=torch.rand(4,3,32,32)
    out=model(x)
    assert list(out.shape)==[4,3]
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    scope={}
    exec(compile(engine.export_code(ir),'model.py','exec'),scope)
    exported=scope['build_model']()
    exported.load_state_dict(model.state_dict())
    torch.testing.assert_close(exported(x),model(x))


def branch_ir(op):
    return dict(schema_version=1,input_shape=[3,16,16],output='head',nodes=[
        dict(id='a',op='Conv2d',inputs=['input'],config={'out_channels':4}),
        dict(id='b',op='Conv2d',inputs=['input'],config={'out_channels':4}),
        dict(id='merge',op=op,inputs=['a','b'],config={}),
        dict(id='gap',op='AdaptiveAvgPool2d',inputs=['merge'],config={}),
        dict(id='flat',op='Flatten',inputs=['gap'],config={}),
        dict(id='head',op='Linear',inputs=['flat'],config={'out_features':2})])


@pytest.mark.parametrize('op',['Add','Concat'])
def test_branches_topological_sort_and_counts(op):
    ir=branch_ir(op)
    random.Random(9).shuffle(ir['nodes'])
    model=engine.GraphModel(ir)
    assert model(torch.ones(2,3,16,16)).shape==(2,2)
    assert sum(p.numel() for p in model.parameters())==engine.analyze(ir)['params']


def test_cycles_mismatch_and_unused_nodes_rejected():
    ir=engine.template()
    ir['nodes'][0]['inputs']=['classifier']
    with pytest.raises(ValueError,match='cycle'):
        engine.analyze(ir)
    ir=branch_ir('Add')
    ir['nodes'][1]['config']['out_channels']=8
    with pytest.raises(ValueError,match='identical'):
        engine.analyze(ir)
    ir=engine.template()
    ir['nodes'].append(dict(id='dead',op='ReLU',inputs=['input'],config={}))
    with pytest.raises(ValueError,match='contribute'):
        engine.analyze(ir)


def test_proxies_are_real_deterministic_and_have_no_weight_updates():
    ir=engine.template(channels=4,depth=1)
    torch.manual_seed(123)
    x=torch.rand(6,3,32,32)
    y=torch.tensor([0,1,2,0,1,2])
    a=engine.proxy(ir,x,y,7)
    b=engine.proxy(ir,x,y,7)
    assert a['scores']==b['scores']
    assert a['scores']['synflow']>0
    assert a['scores']['snip']>0
    assert a['scores']['naswot'] is not None
    assert a['seconds']>0 and a['validity']=='uncalibrated'
    different=engine.proxy(engine.template(channels=8,depth=1),x,y,7)
    assert different['scores']!=a['scores']


def test_grouped_splits_and_reproducible_dataset(tmp_path):
    m=data.builtin(tmp_path)
    a,b,c=[set(m['splits'][k]) for k in ['train','val','test']]
    assert not a&b and not b&c and not a&c and len(a|b|c)==300
    assert data.builtin(tmp_path)['id']==m['id']
    labels=[0]*6+[1]*6
    groups=['a','a','b','c','d','e','f','f','g','h','i','j']
    splits=data.split_indices(labels,42,groups=groups)
    for ids in splits.values():
        assert (0 in ids)==(1 in ids)
        assert (6 in ids)==(7 in ids)


def test_zip_import_quality_and_traversal(tmp_path):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        for c in ['cats','dogs']:
            for i in range(6):
                image=Image.new('RGB',(16,16),(i*30,100 if c=='cats' else 200,0))
                b=io.BytesIO();image.save(b,format='PNG')
                z.writestr(f'{c}/{i}.png',b.getvalue())
        z.writestr('dogs/broken.png',b'broken')
    m=data.import_zip(tmp_path,stream.getvalue(),'uploaded',16,42)
    assert m['samples']==12 and len(m['issues'])==1 and m['classes']==['cats','dogs']
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('../outside.png',b'bad')
    with pytest.raises(ValueError,match='Unsafe'):
        data.import_zip(tmp_path,stream.getvalue(),'bad',16,42)


def test_rank_ties_and_pareto():
    cs=[dict(id=str(i),proxy={'params':p,'scores':{'snip':v,'synflow':v,'naswot':None}}) for i,(p,v) in enumerate([(100,1),(200,2),(300,2)])]
    engine.rank_candidates(cs)
    byid={c['id']:c for c in cs}
    assert byid['1']['score']==byid['2']['score']
    assert byid['1']['pareto'] and not byid['2']['pareto']


def test_reserved_names_and_malformed_nodes_are_rejected():
    ir=engine.template()
    ir['nodes'][0]['id']='training'
    with pytest.raises(ValueError,match='reserved'):
        engine.analyze(ir)
    ir['nodes']=[42]
    with pytest.raises(ValueError,match='object'):
        engine.analyze(ir)
