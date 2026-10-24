import copy
import io
import json
import random
import xml.etree.ElementTree as ET
import pytest
import torch
from PIL import Image
from neuroforge import engine,search_space,runtime,diagrams


@pytest.mark.parametrize('kind',['vit','tokens','linear_attention'])
def test_modern_graph_shapes_counts_gradients_and_standalone_export(kind):
    ir=engine.transformer_template(embed_dim=32,depth=1,num_heads=4) if kind=='tokens' else engine.vit_template(embed_dim=32,depth=2,num_heads=4)
    if kind=='linear_attention':next(n for n in ir['nodes'] if n['id']=='encoder')['config']['attention_type']='Linear'
    model=engine.GraphModel(ir);spec=engine.analyze(ir)
    assert spec['params']==sum(p.numel() for p in model.parameters())
    x=torch.randint(0,100,(2,32)) if kind=='tokens' else torch.rand(2,3,32,32)
    model.eval();y=model(x);assert y.shape==(2,3)
    y.sum().backward();assert all(p.grad is not None for p in model.parameters())
    scope={};exec(compile(engine.export_code(ir),'model.py','exec'),scope)
    exported=scope['build_model']();exported.load_state_dict(model.state_dict());exported.eval()
    torch.testing.assert_close(exported(x),model(x))


def test_transformer_shape_legality_and_patch_counts():
    ir=engine.vit_template(embed_dim=32,patch_size=4,num_heads=4)
    assert engine.analyze(ir)['nodes'][0]['shape']==[64,32]
    ir['nodes'][0]['config']['patch_size']=5
    with pytest.raises(ValueError,match='divisible'):engine.analyze(ir)
    ir=engine.vit_template(embed_dim=30,num_heads=8)
    with pytest.raises(ValueError,match='divisible'):engine.analyze(ir)


def space():
    return dict(schema_version=1,nodes={'block1':{'state':'Searchable'}},variables=[
        dict(id='width',node_id='block1',target='config.out_channels',kind='discrete',candidate_values=[4,8]),
        dict(id='repeat',node_id='block1',target='repeat',kind='range',numeric_range={'min':1,'max':2,'step':1})],global_constraints={'max_params':20000},local_constraints=[])


def test_local_search_repeats_and_fixed_graph_integrity():
    ir=engine.template(channels=4,depth=2);s=space();summary=search_space.summary(s,ir)
    assert summary['combinations']==4 and summary['legal_combinations']==4
    model=search_space.instantiate(ir,s,{'width':8,'repeat':2})
    assert any(n['id']=='block1_repeat2' for n in model['nodes'])
    assert next(n for n in model['nodes'] if n['id']=='pool1')['inputs']==['block1_repeat2']
    assert next(n for n in model['nodes'] if n['id']=='block2')['config']==ir['nodes'][2]['config']
    assert engine.GraphModel(model)(torch.rand(2,3,32,32)).shape==(2,3)
    s['nodes']['block1']['state']='Fixed'
    assert search_space.instantiate(ir,s,{'width':8,'repeat':2})==ir


def test_conditional_branch_toggle_and_circular_dependencies():
    ir=engine.template(channels=4,depth=1);s=space();s['variables'][1]=dict(id='pool_on',node_id='pool1',target='enabled',kind='boolean')
    s['variables'][0]['conditional_dependencies']={'variable':'pool_on','operator':'eq','value':True}
    model=search_space.instantiate(ir,s,{'width':8,'pool_on':False})
    assert not any(n['id']=='pool1' for n in model['nodes'])
    assert model['nodes'][0]['config']['out_channels']==4
    s['variables'][1]['conditional_dependencies']={'variable':'width','operator':'eq','value':8}
    with pytest.raises(ValueError,match='cycle'):search_space.validate(s,ir)


def test_reusable_module_exposed_parameters_and_multi_input_graph():
    ir=engine.template(channels=8,depth=2)
    graph,node=search_space.package_module(ir,['block1','pool1'],'Stage One',bindings={'channels':[{'node':'block1','parameter':'out_channels'}]})
    model=engine.GraphModel(graph)
    assert model(torch.rand(2,3,32,32)).shape==(2,3)
    assert engine.analyze(graph)['params']==engine.analyze(ir)['params']
    node=next(n for n in graph['nodes'] if n['op']=='Module');node['config']['channels']=12
    model=engine.GraphModel(graph)
    assert sum(p.numel() for p in model.parameters())==engine.analyze(graph)['params']
    assert model(torch.rand(2,3,32,32)).shape==(2,3)
    scope={};exec(engine.export_code(graph),scope)
    scope['build_model']().load_state_dict(model.state_dict())
    nested=copy.deepcopy(node['module']);nested['input_ports']=[{'id':'input','shape':[3,32,32]},{'id':'other','shape':[3,32,32]}]
    nested['nodes'].insert(0,dict(id='merge',op='Add',inputs=['input','other'],config={}))
    nested['nodes'][1]['inputs']=['merge']
    assert engine.GraphModel(nested)(torch.rand(2,3,32,32),torch.rand(2,3,32,32)).shape[1]==8


@pytest.mark.parametrize('mode',['2d','3d'])
def test_diagrams_are_real_vector_and_high_resolution_exports(mode):
    ir=engine.vit_template(embed_dim=32,depth=2)
    options={'mode':mode,'format':'svg','title':'Vision Transformer','resolution':1600}
    svg,mime=diagrams.render(ir,options);ET.fromstring(svg)
    assert b'Vision Transformer' in svg and b'<image' not in svg
    options['format']='png';png,mime=diagrams.render(ir,options)
    im=Image.open(io.BytesIO(png));assert im.width>=1600 and len(png)>5000
    options['format']='pdf';pdf,mime=diagrams.render(ir,options);assert pdf.startswith(b'%PDF')


def test_device_validation_and_error_context():
    assert str(runtime.resolve('cpu'))=='cpu'
    with pytest.raises(ValueError):runtime.resolve('not-a-device')
    assert runtime.error_context(RuntimeError('CUDA out of memory'))['code']=='oom'
    assert runtime.error_context(RuntimeError('no kernel image is available'))['code']=='driver_mismatch'


def test_nested_module_search_and_module_replacement():
    ir=engine.template(channels=8,depth=2)
    graph,node=search_space.package_module(ir,['block1','pool1'],'Stage',bindings={'channels':[{'node':'block1','parameter':'out_channels'}]})
    path=node['id']+'/block1'
    space=dict(schema_version=1,nodes={},variables=[dict(id='inner_width',node_id=path,target='config.out_channels',kind='discrete',candidate_values=[4,8])],global_constraints={},local_constraints=[])
    candidate=search_space.instantiate(graph,space,{'inner_width':4})
    module=next(n for n in candidate['nodes'] if n['op']=='Module')
    assert module['config']['channels']==4
    assert engine.GraphModel(candidate)(torch.rand(2,3,32,32)).shape==(2,3)

    alt=copy.deepcopy(node);alt['module']['nodes'][0]['op']='Conv2d';alt['bindings']={};alt['config']={}
    space.update(variables=[dict(id='kind',node_id=node['id'],target='module',kind='discrete',candidate_values=['a','b'])],module_choices={'a':node,'b':alt})
    candidate=search_space.instantiate(graph,space,{'kind':'b'})
    assert engine.GraphModel(candidate)(torch.rand(2,3,32,32)).shape==(2,3)

def test_fixed_dependency_and_long_diagram_labels():
    ir=engine.template(channels=4,depth=1);s=space()
    s['nodes']['block1']={'state':'Fixed'}
    s['variables'].append(dict(id='pool_type',node_id='pool1',target='op',kind='discrete',candidate_values=['AvgPool2d'],conditional_dependencies={'variable':'width','operator':'eq','value':4}))
    model=search_space.instantiate(ir,s,{'pool_type':'AvgPool2d'})
    assert next(n for n in model['nodes'] if n['id']=='pool1')['op']=='AvgPool2d'
    lines=diagrams.wrap_label('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW',18,130)
    assert len(lines)==2 and lines[-1].endswith('…')



@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not installed')
def test_real_cuda_cnn_and_vit_forward_backward_proxy_memory():
    for ir in [engine.template(channels=4,depth=1),engine.vit_template(embed_dim=32,depth=1)]:
        model=engine.GraphModel(ir).to('cuda:0')
        x=torch.rand(4,3,32,32,device='cuda:0');y=torch.tensor([0,1,2,0],device='cuda:0')
        torch.nn.functional.cross_entropy(model(x),y).backward()
        p=engine.proxy(ir,x,y,42,device='cuda:0')
        assert p['device']=='cuda:0' and p['memory_peak_mb']>0
        assert p['methods']['snip']['status']=='completed'
