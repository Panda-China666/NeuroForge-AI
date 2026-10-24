import copy,io,json,zipfile,subprocess,sys,os
from pathlib import Path
import pytest,torch
from fastapi.testclient import TestClient
from neuroforge import copilot,engine,store,ecosystem,custom_ops,search_space
from neuroforge.app import app,asset_export_files
from test_api import payload,wait

client=TestClient(app)
def project(ir=None):
    p=payload();return client.post('/api/projects',json={'name':'Agent test','ir':ir or p['ir'],'dataset_id':p['dataset_id']}).json()
def plan(steps):return dict(title='Reviewed architecture change',rationale='Explicit structured tools.',assumptions=[],warnings=[],steps=[dict(action=a,node_id=n,parameter=k,value_json=json.dumps(v)) for a,n,k,v in steps])
def draft(p):return dict(project_id=p['id'],ir=p['ir'],search_space=p.get('search_space') or {'schema_version':1,'variables':[],'nodes':{},'global_constraints':{},'local_constraints':[]})
def test_agent_plan_is_nonexecuting_stale_safe_and_once_only():
    p=project();body=draft(p);before=len(store.all_of('job'));body['plan']=plan([('set','block1','out_channels',8)])
    r=client.post('/api/copilot/preview',json=body);assert r.status_code==200,r.text
    preview=r.json();assert preview['status']=='preview' and len(store.all_of('job'))==before
    assert store.get('project',p['id'])['ir']==p['ir']
    bad=copy.deepcopy(p['ir']);bad['name']='Newer unsaved draft'
    approve=dict(confirm=True,ir=bad,search_space=draft(p)['search_space'])
    assert client.post('/api/copilot/plans/'+preview['id']+'/apply',json=approve).status_code==400
    approve['ir']=p['ir'];r=client.post('/api/copilot/plans/'+preview['id']+'/apply',json=approve);assert r.status_code==200,r.text
    assert r.json()['project']['revision']==p['revision']+1
    assert client.post('/api/copilot/plans/'+preview['id']+'/apply',json=approve).status_code==400

def test_vit_requested_search_budget_and_cuda_plan_compiles_without_execution():
    p=project(engine.vit_template(embed_dim=32,depth=6));c=copilot.compile_plan(p,plan([('search','encoder','depth',{'kind':'range','numeric_range':{'min':4,'max':12,'step':1}}),('search','encoder','num_heads',{'kind':'discrete','candidate_values':[4,8,16]}),('constraints','','',{'max_params':50000000}),('run','','search',{'strategy':'bayesian','count':100,'device':'cuda:0','hardware_aware':True,'proxy_methods':['gradnorm']})]))
    assert c['space_summary']['combinations']==27 and c['space_summary']['legal_combinations']==27
    assert c['jobs'][0]['payload']['count']==100 and c['jobs'][0]['payload']['max_params']==50000000
    assert c['requires_confirmation'] and c['warnings']
    with pytest.raises(ValueError):copilot.compile_plan(p,plan([('set','encoder','num_heads',7)]))
    with pytest.raises(ValueError):copilot.compile_plan(p,plan([('run','','train',{'project_id':'hijack'})]))

@pytest.mark.parametrize('protocol',['responses','chat'])
def test_agent_real_tool_loop_with_explicit_transport_fixture(monkeypatch,protocol):
    # Fixture tests protocol/tool validation, not a live model's language understanding.
    p=project();expected=plan([('set','block1','out_channels',8)]);seen=[]
    actions=[('get_context',{}),('inspect_nodes',{'paths':['block1']}),('submit_plan',expected),('preview_plan',expected),('submit_plan',expected)]
    def fixture(config,body):
        index=len(seen);seen.append(body);name,args=actions[index]
        call=dict(name=name,arguments=json.dumps(args),call_id=str(index),type='function_call')
        if protocol=='responses':return {'output':[call]}
        return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':str(index),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}
    monkeypatch.setattr(copilot,'_provider_request',fixture)
    token=copilot.configure(None,'https://provider.example/v1','fixture-model','test-key-do-not-persist',protocol)
    preview=copilot.agent(p,token,'Change the first block width to eight.')
    assert preview['status']=='preview' and preview['origin']=='agent'
    assert preview['trace'][2]['status']=='invalid' # submit-before-preview was rejected
    assert len(seen)==5 and 'test-key-do-not-persist' not in json.dumps(preview)
    assert store.get('project',p['id'])['ir']==p['ir']
    copilot.forget(token)

def test_api_key_is_session_only_and_validation_never_reflects_it():
    key='test-secret-that-must-not-be-saved';r=client.post('/api/copilot/config',json=dict(endpoint='https://api.openai.com/v1',model='my-model',api_key=key))
    assert r.status_code==200 and key not in r.text and 'HttpOnly' in r.headers['set-cookie']
    assert client.get('/api/copilot/config').json()['configured']
    r=client.post('/api/copilot/config',json=dict(endpoint='https://api.openai.com/v1',model='m',api_key={'nested':key}));assert key not in r.text
    assert key not in (store.ROOT/'neuroforge.sqlite3').read_bytes().decode('latin1')
    assert client.post('/api/copilot/config',json=dict(endpoint='http://remote.example/v1',model='m',api_key=key)).status_code==400
    client.delete('/api/copilot/config');assert not client.get('/api/copilot/config').json()['configured']

@pytest.mark.parametrize('provider,model,options',[
 ('torchvision','resnet18',{}),('timm','vit_tiny_patch16_224',{}),
 ('huggingface','local/bert',{'hf_config':{'model_type':'bert','vocab_size':32,'hidden_size':16,'num_hidden_layers':1,'num_attention_heads':2,'intermediate_size':32,'max_position_embeddings':32}})])
def test_real_provider_models_weights_and_export(provider,model,options,tmp_path):
    j=store.put('job',dict(kind='import_model',status='running',payload={}))
    p=dict(provider=provider,model=model,input_shape=[8] if options else [3,32,32],classes=3,pretrained=False,**options)
    ecosystem.import_model(j['id'],p);r=store.get('job',j['id'])['result'];ir=r['ir'];graph=engine.GraphModel(ir).eval()
    x=torch.randint(0,30,(2,8)) if options else torch.rand(2,3,32,32)
    assert graph(x).shape==(2,3) and r['analysis']['params']==sum(v.numel() for v in graph.parameters())
    assert not r['analysis']['static_macs_available']
    # Fresh reconstruction preserves actual imported weights; a proxy cannot mutate the cache.
    first=next(iter(graph.parameters())).detach().clone();next(iter(graph.parameters())).data.zero_()
    assert torch.equal(next(iter(engine.GraphModel(ir).parameters())),first)
    files=asset_export_files(ir)
    for name,value in {**files,'model.py':engine.export_code(ir)}.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value.read_bytes() if isinstance(value,Path) else value.encode())
    env=dict(os.environ);env.pop('NEUROFORGE_DATA',None)
    result=subprocess.run([sys.executable,'model.py'],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    if provider=='torchvision':
        variant=ecosystem.replacement(ir,'backbone_layer4',8);v=next(n for n in variant['nodes'] if n['id']=='backbone_layer4');assert v['op']=='Module'
        space=dict(schema_version=1,nodes={},variables=[dict(id='width',node_id=v['id'],target='config.width',candidate_values=[4,8])],global_constraints={},local_constraints=[])
        child=search_space.instantiate(variant,space,{'width':4});assert engine.GraphModel(child)(x).shape==(2,3)
        assert child['nodes'][0]==ir['nodes'][0]

def test_custom_operator_registration_probe_reuse_and_disabling():
    p=dict(name='ChannelGate',source=custom_ops.EXAMPLE,parameters={'channels':4},input_shape=[4,32,32],trusted_code=True)
    j=client.post('/api/operators',json=p);assert j.status_code==200,j.text
    done=wait(j.json()['id']);ident=done['result']['id'];ir=engine.template(channels=4,depth=1)
    from neuroforge.graph_tools import edit
    ir=edit(ir,'insert','block1','CustomOperator',{'operator_id':ident,'parameters':{'channels':4}})['ir'];model=engine.GraphModel(ir)
    out=model(torch.randn(2,3,32,32));out.sum().backward();assert out.shape==(2,3)
    assert any('gain' in k for k in model.state_dict())
    assert client.post('/api/operators/'+ident+'/disable',json={}).status_code==200
    with pytest.raises(ValueError,match='disabled'):engine.GraphModel(ir)
    bad={**p,'trusted_code':False};assert client.post('/api/operators',json=bad).status_code==422

def test_single_proxy_never_uses_search_rejection_budgets():
    p=payload(max_params=100,max_macs=1000,proxy_methods=['gradnorm']);assert engine.analyze(p['ir'])['params']>100
    r=client.post('/api/jobs/proxy',json=p);assert r.status_code==200,r.text
    assert wait(r.json()['id'])['candidates']

def test_protected_macs_constraints_are_explicit_and_optimization_requires_matching_protocol():
    from neuroforge.assistant_report import compare
    shared=dict(kind='train',status='completed',device='cuda:0',device_name='fixture device',environment={'torch':'fixture','cuda':True},metrics=[{}],effective_batch_size=16,effective_accumulation=2,test_accuracy=.8,payload=dict(dataset_id='same',data_pipeline=[],seed=42,epochs=1,batch_size=16))
    a=store.put('job',dict(shared,memory_peak_mb=100));b=store.put('job',dict(shared,memory_peak_mb=65,test_accuracy=.795))
    r=compare(a['id'],b['id']);assert r['comparable'] and r['memory_target_observed'] and r['quality_preservation_observed']
    store.update('job',b['id'],payload={**b['payload'],'seed':43});r=compare(a['id'],b['id']);assert not r['comparable'] and not r['memory_target_observed']
