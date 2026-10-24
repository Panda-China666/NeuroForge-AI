import io
import json
import time
import zipfile
import pytest
import torch
from fastapi.testclient import TestClient
from neuroforge.app import app
from neuroforge import engine,store,worker

client=TestClient(app)


def payload(**updates):
    p=client.get('/api/projects').json()[0]
    d=next(d for d in client.get('/api/datasets').json() if not d.get('input_type'))
    result=dict(project_id=p['id'],dataset_id=d['id'],ir=engine.template(channels=4,depth=1),epochs=3,batch_size=32,seed=33,device="cpu")
    result.update(updates)
    return result


def wait(ident,target=None):
    until=time.time()+45
    while time.time()<until:
        j=client.get('/api/jobs/'+ident).json()
        if j['status'] in ('completed','failed','paused','cancelled'):
            assert j['status']==(target or 'completed'),j.get('error') or j['logs']
            return j
        time.sleep(.05)
    raise AssertionError('Worker did not complete')


def submit(kind,p):
    r=client.post('/api/jobs/'+kind,json=p)
    assert r.status_code==200,r.text
    return r.json()['id']


def test_system_projects_validation_and_origin_guard():
    assert client.get('/').status_code==200
    assert 'torch' in client.get('/api/system').json()
    assert client.post('/api/projects',json={'name':'test','ir':engine.template()}).status_code==200
    project=client.get('/api/projects').json()[0]
    old=project['revision']
    r=client.put('/api/projects/'+project['id'],json={'name':'changed','ir':project['ir'],'dataset_id':project.get('dataset_id'),'layout':{'input':{'x':40,'y':30}}})
    assert r.json()['revision']==old+1
    assert r.json()['layout']['input']=={'x':40,'y':30}
    assert client.get('/api/projects/'+project['id']+'/versions').json()[0]['revision']==old
    assert client.get('/api/jobs/no-such-job').status_code==404
    assert client.post('/api/analyze',json={'ir':{'schema_version':999}}).status_code==400
    assert client.post('/api/projects',json={'name':'bad'},headers={'Origin':'https://untrusted.example'}).status_code==403
    assert client.get('/api/system',headers={'Host':'untrusted.example'}).status_code==403


def test_proxy_search_evolution_and_constraints():
    proxy=wait(submit('proxy',payload()))
    assert len(proxy['candidates'])==1
    cs=wait(submit('search',payload(count=4,channels=[4,8],depth=[1,2],residual=[True,False],max_params=20000)))['candidates']
    assert len(cs)==4
    assert len({c['id'] for c in cs})==4
    assert all(c['proxy']['params']<=20000 for c in cs)
    assert sorted(c['rank'] for c in cs)==[1,2,3,4]
    evo=wait(submit('search',payload(strategy='evolution',count=6,channels=[4,8],depth=[1,2],residual=[True,False])))
    assert len(evo['candidates'])==6
    assert all(set(c['proxy']['sample_indices']).issubset(set(store.get('dataset',evo['payload']['dataset_id'])['splits']['train'])) for c in evo['candidates'])


def test_training_pause_resume_exact_and_export(monkeypatch):
    baseline=wait(submit('train',payload()))
    original=worker.atomic_checkpoint
    paused=[]
    def pause_after_first(path,value):
        original(path,value)
        if path.name=='checkpoint.pt' and value['epoch']==1 and not paused:
            paused.append(str(path.parent.name))
            worker.control(path.parent.name,'pause')
    monkeypatch.setattr(worker,'atomic_checkpoint',pause_after_first)
    ident=submit('train',payload())
    j=wait(ident,'paused')
    assert len(j['metrics'])==1
    assert client.post('/api/jobs/'+ident+'/control',json={'action':'resume'}).status_code==200
    j=wait(ident)
    assert len(j['metrics'])==3 and j['test_accuracy'] is not None
    a=torch.load(store.ARTIFACTS/baseline['id']/'checkpoint.pt',weights_only=True)
    b=torch.load(store.ARTIFACTS/ident/'checkpoint.pt',weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert [m['val_accuracy'] for m in a['metrics']]==[m['val_accuracy'] for m in b['metrics']]
    response=client.get('/api/export/job/'+ident)
    assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        assert {'best.pt','checkpoint.pt','model.py','dataset_manifest.json','experiment.json'}.issubset(z.namelist())
        scope={}
        exec(compile(z.read('model.py'),'model.py','exec'),scope)
        model=scope['build_model']()
        model.load_state_dict(torch.load(io.BytesIO(z.read('best.pt')),weights_only=True))
        assert model(torch.rand(2,3,32,32)).shape==(2,3)


def test_dataset_resplit_preview_and_model_mismatch():
    d=next(d for d in client.get('/api/datasets').json() if not d.get('input_type'))
    assert client.get('/api/datasets/'+d['id']+'/preview/0').headers['content-type']=='image/png'
    assert client.get('/api/datasets/'+d['id']+'/preview/-1').status_code==404
    r=client.post('/api/datasets/'+d['id']+'/split',json={'seed':77,'val':.2,'test':.2})
    assert r.status_code==200
    assert r.json()['id']!=d['id'] and r.json()['parent']==d['id']
    ir=engine.template(classes=10)
    assert client.post('/api/jobs/train',json=payload(ir=ir)).status_code==400


def test_recovery_marks_inflight_jobs_interrupted():
    j=store.put('job',{'kind':'train','status':'running','payload':payload()})
    worker.recover()
    assert store.get('job',j['id'])['status']=='paused'
