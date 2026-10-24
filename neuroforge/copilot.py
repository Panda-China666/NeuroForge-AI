"""Bounded tool-using architecture agent. Provider credentials stay in RAM."""
import copy,json,hashlib,secrets,threading,time,urllib.request,urllib.error
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel,ConfigDict,Field
from typing import Literal
from . import engine,store,search_space,graph_tools,runtime

LOCK=threading.RLock();SESSIONS={};PROMPTS=Path(__file__).with_name('prompts')
class Strict(BaseModel):model_config=ConfigDict(extra='forbid')
class Step(Strict):
    action:Literal['set','search','constraints','template','insert','training','run','replacement']
    node_id:str=Field(max_length=250);parameter:str=Field(max_length=80);value_json:str=Field(max_length=16000)
class Plan(Strict):
    title:str=Field(min_length=1,max_length=160)
    rationale:str=Field(max_length=3000)
    assumptions:list[str]=Field(max_length=12)
    warnings:list[str]=Field(max_length=12)
    steps:list[Step]=Field(min_length=1,max_length=24)


def configure(session,endpoint,model,key,protocol='responses'):
    parsed=urlparse(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:raise ValueError('Use a base URL without credentials, query or fragment')
    if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1','::1')):raise ValueError('Use HTTPS, or HTTP for a local provider')
    if protocol not in ('responses','chat'):raise ValueError('Unsupported provider protocol')
    if not model.strip() or len(model)>120 or not key or len(key)>4096:raise ValueError('Model and API key are required')
    session=session if session in SESSIONS else secrets.token_urlsafe(32)
    with LOCK:SESSIONS[session]=dict(endpoint=endpoint.rstrip('/'),model=model.strip(),key=key,protocol=protocol,expires=time.time()+8*3600)
    def expire():
        with LOCK:
            if session in SESSIONS and SESSIONS[session]['expires']<=time.time():SESSIONS.pop(session,None)
    timer=threading.Timer(8*3600+.1,expire);timer.daemon=True;timer.start()
    return session


def configuration(session):
    with LOCK:
        value=SESSIONS.get(session)
        if value and value['expires']<time.time():SESSIONS.pop(session,None);value=None
        return {k:value[k] for k in ('endpoint','model','protocol')}|{'configured':True} if value else {'configured':False,'endpoint':'https://api.openai.com/v1','model':'','protocol':'responses'}


def forget(session):
    with LOCK:SESSIONS.pop(session,None)


def _provider_request(config,body):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    route='/responses' if config['protocol']=='responses' else '/chat/completions'
    request=urllib.request.Request(config['endpoint']+route,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+config['key'],'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=90) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000:raise ValueError('Provider response exceeded the output budget')
        return json.loads(raw)
    except urllib.error.HTTPError as exc:raise ValueError(f'Provider returned HTTP {exc.code}. Check model access, endpoint and API key.') from None
    except (urllib.error.URLError,TimeoutError):raise ValueError('Provider connection failed or timed out. Check the configured endpoint.') from None


def snapshot(project):
    return engine.fingerprint({k:project.get(k) for k in ('revision','ir','search_space','dataset_id','data_pipeline')})


def context(project,ir=None,space=None):
    ir=copy.deepcopy(ir or project['ir']);space=copy.deepcopy(space if space is not None else project.get('search_space') or {'schema_version':1,'nodes':{},'variables':[],'global_constraints':{},'local_constraints':[]})
    spec=engine.analyze(ir);nodes=search_space.index_nodes(ir)
    dataset=store.get('dataset',project['dataset_id']) if project.get('dataset_id') else None
    measured=[dict(id=j['id'],kind=j['kind'],device=j.get('device'),device_name=j.get('device_name'),memory_peak_mb=j.get('memory_peak_mb'),amp=j.get('amp_effective'),micro_batch=j.get('effective_batch_size'),accumulation=j.get('effective_accumulation'),test_accuracy=j.get('test_accuracy'),metrics=j.get('metrics',[])[-2:],benchmark=j.get('result',{}).get('benchmark'),status=j['status']) for j in store.all_of('job') if j.get('payload',{}).get('project_id')==project['id'] and j['status']=='completed' and engine.fingerprint(j['payload'].get('ir',{}))==engine.fingerprint(ir)][:5]
    return dict(project_id=project['id'],revision=project['revision'],model_ir=ir,search_space=space,node_paths=list(nodes),analysis={k:spec[k] for k in ('params','macs','output_shape')},dataset={k:dataset.get(k) for k in ('id','name','shape','input_type','classes','vocab_size','split_counts')} if dataset else None,hardware=runtime.devices(),measurements=measured,capabilities=dict(max_candidates=256,ops=engine.OPS,search=['random','evolution','bayesian','enumerate']),data_policy='Architecture/config/aggregate metrics only. No dataset samples or credentials.')


def _editable(ir,path):
    mapping=search_space.index_nodes(ir)
    if path not in mapping:raise ValueError('Unknown node path: '+path)
    node=mapping[path]
    if node['op'] in ('EcosystemBlock','CustomOperator'):raise ValueError('Protected operator: use a checked replacement or its own SDK, not arbitrary config edits')
    return node


def _set(ir,path,key,value):
    node=_editable(ir,path)
    if key not in node.get('config',{}) and key not in {'depth','num_heads','out_channels','out_features','kernel_size','stride','padding','ffn_ratio','dropout','embed_dim','patch_size','activation','norm','attention_type','p'}:raise ValueError('Unknown editable parameter: '+key)
    if type(value) not in (int,float,bool,str):raise ValueError('Parameter values must be JSON scalars')
    node['config'][key]=value
    # Keep exposed outer bindings consistent when an inner parameter is edited.
    parts=path.split('/')
    if len(parts)>1:
        parent=search_space.index_nodes(ir)['/'.join(parts[:-1])]
        for name,targets in parent.get('bindings',{}).items():
            if any(t['node']==parts[-1] and t['parameter']==key for t in targets):parent['config'][name]=value


def compile_plan(project,plan,ir=None,space=None):
    from .app import Task,validate_task
    plan=Plan.model_validate(plan).model_dump();before=copy.deepcopy(ir or project['ir']);model=copy.deepcopy(before)
    original_space=copy.deepcopy(space if space is not None else project.get('search_space') or {'schema_version':1,'nodes':{},'variables':[],'global_constraints':{},'local_constraints':[]})
    search=copy.deepcopy(original_space);jobs=[];training={};warnings=list(plan['warnings'])
    for step in plan['steps']:
        try:value=json.loads(step['value_json'])
        except json.JSONDecodeError:raise ValueError('value_json must contain valid JSON, not code')
        action,path,key=step['action'],step['node_id'],step['parameter']
        if action=='set':_set(model,path,key,value)
        elif action=='search':
            _editable(model,path);target=key if key in ('repeat','enabled','op') else 'config.'+key
            if not isinstance(value,dict) or set(value)-{'kind','candidate_values','numeric_range','sampling','value_type','default','conditional_dependencies'}:raise ValueError('Unsupported search-domain fields')
            variable=dict(value,id='agent_'+hashlib.sha256((path+target).encode()).hexdigest()[:10],node_id=path,target=target)
            search.setdefault('variables',[]);search['variables']=[v for v in search['variables'] if (v['node_id'],v['target'])!=(path,target)]+[variable]
            search.setdefault('nodes',{})[path]={'state':'Searchable'}
        elif action=='constraints':
            allowed={'max_params','max_macs','max_flops','max_layers','max_memory_mb','input_shape','output_shape','latency_ms'}
            if not isinstance(value,dict) or set(value)-allowed:raise ValueError('Unsupported architecture constraint')
            for field,bound in value.items():
                if field in ('input_shape','output_shape'):
                    if not isinstance(bound,list) or not bound or any(type(v)!=int or v<1 for v in bound):raise ValueError('Shape constraints require positive integer dimensions')
                elif type(bound) not in (int,float) or not __import__('math').isfinite(bound) or bound<=0:raise ValueError('Resource constraints must be positive finite numbers')
            search.setdefault('global_constraints',{}).update(value)
        elif action=='template':
            if not isinstance(value,dict):raise ValueError('Template options must be an object')
            options=dict(value);kind=options.pop('kind',None);dataset=store.get('dataset',project['dataset_id']);classes=len(dataset['classes'])
            if kind in ('cnn','residual'):model=engine.template(classes=classes,size=dataset['shape'][-1],residual=kind=='residual',**options)
            elif kind=='vit':model=engine.vit_template(classes=classes,size=dataset['shape'][-1],**options)
            elif kind=='transformer':
                if dataset.get('input_type')!='tokens':raise ValueError('Select a token dataset before creating a token model')
                model=engine.transformer_template(classes=classes,vocab_size=dataset['vocab_size'],length=dataset['shape'][0],**options)
            else:raise ValueError('Unsupported template')
            search={'schema_version':1,'nodes':{},'variables':[],'global_constraints':{},'local_constraints':[]};warnings.append('The current draft is replaced by a new template.')
        elif action=='insert':
            if '/' in path:raise ValueError('Enter the module editor for nested insertion; the agent currently inserts at model level')
            model=graph_tools.edit(model,'insert',path,key,value)['ir']
        elif action=='replacement':
            from .ecosystem import replacement
            model=replacement(model,path,**value)
        elif action=='training':
            if not isinstance(value,dict) or set(value)-{'amp','auto_batch','batch_size','gradient_accumulation','epochs','lr','weight_decay','early_stopping_patience','early_stopping_delta','device'}:raise ValueError('Unsupported training option')
            training.update(value)
        elif action=='run':
            if key not in ('search','proxy','train','debug','profile','optimize'):raise ValueError('Unsupported compute action')
            if not isinstance(value,dict):raise ValueError('Task options must be an object')
            jobs.append({'kind':key,'options':value})
    spec=engine.analyze(model);search_space.validate(search,model)
    if len(jobs)>4:raise ValueError('A reviewed plan can queue at most four jobs')
    dataset=store.get('dataset',project['dataset_id'])
    if jobs and (model['input_shape']!=dataset['shape'] or spec['output_shape']!=[len(dataset['classes'])]):raise ValueError('Model and selected dataset do not match; fix input / classifier before queuing compute')
    payloads=[]
    for job in jobs:
        options=dict(training if job['kind']=='train' else {});options.update(job['options'])
        base=dict(project_id=project['id'],dataset_id=project['dataset_id'],ir=model,data_pipeline=project.get('data_pipeline',[]),max_params=search.get('global_constraints',{}).get('max_params',50000000),max_macs=search.get('global_constraints',{}).get('max_macs',10000000000))
        if job['kind']=='search':base['search_space']=search
        base.update(options)
        # Immutable targets cannot be overwritten by model-generated task arguments.
        if any(k in options for k in ('project_id','dataset_id','ir','search_space','data_pipeline','source_job','source_candidate')):raise ValueError('Task target / weights cannot be redirected by the agent')
        if options.get('checkpoint_job'):
            if job['kind'] not in ('debug','profile','optimize'):raise ValueError('Checkpoint selection applies to debug, profile and optimization tasks')
            source=store.get('job',options['checkpoint_job'])
            if source.get('kind')!='train' or source.get('status')!='completed':raise ValueError('Select an immutable completed training run')
            if source.get('payload',{}).get('project_id')!=project['id'] or source['payload'].get('dataset_id')!=project['dataset_id'] or engine.fingerprint(source['payload'].get('ir',{}))!=engine.fingerprint(model):raise ValueError('Select saved weights from the same project, dataset and proposed graph')
            if not (store.ARTIFACTS/source['id']/'best.pt').exists():raise ValueError('Selected run has no saved best.pt')
        task=Task.model_validate(base);validate_task(job['kind'],task)
        payloads.append(dict(kind=job['kind'],payload=task.model_dump()))
    estimate=search_space.summary(search,model) if search.get('variables') else None
    if estimate:
        for job in payloads:
            if job['kind']=='search' and job['payload']['count']>estimate['legal_combinations']:warnings.append(f"Candidate budget {job['payload']['count']}; the current static estimate has only {estimate['legal_combinations']} legal distinct candidates. Search can finish early.")
    if training:
        validated=Task.model_validate(dict(project_id=project['id'],dataset_id=project['dataset_id'],ir=model,**training)).model_dump();training={k:validated[k] for k in training}
    return dict(plan=plan,before=before,after=model,search_before=original_space,search_after=search,diff=graph_tools.diff(before,model),space_summary=estimate,jobs=payloads,training=training,warnings=warnings,analysis={k:spec[k] for k in ('params','macs','output_shape')},requires_confirmation=True)


def save_preview(project,compiled,trace=None,prompt='',origin='manual'):
    return store.put('agent_plan',dict(project_id=project['id'],base_snapshot=snapshot(project),draft_hash=engine.fingerprint(compiled['before']),draft_space_hash=engine.fingerprint(compiled['search_before']),status='preview',compiled=compiled,trace=trace or [],prompt=prompt[:6000],origin=origin,expires=time.time()+3600))


def apply_plan(ident,expected_ir,expected_space):
    from . import worker
    with LOCK,worker.CONTROL_LOCK,store.LOCK:
        pending=store.get('agent_plan',ident);project=store.get('project',pending['project_id'])
        if pending['status']!='preview':raise ValueError('This plan has already been applied or discarded')
        if pending['expires']<time.time():raise ValueError('Plan expired. Generate a fresh preview.')
        if snapshot(project)!=pending['base_snapshot'] or engine.fingerprint(expected_ir)!=pending['draft_hash'] or engine.fingerprint(expected_space)!=pending['draft_space_hash']:raise ValueError('Project or draft changed after preview. Review a new plan.')
        c=pending['compiled'];active=sum(j['status'] in ('running','queued') for j in store.all_of('job'))
        if active+len(c['jobs'])>8:raise ValueError('Local queue is full. Nothing has been changed.')
        store.put('version',dict(project_id=project['id'],revision=project['revision'],ir=project['ir'],layout=project.get('layout',{}),search_space=project.get('search_space',{}),dataset_id=project.get('dataset_id')))
        updated=store.update('project',project['id'],ir=c['after'],search_space=c['search_after'],layout={},training_defaults={**project.get('training_defaults',{}),**c['training']},revision=project['revision']+1)
        ids=[]
        store.update('agent_plan',ident,status='applying')
        try:
            for job in c['jobs']:ids.append(worker.submit(job['kind'],job['payload'])['id'])
        except Exception:
            store.update('agent_plan',ident,status='partial',job_ids=ids);raise
        store.update('agent_plan',ident,status='applied',job_ids=ids,applied_revision=updated['revision'])
        return dict(project=updated,job_ids=ids,status='applied')


def _tools():
    schema=Plan.model_json_schema()
    # Bounds remain locally validated; strict API schemas only need structural types.
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ('maxLength','minLength','minItems','maxItems')}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return [dict(type='function',name='get_context',description='Read current graph, search space, data contract and measured hardware.',parameters={'type':'object','properties':{},'required':[],'additionalProperties':False},strict=True),dict(type='function',name='inspect_nodes',description='Read exact node configs and nested paths.',parameters={'type':'object','properties':{'paths':{'type':'array','items':{'type':'string'}}},'required':['paths'],'additionalProperties':False},strict=True),*[dict(type='function',name=name,description=description,parameters=clean(schema),strict=True) for name,description in [('preview_plan','Compile and preview a proposed plan. Does not execute changes.'),('submit_plan','Submit the identical successfully previewed plan for human approval. Never executes it.')]]]


def agent(project,session,prompt,mode='architecture',ir=None,space=None):
    configuration(session)
    with LOCK:config=copy.deepcopy(SESSIONS.get(session))
    if not config:raise ValueError('Configure your API key, endpoint and model in Copilot Settings first')
    ctx=context(project,ir,space);trace=[];successful={};tools=_tools()
    instructions=(PROMPTS/'architecture.md').read_text(encoding='utf-8-sig')
    if mode=='optimization':instructions+='\n'+(PROMPTS/'optimization.md').read_text(encoding='utf-8-sig')
    user=json.dumps({'user_request':prompt,'mode':mode,'context':ctx},ensure_ascii=False)
    if len(user)>300000:raise ValueError('Context is too large. Collapse or simplify nested modules first.')
    messages=[{'role':'user','content':user}]
    if config['protocol']=='chat':messages.insert(0,{'role':'system','content':instructions})
    for round_index in range(8):
        if not configuration(session)['configured']:raise ValueError('Provider session expired or was disconnected')
        if config['protocol']=='responses':
            body=dict(model=config['model'],instructions=instructions,input=messages,tools=tools,tool_choice='required',max_output_tokens=6500,store=False)
        else:body=dict(model=config['model'],messages=messages,tools=[{'type':'function','function':{k:v for k,v in t.items() if k!='type'}} for t in tools],tool_choice='required',max_completion_tokens=6500)
        response=_provider_request(config,body)
        if config['protocol']=='responses':
            output=response.get('output',[]);messages.extend(output);calls=[(c['name'],c['arguments'],c['call_id']) for c in output if c.get('type')=='function_call']
        else:
            message=response.get('choices',[{}])[0].get('message',{});messages.append(message);calls=[(c['function']['name'],c['function']['arguments'],c['id']) for c in message.get('tool_calls',[])]
        if not calls:raise ValueError('The provider returned no tool calls. Select a model that supports function calling.')
        if len(calls)>6:raise ValueError('Provider exceeded the per-round tool budget')
        for name,arguments,call_id in calls:
            try:
                args=json.loads(arguments)
                if name=='get_context':result=ctx
                elif name=='inspect_nodes':
                    mapping=search_space.index_nodes(ctx['model_ir']);paths=args.get('paths',[])[:24];result={p:mapping.get(p,{'error':'Unknown node path'}) for p in paths}
                elif name in ('preview_plan','submit_plan'):
                    plan=Plan.model_validate(args).model_dump();key=engine.fingerprint(plan)
                    if name=='submit_plan':
                        if key not in successful:raise ValueError('Call preview_plan successfully with this exact plan before submitting it')
                        saved=save_preview(project,successful[key],trace+[{'tool':'submit_plan','status':'review_required'}],prompt,'agent')
                        return saved
                    compiled=compile_plan(project,plan,ctx['model_ir'],ctx['search_space']);successful[key]=compiled
                    result={k:compiled[k] for k in ('diff','space_summary','warnings','analysis','requires_confirmation')}
                else:raise ValueError('Unsupported agent tool')
                trace.append({'tool':name,'status':'ok','round':round_index+1})
            except (ValueError,KeyError,TypeError) as exc:
                result={'error':str(exc)[:3000]};trace.append({'tool':name,'status':'invalid','round':round_index+1})
            content=json.dumps(result,ensure_ascii=False)
            if config['protocol']=='responses':messages.append({'type':'function_call_output','call_id':call_id,'output':content})
            else:messages.append({'role':'tool','tool_call_id':call_id,'content':content})
    raise ValueError('Agent reached its eight-round limit without a valid plan. Narrow the request and retry.')
