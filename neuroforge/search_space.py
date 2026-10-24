"""Search Space IR is distinct from the executable model graph. No expression eval."""
import copy
import itertools
import math
import random
from . import engine


def index_nodes(ir,prefix=''):
    result={}
    for n in ir['nodes']:
        path=prefix+n['id'];result[path]=n
        if n['op']=='Module':result.update(index_nodes(n['module'],path+'/'))
    return result


def domain(variable):
    kind=variable.get('kind','discrete')
    if kind=='boolean':return [False,True]
    if kind=='range':
        r=variable.get('numeric_range',{})
        lo,hi,step=r.get('min'),r.get('max'),r.get('step',1)
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (lo,hi,step)) or hi<lo or step<=0:
            raise ValueError('Numeric range requires min <= max and a positive step')
        if variable.get('sampling','linear')=='log':
            count=r.get('samples',5)
            if lo<=0 or type(count)!=int or not 2<=count<=32:raise ValueError('Log sampling requires positive values and 2–32 samples')
            values=[math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*i/(count-1)) for i in range(count)]
        else:
            count=math.floor((hi-lo)/step+1e-8)+1
            if count>64:raise ValueError('A range supports at most 64 sampled values')
            values=[round(lo+i*step,8) for i in range(count)]
        if variable.get('value_type','integer')=='integer':values=[int(round(v)) for v in values]
    else:values=variable.get('candidate_values',variable.get('values',[]))
    if not isinstance(values,list) or not 1<=len(values)<=64 or any(type(v) not in (str,int,float,bool) for v in values):
        raise ValueError('Candidate values require 1–64 scalar values')
    if any(type(v) in (int,float) and not math.isfinite(v) for v in values):raise ValueError('Search candidates must be finite')
    return list(dict.fromkeys(values))


def validate(space,ir):
    if space.get('schema_version',1)!=1:raise ValueError('Unsupported Search Space IR version')
    variables=space.get('variables',[])
    if not isinstance(variables,list) or len(variables)>64:raise ValueError('Search space supports at most 64 variables')
    ids=[v.get('id') for v in variables]
    if any(not isinstance(i,str) or not i.isidentifier() for i in ids) or len(ids)!=len(set(ids)):raise ValueError('Search variable IDs must be unique identifiers')
    nodes=index_nodes(ir)
    targets=[];domains={}
    for v in variables:
        node=v.get('node_id');target=v.get('target','')
        if node not in nodes:raise ValueError(f'Unknown search node: {node}')
        if target not in ('op','repeat','enabled','module') and not (target.startswith('config.') and target.count('.')==1 and target[7:].isidentifier()):raise ValueError('Invalid search parameter target')
        if (node,target) in targets:raise ValueError('Only one search variable is allowed for each node parameter')
        targets.append((node,target));domains[v['id']]=domain(v)
        if target=='op' and any(value not in engine.OPS or value=='Module' for value in domains[v['id']]):raise ValueError('Unsupported operator choice')
        if target=='repeat' and any(type(value)!=int or not 0<=value<=24 for value in domains[v['id']]):raise ValueError('Repeat choices must be integers from 0 to 24')
        if target=='enabled' and any(type(value)!=bool for value in domains[v['id']]):raise ValueError('Enabled choices must be boolean')
        if target=='module' and (nodes[node]['op']!='Module' or any(value not in space.get('module_choices',{}) for value in domains[v['id']])):raise ValueError('Module choices must reference available reusable module definitions')
        condition=v.get('conditional_dependencies')
        if condition:
            if condition.get('variable') not in ids or condition['variable']==v['id']:raise ValueError('Conditional dependency must reference another variable')
    # Detect cycles before sampling, independent of variable declaration order.
    done=set();pending=list(variables)
    while pending:
        ready=[v for v in pending if not v.get('conditional_dependencies') or v['conditional_dependencies']['variable'] in done]
        if not ready:raise ValueError('Conditional variable dependencies contain a cycle')
        for v in ready:done.add(v['id']);pending.remove(v)
    return domains


def active_variables(space):
    states=space.get('nodes',{})
    return [v for v in space.get('variables',[]) if states.get(v['node_id'],{}).get('state','Searchable')!='Fixed']


def condition_true(cond,genes):
    a=genes.get(cond['variable']);b=cond.get('value');op=cond.get('operator','eq')
    if a is None:return False
    if op=='eq':return a==b
    if op=='ne':return a!=b
    if op=='in':return a in b if isinstance(b,list) else False
    raise ValueError('Conditional operator must be eq, ne, or in')


def sample(space,ir,rng,previous=None):
    domains=validate(space,ir);vs=active_variables(space);genes={}
    if previous:
        genes=copy.deepcopy(previous)
        if vs:
            v=rng.choice(vs);genes[v['id']]=rng.choice(domains[v['id']])
    for v in vs:
        genes.setdefault(v['id'],rng.choice(domains[v['id']]))
    return genes


def instantiate(ir,space,genes,check=True):
    domains=validate(space,ir);model=copy.deepcopy(ir);mapping=index_nodes(model);repeats={};enabled={}
    genes=copy.deepcopy(genes)
    active_ids={v['id'] for v in active_variables(space)}
    for v in space.get('variables',[]):
        if v['id'] not in active_ids:
            n=mapping[v['node_id']];target=v['target']
            genes[v['id']]=n['config'].get(target[7:]) if target.startswith('config.') else n['op'] if target=='op' else True if target=='enabled' else 1 if target=='repeat' else None
    vs=active_variables(space);resolved=set();pending=list(vs);applied=set()
    while pending:
        for v in list(pending):
            cond=v.get('conditional_dependencies')
            if cond and cond['variable'] not in resolved:
                if any(x['id']==cond['variable'] for x in pending):continue
            value=genes.get(v['id'],v.get('default',domains[v['id']][0]))
            if value not in domains[v['id']]:raise ValueError(f'Value outside search domain: {v["id"]}')
            resolved.add(v['id']);pending.remove(v)
            if cond and not condition_true(cond,genes):
                genes[v['id']]=None
                continue
            applied.add((v['node_id'],v['target']))
            n=mapping[v['node_id']];target=v['target']
            if target.startswith('config.'):n['config'][target[7:]]=value
            elif target=='op':n['op']=value
            elif target=='repeat':repeats[v['node_id']]=value
            elif target=='enabled':enabled[v['node_id']]=value
            elif target=='module':
                choice=space['module_choices'][value]
                n.update(module=copy.deepcopy(choice['module']),config=copy.deepcopy(choice.get('config',{})),bindings=copy.deepcopy(choice.get('bindings',{})),name=choice.get('name','Module'))
    def expand(graph,base,prefix=''):
        order=[n['id'] for n in engine.analyze(base,_level=1 if prefix else 0)['nodes']]
        local={n['id']:n for n in graph['nodes']};original={n['id']:n for n in base['nodes']};aliases={};result=[]
        for ident in order:
            n=local[ident];path=prefix+ident
            if n['op']=='Module' and original[ident]['op']=='Module':
                prior=original[ident]['module']
                if {x['id'] for x in prior['nodes']}!={x['id'] for x in n['module']['nodes']}:prior=copy.deepcopy(n['module'])
                expand(n['module'],prior,path+'/')
                # An explicitly exposed outer value takes precedence only if its variable is active.
                for name,targets in list(n.get('bindings',{}).items()):
                    if (path,'config.'+name) not in applied and any((path+'/'+t['node'],'config.'+t['parameter']) in applied for t in targets):
                        for target in targets:
                            inner=next((x for x in n['module']['nodes'] if x['id']==target['node']),None)
                            if inner:n['config'][name]=inner['config'].get(target['parameter'],n['config'].get(name))
            n['inputs']=[aliases.get(parent,parent) for parent in n['inputs']]
            count=repeats.get(path,1) if enabled.get(path,True) else 0
            if count!=1 and len(n['inputs'])!=1:raise ValueError('Repeat / branch bypass requires a single boundary input')
            if count==0:aliases[ident]=n['inputs'][0];continue
            for i in range(count):
                node=copy.deepcopy(n);node['id']=ident if i==0 else f'{ident}_repeat{i+1}'
                if i:node['inputs']=[result[-1]['id']]
                result.append(node)
            aliases[ident]=result[-1]['id']
        graph['nodes']=result;graph['output']=aliases.get(graph['output'],graph['output'])
    expand(model,ir)
    for rule in space.get('local_constraints',[]):
        def value(ref):
            if not isinstance(ref,str):return ref
            if ref.startswith('$'):return genes.get(ref[1:])
            if '.' in ref:
                ident,key=ref.split('.',1);return mapping[ident]['config'].get(key)
            return ref
        a,b=value(rule.get('left')),value(rule.get('right'));op=rule.get('operator','eq')
        ok=(a==b) if op=='eq' else (a!=b) if op=='ne' else (a<=b) if op=='le' else (a>=b) if op=='ge' else (b!=0 and a%b==0) if op=='divisible_by' else False
        if not ok:raise ValueError('Local relationship constraint rejected this candidate')
    if check:check_constraints(model,space.get('global_constraints',{}))
    return model


def check_constraints(ir,constraints):
    s=engine.analyze(ir)
    if not s.get('static_macs_available',True) and any(constraints.get(k) is not None for k in ('max_macs','max_flops')):raise ValueError('Static MACs are unavailable inside protected blocks; use measured latency constraints instead')
    for key,actual in [('max_params',s['params']),('max_macs',s['macs']),('max_flops',s['flops']),('max_layers',s['expanded_nodes']),('max_memory_mb',s['parameter_memory_mb']+s['activation_memory_mb'])]:
        bound=constraints.get(key)
        if bound is not None and actual>bound:raise ValueError(f'{key} exceeded: {actual:g} > {bound:g}')
    if constraints.get('input_shape') and ir['input_shape']!=constraints['input_shape']:raise ValueError('Input shape constraint failed')
    if constraints.get('output_shape') and s['output_shape']!=constraints['output_shape']:raise ValueError('Output shape constraint failed')
    return s


def summary(space,ir):
    domains=validate(space,ir);vs=active_variables(space);total=math.prod(len(domains[v['id']]) for v in vs)
    cap=256;rng=random.Random(42);legal=0;reasons={};tested=0;seen=set()
    combinations=itertools.product(*[domains[v['id']] for v in vs]) if total<=cap else None
    for i in range(min(total,cap)):
        genes=dict(zip([v['id'] for v in vs],next(combinations))) if combinations is not None else sample(space,ir,rng)
        try:
            model=instantiate(ir,space,genes);key=engine.fingerprint(model)
            if total>cap or key not in seen:legal+=1
            seen.add(key)
        except (ValueError,KeyError,TypeError) as exc:
            text=str(exc);reasons[text]=reasons.get(text,0)+1
        tested+=1
    searched={v['node_id'] for v in vs}
    return dict(variables=len(vs),discrete_variables=sum(v.get('kind')!='range' for v in vs),conditional_variables=sum(bool(v.get('conditional_dependencies')) for v in vs),
                combinations=total,legal_combinations=legal if total<=cap else round(total*legal/tested),estimate=total>cap,checked=tested,
                fixed_nodes=len(index_nodes(ir))-len(searched),searchable_nodes=len(searched),recommended_candidates=min(64,max(1,min(total,legal if total<=cap else 24))),
                rejections=reasons,latency_enforced=False)


def package_module(ir,selected,name,output=None,bindings=None):
    spec=engine.analyze(ir);selected=set(selected);mapping={n['id']:n for n in ir['nodes']};shapes={n['id']:n['shape'] for n in spec['nodes']};shapes['input']=ir['input_shape']
    if not selected or not selected.issubset(mapping):raise ValueError('Select valid model nodes to package')
    exits=[i for i in selected if i==ir['output'] or any(i in n['inputs'] for n in ir['nodes'] if n['id'] not in selected)]
    if len(exits)!=1:raise ValueError('Choose a connected subgraph with one output boundary')
    output=output or exits[0]
    if output!=exits[0]:raise ValueError('Output must match the subgraph boundary')
    external=[]
    for n in spec['nodes']:
        if n['id'] in selected:
            for parent in mapping[n['id']]['inputs']:
                if parent not in selected and parent not in external:external.append(parent)
    if not external:raise ValueError('Module requires at least one input boundary')
    ports={p:'input' if i==0 else f'input_{i+1}' for i,p in enumerate(external)}
    internal=[]
    for n in ir['nodes']:
        if n['id'] in selected:
            node=copy.deepcopy(n);node['inputs']=[ports.get(p,p) for p in node['inputs']];internal.append(node)
    graph=dict(schema_version=1,name=name,input_shape=shapes[external[0]],input_ports=[{'id':ports[p],'shape':shapes[p]} for p in external],nodes=internal,output=output)
    engine.analyze(graph,_level=1)
    ident='module_'+engine.fingerprint(graph)[:8]
    config={};bindings=bindings or {}
    for key,targets in bindings.items():
        if not key.isidentifier() or not targets:raise ValueError('Invalid exposed parameter name')
        for target in targets:
            if target['node'] not in selected or target['parameter'] not in mapping[target['node']]['config']:raise ValueError('Exposed parameter must reference an internal configuration field')
        config[key]=mapping[targets[0]['node']]['config'][targets[0]['parameter']]
    module=dict(id=ident,op='Module',name=name,inputs=external,config=config,module=graph,bindings=bindings)
    result=copy.deepcopy(ir);result['nodes']=[n for n in result['nodes'] if n['id'] not in selected]+[module]
    for node in result['nodes']:
        if node['id']!=ident:node['inputs']=[ident if p==output else p for p in node['inputs']]
    if result['output']==output:result['output']=ident
    engine.analyze(result)
    return result,module
