"""Non-executing Python AST import, structural diff, and checked graph edits."""
import ast
import copy
import json
import re
from . import engine,search_space


def import_code(source,input_shape,class_name=None,input_type='image'):
    if len(source)>500000:raise ValueError('Python source exceeds 500KB')
    try:tree=ast.parse(source)
    except SyntaxError as e:raise ValueError(f'Line {e.lineno}: {e.msg}')
    def fail(node,msg):raise ValueError(f'Line {getattr(node,"lineno",0)}: {msg}')
    # Our generated standalone files carry an exact architecture literal.
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARCHITECTURE' for t in node.targets):
            try:
                value=node.value
                ir=json.loads(ast.literal_eval(value.args[0])) if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='loads' else ast.literal_eval(value)
                spec=engine.analyze(ir);return dict(ir=ir,analysis=spec,mode='embedded IR',warnings=[],source=source)
            except (ValueError,TypeError,KeyError):pass
    env={};modules={};nodes=[];used=set();counter=0
    prefixes={'torch.nn'};constructors=set()
    for statement in tree.body:
        if isinstance(statement,ast.Import):
            for alias in statement.names:
                if alias.name=='torch.nn':prefixes.add(alias.asname or alias.name)
        elif isinstance(statement,ast.ImportFrom):
            for alias in statement.names:
                if statement.module=='torch' and alias.name=='nn':prefixes.add(alias.asname or 'nn')
                if statement.module=='torch.nn':constructors.add(alias.asname or alias.name)
    def path(node):
        if isinstance(node,ast.Name):return node.id
        if isinstance(node,ast.Attribute):return path(node.value)+'.'+node.attr
        return ''
    def literal(node):
        if isinstance(node,ast.Name) and node.id in env:return env[node.id]
        if isinstance(node,ast.Attribute) and path(node) in env:return env[path(node)]
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Add,ast.Sub,ast.FloorDiv)):
            a,b=literal(node.left),literal(node.right)
            return a*b if isinstance(node.op,ast.Mult) else a+b if isinstance(node.op,ast.Add) else a-b if isinstance(node.op,ast.Sub) else a//b
        try:return ast.literal_eval(node)
        except Exception:fail(node,'Only literal values or constant arithmetic are supported')
    def scalar(v,node):
        if isinstance(v,(list,tuple)):
            if not v or len(set(v))!=1:fail(node,'Only square kernels / uniform dimensions are supported')
            return v[0]
        return v
    def layer(node):
        if isinstance(node,ast.Attribute) and path(node) in modules:return copy.deepcopy(modules[path(node)])
        if not isinstance(node,ast.Call):fail(node,'Expected a torch.nn layer constructor')
        full=path(node.func);name=full.split('.')[-1]
        if not (full in constructors or any(full==prefix+'.'+name for prefix in prefixes)):fail(node,'Only imported torch.nn constructors are supported')
        if name=='Sequential':return [v for a in node.args for v in layer(a)]
        signatures={'Conv2d':['in_channels','out_channels','kernel_size','stride','padding','dilation','groups','bias','padding_mode'],'Linear':['in_features','out_features','bias'],'MaxPool2d':['kernel_size','stride','padding','dilation','return_indices','ceil_mode'],'AvgPool2d':['kernel_size','stride','padding','ceil_mode','count_include_pad'],'BatchNorm2d':['num_features','eps','momentum','affine','track_running_stats'],'AdaptiveAvgPool2d':['output_size'],'Flatten':['start_dim','end_dim'],'Dropout':['p','inplace'],'ReLU':['inplace'],'GELU':['approximate'],'SiLU':['inplace'],'LayerNorm':['normalized_shape','eps','elementwise_affine','bias'],'Embedding':['num_embeddings','embedding_dim'],'TransformerEncoderLayer':['d_model','nhead','dim_feedforward','dropout','activation','layer_norm_eps','batch_first','norm_first','bias'],'TransformerEncoder':['encoder_layer','num_layers'],'Identity':[]}
        if name not in signatures:fail(node,f'Unsupported layer {name}; keep this model in code until the operator is supported')
        if name=='TransformerEncoder':
            if any(k.arg not in signatures[name] for k in node.keywords):fail(node,'Only encoder_layer and num_layers are supported in TransformerEncoder')
            args={k:v for k,v in zip(signatures[name],node.args)};args.update({k.arg:k.value for k in node.keywords})
            base=layer(args['encoder_layer'])
            if len(base)!=1 or base[0][0]!='TransformerEncoder':fail(node,'Encoder stack requires a supported encoder layer')
            return [('TransformerStack',dict(base[0][1],depth=literal(args['num_layers'])),{})]
        args={k:literal(v) for k,v in zip(signatures[name],node.args)}
        for k in node.keywords:
            if k.arg not in signatures[name]:fail(k,'Unsupported constructor argument: '+str(k.arg))
            args[k.arg]=literal(k.value)
        if len(node.args)>len(signatures[name]):fail(node,'Too many constructor arguments')
        defaults={'dilation':1,'groups':1,'padding_mode':'zeros','return_indices':False,'ceil_mode':False,'eps':1e-5,'momentum':.1,'affine':True,'track_running_stats':True,'elementwise_affine':True,'inplace':False,'approximate':'none','layer_norm_eps':1e-5}
        for k,v in defaults.items():
            if k in args and args[k]!=v:fail(node,f'{name}.{k}={args[k]} is not supported without changing semantics')
        checks={};c={}
        if name=='Conv2d':
            c={k:scalar(args.get(k,v),node) for k,v in [('out_channels',16),('kernel_size',3),('stride',1),('padding',0),('bias',True)]};checks['in_channels']=args['in_channels']
        elif name=='Linear':c=dict(out_features=args['out_features'],bias=args.get('bias',True));checks['in_features']=args['in_features']
        elif name in ('MaxPool2d','AvgPool2d'):
            if args.get('padding',0)!=0 or args.get('count_include_pad',True)!=True:fail(node,'Pooling padding / count_include_pad not supported')
            c={'kernel_size':scalar(args['kernel_size'],node)}
            if args.get('stride') is not None:c['stride']=scalar(args['stride'],node)
        elif name=='AdaptiveAvgPool2d':
            if scalar(args.get('output_size',1),node)!=1:fail(node,'Adaptive pooling currently supports output_size=1')
        elif name=='Flatten':
            if args.get('start_dim',1)!=1 or args.get('end_dim',-1)!=-1:fail(node,'Flatten must preserve only the batch dimension')
        elif name=='Dropout':c={'p':args.get('p',.5)}
        elif name=='LayerNorm':checks['normalized_shape']=args['normalized_shape']
        elif name=='BatchNorm2d':checks['num_features']=args['num_features']
        elif name=='Embedding':c=dict(vocab_size=args['num_embeddings'],embed_dim=args['embedding_dim'])
        elif name=='TransformerEncoderLayer':
            if not args.get('batch_first',False) or not args.get('norm_first',False) or not args.get('bias',True):fail(node,'Encoder import requires batch_first=True, norm_first=True, bias=True')
            dim=args['d_model'];ffn=args.get('dim_feedforward',2048)
            if ffn%dim:fail(node,'FFN dimension must be a multiple of d_model')
            activation=args.get('activation','relu').upper()
            if activation not in ('RELU','GELU'):fail(node,'Unsupported encoder activation')
            c=dict(num_heads=args['nhead'],ffn_ratio=ffn//dim,dropout=args.get('dropout',.1),activation='ReLU' if activation=='RELU' else 'GELU',qkv_bias=True);name='TransformerEncoder';checks['hidden_dim']=dim
        return [(name,c,checks)]
    ir=dict(schema_version=1,name=class_name or 'Imported PyTorch',input_shape=input_shape,nodes=nodes,output='')
    if input_type=='tokens':ir.update(input_type='tokens',vocab_size=256)
    checks_all={}
    def emit(op,c,ins,hint,checks=None):
        nonlocal counter
        counter+=1;ident=re.sub(r'\W','_',hint).strip('_')[:30] or 'node'
        if ident in {n['id'] for n in nodes} or ident=='input' or hasattr(__import__('torch').nn.ModuleDict(),ident):ident='op_'+ident+'_'+str(counter)
        nodes.append(dict(id=ident,op=op,config=c,inputs=ins));checks_all[ident]=checks or {};return ident
    def use(key,arg,node):
        if key not in modules:fail(node,'Unknown module '+key)
        layers=modules[key]
        if key in used and any(op not in ['ReLU','GELU','SiLU','Identity','Flatten','Dropout','MaxPool2d','AvgPool2d','AdaptiveAvgPool2d'] for op,c,k in layers):fail(node,'Shared-weight modules cannot be cloned safely; import a non-shared graph')
        used.add(key)
        for i,(op,c,checks) in enumerate(layers):arg=emit(op,c,[arg],key.replace('self.','').replace('.','_')+('_'+str(i) if len(layers)>1 else ''),checks)
        return arg
    variables={}
    def expr(node):
        if isinstance(node,ast.Name) and node.id in variables:return variables[node.id]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return emit('Add',{},[expr(node.left),expr(node.right)],'add')
        if not isinstance(node,ast.Call):fail(node,'Unsupported forward expression')
        key=path(node.func);tail=key.split('.')[-1]
        if key in modules:return use(key,expr(node.args[0]),node)
        if tail in ('relu','gelu','silu'):
            if node.keywords:fail(node,'Functional activation keyword arguments are not supported')
            return emit({'relu':'ReLU','gelu':'GELU','silu':'SiLU'}[tail],{},[expr(node.args[0])],tail)
        if tail=='flatten':
            method=isinstance(node.func,ast.Attribute) and not key.startswith(('torch.','F.','functional.'))
            arg=node.func.value if method else node.args[0];rest=node.args if method else node.args[1:]
            kw={k.arg:literal(k.value) for k in node.keywords}
            if set(kw)-{'start_dim','end_dim'}:fail(node,'Unsupported flatten argument')
            start=literal(rest[0]) if rest else kw.get('start_dim',0)
            end=literal(rest[1]) if len(rest)>1 else kw.get('end_dim',-1)
            if start!=1 or end!=-1:fail(node,'flatten must use start_dim=1, end_dim=-1')
            return emit('Flatten',{},[expr(arg)],'flatten')
        if tail in ('view','reshape') and len(node.args)==2 and literal(node.args[1])==-1:
            first=node.args[0]
            if not (isinstance(first,ast.Call) and isinstance(first.func,ast.Attribute) and first.func.attr=='size' and len(first.args)==1 and literal(first.args[0])==0 and ast.dump(first.func.value)==ast.dump(node.func.value)):fail(node,'reshape/view must use x.size(0), -1 to preserve batch')
            return emit('Flatten',{},[expr(node.func.value)],'flatten')
        if tail=='cat':
            dim=literal(node.args[1]) if len(node.args)>1 else next((literal(k.value) for k in node.keywords if k.arg=='dim'),0)
            if dim!=1:fail(node,'Concatenation requires dim=1')
            return emit('Concat',{},[expr(v) for v in node.args[0].elts],'concat')
        if tail=='mean' and len(node.keywords)==1 and node.keywords[0].arg=='dim' and literal(node.keywords[0].value)==1:return emit('TokenPooling',{'mode':'mean'},[expr(node.func.value)],'mean_pool')
        fail(node,'Unsupported forward call '+key)
    classes=[c for c in tree.body if isinstance(c,ast.ClassDef)]
    chosen=next((c for c in classes if class_name and c.name==class_name),None) if class_name else next((c for c in classes if any(isinstance(m,ast.FunctionDef) and m.name=='forward' for m in c.body)),None)
    if chosen:
        ir['name']=chosen.name
        init=next((m for m in chosen.body if isinstance(m,ast.FunctionDef) and m.name=='__init__'),None)
        if init:
            params=init.args.args[1:];defaults=init.args.defaults
            if len(params)>len(defaults):fail(init,'Constructor parameters require literal defaults')
            for p,d in zip(params[-len(defaults):] if defaults else [],defaults):env[p.arg]=literal(d)
            for stmt in init.body:
                if isinstance(stmt,(ast.Expr,ast.Pass)):continue
                if not isinstance(stmt,ast.Assign) or len(stmt.targets)!=1:fail(stmt,'Only static layer declarations are supported in __init__')
                key=path(stmt.targets[0])
                if isinstance(stmt.value,ast.Call):modules[key]=layer(stmt.value)
                else:env[key]=literal(stmt.value)
        forward=next(m for m in chosen.body if isinstance(m,ast.FunctionDef) and m.name=='forward')
        if len(forward.args.args)!=2:fail(forward,'Exactly one model input is supported')
        variables[forward.args.args[1].arg]='input'
        for stmt in forward.body:
            if isinstance(stmt,ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name):variables[stmt.targets[0].id]=expr(stmt.value)
            elif isinstance(stmt,ast.Return):ir['output']=expr(stmt.value)
            elif isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant):continue
            else:fail(stmt,'Dynamic control flow, loops, and in-place mutation are not supported')
    else:
        for stmt in tree.body:
            if isinstance(stmt,ast.Assign) and isinstance(stmt.value,ast.Call) and path(stmt.value.func).endswith('Sequential'):
                modules['model']=layer(stmt.value);ir['output']=use('model','input',stmt);break
        if not ir['output']:raise ValueError('No supported nn.Module.forward or nn.Sequential found')
    spec=engine.analyze(ir)
    for n in spec['nodes']:
        expected=checks_all[n['id']];shape=n['input_shapes'][0]
        for key,value in expected.items():
            actual=shape[0] if key in ('in_channels','num_features') else shape[-1]
            target=scalar(value,None)
            if target!=actual:raise ValueError(f'{n["id"]}: declared {key}={value}, inferred {actual}')
    embeds=[n for n in nodes if n['op']=='Embedding']
    if embeds:ir['vocab_size']=embeds[0]['config']['vocab_size']
    return dict(ir=ir,analysis=spec,mode='static AST / source never executed',warnings=['Weights are newly initialized; source checkpoint is not imported.'],source=source)


def diff(left,right):
    a,b=search_space.index_nodes(left),search_space.index_nodes(right);sa=engine.analyze(left);sb=engine.analyze(right)
    changed=[]
    for key in sorted(a.keys()&b.keys()):
        fields={k:{'before':a[key].get(k),'after':b[key].get(k)} for k in ('op','name','config','inputs','bindings') if a[key].get(k)!=b[key].get(k)}
        if fields:changed.append(dict(id=key,fields=fields))
    edges=lambda ns:{(prefix+i,path) for path,n in ns.items() for i in n['inputs'] for prefix in [path.rsplit('/',1)[0]+'/' if '/' in path else '']}
    ea,eb=edges(a),edges(b)
    return dict(added=sorted(b.keys()-a.keys()),removed=sorted(a.keys()-b.keys()),changed=changed,edges_added=sorted(eb-ea),edges_removed=sorted(ea-eb),before={k:sa[k] for k in ('params','macs','output_shape')},after={k:sb[k] for k in ('params','macs','output_shape')},input_changed=left['input_shape']!=right['input_shape'])


def edit(ir,action,node_id,op='Identity',config=None,target=None):
    model=copy.deepcopy(ir);nodes=model['nodes'];source=next((n for n in nodes if n['id']==node_id),None)
    if node_id!='input' and not source:raise ValueError('Select a source node')
    spec=engine.analyze(model);shape=model['input_shape'] if node_id=='input' else next(n['shape'] for n in spec['nodes'] if n['id']==node_id)
    if action=='insert':
        c=copy.deepcopy(config or {})
        if op=='Linear' and len(shape)==3:
            flattened=edit(model,'insert',node_id,'Flatten');model=flattened['ir'];node_id=flattened['inserted'];nodes=model['nodes']
        ident='insert_'+str(len(nodes)+1)
        while any(n['id']==ident for n in nodes):ident+='x'
        consumers=[n for n in nodes if node_id in n['inputs']]
        for n in consumers:n['inputs']=[ident if i==node_id else i for i in n['inputs']]
        nodes.append(dict(id=ident,op=op,inputs=[node_id],config=c))
        if model['output']==node_id:model['output']=ident
        result=dict(ir=model,inserted=ident)
    elif action=='connect':
        dest=next((n for n in nodes if n['id']==target),None)
        if not dest:raise ValueError('Target node not found')
        if dest['op'] in ('Add','Concat'):dest['inputs']=list(dict.fromkeys(dest['inputs']+[node_id]))
        else:dest['inputs']=[node_id]
        # Orphan nodes are rejected, never silently removed.
        result=dict(ir=model,inserted=None)
    else:raise ValueError('Unknown edit action')
    result['analysis']=engine.analyze(model);return result
