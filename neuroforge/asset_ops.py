"""Declarative contracts for protected ecosystem and trusted custom blocks."""
import os,json,re
from pathlib import Path


def asset_root():return Path(os.environ.get('NEUROFORGE_DATA',Path(__file__).resolve().parents[1]/'runtime'))


def metadata(kind,ident):
    if not isinstance(ident,str) or not re.fullmatch('[a-f0-9]{16}',ident):raise ValueError('Invalid local asset ID')
    path=asset_root()/('models' if kind=='model' else 'custom_operators')/ident/'metadata.json'
    if not path.exists():raise ValueError('Local operator/model asset is missing')
    return json.loads(path.read_text(encoding='utf-8'))


def contract(n,input_shape):
    c=n['config'];kind='model' if n['op']=='EcosystemBlock' else 'operator';m=metadata(kind,c.get('asset_id') or c.get('operator_id'))
    if kind=='operator' and not m.get('enabled'):raise ValueError('This custom operator is disabled')
    key=c.get('module_path','') if kind=='model' else ''
    if kind=='operator' and c.get('parameters',{})!=m['parameters']:raise ValueError('Custom configuration changed: register and validate a new operator version')
    spec=m['contracts'].get(key)
    if not spec:raise ValueError('Unsupported module boundary')
    if input_shape!=spec['input_shape']:raise ValueError(f'Protected block expects {spec["input_shape"]}; received {input_shape}. Use a contract-preserving replacement.')
    return spec['output_shape'],spec['params'],0,{'protected':True,'static_macs_available':False,'asset_id':m['id'],'source':m.get('provider','custom')}


def layer(n):
    if n['op']=='EcosystemBlock':
        from .ecosystem import load_block
        return load_block(n['config']['asset_id'],n['config'].get('module_path',''))
    from .custom_ops import load_operator
    return load_operator(n['config']['operator_id'],n['config'].get('parameters',{}))


def asset_export_files(ir):
    from . import search_space,store
    nodes=search_space.index_nodes(ir);assets={};needed=False
    for node in nodes.values():
        if node['op'] not in ('EcosystemBlock','CustomOperator'):continue
        needed=True;custom=node['op']=='CustomOperator';ident=node['config']['operator_id' if custom else 'asset_id'];folder='custom_operators' if custom else 'models'
        from .asset_ops import metadata
        metadata('operator' if custom else 'model',ident)
        for path in (store.ROOT/folder/ident).iterdir():
            if path.name in ('metadata.json','weights.pt','operator.py'):assets['runtime/'+folder+'/'+ident+'/'+path.name]=path
    if needed:
        assets['neuroforge/__init__.py']=''
        for name in ('asset_ops.py','ecosystem.py','custom_ops.py'):assets['neuroforge/'+name]=Path(__file__).with_name(name)
        assets['requirements-assets.txt']='torchvision==0.23.0\ntimm==1.0.22\ntransformers==4.57.1\nsafetensors==0.6.2\n'
        assets['ASSETS.txt']='Contains local weight snapshots and explicitly trusted custom operator code. Review custom code before running. Install requirements.txt and requirements-assets.txt, then run model.py from this directory. No provider downloads are needed to rebuild an imported snapshot.\n'
    return assets


