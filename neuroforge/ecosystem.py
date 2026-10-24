"""Curated ecosystem adapters preserve real modules and local weight snapshots."""
import copy,functools,hashlib,json,os,re,uuid,time
from pathlib import Path
import torch
from torch import nn
from .asset_ops import asset_root,metadata

CATALOG={
 'torchvision':{'resnet18':'Residual CNN / editable stage boundaries','resnet34':'Residual CNN / editable stage boundaries','resnet50':'Residual CNN / editable stage boundaries','efficientnet_b0':'EfficientNet / protected backbone','convnext_tiny':'ConvNeXt / protected backbone','vit_b_16':'Vision Transformer / protected backbone','swin_t':'Swin Transformer / protected backbone'},
 'timm':{'resnet18':'Residual CNN','efficientnet_b0':'EfficientNet','convnext_tiny':'ConvNeXt','vit_tiny_patch16_224':'Vision Transformer','deit_tiny_patch16_224':'DeiT','swin_tiny_patch4_window7_224':'Swin'},
 'huggingface':{'hf-internal-testing/tiny-random-bert':'Small BERT integration fixture; not a trained capability benchmark','google/bert_uncased_L-2_H-128_A-2':'BERT encoder','openai-community/gpt2':'GPT-2 sequence classification; no generation UI','google/vit-base-patch16-224-in21k':'Vision Transformer'}}


class HFClassifier(nn.Module):
    def __init__(self,model,input_type):super().__init__();self.model=model;self.input_type=input_type
    def forward(self,x):
        if self.input_type=='tokens':return self.model(input_ids=x,attention_mask=torch.ones_like(x)).logits
        kwargs={'pixel_values':x}
        if self.model.config.model_type=='vit':kwargs['interpolate_pos_encoding']=True
        return self.model(**kwargs).logits


def _builder(m):
    provider=m['provider'];name=m['model'];kwargs=m.get('model_kwargs',{})
    if provider=='torchvision':
        import torchvision.models as models
        model=models.get_model(name,weights=None,**kwargs)
        return strip_head(model,name) if m.get('feature_adapter') else model
    if provider=='timm':
        import timm
        return timm.create_model(name,pretrained=False,**kwargs)
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoModelForImageClassification
    config=copy.deepcopy(m['hf_config']);kind=config.pop('model_type');cfg=AutoConfig.for_model(kind,**config)
    klass=AutoModelForSequenceClassification if m['input_type']=='tokens' else AutoModelForImageClassification
    return HFClassifier(klass.from_config(cfg,trust_remote_code=False),m['input_type'])


@functools.lru_cache(maxsize=2)
def _base(ident):
    m=metadata('model',ident);model=_builder(m)
    if hashlib.sha256((asset_root()/'models'/ident/'weights.pt').read_bytes()).hexdigest()!=m['weights_sha256']:raise ValueError('Imported weight snapshot checksum mismatch')
    model.load_state_dict(torch.load(asset_root()/'models'/ident/'weights.pt',map_location='cpu',weights_only=True));return model.eval()


def load_block(ident,path=''):
    m=metadata('model',ident)
    if path not in m['contracts']:raise ValueError('Unregistered model boundary')
    return copy.deepcopy(_base(ident).get_submodule(path) if path else _base(ident))


def catalog():
    import importlib.util
    modules={'torchvision':'torchvision','timm':'timm','huggingface':'transformers'}
    return [dict(provider=p,installed=importlib.util.find_spec(modules[p]) is not None,models=[dict(id=k,description=v) for k,v in names.items()]) for p,names in CATALOG.items()]


def strip_head(model,name):
    if name.startswith('convnext'):model.classifier[-1]=nn.Identity()
    elif name.startswith('efficientnet'):model.classifier=nn.Identity()
    elif name.startswith('vit_'):model.heads=nn.Identity()
    elif name.startswith('swin'):model.head=nn.Identity()
    else:raise ValueError('No feature adapter for this family')
    return model


def import_model(ident,p):
    from . import store,engine
    provider,name=p['provider'],p['model'];pretrained=p.get('pretrained',False)
    if provider not in CATALOG:raise ValueError('Unsupported model provider')
    if provider!='huggingface' and name not in CATALOG[provider]:raise ValueError('Choose a curated model')
    if provider=='huggingface' and not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',name):raise ValueError('Use a public Hugging Face repository ID')
    if provider=='huggingface' and not p.get('hf_config') and not p.get('allow_download'):raise ValueError('Confirm downloading the public model configuration')
    if pretrained and not p.get('allow_download'):raise ValueError('Confirm the pretrained weight download before importing')
    shape=p.get('input_shape',[3,32,32]);classes=p.get('classes',3);torch.manual_seed(p.get('seed',42));input_type='image';kwargs={};hf_config=None;notes=[];feature_adapter=False
    if provider=='torchvision':
        torch.hub.set_dir(str(asset_root()/'torch_cache'))
        import torchvision.models as tv
        kwargs={'num_classes':1000}
        if name.startswith('vit_'):kwargs['image_size']=shape[-1]
        if pretrained and name.startswith('vit_') and shape[-1]!=224:raise ValueError('Torchvision pretrained ViT requires 224×224 input; use timm for resized embeddings')
        model=tv.get_model(name,weights='DEFAULT' if pretrained else None,**kwargs)
        if not name.startswith('resnet'):model=strip_head(model,name);feature_adapter=True
        notes.append('Provider head is replaced with a new classifier when the selected dataset has a different class count.')
    elif provider=='timm':
        import timm
        kwargs={'num_classes':0}
        if any(k in name for k in ('vit_','deit_','swin_')):kwargs['img_size']=shape[-1]
        model=timm.create_model(name,pretrained=pretrained,**kwargs)
    else:
        from transformers import AutoConfig,AutoModelForSequenceClassification,AutoModelForImageClassification
        if p.get('hf_config'):
            if pretrained:raise ValueError('Explicit local HF config cannot be combined with remote pretrained weights')
            values=copy.deepcopy(p['hf_config']);kind=values.pop('model_type');cfg=AutoConfig.for_model(kind,**values)
        else:cfg=AutoConfig.from_pretrained(name,trust_remote_code=False,cache_dir=str(asset_root()/'model_cache'))
        if cfg.model_type not in ('bert','gpt2','vit','swin'):raise ValueError('Supported HF adapters: BERT, GPT-2 classification, ViT and Swin')
        input_type='tokens' if cfg.model_type in ('bert','gpt2') else 'image';cfg.num_labels=classes
        if cfg.model_type=='gpt2' and cfg.pad_token_id is None:cfg.pad_token_id=cfg.eos_token_id
        klass=AutoModelForSequenceClassification if input_type=='tokens' else AutoModelForImageClassification
        if pretrained:
            base,loading=klass.from_pretrained(name,config=cfg,trust_remote_code=False,use_safetensors=True,ignore_mismatched_sizes=True,output_loading_info=True,cache_dir=str(asset_root()/'model_cache'))
            notes.append('New / mismatched classifier parameters: '+str(loading.get('missing_keys',[])+loading.get('mismatched_keys',[]))[:1200])
        else:base=klass.from_config(cfg,trust_remote_code=False)
        hf_config=base.config.to_dict();model=HFClassifier(base,input_type)
    model=model.cpu().eval()
    if sum(v.numel() for v in model.parameters())>250000000:raise ValueError('Model exceeds the 250M local asset budget')
    if provider=='huggingface' and input_type=='tokens':x=torch.zeros(2,*shape,dtype=torch.long)
    else:x=torch.zeros(2,*shape)
    if input_type=='image' and len(shape)!=3 or input_type=='tokens' and len(shape)!=1:raise ValueError('Selected model and input type do not match')
    contracts={};nodes=[]
    def add(path,module,input_value):
        with torch.no_grad():value=module(input_value)
        contracts[path]=dict(input_shape=list(input_value.shape[1:]),output_shape=list(value.shape[1:]),params=sum(v.numel() for v in module.parameters()))
        return value
    asset_id=uuid.uuid4().hex[:16];root=asset_root()/'models'/asset_id;root.mkdir(parents=True)
    if provider=='torchvision' and name.startswith('resnet'):
        prev='input';value=x
        for path in ('conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool'):
            value=add(path,model.get_submodule(path),value);nid='backbone_'+path
            nodes.append(dict(id=nid,name=path,op='EcosystemBlock',inputs=[prev],config={'asset_id':asset_id,'module_path':path}));prev=nid
        nodes.append(dict(id='flatten',op='Flatten',inputs=[prev],config={}));prev='flatten';dim=value.flatten(1).shape[1]
        if classes==model.fc.out_features:
            add('fc',model.fc,value.flatten(1));nodes.append(dict(id='classifier',name='Pretrained classifier' if pretrained else 'Classifier',op='EcosystemBlock',inputs=[prev],config={'asset_id':asset_id,'module_path':'fc'}))
        else:nodes.append(dict(id='classifier',op='Linear',inputs=[prev],config={'out_features':classes}))
    else:
        value=add('',model,x);nodes=[dict(id='backbone',name=name,op='EcosystemBlock',inputs=['input'],config={'asset_id':asset_id,'module_path':''})]
        if value.ndim!=2:raise ValueError('Adapter did not return classification features')
        if value.shape[-1]!=classes:nodes.append(dict(id='classifier',op='Linear',inputs=['backbone'],config={'out_features':classes}));notes.append('Pooled provider features feed a newly initialized task classifier.')
    m=dict(id=asset_id,provider=provider,model=name,pretrained=pretrained,input_type=input_type,input_shape=shape,classes=classes,model_kwargs=kwargs,feature_adapter=feature_adapter,hf_config=hf_config,contracts=contracts,notes=notes,created_at=time.time(),module_tree=[{'path':path,'type':type(mod).__name__,'parameters':sum(p.numel() for p in mod.parameters(recurse=False))} for path,mod in model.named_modules()][:2000],weight_status='provider pretrained snapshot' if pretrained else 'initialized snapshot',protected=True)
    torch.save(model.state_dict(),root/'weights.pt');m['weights_sha256']=hashlib.sha256((root/'weights.pt').read_bytes()).hexdigest();(root/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    ir=dict(schema_version=1,name=name,input_shape=shape,nodes=nodes,output=nodes[-1]['id'])
    if input_type=='tokens':ir.update(input_type='tokens',vocab_size=hf_config.get('vocab_size',256))
    spec=engine.analyze(ir);store.put('model_asset',dict(m,ir=ir,params=spec['params']))
    store.update('job',ident,status='completed',progress=1,result=dict(asset=m,ir=ir,analysis=spec),finished_at=time.time())


def replacement(ir,path,width=32):
    from . import search_space,engine
    model=copy.deepcopy(ir);node=search_space.index_nodes(model).get(path)
    if not node or node['op']!='EcosystemBlock':raise ValueError('Choose a protected ecosystem block')
    m=metadata('model',node['config']['asset_id']);c=m['contracts'][node['config'].get('module_path','')];s,t=c['input_shape'],c['output_shape']
    if len(s)!=3 or len(t)!=3 or not 1<=width<=1024:raise ValueError('Searchable replacement currently supports image feature-map boundaries')
    ratio=s[1]/t[1]
    if s[2]/t[2]!=ratio or ratio not in (1,2,4):raise ValueError('This spatial contract needs a custom adapter')
    stride=int(ratio);inner=dict(schema_version=1,input_shape=s,input_ports=[{'id':'input','shape':s}],nodes=[dict(id='bottleneck',op='ConvBlock',inputs=['input'],config={'out_channels':width,'kernel_size':3,'stride':stride,'padding':1,'norm':'BatchNorm','activation':'GELU'}),dict(id='projection',op='Conv2d',inputs=['bottleneck'],config={'out_channels':t[0],'kernel_size':1,'padding':0})],output='projection')
    node.update(op='Module',name='Searchable replacement',config={'width':width},module=inner,bindings={'width':[{'node':'bottleneck','parameter':'out_channels'}]})
    spec=engine.analyze(model)
    actual=next(n['shape'] for n in spec['nodes'] if n['id']==path) if '/' not in path else engine.analyze(inner,_level=1)['output_shape']
    if actual!=t:raise ValueError('Replacement does not preserve the output contract')
    return model
