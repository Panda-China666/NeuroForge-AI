"""Modern sequence operators and reusable modules, shared by compiler and exports."""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

MODERN_OPS=['ConvBlock','GELU','SiLU','Identity','Embedding','PatchEmbedding','ClassToken',
            'PositionalEmbedding','PositionalEncoding','MultiHeadAttention','FeedForward',
            'LayerNorm','TransformerEncoder','TransformerStack','TokenPooling','Module','EcosystemBlock','CustomOperator']


def vit_template(classes=3,size=32,embed_dim=64,depth=2,num_heads=4,patch_size=4,class_token=True):
    nodes=[dict(id='patch',op='PatchEmbedding',inputs=['input'],config={'patch_size':patch_size,'embed_dim':embed_dim})]
    prev='patch'
    if class_token:
        nodes.append(dict(id='cls_token',op='ClassToken',inputs=[prev],config={}));prev='cls_token'
    nodes += [dict(id='position',op='PositionalEmbedding',inputs=[prev],config={}),
              dict(id='encoder',op='TransformerStack',inputs=['position'],config={'depth':depth,'num_heads':num_heads,'ffn_ratio':4,'dropout':.1,'activation':'GELU','qkv_bias':True}),
              dict(id='norm',op='LayerNorm',inputs=['encoder'],config={}),
              dict(id='pool',op='TokenPooling',inputs=['norm'],config={'mode':'cls' if class_token else 'mean'}),
              dict(id='classifier',op='Linear',inputs=['pool'],config={'out_features':classes})]
    return dict(schema_version=1,name='Vision Transformer',input_shape=[3,size,size],nodes=nodes,output='classifier')


def transformer_template(vocab_size=256,length=32,classes=3,embed_dim=64,depth=2,num_heads=4):
    ir=vit_template(classes=classes,embed_dim=embed_dim,depth=depth,num_heads=num_heads,class_token=False)
    ir.update(name='Transformer Encoder',input_shape=[length],input_type='tokens',vocab_size=vocab_size)
    ir['nodes'][0]=dict(id='patch',op='Embedding',inputs=['input'],config={'vocab_size':vocab_size,'embed_dim':embed_dim})
    return ir


def int_value(c,key,default,lo=1,hi=4096):
    v=c.get(key,default)
    if type(v)!=int or not lo<=v<=hi:raise ValueError(f'{key}: integer in [{lo}, {hi}] required')
    return v


def probability(c,key,default=0.):
    v=c.get(key,default)
    if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<1:raise ValueError(f'{key} must be in [0, 1)')
    return float(v)


def effective_module(n,shapes):
    ir=copy.deepcopy(n.get('module'))
    if not isinstance(ir,dict):raise ValueError('Module definition is missing')
    ports=ir.get('input_ports',[{'id':'input'}])
    if len(ports)!=len(shapes):raise ValueError('Module input port count does not match connected tensors')
    ir['input_shape']=shapes[0];ir['input_ports']=[dict(id=p['id'],shape=s) for p,s in zip(ports,shapes)]
    mapping={v['id']:v for v in ir['nodes']}
    for key,targets in n.get('bindings',{}).items():
        if key in n['config']:
            for target in targets:
                if target['node'] not in mapping:raise ValueError('Exposed parameter references a missing internal node')
                mapping[target['node']]['config'][target['parameter']]=n['config'][key]
    return ir


def modern_shape(n,inputs,analyze,level):
    op,c,s=n['op'],n.get('config',{}),inputs[0];p=m=0;extra={}
    if op not in ('Module',) and len(inputs)!=1:raise ValueError(f'{op} requires one input')
    if op in ('EcosystemBlock','CustomOperator'):
        try:from .asset_ops import contract
        except ImportError:from neuroforge.asset_ops import contract
        return contract(n,s)
    if op=='ConvBlock':
        if len(s)!=3:raise ValueError('ConvBlock requires an image feature map')
        out=int_value(c,'out_channels',16,hi=1024);k=int_value(c,'kernel_size',3,hi=11);st=int_value(c,'stride',1,hi=4)
        pad=int_value(c,'padding',k//2,lo=0,hi=10);h,w=[(x+2*pad-k)//st+1 for x in s[1:]]
        if c.get('norm','BatchNorm') not in ('BatchNorm','LayerNorm','None') or c.get('activation','ReLU') not in ('ReLU','GELU','SiLU','Identity'):raise ValueError('Unsupported ConvBlock normalization or activation')
        p=out*(s[0]*k*k+1)+(2*out if c.get('norm','BatchNorm')!='None' else 0);m=h*w*out*s[0]*k*k;t=[out,h,w]
    elif op=='PatchEmbedding':
        if len(s)!=3:raise ValueError('PatchEmbedding requires [C,H,W]')
        patch=int_value(c,'patch_size',4,hi=64);dim=int_value(c,'embed_dim',64,hi=2048)
        if s[1]%patch or s[2]%patch:raise ValueError('Image dimensions must be divisible by patch_size')
        length=(s[1]//patch)*(s[2]//patch);t=[length,dim];p=dim*(s[0]*patch*patch+1);m=length*dim*s[0]*patch*patch
        extra['patch_count']=length
    elif op=='Embedding':
        if len(s)!=1:raise ValueError('Embedding requires token input [sequence_length]')
        dim=int_value(c,'embed_dim',64,hi=2048);vocab=int_value(c,'vocab_size',256,hi=65536);p=vocab*dim;t=[s[0],dim]
    elif op=='Module':
        if level>=6:raise ValueError('Module nesting exceeds six levels')
        ir=effective_module(n,inputs);spec=analyze(ir,_level=level+1)
        p,m,t=spec['params'],spec['macs'],spec['output_shape']
        extra=dict(static_macs_available=spec.get('static_macs_available',True),module_ir=ir,internal_nodes=spec.get('expanded_nodes',len(spec['nodes'])),nested_activation=spec['activation_memory_mb']*2**20/4)
    elif op in ('GELU','SiLU','Identity'):t=list(s)
    elif op=='LayerNorm':p=2*s[-1];t=list(s)
    else:
        if len(s)!=2:raise ValueError(f'{op} requires [sequence_length, hidden_dim]')
        length,dim=s;t=list(s)
        if op=='ClassToken':p=dim;t=[length+1,dim]
        elif op=='PositionalEmbedding':p=length*dim
        elif op=='PositionalEncoding':pass
        elif op=='TokenPooling':
            if c.get('mode','cls') not in ('cls','mean'):raise ValueError('TokenPooling mode must be cls or mean')
            t=[dim]
        elif op=='FeedForward':
            ratio=c.get('expansion_ratio',4)
            if type(ratio) not in (int,float) or not 1<=ratio<=16:raise ValueError('expansion_ratio must be in [1,16]')
            hidden=int_value(c,'hidden_dim',int(dim*ratio),hi=32768)
            p=2*dim*hidden+hidden+dim;m=2*length*dim*hidden;probability(c,'dropout')
        elif op in ('MultiHeadAttention','TransformerEncoder','TransformerStack'):
            heads=int_value(c,'num_heads',4,hi=32)
            if dim%heads:raise ValueError(f'Hidden dimension {dim} must be divisible by {heads} attention heads')
            depth=int_value(c,'depth',1,hi=24) if op=='TransformerStack' else 1
            if c.get('attention_type','MHA') not in ('MHA','Linear'):raise ValueError('attention_type must be MHA or Linear')
            if type(c.get('qkv_bias',True))!=bool:raise ValueError('qkv_bias must be boolean')
            p=4*dim*dim+(3*dim if c.get('qkv_bias',True) else 0)+dim
            m=4*length*dim*dim+(2*length*length*dim if c.get('attention_type','MHA')=='MHA' else 2*length*dim*dim//heads)
            for key in ('dropout','attn_dropout','proj_dropout'):probability(c,key)
            if op!='MultiHeadAttention':
                ratio=c.get('ffn_ratio',4)
                if type(ratio) not in (int,float) or not 1<=ratio<=16:raise ValueError('ffn_ratio must be in [1,16]')
                hidden=int(dim*ratio);p+=2*dim*hidden+hidden+dim+4*dim;m+=2*length*dim*hidden
            p*=depth;m*=depth
        if c.get('activation','GELU') not in ('GELU','ReLU','SiLU'):raise ValueError('Unsupported FFN activation')
    return t,p,m,extra


def act(name):return {'ReLU':nn.ReLU,'GELU':nn.GELU,'SiLU':nn.SiLU,'Identity':nn.Identity}[name]()


class ChannelLayerNorm(nn.Module):
    def __init__(self,dim):super().__init__();self.norm=nn.LayerNorm(dim)
    def forward(self,x):return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class PatchEmbedding(nn.Module):
    def __init__(self,cin,dim,patch):super().__init__();self.proj=nn.Conv2d(cin,dim,patch,patch)
    def forward(self,x):return self.proj(x).flatten(2).transpose(1,2)


class ClassToken(nn.Module):
    def __init__(self,dim):super().__init__();self.token=nn.Parameter(torch.empty(1,1,dim));nn.init.normal_(self.token,std=.02)
    def forward(self,x):return torch.cat([self.token.expand(x.shape[0],-1,-1),x],1)


class Position(nn.Module):
    def __init__(self,length,dim,learned=True):
        super().__init__()
        if learned:self.position=nn.Parameter(torch.empty(1,length,dim));nn.init.normal_(self.position,std=.02)
        else:
            pos=torch.arange(length).unsqueeze(1);div=torch.exp(torch.arange(0,dim,2)*(-math.log(10000.)/dim));value=torch.zeros(1,length,dim)
            value[0,:,0::2]=torch.sin(pos*div);value[0,:,1::2]=torch.cos(pos*div[:dim//2]);self.register_buffer('position',value)
    def forward(self,x):return x+self.position[:,:x.shape[1]]


class Attention(nn.Module):
    def __init__(self,dim,c):
        super().__init__();self.heads=c.get('num_heads',4);self.kind=c.get('attention_type','MHA')
        self.qkv=nn.Linear(dim,dim*3,bias=c.get('qkv_bias',True));self.proj=nn.Linear(dim,dim)
        self.attn_drop=nn.Dropout(c.get('attn_dropout',c.get('dropout',0.)));self.proj_drop=nn.Dropout(c.get('proj_dropout',c.get('dropout',0.)))
    def forward(self,x):
        b,l,d=x.shape;h=self.heads;k=d//h;q,key,v=self.qkv(x).reshape(b,l,3,h,k).permute(2,0,3,1,4).unbind(0)
        if self.kind=='Linear':
            q,key=F.elu(q)+1,F.elu(key)+1;out=(q@(key.transpose(-2,-1)@v))/(q@key.sum(-2,keepdim=True).transpose(-2,-1)+1e-6)
        else:out=self.attn_drop(((q@key.transpose(-2,-1))*(k**-.5)).softmax(-1))@v
        return self.proj_drop(self.proj(out.transpose(1,2).reshape(b,l,d)))


class FeedForward(nn.Module):
    def __init__(self,dim,c):
        super().__init__();hidden=c.get('hidden_dim',int(dim*c.get('expansion_ratio',c.get('ffn_ratio',4))))
        self.layers=nn.Sequential(nn.Linear(dim,hidden),act(c.get('activation','GELU')),nn.Dropout(c.get('dropout',0.)),nn.Linear(hidden,dim),nn.Dropout(c.get('dropout',0.)))
    def forward(self,x):return self.layers(x)


class Encoder(nn.Module):
    def __init__(self,dim,c):super().__init__();self.norm1=nn.LayerNorm(dim);self.attn=Attention(dim,c);self.norm2=nn.LayerNorm(dim);self.ffn=FeedForward(dim,c)
    def forward(self,x):x=x+self.attn(self.norm1(x));return x+self.ffn(self.norm2(x))


class TokenPooling(nn.Module):
    def __init__(self,mode):super().__init__();self.mode=mode
    def forward(self,x):return x[:,0] if self.mode=='cls' else x.mean(1)


def modern_layer(n,s,GraphModel,extra):
    op,c=n['op'],n['config']
    if op in ('EcosystemBlock','CustomOperator'):
        try:from .asset_ops import layer
        except ImportError:from neuroforge.asset_ops import layer
        return layer(n)
    if op=='ConvBlock':
        out=c.get('out_channels',16);norm=c.get('norm','BatchNorm');k=c.get('kernel_size',3)
        return nn.Sequential(nn.Conv2d(s[0],out,k,c.get('stride',1),c.get('padding',k//2)),nn.BatchNorm2d(out) if norm=='BatchNorm' else ChannelLayerNorm(out) if norm=='LayerNorm' else nn.Identity(),act(c.get('activation','ReLU')))
    if op in ('GELU','SiLU'):return act(op)
    if op=='Identity':return nn.Identity()
    if op=='Embedding':return nn.Embedding(c.get('vocab_size',256),c.get('embed_dim',64))
    if op=='PatchEmbedding':return PatchEmbedding(s[0],c.get('embed_dim',64),c.get('patch_size',4))
    if op=='ClassToken':return ClassToken(s[-1])
    if op in ('PositionalEmbedding','PositionalEncoding'):return Position(*s,learned=op=='PositionalEmbedding')
    if op=='LayerNorm':return nn.LayerNorm(s[-1])
    if op=='MultiHeadAttention':return Attention(s[-1],c)
    if op=='FeedForward':return FeedForward(s[-1],c)
    if op=='TransformerEncoder':return Encoder(s[-1],c)
    if op=='TransformerStack':return nn.Sequential(*[Encoder(s[-1],c) for _ in range(c.get('depth',1))])
    if op=='TokenPooling':return TokenPooling(c.get('mode','cls'))
    if op=='Module':return GraphModel(extra['module_ir'])
    raise ValueError('Unsupported modern operator')
