"""Versioned tensor preprocessing pipeline, train-only augmentation and previews."""
import torch
from torch.nn import functional as F

OPS={'flip':{'probability':.5},'rotate':{'quarter_turns':1},'brightness':{'factor':1.2},'contrast':{'factor':1.2},'noise':{'std':.03},'cutout':{'size':8},'crop':{'padding':4},'normalize':{'mean':.5,'std':.5},'resize':{'size':32}}
RANDOM={'flip','noise','cutout','crop','rotate'}


def validate(steps,shape=None):
    if not isinstance(steps,list) or len(steps)>16:raise ValueError('Pipeline requires at most 16 steps')
    for step in steps:
        op=step.get('op');params=step.get('params',{})
        if op not in OPS or not isinstance(params,dict) or set(params)-set(OPS[op]):raise ValueError('Unsupported pipeline operator / parameter')
        if step.get('scope','train' if op in RANDOM else 'all') not in ('train','all'):raise ValueError('Pipeline scope must be train or all')
        p=dict(OPS[op],**params)
        if any(type(v) not in (int,float) or not torch.isfinite(torch.tensor(float(v))) for v in p.values()):raise ValueError('Pipeline parameters must be finite')
        if op=='flip' and not 0<=p['probability']<=1:raise ValueError('Flip probability must be 0..1')
        if op=='rotate' and p['quarter_turns'] not in (0,1,2,3):raise ValueError('Rotation must use 0..3 quarter turns')
        if op in ('brightness','contrast') and not 0<=p['factor']<=3:raise ValueError('Factor must be 0..3')
        if op=='noise' and not 0<=p['std']<=1:raise ValueError('Noise std must be 0..1')
        if op=='normalize' and not .001<=p['std']<=10:raise ValueError('Normalize std must be positive')
        if op in ('cutout','crop','resize'):
            key='padding' if op=='crop' else 'size';lo=0 if op=='crop' else 1
            if type(p[key])!=int or not lo<=p[key]<=128:raise ValueError(key+' must be an integer within runtime bounds')
        if op=='resize' and shape and shape[-2:]!=[p['size'],p['size']]:raise ValueError('Resize must match model input H/W; edit the model first')
    return steps


def apply(x,steps,seed=42,training=True):
    if not steps:return x
    if x.ndim!=4 or not x.is_floating_point():raise ValueError('Image pipeline requires N,C,H,W floating tensors')
    validate(steps);out=x.clone();g=torch.Generator(device=x.device).manual_seed(seed)
    for step in steps:
        op=step['op'];p=dict(OPS[op],**step.get('params',{}))
        if not training and step.get('scope','train' if op in RANDOM else 'all')=='train':continue
        if op=='flip':
            mask=torch.rand(len(x),generator=g,device=x.device)<p['probability'];out[mask]=out[mask].flip(-1)
        elif op=='rotate':out=torch.rot90(out,int(p['quarter_turns']),(-2,-1))
        elif op=='brightness':out=(out*p['factor']).clamp(0,1)
        elif op=='contrast':
            mean=out.mean((-2,-1),keepdim=True);out=((out-mean)*p['factor']+mean).clamp(0,1)
        elif op=='noise':out=(out+torch.randn(out.shape,generator=g,device=x.device)*p['std']).clamp(0,1)
        elif op=='normalize':out=(out-p['mean'])/p['std']
        elif op=='resize':out=F.interpolate(out,size=(p['size'],p['size']),mode='bilinear',align_corners=False)
        elif op=='cutout':
            h,w=out.shape[-2:];size=min(p['size'],h,w)
            for v in out:
                top=int(torch.randint(h-size+1,(),generator=g,device=x.device));left=int(torch.randint(w-size+1,(),generator=g,device=x.device));v[:,top:top+size,left:left+size]=0
        elif op=='crop':
            pad=p['padding'];h,w=out.shape[-2:];padded=F.pad(out,(pad,pad,pad,pad),mode='replicate')
            for i in range(len(out)):
                top=int(torch.randint(2*pad+1,(),generator=g,device=x.device));left=int(torch.randint(2*pad+1,(),generator=g,device=x.device));out[i]=padded[i,:,top:top+h,left:left+w]
    return out
