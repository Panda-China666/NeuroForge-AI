"""Additional batch-budgeted zero-training scores; no optimizer updates."""
import torch
from torch import nn
from torch.nn import functional as F

REGISTRY={
 'snip':{'label':'SNIP','description':'Weight × supervised gradient sensitivity'},
 'synflow':{'label':'SynFlow','description':'Positive-weight signal flow, FP64'},
 'naswot':{'label':'NASWOT','description':'ReLU activation agreement log determinant'},
 'gradnorm':{'label':'GradNorm','description':'Sum of layer gradient L2 norms'},
 'zico':{'label':'ZiCo','description':'Log inverse gradient variation across 3 fixed microbatches'},
 'fisher':{'label':'Fisher','description':'Channel activation × gradient squared sensitivity'},
 'jacov':{'label':'Jacobian Covariance','description':'Input Jacobian correlation spectrum; floating inputs only'}
}


def compute(name,model,x,y):
    if name=='gradnorm':
        F.cross_entropy(model(x),y).backward()
        return sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    if name=='zico':
        if len(x)<3:raise ValueError('ZiCo requires at least 3 samples')
        layers=[m for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
        gradients=[[] for _ in layers]
        for ids in torch.tensor_split(torch.arange(len(x),device=x.device),3):
            model.zero_grad(set_to_none=True);F.cross_entropy(model(x[ids]),y[ids]).backward()
            for rows,m in zip(gradients,layers):
                if m.weight.grad is not None:rows.append(m.weight.grad.detach().flatten().clone())
        score=0.;valid=0
        for rows in gradients:
            if len(rows)!=3:continue
            g=torch.stack(rows);std=g.std(0,unbiased=False);mask=std>1e-12
            if mask.any():score+=(g.abs().mean(0)[mask]/std[mask]).sum().log().item();valid+=1
        if not valid:raise ValueError('ZiCo found no varying weight gradients')
        return score
    if name=='fisher':
        activations=[];hooks=[]
        def capture(m,args,out):
            if out.requires_grad:out.retain_grad();activations.append(out)
        try:
            for m in model.modules():
                if isinstance(m,(nn.Conv2d,nn.Linear)):hooks.append(m.register_forward_hook(capture))
            F.cross_entropy(model(x),y).backward();score=0.
            for a in activations:
                if a.grad is None:continue
                v=a.detach()*a.grad
                if v.ndim==4:v=v.sum((2,3))
                elif v.ndim==3:v=v.sum(1)
                score+=.5*v.square().mean(0).sum().item()
            return score
        finally:
            for h in hooks:h.remove()
    if name=='jacov':
        if not x.is_floating_point():raise ValueError('Jacobian Covariance requires differentiable floating inputs; token IDs are discrete')
        inputs=x.detach().clone().requires_grad_(True)
        g=torch.autograd.grad(model(inputs).sum(),inputs)[0].flatten(1).double();g=g-g.mean(1,keepdim=True)
        norm=g.norm(dim=1,keepdim=True)
        if (norm<1e-12).any():raise ValueError('Input Jacobian has constant rows')
        corr=(g/norm)@(g/norm).T;eigen=torch.linalg.eigvalsh(corr).clamp_min(1e-5)
        return -(torch.log(eigen)+1/eigen).sum().item()
    raise ValueError('Unknown proxy '+name)
