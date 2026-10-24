"""CUDA discovery, synchronized telemetry, and per-method proxy execution."""
import gc
import math
import os
import time
import torch
from torch import nn
from torch.nn import functional as F


def devices():
    result=[dict(id='auto',name='Auto'),dict(id='cpu',name='CPU')]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            result.append(dict(id=f'cuda:{i}',name=p.name,total_memory_mb=p.total_memory/2**20,capability=list(torch.cuda.get_device_capability(i))))
    return result


def resolve(value='auto'):
    if value=='auto':value='cuda:0' if torch.cuda.is_available() else 'cpu'
    if value=='multi-gpu':raise ValueError('Multi-GPU is not enabled. Choose a single CUDA device.')
    if value=='cuda':value='cuda:0'
    if value!='cpu' and not (value.startswith('cuda:') and value[5:].isdigit()):raise ValueError('Device must be auto, cpu or cuda:N')
    if value.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable. Install a CUDA build of PyTorch and check the NVIDIA driver.')
        if int(value[5:])>=torch.cuda.device_count():raise ValueError('The selected CUDA device does not exist.')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    return torch.device(value)


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def memory(device):
    if device.type!='cuda':return dict(memory_current_mb=None,memory_peak_mb=None,memory_reserved_mb=None)
    free,total=torch.cuda.mem_get_info(device)
    return dict(memory_current_mb=torch.cuda.memory_allocated(device)/2**20,memory_peak_mb=torch.cuda.max_memory_allocated(device)/2**20,
                memory_reserved_mb=torch.cuda.memory_reserved(device)/2**20,memory_device_used_mb=(total-free)/2**20,memory_total_mb=total/2**20)


def error_context(exc,device=None):
    text=str(exc);lower=text.lower()
    code='oom' if 'out of memory' in lower else 'cuda_unavailable' if 'cuda' in lower and ('available' in lower or 'Unavailable' in lower) else 'driver_mismatch' if any(v in lower for v in ('driver','no kernel image','not compiled with cuda')) else 'device_error' if any(v in lower for v in ('same device','device-side','tensor migration')) else 'execution_error'
    hints={'oom':'GPU out of memory. Reduce batch size, channel width, sequence length or depth.','cuda_unavailable':'Check the CUDA build of PyTorch and the NVIDIA driver.','driver_mismatch':'The driver does not match this PyTorch CUDA build.','device_error':'Check that model and inputs use the same device.'}
    return dict(code=code,message=text,hint=hints.get(code,'Inspect the task logs for details.'),device=str(device) if device else None)


def proxy(ir,x,y,seed=42,device='cpu',selected=None):
    from . import engine
    start=time.perf_counter();spec=engine.analyze(ir);dev=resolve(device)
    x,y=x.to(dev),y.to(dev);scores={};methods={};timings={}
    from .proxies import REGISTRY,compute
    selected=selected or ['snip','synflow','naswot']
    if not selected or len(set(selected))!=len(selected) or any(n not in REGISTRY for n in selected):raise ValueError('Select known unique proxy algorithms')
    for name in selected:
        hooks=[];model=None;now=time.perf_counter()
        if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
        try:
            torch.manual_seed(seed)
            if dev.type=='cuda':torch.cuda.manual_seed_all(seed)
            model=engine.GraphModel(ir).to(dev).eval();sync(dev);now=time.perf_counter()
            if name=='snip':
                F.cross_entropy(model(x),y).backward()
                score=sum((p*p.grad).abs().sum().item() for p in model.parameters() if p.grad is not None)
            elif name=='synflow':
                model=model.double()
                with torch.no_grad():
                    for p in model.parameters():p.abs_()
                ones=torch.ones(1,*ir['input_shape'],device=dev,dtype=torch.long if ir.get('input_type')=='tokens' else torch.float64)
                model(ones).sum().backward()
                score=sum((p*p.grad).abs().sum().item() for p in model.parameters() if p.grad is not None)
            elif name=='naswot':
                kernels=[]
                def capture(module,args,out):
                    a=(args[0].detach().flatten(1)>0).float();kernels.append(a@a.T+(1-a)@(1-a).T)
                for layer in model.modules():
                    if isinstance(layer,nn.ReLU):hooks.append(layer.register_forward_hook(capture))
                with torch.no_grad():model(x)
                if not kernels:raise ValueError('NASWOT is unavailable because this architecture has no ReLU.')
                sign,val=torch.linalg.slogdet(sum(kernels).double())
                if sign<=0 or not torch.isfinite(val):raise ValueError('NASWOT has a singular activation kernel.')
                score=val.item()
            else:
                score=compute(name,model,x,y)
            sync(dev)
            if not math.isfinite(score):raise ValueError('Proxy produced a nonfinite score.')
            scores[name]=score;methods[name]=dict(status='completed',score=score)
        except Exception as exc:
            context=error_context(exc,dev)
            if context['code'] in ('oom','driver_mismatch','device_error'):raise
            scores[name]=None;methods[name]=dict(status='unavailable',reason=str(exc))
        finally:
            for h in hooks:h.remove()
            sync(dev)
            timings[name]=time.perf_counter()-now
            methods.setdefault(name,{}).update(device=str(dev),seconds=timings[name],**memory(dev))
            del model
            if dev.type=='cuda':torch.cuda.empty_cache()
    telemetry=memory(dev)
    if dev.type=='cuda':telemetry['memory_peak_mb']=max(m.get('memory_peak_mb',0) for m in methods.values())
    return dict(scores=scores,method_seconds=timings,methods=methods,seconds=time.perf_counter()-start,
                static_macs_available=spec.get('static_macs_available',True),params=spec['params'],macs=spec['macs'],flops=spec['flops'],seed=seed,batch_size=len(x),
                device=str(dev),device_name=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU',
                initialization='Preserved imported weights / seeded new operators' if not spec.get('static_macs_available',True) else 'PyTorch defaults / fixed seed',implementation_version=engine.ENGINE_VERSION,
                torch_version=str(torch.__version__),validity='uncalibrated',**telemetry)
