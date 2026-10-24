"""Real tensor capture and synchronized per-node inference profiling."""
import time
import math
import torch
from torch import nn
from torch.nn import functional as F
from . import engine,runtime,store


def stats(t):
    if t is None:return None
    v=t.detach().float().cpu();flat=v.flatten();finite=flat[torch.isfinite(flat)]
    result=dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),numel=t.numel(),bytes=t.numel()*t.element_size(),nonfinite=int(flat.numel()-finite.numel()))
    if not len(finite):return dict(result,histogram=[],preview=[])
    lo,hi=finite.min().item(),finite.max().item()
    hist=torch.histc(finite,bins=32,min=lo if lo!=hi else lo-.5,max=hi if lo!=hi else hi+.5)
    result.update(min=lo,max=hi,mean=finite.mean().item(),std=finite.std(unbiased=False).item(),zero_fraction=(finite==0).float().mean().item(),l2=finite.norm().item(),histogram=hist.int().tolist(),histogram_range=[lo,hi],preview=[float(v) if torch.isfinite(v) else None for v in flat[:128]])
    sample=v[0] if v.ndim>1 else v
    if sample.ndim==3:
        maps=F.adaptive_avg_pool2d(sample[:8].unsqueeze(0),(min(32,sample.shape[-2]),min(32,sample.shape[-1])))[0]
        result['maps']=maps.tolist();result['map_kind']='feature channels / adaptive mean if >32 pixels'
    elif sample.ndim==2:
        result['maps']=[sample[:64,:64].tolist()];result['map_kind']='sequence / first 64 × 64 values'
    return result


class Observer:
    def __init__(self,prefix,device,rows,tensors,capture):
        self.prefix=prefix;self.device=device;self.rows=rows;self.tensors=tensors;self.capture=capture;self.starts={}
    def input(self,x):
        if self.capture:self.tensors[self.prefix+'input']=x
    def before(self,n):
        if self.device.type=='cuda':
            mem=runtime.memory(self.device)['memory_current_mb'];e=torch.cuda.Event(enable_timing=True);e.record();self.starts[n['id']]=(e,mem)
        else:self.starts[n['id']]=(time.perf_counter(),None)
    def after(self,n,v):
        start,mem=self.starts[n['id']];path=self.prefix+n['id']
        if self.device.type=='cuda':
            end=torch.cuda.Event(enable_timing=True);end.record();end.synchronize();ms=start.elapsed_time(end)
            delta=runtime.memory(self.device)['memory_current_mb']-mem
        else:ms=(time.perf_counter()-start)*1000;delta=None
        row=self.rows.setdefault(path,dict(id=path,op=n['op'],times_ms=[],activation_bytes=v.numel()*v.element_size(),allocation_delta_mb=[]))
        row['times_ms'].append(ms)
        if delta is not None:row['allocation_delta_mb'].append(delta)
        if self.capture:
            if v.requires_grad:v.retain_grad()
            self.tensors[path]=v


def attach(model,dev,rows,tensors,capture,prefix=''):
    model.observer=Observer(prefix,dev,rows,tensors,capture)
    for key,layer in model.layers.items():
        if isinstance(layer,engine.GraphModel):attach(layer,dev,rows,tensors,capture,prefix+key+'/')


def benchmark(model,x,device,warmup=3,repeats=10):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):model(x)
        runtime.sync(device)
        times=[]
        for _ in range(repeats):
            start=time.perf_counter();model(x);runtime.sync(device);times.append((time.perf_counter()-start)*1000)
    ordered=sorted(times);median=ordered[len(ordered)//2]
    return dict(latency_ms=median,p95_ms=ordered[min(len(ordered)-1,math.ceil(.95*len(ordered))-1)],samples_per_second=len(x)/(median/1000),batch_size=len(x),warmup=warmup,repeats=repeats,device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',**runtime.memory(device))


def run(ir,x,y,device='auto',seed=42,checkpoint_job=None,capture=True,repeats=5,ident=None):
    dev=runtime.resolve(device);torch.manual_seed(seed)
    model=engine.GraphModel(ir).to(dev).eval();source='seeded initialization'
    if checkpoint_job:
        j=store.get('job',checkpoint_job)
        if engine.fingerprint(j['payload']['ir'])!=engine.fingerprint(ir):raise ValueError('Checkpoint architecture does not match the current graph')
        cp=store.ARTIFACTS/checkpoint_job/'best.pt'
        if not cp.exists():raise ValueError('Selected run has no saved weights')
        model.load_state_dict(torch.load(cp,map_location=dev,weights_only=True));source='best.pt / '+checkpoint_job
    x=x.to(dev);y=y.to(dev)
    if x.is_floating_point():x.requires_grad_(capture)
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
    bench=benchmark(model,x,dev,repeats=repeats)
    rows,tensors={},{};attach(model,dev,rows,tensors,capture)
    if capture:
        logits=model(x)
        loss=F.cross_entropy(logits,y);loss.backward()
    else:
        with torch.no_grad():
            for _ in range(repeats):model(x)
    result=[];files={};budget=64*2**20
    for path,row in rows.items():
        times=row.pop('times_ms');deltas=row.pop('allocation_delta_mb')
        row.update(milliseconds=sum(times)/len(times),calls=len(times),allocation_delta_mb=max(deltas) if deltas else None,inclusive=row['op']=='Module')
        if capture:
            tensor=tensors[path];row.update(activation=stats(tensor),gradient=stats(tensor.grad))
            if tensor.numel()*tensor.element_size()<=budget:
                files[path]={'activation':tensor.detach().cpu(),'gradient':tensor.grad.detach().cpu() if tensor.grad is not None else None};budget-=tensor.numel()*tensor.element_size()*2
        result.append(row)
    for path,t in tensors.items():
        if path.endswith('input') and capture:
            result.append(dict(id=path,op='Input',activation=stats(t),gradient=stats(t.grad),milliseconds=None,activation_bytes=t.numel()*t.element_size()))
            if t.numel()*t.element_size()*2<=budget:
                files[path]={'activation':t.detach().cpu(),'gradient':t.grad.detach().cpu() if t.grad is not None else None};budget-=t.numel()*t.element_size()*2
    weights=[]
    if capture:
        for name,p in model.named_parameters():
            weights.append(dict(name=name,shape=list(p.shape),weight_norm=p.detach().norm().item(),gradient_norm=p.grad.norm().item() if p.grad is not None else None))
    if ident and capture:
        out=store.ARTIFACTS/ident;out.mkdir(exist_ok=True);torch.save(files,out/'tensors.pt')
    return dict(nodes=result,benchmark=bench,weights=weights,weights_source=source,loss=loss.item() if capture else None,graph_hash=engine.fingerprint(ir),sample_count=len(x),measurement='CUDA events or CPU wall clock; node timings instrumented and inclusive for modules',**runtime.memory(dev))
