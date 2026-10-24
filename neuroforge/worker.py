"""One local compute worker, persisted tasks and atomic epoch-boundary checkpoints."""
import copy
import math
import json
import os
import platform
import random
import threading
import time
import traceback
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn
from . import data, engine, store, runtime, search_space

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='neuroforge')
CONTROL_LOCK = threading.RLock()
ACCEPTING_JOBS=True


def dispatch(ident):
    if os.environ.get('NEUROFORGE_INLINE_WORKER')=='1':
        return execute(ident)
    process=subprocess.Popen([sys.executable,'-m','neuroforge.job_process',ident],cwd=str(store.ROOT.parent if (store.ROOT.parent/'neuroforge').exists() else __import__('pathlib').Path(__file__).resolve().parents[1]),
                             env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    store.update('job',ident,worker_pid=process.pid)
    code=process.wait()
    j=store.get('job',ident)
    if j.get('worker_pid')==process.pid and (j['status']=='running' or code!=0 and j['status']=='queued'):
        store.update('job',ident,status='failed',error=f'Worker exited unexpectedly ({code}). Logs and checkpoints are preserved.',finished_at=time.time())
    elif j.get('worker_pid')==process.pid and j['status'] in ('completed','paused','failed','cancelled'):
        store.update('job',ident,worker_exit_code=code,memory_current_mb=0 if str(j.get('device','')).startswith('cuda') else None,
                     memory_reserved_mb=0 if str(j.get('device','')).startswith('cuda') else None)


def recover():
    for j in store.all_of('job'):
        if j['status'] in ('running','queued','pausing','cancelling'):
            store.update('job', j['id'], status='paused', error='Server stopped. Training can resume from the latest completed epoch.')


def log(ident, text):
    j = store.get('job', ident)
    store.update('job', ident, logs=(j.get('logs', [])+[{'time':time.time(),'text':text}])[-300:])


def submit(kind, payload):
    with CONTROL_LOCK:
        if not ACCEPTING_JOBS:raise ValueError('Runtime is stopping. Restart NeuroForge before queuing new tasks.')
        active = [j for j in store.all_of('job') if j['status'] in ('queued','running','pausing','cancelling')]
        if len(active) >= 8:
            raise ValueError('Local queue is full (8 tasks)')
        job = store.put('job', dict(kind=kind, payload=payload, status='queued', progress=0, metrics=[], logs=[], candidates=[],
                                   environment={'torch':str(torch.__version__),'python':platform.python_version(),
                                                'platform':platform.platform(),'cuda':torch.cuda.is_available(),
                                                'engine':engine.ENGINE_VERSION}))
        POOL.submit(dispatch, job['id'])
        return job


def control(ident, action):
    with CONTROL_LOCK:
        if not ACCEPTING_JOBS:raise ValueError('Runtime is stopping. Restart NeuroForge before changing task state.')
        j = store.get('job', ident)
        if action == 'resume':
            if j['kind'] != 'train' or j['status'] not in ('paused','interrupted','failed'):
                raise ValueError('Only paused/interrupted/failed training can resume')
            store.update('job', ident, status='queued', error=None, command=None)
            POOL.submit(dispatch, ident)
        elif action in ('pause','cancel'):
            if j['status'] not in ('running','queued'):
                raise ValueError('Task is not active')
            if action == 'pause' and j['kind'] != 'train':
                raise ValueError('Search and proxy tasks support cancel only')
            store.update('job', ident, command=action, control_pending=action)
        else:
            raise ValueError('Unknown task action')
    return store.get('job', ident)


def stopped(ident):
    command = store.get('job', ident).get('command')
    if command:
        store.update('job', ident, status='paused' if command == 'pause' else 'cancelled', command=None, control_pending=None)
        log(ident, 'Paused at checkpoint.' if command == 'pause' else 'Cancelled.')
        return True
    return False


def execute(ident):
    try:
        if stopped(ident):
            return
        store.update('job', ident, status='running', started_at=time.time())
        job = store.get('job', ident)
        log(ident, f'Worker started: {job["kind"]}; all results use real PyTorch computation.')
        if job['kind']=='import_model':
            from .ecosystem import import_model
            import_model(ident,job['payload'])
        elif job['kind']=='register_operator':
            from .custom_ops import register
            register(ident,job['payload'])
        elif job['kind'] == 'train':
            train(ident, job['payload'])
        elif job['kind'] in ('debug','profile','optimize'):
            advanced(ident,job['payload'],job['kind'])
        else:
            search(ident, job['payload'], single=job['kind']=='proxy')
    except Exception as exc:
        context=runtime.error_context(exc,store.get('job',ident).get('device'))
        log(ident, traceback.format_exc()[-3500:])
        store.update('job', ident, status='failed', error=str(exc), error_context=context, finished_at=time.time())


def input_data(payload):
    x, y, manifest = data.load(store.DATA, payload['dataset_id'])
    if payload['ir']['input_shape'] != manifest['shape']:
        raise ValueError('Model input shape does not match dataset. Apply dataset to the model first.')
    if engine.analyze(payload['ir'])['output_shape'] != [len(manifest['classes'])]:
        raise ValueError('Classifier output must equal the dataset class count')
    return x,y,manifest


def search(ident, p, single=False):
    x,y,manifest = input_data(p)
    seed = p.get('seed',42)
    rng = random.Random(seed)
    train_ids = manifest['splits']['train']
    # Balanced batch, reproducible across candidates; no validation/test samples.
    batch = []
    for cls in range(len(manifest['classes'])):
        choices = [i for i in train_ids if int(y[i]) == cls]
        rng.shuffle(choices)
        batch.extend(choices[:max(1,16//len(manifest['classes']))])
    batch = batch[:32]
    xs, ys = x[batch],y[batch]
    from . import search_algorithms,pipeline,inspection
    xs=pipeline.apply(xs,p.get('data_pipeline',[]),seed,training=False)
    space=p.get('search_space')
    if space:search_space.validate(space,p['ir'])
    device=runtime.resolve(p.get('device','auto'))
    store.update('job',ident,device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    enumerator=iter(search_algorithms.enumerate_genes(space)) if space and p.get('strategy')=='enumerate' else None
    if not engine.analyze(p['ir']).get('static_macs_available',True):log(ident,'Protected model internals: static MACs limit cannot be verified. Use measured hardware latency / memory constraints. Imported weights are preserved.')
    count = 1 if single else p.get('count',12)
    candidates, seen, rejected = [], set(), 0
    attempts, total_seconds = 0, time.perf_counter()
    while len(candidates) < count and attempts < count*30:
        if stopped(ident):
            return
        attempts += 1
        if p.get('budget_seconds') and time.perf_counter()-total_seconds>p['budget_seconds']:break
        if single:
            ir = copy.deepcopy(p['ir'])
        elif space:
            parent=rng.choice(engine.rank_candidates(candidates)[:max(2,len(candidates)//2)])['genes'] if p.get('strategy')=='evolution' and len(candidates)>=4 else None
            if enumerator is not None:
                try:genes=next(enumerator)
                except StopIteration:break
            elif p.get('strategy')=='bayesian':
                genes=search_algorithms.propose(space,p['ir'],rng,candidates,seen)
                if genes is None:break
            else:genes=search_space.sample(space,p['ir'],rng,parent)
            try:ir=search_space.instantiate(p['ir'],space,genes)
            except (ValueError,KeyError,TypeError) as exc:
                rejected+=1
                if rejected<=10:log(ident,str(exc))
                continue
        elif p.get('strategy') == 'evolution' and len(candidates) >= 4:
            parent = copy.deepcopy(rng.choice(engine.rank_candidates(candidates)[:max(2,len(candidates)//2)]))
            genes = parent['genes']
            key = rng.choice(['channels','depth','residual'])
            genes[key] = rng.choice(p[key])
            ir = engine.template(**genes, classes=len(manifest['classes']), size=manifest['shape'][1])
        else:
            genes = {key:rng.choice(p[key]) for key in ['channels','depth','residual']}
            ir = engine.template(**genes,classes=len(manifest['classes']),size=manifest['shape'][1])
        h = engine.fingerprint(ir)
        if h in seen:
            continue
        seen.add(h)
        try:
            spec = engine.analyze(ir)
            if not single and (spec['params'] > p.get('max_params',20_000_000) or spec.get('static_macs_available',True) and spec['macs'] > p.get('max_macs',2_000_000_000)):
                rejected += 1
                continue
            log(ident, f'Candidate {len(candidates)+1}/{count}: {spec["params"]:,} parameters, {spec["macs"]:,} MACs')
            result = engine.proxy(ir,xs,ys,seed,device=str(device),methods=p.get('proxy_methods'))
            if p.get('hardware_aware'):
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                torch.manual_seed(seed);benchmark_model=engine.GraphModel(ir).to(device).eval()
                hw=inspection.benchmark(benchmark_model,xs.to(device),device,repeats=p.get('benchmark_repeats',10))
                del benchmark_model
                if device.type=='cuda':torch.cuda.empty_cache()
                result['hardware']=hw
                if (p.get('max_latency_ms') and hw['latency_ms']>p['max_latency_ms']) or (p.get('max_measured_memory_mb') and (hw['memory_peak_mb'] or 0)>p['max_measured_memory_mb']) or (p.get('min_throughput') and hw['samples_per_second']<p['min_throughput']):
                    rejected+=1;log(ident,'Hardware constraint rejected candidate '+h[:12]);continue
            result['sample_indices'] = batch
            candidates.append(dict(id=h[:12],ir=ir,proxy=result,genes={} if single else genes))
            engine.rank_candidates(candidates)
            store.update('job',ident,candidates=candidates,progress=len(candidates)/count,
                         rejected=rejected,attempts=attempts,**runtime.memory(device))
        except (ValueError,RuntimeError) as exc:
            if runtime.error_context(exc)['code'] in ('oom','driver_mismatch','device_error'):raise
            rejected += 1
            log(ident, f'Candidate invalid: {exc}')
    if not candidates:
        raise ValueError('No legal candidates in the selected search space and constraints')
    if stopped(ident):
        return
    log(ident, f'Completed {len(candidates)} candidates; {rejected} rejected; proxy percentile ranking, no calibrated accuracy.')
    store.update('job',ident,status='completed',progress=1,candidates=engine.rank_candidates(candidates),
                 seconds=time.perf_counter()-total_seconds,finished_at=time.time())


def atomic_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def advanced(ident,p,kind):
    from . import inspection,optimization,pipeline
    x,y,m=input_data(p);ids=m['splits']['train'][:min(p.get('inspect_batch',4),len(m['splits']['train']))]
    x=pipeline.apply(x[ids],p.get('data_pipeline',[]),p['seed'],training=False);y=y[ids]
    device=runtime.resolve(p.get('device','auto'))
    store.update('job',ident,device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    if kind=='optimize':result=optimization.run(ident,p,x,y,device)
    else:result=inspection.run(p['ir'],x,y,str(device),p['seed'],p.get('checkpoint_job'),kind=='debug',p.get('benchmark_repeats',5),ident)
    result['sample_indices']=ids
    finished_device=runtime.resolve(store.get('job',ident).get('device',str(device)))
    store.update('job',ident,status='completed',result=result,progress=1,finished_at=time.time(),**runtime.memory(finished_device))
    log(ident,f'{kind} completed on {device}; captured actual execution results.')


def train(ident,p):
    from . import pipeline
    x,y,manifest=input_data(p);seed=p.get('seed',42);torch.manual_seed(seed)
    device=runtime.resolve(p.get('device','auto'));amp=bool(p.get('amp') and device.type=='cuda')
    if p.get('amp') and not amp:log(ident,'AMP is enabled on CUDA only; using CPU FP32.')
    store.update('job',ident,device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',amp_effective=amp)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True,warn_only=True)
    model=engine.GraphModel(p['ir']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p.get('lr',.001),weight_decay=p.get('weight_decay',.0001))
    epochs,batch_size=p.get('epochs',5),p.get('batch_size',32);accum=p.get('gradient_accumulation',1)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    out=store.ARTIFACTS/ident;out.mkdir(exist_ok=True);checkpoint=out/'checkpoint.pt'
    first,metrics,best,bad_epochs=0,[],-1.,0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['rng'].cpu())
        if device.type=='cuda' and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        if saved.get('scaler'):scaler.load_state_dict(saved['scaler'])
        first,metrics,best=saved['epoch'],saved['metrics'],saved['best'];bad_epochs=saved.get('bad_epochs',0);batch_size=saved.get('batch_size',batch_size)
        log(ident,f'Restored model, optimizer, scheduler, scaler and RNG from epoch {first}.');store.update('job',ident,recovered_from_epoch=first)
    train_ids,val_ids=manifest['splits']['train'],manifest['splits']['val'];loss_fn=nn.CrossEntropyLoss();steps=p.get('data_pipeline',[])
    def autocast():return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp)
    if p.get('auto_batch'):
        original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
        while True:
            try:
                model.train();probe=x[train_ids[:batch_size]].to(device)
                with autocast():loss=loss_fn(model(probe),y[train_ids[:batch_size]].to(device))
                loss.backward();del loss,probe;optimizer.zero_grad(set_to_none=True);break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if device.type=='cuda':torch.cuda.empty_cache()
                if batch_size<=2:raise
                batch_size=max(2,batch_size//2);log(ident,f'Automatic batch preflight: OOM, reducing to {batch_size}.')
        model.load_state_dict(original);del original;torch.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    store.update('job',ident,effective_batch_size=batch_size,effective_accumulation=accum)
    log(ident,f'Device {device}; AMP={amp}; batch={batch_size}; accumulation={accum}; train/val/test={manifest["split_counts"]}.')
    def evaluate(ids):
        model.eval();loss,total,correct=0.,0,0
        with torch.no_grad():
            for offset in range(0,len(ids),batch_size):
                inds=ids[offset:offset+batch_size];xb=pipeline.apply(x[inds].to(device),steps,seed,training=False);yb=y[inds].to(device)
                with autocast():logits=model(xb);value=loss_fn(logits,yb)
                loss+=value.item()*len(inds);correct+=(logits.argmax(1)==yb).sum().item();total+=len(inds)
        return loss/total,correct/total
    for epoch in range(first,epochs):
        if stopped(ident):return
        runtime.sync(device);start=time.perf_counter();order=torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(seed+epoch)).tolist();ordered=[train_ids[i] for i in order]
        model.train();total_loss,correct,total,grad=0.,0,0,0.;lr=optimizer.param_groups[0]['lr'];step_seconds=[];optimizer.zero_grad(set_to_none=True)
        for step,offset in enumerate(range(0,len(ordered),batch_size)):
            runtime.sync(device);step_start=time.perf_counter();inds=ordered[offset:offset+batch_size]
            xb=pipeline.apply(x[inds].to(device),steps,seed+epoch*100000+step,training=True);yb=y[inds].to(device)
            group_start=(step//accum)*accum*batch_size;group_count=min(accum*batch_size,len(ordered)-group_start)
            with autocast():logits=model(xb);loss=loss_fn(logits,yb)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            scaler.scale(loss*(len(inds)/group_count)).backward()
            if (step+1)%accum==0 or offset+batch_size>=len(ordered):
                scaler.unscale_(optimizer);grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.));scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            total_loss+=loss.item()*len(inds);correct+=(logits.argmax(1)==yb).sum().item();total+=len(inds)
            runtime.sync(device);step_seconds.append(time.perf_counter()-step_start)
            store.update('job',ident,current_epoch=epoch+1,current_step=step+1,steps_per_epoch=math.ceil(len(ordered)/batch_size),step_seconds=step_seconds[-1],**runtime.memory(device))
        val_loss,val_acc=evaluate(val_ids);runtime.sync(device);seconds=time.perf_counter()-start
        metrics.append(dict(epoch=epoch+1,train_loss=total_loss/total,train_accuracy=correct/total,val_loss=val_loss,val_accuracy=val_acc,lr=lr,grad_norm=grad,seconds=seconds,samples_per_second=total/seconds,step_seconds=sum(step_seconds)/len(step_seconds),gpu_memory_mb=runtime.memory(device)['memory_peak_mb']))
        scheduler.step();improved=best<0 or val_acc>best+p.get('early_stopping_delta',0.)
        if improved:best=val_acc;bad_epochs=0;atomic_checkpoint(out/'best.pt',model.state_dict())
        else:bad_epochs+=1
        atomic_checkpoint(checkpoint,dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,metrics=metrics,best=best,bad_epochs=bad_epochs,batch_size=batch_size,ir=p['ir'],dataset_id=p['dataset_id']))
        store.update('job',ident,metrics=metrics,progress=(epoch+1)/epochs,best_val_accuracy=best,checkpoint=True,latest_checkpoint_epoch=epoch+1,**runtime.memory(device))
        log(ident,f'Epoch {epoch+1}/{epochs}: loss={total_loss/total:.4f} val_acc={val_acc:.4f}; checkpoint saved')
        if stopped(ident):return
        if p.get('early_stopping_patience',0) and bad_epochs>=p['early_stopping_patience']:
            store.update('job',ident,early_stopped=True);log(ident,'Early stopping: validation accuracy did not improve.');break
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location=device));test_loss,test_acc=evaluate(manifest['splits']['test'])
    store.update('job',ident,status='completed',progress=1,finished_at=time.time(),test_accuracy=test_acc,test_loss=test_loss,metrics=metrics,best_val_accuracy=best)
    log(ident,f'Completed; best-validation weights evaluated on held-out test: {test_acc:.4f}')
