"""Standalone script copied into training experiment exports.

python reproduce.py --dataset /path/to/runtime/datasets/<dataset-id>/images.npz --output rerun
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from torch import nn
from model import build_model
from pipeline import apply as apply_pipeline


def main():
    parser=argparse.ArgumentParser(description='Reproduce an exported NeuroForge training run on CPU or CUDA')
    parser.add_argument('--dataset',type=Path,required=True,help='Original versioned images.npz')
    parser.add_argument('--device',default='cpu',help='cpu, auto, or cuda:N')
    parser.add_argument('--output',type=Path,default=Path('rerun'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    experiment=json.loads((root/'experiment.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
    p=experiment['payload']
    p['batch_size']=experiment.get('effective_batch_size',p['batch_size'])
    if experiment['kind']!='train':
        raise ValueError('This reproduction entry point is for training exports')
    with np.load(args.dataset,allow_pickle=False) as a:
        is_tokens='tokens' in a
        pixels,labels=a['tokens' if is_tokens else 'images'].copy(),a['labels'].copy()
    split=manifest['pipeline'][1]
    identity=[manifest['classes'],manifest['seed'],manifest['vocab_size']] if is_tokens else [manifest['classes'],manifest['seed'],split['val'],split['test']]
    digest=hashlib.sha256(pixels.tobytes()+labels.astype(np.int64).tobytes()+json.dumps(identity,ensure_ascii=is_tokens).encode()).hexdigest()
    if digest!=manifest['id']:
        raise ValueError('Dataset content does not match exported version hash')
    device=torch.device('cuda:0' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    x=(torch.from_numpy(pixels).long() if is_tokens else torch.from_numpy(pixels).permute(0,3,1,2).float()/255.).to(device)
    y=torch.from_numpy(labels).to(device)
    torch.set_num_threads(4)
    torch.manual_seed(p['seed'])
    torch.use_deterministic_algorithms(True,warn_only=True)
    model=build_model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,p['epochs'])
    loss_fn=nn.CrossEntropyLoss()
    args.output.mkdir(parents=True,exist_ok=True)
    metrics,best,bad_epochs=[], -1.,0
    amp=bool(p.get('amp') and device.type=='cuda');accum=p.get('gradient_accumulation',1)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    def autocast():return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp)
    def evaluate(ids):
        model.eval()
        loss,correct=0.,0
        with torch.no_grad():
            for offset in range(0,len(ids),p['batch_size']):
                inds=ids[offset:offset+p['batch_size']]
                xb=apply_pipeline(x[inds],p.get('data_pipeline',[]),p['seed'],training=False)
                with autocast():logits=model(xb);value=loss_fn(logits,y[inds])
                loss+=value.item()*len(inds)
                correct+=(logits.argmax(1)==y[inds]).sum().item()
        return loss/len(ids),correct/len(ids)
    for epoch in range(p['epochs']):
        train_ids=manifest['splits']['train']
        order=torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(p['seed']+epoch)).tolist()
        ids=[train_ids[i] for i in order]
        model.train()
        total_loss=0.
        opt.zero_grad(set_to_none=True)
        for step,offset in enumerate(range(0,len(ids),p['batch_size'])):
            inds=ids[offset:offset+p['batch_size']]
            xb=apply_pipeline(x[inds],p.get('data_pipeline',[]),p['seed']+epoch*100000+step,training=True)
            with autocast():logits=model(xb);loss=loss_fn(logits,y[inds])
            group_start=(step//accum)*accum*p['batch_size'];group_count=min(accum*p['batch_size'],len(ids)-group_start)
            scaler.scale(loss*(len(inds)/group_count)).backward()
            if (step+1)%accum==0 or offset+p['batch_size']>=len(ids):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            total_loss+=loss.item()*len(inds)
        vl,va=evaluate(manifest['splits']['val'])
        schedule.step()
        if best<0 or va>best+p.get('early_stopping_delta',0):
            best=va;bad_epochs=0
            torch.save(model.state_dict(),args.output/'best.pt')
        else:bad_epochs+=1
        row={'epoch':epoch+1,'train_loss':total_loss/len(ids),'val_loss':vl,'val_accuracy':va}
        metrics.append(row)
        print(json.dumps(row),flush=True)
        if p.get('early_stopping_patience',0) and bad_epochs>=p['early_stopping_patience']:break
    model.load_state_dict(torch.load(args.output/'best.pt',weights_only=True,map_location=device))
    tl,ta=evaluate(manifest['splits']['test'])
    result={'metrics':metrics,'best_val_accuracy':best,'test_loss':tl,'test_accuracy':ta,
            'torch':str(torch.__version__),'python':platform.python_version()}
    (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Held-out test accuracy:',ta)


if __name__=='__main__':
    main()
