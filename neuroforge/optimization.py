"""Explicit optimization jobs: magnitude pruning, dynamic INT8 and ONNX."""
import copy,json,io,zipfile,time
from pathlib import Path
import torch
from torch import nn
from . import engine,store,runtime,inspection


def run(ident,p,x,y,device):
    torch.manual_seed(p['seed']);model=engine.GraphModel(p['ir']).eval();source='initialized'
    if p.get('checkpoint_job'):
        j=store.get('job',p['checkpoint_job'])
        if engine.fingerprint(j['payload']['ir'])!=engine.fingerprint(p['ir']):raise ValueError('Checkpoint architecture mismatch')
        model.load_state_dict(torch.load(store.ARTIFACTS/j['id']/'best.pt',map_location='cpu',weights_only=True));source=j['id']
    operation=p.get('optimization','prune');out=store.ARTIFACTS/ident;out.mkdir(exist_ok=True)
    if operation=='dynamic_int8':device=torch.device('cpu');store.update('job',ident,device='cpu',device_name='CPU')
    model=model.to(device);x=x.to(device);y=y.to(device)
    before=inspection.benchmark(model,x,device);before_buffer=io.BytesIO();torch.save(model.state_dict(),before_buffer)
    with torch.inference_mode():original=model(x);before_loss=nn.functional.cross_entropy(original,y).item()
    report=dict(operation=operation,source=source,before=before,before_bytes=len(before_buffer.getvalue()),before_loss=before_loss)
    (out/'model.py').write_text(engine.export_code(p['ir']),encoding='utf-8')
    if operation=='prune':
        from torch.nn.utils import prune
        targets=[(m,'weight') for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
        if not targets:raise ValueError('No prunable Conv2d / Linear layers')
        prune.global_unstructured(targets,pruning_method=prune.L1Unstructured,amount=p.get('prune_amount',.3))
        for m,key in targets:prune.remove(m,key)
        report['weight_sparsity']=sum((m.weight==0).sum().item() for m,k in targets)/sum(m.weight.numel() for m,k in targets)
        report['note']='Unstructured zero weights; dense kernels do not guarantee speed or file-size reduction.'
    elif operation=='dynamic_int8':
        model=torch.ao.quantization.quantize_dynamic(model,{nn.Linear},dtype=torch.qint8,inplace=False)
        report['note']='Dynamic INT8 Linear layers on CPU; Conv2d remains floating point.'
    elif operation=='onnx':
        import onnx,onnxruntime as ort,numpy as np
        path=out/'model.onnx'
        torch.onnx.export(model,(x,),str(path),opset_version=17,input_names=['input'],output_names=['output'],dynamic_axes={'input':{0:'batch'},'output':{0:'batch'}},dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)));session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
        values={'input':x.detach().cpu().numpy()};pred=session.run(None,values)[0]
        report.update(onnx_max_abs_error=float(np.max(np.abs(pred-original.detach().cpu().numpy()))),onnx_output_shape=list(pred.shape),onnx_bytes=path.stat().st_size,onnx_provider='CPUExecutionProvider')
        for _ in range(3):session.run(None,values)
        now=time.perf_counter()
        for _ in range(10):session.run(None,values)
        report['onnx_latency_ms']=(time.perf_counter()-now)*100
    else:raise ValueError('Unknown optimization operation')
    torch.save(model.state_dict(),out/'optimized.pt');report['after_bytes']=(out/'optimized.pt').stat().st_size
    report['after']=inspection.benchmark(model,x,device)
    with torch.inference_mode():pred=model(x);report.update(after_loss=nn.functional.cross_entropy(pred,y).item(),sample_accuracy=(pred.argmax(1)==y).float().mean().item(),output_max_abs_change=(pred-original).abs().max().item())
    report['accuracy_scope']='Selected training samples only; run full evaluation before using optimized weights.'
    (out/'optimization.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    loader='''import torch\nfrom model import build_model\nmodel=build_model().eval()\n'''+("model=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)\n" if operation=='dynamic_int8' else '')+"model.load_state_dict(torch.load('optimized.pt',map_location='cpu',weights_only=True))\nprint(model)\n"
    (out/'load_optimized.py').write_text(loader,encoding='utf-8')
    with zipfile.ZipFile(out/'optimization.zip','w',zipfile.ZIP_DEFLATED) as z:
        from .asset_ops import asset_export_files
        for name,value in asset_export_files(p['ir']).items():
            if isinstance(value,Path):z.write(value,name)
            else:z.writestr(name,value)
        z.writestr('requirements.txt','torch=='+str(torch.__version__)+'\n')
        for name in ('model.py','optimized.pt','model.onnx','load_optimized.py','optimization.json'):
            if (out/name).exists():z.write(out/name,name)
    return report
