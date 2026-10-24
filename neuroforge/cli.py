"""python -m neuroforge --help"""
import argparse,json,sys
from pathlib import Path
from .sdk import NeuroForge


def main():
    parser=argparse.ArgumentParser(prog='neuroforge',description='Local neural network design and compute CLI')
    parser.add_argument('--url',default='http://127.0.0.1:8787');commands=parser.add_subparsers(dest='command',required=True)
    for name in ['system','projects','datasets']:commands.add_parser(name)
    code=commands.add_parser('import-code');code.add_argument('source',type=Path);code.add_argument('--shape',default='3,32,32');code.add_argument('--input-type',choices=['image','tokens','sequence'],default='image');code.add_argument('--class-name');code.add_argument('--output',type=Path,required=True)
    for kind in ['train','search','proxy','debug','profile','optimize']:
        c=commands.add_parser(kind);c.add_argument('--project',required=True);c.add_argument('--device',default='auto');c.add_argument('--config',type=Path);c.add_argument('--wait',action='store_true')
    for name in ['job','wait']:c=commands.add_parser(name);c.add_argument('id')
    c=commands.add_parser('control');c.add_argument('id');c.add_argument('action',choices=['pause','resume','cancel','rerun'])
    c=commands.add_parser('export');c.add_argument('id');c.add_argument('--output',type=Path,required=True)
    commands.add_parser('model-catalog');commands.add_parser('model-assets');commands.add_parser('operators')
    c=commands.add_parser('import-model');c.add_argument('--project',required=True);c.add_argument('--provider',choices=['torchvision','timm','huggingface'],required=True);c.add_argument('--model',required=True);c.add_argument('--pretrained',action='store_true');c.add_argument('--allow-download',action='store_true');c.add_argument('--wait',action='store_true')
    c=commands.add_parser('register-operator');c.add_argument('source',type=Path);c.add_argument('--name',required=True);c.add_argument('--shape',required=True);c.add_argument('--parameters',default='{}');c.add_argument('--trust-code',action='store_true');c.add_argument('--wait',action='store_true')
    c=commands.add_parser('preview-plan');c.add_argument('plan',type=Path);c.add_argument('--project',required=True)
    a=parser.parse_args();client=NeuroForge(a.url)
    try:
        if a.command in ('system','projects','datasets'):result=getattr(client,a.command)()
        elif a.command in ('model-catalog','model-assets','operators'):result=getattr(client,a.command.replace('-','_'))()
        elif a.command=='import-model':
            result=client.import_model(a.project,a.provider,a.model,a.pretrained,a.allow_download)
            if a.wait:result=client.wait(result['id'])
        elif a.command=='register-operator':
            result=client.register_operator(a.name,a.source.read_text(encoding='utf-8-sig'),[int(v) for v in a.shape.split(',')],json.loads(a.parameters),trusted_code=a.trust_code)
            if a.wait:result=client.wait(result['id'])
        elif a.command=='preview-plan':result=client.preview_plan(a.project,json.loads(a.plan.read_text(encoding='utf-8-sig')))
        elif a.command=='import-code':
            result=client.import_code(a.source.read_text(encoding='utf-8-sig'),[int(v) for v in a.shape.split(',')],input_type=a.input_type,class_name=a.class_name);a.output.write_text(json.dumps(result['ir'],indent=2),encoding='utf-8')
        elif a.command in ('job','wait'):result=getattr(client,a.command)(a.id)
        elif a.command=='control':result=client.control(a.id,a.action)
        elif a.command=='export':result={'path':str(client.export(a.id,a.output))}
        else:
            options=json.loads(a.config.read_text(encoding='utf-8')) if a.config else {};options['device']=a.device
            result=client.submit(a.command,a.project,**options)
            if a.wait:result=client.wait(result['id'])
        print(json.dumps(result,indent=2,ensure_ascii=False))
        if isinstance(result,dict) and result.get('status')=='failed':return 1
        return 0
    except Exception as e:print(str(e),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
