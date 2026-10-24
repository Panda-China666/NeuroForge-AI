"""Framework-independent architecture IR, deterministic compiler, and zero-cost signals."""
import copy
import hashlib
import json
import math
import time
import torch
from torch import nn
import torch.nn.functional as F

from .modern import *

ENGINE_VERSION = '0.4.0'
OPS = ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'AvgPool2d', 'ResidualBlock',
       'AdaptiveAvgPool2d', 'Flatten', 'Linear', 'Dropout', 'Add', 'Concat'] + MODERN_OPS


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def template(channels=16, depth=2, residual=True, classes=3, size=32):
    nodes = []
    prev = 'input'
    for i in range(depth):
        ident = f'block{i+1}'
        nodes.append(dict(id=ident, op='ResidualBlock' if residual else 'Conv2d', inputs=[prev],
                          config=dict(out_channels=channels * min(2**i, 4), kernel_size=3, stride=1, padding=1)))
        if not residual:
            nodes.append(dict(id=f'act{i+1}', op='ReLU', inputs=[ident], config={}))
            ident = f'act{i+1}'
        nodes.append(dict(id=f'pool{i+1}', op='MaxPool2d', inputs=[ident], config={'kernel_size': 2}))
        prev = f'pool{i+1}'
    nodes += [dict(id='gap', op='AdaptiveAvgPool2d', inputs=[prev], config={}),
              dict(id='flatten', op='Flatten', inputs=['gap'], config={}),
              dict(id='classifier', op='Linear', inputs=['flatten'], config={'out_features': classes})]
    return dict(schema_version=1, name='ForgeNet', input_shape=[3, size, size], nodes=nodes, output='classifier')


def integer(c, key, default, lo=1, hi=1024):
    v = c.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f'{key} must be an integer in [{lo}, {hi}]')
    return v


def analyze(ir, _level=0):
    if ir.get('schema_version') != 1:
        raise ValueError('Unsupported IR schema_version')
    shape = ir.get('input_shape', [])
    if not 1 <= len(shape) <= 3 or any(type(x) is not int or not 1 <= x <= 65536 for x in shape):
        raise ValueError('Invalid input shape')
    if len(shape) == 3 and (max(shape[1:]) > 512 or (_level == 0 and not ir.get('input_ports') and shape[0] not in (1,3))):
        raise ValueError('Invalid image input shape')
    nodes = ir.get('nodes', [])
    if not 1 <= len(nodes) <= 256:
        raise ValueError('Architecture requires 1–256 nodes')
    ports=ir.get('input_ports',[{'id':'input','shape':shape}])
    if len(ports)>16 or not ports or any(not isinstance(p.get('id'),str) or not p['id'].isidentifier() or not p.get('shape') for p in ports):
        raise ValueError('Invalid input ports')
    specs, shapes, done = [], {p['id']:p['shape'] for p in ports}, {p['id'] for p in ports}
    if any(not isinstance(n,dict) for n in nodes):
        raise ValueError('Every node must be an object')
    if any(not isinstance(n.get('inputs'),list) or not all(isinstance(x,str) for x in n['inputs']) or not isinstance(n.get('config',{}),dict) for n in nodes):
        raise ValueError('Node inputs must be a list of IDs and config must be an object')
    ids = [n.get('id') for n in nodes]
    if any(not isinstance(i, str) or not i.isidentifier() or len(i) > 48 or i in shapes for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Node IDs must be unique Python identifiers (input is reserved)')
    if any(hasattr(nn.ModuleDict(),i) for i in ids):
        raise ValueError('Node ID conflicts with a reserved module attribute')
    pending = list(nodes)
    params = macs = activations = expanded = 0
    while pending:
        ready = [n for n in pending if n.get('inputs') and all(i in done for i in n['inputs'])]
        if not ready:
            raise ValueError('Graph has a cycle, missing input, or disconnected node')
        for n in ready:
            ident, op, c = n['id'], n.get('op'), n.get('config', {})
            if op not in OPS:
                raise ValueError(f'{ident}: unsupported operator {op}')
            inputs = [shapes[i] for i in n['inputs']]
            s = inputs[0]
            p = m = 0
            extra = {}
            if op not in ('Add', 'Concat', 'Module') and len(inputs) != 1:
                raise ValueError(f'{ident}: operator requires exactly one input')
            if op in ('Conv2d', 'ResidualBlock', 'MaxPool2d', 'AvgPool2d', 'BatchNorm2d', 'AdaptiveAvgPool2d') and len(s) != 3:
                raise ValueError(f'{ident}: expected C,H,W tensor, received {s}')
            if op in MODERN_OPS:
                t,p,m,extra=modern_shape(n,inputs,analyze,_level)
            elif op in ('Conv2d', 'ResidualBlock'):
                out = integer(c, 'out_channels', 16, hi=512)
                k, stride, pad = integer(c, 'kernel_size', 3, hi=7), integer(c, 'stride', 1, hi=4), integer(c, 'padding', 1, lo=0, hi=6)
                h, w = [(x + 2*pad-k)//stride+1 for x in s[1:]]
                if op == 'ResidualBlock' and (k != 3 or pad != 1):
                    raise ValueError(f'{ident}: ResidualBlock uses kernel_size=3, padding=1')
                p, m = out*(s[0]*k*k+int(c.get('bias',True))), h*w*out*s[0]*k*k
                if op == 'ResidualBlock':
                    p += out*(out*9+1)
                    m += h*w*out*out*9
                    if s[0] != out or stride != 1:
                        p += out*(s[0]+1)
                        m += h*w*out*s[0]
                t = [out, h, w]
            elif op in ('MaxPool2d', 'AvgPool2d'):
                k = integer(c, 'kernel_size', 2, hi=8)
                st = integer(c, 'stride', k, hi=8)
                t = [s[0], (s[1]-k)//st+1, (s[2]-k)//st+1]
            elif op == 'AdaptiveAvgPool2d':
                t = [s[0], 1, 1]
            elif op == 'Flatten':
                t = [math.prod(s)]
            elif op == 'Linear':
                if len(s) not in (1,2):
                    raise ValueError(f'{ident}: add Flatten before Linear')
                out = integer(c, 'out_features', 3, hi=4096)
                p, m, t = (s[-1]+int(c.get('bias',True)))*out, math.prod(s[:-1])*s[-1]*out, s[:-1]+[out]
            elif op == 'BatchNorm2d':
                p, t = 2*s[0], list(s)
            elif op == 'Add':
                if len(inputs) < 2 or any(x != s for x in inputs):
                    raise ValueError(f'{ident}: Add requires >=2 identical shapes, received {inputs}')
                t = list(s)
            elif op == 'Concat':
                if len(inputs) < 2 or any(x[1:] != s[1:] for x in inputs):
                    raise ValueError(f'{ident}: Concat requires matching non-channel dimensions')
                t = [sum(x[0] for x in inputs)] + s[1:]
            else:
                t = list(s)
                if op == 'Dropout' and not 0 <= float(c.get('p', .2)) < 1:
                    raise ValueError('Dropout p must be in [0,1)')
            if min(t) < 1:
                raise ValueError(f'{ident}: tensor has nonpositive dimension {t}')
            params += p
            macs += m
            activations += math.prod(t)+extra.get('nested_activation',0)
            expanded += extra.get('internal_nodes',1)
            if params > 250_000_000 or activations > 32_000_000 or macs > 10_000_000_000 or expanded > 2048:
                raise ValueError('Model exceeds local resource budget')
            shapes[ident] = t
            specs.append(dict(id=ident, op=op, inputs=n['inputs'], config=c, input_shapes=inputs,
                              shape=t, params=p, macs=m, **extra))
            done.add(ident)
            pending.remove(n)
    if ir.get('output') not in shapes or ir.get('output') == 'input':
        raise ValueError('Output must reference a model node')
    ancestors = set()
    mapping = {n['id']: n for n in nodes}
    def visit(i):
        if i not in mapping or i in ancestors:
            return
        ancestors.add(i)
        for parent in mapping[i]['inputs']:
            visit(parent)
    visit(ir['output'])
    if ancestors != set(ids):
        raise ValueError('Every node must contribute to the output')
    return dict(static_macs_available=all(n.get('static_macs_available',True) for n in specs),nodes=specs, params=params, macs=macs, flops=macs*2, expanded_nodes=expanded,
                parameter_memory_mb=params*4/2**20, activation_memory_mb=activations*4/2**20,
                output_shape=shapes[ir['output']], hash=fingerprint(ir),
                notes=['FP32 parameter bytes; summed forward activations, not peak training VRAM.',
                       'MACs count Conv/Linear only; FLOPs = 2 × MACs. No accuracy prediction is calibrated.'])


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))) + self.skip(x))


class GraphModel(nn.Module):
    def __init__(self, ir):
        super().__init__()
        self.ir = copy.deepcopy(ir)
        self.spec = analyze(ir, _level=1 if ir.get("input_ports") else 0)
        self.layers = nn.ModuleDict()
        for n in self.spec['nodes']:
            op, c, s = n['op'], n['config'], n['input_shapes'][0]
            if op in MODERN_OPS:
                layer = modern_layer(n,s,GraphModel,n)
            elif op == 'Conv2d':
                layer = nn.Conv2d(s[0], c.get('out_channels', 16), c.get('kernel_size', 3), c.get('stride', 1), c.get('padding', 1),bias=c.get('bias',True))
            elif op == 'ResidualBlock':
                layer = ResidualBlock(s[0], c.get('out_channels', 16), c.get('stride', 1))
            elif op == 'Linear':
                layer = nn.Linear(s[-1], c.get('out_features', 3),bias=c.get('bias',True))
            elif op in ('MaxPool2d', 'AvgPool2d'):
                layer = getattr(nn, op)(c.get('kernel_size', 2), c.get('stride', c.get('kernel_size', 2)))
            elif op == 'AdaptiveAvgPool2d':
                layer = nn.AdaptiveAvgPool2d(1)
            elif op == 'BatchNorm2d':
                layer = nn.BatchNorm2d(s[0])
            elif op == 'ReLU':
                layer = nn.ReLU()
            elif op == 'Flatten':
                layer = nn.Flatten(1)
            elif op == 'Dropout':
                layer = nn.Dropout(float(c.get('p', .2)))
            else:
                layer = nn.Identity()
            self.layers[n['id']] = layer

    def forward(self, *args):
        ports=self.ir.get('input_ports',[{'id':'input'}])
        values = {p['id']:x for p,x in zip(ports,args)}
        observer=getattr(self,'observer',None)
        if observer:observer.input(args[0])
        for n in self.spec['nodes']:
            if observer:observer.before(n)
            xs = [values[i] for i in n['inputs']]
            if n['op'] == 'Add':
                v = torch.stack(xs).sum(0)
            elif n['op'] == 'Concat':
                v = torch.cat(xs, 1)
            elif n['op']=='Module':
                v=self.layers[n['id']](*xs)
            else:
                v = self.layers[n['id']](xs[0])
            values[n['id']] = v
            if observer:observer.after(n,v)
        return values[self.ir['output']]


def proxy(ir, x, y, seed=42, device='cpu', methods=None):
    from .runtime import proxy as execute_proxy
    return execute_proxy(ir,x,y,seed,device,methods)


def rank_candidates(candidates):
    """Equal weight empirical percentile rank of available, nonconstant proxy signals."""
    names = sorted({k for c in candidates for k in c['proxy']['scores']})
    for c in candidates:
        c['proxy_ranks'] = {}
    for name in names:
        valid = sorted([c for c in candidates if c['proxy']['scores'].get(name) is not None],
                       key=lambda c: c['proxy']['scores'][name])
        vals = [c['proxy']['scores'][name] for c in valid]
        if len(set(vals)) <= 1:
            continue
        for c in valid:
            v = c['proxy']['scores'][name]
            tied = [i for i, val in enumerate(vals) if val == v]
            c['proxy_ranks'][name] = (sum(tied)/len(tied))/(len(valid)-1)
    for c in candidates:
        r = list(c['proxy_ranks'].values())
        c['score'] = sum(r)/len(r) if r else 0.5
        c['disagreement'] = max(r)-min(r) if r else None
    candidates.sort(key=lambda c: c['score'], reverse=True)
    for i, c in enumerate(candidates):
        c['rank'] = i+1
        def dominates(d):
            values=[(d['score'],c['score'],True),(d['proxy']['params'],c['proxy']['params'],False)]
            a,b=d['proxy'].get('hardware'),c['proxy'].get('hardware')
            if a and b:
                values.append((a['latency_ms'],b['latency_ms'],False))
                if a.get('memory_peak_mb') is not None and b.get('memory_peak_mb') is not None:values.append((a['memory_peak_mb'],b['memory_peak_mb'],False))
            return all(x>=y if maximize else x<=y for x,y,maximize in values) and any(x>y if maximize else x<y for x,y,maximize in values)
        c['pareto'] = not any(dominates(d) for d in candidates if d is not c)
    return candidates


def export_code(ir):
    # Bundle the actual compiler implementation: identical node names/state_dict keys.
    from pathlib import Path
    source = Path(__file__).read_text(encoding='utf-8')
    source = source[:source.index('\ndef proxy(')]
    modern_source=Path(__file__).with_name('modern.py').read_text(encoding='utf-8')
    source=source.replace('from .modern import *',modern_source)
    return source + '\n\nARCHITECTURE = json.loads(' + repr(json.dumps(ir)) + ')\n\ndef build_model():\n    return GraphModel(ARCHITECTURE)\n\nif __name__ == "__main__":\n    model = build_model()\n    print(model)\n    print(model(torch.zeros(1, *ARCHITECTURE["input_shape"], dtype=torch.long if ARCHITECTURE.get("input_type")=="tokens" else torch.float32)).shape)\n'
