"""Checked serializable vector scene edits, independent of the model IR."""
import re,math
from .diagrams import Scene,two_d,three_d


def create(ir,options):
    scene=three_d(ir,options) if options.get('mode')=='3d' else two_d(ir,options)
    return dict(schema_version=1,width=scene.w,height=scene.h,background=scene.bg,objects=[list(op) for op in scene.ops])


def parse(doc):
    if doc.get('schema_version')!=1:raise ValueError('Unsupported figure schema')
    def number(v,lo=-30000,hi=30000):
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Invalid figure coordinate or size')
        return v
    def color(c):
        if c is not None and not re.fullmatch(r'#[0-9a-fA-F]{6}',c):raise ValueError('Colors must be six-digit hexadecimal RGB')
        return c
    scene=Scene(number(doc['width'],100,30000),number(doc['height'],100,30000),color(doc.get('background','#ffffff')))
    objects=doc.get('objects',[])
    if not 1<=len(objects)<=12000:raise ValueError('Figure requires 1..12000 primitives')
    if doc.get('background') is None:raise ValueError('Figure background must be RGB')
    groups=doc.get('groups',[])
    if not isinstance(groups,list) or len(groups)>256 or any(not isinstance(g,list) or len(g)>len(objects) or any(type(i)!=int or not 0<=i<len(objects) for i in g) for g in groups):raise ValueError('Invalid figure groups')
    for item in objects:
        op=list(item);kind=op[0]
        if kind=='rect' and len(op)==9:
            for i in (1,2):number(op[i])
            for i in (3,4):number(op[i],0,30000)
            color(op[5]);color(op[6]);number(op[7],0,100);number(op[8],.1,20)
        elif kind in ('poly','line') and len(op)==(5 if kind=='poly' else 4):
            if not 2<=len(op[1])<=1000:raise ValueError('Invalid point count')
            for x,y in op[1]:number(x);number(y)
            color(op[2]);number(op[-1],.1,20)
            if kind=='poly':color(op[3])
        elif kind=='text' and len(op)==8:
            number(op[1]);number(op[2]);number(op[4],5,120);color(op[5]);number(op[6],100,900)
            if not isinstance(op[3],str) or len(op[3])>1000 or op[7] not in ('start','middle','end'):raise ValueError('Invalid text primitive')
        else:raise ValueError('Unsupported scene object')
        scene.ops.append(tuple(op))
    return scene
