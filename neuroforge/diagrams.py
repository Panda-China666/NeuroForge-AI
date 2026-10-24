"""Architecture-driven vector and high-resolution diagram renderer.

Shared scene primitives render SVG, antialiased PNG and vector PDF. The spatial
view uses projected extruded cards; no image generator or browser screenshot.
"""
import html
import io
import math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from . import engine

PALETTES={
 'clean':['#e9f0fb','#ede9fa','#e8f3ee','#fff1df','#fce8eb'],
 'minimal':['#f6f7f8']*5,
 'color':['#dceafe','#ede1ff','#dff4e6','#ffedcc','#ffe2e8'],
 'mono':['#f2f2f2']*5,
 'poster':['#dce9f7','#e4dcf2','#dcf0e8','#f8e9d2','#efd9e2'],
 'product':['#e7eefa','#ebe6fa','#e3f3eb','#f8ecd9','#f6e3e8'],
 'white':['#eef2f5','#f1eef7','#edf5f0','#f6f1e8','#f7eef0'],
 'dark':['#23364e','#3c3155','#1e413e','#49402d','#4e2d41'],
 'cover':['#d7e4f8','#e2daf5','#d7ede4','#f5e4d0','#f3dbe4'],
 'investor':['#e6ebf5','#e8e6f2','#e2eee8','#f2eade','#f1e4e8']}


def category(op):
    if op in ('Input','Embedding','PatchEmbedding','ClassToken','PositionalEmbedding','PositionalEncoding'):return 0
    if op in ('Module','TransformerEncoder','TransformerStack','MultiHeadAttention'):return 1
    if op in ('Conv2d','ConvBlock','ResidualBlock'):return 2
    if op in ('Linear','FeedForward','TokenPooling','Flatten'):return 3
    return 4


def short(v):
    return f'{v/1e6:.2f}M' if v>=1e6 else f'{v/1000:.1f}K' if v>=1000 else str(v)


def wrap_label(value,size,width,max_lines=2):
    fontpath=next((p for p in ['C:/Windows/Fonts/msyh.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()),None)
    font=ImageFont.truetype(fontpath,round(size)) if fontpath else ImageFont.load_default()
    lines=[];line=''
    for char in value:
        if font.getlength(line+char)>width and line:
            lines.append(line);line=char
        else:line+=char
    if line:lines.append(line)
    if len(lines)>max_lines:
        lines=lines[:max_lines];last=lines[-1]
        while last and font.getlength(last+'…')>width:last=last[:-1]
        lines[-1]=last+'…'
    return lines or ['']


def label(n):
    return n.get('name') or {'Conv2d':'Convolution','ResidualBlock':'Residual block','TransformerStack':'Transformer encoder',
          'PatchEmbedding':'Patch embedding','PositionalEmbedding':'Position embedding','TokenPooling':'Token pooling',
          'AdaptiveAvgPool2d':'Global average pool','MultiHeadAttention':'Self-attention','Linear':'Linear projection',
          'ClassToken':'Class token','LayerNorm':'Layer normalization','PositionalEncoding':'Position encoding'}.get(n['op'],n['op'])


class Scene:
    def __init__(self,w,h,bg='#ffffff'):self.w=w;self.h=h;self.bg=bg;self.ops=[]
    def rect(self,x,y,w,h,fill,stroke=None,r=8,lw=1):self.ops.append(('rect',x,y,w,h,fill,stroke,r,lw))
    def poly(self,points,fill,stroke=None,lw=1):self.ops.append(('poly',points,fill,stroke,lw))
    def line(self,points,color='#8993a2',width=1.5):self.ops.append(('line',points,color,width))
    def text(self,x,y,value,size=12,color='#263244',weight=400,anchor='start'):self.ops.append(('text',x,y,str(value),size,color,weight,anchor))
    def svg(self):
        out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}"><rect width="100%" height="100%" fill="{self.bg}"/>']
        for op in self.ops:
            if op[0]=='rect':
                _,x,y,w,h,fill,stroke,r,lw=op
                out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')
            elif op[0]=='poly':
                _,pts,fill,stroke,lw=op;points=' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)
                out.append(f'<polygon points="{points}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}" stroke-linejoin="round"/>')
            elif op[0]=='line':
                _,pts,color,lw=op;points=' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)
                out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round" stroke-linecap="round"/>')
            else:
                _,x,y,value,size,color,weight,anchor=op
                out.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Segoe UI,Arial,Microsoft YaHei,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(value)}</text>')
        return (''.join(out)+'</svg>').encode()
    def png(self,width=2400):
        scale=min(4,width/self.w);w,h=round(self.w*scale),round(self.h*scale)
        if w*h>32_000_000:raise ValueError('Export exceeds 32 megapixels; reduce resolution')
        im=Image.new('RGB',(w,h),self.bg);d=ImageDraw.Draw(im)
        fonts={}
        for op in self.ops:
            if op[0]=='rect':
                _,x,y,bw,bh,fill,stroke,r,lw=op
                d.rounded_rectangle((round(x*scale),round(y*scale),round((x+bw)*scale),round((y+bh)*scale)),radius=round(r*scale),fill=fill,outline=stroke,width=max(1,round(lw*scale)))
            elif op[0]=='poly':
                _,pts,fill,stroke,lw=op;pts=[(round(x*scale),round(y*scale)) for x,y in pts]
                d.polygon(pts,fill=fill)
                if stroke:d.line(pts+[pts[0]],fill=stroke,width=max(1,round(lw*scale)),joint='curve')
            elif op[0]=='line':
                _,pts,color,lw=op;d.line([(round(x*scale),round(y*scale)) for x,y in pts],fill=color,width=max(1,round(lw*scale)),joint='curve')
            else:
                _,x,y,value,size,color,weight,anchor=op;key=(size,weight)
                if key not in fonts:
                    candidates=['C:/Windows/Fonts/msyhbd.ttc' if weight>=600 else 'C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/segoeuib.ttf' if weight>=600 else 'C:/Windows/Fonts/segoeui.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
                    fontpath=next((p for p in candidates if Path(p).exists()),None)
                    fonts[key]=ImageFont.truetype(fontpath,round(size*scale)) if fontpath else ImageFont.load_default()
                f=fonts[key];tw=d.textlength(value,font=f);offset=tw/2 if anchor=='middle' else tw if anchor=='end' else 0
                d.text((x*scale-offset,y*scale),value,font=f,fill=color,anchor='ls')
        out=io.BytesIO();im.save(out,format='PNG',dpi=(300,300));return out.getvalue()
    def pdf(self,paper='auto'):
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        font='Helvetica'
        try:
            if 'NeuroForge' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('NeuroForge','C:/Windows/Fonts/msyh.ttc'))
            font='NeuroForge'
        except Exception:
            if 'STSong-Light' not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
            font='STSong-Light'
        pages={'a4':(842,595),'letter':(792,612),'poster':(1684,1190)}
        pw,ph=pages.get(paper,(self.w,self.h));sc=min((pw-36)/self.w,(ph-36)/self.h)
        out=io.BytesIO();c=canvas.Canvas(out,pagesize=(pw,ph));c.setTitle('NeuroForge architecture diagram')
        c.translate((pw-self.w*sc)/2,(ph-self.h*sc)/2);c.scale(sc,sc)
        def fill(color):c.setFillColor(HexColor(color or '#ffffff'))
        def stroke(color):c.setStrokeColor(HexColor(color or '#ffffff'))
        fill(self.bg);c.rect(0,0,self.w,self.h,stroke=0,fill=1)
        for op in self.ops:
            if op[0]=='rect':
                _,x,y,w,h,fc,st,r,lw=op;fill(fc);stroke(st);c.setLineWidth(lw);c.roundRect(x,self.h-y-h,w,h,r,stroke=int(bool(st)),fill=int(bool(fc)))
            elif op[0] in ('poly','line'):
                if op[0]=='poly':_,pts,fc,st,lw=op
                else:_,pts,st,lw=op;fc=None
                p=c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
                for x,y in pts[1:]:p.lineTo(x,self.h-y)
                if op[0]=='poly':p.close()
                fill(fc);stroke(st);c.setLineWidth(lw);c.drawPath(p,stroke=int(bool(st)),fill=int(bool(fc)))
            else:
                _,x,y,value,size,color,weight,anchor=op;c.setFont(font,size);fill(color)
                (c.drawCentredString if anchor=='middle' else c.drawRightString if anchor=='end' else c.drawString)(x,self.h-y,value)
        c.showPage();c.save();return out.getvalue()


def graph(ir):
    spec=engine.analyze(ir);mapping={n['id']:n for n in ir['nodes']}
    nodes=[dict(id='input',op='Input',shape=ir['input_shape'],params=0,inputs=[],name='Token input' if ir.get('input_type')=='tokens' else 'Image input')]
    for n in spec['nodes']:
        item=dict(n);item['name']=label(mapping[n['id']]);item['repeat']=mapping[n['id']].get('config',{}).get('depth',1) if n['op']=='TransformerStack' else 1
        item['annotation']=mapping[n['id']].get('annotation','');nodes.append(item)
        item['stage']=mapping[n['id']].get('stage','')
    return nodes,spec


def arrow(scene,points,color,width):
    scene.line(points,color,width);x,y=points[-1];px,py=points[-2];angle=math.atan2(y-py,x-px)
    pts=[(x,y),(x-7*math.cos(angle-.4),y-7*math.sin(angle-.4)),(x-7*math.cos(angle+.4),y-7*math.sin(angle+.4))]
    scene.poly(pts,color)


def two_d(ir,o):
    nodes,spec=graph(ir);vertical=o.get('orientation')=='vertical';style=o.get('style','clean');palette=PALETTES.get(style,PALETTES['clean'])
    bw,bh=190,105;font=o.get('font_size',12);gap=58;levels={'input':0};rows={};positions={}
    for n in nodes:
        level=max([levels[i]+1 for i in n['inputs']] or [0]);levels[n['id']]=level
        row=rows.get(level,0);rows[level]=row+1
        positions[n['id']]=(70+row*(bw+38),155+level*(bh+gap)) if vertical else (60+level*(bw+gap),155+row*(bh+50))
    width=max(x+bw for x,y in positions.values())+65;height=max(y+bh for x,y in positions.values())+100
    sc=Scene(width,height);sc.text(60,52,o.get('title') or ir.get('name','Neural architecture'),27,'#202735',600)
    sc.text(60,80,o.get('subtitle') or f'{len(nodes)-1} blocks  /  {short(spec["params"])} parameters  /  {short(spec["macs"])} MACs',11,'#7d8592')
    groups={}
    for n in nodes:
        if n.get('stage'):groups.setdefault(n['stage'],[]).append(positions[n['id']])
    for name,pts in groups.items():
        x=min(p[0] for p in pts)-18;y=min(p[1] for p in pts)-30
        gw=max(p[0] for p in pts)+bw+18-x;gh=max(p[1] for p in pts)+bh+16-y
        sc.rect(x,y,gw,gh,'#fafbfc','#e4e8ed',r=12,lw=.8);sc.text(x+14,y+18,name[:48],10,'#939dad',500)
    for n in nodes:
        bx,by=positions[n['id']]
        for parent in n['inputs']:
            ax,ay=positions[parent]
            points=[(ax+bw/2,ay+bh),(ax+bw/2,by-25),(bx+bw/2,by-25),(bx+bw/2,by)] if vertical else [(ax+bw,ay+bh/2),(bx-25,ay+bh/2),(bx-25,by+bh/2),(bx,by+bh/2)]
            arrow(sc,points,'#6c7685',o.get('line_width',1.5))
    for n in nodes:
        x,y=positions[n['id']];color=palette[category(n['op'])] if o.get('color_coding',True) else '#f5f5f5'
        sc.rect(x,y,bw,bh,color,'#cdd4de',r=o.get('radius',8));sc.rect(x+12,y+14,4,18,'#7889ac',r=2)
        if o.get('show_labels',True):
            parts=wrap_label(n['name'],font,bw-60)
            for j,part in enumerate(parts):sc.text(x+26,y+27+j*(font+3),part,font,'#263244',600)
        if n.get('repeat',1)>1:sc.text(x+bw-12,y+28,'×'+str(n['repeat']),10,'#61538a',600,'end')
        if o.get('show_shapes',True):sc.text(x+15,y+66,'N × '+' × '.join(map(str,n['shape'])),font-2,'#65748a')
        if o.get('show_params',True):sc.text(x+15,y+87,short(n['params'])+' parameters',font-3,'#7d8592')
        if n.get('annotation'):sc.text(x,y+bh+20,n['annotation'][:34],9,'#7d8592')
    if o.get('legend',True):sc.text(60,height-30,'NEUROFORGE  /  '+('Feature maps · Attention · Projection · Tensor flow' if style!='mono' else 'Architecture diagram'),9,'#969ca6')
    return sc


def blend(color,amount):
    rgb=[int(color[i:i+2],16) for i in (1,3,5)];return '#'+''.join(f'{max(0,min(255,round(v*amount))):02x}' for v in rgb)


def three_d(ir,o):
    nodes,spec=graph(ir);theme=o.get('theme','product');dark=theme=='dark';palette=PALETTES.get(theme,PALETTES['product'])
    compact=o.get('layout','compact')=='compact';cols=min(4,max(2,math.ceil(math.sqrt(len(nodes))))) if compact else min(6,len(nodes))
    angle=o.get('camera_angle',28);skew=math.tan(math.radians(angle))*.4;depth=o.get('depth',22);shadow=o.get('shadow',.6)
    bw,bh=225,132;positions={}
    for i,n in enumerate(nodes):
        row,col=divmod(i,cols);col=cols-1-col if row%2 else col
        positions[n['id']]=(100+col*290+row*90,220+row*230-col*18)
    w=max(x+bw+bh*skew for x,y in positions.values())+110;h=max(y+bh+depth for x,y in positions.values())+140
    bg='#101620' if dark else '#fafbfe';text='#ecf2fa' if dark else '#253147';muted='#8292aa' if dark else '#8490a5'
    sc=Scene(math.ceil(w),math.ceil(h),bg)
    sc.text(88,60,'NEUROFORGE   /   ARCHITECTURE COLLECTION',10,muted,600)
    sc.text(88,112,o.get('title') or ir.get('name','Neural architecture'),36,text,600)
    sc.text(90,142,o.get('subtitle') or 'Designed to think. Built to run.',14,muted)
    # A low-contrast floor gives the projected cards a shared spatial reference.
    if not dark:sc.poly([(55,h-110),(w-140,h-165),(w-55,h-62),(150,h-12)],'#f0f3fa')
    centers={k:(x+bw/2+bh*skew/2,y+bh/2) for k,(x,y) in positions.items()}
    for n in nodes:
        for parent in n['inputs']:
            a=centers[parent];b=centers[n['id']];dx=b[0]-a[0];dy=b[1]-a[1];length=math.hypot(dx,dy)
            start=(a[0]+dx/length*75,a[1]+dy/length*45);end=(b[0]-dx/length*78,b[1]-dy/length*49)
            arrow(sc,[start,((start[0]+end[0])/2,(start[1]+end[1])/2),end],'#53657f' if dark else '#b7c4db',2)
    for n in sorted(nodes,key=lambda n:positions[n['id']][1]):
        x,y=positions[n['id']];sx=bh*skew;color=palette[category(n['op'])];top=[(x+sx,y),(x+bw+sx,y),(x+bw,y+bh),(x,y+bh)]
        for layer in range(6,0,-1):
            shift=8+layer*3;shade='#121c29' if dark else ['#f7f8fc','#f3f5fa','#eef1f7','#e9edf5','#e6eaf2','#e3e8f1'][6-layer]
            if shadow>0:sc.poly([(px+shift*.6,py+depth+shift*shadow) for px,py in top],shade)
        sc.poly([(x,y+bh),(x+bw,y+bh),(x+bw,y+bh+depth),(x,y+bh+depth)],blend(color,.84),'#9aa9bd' if dark else '#d2dbea')
        sc.poly([(x+bw+sx,y),(x+bw,y+bh),(x+bw,y+bh+depth),(x+bw+sx,y+depth)],blend(color,.92),'#60728e' if dark else '#d6deeb')
        sc.poly(top,color,'#5b6b83' if dark else '#ffffff',1.5)
        # Stacked sheets convey a repeated block without duplicating labels.
        if n.get('repeat',1)>1:
            for layer in (2,1):sc.line([(x+5+sx+layer*3,y-layer*5),(x+bw+sx-5+layer*3,y-layer*5),(x+bw-5+layer*3,y+bh-10-layer*5)],'#776ba5' if dark else '#c3bdd8',1.2)
        tx=x+24+sx*.55
        sc.rect(tx,y+20,24,4,'#9584be' if category(n['op'])==1 else '#8da6bd',r=2)
        parts=wrap_label(n['name'],15,bw-38)
        for j,part in enumerate(parts):sc.text(tx,y+53+j*18,part,15,text,600)
        if o.get('show_shapes',True):sc.text(tx,y+91,' × '.join(map(str,n['shape'])),11,muted)
        if o.get('show_params',True):sc.text(tx,y+113,short(n['params'])+' params',10,muted)
        if n.get('repeat',1)>1:sc.text(x+bw-13,y+bh-20,'×'+str(n['repeat']),12,'#9482b9',600,'end')
    sc.text(88,h-45,f'{len(nodes)-1:02d} BLOCKS     /     {short(spec["params"])} PARAMETERS     /     {short(spec["macs"])} MACs',10,muted,500)
    sc.text(w-75,h-45,'NEUROFORGE AI',10,muted,600,'end')
    return sc


def render(ir,options):
    scene=three_d(ir,options) if options.get('mode')=='3d' else two_d(ir,options)
    kind=options.get('format','svg')
    if kind=='svg':return scene.svg(),'image/svg+xml'
    if kind=='png':return scene.png(options.get('resolution',2400)),'image/png'
    if kind=='pdf':return scene.pdf(options.get('paper_size','auto')),'application/pdf'
    raise ValueError('Supported formats: SVG, PNG, PDF')
