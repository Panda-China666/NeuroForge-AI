"""Versioned, content-addressed image assets. ZIP entries are read without extraction."""
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image, ImageDraw
import torch

MAX_BYTES = 100 * 1024 * 1024


def split_indices(labels, seed, val=.15, test=.15, groups=None):
    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    # Identical pixel content stays in the same split, preventing duplicate leakage.
    groups = groups or [str(i) for i in range(len(labels))]
    for cls in sorted(set(labels)):
        by_hash = {}
        for i, y in enumerate(labels):
            if y == cls:
                by_hash.setdefault(groups[i], []).append(i)
        buckets = list(by_hash.values())
        if len(buckets) < 3:
            raise ValueError(f'Class {cls} requires at least 3 unique images for train/val/test')
        rng.shuffle(buckets)
        nt, nv = max(1, round(len(buckets)*test)), max(1, round(len(buckets)*val))
        if nt+nv >= len(buckets):
            raise ValueError('Split ratios leave no training data')
        for name, selected in [('test', buckets[:nt]), ('val', buckets[nt:nt+nv]), ('train', buckets[nt+nv:])]:
            splits[name] += [i for group in selected for i in group]
    for values in splits.values():
        rng.shuffle(values)
    return splits


def finalize(root, images, labels, classes, name, seed=42, issues=None, source='upload', val=.15, test=.15, parent=None):
    arr = np.stack(images).astype(np.uint8)
    groups = [hashlib.sha256(a.tobytes()).hexdigest() for a in arr]
    known, conflicts = {}, 0
    for h, label in zip(groups, labels):
        if h in known and known[h] != label:
            conflicts += 1
        known[h] = label
    if conflicts:
        raise ValueError(f'{conflicts} identical images have conflicting labels; fix labels before import')
    splits = split_indices(labels, seed, val, test, groups)
    identity = hashlib.sha256(arr.tobytes()+np.array(labels, dtype=np.int64).tobytes()+
                              json.dumps([classes, seed, val, test], ensure_ascii=False).encode()).hexdigest()
    path = root / identity
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path/'images.npz', images=arr, labels=np.array(labels, dtype=np.int64))
    counts = [labels.count(i) for i in range(len(classes))]
    manifest = dict(id=identity, name=name, source=source, parent=parent, samples=len(labels), classes=classes,
                    counts=counts, shape=[3, arr.shape[1], arr.shape[2]], seed=seed,
                    splits=splits, split_counts={k: len(v) for k, v in splits.items()},
                    duplicates=len(groups)-len(set(groups)), issues=issues or [],
                    pipeline=[{'op':'decode_rgb_resize','size':int(arr.shape[1]),'resample':'bilinear','version':'0.1.0'},
                              {'op':'stratified_group_split','seed':seed,'val':val,'test':test,'version':'0.1.0'},
                              {'op':'scale_float','range':[0,1],'version':'0.1.0'}],
                    imbalance_ratio=max(counts)/min(counts))
    (path/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def builtin(root):
    rng = np.random.default_rng(42)
    images, labels = [], []
    for cls in range(3):
        for _ in range(100):
            img = Image.new('RGB', (32, 32), (15, 18, 24))
            draw = ImageDraw.Draw(img)
            x, y = map(int, rng.integers(3, 10, 2))
            extent = int(rng.integers(13, 22))
            color = tuple(map(int, rng.integers(120, 255, 3)))
            if cls == 0:
                draw.ellipse((x, y, x+extent, y+extent), fill=color)
            elif cls == 1:
                draw.rectangle((x, y, x+extent, y+extent), fill=color)
            else:
                draw.polygon([(x+extent//2,y),(x,y+extent),(x+extent,y+extent)], fill=color)
            a = np.array(img).astype(np.int16)
            a = np.clip(a+rng.integers(-12, 13, a.shape),0,255).astype(np.uint8)
            images.append(a)
            labels.append(cls)
    return finalize(root, images, labels, ['circle', 'square', 'triangle'], 'Shapes-300', source='synthetic:shapes-v1')


def import_zip(root, content, name, size, seed):
    if len(content) > MAX_BYTES:
        raise ValueError('ZIP exceeds 100 MB')
    images, class_names, issues, seen_paths = [], [], [], set()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        entries = [e for e in z.infolist() if not e.is_dir()]
        if len(entries) > 5000 or sum(e.file_size for e in entries) > 256*1024*1024:
            raise ValueError('Archive exceeds 5000 files or 256 MB uncompressed')
        for e in entries:
            p = PurePosixPath(e.filename.replace('\\', '/'))
            if p.is_absolute() or '..' in p.parts or ':' in e.filename:
                raise ValueError('Unsafe ZIP path')
            if any(part.startswith('.') or part == '__MACOSX' for part in p.parts):
                continue
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
                continue
            if e.filename in seen_paths or e.file_size > 10*1024*1024:
                raise ValueError('Duplicate ZIP filename or image exceeds 10 MB')
            seen_paths.add(e.filename)
            if len(p.parts) < 2:
                raise ValueError('ZIP must contain class_name/image.png folders')
            try:
                with Image.open(io.BytesIO(z.read(e))) as img:
                    if img.width*img.height > 16_000_000:
                        raise ValueError('Image exceeds 16 megapixels')
                    a = np.asarray(img.convert('RGB').resize((size,size), Image.Resampling.BILINEAR))
                images.append(a)
                class_names.append(p.parent.name)
            except Exception as exc:
                issues.append({'file':e.filename,'error':str(exc)[:160]})
    classes = sorted(set(class_names))
    if not 2 <= len(classes) <= 100:
        raise ValueError('Require 2–100 classes')
    labels = [classes.index(c) for c in class_names]
    return finalize(root, images, labels, classes, name, seed, issues, source='zip-upload')


def load(root, ident):
    path = root / ident
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    with np.load(path/'images.npz', allow_pickle=False) as a:
        x = torch.from_numpy(a['tokens'].copy()).long() if 'tokens' in a else torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255.
        y = torch.from_numpy(a['labels'].copy())
    return x, y, manifest


def tokens(root,records,name='Token dataset',seed=42,vocab_size=128):
    if not isinstance(records,list) or not 6<=len(records)<=10000:raise ValueError('Token dataset requires 6–10000 samples')
    classes=sorted({str(r['label']) for r in records})
    if not 2<=len(classes)<=100:raise ValueError('Token dataset requires 2–100 classes')
    length=len(records[0]['tokens'])
    if not 1<=length<=512:raise ValueError('Token sequence length must be 1–512')
    if not 2<=vocab_size<=65536:raise ValueError('vocab_size must be in [2,65536]')
    for r in records:
        if len(r['tokens'])!=length or any(type(v)!=int or not 0<=v<vocab_size for v in r['tokens']):raise ValueError('Token IDs must be within vocabulary and all sequences must have the same length')
    array=np.asarray([r['tokens'] for r in records],dtype=np.int64);labels=[classes.index(str(r['label'])) for r in records]
    groups=[hashlib.sha256(row.tobytes()).hexdigest() for row in array];known={}
    for h,y in zip(groups,labels):
        if h in known and known[h]!=y:raise ValueError('Identical token sequences have conflicting labels')
        known[h]=y
    splits=split_indices(labels,seed,groups=groups)
    ident=hashlib.sha256(array.tobytes()+np.asarray(labels,dtype=np.int64).tobytes()+json.dumps([classes,seed,vocab_size]).encode()).hexdigest()
    path=root/ident;path.mkdir(exist_ok=True);np.savez_compressed(path/'images.npz',tokens=array,labels=np.asarray(labels,dtype=np.int64))
    counts=[labels.count(i) for i in range(len(classes))]
    m=dict(id=ident,name=name,source='tokens-json',input_type='tokens',vocab_size=vocab_size,samples=len(records),classes=classes,counts=counts,shape=[length],seed=seed,
           splits=splits,split_counts={k:len(v) for k,v in splits.items()},duplicates=len(groups)-len(set(groups)),issues=[],imbalance_ratio=max(counts)/min(counts),
           pipeline=[{'op':'validate_integer_tokens','vocab_size':vocab_size},{'op':'stratified_group_split','seed':seed,'val':.15,'test':.15}])
    (path/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');return m


def builtin_tokens(root):
    rng=np.random.default_rng(42);records=[]
    for cls in range(3):
        for i in range(100):
            values=rng.integers(cls*40+1,cls*40+40,32).tolist()
            records.append({'tokens':values,'label':['pattern-a','pattern-b','pattern-c'][cls]})
    return tokens(root,records,'Token Patterns · 300')
