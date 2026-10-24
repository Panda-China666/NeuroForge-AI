import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(os.environ.get('NEUROFORGE_DATA', Path(__file__).resolve().parents[1]/'runtime')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
DATA = ROOT/'datasets'
ARTIFACTS = ROOT/'artifacts'
DATA.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
LOCK = threading.RLock()


@contextmanager
def connection():
    c = sqlite3.connect(ROOT/'neuroforge.sqlite3', timeout=30)
    try:
        c.execute('PRAGMA journal_mode=WAL')
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


with connection() as c:
    c.execute('CREATE TABLE IF NOT EXISTS objects (kind TEXT, id TEXT, body TEXT, updated REAL, PRIMARY KEY(kind,id))')


def put(kind, obj):
    obj = dict(obj)
    obj.setdefault('id', uuid.uuid4().hex[:16])
    obj.setdefault('created_at', time.time())
    obj['updated_at'] = time.time()
    with LOCK, connection() as c:
        c.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?,?)',
                  (kind, obj['id'], json.dumps(obj, ensure_ascii=False, allow_nan=False), obj['updated_at']))
    return obj


def get(kind, ident):
    with connection() as c:
        row = c.execute('SELECT body FROM objects WHERE kind=? AND id=?', (kind,ident)).fetchone()
    if row is None:
        raise KeyError(f'{kind} {ident} not found')
    return json.loads(row[0])


def all_of(kind):
    with connection() as c:
        return [json.loads(r[0]) for r in c.execute('SELECT body FROM objects WHERE kind=? ORDER BY updated DESC',(kind,))]


def update(kind, ident, **changes):
    with LOCK:
        obj = get(kind, ident)
        obj.update(changes)
        return put(kind, obj)
