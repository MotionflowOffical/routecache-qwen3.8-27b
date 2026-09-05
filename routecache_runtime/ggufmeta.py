from __future__ import annotations
import json, struct
from pathlib import Path
from typing import Any

_SIMPLE = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

class Reader:
    def __init__(self, f): self.f=f
    def take(self,n):
        b=self.f.read(n)
        if len(b)!=n: raise EOFError("truncated GGUF header")
        return b
    def u32(self): return struct.unpack('<I',self.take(4))[0]
    def u64(self): return struct.unpack('<Q',self.take(8))[0]
    def string(self): return self.take(self.u64()).decode('utf-8',errors='replace')

def _read_value(r:Reader,t:int,keep:bool=False):
    if t in _SIMPLE:
        fmt,n=_SIMPLE[t]; v=struct.unpack(fmt,r.take(n))[0]; return v if keep else None
    if t==8:
        s=r.string(); return s if keep else None
    if t==9:
        et=r.u32(); n=r.u64(); vals=[] if keep and n<=4096 else None
        for _ in range(n):
            x=_read_value(r,et,keep=vals is not None)
            if vals is not None: vals.append(x)
        return vals
    raise RuntimeError(f"unsupported GGUF metadata type {t}")

def routecache_metadata(path:Path)->dict[str,Any]:
    out={}
    with Path(path).open('rb') as f:
        r=Reader(f)
        if r.take(4)!=b'GGUF': raise RuntimeError('not a GGUF file')
        _version=r.u32(); _n_tensors=r.u64(); n_kv=r.u64()
        for _ in range(n_kv):
            key=r.string(); typ=r.u32(); keep=key.startswith('routecache.')
            val=_read_value(r,typ,keep=keep)
            if keep: out[key]=val
    return out

def embedded_profile(path:Path)->dict[str,Any] | None:
    md=routecache_metadata(path)
    raw=md.get('routecache.profile.json')
    if not isinstance(raw,str): return None
    try:
        obj=json.loads(raw)
        return obj if isinstance(obj,dict) else None
    except Exception:
        return None
