from __future__ import annotations
import os, shutil, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .util import read_json
from .ggufmeta import embedded_profile

def repo_root()->Path:return Path(__file__).resolve().parents[1]

def _merge_embedded(base:dict[str,Any], emb:dict[str,Any])->dict[str,Any]:
    out=dict(base)
    out['gguf_profile_embedded']=True
    if emb.get('certified_hardware'):out['certified_hardware']=emb['certified_hardware']
    out['certified_route']={
        'mode':'certified-explicit','name':emb.get('profile_id','routecache-embedded'),'context':int(emb.get('context',4096)),
        'batch':int(emb.get('batch',256)),'ubatch':int(emb.get('ubatch',64)),'kv':emb.get('kv','q8_0'),'override':emb.get('override','')}
    out['embedded_routecache_profile']=emb
    return out

def load_profile(model_path:Path|None=None)->dict[str,Any]:
    base=read_json(repo_root()/'routecache_profile.json',{}) or {}
    if model_path and Path(model_path).exists():
        try:
            emb=embedded_profile(Path(model_path))
            if emb:return _merge_embedded(base,emb)
        except Exception:pass
    return base

@dataclass
class Hardware:
    gpu_name:str='unknown';vram_mib:int=0;compute_capability:str=''
    @property
    def cuda_arch(self)->str:return self.compute_capability.replace('.','')

def detect_hardware()->Hardware:
    exe=shutil.which('nvidia-smi.exe' if os.name=='nt' else 'nvidia-smi')
    if not exe:return Hardware()
    for q in [['--query-gpu=name,memory.total,compute_cap','--format=csv,noheader,nounits'],['--query-gpu=name,memory.total','--format=csv,noheader,nounits']]:
        try:
            cp=subprocess.run([exe,*q],text=True,capture_output=True,timeout=10)
            if cp.returncode or not cp.stdout.strip():continue
            p=[x.strip() for x in cp.stdout.splitlines()[0].split(',')];name=p[0];mem=int(float(p[1])) if len(p)>1 else 0;cc=p[2] if len(p)>2 else ('8.9' if 'RTX 40' in name else '')
            return Hardware(name,mem,cc)
        except Exception:continue
    return Hardware()

def is_certified_profile(hw:Hardware,profile:dict[str,Any])->bool:
    cert=profile.get('certified_hardware') or {};expected=str(cert.get('gpu') or '').lower().replace('nvidia geforce ','').replace(' 8gb','').strip();actual=str(hw.gpu_name or '').lower().replace('nvidia geforce ','').replace(' 8gb','').strip()
    if expected and expected!=actual:return False
    if str(cert.get('compute_capability') or '') and str(cert['compute_capability'])!=hw.compute_capability:return False
    return int(cert.get('vram_mib_min') or 7000)<=hw.vram_mib<=int(cert.get('vram_mib_max') or 9000)
