from __future__ import annotations
import hashlib, json, os, shutil, struct, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path
from typing import Any
from huggingface_hub import hf_hub_download, snapshot_download
from .profile import load_profile, repo_root
from .util import log, read_json, write_json

UPSTREAM_REPO="Qwen/Qwen3.8-27B"
REFERENCE_REPO="unsloth/Qwen3.8-27B-GGUF"
REFERENCE_FILE="Qwen3.8-27B-UD-IQ2_S.gguf"
IMATRIX_FILE="imatrix_unsloth.gguf"
REFERENCE_SHA256="7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe"
MIN_FREE_BYTES=120*1024**3
GGUF_TYPES={0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4",40:"NVFP4",41:"Q1_0",42:"Q2_0"}
UNQUANTIZED={"F32","F16","BF16","I8","I16","I32","I64","F64"}

class NeedMore(Exception): pass
class Buf:
    def __init__(self,d:bytes): self.d=d; self.p=0
    def take(self,n):
        if self.p+n>len(self.d): raise NeedMore
        x=self.d[self.p:self.p+n]; self.p+=n; return x
    def u8(self): return struct.unpack('<B',self.take(1))[0]
    def i8(self): return struct.unpack('<b',self.take(1))[0]
    def u16(self): return struct.unpack('<H',self.take(2))[0]
    def i16(self): return struct.unpack('<h',self.take(2))[0]
    def u32(self): return struct.unpack('<I',self.take(4))[0]
    def i32(self): return struct.unpack('<i',self.take(4))[0]
    def u64(self): return struct.unpack('<Q',self.take(8))[0]
    def i64(self): return struct.unpack('<q',self.take(8))[0]
    def f32(self): return struct.unpack('<f',self.take(4))[0]
    def f64(self): return struct.unpack('<d',self.take(8))[0]
    def string(self): return self.take(self.u64()).decode('utf-8',errors='replace')

def _v(b:Buf,t:int):
    if t==0:return b.u8()
    if t==1:return b.i8()
    if t==2:return b.u16()
    if t==3:return b.i16()
    if t==4:return b.u32()
    if t==5:return b.i32()
    if t==6:return b.f32()
    if t==7:return bool(b.u8())
    if t==8:return b.string()
    if t==9:
        et=b.u32(); n=b.u64(); small=[] if n<=64 else None
        for _ in range(n):
            x=_v(b,et)
            if small is not None: small.append(x)
        return small if small is not None else {'array_len':n,'elem_type':et}
    if t==10:return b.u64()
    if t==11:return b.i64()
    if t==12:return b.f64()
    raise RuntimeError(f'unsupported GGUF metadata type {t}')

def parse_gguf_header(data:bytes)->dict[str,Any]:
    b=Buf(data)
    if b.take(4)!=b'GGUF': raise RuntimeError('not GGUF')
    version=b.u32(); nt=b.u64(); nk=b.u64(); md={}
    for _ in range(nk):
        key=b.string(); typ=b.u32(); md[key]=_v(b,typ)
    ts=[]
    for _ in range(nt):
        name=b.string(); nd=b.u32(); shape=[b.u64() for _ in range(nd)]; typ=b.u32(); off=b.u64()
        ts.append({'name':name,'shape':shape,'dtype_id':typ,'dtype':GGUF_TYPES.get(typ,f'TYPE_{typ}'),'offset':off})
    return {'version':version,'tensor_count':nt,'kv_count':nk,'metadata':md,'tensors':ts,'header_bytes':b.p}

def fetch_remote_header(url:str,max_bytes:int=128*1024**2):
    n=2*1024**2
    while n<=max_bytes:
        req=urllib.request.Request(url,headers={'Range':f'bytes=0-{n-1}','User-Agent':'RouteCache-reproducer/1.2'})
        with urllib.request.urlopen(req,timeout=180) as r: data=r.read(n)
        try:return parse_gguf_header(data)
        except NeedMore:n*=2
    raise RuntimeError('GGUF header exceeded 128 MiB safety cap')

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4*1024**2),b''): h.update(x)
    return h.hexdigest()

def _run(cmd,cwd=None,timeout=7200):
    log('RUN: '+subprocess.list2cmdline([str(x) for x in cmd]))
    cp=subprocess.run([str(x) for x in cmd],cwd=cwd,text=True,capture_output=True,timeout=timeout)
    if cp.returncode: raise RuntimeError(f'command failed ({cp.returncode})\n{cp.stdout[-5000:]}\n{cp.stderr[-5000:]}')
    return cp

def _llama_source(commit:str,work:Path):
    srcroot=work/'llama-src'; srcroot.mkdir(parents=True,exist_ok=True); z=srcroot/'llama.zip'; dst=srcroot/'source'
    if dst.exists():
        ds=[p for p in dst.iterdir() if p.is_dir()]
        if len(ds)==1:return ds[0]
    shutil.rmtree(dst,ignore_errors=True); dst.mkdir()
    urllib.request.urlretrieve(f'https://codeload.github.com/ggml-org/llama.cpp/zip/{commit}',z)
    with zipfile.ZipFile(z) as q:q.extractall(dst)
    ds=[p for p in dst.iterdir() if p.is_dir()]
    if len(ds)!=1:raise RuntimeError('unexpected llama.cpp source layout')
    return ds[0]

def _download_repro_tools(root:Path,profile:dict):
    url=str(profile.get('repro_tools_url') or '')
    if not url:return None
    out=root/'runtime'/'repro-tools'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True); z=out/'tools.zip'
    try:
        urllib.request.urlretrieve(url,z)
        with zipfile.ZipFile(z) as q:q.extractall(out/'unpacked')
        for n in ('llama-quantize.exe','llama-quantize'):
            h=list((out/'unpacked').rglob(n))
            if h:return h[0]
    except Exception as e: log(f'Prebuilt repro tools unavailable: {e}')
    return None

def _build_quantizer(src:Path,work:Path):
    cmake=shutil.which('cmake.exe' if os.name=='nt' else 'cmake')
    if not cmake:raise RuntimeError('CMake is required because the published repro-tools asset was unavailable.')
    b=work/'llama-build'; _run([cmake,'-S',src,'-B',b,'-DGGML_CUDA=OFF','-DGGML_NATIVE=ON','-DGGML_BACKEND_DL=OFF','-DBUILD_SHARED_LIBS=ON','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_BUILD_SERVER=OFF','-DCMAKE_BUILD_TYPE=Release'],timeout=1800)
    _run([cmake,'--build',b,'--config','Release','--target','llama-quantize','-j','6'],timeout=3600)
    for n in ('llama-quantize.exe','llama-quantize'):
        h=list(b.rglob(n))
        if h:return h[0]
    raise RuntimeError('llama-quantize build completed but executable was not found')

def _map(ref,out:Path):
    m={}
    for t in ref['tensors']:
        typ=t['dtype']
        if typ.startswith('TYPE_'):raise RuntimeError(f'unknown GGML type {typ}')
        if typ not in UNQUANTIZED:m[t['name']]=typ.lower()
    out.write_text('\n'.join(f'{k}={v}' for k,v in m.items())+'\n',encoding='utf-8');return m

def _validate(path:Path,ref):
    with path.open('rb') as f:data=f.read(128*1024**2)
    got=parse_gguf_header(data); a={t['name']:t['dtype'] for t in ref['tensors']}; b={t['name']:t['dtype'] for t in got['tensors']}; md=got['metadata']
    return {'tensor_count':got['tensor_count'],'reference_tensor_count':ref['tensor_count'],'same_tensor_names':set(a)==set(b),'same_tensor_types':set(a)==set(b) and all(a[k]==b[k] for k in a),'block_count':md.get('qwen35.block_count'),'nextn_predict_layers':md.get('qwen35.nextn_predict_layers')}

def reproduce(keep_intermediates=False):
    root=repo_root(); profile=load_profile(); work=root/'reproduction-work'; work.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(work).free
    if free<MIN_FREE_BYTES:raise RuntimeError(f'Need at least ~120 GiB free; only {free/2**30:.1f} GiB available')
    ref_url=f'https://huggingface.co/{REFERENCE_REPO}/resolve/main/{REFERENCE_FILE}'
    log('Reading public UD-IQ2_S tensor recipe via HTTP range requests...'); ref=fetch_remote_header(ref_url); md=ref['metadata']
    if ref['tensor_count']!=498 or int(md.get('qwen35.block_count',-1))!=64:raise RuntimeError('reference topology changed; refusing silent reproduction')
    typemap=work/'UD-IQ2_S.tensor-types.txt'; mapping=_map(ref,typemap); log(f'Captured {len(mapping)} quantized tensor assignments.')
    src=_llama_source(profile['llama_commit'],work)
    req=src/'requirements.txt'; _run([sys.executable,'-m','pip','install','-r',req],timeout=3600)
    upstream=Path(snapshot_download(repo_id=UPSTREAM_REPO,repo_type='model',local_dir=str(work/'qwen-upstream')))
    bf16=work/'Qwen3.8-27B-BF16.gguf'; _run([sys.executable,src/'convert_hf_to_gguf.py',upstream,'--outfile',bf16,'--outtype','bf16'],cwd=src,timeout=14400)
    imatrix=Path(hf_hub_download(repo_id=REFERENCE_REPO,filename=IMATRIX_FILE,repo_type='model',local_dir=str(work/'recipe')))
    quant=_download_repro_tools(root,profile) or _build_quantizer(src,work)
    out=root/'models'/profile['model_filename']; out.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(out)+'.building')
    if tmp.exists():tmp.unlink()
    threads=max(1,min(os.cpu_count() or 6,12))
    _run([quant,'--imatrix',imatrix,'--tensor-type-file',typemap,'--prune-layers','64','--override-kv','qwen35.block_count=int:64','--override-kv','qwen35.nextn_predict_layers=int:0',bf16,tmp,'IQ2_S',str(threads)],timeout=21600)
    if not tmp.exists() or tmp.stat().st_size<7*1024**3:raise RuntimeError('quantizer produced no plausible GGUF')
    st=_validate(tmp,ref)
    if not st['same_tensor_names'] or not st['same_tensor_types'] or st['block_count']!=64 or int(st.get('nextn_predict_layers') or 0)!=0:raise RuntimeError(f'reproduced topology/type validation failed: {st}')
    tmp.replace(out); digest=sha256(out)
    report={'schema':1,'status':'ready','upstream_repo':UPSTREAM_REPO,'reference_recipe_repo':REFERENCE_REPO,'reference_file':REFERENCE_FILE,'reference_sha256':REFERENCE_SHA256,'output':str(out.resolve()),'output_sha256':digest,'byte_identical_to_reference':digest.lower()==REFERENCE_SHA256.lower(),'structure':st,'llama_commit':profile['llama_commit'],'recipe':'official Qwen BF16 + Unsloth public imatrix/per-tensor UD-IQ2_S recipe + MTP prune/metadata rewrite'}
    write_json(root/'runtime'/'reproduction_report.json',report)
    if not keep_intermediates:
        shutil.rmtree(work/'qwen-upstream',ignore_errors=True)
        try:bf16.unlink()
        except Exception:pass
    return report
