from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def _tensor_digest(reader):
    h=hashlib.sha256(); sig=[]
    for t in reader.tensors:
        name=str(t.name); typ=int(t.tensor_type); n=int(t.n_bytes); shape=tuple(int(x) for x in t.data.shape)
        sig.append((name,typ,n,shape)); h.update(name.encode()); h.update(b'\0'); h.update(typ.to_bytes(4,'little')); h.update(n.to_bytes(8,'little')); h.update(memoryview(t.data).cast('B'))
    return h.hexdigest(),sig

def build_embedded_profile(public_profile:dict[str,Any],source_sha:str)->dict[str,Any]:
    r=dict(public_profile.get('certified_route') or {})
    q=dict(public_profile.get('quality') or {})
    c=dict(public_profile.get('calibration') or {})
    return {
        'schema':1,'routecache_version':'1.3.0','profile_id':r.get('name','routecache-rtx4060'),
        'model_family':'Qwen3.8-27B','quantization':'UD-IQ2_S','context':int(r.get('context',4096)),
        'batch':int(r.get('batch',256)),'ubatch':int(r.get('ubatch',64)),'kv':r.get('kv','q8_0'),'override':r.get('override',''),'fit':False,
        'warmup_rounds':2,'context_compaction':True,'context_shift':True,
        'certified_hardware':public_profile.get('certified_hardware') or {},
        'quality':{'teacher_ppl':q.get('teacher_ppl',1.55),'model_ppl':q.get('candidate_ppl',1.6384),'ratio':q.get('ppl_ratio',1.0570322581)},
        'performance':{'stable_tps':c.get('stable_tps_floor',11.3916054301),'mtp_production':False},
        'source':{'repo':'unsloth/Qwen3.8-27B-GGUF','filename':'Qwen3.8-27B-UD-IQ2_S.gguf','sha256':source_sha,'base_model':'Qwen/Qwen3.8-27B'},
        'distribution':{'github_repo':public_profile.get('github_repo',''),'hf_repo':public_profile.get('hf_repo','')},
    }

def bake_file(source:Path,output:Path,public_profile:dict[str,Any])->dict[str,Any]:
    import gguf
    source=source.resolve(); output=output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    if source==output: raise RuntimeError('source and baked output must differ')
    source_sha=sha256(source); profile=build_embedded_profile(public_profile,source_sha)
    src=gguf.GGUFReader(source,'r'); payload,sig=_tensor_digest(src)
    arch=src.get_field(gguf.Keys.General.ARCHITECTURE).contents(); w=gguf.GGUFWriter(output,arch=arch,endianess=src.endianess)
    align=src.get_field(gguf.Keys.General.ALIGNMENT)
    if align is not None:w.data_alignment=int(align.contents())
    replacements={'general.name':(gguf.GGUFValueType.STRING,'Qwen3.8-27B UD-IQ2_S RouteCache'),'general.description':(gguf.GGUFValueType.STRING,'Qwen3.8-27B UD-IQ2_S with embedded RouteCache execution profile; tensor payload preserved.')}
    for f in src.fields.values():
        if f.name==gguf.Keys.General.ARCHITECTURE or f.name.startswith('GGUF.') or f.name.startswith('routecache.'):continue
        typ=f.types[0]; sub=f.types[-1] if typ==gguf.GGUFValueType.ARRAY else None
        if f.name in replacements:
            t,v=replacements.pop(f.name); w.add_key_value(f.name,v,t)
        else:w.add_key_value(f.name,f.contents(),typ,sub_type=sub)
    for k,(t,v) in replacements.items():w.add_key_value(k,v,t)
    compact=json.dumps(profile,sort_keys=True,separators=(',',':'))
    extras={
        'routecache.schema_version':(gguf.GGUFValueType.UINT32,1),'routecache.version':(gguf.GGUFValueType.STRING,'1.3.0'),
        'routecache.profile.id':(gguf.GGUFValueType.STRING,profile['profile_id']),'routecache.profile.json':(gguf.GGUFValueType.STRING,compact),
        'routecache.source.sha256':(gguf.GGUFValueType.STRING,source_sha),'routecache.source.tensor_payload_sha256':(gguf.GGUFValueType.STRING,payload),
        'routecache.runtime.context':(gguf.GGUFValueType.UINT32,profile['context']),'routecache.runtime.batch':(gguf.GGUFValueType.UINT32,profile['batch']),
        'routecache.runtime.ubatch':(gguf.GGUFValueType.UINT32,profile['ubatch']),'routecache.runtime.kv_type':(gguf.GGUFValueType.STRING,profile['kv']),
        'routecache.runtime.tensor_override':(gguf.GGUFValueType.STRING,profile['override']),'routecache.runtime.context_compaction':(gguf.GGUFValueType.BOOL,True),
        'routecache.runtime.context_shift':(gguf.GGUFValueType.BOOL,True),'routecache.runtime.warmup_rounds':(gguf.GGUFValueType.UINT32,2),
    }
    for k,(t,v) in extras.items():w.add_key_value(k,v,t)
    for t in src.tensors:w.add_tensor_info(t.name,t.data.shape,t.data.dtype,t.data.nbytes,t.tensor_type)
    w.write_header_to_file();w.write_kv_data_to_file();w.write_ti_data_to_file()
    for t in src.tensors:w.write_tensor_data(t.data,tensor_endianess=src.endianess)
    w.close()
    dst=gguf.GGUFReader(output,'r'); payload2,sig2=_tensor_digest(dst)
    if payload2!=payload or sig2!=sig: raise RuntimeError('tensor payload/signature changed during bake')
    return {'output':str(output),'source_sha256':source_sha,'baked_sha256':sha256(output),'tensor_payload_sha256':payload,'tensor_payload_identical':True,'embedded_profile':profile}
