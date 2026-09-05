from __future__ import annotations

import argparse
import json
from .installer import install, install_local
from .kernel import build_kernel
from .ollama import create as ollama_create
from .profile import detect_hardware, load_profile, repo_root
from .reproduce import reproduce
from .runtime import serve
from .util import read_json, write_json


def cmd_install(a):
    m=install(a.hf_repo,a.no_kernel); print(json.dumps(m,indent=2)); return 0

def cmd_install_local(a):
    m=install_local(no_kernel=a.no_kernel); print(json.dumps(m,indent=2)); return 0

def cmd_run(a): return serve(a.port,a.ui,not a.no_warmup,not a.no_compaction)

def cmd_kernel(a):
    root=repo_root(); p=load_profile(); hw=detect_hardware(); arch=hw.cuda_arch or str(p.get('kernel',{}).get('default_arch','89'))
    r=build_kernel(root,p['llama_commit'],arch,int(p.get('kernel',{}).get('jobs',6))); print(json.dumps(r,indent=2))
    if r.get('ok'):
        m=read_json(root/'runtime'/'manifest.json',{}) or {}; m['server']=r['server']; m['runtime_kind']=f'custom-sm{arch}'; m['kernel_attempt']=r; write_json(root/'runtime'/'manifest.json',m); return 0
    return 4

def cmd_ollama(a): return ollama_create(a.name)

def cmd_reproduce(a):
    r=reproduce(a.keep_intermediates); print(json.dumps(r,indent=2)); return 0

def main(argv=None):
    ap=argparse.ArgumentParser('routecache-runtime'); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('install'); p.add_argument('hf_repo',nargs='?'); p.add_argument('--no-kernel',action='store_true'); p.set_defaults(func=cmd_install)
    p=sp.add_parser('install-local'); p.add_argument('--no-kernel',action='store_true'); p.set_defaults(func=cmd_install_local)
    p=sp.add_parser('run'); p.add_argument('--port',type=int,default=8080); p.add_argument('--ui',action='store_true'); p.add_argument('--no-warmup',action='store_true'); p.add_argument('--no-compaction',action='store_true'); p.set_defaults(func=cmd_run)
    p=sp.add_parser('rebuild-kernel'); p.set_defaults(func=cmd_kernel)
    p=sp.add_parser('ollama-create'); p.add_argument('--name',default='qwen3.8-27b-routecache'); p.set_defaults(func=cmd_ollama)
    p=sp.add_parser('reproduce'); p.add_argument('--keep-intermediates',action='store_true'); p.set_defaults(func=cmd_reproduce)
    a=ap.parse_args(argv); return int(a.func(a))
