from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .download import download_model
from .kernel import build_kernel
from .profile import detect_hardware, load_profile, repo_root
from .util import log, write_json


def _server_in_dir(d: Path) -> Path | None:
    for n in ('llama-server.exe','llama-server'):
        hits=list(d.rglob(n)) if d.exists() else []
        if hits: return hits[0]
    return None


def _download_fallback(url: str, root: Path) -> Path | None:
    if not url: return None
    out=root/'runtime'/'fallback-download'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    z=out/'runtime.zip'; log('Downloading known-good fallback runtime release asset ...')
    urllib.request.urlretrieve(url,z)
    with zipfile.ZipFile(z) as zf: zf.extractall(out/'unpacked')
    return _server_in_dir(out/'unpacked')


def _install_with_model(root: Path, profile: dict[str,Any], model: Path, repo: str, no_kernel: bool=False) -> dict[str,Any]:
    hw=detect_hardware(); arch=hw.cuda_arch or str(profile.get('kernel',{}).get('default_arch') or '89')
    kernel={'ok':False,'reason':'disabled'}
    server=None; kind=''
    if not no_kernel and hw.compute_capability:
        kernel=build_kernel(root,profile['llama_commit'],arch,int(profile.get('kernel',{}).get('jobs',6)))
        if kernel.get('ok'):
            server=Path(kernel['server']); kind=f'custom-sm{arch}'
    if server is None:
        bundled=_server_in_dir(root/'vendor'/'win-x64-cuda')
        if bundled: server=bundled; kind='bundled-known-good-fallback'
    if server is None:
        server=_download_fallback(str(profile.get('fallback_runtime_url') or ''),root)
        if server: kind='downloaded-known-good-fallback'
    if server is None:
        # Last resort only: an externally installed server is accepted only when
        # it is the same pinned commit and exposes the RouteCache-required flags.
        q=shutil.which('llama-server.exe' if os.name=='nt' else 'llama-server')
        if q:
            try:
                import subprocess
                ver=subprocess.run([q,'--version'],text=True,capture_output=True,timeout=15)
                help_cp=subprocess.run([q,'--help'],text=True,capture_output=True,timeout=15)
                vt=(ver.stdout or '')+(ver.stderr or '')
                ht=(help_cp.stdout or '')+(help_cp.stderr or '')
                if profile['llama_commit'][:8] in vt and 'override-tensor' in ht and '--context-shift' in ht:
                    server=Path(q); kind='system-exact-commit-llama-server'
            except Exception:
                pass
    if server is None:
        raise RuntimeError('No usable llama-server runtime. Install CMake + CUDA Toolkit for custom build, or publish/configure the fallback GitHub Release asset URL.')
    manifest={'schema':1,'hf_repo':repo,'model':str(model.resolve()),'server':str(server.resolve()),'runtime_kind':kind,'kernel_attempt':kernel,'hardware':hw.__dict__}
    write_json(root/'runtime'/'manifest.json',manifest)
    log(f'Installed runtime: {kind}')
    return manifest

def install(hf_repo: str|None=None, no_kernel: bool=False) -> dict[str,Any]:
    root=repo_root(); profile=load_profile()
    repo=hf_repo or os.environ.get('ROUTECACHE_HF_REPO') or profile.get('hf_repo')
    if not repo or str(repo).startswith('YOUR_'):
        raise RuntimeError('Set the Hugging Face repo: INSTALL.bat USER/REPO or ROUTECACHE_HF_REPO=USER/REPO')
    model=download_model(str(repo),profile['model_filename'],root/'models')
    return _install_with_model(root,profile,model,str(repo),no_kernel)

def install_local(model: Path|None=None, no_kernel: bool=False) -> dict[str,Any]:
    root=repo_root(); profile=load_profile()
    model=Path(model) if model else root/'models'/profile['model_filename']
    if not model.exists():
        raise RuntimeError(f'Local reproduced model is missing: {model}')
    return _install_with_model(root,profile,model,'local-reproduction',no_kernel)
