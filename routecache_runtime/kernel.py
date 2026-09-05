from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .util import log, write_json


def _run(cmd, timeout=1800, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)


def find_nvcc() -> Path | None:
    q=shutil.which('nvcc.exe' if os.name=='nt' else 'nvcc')
    if q:
        return Path(q)
    base=os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if base:
        p=Path(base)/'bin'/('nvcc.exe' if os.name=='nt' else 'nvcc')
        if p.exists(): return p
    return None


def _source(commit: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    z=work/f'llama-{commit}.zip'
    url=f'https://codeload.github.com/ggml-org/llama.cpp/zip/{commit}'
    log(f'Downloading llama.cpp {commit} ...')
    req=urllib.request.Request(url, headers={'User-Agent':'RouteCache-public/1.2'})
    with urllib.request.urlopen(req, timeout=180) as r, z.open('wb') as f:
        shutil.copyfileobj(r,f,1024*1024)
    src=work/'source'
    shutil.rmtree(src, ignore_errors=True); src.mkdir()
    with zipfile.ZipFile(z) as zf: zf.extractall(src)
    dirs=[p for p in src.iterdir() if p.is_dir()]
    if len(dirs)!=1: raise RuntimeError('unexpected llama.cpp archive layout')
    return dirs[0]


def _built_server(build: Path) -> Path | None:
    names={'llama-server.exe','llama-server'}
    hits=[p for p in build.rglob('*') if p.is_file() and p.name in names]
    hits.sort(key=lambda p:(0 if 'release' in str(p).lower() else 1,len(str(p))))
    return hits[0] if hits else None


def build_kernel(root: Path, commit: str, arch: str, jobs: int=6) -> dict[str, Any]:
    cmake=shutil.which('cmake.exe' if os.name=='nt' else 'cmake')
    nvcc=find_nvcc()
    report: dict[str,Any]={'ok':False,'commit':commit,'arch':arch}
    if not cmake or not nvcc:
        report.update(reason='missing-build-tools', missing=[x for x,v in [('cmake',cmake),('nvcc',nvcc)] if not v])
        return report
    work=root/'runtime'/'build-sm'
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    try:
        src=_source(commit, work)
        build=work/'build'
        cmd=[
            cmake,'-S',str(src),'-B',str(build),
            '-DGGML_CUDA=ON', f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCMAKE_CUDA_ARCHITECTURES={arch}',
            '-DGGML_NATIVE=ON','-DGGML_BACKEND_DL=OFF','-DBUILD_SHARED_LIBS=ON',
            '-DGGML_CUDA_GRAPHS=ON','-DGGML_CUDA_FA=ON','-DGGML_CUDA_NCCL=OFF','-DGGML_RPC=OFF',
            '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=OFF',
            '-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_APP=OFF','-DLLAMA_BUILD_UI=ON','-DLLAMA_USE_PREBUILT_UI=ON',
            '-DLLAMA_OPENSSL=OFF','-DCMAKE_BUILD_TYPE=Release'
        ]
        env=dict(os.environ); env.setdefault('CUDACXX',str(nvcc))
        log(f'Compiling custom CUDA runtime for sm_{arch} ...')
        cp=_run(cmd,1800,env=env)
        report['configure_command']=cmd; report['configure_stdout_tail']=cp.stdout[-6000:]; report['configure_stderr_tail']=cp.stderr[-6000:]
        if cp.returncode!=0:
            report['reason']='cmake-configure-failed'; return report
        bcmd=[cmake,'--build',str(build),'--config','Release','--target','llama-server','-j',str(max(1,jobs))]
        bp=_run(bcmd,3600,env=env)
        report['build_command']=bcmd; report['build_stdout_tail']=bp.stdout[-6000:]; report['build_stderr_tail']=bp.stderr[-6000:]
        if bp.returncode!=0:
            report['reason']='cmake-build-failed'; return report
        server=_built_server(build)
        if not server:
            report['reason']='server-not-found'; return report
        out=root/'runtime'/'custom-bin'; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
        for p in server.parent.iterdir():
            if p.is_file() and (p.name==server.name or p.suffix.lower()=='.dll'):
                shutil.copy2(p,out/p.name)
        report.update(ok=True, reason='ok', server=str(out/server.name), bin_dir=str(out))
        return report
    except Exception as e:
        report.update(reason='exception', error=str(e)); return report
    finally:
        write_json(root/'runtime'/'kernel_build.json', report)
