from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from .profile import load_profile, repo_root
from .util import log


def write_modelfile(model: Path|None=None) -> Path:
    root=repo_root(); profile=load_profile()
    model=model or (root/'models'/profile['model_filename'])
    if not model.exists(): raise RuntimeError(f'model not found: {model}')
    p=root/'runtime'/'Modelfile'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(f'FROM "{model}"\nPARAMETER num_ctx 4096\n',encoding='utf-8')
    return p


def create(name: str='qwen3.8-27b-routecache') -> int:
    exe=shutil.which('ollama.exe') or shutil.which('ollama')
    if not exe: raise RuntimeError('Ollama is not installed or not on PATH.')
    mf=write_modelfile(); log(f'Creating Ollama model {name} with num_ctx=4096 ...')
    return subprocess.call([exe,'create',name,'-f',str(mf)])
