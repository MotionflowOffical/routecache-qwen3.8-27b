from __future__ import annotations

from pathlib import Path
from .util import log


def download_model(repo_id: str, filename: str, model_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {repo_id}/{filename} ...")
    path=hf_hub_download(repo_id=repo_id, filename=filename, repo_type='model', local_dir=str(model_dir))
    p=Path(path)
    if not p.exists() or p.stat().st_size < 1024*1024:
        raise RuntimeError(f"invalid model download: {p}")
    log(f"Model ready: {p.name} ({p.stat().st_size/2**30:.2f} GiB)")
    return p
