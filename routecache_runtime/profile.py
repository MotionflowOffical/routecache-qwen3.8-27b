from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import read_json


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_profile() -> dict[str, Any]:
    p = repo_root() / "routecache_profile.json"
    return read_json(p, {}) or {}


@dataclass
class Hardware:
    gpu_name: str = "unknown"
    vram_mib: int = 0
    compute_capability: str = ""

    @property
    def cuda_arch(self) -> str:
        return self.compute_capability.replace('.', '')


def detect_hardware() -> Hardware:
    exe = shutil.which('nvidia-smi.exe' if os.name == 'nt' else 'nvidia-smi')
    if not exe:
        return Hardware()
    queries = [
        ['--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'],
        ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
    ]
    for q in queries:
        try:
            cp=subprocess.run([exe,*q], text=True, capture_output=True, timeout=10)
            if cp.returncode != 0 or not cp.stdout.strip():
                continue
            parts=[x.strip() for x in cp.stdout.splitlines()[0].split(',')]
            name=parts[0]
            mem=int(float(parts[1])) if len(parts)>1 else 0
            cc=parts[2] if len(parts)>2 else ('8.9' if 'RTX 40' in name else '')
            return Hardware(name, mem, cc)
        except Exception:
            continue
    return Hardware()


def is_certified_profile(hw: Hardware, profile: dict[str, Any]) -> bool:
    cert=profile.get('certified_hardware') or {}
    expected_name=str(cert.get('gpu') or '').strip().lower()
    actual_name=str(hw.gpu_name or '').strip().lower()
    # The measured tensor split is hardware-specific. Do not apply the RTX 4060
    # calibration to a different sm_89 card merely because VRAM size matches.
    if expected_name:
        key=expected_name.replace('nvidia geforce ', '').replace(' 8gb', '').strip()
        actual_key=actual_name.replace('nvidia geforce ', '').replace(' 8gb', '').strip()
        if key and key != actual_key:
            return False
    if str(cert.get('compute_capability') or '') and str(cert['compute_capability']) != hw.compute_capability:
        return False
    lo=int(cert.get('vram_mib_min') or 7000)
    hi=int(cert.get('vram_mib_max') or 9000)
    return lo <= hw.vram_mib <= hi
