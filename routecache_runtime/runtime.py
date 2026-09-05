from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any

from .context_proxy import CompactionSettings, ContextProxy, wait_for_health, warm_backend
from .profile import detect_hardware, is_certified_profile, load_profile, repo_root
from .util import log, read_json, write_json


def _server_help(server: Path) -> str:
    cp=subprocess.run([str(server),'--help'], text=True, capture_output=True, timeout=20)
    return (cp.stdout or '')+(cp.stderr or '')


def _runtime_manifest(root: Path) -> dict[str, Any]:
    m=read_json(root/'runtime'/'manifest.json',{}) or {}
    if not m: raise RuntimeError('Runtime is not installed. Run INSTALL.bat first.')
    return m


def route_for_hardware(profile: dict[str,Any]) -> tuple[dict[str,Any],bool]:
    hw=detect_hardware(); certified=is_certified_profile(hw,profile)
    return (dict(profile['certified_route']) if certified else dict(profile['portable_route'])), certified


def server_command(server: Path, model: Path, route: dict[str,Any], port: int, ui: bool) -> list[str]:
    helptext=_server_help(server)
    cmd=[str(server),'-m',str(model),'--host','127.0.0.1','--port',str(port),'-c',str(route.get('context',4096)),'--parallel','1']
    if route.get('mode')=='certified-explicit':
        cmd += ['-b',str(route['batch']),'-ub',str(route['ubatch']),'-fa','auto','-ctk',route['kv'],'-ctv',route['kv'],'-ngl','all','-fit','off','--jinja']
        ov=route.get('override') or ''
        if ov: cmd += ['-ot',ov.replace(';',',')]
        if '--cache-ram' in helptext: cmd += ['--cache-ram','0']
        if '--ctx-checkpoints' in helptext: cmd += ['--ctx-checkpoints','0']
    else:
        cmd += ['-fa','auto','-ngl','auto','-fit','on','--jinja']
    if '--context-shift' in helptext: cmd += ['--context-shift']
    if '--keep' in helptext: cmd += ['--keep','512']
    if '--reasoning' in helptext: cmd += ['--reasoning','auto']
    if '--metrics' in helptext: cmd += ['--metrics']
    if ui:
        if '--ui' in helptext: cmd += ['--ui']
        elif '--webui' in helptext: cmd += ['--webui']
    else:
        if '--no-ui' in helptext: cmd += ['--no-ui']
        elif '--no-webui' in helptext: cmd += ['--no-webui']
    return cmd


def serve(port: int=8080, ui: bool=False, warmup: bool=True, compact: bool=True) -> int:
    root=repo_root(); manifest=_runtime_manifest(root)
    server=Path(manifest['server']); model=Path(manifest['model']); profile=load_profile(model)
    if not server.exists(): raise RuntimeError(f'missing llama-server: {server}')
    if not model.exists(): raise RuntimeError(f'missing model: {model}')
    route, certified=route_for_hardware(profile)
    backend_port=port+1 if compact else port
    cmd=server_command(server,model,route,backend_port,ui)
    log('Certified RTX 4060 route active.' if certified else 'Portable auto-fit route active; no RTX 4060 TPS claim applies.')
    log('Command: '+subprocess.list2cmdline(cmd))
    flags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0) if os.name=='nt' else 0
    proc=subprocess.Popen(cmd, creationflags=flags)
    proxy=None
    try:
        if not wait_for_health(backend_port,180): raise RuntimeError(f'llama-server failed to become healthy (exit={proc.poll()})')
        if warmup: warm_backend(backend_port,rounds=2,tokens=96)
        if compact:
            proxy=ContextProxy(port,backend_port,CompactionSettings(context_tokens=int(route.get('context',4096))))
            proxy.start(); log(f'Automatic context compaction proxy: http://127.0.0.1:{port}')
        if ui:
            webbrowser.open(f'http://127.0.0.1:{port}')
        log('Ready. This console remains open while the server is running.')
        while proc.poll() is None: time.sleep(.5)
        return int(proc.returncode or 0)
    except KeyboardInterrupt:
        return 130
    finally:
        if proxy: proxy.stop()
        if proc.poll() is None:
            try: proc.terminate(); proc.wait(5)
            except Exception:
                try: proc.kill()
                except Exception: pass
