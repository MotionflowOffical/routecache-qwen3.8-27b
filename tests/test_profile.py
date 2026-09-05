from routecache_runtime.profile import Hardware, is_certified_profile

def test_profile_gate():
    p={'certified_hardware':{'compute_capability':'8.9','vram_mib_min':7000,'vram_mib_max':9000}}
    assert is_certified_profile(Hardware('RTX 4060',8188,'8.9'),p)
    assert not is_certified_profile(Hardware('RTX 3060',12000,'8.6'),p)


def test_streaming_proxy_uses_low_latency_chunking_source():
    from pathlib import Path
    p = Path(__file__).parents[1] / 'routecache_runtime' / 'context_proxy.py'
    s = p.read_text(encoding='utf-8')
    assert 'chunk_size=8192' not in s
    assert 'Transfer-Encoding", "chunked"' in s
    assert 'TCP_NODELAY' in s
    assert 'r.raw.readline()' in s
