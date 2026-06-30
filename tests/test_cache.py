from mars_core.cache import DiskCache


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path, enabled=True)
    payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    assert cache.get(payload) is None
    cache.set(payload, {"content": "hello"})
    assert cache.get(payload) == {"content": "hello"}
