from mars_core.cache import DiskCache


def test_disk_cache_roundtrip(tmp_path):
    cache = DiskCache(tmp_path, enabled=True)
    payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    assert cache.get(payload) is None
    cache.set(payload, {"content": "hello"})
    assert cache.get(payload) == {"content": "hello"}


def test_disk_cache_disabled_returns_none(tmp_path):
    cache = DiskCache(tmp_path, enabled=False)
    payload = {"x": 1}
    cache.set(payload, {"value": 1})
    assert cache.get(payload) is None


def test_disk_cache_same_payload_same_key(tmp_path):
    cache = DiskCache(tmp_path, enabled=True)
    payload = {"b": 2, "a": 1}
    assert cache.make_key(payload) == cache.make_key({"a": 1, "b": 2})
