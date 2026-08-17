"""Async-safe cache behavior including per-key single-flight."""

import asyncio

import pytest

from agentic_service.cache import AsyncTTLCache


@pytest.mark.asyncio
async def test_get_set_roundtrip() -> None:
    cache = AsyncTTLCache(ttl_seconds=60)
    await cache.set("k", {"v": 1})
    assert await cache.get("k") == {"v": 1}


@pytest.mark.asyncio
async def test_ttl_expiry() -> None:
    cache = AsyncTTLCache(ttl_seconds=0)
    await cache.set("k", "x")
    await asyncio.sleep(0.01)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_concurrent_sets_do_not_raise() -> None:
    cache = AsyncTTLCache(ttl_seconds=60)

    async def writer(i: int) -> None:
        await cache.set(f"k{i}", i)
        await cache.get(f"k{i}")

    await asyncio.gather(*(writer(i) for i in range(50)))


@pytest.mark.asyncio
async def test_same_key_single_flight_computes_once() -> None:
    cache = AsyncTTLCache(ttl_seconds=60)
    calls = {"n": 0}

    async def factory() -> dict:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return {"value": calls["n"]}

    results = await asyncio.gather(
        cache.get_or_compute("same-key", factory),
        cache.get_or_compute("same-key", factory),
        cache.get_or_compute("same-key", factory),
    )
    assert calls["n"] == 1
    values = [r[0] for r in results]
    assert values == [{"value": 1}, {"value": 1}, {"value": 1}]
    computed_flags = [r[1] for r in results]
    assert computed_flags.count(True) == 1
    assert computed_flags.count(False) == 2


@pytest.mark.asyncio
async def test_different_keys_compute_concurrently() -> None:
    cache = AsyncTTLCache(ttl_seconds=60)
    started = asyncio.Event()
    release = asyncio.Event()
    in_flight = {"n": 0}

    async def factory_a() -> str:
        in_flight["n"] += 1
        if in_flight["n"] >= 2:
            started.set()
        await release.wait()
        return "a"

    async def factory_b() -> str:
        in_flight["n"] += 1
        if in_flight["n"] >= 2:
            started.set()
        await release.wait()
        return "b"

    task_a = asyncio.create_task(cache.get_or_compute("key-a", factory_a))
    task_b = asyncio.create_task(cache.get_or_compute("key-b", factory_b))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    # Both factories entered before either finished → not serialized by a global lock.
    assert in_flight["n"] == 2
    release.set()
    (va, ca), (vb, cb) = await asyncio.gather(task_a, task_b)
    assert va == "a" and vb == "b"
    assert ca is True and cb is True
