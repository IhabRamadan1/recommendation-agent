"""Async-safe cache behavior."""

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
    # ttl 0 => expires_at == now; treat as expired on next get
    await asyncio.sleep(0.01)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_concurrent_sets_do_not_raise() -> None:
    cache = AsyncTTLCache(ttl_seconds=60)

    async def writer(i: int) -> None:
        await cache.set(f"k{i}", i)
        await cache.get(f"k{i}")

    await asyncio.gather(*(writer(i) for i in range(50)))
