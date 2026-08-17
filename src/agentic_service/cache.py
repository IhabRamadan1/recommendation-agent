"""Async-safe request cache with TTL and idempotency-key support.

Bug #3 fix: a bare shared dict is not safe under concurrent await points.
All get/set operations take an asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class AsyncTTLCache:
    """Per-process async cache. Suitable for idempotency within one service instance."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._store[key]
                return None
            return entry.value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
            )

    async def get_or_set(self, key: str, factory) -> Any:
        """Return cached value or compute under lock (single-flight per key).

        Note: factory is awaited outside the lock after a double-check so we do not
        hold the lock during slow I/O — then we re-acquire to store. A second
        concurrent caller may still compute once; idempotent factories are required.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached

        value = await factory()
        # If another coroutine stored first, prefer the existing cached response
        # so retries stay idempotent.
        async with self._lock:
            entry = self._store.get(key)
            if entry is not None and entry.expires_at >= time.monotonic():
                return entry.value
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
            )
            return value

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
