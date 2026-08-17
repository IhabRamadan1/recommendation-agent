"""Async-safe request cache with TTL and per-key single-flight.

Bug #3 fix: a bare shared dict is not safe under concurrent await points.
Idempotency: different keys run concurrently; same key computes at most once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class AsyncTTLCache:
    """Per-process async cache with per-key locks (not one global compute lock)."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _CacheEntry] = {}
        self._store_lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._key_locks_guard = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._store_lock:
            return self._read_unlocked(key)

    async def set(self, key: str, value: Any) -> None:
        async with self._store_lock:
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
            )

    async def _key_lock(self, key: str) -> asyncio.Lock:
        async with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._key_locks[key] = lock
            return lock

    def _read_unlocked(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.monotonic():
            del self._store[key]
            return None
        return entry.value

    async def get_or_compute(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        """Single-flight per key: check → per-key lock → re-check → compute once.

        Returns (value, computed) where computed is True iff this caller ran factory.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached, False

        lock = await self._key_lock(key)
        async with lock:
            async with self._store_lock:
                cached = self._read_unlocked(key)
                if cached is not None:
                    return cached, False

            value = await factory()

            async with self._store_lock:
                # Prefer an entry another waiter might have stored (defensive).
                existing = self._read_unlocked(key)
                if existing is not None:
                    return existing, False
                self._store[key] = _CacheEntry(
                    value=value,
                    expires_at=time.monotonic() + self._ttl,
                )
                return value, True

    async def clear(self) -> None:
        async with self._store_lock:
            self._store.clear()
