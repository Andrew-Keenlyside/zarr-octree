"""A store wrapper that counts requests and can add latency to each one.

Used by the tests (to check how many objects a read touches) and by the
benchmarks (to emulate object storage: every request costs a round trip,
concurrent requests overlap).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import WrapperStore


@dataclass
class Counts:
    gets: int = 0
    hits: int = 0  # gets that found an object
    bytes: int = 0
    sets: int = 0
    keys: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.gets = self.hits = self.bytes = self.sets = 0
        self.keys.clear()


class CountingStore(WrapperStore[Store]):
    """Wrap ``store``; record every get and sleep ``latency`` seconds per request.

    ``max_inflight`` caps the requests in flight at once, like the connection
    pool of an object-storage client; ``None`` leaves them unbounded.
    """

    def __init__(
        self, store: Store, latency: float = 0.0, record_keys: bool = False, max_inflight: int | None = None
    ) -> None:
        super().__init__(store)
        self.latency = latency
        self.record_keys = record_keys
        self.max_inflight = max_inflight
        self._slots = asyncio.Semaphore(max_inflight) if max_inflight else None
        self.counts = Counts()

    def _with_store(self, store: Store) -> CountingStore:
        other = CountingStore(store, self.latency, self.record_keys, self.max_inflight)
        other.counts = self.counts  # views (e.g. read-only) count into the same totals
        other._slots = self._slots  # and share the same pool
        return other

    # Async only: the sync fast path would bypass the counts and the latency.
    _supports_sync_io = False

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        if self._slots is None:
            buf = await self._get(key, prototype, byte_range)
        else:
            async with self._slots:
                buf = await self._get(key, prototype, byte_range)
        self.counts.gets += 1
        if self.record_keys:
            self.counts.keys.append(key)
        if buf is not None:
            self.counts.hits += 1
            self.counts.bytes += len(buf)
        return buf

    async def _get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None) -> Buffer | None:
        if self.latency:
            await asyncio.sleep(self.latency)
        return await self._store.get(key, prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):  # type: ignore[no-untyped-def]
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def _get_many(self, requests):  # type: ignore[no-untyped-def]
        requests = list(requests)
        results = await asyncio.gather(*(self.get(*r) for r in requests))
        for req, res in zip(requests, results):
            yield req[0], res

    if hasattr(Store, "get_ranges"):  # zarr >= 3.4; the base version reads through self.get
        get_ranges = Store.get_ranges

    async def set(self, key: str, value: Buffer) -> None:
        self.counts.sets += 1
        await self._store.set(key, value)
