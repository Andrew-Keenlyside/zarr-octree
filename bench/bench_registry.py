"""``octree_sharding`` against ``sharding_indexed``, including ``sharding_indexed`` with its chunks in Z-order.

The comparison for the zarr-extensions proposal: ``sharding_indexed`` already
lets a writer place inner chunks in any order, so a writer could store the
octree codec's Z-order layout today.  What ``octree_sharding`` adds is its sparse index
and its run table; this measures what each is worth.

Pure zarr (no zarr-vectors).  Data: a clustered point cloud binned into a
128³ grid of cells, each cell's float32 xyz as one ``bytes`` inner chunk
(VLenBytes + zstd), sharded 16³ cells per shard (4,096 inner chunks a shard).
Three stores of the same chunks:

* ``indexed``: zarr's ``sharding_indexed`` as zarr writes it (chunks in C order);
* ``indexed-zorder``: the same shards rewritten with the chunks in Z-order (the
  dense index updated) -- still plain ``sharding_indexed``;
* ``octree``: the ``octree_sharding`` codec, default layout (octree leaves aimed at 100 kB).

Measured: store and index size; for box queries through zarr's API, requests,
bytes and time behind a store with 20 ms latency and 100 MB/s per request;
and, from the indexes alone, how many byte ranges a reader that merges
adjacent chunks needs (the reader-side saving available to either format).

    python bench/bench_registry.py [--quick]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import google_crc32c
import numpy as np
import zarr
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, VLenBytesCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from zarr_octree.codec import OctreeShardingCodec, _bits, index_from_shard
from zarr_octree.instrument import CountingStore
from zarr_octree.tree import morton

HERE = Path(__file__).parent
GRID, SHARD = 128, 16
CPS = (SHARD,) * 3
NONE = np.iinfo(np.uint64).max
PROTO = default_buffer_prototype()


def make_cells(n: int, seed: int = 0) -> dict[tuple[int, int, int], bytes]:
    """Clustered points (broad-tailed clusters plus a sparse background), binned into cells."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.2, 0.8, (6, 3)) * GRID
    spread = rng.uniform(3, 12, 6)
    k = rng.integers(0, 6, n)
    pts = centres[k] + rng.standard_t(3, (n, 3)) * spread[k, None]
    pts = np.vstack([pts, rng.uniform(0, GRID, (n // 50, 3))])
    pts = pts[((pts >= 0) & (pts < GRID)).all(1)].astype(np.float32)
    cell = np.floor(pts).astype(np.int64)
    key = np.ravel_multi_index(cell.T, (GRID,) * 3)
    order = np.argsort(key, kind="stable")
    key, pts = key[order], pts[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    stops = np.r_[starts[1:], len(key)]
    coords = np.array(np.unravel_index(key[starts], (GRID,) * 3)).T
    return {tuple(int(x) for x in c): pts[a:b].tobytes() for c, a, b in zip(coords, starts, stops)}


def write(store, serializer, cells) -> zarr.Array:
    arr = zarr.create_array(store, name="a", shape=(GRID,) * 3, chunks=CPS, serializer=serializer, compressors=None, dtype="bytes")
    idx = np.array(list(cells)).T
    vals = np.empty(len(cells), dtype=object)
    vals[:] = list(cells.values())
    arr.vindex[tuple(idx)] = vals
    return arr


async def _items(store):
    return {k: (await store.get(k, PROTO)).to_bytes() async for k in store.list_prefix("a/c/")}


def shards(store) -> dict[str, bytes]:
    return asyncio.run(_items(store))


def dense_index(blob: bytes) -> np.ndarray:
    """``sharding_indexed`` index at the end of ``blob``: ``(4096, 2)`` offset, length (C order)."""
    n = int(np.prod(CPS))
    return np.frombuffer(blob[-(16 * n + 4) : -4], "<u8").reshape(n, 2)


def to_zorder(blob: bytes) -> bytes:
    """The same ``sharding_indexed`` shard with its chunks stored in Z-order."""
    rows = dense_index(blob)
    present = np.flatnonzero(rows[:, 0] != NONE)
    coords = np.array(np.unravel_index(present, CPS)).T
    order = present[np.argsort(morton(coords, _bits(CPS)))]
    out, new = bytearray(), np.full(rows.shape, NONE, np.uint64)
    for i in order:
        o, n = int(rows[i, 0]), int(rows[i, 1])
        new[i] = (len(out), n)
        out += blob[o : o + n]
    body = new.astype("<u8").tobytes()
    return bytes(out) + body + google_crc32c.value(body).to_bytes(4, "little")


def stored_positions(name: str, blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per stored chunk, in storage order: C-order position, offset, length, run (-1 if none)."""
    if name == "octree":
        ix = index_from_shard(blob, CPS)
        pos = np.ravel_multi_index(ix.coords.T, CPS)
        return pos, ix.offsets.astype(np.int64), ix.lengths.astype(np.int64), ix.run_of()
    rows = dense_index(blob)
    present = np.flatnonzero(rows[:, 0] != NONE)
    order = np.argsort(rows[present, 0])
    p = present[order]
    return p, rows[p, 0].astype(np.int64), rows[p, 1].astype(np.int64), np.full(len(p), -1)


def merged_ranges(stored, wanted: set[int]) -> tuple[int, int]:
    """Byte ranges and bytes for a reader that fetches wanted chunks, merging adjacent ones (no gap)."""
    pos, _off, ln, _ = stored
    hit = np.isin(pos, list(wanted))
    if not hit.any():
        return 0, 0
    starts = np.flatnonzero(hit & ~np.r_[False, hit[:-1]])
    return len(starts), int(ln[hit].sum())


def queries(cells, sizes, per_size, seed=1):
    rng = np.random.default_rng(seed)
    keys = np.array(list(cells))
    weights = np.array([len(v) for v in cells.values()], float)
    out = []
    for s in sizes:
        for c in keys[rng.choice(len(keys), per_size, p=weights / weights.sum())]:
            lo = np.clip(c - s // 2, 0, GRID - s)
            out.append((s, tuple(slice(int(a), int(a) + s) for a in lo)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 300_000 if args.quick else 2_000_000
    sizes, per_size = ((2, 8, 32), 3) if args.quick else ((2, 8, 32), 10)

    t0 = time.time()
    cells = make_cells(n)
    print(f"{n:,} points -> {len(cells):,} occupied cells of {GRID**3:,} ({time.time() - t0:.0f} s)", flush=True)

    inner = [VLenBytesCodec(), ZstdCodec()]
    stores = {"indexed": MemoryStore(), "octree": MemoryStore()}
    write(stores["indexed"], ShardingCodec(chunk_shape=(1, 1, 1), codecs=inner, index_codecs=[BytesCodec(), Crc32cCodec()]), cells)
    write(stores["octree"], OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=inner), cells)
    zstore = MemoryStore()
    asyncio.run(zstore.set("a/zarr.json", asyncio.run(stores["indexed"].get("a/zarr.json", PROTO))))
    for k, v in shards(stores["indexed"]).items():
        asyncio.run(zstore.set(k, PROTO.buffer.from_bytes(to_zorder(v))))
    stores = {"indexed": stores["indexed"], "indexed-zorder": zstore, "octree": stores["octree"]}
    print(f"written ({time.time() - t0:.0f} s)", flush=True)

    blobs = {name: shards(s) for name, s in stores.items()}
    data_bytes = sum(len(v) for v in cells.values())
    result = {"zarr": zarr.__version__, "points": n, "cells": len(cells), "grid": GRID, "shard_cells": SHARD, "stores": {}, "queries": {}}
    for name, b in blobs.items():
        total = sum(len(x) for x in b.values())
        idx = sum(len(x) - int(stored_positions(name, x)[2].sum()) for x in b.values())
        result["stores"][name] = {"shards": len(b), "bytes": total, "index_bytes": idx}
        print(f"{name:15s} {len(b)} shards, {total / 1e6:8.2f} MB, index {idx / 1e6:6.2f} MB "
              f"({idx / len(b) / 1024:5.1f} KiB a shard)", flush=True)
    print(f"(raw float32 xyz: {data_bytes / 1e6:.1f} MB)")

    stored = {name: {k: stored_positions(name, x) for k, x in b.items()} for name, b in blobs.items()}
    reference = None
    qs = queries(cells, sizes, per_size)
    for name, store in stores.items():
        counter = CountingStore(store, latency=0.02, bandwidth=100e6)
        arr = zarr.open_array(counter, path="a", mode="r")
        rows = []
        for size, sel in qs:
            counter.counts.reset()
            t = time.perf_counter()
            got = arr[sel]
            dt = time.perf_counter() - t
            # Reader-side: ranges a reader merging adjacent chunks would fetch (plus one index read a shard).
            touched: dict[str, set[int]] = {}
            for c in np.array(np.meshgrid(*[np.arange(s.start, s.stop) for s in sel], indexing="ij")).reshape(3, -1).T:
                key = "a/c/" + "/".join(str(int(x) // SHARD) for x in c)
                touched.setdefault(key, set()).add(int(np.ravel_multi_index(tuple(int(x) % SHARD for x in c), CPS)))
            ranges = sum(merged_ranges(stored[name][k], w)[0] for k, w in touched.items() if k in stored[name])
            rows.append({"size": size, "gets": counter.counts.gets, "bytes": counter.counts.bytes, "seconds": dt,
                         "merged_ranges": ranges, "shards": sum(k in stored[name] for k in touched)})
            if reference is None:
                reference = []
            if name == "indexed":
                reference.append(got)
            else:
                assert all(a == b for a, b in zip(got.ravel(), reference[len(rows) - 1].ravel())), (name, sel)
        result["queries"][name] = rows

    print(f"\nBox queries through zarr's API (20 ms latency, 100 MB/s a request); median per query, "
          f"{per_size} boxes a size")
    print(f"{'store':15s} {'box':>4s} {'requests':>9s} {'MB read':>8s} {'ms':>6s} {'merged ranges':>14s}")
    for name, rows in result["queries"].items():
        for s in sizes:
            r = [x for x in rows if x["size"] == s]
            med = {k: float(np.median([x[k] for x in r])) for k in ("gets", "bytes", "seconds", "merged_ranges")}
            print(f"{name:15s} {s:>3d}³ {med['gets']:9.0f} {med['bytes'] / 1e6:8.2f} {med['seconds'] * 1e3:6.0f} "
                  f"{med['merged_ranges']:14.0f}")
    out = HERE / "results" / ("bench_registry_quick.json" if args.quick else "bench_registry.json")
    out.write_text(json.dumps(result, indent=1))
    print(f"\n{out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
