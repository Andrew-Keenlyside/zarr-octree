"""Chunk layouts compared: uniform grids of every size against octree runs bounded in vertices.

The same 3.99 M points (``make_points.py``), stored as vlen-bytes chunks of
float32 xyz (zstd), the way zarr-vectors stores vertices:

* ``u<G>``        -- a regular ``G``³ chunk grid (G = 128 … 8: chunks 7.8 … 125 units wide)
* ``o<G>-<lo>-<hi>`` -- a ``G``³ leaf grid packed into runs of the octree's Z-order
  holding ``lo`` to ``hi`` vertices (``OctreeConfig(unit="vertices")``), one
  root for the whole grid

Each layout is read two ways:

* ``zarr``   -- through zarr's own API (``arr.vindex``), as zarr-vectors reads today;
* ``direct`` -- the chunks (uniform) or nodes (octree, ``OctreeStore.get_cells``)
  fetched concurrently and decoded here, without zarr's per-chunk machinery:
  what each layout costs at the storage level.

Queries (vertex addresses are looked up in memory -- the lookup is the same
for every layout -- and only the vertex reads are timed): random vertices,
whole objects (``compact`` / ``cluster`` / ``curve``), and boxes centred on a
vertex or placed uniformly.  Every result is checked against the source.

    python bench/bench_bounds.py --quick
    python bench/bench_bounds.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import time
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import LocalStore

from bench import BOX, HERE, bin_points, obj_values, query_box
from bench_index import addresses, make_objects
from zarr_octree import OctreeConfig, OctreeStore, convert
from zarr_octree.instrument import CountingStore

UNIFORM = [128, 64, 32, 16, 8]
OCTREE = [  # (leaf grid, min vertices, max vertices)
    (128, 1_000, 4_000), (128, 4_000, 16_000), (128, 16_000, 64_000), (128, 64_000, 256_000), (128, 0, 16_000),
    (64, 4_000, 16_000), (64, 16_000, 64_000), (64, 64_000, 256_000),
    (32, 4_000, 16_000), (32, 16_000, 64_000), (32, 64_000, 256_000),
    (16, 16_000, 64_000),
]
ZSTD = numcodecs.Zstd()


def vertices(chunk: bytes) -> float:
    """Vertices in an encoded chunk: zstd over a one-item vlen-bytes block of float32 xyz."""
    return (len(ZSTD.decode(chunk)) - 8) / 12


def decode(chunk: bytes) -> np.ndarray:
    return np.frombuffer(ZSTD.decode(chunk)[8:], dtype=np.float32).reshape(-1, 3)


def name_of(spec) -> str:
    if isinstance(spec, int):
        return f"u{spec}"
    g, lo, hi = spec
    k = lambda v: f"{v // 1000}k" if v else "0"
    return f"o{g}-{k(lo)}-{k(hi)}"


def build(root: Path, pts: np.ndarray, rebuild: bool) -> dict[str, float]:
    seconds = {}
    for g in UNIFORM:
        path = root / name_of(g)
        if path.exists() and not rebuild:
            continue
        shutil.rmtree(path, ignore_errors=True)
        coords, blobs = bin_points(pts, g, BOX / g)
        t0 = time.perf_counter()
        arr = zarr.create_array(LocalStore(str(path)), name="pts", shape=(g,) * 3, chunks=(1, 1, 1), dtype="bytes")
        arr.vindex[tuple(coords.T)] = obj_values(blobs)
        seconds[name_of(g)] = time.perf_counter() - t0
        print(f"built {name_of(g)}: {seconds[name_of(g)]:.1f} s", flush=True)
    for spec in OCTREE:
        g, lo, hi = spec
        path = root / name_of(spec)
        if path.exists() and not rebuild:
            continue
        shutil.rmtree(path, ignore_errors=True)
        config = OctreeConfig(max_depth=int(math.log2(g)), min_weight=lo, max_weight=hi, unit="vertices", weigh=vertices)
        t0 = time.perf_counter()
        convert(LocalStore(str(root / name_of(g)), read_only=True), "pts", LocalStore(str(path)), config)
        seconds[name_of(spec)] = time.perf_counter() - t0
        print(f"built {name_of(spec)}: {seconds[name_of(spec)]:.1f} s", flush=True)
    return seconds


def object_sizes(root: Path, name: str, spec, pts: np.ndarray) -> np.ndarray:
    """Vertices in every stored object."""
    if isinstance(spec, int):
        return np.array([len(b) / 12 for b in bin_points(pts, spec, BOX / spec)[1]])
    return np.array([i.weight for i in OctreeStore(LocalStore(str(root / name), read_only=True)).nodes("pts").values()])


class Layout:
    """One layout opened for queries over a latency-injecting counting store."""

    def __init__(self, root: Path, spec, latency: float, max_inflight: int, bandwidth: float | None):
        self.name = name_of(spec)
        self.grid = spec if isinstance(spec, int) else spec[0]
        self.octree = not isinstance(spec, int)
        self.counter = CountingStore(LocalStore(str(root / self.name), read_only=True), latency=latency,
                                     max_inflight=max_inflight, bandwidth=bandwidth)
        self.store = OctreeStore(self.counter) if self.octree else self.counter
        self.arr = zarr.open_array(self.store, path="pts", mode="r")
        self.proto = default_buffer_prototype()

    def reset(self) -> None:
        if self.octree:
            self.store._cache.clear()
        self.counter.counts.reset()

    def chunks(self, cells: np.ndarray, reader: str) -> list[np.ndarray]:
        """The vertices of each cell in ``cells`` (``(n, 3)`` chunk coordinates)."""
        if not len(cells):
            return []
        if reader == "zarr":
            return [np.frombuffer(b, dtype=np.float32).reshape(-1, 3) for b in self.arr.vindex[tuple(cells.T)]]
        if self.octree:
            raw = self.store.get_cells("pts", cells.tolist())
        else:
            keys = ["pts/c/" + "/".join(map(str, c)) for c in cells.tolist()]

            async def fetch():
                bufs = await asyncio.gather(*(self.counter.get(k, self.proto) for k in keys))
                return [None if b is None else b.to_bytes() for b in bufs]

            raw = sync(fetch())
        return [np.zeros((0, 3), np.float32) if r is None else decode(r) for r in raw]


def vertex_query(layout: Layout, addr: np.ndarray, reader: str) -> tuple[dict, np.ndarray]:
    layout.reset()
    t0 = time.perf_counter()
    cells, inverse = np.unique(addr[:, :3], axis=0, return_inverse=True)
    chunks = layout.chunks(cells, reader)
    sizes = np.array([len(c) for c in chunks])
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = np.concatenate(chunks)[start[inverse.reshape(-1)] + addr[:, 3]]
    dt = time.perf_counter() - t0
    c = layout.counter.counts
    return {"seconds": dt, "requests": c.gets, "bytes_read": c.bytes, "cells": int(len(cells)), "points": int(len(addr))}, out


def box_query(layout: Layout, occ: np.ndarray, lo: np.ndarray, hi: np.ndarray, reader: str) -> dict:
    cell = BOX / layout.grid
    a = np.floor(lo / cell).astype(int)
    b = np.minimum(np.ceil(hi / cell).astype(int), layout.grid)
    layout.reset()
    t0 = time.perf_counter()
    idx = np.argwhere(occ[a[0] : b[0], a[1] : b[1], a[2] : b[2]]) + a
    chunks = layout.chunks(idx, reader)
    p = np.concatenate(chunks) if chunks else np.zeros((0, 3), np.float32)
    inside = int(((p >= lo) & (p < hi)).all(1).sum())
    dt = time.perf_counter() - t0
    c = layout.counter.counts
    return {"seconds": dt, "requests": c.gets, "bytes_read": c.bytes, "cells": int(len(idx)), "points": inside}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "_data" / "densmap_points.npz")
    ap.add_argument("--stores", type=Path, default=HERE / "_stores_bounds")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "bench_bounds.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4096)
    ap.add_argument("--max-inflight", type=int, default=128)
    ap.add_argument("--bandwidth", type=float, default=100e6, help="bytes/s per request when latency > 0")
    args = ap.parse_args()
    zarr.config.set({"async.concurrency": args.concurrency})

    data = np.load(args.data)
    pts = np.clip(data["points"], 0, BOX - 1e-3)
    specs = UNIFORM + OCTREE
    results: dict = {"setup": {"points": int(len(pts)), "uniform": UNIFORM, "octree": OCTREE, "max_inflight": args.max_inflight,
                               "bandwidth": args.bandwidth, "concurrency": args.concurrency, "zarr": zarr.__version__}}
    results["build_seconds"] = build(args.stores, pts, args.rebuild)
    results["objects"] = {}
    for spec in specs:
        name = name_of(spec)
        v = object_sizes(args.stores, name, spec, pts)
        assert v.sum() == len(pts), name
        disk = sum(p.stat().st_size for p in (args.stores / name / "pts").rglob("*") if p.is_file() and p.name != "zarr.json")
        results["objects"][name] = {"objects": int(len(v)), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                                    "max": float(v.max()), "min": float(v.min()), "cv": float(v.std() / v.mean()), "disk_bytes": int(disk)}
        print(name, results["objects"][name], flush=True)

    grids = sorted({s if isinstance(s, int) else s[0] for s in specs})
    table = {g: addresses(pts, g, BOX / g) for g in grids}
    occ = {}
    for g in grids:
        o = np.zeros((g,) * 3, dtype=bool)
        o[tuple(bin_points(pts, g, BOX / g)[0].T)] = True
        occ[g] = o
    n_q = 3 if args.quick else 5
    objects = make_objects(pts, data["labels"], [1000, 10000], n_q)
    random_k = [1, 32, 1024] if args.quick else [1, 32, 1024, 16384]
    box_sizes = [15.625, 62.5, 250.0] if args.quick else [15.625, 62.5, 125.0, 250.0]
    latencies = [0.0, 0.02] if args.quick else [0.0, 0.02, 0.08]

    queries = []
    for latency in latencies:
        bw = args.bandwidth if latency and args.bandwidth else None
        layouts = {name_of(s): Layout(args.stores, s, latency, args.max_inflight, bw) for s in specs}
        families = []
        for k in random_k:
            r = np.random.default_rng(k)
            families.append(("random", k, [np.sort(r.choice(len(pts), k, replace=False)) for _ in range(n_q)]))
        for (kind, k), ids in objects.items():
            families.append((kind, k, ids[:n_q]))
        # Through zarr at 80 ms adds nothing the 0 and 20 ms runs do not show, at the most cost.
        for reader in ("direct",) if latency >= 0.08 else ("direct", "zarr"):
            for kind, k, sets in families:
                for name, lay in layouts.items():
                    rows = []
                    for qi, ids in enumerate(sets):
                        r, got = vertex_query(lay, table[lay.grid][ids], reader)
                        assert (got == pts[ids]).all(), (name, kind, k)
                        r.update(layout=name, reader=reader, latency=latency, kind=kind, size=k, query=qi)
                        rows.append(r)
                    queries += rows
                    print(f"lat {latency*1e3:3.0f} {reader:6s} {kind:8s} {k:6d} {name:16s} {np.median([q['seconds'] for q in rows])*1e3:8.1f} ms"
                          f"  req {np.median([q['requests'] for q in rows]):6.0f}  MB {np.median([q['bytes_read'] for q in rows])/1e6:6.2f}", flush=True)
            for size in box_sizes:
                for placement in ("dense", "uniform"):
                    brng = np.random.default_rng([int(size * 8), placement == "dense"])
                    boxes = [query_box(brng, size, placement, pts) for _ in range(n_q)]
                    found = None
                    for name, lay in layouts.items():
                        rows = []
                        for qi, (lo, hi) in enumerate(boxes):
                            r = box_query(lay, occ[lay.grid], lo, hi, reader)
                            r.update(layout=name, reader=reader, latency=latency, kind=f"box-{placement}", size=size, query=qi)
                            rows.append(r)
                        got = [q["points"] for q in rows]
                        assert found is None or got == found, (name, size, placement)
                        found = got
                        queries += rows
                        print(f"lat {latency*1e3:3.0f} {reader:6s} box-{placement[:4]} {size:6.1f} {name:16s} "
                              f"{np.median([q['seconds'] for q in rows])*1e3:8.1f} ms  req {np.median([q['requests'] for q in rows]):6.0f}", flush=True)
    results["queries"] = queries
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
