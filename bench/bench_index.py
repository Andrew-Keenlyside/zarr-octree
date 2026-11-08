"""Spatial and vertex-index query speed: octree against regular chunking, through zarr.

The stores are the ones ``bench.py`` builds (``bench/_stores``).  zarr-vectors
addresses a vertex as (chunk, row in that chunk) -- object manifests and links
hold these addresses -- so a vertex-index query is: look the addresses up, then
fetch the chunks that hold them and take the rows.  Two kinds:

* **random vertices** -- ``k`` vertex ids drawn uniformly (so, in space,
  following the density), looked up in a dense address table (one row per
  vertex id, 65,536 rows per chunk);
* **objects** -- one object's vertices, looked up in an object manifest (one
  vlen row per object, as zarr-vectors' ``object_index/manifests``).  Objects
  are ``compact`` (the ``k`` nearest neighbours of a vertex), ``cluster``
  (``k`` vertices of one Gaussian cluster) or ``curve`` (``k`` vertices of one
  curved 1-D manifold).

The address tables and manifests are plain zarr arrays in each layout's store;
in the octree store they pass through untouched.  Every fetched vertex is
checked against the source points.  Spatial queries are the occupied-cell box
queries of ``bench.py``.  With latency, every request also pays a transfer
time (``--bandwidth`` per request), so reading a large chunk costs more.

    python bench/bench_index.py --quick
    python bench/bench_index.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial import cKDTree
from zarr.storage import LocalStore

from bench import BOX, COARSE, GRID, HERE, LAYOUTS, LEAF, Reader, bin_points, obj_values, query_box, run_query

INDEX_ROWS = 65536


def addresses(pts: np.ndarray, grid: int, cell: float) -> np.ndarray:
    """``(n, 4)`` int32 address of every vertex: its chunk coordinates and its row there.

    Rows follow ``bin_points``: a chunk's vertices in input order.
    """
    c = np.clip(np.floor(pts / cell).astype(np.int64), 0, grid - 1)
    lin = (c[:, 0] * grid + c[:, 1]) * grid + c[:, 2]
    order = np.argsort(lin, kind="stable")
    _, start, inverse = np.unique(lin[order], return_index=True, return_inverse=True)
    row = np.empty(len(pts), dtype=np.int64)
    row[order] = np.arange(len(pts)) - start[inverse.reshape(-1)]
    return np.concatenate([c, row[:, None]], 1).astype(np.int32)


def make_objects(pts: np.ndarray, labels: np.ndarray, sizes: list[int], per_kind: int) -> dict[tuple[str, int], list[np.ndarray]]:
    """Vertex ids of each test object, by (kind, size)."""
    rng = np.random.default_rng(11)
    tree = cKDTree(pts)
    clusters = [np.flatnonzero(labels == k) for k in range(24)]
    curves = [np.flatnonzero(labels == k) for k in (24, 25, 26)]
    out: dict[tuple[str, int], list[np.ndarray]] = {}
    for k in sizes:
        out["compact", k] = [np.sort(tree.query(pts[rng.integers(len(pts))], k)[1]) for _ in range(per_kind)]
        out["cluster", k] = [np.sort(rng.choice(m, k, replace=False)) for m in (clusters[j] for j in rng.integers(len(clusters), size=per_kind))]
        out["curve", k] = [np.sort(rng.choice(m, k, replace=False)) for m in (curves[j] for j in rng.integers(len(curves), size=per_kind))]
    return out


def write_indexes(root: Path, layout: str, table: np.ndarray, objects: list[np.ndarray]) -> None:
    """The address table and the object manifests, as plain zarr arrays beside ``pts``."""
    store = LocalStore(str(root / layout))
    a = zarr.create_array(store, name="vertex_index", shape=table.shape, chunks=(INDEX_ROWS, 4), dtype="int32", overwrite=True)
    a[:] = table
    m = zarr.create_array(store, name="object_index", shape=(len(objects),), chunks=(1,), dtype="bytes", overwrite=True)
    m[:] = obj_values([table[ids].tobytes() for ids in objects])


def fetch(reader: Reader, addr: np.ndarray) -> np.ndarray:
    """The vertices at ``addr`` (``(n, 4)``: chunk coords, row), in the order given."""
    cells, inverse = np.unique(addr[:, :3], axis=0, return_inverse=True)
    blobs = reader.arr.vindex[tuple(cells.T)]
    chunks = [np.frombuffer(b, dtype=np.float32).reshape(-1, 3) for b in blobs]
    sizes = np.array([len(c) for c in chunks])
    flat = np.concatenate(chunks)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return flat[start[inverse.reshape(-1)] + addr[:, 3]]


def vertex_query(reader: Reader, index: zarr.Array, gids: np.ndarray | int, kind: str) -> tuple[dict, np.ndarray]:
    reader.reset()
    t0 = time.perf_counter()
    if kind == "random":
        addr = index.get_orthogonal_selection((gids, slice(None)))
    else:
        addr = np.frombuffer(index[gids : gids + 1][0], dtype=np.int32).reshape(-1, 4)
    t1 = time.perf_counter()
    lookup_requests = reader.counter.counts.gets
    out = fetch(reader, addr)
    t2 = time.perf_counter()
    c = reader.counter.counts
    return {
        "seconds": t2 - t0, "lookup_seconds": t1 - t0, "fetch_seconds": t2 - t1,
        "lookup_requests": lookup_requests, "requests": c.gets - lookup_requests, "bytes_read": c.bytes,
        "cells": int(len(np.unique(addr[:, :3], axis=0))), "points": int(len(addr)),
    }, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "_data" / "densmap_points.npz")
    ap.add_argument("--stores", type=Path, default=HERE / "_stores")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "bench_index.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4096)
    ap.add_argument("--max-inflight", type=int, default=128)
    ap.add_argument("--bandwidth", type=float, default=100e6, help="bytes/s per request when latency > 0 (0 = none)")
    args = ap.parse_args()
    zarr.config.set({"async.concurrency": args.concurrency})

    data = np.load(args.data)
    pts = np.clip(data["points"], 0, BOX - 1e-3)
    tables = {"leaf": addresses(pts, GRID, LEAF), "coarse": addresses(pts, GRID // COARSE, LEAF * COARSE)}
    # The addresses agree with how the stores were written.
    for f, key in ((1, "leaf"), (COARSE, "coarse")):
        coords, blobs = bin_points(pts, GRID // f, LEAF * f)
        i = np.random.default_rng(0).integers(len(pts), size=1000)
        t = tables[key][i]
        lookup = {tuple(c): b for c, b in zip(coords.tolist(), blobs)}
        got = np.stack([np.frombuffer(lookup[tuple(a[:3])], np.float32).reshape(-1, 3)[a[3]] for a in t.tolist()])
        assert (got == pts[i]).all()

    sizes = [1000, 10000]
    per_kind = 3 if args.quick else 8
    objects = make_objects(pts, data["labels"], sizes, per_kind)
    order = list(objects)
    flat = [ids for key in order for ids in objects[key]]
    for layout in LAYOUTS:
        write_indexes(args.stores, layout, tables["coarse" if layout == "coarse" else "leaf"], flat)

    latencies = [0.0, 0.02] if args.quick else [0.0, 0.02, 0.08]
    random_k = [1, 32, 1024] if args.quick else [1, 32, 1024, 16384]
    n_q = 3 if args.quick else 8
    results: dict = {"setup": {"points": int(len(pts)), "latencies": latencies, "max_inflight": args.max_inflight,
                               "concurrency": args.concurrency, "bandwidth": args.bandwidth, "zarr": zarr.__version__, "index_rows": INDEX_ROWS}}
    vertex, spatial = [], []
    occ = {}
    for f in (1, COARSE):
        g = np.zeros((GRID // f,) * 3, dtype=bool)
        g[tuple(bin_points(pts, GRID // f, LEAF * f)[0].T)] = True
        occ[f] = g

    def report(kind: str, size: int, layout: str, rows: list[dict], latency: float) -> None:
        print(f"lat {latency*1e3:3.0f} ms  {kind:16s} {size:6d}  {layout:7s} median {np.median([r['seconds'] for r in rows])*1e3:8.1f} ms"
              f"  cells {np.median([r.get('cells', 0) for r in rows]):7.0f}  requests {np.median([r['requests'] for r in rows]):6.0f}", flush=True)

    for latency in latencies:
        # No latency means local disk; with latency, object storage: a round trip plus a transfer time.
        bandwidth = args.bandwidth if latency and args.bandwidth else None
        readers = {layout: Reader(layout, args.stores, latency, args.max_inflight, bandwidth) for layout in LAYOUTS}
        index = {layout: zarr.open_array(r.store, path="vertex_index", mode="r") for layout, r in readers.items()}
        manifests = {layout: zarr.open_array(r.store, path="object_index", mode="r") for layout, r in readers.items()}

        # Random vertices.
        for k in random_k:
            if latency >= 0.08 and k > 1024:
                continue
            qrng = np.random.default_rng(k)
            draws = [np.sort(qrng.choice(len(pts), k, replace=False)) for _ in range(n_q)]
            for layout in LAYOUTS:
                rows = []
                for qi, gids in enumerate(draws):
                    r, got = vertex_query(readers[layout], index[layout], gids, "random")
                    assert (got == pts[gids]).all(), (layout, k)
                    r.update(layout=layout, latency=latency, kind="random", size=k, query=qi)
                    rows.append(r)
                vertex += rows
                report("random", k, layout, rows, latency)

        # Objects.
        for oi_start, key in zip(np.cumsum([0] + [len(objects[k]) for k in order[:-1]]), order):
            kind, k = key
            for layout in LAYOUTS:
                rows = []
                for qi, ids in enumerate(objects[key][:n_q]):
                    r, got = vertex_query(readers[layout], manifests[layout], int(oi_start + qi), "object")
                    assert (got == pts[ids]).all(), (layout, key)
                    r.update(layout=layout, latency=latency, kind=kind, size=k, query=qi)
                    rows.append(r)
                vertex += rows
                report(f"object {kind}", k, layout, rows, latency)

        # Boxes.
        for size in [2, 8, 32] if args.quick else [2, 8, 16, 32]:
            if latency >= 0.08 and size > 16:
                continue
            for placement in ("dense", "uniform"):
                brng = np.random.default_rng([size, placement == "dense"])
                boxes = [query_box(brng, size * LEAF, placement, pts) for _ in range(n_q)]
                for layout in LAYOUTS:
                    rows = []
                    for qi, (lo, hi) in enumerate(boxes):
                        r = run_query(readers[layout], occ, lo, hi, "occupied")
                        r.update(layout=layout, latency=latency, placement=placement, size=size, query=qi)
                        rows.append(r)
                    spatial += rows
                    report(f"box {placement}", size, layout, rows, latency)
        # Box results must agree across layouts.
        for size in {q["size"] for q in spatial if q["latency"] == latency}:
            for placement in ("dense", "uniform"):
                counts = {l: [q["points"] for q in spatial if q["latency"] == latency and q["size"] == size and q["placement"] == placement and q["layout"] == l] for l in LAYOUTS}
                assert all(v == counts["leaf"] for v in counts.values()), (size, placement)

    results["vertex"] = vertex
    results["spatial"] = spatial
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
