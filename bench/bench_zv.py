"""Prototype B in zarr-vectors: ``sharding_indexed`` against ``zarr_octree_sharding``, through zarr-vectors' API.

Run in the zarr-vectors environment, with the prototype branch first on the path:

    PYTHONPATH=~/scripts/zarr-vectors-py-octree python bench/bench_zv.py [--quick]

The same 3.99 M densMAP points (``make_points.py``) are written as a
zarr-vectors point cloud (64³ cells, zstd) whose spatial arrays are sharded
16³ or 32³ cells per shard, by either codec.  Objects: each cluster / curve /
the background split at random into objects of ~1,000 vertices (spread over
their cluster, as in ``bench_index.py``).

Reads, through ``zv.open(store)``: boxes (``select(bbox=)``) centred on a
vertex or placed uniformly, and objects (``level.objects[ids]``).  Every store
must return the same points.  Latency: the local store as it is (zarr-vectors
reads shard files directly), or the store behind ``CountingStore`` with 20 /
80 ms and 100 MB/s per request (zarr-vectors' object-store path).  The octree
store is read twice: with the pooled reader that knows its sparse index
(``octree``), and with that support switched off (``octree-zarr``: zarr-vectors
reads it the way it reads any codec it does not recognise).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from zarr.storage import LocalStore

HERE = Path(__file__).parent
BOX = 1000.0
CELLS = 64
OBJECT_SIZE = 1000
STORES = {  # name: (shard cells, codec, codec config)
    "indexed-16": (16, "sharding_indexed", {}),
    "octree-16": (16, "zarr_octree_sharding", {"min_weight": 160_000, "max_weight": 640_000}),
    "indexed-32": (32, "sharding_indexed", {}),
    "octree-32": (32, "zarr_octree_sharding", {"min_weight": 160_000, "max_weight": 640_000}),
}


def object_ids(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Each label's vertices split at random into objects of ~OBJECT_SIZE."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=np.int64)
    next_id = 0
    for lab in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == lab))
        k = max(1, round(len(idx) / OBJECT_SIZE))
        out[idx] = next_id + np.arange(len(idx)) % k
        next_id += k
    return out


def build(root: Path, name: str, pts: np.ndarray, ids: np.ndarray) -> dict:
    import zarr_vectors as zv

    shard, codec, config = STORES[name]
    path = root / f"{name}.zv"
    shutil.rmtree(path, ignore_errors=True)
    os.environ["ZARR_VECTORS_SHARD_CODEC"] = codec
    os.environ["ZARR_VECTORS_SHARD_CODEC_CONFIG"] = json.dumps(config)
    schema = zv.Schema(bounds=([0, 0, 0], [BOX, BOX, BOX]), kind="point_cloud",
                       layout=zv.Layout(cells=CELLS, shard_cells=shard, compression="zstd"))
    t0 = time.perf_counter()
    ds = zv.create(str(path), schema=schema)
    ds.add_points(pts, object_ids=ids)
    seconds = time.perf_counter() - t0
    files = [p for p in path.rglob("*") if p.is_file()]
    vert = [p for p in (path / "0" / "vertices").rglob("*") if p.is_file() and p.name != "zarr.json"]
    meta = json.loads((path / "0" / "vertices" / "zarr.json").read_text())
    assert meta["codecs"][0]["name"] == codec, meta["codecs"][0]["name"]
    return {"write_seconds": seconds, "disk_bytes": sum(p.stat().st_size for p in files),
            "vertex_bytes": sum(p.stat().st_size for p in vert), "vertex_objects": len(vert), "files": len(files)}


class Reader:
    def __init__(self, root: Path, name: str, latency: float, bandwidth: float | None, max_inflight: int):
        import zarr_vectors as zv

        from zarr_octree.instrument import CountingStore

        self.local = LocalStore(str(root / f"{name}.zv"), read_only=True)
        if latency:
            self.counter = CountingStore(self.local, latency=latency, bandwidth=bandwidth, max_inflight=max_inflight)
            self.ds = zv.open(self.counter)
        else:
            self.counter = None
            self.ds = zv.open(str(root / f"{name}.zv"))
        self.level = self.ds.level(0)

    def counts(self) -> tuple[int, int]:
        return (self.counter.counts.gets, self.counter.counts.bytes) if self.counter else (0, 0)

    def reset(self) -> None:
        if self.counter:
            self.counter.counts.reset()


def octree_pooled(on: bool) -> None:
    """Switch zarr-vectors' pooled reader support for the octree index on or off."""
    import zarr_vectors.core._cells_on_disk as cod

    if not hasattr(cod, "_cell_source_full"):
        cod._cell_source_full = cod.cell_source

    def without(name, node, *, any_codecs=False):
        src = cod._cell_source_full(name, node, any_codecs=any_codecs)
        return None if src is not None and src.index_format == "zarr_octree" else src

    cod.cell_source = cod._cell_source_full if on else without
    import zarr_vectors.core._batch_reader as br

    if hasattr(br, "cell_source"):
        br.cell_source = cod.cell_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "_data" / "densmap_points.npz")
    ap.add_argument("--stores", type=Path, default=HERE / "_stores_zv")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "bench_zv.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--points", type=int, default=0, help="use only this many points (0: all)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--max-inflight", type=int, default=128)
    ap.add_argument("--bandwidth", type=float, default=100e6)
    args = ap.parse_args()
    import zarr

    zarr.config.set({"async.concurrency": 4096})

    data = np.load(args.data)
    pts = np.clip(data["points"], 0, BOX - 1e-3).astype(np.float32)
    labels = data["labels"]
    if args.points:
        keep = np.random.default_rng(1).choice(len(pts), args.points, replace=False)
        pts, labels = pts[keep], labels[keep]
    ids = object_ids(labels)
    results: dict = {"setup": {"points": int(len(pts)), "cells": CELLS, "stores": STORES, "objects": int(ids.max() + 1),
                               "zarr": zarr.__version__, "bandwidth": args.bandwidth, "max_inflight": args.max_inflight}}
    args.stores.mkdir(parents=True, exist_ok=True)
    built_path = args.out.with_name(args.out.stem + "_build.json")
    if args.skip_build and built_path.exists():
        results["build"] = json.loads(built_path.read_text())
    else:
        results["build"] = {}
        for name in STORES:
            results["build"][name] = build(args.stores, name, pts, ids)
            print(name, results["build"][name], flush=True)
        built_path.write_text(json.dumps(results["build"], indent=1))

    n_q = 3 if args.quick else 5
    latencies = [0.0, 0.02] if args.quick else [0.0, 0.02, 0.08]
    box_sizes = [31.25, 125.0] if args.quick else [31.25, 62.5, 125.0, 250.0]
    obj_counts = [1, 10]
    rng = np.random.default_rng(5)
    boxes = {}
    for size in box_sizes:
        for placement in ("dense", "uniform"):
            b = []
            for _ in range(n_q):
                c = pts[rng.integers(len(pts))] if placement == "dense" else rng.uniform(0, BOX, 3)
                lo = np.clip(c - size / 2, 0, BOX - size)
                b.append((lo, lo + size))
            boxes[size, placement] = b
    objs = {k: [rng.choice(int(ids.max()) + 1, k, replace=False) for _ in range(n_q)] for k in obj_counts}

    variants = [("indexed-16", True), ("octree-16", True), ("octree-16", False),
                ("indexed-32", True), ("octree-32", True), ("octree-32", False)]
    queries = []
    for latency in latencies:
        bw = args.bandwidth if latency else None
        for name, pooled in variants:
            label = name if pooled or not name.startswith("octree") else name.replace("octree", "octree-zarr")
            octree_pooled(pooled)
            reader = Reader(args.stores, name, latency, bw, args.max_inflight)
            for (size, placement), bs in boxes.items():
                for qi, (lo, hi) in enumerate(bs):
                    reader.reset()
                    t0 = time.perf_counter()
                    r = reader.ds.select(bbox=(lo.tolist(), hi.tolist())).read()
                    dt = time.perf_counter() - t0
                    gets, nbytes = reader.counts()
                    p = np.asarray(r.positions)
                    queries.append({"layout": label, "latency": latency, "kind": f"box-{placement}", "size": size, "query": qi,
                                    "seconds": dt, "requests": gets, "bytes_read": nbytes, "points": int(len(p)),
                                    "checksum": float(np.sort(p, axis=0).astype(np.float64).sum()) if len(p) else 0.0})
            for k, sets in objs.items():
                for qi, oid in enumerate(sets):
                    reader.reset()
                    t0 = time.perf_counter()
                    r = reader.level.objects[[int(x) for x in oid]]
                    dt = time.perf_counter() - t0
                    gets, nbytes = reader.counts()
                    p = np.asarray(r.positions)
                    queries.append({"layout": label, "latency": latency, "kind": "objects", "size": k, "query": qi,
                                    "seconds": dt, "requests": gets, "bytes_read": nbytes, "points": int(len(p)),
                                    "checksum": float(np.sort(p, axis=0).astype(np.float64).sum()) if len(p) else 0.0})
            mine = [q for q in queries if q["layout"] == label and q["latency"] == latency]
            print(f"lat {latency*1e3:3.0f}  {label:16s} median {np.median([q['seconds'] for q in mine])*1e3:8.1f} ms  "
                  f"requests {np.median([q['requests'] for q in mine]):6.0f}", flush=True)
    octree_pooled(True)
    # Every layout must return the same points for the same query.
    for latency in latencies:
        groups: dict = {}
        for q in queries:
            if q["latency"] == latency:
                groups.setdefault((q["kind"], q["size"], q["query"]), []).append((q["layout"], q["points"], q["checksum"]))
        for key, rows in groups.items():
            ref = rows[0]
            for row in rows[1:]:
                assert row[1] == ref[1] and abs(row[2] - ref[2]) <= 1e-6 * max(1.0, abs(ref[2])), (latency, key, ref, row)
    results["queries"] = queries
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
