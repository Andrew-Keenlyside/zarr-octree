"""Benchmark octree packing against regular chunk layouts, through zarr's own API.

One point cloud (``make_points.py``), binned into a regular grid of leaf
cells; each cell is one chunk of a vlen-bytes array holding its points as
float32 xyz -- the way zarr-vectors stores vertices.  Four layouts of the same
points:

* ``coarse``  -- regular chunks ``COARSE`` leaves wide (one object per coarse cell)
* ``leaf``    -- regular chunks one leaf wide (one object per leaf cell)
* ``sharded`` -- leaf chunks inside zarr's ``sharding_indexed``, ``COARSE`` leaves per shard
* ``octree``  -- leaf chunks packed by ``OctreeStore`` (roots ``2 ** DEPTH`` leaves wide)

Every store sits on a ``CountingStore`` that counts requests and adds a fixed
latency to each one (concurrent requests overlap), emulating object storage.

    python bench/bench.py            # everything
    python bench/bench.py --quick    # a few queries, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

from zarr_octree import OctreeConfig, OctreeStore
from zarr_octree.instrument import CountingStore
from zarr_octree.tree import decode_node

warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")

HERE = Path(__file__).parent
BOX = 1000.0
GRID = 128  # leaf cells per axis
LEAF = BOX / GRID
COARSE = 16  # leaves per coarse chunk / shard / octree root, per axis
DEPTH = 5  # octree root: 2**DEPTH = 32 leaves per axis
TARGET = 128 << 10
LAYOUTS = ["coarse", "leaf", "sharded", "octree"]
POINT_BYTES = 12


def bin_points(pts: np.ndarray, grid: int, cell: float) -> tuple[np.ndarray, list[bytes]]:
    """Occupied cells (``(n, 3)``) and each cell's points as bytes."""
    c = np.clip(np.floor(pts / cell).astype(np.int64), 0, grid - 1)
    lin = (c[:, 0] * grid + c[:, 1]) * grid + c[:, 2]
    order = np.argsort(lin, kind="stable")
    lin, pts = lin[order], pts[order]
    uniq, start = np.unique(lin, return_index=True)
    stop = np.append(start[1:], len(lin))
    coords = np.stack(np.unravel_index(uniq, (grid,) * 3), 1)
    return coords, [pts[a:b].tobytes() for a, b in zip(start, stop)]


def obj_values(blobs: list[bytes]) -> np.ndarray:
    v = np.empty(len(blobs), dtype=object)
    v[:] = blobs
    return v


def build(layout: str, root: Path, leaf: tuple, coarse: tuple, config: OctreeConfig) -> float:
    """Write one layout; return the wall time of the write (and pack)."""
    path = root / layout
    shutil.rmtree(path, ignore_errors=True)
    inner = LocalStore(str(path))
    t0 = time.perf_counter()
    if layout == "coarse":
        coords, blobs = coarse
        arr = zarr.create_array(inner, name="pts", shape=(GRID // COARSE,) * 3, chunks=(1, 1, 1), dtype="bytes")
        arr.vindex[tuple(coords.T)] = obj_values(blobs)
    elif layout == "leaf":
        coords, blobs = leaf
        arr = zarr.create_array(inner, name="pts", shape=(GRID,) * 3, chunks=(1, 1, 1), dtype="bytes")
        arr.vindex[tuple(coords.T)] = obj_values(blobs)
    elif layout == "sharded":
        coords, blobs = leaf
        arr = zarr.create_array(inner, name="pts", shape=(GRID,) * 3, chunks=(1, 1, 1), shards=(COARSE,) * 3, dtype="bytes")
        # One write per occupied shard: a whole-array write would encode every empty inner chunk too.
        shard = coords // COARSE
        for sh in np.unique(shard, axis=0):
            m = (shard == sh).all(1)
            block = np.full((COARSE,) * 3, b"", dtype=object)
            block[tuple((coords[m] - sh * COARSE).T)] = obj_values([b for b, k in zip(blobs, m) if k])
            arr[tuple(slice(c * COARSE, (c + 1) * COARSE) for c in sh)] = block
    elif layout == "octree":
        coords, blobs = leaf
        store = OctreeStore(inner, arrays={"pts": config}, max_buffer_bytes=1 << 34)
        arr = zarr.create_array(store, name="pts", shape=(GRID,) * 3, chunks=(1, 1, 1), dtype="bytes")
        arr.vindex[tuple(coords.T)] = obj_values(blobs)
        store.flush()
    return time.perf_counter() - t0


def disk_bytes(layout: str, root: Path) -> int:
    """Bytes of every stored data object of a layout (shard indexes and the manifest included)."""
    base = root / layout / "pts"
    return int(sum(p.stat().st_size for p in base.rglob("*") if p.is_file() and p.name != "zarr.json"))


def object_points(layout: str, root: Path, leaf: tuple, coarse: tuple) -> np.ndarray:
    """Points in every stored object of a layout, counted from the data.

    For the octree, each stored node is read back and its points are summed
    over the cells it actually holds.
    """
    if layout in ("coarse", "sharded"):  # a shard holds exactly the leaves of one coarse cell
        return np.array([len(b) for b in coarse[1]]) / POINT_BYTES
    npts = {tuple(int(x) for x in c): len(b) / POINT_BYTES for c, b in zip(*leaf)}
    if layout == "leaf":
        return np.array(list(npts.values()))
    out = []
    for f in (root / "octree" / "pts" / "o").rglob("*"):
        if f.is_file() and f.name != "manifest":
            node = decode_node(f.read_bytes())
            out.append(sum(npts[tuple(int(x) for x in c)] for c in node.absolute()))
    assert sum(out) == sum(npts.values())
    return np.array(out)


def batch_efficiency(points: np.ndarray, batch: int = 64, seed: int = 0) -> float:
    """Useful fraction of a GPU batch that pads every object to the largest in its batch."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(points)
    p = p[: len(p) // batch * batch].reshape(-1, batch) if len(p) >= batch else p[None]
    return float(p.sum() / (p.max(1, keepdims=True) * p.shape[1]).sum())


def object_stats(pts: np.ndarray, disk: int) -> dict:
    return {
        "objects": int(len(pts)),
        "disk_bytes": disk,
        "disk_over_payload": float(disk / (pts.sum() * POINT_BYTES)),
        "points_median": float(np.median(pts)),
        "points_p99": float(np.percentile(pts, 99)),
        "points_max": float(pts.max()),
        "cv": float(pts.std() / pts.mean()),
        "max_over_median": float(pts.max() / max(np.median(pts), 1)),
        "gpu_batch_efficiency": batch_efficiency(pts),
    }


class Reader:
    """One layout opened for queries, over a latency-injecting counting store."""

    def __init__(self, layout: str, root: Path, latency: float, max_inflight: int | None, bandwidth: float | None = None):
        self.layout = layout
        self.factor = COARSE if layout == "coarse" else 1
        self.counter = CountingStore(LocalStore(str(root / layout), read_only=True), latency=latency, max_inflight=max_inflight, bandwidth=bandwidth)
        self.store = OctreeStore(self.counter) if layout == "octree" else self.counter
        self.arr = zarr.open_array(self.store, path="pts", mode="r")

    def reset(self) -> None:
        """Forget node data between queries; metadata and the manifest stay loaded."""
        if isinstance(self.store, OctreeStore):
            self.store._cache.clear()
        self.counter.counts.reset()


def query_box(rng: np.random.Generator, size: float, placement: str, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = pts[rng.integers(len(pts))] if placement == "dense" else rng.uniform(0, BOX, 3)
    lo = np.clip(centre - size / 2, 0, BOX - size)
    return lo, lo + size


def run_query(reader: Reader, occ: dict[int, np.ndarray], lo: np.ndarray, hi: np.ndarray, mode: str) -> dict:
    cell = LEAF * reader.factor
    a = np.floor(lo / cell).astype(int)
    b = np.minimum(np.ceil(hi / cell).astype(int), GRID // reader.factor)
    reader.reset()
    t0 = time.perf_counter()
    if mode == "occupied":
        # Only the occupied cells, as zarr-vectors does with its ``nonempty_chunks`` list.
        sub = occ[reader.factor][a[0] : b[0], a[1] : b[1], a[2] : b[2]]
        idx = np.argwhere(sub) + a
        blobs = reader.arr.vindex[tuple(idx.T)] if len(idx) else []
    else:
        # Plain zarr slicing: zarr asks for every chunk in the box, occupied or not.
        blobs = reader.arr[a[0] : b[0], a[1] : b[1], a[2] : b[2]].ravel()
    p = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(-1, 3)
    inside = int(((p >= lo) & (p < hi)).all(1).sum())
    dt = time.perf_counter() - t0
    c = reader.counter.counts
    return {"seconds": dt, "requests": c.gets, "hits": c.hits, "bytes_read": c.bytes, "points": inside}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "_data" / "densmap_points.npz")
    ap.add_argument("--stores", type=Path, default=HERE / "_stores")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "bench.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4096, help="zarr's async.concurrency: chunk requests zarr issues at once")
    ap.add_argument("--max-inflight", type=int, default=128, help="storage requests in flight at once (0 = unbounded)")
    args = ap.parse_args()
    args.max_inflight = args.max_inflight or None
    zarr.config.set({"async.concurrency": args.concurrency})
    rng = np.random.default_rng(0)

    pts = np.load(args.data)["points"]
    pts = np.clip(pts, 0, BOX - 1e-3)
    leaf = bin_points(pts, GRID, LEAF)
    coarse = bin_points(pts, GRID // COARSE, LEAF * COARSE)
    config = OctreeConfig(max_depth=DEPTH, max_weight=TARGET)
    results: dict = {
        "setup": {
            "points": int(len(pts)), "grid": GRID, "leaf": LEAF, "coarse": COARSE, "max_depth": DEPTH,
            "target_bytes": TARGET, "concurrency": args.concurrency, "max_inflight": args.max_inflight, "zarr": zarr.__version__,
            "occupied_leaves": int(len(leaf[0])), "occupied_coarse": int(len(coarse[0])),
        }
    }
    print(json.dumps(results["setup"]), flush=True)

    # 1. Build every layout; the size of every stored object.
    previous = args.out.parent / "bench.json"
    results["build_seconds"] = json.loads(previous.read_text())["build_seconds"] if args.skip_build and previous.exists() else {}
    for layout in LAYOUTS:
        if not args.skip_build:
            results["build_seconds"][layout] = build(layout, args.stores, leaf, coarse, config)
            print(f"built {layout}: {results['build_seconds'][layout]:.1f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results["objects"] = {}
    for layout in LAYOUTS:
        pts_l = object_points(layout, args.stores, leaf, coarse)
        np.save(args.out.parent / f"objects_{layout}.npy", pts_l)
        results["objects"][layout] = object_stats(pts_l, disk_bytes(layout, args.stores))
    for layout, s in results["objects"].items():
        print(layout, {k: round(v, 3) for k, v in s.items()}, flush=True)
    nodes = OctreeStore(LocalStore(str(args.stores / "octree"))).nodes("pts")
    results["octree_depths"] = {str(d): int(sum(1 for k in nodes if k[0] == d)) for d in range(DEPTH + 1)}
    manifest = args.stores / "octree" / "pts" / "o" / "manifest"
    results["manifest_bytes"] = manifest.stat().st_size

    # 2. (Format 1 swept octree depth and target size here; bench_bounds.py sweeps format 2's bounds.)

    # 3. Box queries.
    occ = {}
    for f, (c, _) in ((1, leaf), (COARSE, coarse)):
        g = np.zeros((GRID // f,) * 3, dtype=bool)
        g[tuple(c.T)] = True
        occ[f] = g
    sizes_q = [2, 8, 16, 32, 64] if not args.quick else [8, 32]
    latencies = [0.0, 0.02, 0.08] if not args.quick else [0.0, 0.02]
    queries = []
    for latency in latencies:
        n_q = 2 if args.quick else (12 if latency == 0 else 6)
        readers = {layout: Reader(layout, args.stores, latency, args.max_inflight) for layout in LAYOUTS}
        for size in sizes_q:
            if latency >= 0.08 and size >= 64:
                continue  # ~30k leaf requests per query; the 0 and 20 ms runs cover this size
            for placement in ("dense", "uniform"):
                brng = np.random.default_rng([size, placement == "dense"])  # same boxes at every latency
                boxes = [query_box(brng, size * LEAF, placement, pts) for _ in range(n_q)]
                for mode in ("occupied", "slice"):
                    if mode == "slice" and size > 16:
                        continue  # a slice asks for every chunk in the box: 32^3 leaf requests is too many
                    for layout in LAYOUTS:
                        for qi, (lo, hi) in enumerate(boxes):
                            r = run_query(readers[layout], occ, lo, hi, mode)
                            r.update(layout=layout, latency=latency, size=size, placement=placement, mode=mode, query=qi)
                            queries.append(r)
                        last = [q for q in queries[-n_q:]]
                        print(f"lat {latency*1e3:.0f} ms  {mode:8s} {placement:7s} box {size:2d}  {layout:7s} "
                              f"median {np.median([q['seconds'] for q in last])*1e3:8.1f} ms  "
                              f"requests {np.median([q['requests'] for q in last]):7.0f}", flush=True)
    results["queries"] = queries

    # 4. Cold open: first query from a fresh process-level store (metadata + manifest).
    cold = []
    for latency in latencies:
        for layout in LAYOUTS:
            crng = np.random.default_rng(7)
            for qi in range(3 if args.quick else 8):
                lo, hi = query_box(crng, 16 * LEAF, "dense", pts)
                t0 = time.perf_counter()
                reader = Reader(layout, args.stores, latency, args.max_inflight)
                opened = reader.counter.counts.gets
                reader.counter.counts.gets = 0
                r = run_query(reader, occ, lo, hi, "occupied")
                r.update(layout=layout, latency=latency, open_requests=opened, open_and_query_seconds=time.perf_counter() - t0, query=qi)
                cold.append(r)
    results["cold"] = cold

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
