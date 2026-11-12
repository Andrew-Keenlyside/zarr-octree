"""Prototype B: the zarr_octree_sharding codec, through zarr's own API."""

import json
import subprocess
import sys

import numpy as np
import pytest
import zarr
from zarr.codecs import BytesCodec, VLenBytesCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from zarr_octree.codec import OctreeShardingCodec, dense_rows, index_from_shard, index_read_nbytes
from zarr_octree.instrument import CountingStore

from test_store import GRID, as_object_array, point_cells, write_cells


def sharded(store, shard=(16, 16, 16), lo=0, hi=8_000, name="pts", grid=GRID, layout="runs", target=4_000):
    bounds = {"min_weight": lo, "max_weight": hi} if layout == "runs" else {"target": target}
    codec = OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=[VLenBytesCodec(), ZstdCodec()], layout=layout, **bounds)
    return zarr.create_array(store, name=name, shape=grid, chunks=shard, serializer=codec, compressors=None, dtype="bytes")


def shard_objects(store, name="pts"):
    async def run():
        return {k: (await store.get(k, default_buffer_prototype())).to_bytes() async for k in store.list_prefix(f"{name}/c/")}
    return sync(run())


@pytest.mark.parametrize("layout", ["octree", "runs"])
def test_vlen_cells_round_trip_and_zarr_sees_a_sharded_array(layout):
    cells = point_cells()
    store = MemoryStore()
    arr = sharded(store, layout=layout)
    write_cells(arr, cells)
    arr = zarr.open_array(store, path="pts", mode="r")
    assert arr.shards == (16, 16, 16) and arr.chunks == (1, 1, 1)
    np.testing.assert_array_equal(arr[...], as_object_array(cells))
    idx = np.array(list(cells)[::7]).T
    assert list(arr.vindex[tuple(idx)]) == list(cells.values())[::7]
    np.testing.assert_array_equal(arr[3:19, 5:9, 20:31], as_object_array(cells)[3:19, 5:9, 20:31])


def test_runs_respect_the_bounds_and_the_index_is_sparse():
    cells = point_cells()
    store = MemoryStore()
    write_cells(sharded(store, lo=2_000, hi=8_000), cells)
    weights, n_stored = [], 0
    for key, blob in shard_objects(store).items():
        index = index_from_shard(blob, (16, 16, 16))
        n_stored += len(index.codes)
        for r, (a, b, w) in enumerate(index.runs()):
            single = index.run_start[r + 1] - index.run_start[r] == 1
            assert single or w <= 8_000
            weights.append(w)
        index_nbytes = len(blob) - int(sum(index.lengths))
        assert index_nbytes <= 64 + 6 * len(index.codes) + 12 * len(index.runs())  # 6 bytes a stored chunk, not 16 a possible one
        assert index_nbytes <= index_read_nbytes((16, 16, 16))  # a reader's one suffix read holds it
    assert n_stored == len(cells)
    assert (np.array(weights) >= 2_000).mean() > 0.8


def test_partial_reads_fetch_one_range_per_run():
    cells = point_cells()
    inner = MemoryStore()
    write_cells(sharded(inner, hi=8_000), cells)
    counter = CountingStore(inner, record_keys=True)
    arr = zarr.open_array(counter, path="pts", mode="r")
    want = [c for c in cells if all(0 <= x < 16 for x in c)][:400]  # one shard
    counter.counts.reset()
    got = arr.vindex[tuple(np.array(want).T)]
    assert list(got) == [cells[c] for c in want]
    blob = shard_objects(inner)["pts/c/0/0/0"]
    index = index_from_shard(blob, (16, 16, 16))
    pos = {tuple(int(x) for x in c): i for i, c in enumerate(index.coords)}
    runs_touched = len({int(index.run_of()[pos[c]]) for c in want})
    shard_reads = [k for k in counter.counts.keys if k.startswith("pts/c/")]
    # Index reads (zarr may decode the shard's selection in several passes) plus one per run touched.
    assert len(shard_reads) <= runs_touched + 4


def test_edits_and_deletes():
    cells = point_cells()
    store = MemoryStore()
    arr = sharded(store)
    write_cells(arr, cells)
    expect = dict(cells)
    keys = list(cells)
    rng = np.random.default_rng(0)
    for c in [keys[i] for i in rng.choice(len(keys), 40, replace=False)]:
        expect[c] = b"\x02" * int(rng.integers(1, 5000))
        v = np.empty((1, 1, 1), dtype=object); v.flat[0] = expect[c]
        arr[c[0]:c[0] + 1, c[1]:c[1] + 1, c[2]:c[2] + 1] = v
    for c in [keys[i] for i in rng.choice(len(keys), 40, replace=False)]:
        expect.pop(c, None)
        v = np.empty((1, 1, 1), dtype=object); v.flat[0] = b""
        arr[c[0]:c[0] + 1, c[1]:c[1] + 1, c[2]:c[2] + 1] = v
    np.testing.assert_array_equal(zarr.open_array(store, path="pts")[...], as_object_array(expect))


def test_numeric_array_and_multi_cell_inner_chunks():
    data = np.zeros((64, 48, 40), dtype="float32")
    data[10:30, 5:20, 0:12] = np.random.default_rng(3).normal(size=(20, 15, 12))
    store = MemoryStore()
    codec = OctreeShardingCodec(chunk_shape=(4, 4, 4), codecs=[BytesCodec(), ZstdCodec()], max_weight=2_000)
    arr = zarr.create_array(store, name="img", shape=data.shape, chunks=(32, 16, 40), serializer=codec, compressors=None, dtype="float32", fill_value=0)
    arr[...] = data
    arr = zarr.open_array(store, path="img")
    np.testing.assert_array_equal(arr[...], data)
    np.testing.assert_array_equal(arr[11:29, 3:17, 5:9], data[11:29, 3:17, 5:9])


def test_metadata_names_the_codec_and_a_reader_without_it_refuses():
    store = MemoryStore()
    write_cells(sharded(store, lo=10, hi=20), {(0, 0, 0): b"x"})
    meta = json.loads(sync(store.get("pts/zarr.json", default_buffer_prototype())).to_bytes())
    codec = meta["codecs"][0]
    assert codec["name"] == "zarr_octree_sharding" and codec["configuration"]["max_weight"] == 20
    # A fresh interpreter that has not imported zarr_octree (and has no entry point for it).
    code = (
        "import json, sys, zarr\n"
        "from zarr.storage import MemoryStore\n"
        "from zarr.core.buffer import default_buffer_prototype as p\n"
        "from zarr.core.sync import sync\n"
        "s = MemoryStore()\n"
        "sync(s.set('a/zarr.json', p().buffer.from_bytes(sys.stdin.read().encode())))\n"
        "try:\n    zarr.open_array(s, path='a', mode='r')\nexcept Exception as e:\n    print('refused', type(e).__name__)\n"
        "else:\n    print('opened')\n"
    )
    probe = "import importlib.metadata as md; print(any(e.name == 'zarr_octree_sharding' for e in md.entry_points(group='zarr.codecs')))"
    if subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True).stdout.strip() == "True":
        pytest.skip("zarr_octree is installed with its entry point in this environment; zarr would load it")
    r = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps(meta), capture_output=True, text=True)
    assert r.stdout.startswith("refused"), r.stdout + r.stderr


def test_codec_round_trips_through_its_dict():
    c = OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()], min_weight=5, max_weight=50)
    assert c.layout == "runs" and OctreeShardingCodec.from_dict(c.to_dict()) == c
    d = OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()], target=500)
    assert d.layout == "octree" and OctreeShardingCodec.from_dict(d.to_dict()) == d
    assert "min_weight" not in d.to_dict()["configuration"]


def test_default_is_an_octree_aimed_at_100k_and_old_metadata_reads_as_runs():
    c = OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()])
    assert (c.layout, c.target) == ("octree", 100_000)
    old = {"name": "zarr_octree_sharding", "configuration": {"version": 2, "chunk_shape": (2, 2), "codecs": [BytesCodec().to_dict()],
                                                             "min_weight": 5, "max_weight": 50, "unit": "bytes"}}
    assert OctreeShardingCodec.from_dict(old).layout == "runs"
    with pytest.raises(ValueError):
        OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()], layout="grid")


def test_octree_layout_runs_are_aligned_octree_blocks():
    cells = point_cells()
    store = MemoryStore()
    write_cells(sharded(store, layout="octree", target=4_000), cells)
    weights = []
    for blob in shard_objects(store).values():
        index = index_from_shard(blob, (16, 16, 16))
        codes = [int(c) for c in index.codes]
        for r in range(len(index.run_weight)):
            a, b = int(index.run_start[r]), int(index.run_start[r + 1])
            bits = (codes[a] ^ codes[b - 1]).bit_length()
            block = 1 << (3 * -(-bits // 3))
            start = codes[a] - codes[a] % block
            assert (a == 0 or codes[a - 1] < start) and (b == len(codes) or codes[b] >= start + block)
            weights.append(float(index.run_weight[r]))
    assert 4_000 / 4 < np.median(weights) < 4_000 * 4


@pytest.mark.parametrize("heavy_at", [(1, 1, 1), (0, 0, 0), (1, 0, 1)])
def test_runs_layout_writes_a_cell_heavier_than_max_weight(heavy_at):
    # The last cell in Z-order (1, 1, 1) heavier than max_weight once crashed the runs layout's writer.
    rng = np.random.default_rng(0)
    store = MemoryStore()
    codec = OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=[VLenBytesCodec(), ZstdCodec()], layout="runs", min_weight=500, max_weight=2_000)
    arr = zarr.create_array(store, name="pts", shape=(4, 4, 4), chunks=(2, 2, 2), serializer=codec, compressors=None, dtype="bytes")
    cells = {c: rng.integers(0, 256, 5_000 if c == heavy_at else 50, dtype=np.uint8).tobytes()  # incompressible
             for c in [(0, 1, 0), (1, 1, 0), heavy_at, (2, 3, 2)]}
    write_cells(arr, cells)
    arr = zarr.open_array(store, path="pts", mode="r")
    np.testing.assert_array_equal(arr[...], as_object_array(cells, (4, 4, 4)))
    index = index_from_shard(shard_objects(store)["pts/c/0/0/0"], (2, 2, 2))
    heavy = [r for r, (a, b, w) in enumerate(index.runs()) if w > 2_000]
    assert len(heavy) == 1 and index.run_start[heavy[0] + 1] - index.run_start[heavy[0]] == 1


@pytest.mark.parametrize("keep", [0, 3, 15, 20, "half"])
def test_a_truncated_shard_is_a_clear_format_error(keep):
    store = MemoryStore()
    write_cells(sharded(store), point_cells())
    key, blob = next(iter(shard_objects(store).items()))
    cut = blob[: len(blob) // 2] if keep == "half" else blob[:keep]
    with pytest.raises(ValueError, match="zarr_octree_sharding"):
        index_from_shard(cut, (16, 16, 16))
    tail = blob[-16:] if keep == "half" else cut  # a tail that is only the footer, or a short object
    with pytest.raises(ValueError, match="zarr_octree_sharding|shorter than"):
        dense_rows(tail, (16, 16, 16))
    sync(store.set(key, default_buffer_prototype().buffer.from_bytes(cut)))
    arr = zarr.open_array(store, path="pts", mode="r")
    with pytest.raises(ValueError, match="zarr_octree_sharding"):
        arr[...]
    first = tuple(16 * int(c) for c in key.split("/")[-3:])
    with pytest.raises(ValueError, match="zarr_octree_sharding"):
        arr[first]  # a partial read: the index alone
