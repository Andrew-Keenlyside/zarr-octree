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


def sharded(store, shard=(16, 16, 16), lo=0, hi=8_000, name="pts", grid=GRID):
    codec = OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=[VLenBytesCodec(), ZstdCodec()], min_weight=lo, max_weight=hi)
    return zarr.create_array(store, name=name, shape=grid, chunks=shard, serializer=codec, compressors=None, dtype="bytes")


def shard_objects(store, name="pts"):
    async def run():
        return {k: (await store.get(k, default_buffer_prototype())).to_bytes() async for k in store.list_prefix(f"{name}/c/")}
    return sync(run())


def test_vlen_cells_round_trip_and_zarr_sees_a_sharded_array():
    cells = point_cells()
    store = MemoryStore()
    arr = sharded(store)
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
    assert OctreeShardingCodec.from_dict(c.to_dict()) == c


def test_dense_rows_for_external_readers():
    cells = point_cells()
    store = MemoryStore()
    write_cells(sharded(store, lo=2_000, hi=8_000), cells)
    blob = shard_objects(store)["pts/c/0/0/0"]
    index = index_from_shard(blob, (16, 16, 16))
    tail = blob[-index_read_nbytes((16, 16, 16)):]
    rows = dense_rows(tail, (16, 16, 16), runs=True)
    stored = rows[:, 0] != np.iinfo(np.uint64).max
    assert stored.sum() == len(index.codes) and np.array_equal(rows[:, :2], dense_rows(tail, (16, 16, 16)))
    pos = {tuple(int(x) for x in c): i for i, c in enumerate(index.coords)}
    for (off, n, run), flat in zip(rows[stored], np.flatnonzero(stored)):
        i = pos[tuple(int(x) for x in np.unravel_index(flat, (16, 16, 16)))]
        assert (int(index.offsets[i]), int(index.lengths[i]), int(index.run_of()[i])) == (off, n, run)
