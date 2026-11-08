"""The extension through zarr's own API: what zarr sees must not change."""

import json

import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.errors import MetadataValidationError
from zarr.storage import MemoryStore

from zarr_octree import EXTENSION, OctreeConfig, OctreeStore, convert
from zarr_octree.instrument import CountingStore

GRID = (32, 32, 32)


def point_cells(seed=0, n=20_000, grid=GRID):
    """A clustered point cloud binned into cells: ``{cell: float32 xyz bytes}``."""
    rng = np.random.default_rng(seed)
    g = np.array(grid)
    centres = rng.uniform(0.2, 0.8, (3, 3)) * g
    pts = np.vstack(
        [
            centres[rng.integers(0, 3, n)] + rng.normal(0, 1.5, (n, 3)),
            rng.uniform(0, 1, (n // 20, 3)) * g,
        ]
    )
    pts = pts[((pts >= 0) & (pts < g)).all(1)].astype(np.float32)
    cell = np.floor(pts).astype(int)
    out = {}
    for c in np.unique(cell, axis=0):
        out[tuple(int(x) for x in c)] = pts[(cell == c).all(1)].tobytes()
    return out


def as_object_array(cells, grid=GRID):
    a = np.full(grid, b"", dtype=object)
    for c, b in cells.items():
        a[c] = b
    return a


def write_cells(arr, cells):
    idx = np.array(list(cells)).T
    vals = np.empty(len(cells), dtype=object)
    vals[:] = list(cells.values())
    arr.vindex[tuple(idx)] = vals


def set_cell(arr, cell, data):
    v = np.empty((1,) * len(cell), dtype=object)
    v.flat[0] = data
    arr[tuple(slice(c, c + 1) for c in cell)] = v


def octree_array(inner, cells, config=OctreeConfig(max_depth=4, target_bytes=8_000), name="pts"):
    store = OctreeStore(inner, arrays={name: config})
    arr = zarr.create_array(store, name=name, shape=GRID, chunks=(1, 1, 1), dtype="bytes")
    write_cells(arr, cells)
    store.flush()
    return store


def test_vlen_point_cells_round_trip_through_zarr():
    cells = point_cells()
    inner = MemoryStore()
    octree_array(inner, cells)
    # A fresh store over the same bytes: nothing carried over in memory.
    arr = zarr.open_array(OctreeStore(inner), path="pts", mode="r")
    np.testing.assert_array_equal(arr[...], as_object_array(cells))
    idx = np.array(list(cells)[:500]).T
    assert list(arr.vindex[tuple(idx)]) == list(cells.values())[:500]


def test_physical_layout_is_nodes_and_a_manifest():
    inner = MemoryStore()
    store = octree_array(inner, point_cells())
    keys = [k for k in inner._store_dict]
    assert not any(k.startswith("pts/c/") for k in keys)
    assert "pts/o/manifest" in keys
    nodes = store.nodes("pts")
    assert {f"pts/o/{d}/" + "/".join(map(str, c)) for d, c in nodes} == {k for k in keys if k.startswith("pts/o/") and k != "pts/o/manifest"}
    assert len({d for d, _ in nodes}) > 1  # more than one depth was used
    assert len(nodes) < len(point_cells()) / 5


def test_nodes_hold_similar_amounts():
    cells = point_cells()
    store = octree_array(MemoryStore(), cells)
    sizes = np.array([info.nbytes for (d, _), info in store.nodes("pts").items() if d < 4])
    assert sizes.max() <= 8_000 + 64  # header slack
    cell_sizes = np.array([len(b) for b in cells.values()])
    assert sizes.std() / sizes.mean() < cell_sizes.std() / cell_sizes.mean()


def test_plain_zarr_refuses_the_array_and_wrapped_zarr_sees_plain_metadata():
    inner = MemoryStore()
    octree_array(inner, point_cells())
    with pytest.raises(MetadataValidationError):
        zarr.open_array(inner, path="pts", mode="r")
    raw = json.loads(sync(inner.get("pts/zarr.json", default_buffer_prototype())).to_bytes())
    assert raw[EXTENSION]["must_understand"] is True
    assert raw["storage_transformers"][0]["configuration"]["max_depth"] == 4
    seen = json.loads(sync(OctreeStore(inner).get("pts/zarr.json", default_buffer_prototype())).to_bytes())
    assert EXTENSION not in seen and "storage_transformers" not in seen


def test_metadata_rewrites_keep_the_extension():
    inner = MemoryStore()
    store = octree_array(inner, point_cells())
    arr = zarr.open_array(store, path="pts", mode="r+")
    arr.attrs["note"] = "edited"
    arr.resize((40, 32, 32))
    with pytest.raises(MetadataValidationError):
        zarr.open_array(inner, path="pts", mode="r")
    arr = zarr.open_array(OctreeStore(inner), path="pts", mode="r")
    assert arr.attrs["note"] == "edited" and arr.shape == (40, 32, 32)


def test_reads_see_buffered_writes_before_flush():
    cells = point_cells()
    store = OctreeStore(MemoryStore(), arrays={"pts": OctreeConfig(4, 8_000)})
    arr = zarr.create_array(store, name="pts", shape=GRID, chunks=(1, 1, 1), dtype="bytes")
    write_cells(arr, cells)
    np.testing.assert_array_equal(arr[...], as_object_array(cells))


def test_edits_overwrite_delete_and_add():
    cells = point_cells()
    inner = MemoryStore()
    store = octree_array(inner, cells)
    arr = zarr.open_array(store, path="pts", mode="r+")
    keys = list(cells)
    rng = np.random.default_rng(1)
    expect = dict(cells)
    for c in [keys[i] for i in rng.choice(len(keys), 50, replace=False)]:
        expect[c] = b"\x01" * int(rng.integers(1, 3000))  # some grow past the target
        set_cell(arr, c, expect[c])
    for c in [keys[i] for i in rng.choice(len(keys), 50, replace=False)]:
        expect.pop(c, None)
        set_cell(arr, c, b"")  # all-fill chunk: zarr deletes it
    empty = [c for c in np.ndindex(*GRID) if c not in cells][:20:3]
    for c in empty:
        expect[c] = b"new"
        set_cell(arr, c, b"new")
    store.flush()
    arr = zarr.open_array(OctreeStore(inner), path="pts", mode="r")
    np.testing.assert_array_equal(arr[...], as_object_array(expect))
    check_partition(OctreeStore(inner).nodes("pts"), max_depth=4)


def check_partition(nodes, max_depth):
    keys = set(nodes)
    for depth, coords in keys:
        for up in range(depth):
            assert (up, tuple(c >> (depth - up) for c in coords)) not in keys


def test_repack_matches_a_fresh_write():
    cells = point_cells()
    inner = MemoryStore()
    store = octree_array(inner, cells)
    arr = zarr.open_array(store, path="pts", mode="r+")
    for c in list(cells)[::2]:
        set_cell(arr, c, b"")
    store.flush()
    store.repack("pts")
    fresh = octree_array(MemoryStore(), {c: b for i, (c, b) in enumerate(cells.items()) if i % 2})
    assert set(store.nodes("pts")) == set(fresh.nodes("pts"))


def test_one_request_per_node():
    cells = point_cells()
    inner = MemoryStore()
    octree_array(inner, cells)
    counter = CountingStore(inner, record_keys=True)
    arr = zarr.open_array(OctreeStore(counter), path="pts", mode="r")
    counter.counts.reset()
    arr[...]
    node_reads = [k for k in counter.counts.keys if "/o/" in k and not k.endswith("manifest")]
    assert len(node_reads) == len(set(node_reads)) == len(OctreeStore(inner).nodes("pts"))
    assert counter.counts.keys.count("pts/o/manifest") == 1
    # Empty cells are answered from the manifest: no request per missing chunk.
    assert counter.counts.gets == len(node_reads) + 1


def test_numeric_array_with_multi_cell_chunks():
    rng = np.random.default_rng(3)
    data = np.zeros((64, 48, 40), dtype="float32")
    data[10:30, 5:20, 0:12] = rng.normal(size=(20, 15, 12))
    data[50:60, 40:48, 30:40] = 1.5
    inner = MemoryStore()
    store = OctreeStore(inner, arrays={"img": OctreeConfig(3, 4_000)})
    arr = zarr.create_array(store, name="img", shape=data.shape, chunks=(4, 4, 4), dtype="float32", fill_value=0)
    arr[...] = data
    store.flush()
    np.testing.assert_array_equal(zarr.open_array(OctreeStore(inner), path="img")[...], data)


def test_sharded_array_byte_range_reads():
    data = np.arange(32 * 32, dtype="int32").reshape(32, 32)
    inner = MemoryStore()
    store = OctreeStore(inner, arrays={"s": OctreeConfig(2, 2_000)})
    arr = zarr.create_array(store, name="s", shape=data.shape, chunks=(4, 4), shards=(8, 8), dtype="int32")
    arr[...] = data
    store.flush()
    arr = zarr.open_array(OctreeStore(inner), path="s", mode="r")
    np.testing.assert_array_equal(arr[...], data)
    np.testing.assert_array_equal(arr[3:9, 17:30], data[3:9, 17:30])


@pytest.mark.parametrize("sep", ["/", "."])
@pytest.mark.parametrize("ndim", [2, 4])
def test_other_ranks_and_separators(ndim, sep):
    shape = (16,) * ndim
    rng = np.random.default_rng(ndim)
    data = (rng.random(shape) > 0.9).astype("uint8") * rng.integers(1, 255, shape).astype("uint8")
    inner = MemoryStore()
    store = OctreeStore(inner, arrays={"a": OctreeConfig(2, 64)})
    arr = zarr.create_array(
        store, name="a", shape=shape, chunks=(2,) * ndim, dtype="uint8",
        chunk_key_encoding={"name": "default", "separator": sep},
    )
    arr[...] = data
    store.flush()
    np.testing.assert_array_equal(zarr.open_array(OctreeStore(inner), path="a")[...], data)


def test_convert_and_listing_match_the_regular_array():
    cells = point_cells()
    regular = MemoryStore()
    write_cells(zarr.create_array(regular, name="pts", shape=GRID, chunks=(1, 1, 1), dtype="bytes"), cells)
    inner = MemoryStore()
    store = convert(regular, "pts", inner, OctreeConfig(4, 8_000))
    np.testing.assert_array_equal(zarr.open_array(store, path="pts")[...], as_object_array(cells))

    async def keys(s, prefix):
        return sorted([k async for k in s.list_prefix(prefix)])

    assert sync(keys(OctreeStore(inner), "pts/c/")) == sync(keys(regular, "pts/c/"))
    assert sync(keys(OctreeStore(inner), "pts")) == sync(keys(regular, "pts"))


def test_other_arrays_and_groups_pass_through():
    inner = MemoryStore()
    store = OctreeStore(inner, arrays={"g/pts": OctreeConfig(4, 8_000)})
    root = zarr.open_group(store, mode="w")
    g = root.create_group("g")
    plain = g.create_array("dense", shape=(8, 8), chunks=(4, 4), dtype="int16")
    plain[...] = 7
    pts = g.create_array("pts", shape=GRID, chunks=(1, 1, 1), dtype="bytes")
    write_cells(pts, point_cells())
    store.flush()
    assert "g/dense/c/0/0" in inner._store_dict
    assert zarr.open_array(inner, path="g/dense")[0, 0] == 7
    assert sorted(zarr.open_group(OctreeStore(inner), path="g").array_keys()) == ["dense", "pts"]


def test_auto_flush_when_the_buffer_is_full():
    cells = point_cells()
    inner = MemoryStore()
    store = OctreeStore(inner, arrays={"pts": OctreeConfig(4, 8_000)}, max_buffer_bytes=20_000)
    arr = zarr.create_array(store, name="pts", shape=GRID, chunks=(1, 1, 1), dtype="bytes")
    write_cells(arr, cells)
    assert "pts/o/manifest" in inner._store_dict
    store.flush()
    np.testing.assert_array_equal(zarr.open_array(OctreeStore(inner), path="pts")[...], as_object_array(cells))
    check_partition(OctreeStore(inner).nodes("pts"), 4)


def test_overwriting_the_array_drops_its_nodes():
    inner = MemoryStore()
    octree_array(inner, point_cells())
    store = OctreeStore(inner, arrays={"pts": OctreeConfig(4, 8_000)})
    zarr.create_array(store, name="pts", shape=GRID, chunks=(1, 1, 1), dtype="bytes", overwrite=True)
    assert not any(k.startswith("pts/o/") for k in inner._store_dict)
    assert (zarr.open_array(store, path="pts")[...] == b"").all()


def test_read_only_view_sees_buffered_writes():
    cells = point_cells()
    store = OctreeStore(MemoryStore(), arrays={"pts": OctreeConfig(4, 8_000)})
    write_cells(zarr.create_array(store, name="pts", shape=GRID, chunks=(1, 1, 1), dtype="bytes"), cells)
    arr = zarr.open_array(store, path="pts", mode="r")  # zarr reads through store.with_read_only(True)
    np.testing.assert_array_equal(arr[...], as_object_array(cells))
    with pytest.raises(ValueError):
        sync(store.with_read_only(True).set("pts/c/0/0/0", default_buffer_prototype().buffer.from_bytes(b"x")))
