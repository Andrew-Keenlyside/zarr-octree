import numpy as np
import pytest

from zarr_octree.tree import (
    build_partition,
    decode_manifest,
    decode_node,
    encode_manifest,
    encode_node,
    NodeInfo,
)


def clustered_cells(rng, n=4000, grid=64, ndim=3):
    """Occupied cells around a few tight clusters plus a sparse background."""
    centres = rng.integers(8, grid - 8, size=(4, ndim))
    near = centres[rng.integers(0, 4, n)] + rng.normal(0, 2.0, (n, ndim))
    far = rng.uniform(0, grid, (n // 10, ndim))
    cells = np.clip(np.vstack([near, far]), 0, grid - 1).astype(np.int64)
    cells, counts = np.unique(cells, axis=0, return_counts=True)
    return cells, counts * 120  # bytes grow with the points in the cell


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_partition_covers_each_cell_once_and_respects_target(ndim):
    rng = np.random.default_rng(ndim)
    cells, nbytes = clustered_cells(rng, grid=32 if ndim == 4 else 64, ndim=ndim)
    D, target = 5, 20_000
    nodes = build_partition(cells, nbytes, D, target)
    seen = np.concatenate([m for _, _, m in nodes])
    assert sorted(seen) == list(range(len(cells)))
    keys = set()
    for depth, coords, members in nodes:
        assert (cells[members] >> (D - depth) == np.array(coords)).all()
        assert nbytes[members].sum() <= target or depth == D
        keys.add((depth, coords))
    # No stored node is an ancestor of another: the layers are mutually exclusive.
    for depth, coords in keys:
        for up in range(depth):
            assert (up, tuple(c >> (depth - up) for c in coords)) not in keys


def test_partition_adapts_depth_to_density():
    rng = np.random.default_rng(0)
    cells, nbytes = clustered_cells(rng)
    nodes = build_partition(cells, nbytes, 5, 20_000)
    depth_of_cell = np.empty(len(cells), dtype=int)
    for depth, _, members in nodes:
        depth_of_cell[members] = depth
    dense = nbytes >= np.quantile(nbytes, 0.9)
    assert depth_of_cell[dense].mean() > depth_of_cell[~dense].mean() + 0.5


def test_node_round_trip():
    local = np.array([[3, 1, 0], [0, 0, 0], [1, 2, 3]])
    payloads = [b"three", b"", b"x" * 1000]
    node = decode_node(encode_node(2, (5, 6, 7), local, payloads, max_depth=4))
    assert node.depth == 2 and node.coords == (5, 6, 7)
    assert node.cell((3, 1, 0)) == b"three"
    assert node.cell((0, 0, 0)) == b""
    assert node.cell((1, 2, 3)) == b"x" * 1000
    assert node.cell((1, 1, 1)) is None
    assert node.cell((4, 0, 0)) is None  # outside the node (side 4)
    # Absolute coords: node (5,6,7) at depth 2 of max depth 4 starts at cell (20,24,28).
    assert node.cells()[(23, 25, 28)] == b"three"


def test_node_bytes_do_not_depend_on_insertion_order():
    local = np.array([[1, 0], [0, 1], [0, 0]])
    payloads = [b"a", b"b", b"c"]
    perm = [2, 0, 1]
    assert encode_node(1, (0, 0), local, payloads, max_depth=2) == encode_node(1, (0, 0), local[perm], [payloads[i] for i in perm], max_depth=2)


def test_manifest_round_trip():
    nodes = {(0, (0, 0, 0)): NodeInfo(10, 2), (3, (9, 8, 7)): NodeInfo(5_000_000_000, 70_000)}
    ndim, D, back = decode_manifest(encode_manifest(3, 6, nodes))
    assert (ndim, D) == (3, 6)
    assert back == nodes


def test_rejects_negative_coords():
    with pytest.raises(ValueError):
        build_partition(np.array([[-1, 0]]), np.array([1]), 2, 10)
