import numpy as np
import pytest

from zarr_octree.tree import (
    NodeInfo,
    block_of,
    boundary_of,
    build_octree,
    build_runs,
    decode_manifest,
    decode_node,
    encode_manifest,
    encode_node,
    morton,
    unmorton,
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
def test_morton_round_trip_and_blocks_are_contiguous(ndim):
    bits = 4 if ndim == 4 else 5
    rng = np.random.default_rng(ndim)
    local = rng.integers(0, 2**bits, (500, ndim))
    assert (unmorton(morton(local, bits), ndim, bits) == local).all()
    # Every aligned block is one contiguous range of codes.
    every = np.stack(np.unravel_index(np.arange(2 ** (bits * ndim)), (2**bits,) * ndim), 1)
    codes = morton(every, bits)
    for level in range(1, bits):
        block = every >> level
        for b in np.unique(block, axis=0)[:5]:
            c = np.sort(codes[(block == b).all(1)])
            assert (np.diff(c) == 1).all()


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_runs_cover_each_cell_once_and_respect_the_bounds(ndim):
    rng = np.random.default_rng(ndim)
    D = 5
    cells, w = clustered_cells(rng, grid=2**D, ndim=ndim)
    codes = morton(cells, D)
    order = np.argsort(codes)
    codes, w = codes[order], w[order]
    lo, hi = 5_000, 20_000
    runs = build_runs(codes, w, ndim, lo, hi)
    assert runs[0][0] == 0 and runs[0][1] == 0 and runs[-1][2] == len(codes)
    for (b0, a0, s0), (b1, a1, s1) in zip(runs, runs[1:]):
        assert s0 == a1 and b0 < b1  # consecutive, increasing ranges
        assert codes[s0 - 1] < b1 <= codes[a1]  # the boundary separates the runs
    weights = np.array([w[a:b].sum() for _, a, b in runs])
    single = np.array([b - a == 1 for _, a, b in runs])
    assert (weights[~single] <= hi).all()
    assert (weights[:-1] >= lo).mean() > 0.9  # light runs only where forced
    # Boundaries sit on octree blocks: each names the block it starts.
    for b, _, _ in runs:
        depth, at = block_of(b, ndim, D)
        assert boundary_of((depth, tuple(int(x) for x in at)), D) == b


def test_boundaries_prefer_coarse_octree_blocks():
    rng = np.random.default_rng(0)
    cells, w = clustered_cells(rng, grid=32)
    codes = np.sort(morton(cells, 5))
    w = np.full(len(codes), 100)
    runs = build_runs(codes, w, 3, 2_000, 8_000)
    depths = [block_of(b, 3, 5)[0] for b, _, _ in runs]
    # Any cut point would do for uniform weights; the chosen ones are far coarser than a cell.
    assert np.mean(depths) < 4


def test_min_weight_merges_light_neighbours():
    rng = np.random.default_rng(1)
    cells, w = clustered_cells(rng, grid=32)
    codes = morton(cells, 5)
    order = np.argsort(codes)
    without = build_runs(codes[order], w[order], 3, 0, 20_000)
    with_min = build_runs(codes[order], w[order], 3, 10_000, 20_000)
    light = lambda runs: sum(w[order][a:b].sum() < 10_000 for _, a, b in runs)
    assert light(with_min) < light(without)


def test_node_round_trip():
    codes = np.array([7, 0, 300], dtype=np.uint64)
    payloads = [b"seven", b"", b"x" * 1000]
    node = decode_node(encode_node(2, (5, 6, 7), codes, payloads, max_depth=4))
    assert node.depth == 2 and node.coords == (5, 6, 7) and node.root == (1, 1, 1)
    assert node.cell(7) == b"seven" and node.cell(0) == b"" and node.cell(300) == b"x" * 1000
    assert node.cell(8) is None
    local = unmorton(np.array([7], dtype=np.uint64), 3, 4)[0]
    assert node.cells()[tuple(int(x) for x in local + 16)] == b"seven"


def test_node_bytes_do_not_depend_on_insertion_order():
    codes = np.array([5, 1, 3], dtype=np.uint64)
    payloads = [b"a", b"b", b"c"]
    perm = [2, 0, 1]
    assert encode_node(1, (0, 0), codes, payloads, max_depth=2) == encode_node(1, (0, 0), codes[perm], [payloads[i] for i in perm], max_depth=2)


def test_manifest_round_trip():
    nodes = {(0, (0, 0, 0)): NodeInfo(10, 2, 0, 9, 1.5), (3, (9, 8, 7)): NodeInfo(5_000_000_000, 70_000, 2**40, 2**62, 7e4)}
    ndim, D, back = decode_manifest(encode_manifest(3, 6, nodes))
    assert (ndim, D) == (3, 6)
    assert back == nodes


def test_rejects_unsorted_codes_and_oversized_codes():
    with pytest.raises(ValueError):
        build_runs(np.array([3, 1], dtype=np.uint64), np.array([1, 1]), 2, 0, 10)
    with pytest.raises(ValueError):
        morton(np.zeros((1, 4)), 17)


@pytest.mark.parametrize("target", [500, 5_000, 50_000])
def test_octree_leaves_are_aligned_blocks_covering_every_cell(target):
    rng = np.random.default_rng(4)
    cells, weights = clustered_cells(rng)
    codes = morton(cells, 6)
    order = np.argsort(codes)
    codes, weights = codes[order], weights[order]
    leaves = build_octree(codes, weights, 3, 6, target)
    assert [a for _, a, _ in leaves] == [0] + [b for _, _, b in leaves[:-1]] and leaves[-1][2] == len(codes)
    for start, a, b in leaves:
        block = next(1 << (3 * lv) for lv in range(7) if start % (1 << (3 * lv)) == 0 and int(codes[b - 1]) < start + (1 << (3 * lv)))
        assert start <= int(codes[a])  # the run's cells lie in one aligned block
        assert (a == 0 or int(codes[a - 1]) < start) and (b == len(codes) or int(codes[b]) >= start + block)  # which it owns


def test_octree_spreads_around_the_target_with_no_hard_bound():
    rng = np.random.default_rng(5)
    cells, weights = clustered_cells(rng, n=20_000)
    codes = morton(cells, 6)
    order = np.argsort(codes)
    codes, weights = codes[order], weights[order]
    w = np.array([weights[a:b].sum() for _, a, b in build_octree(codes, weights, 3, 6, 20_000)])
    assert w.sum() == weights.sum()
    assert 20_000 / 4 < np.median(w) < 20_000 * 4  # aimed at the target
    assert (w > 20_000).any() and (w < 20_000).any()  # on both sides of it: no limit


def test_octree_keeps_a_light_root_and_splits_nothing_below_one_cell():
    codes = np.array([0, 9, 200], dtype=np.uint64)
    assert build_octree(codes, np.array([1.0, 1.0, 1.0]), 3, 3, 10_000) == [(0, 0, 3)]
    leaves = build_octree(codes, np.array([5e5, 5e5, 5e5]), 3, 3, 10)
    assert [(a, b) for _, a, b in leaves] == [(0, 1), (1, 2), (2, 3)]
