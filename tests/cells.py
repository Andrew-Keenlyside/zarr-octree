"""Test data: a clustered point cloud binned into cells, and writing cells into an array."""

import numpy as np

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
