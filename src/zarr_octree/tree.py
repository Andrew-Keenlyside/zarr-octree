"""Morton (Z-order) codes and the two ways a shard's inner chunks are cut into runs.

Pure numpy: no zarr, no I/O.  A shard is a block of ``2 ** bits`` inner
chunks along every axis; a chunk's **code** is its local coordinates' bits
interleaved, so every aligned octree block of the shard is one contiguous
range of codes.  A **run** is a contiguous range of codes stored as one byte
range.  :func:`build_octree` makes the runs the leaves of an octree aimed at a
target weight; :func:`build_runs` cuts Z-order runs between a minimum and a
maximum weight, on the coarsest octree boundaries.
"""

from __future__ import annotations

import numpy as np


# -- Morton codes ---------------------------------------------------------------


def morton(local: np.ndarray, bits: int) -> np.ndarray:
    """Z-order codes of ``(m, ndim)`` local coordinates in ``[0, 2 ** bits)``; axis 0 is most significant."""
    local = np.asarray(local, dtype=np.uint64)
    m, n = local.shape
    if n * bits > 64:
        raise ValueError(f"{n}-d codes of {bits} bits per axis do not fit in 64 bits")
    code = np.zeros(m, dtype=np.uint64)
    one = np.uint64(1)
    for b in range(bits):
        for a in range(n):
            code |= ((local[:, a] >> np.uint64(b)) & one) << np.uint64(b * n + n - 1 - a)
    return code


def unmorton(code: np.ndarray, ndim: int, bits: int) -> np.ndarray:
    """Inverse of :func:`morton`: ``(m, ndim)`` int64 local coordinates."""
    code = np.asarray(code, dtype=np.uint64)
    out = np.zeros((len(code), ndim), dtype=np.uint64)
    one = np.uint64(1)
    for b in range(bits):
        for a in range(ndim):
            out[:, a] |= ((code >> np.uint64(b * ndim + ndim - 1 - a)) & one) << np.uint64(b)
    return out.astype(np.int64)


def _high_bit(x: np.ndarray) -> np.ndarray:
    """Index of the highest set bit of each (non-zero) uint64."""
    x = np.asarray(x, dtype=np.uint64).copy()
    h = np.zeros(len(x), dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = (x >> np.uint64(s)) != 0
        h[big] += s
        x[big] >>= np.uint64(s)
    return h


# -- partition --------------------------------------------------------------------


def build_runs(
    codes: np.ndarray,
    weights: np.ndarray,
    ndim: int,
    min_weight: float,
    max_weight: float,
) -> list[tuple[int, int, int]]:
    """Cut one root's occupied cells, in code order, into runs of bounded weight.

    Greedy along the Z-order: each run takes the cells from where the last one
    stopped, and ends at the candidate boundary that keeps its weight within
    ``[min_weight, max_weight]`` and lies on the coarsest octree boundary
    (ties go to the heavier run).  A boundary that would leave a remainder
    lighter than ``min_weight`` is avoided when possible.

    Args:
        codes: ``(m,)`` sorted, unique codes of the occupied cells.
        weights: ``(m,)`` weight of each cell.

    Returns:
        ``[(boundary, start, stop), ...]``: the code where each run starts (an
        octree boundary; 0 for the first) and its cells ``codes[start:stop]``.
    """
    codes = np.asarray(codes, dtype=np.uint64)
    m = len(codes)
    if m == 0:
        return []
    if m > 1 and not (codes[1:] > codes[:-1]).all():
        raise ValueError("codes must be sorted and unique")
    W = np.concatenate([[0.0], np.cumsum(np.asarray(weights, dtype=np.float64))])
    out: list[tuple[int, int, int]] = []
    boundary, i = 0, 0
    while i < m:
        if W[m] - W[i] <= max_weight or i == m - 1:
            # The rest fits, or it is one cell heavier than max_weight: a run of its own.
            out.append((boundary, i, m))
            break
        jmax = max(i + 1, int(np.searchsorted(W, W[i] + max_weight, side="right")) - 1)
        # run = cells[i:j]; j < m because the rest exceeds max_weight and holds at
        # least two cells (a single heavy cell before the end gives j = i + 1).
        js = np.arange(i + 1, jmax + 1)
        w = W[js] - W[i]
        tail = W[m] - W[js]
        ok = (w >= min_weight) & ~((tail > 0) & (tail < min_weight))
        if not ok.any():
            ok = w >= min_weight
        if not ok.any():
            ok = js == jmax
        split = _high_bit(codes[js - 1] ^ codes[js])  # the boundary between j-1 and j is aligned to 2**split
        level = split // ndim
        score = np.where(ok, level * (W[m] + 1.0) + w, -1.0)
        k = int(np.argmax(score))
        j = int(js[k])
        out.append((boundary, i, j))
        s = np.uint64(split[k])
        boundary = int((codes[j] >> s) << s)
        i = j
    return out


def build_octree(
    codes: np.ndarray,
    weights: np.ndarray,
    ndim: int,
    bits: int,
    target: float,
) -> list[tuple[int, int, int]]:
    """Cut one root's occupied cells into the leaves of an octree aimed at ``target``.

    Starting from the root block (``bits`` levels), a block heavier than
    ``target`` is split into its ``2 ** ndim`` children when the occupied
    children are, on average, closer to ``target`` in log weight than the block
    itself; otherwise it is a leaf.  No weight is a hard bound: a leaf can be
    lighter or heavier than ``target``, so leaf weights taper off on both sides
    of it rather than piling up under a limit.  Every leaf is an aligned octree
    block, so it is one contiguous range of codes: the leaves, in code order,
    are runs as :func:`build_runs` returns them.

    Args:
        codes: ``(m,)`` sorted, unique codes of the occupied cells.
        weights: ``(m,)`` weight of each cell.
        bits: Octree levels of the root (codes hold ``ndim * bits`` bits).

    Returns:
        ``[(block start code, start, stop), ...]``: each leaf block's first code
        and its cells ``codes[start:stop]``.
    """
    codes = np.asarray(codes, dtype=np.uint64)
    m = len(codes)
    if m == 0:
        return []
    if m > 1 and not (codes[1:] > codes[:-1]).all():
        raise ValueError("codes must be sorted and unique")
    if target <= 0:
        raise ValueError("target must be positive")
    W = np.concatenate([[0.0], np.cumsum(np.asarray(weights, dtype=np.float64))])
    out: list[tuple[int, int, int]] = []
    log_t = np.log(target)

    def visit(start_code: int, level: int, i: int, j: int) -> None:
        w = W[j] - W[i]
        if level > 0 and j - i > 1 and w > target:
            step = 1 << (ndim * (level - 1))
            edges = np.searchsorted(codes[i:j], np.uint64(start_code) + np.arange(1, 1 << ndim, dtype=np.uint64) * np.uint64(step)) + i
            bounds = np.concatenate([[i], edges, [j]])
            kw = W[bounds[1:]] - W[bounds[:-1]]
            occupied = np.diff(bounds) > 0
            heavy = kw[occupied & (kw > 0)]
            if len(heavy) and np.mean(np.abs(np.log(heavy) - log_t)) < abs(np.log(w) - log_t):
                for k in range(1 << ndim):
                    if bounds[k + 1] > bounds[k]:
                        visit(start_code + k * step, level - 1, int(bounds[k]), int(bounds[k + 1]))
                return
        out.append((start_code, i, j))

    visit(0, bits, 0, m)
    return out
