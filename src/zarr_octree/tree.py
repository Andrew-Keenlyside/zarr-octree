"""Partition of a sparse chunk grid into runs of an octree's Z-order, and the on-disk formats.

Pure numpy and ``struct``: no zarr, no I/O.

Terms used throughout:

* **cell** -- one chunk of the logical Zarr array, addressed by its chunk
  coordinates.  Zarr reads and writes cells; it never sees nodes.
* **root** -- a block of ``2 ** max_depth`` cells along every axis.  Roots
  are partitioned independently, so an edit never reaches past its root.
* **code** -- a cell's Morton (Z-order) code within its root: the bits of its
  local coordinates interleaved.  Every aligned octree block of a root is one
  contiguous range of codes.
* **node** -- one stored object: the occupied cells of a contiguous range of
  codes (a *run*).  A run's range starts on an octree boundary, chosen as
  coarse as the size bounds allow, so a run is a union of a few octree blocks.
  The node is named by the largest octree block that starts where it starts.
* **weight** -- what the size bounds count: encoded bytes by default, or
  anything a writer measures per cell (vertices, say).  Every node weighs at
  most ``max_weight`` unless it is a single cell, and at least ``min_weight``
  unless its root holds less in total or a lighter remainder could not be
  merged without exceeding ``max_weight``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

NODE_MAGIC = b"ZOCN"
MANIFEST_MAGIC = b"ZOCM"
FORMAT_VERSION = 2

_NODE_HEADER = struct.Struct("<4sBBBBI")  # magic, version, ndim, depth, max_depth, n_cells
_MANIFEST_HEADER = struct.Struct("<4sBBBxQ")  # magic, version, ndim, max_depth, n_nodes

NodeKey = tuple[int, tuple[int, ...]]  # (depth, coords of the octree block the node starts at)


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


def block_of(boundary: int, ndim: int, max_depth: int) -> tuple[int, np.ndarray]:
    """The largest octree block starting at code ``boundary``: ``(depth, local coords at that depth)``."""
    if boundary == 0:
        return 0, np.zeros(ndim, dtype=np.int64)
    tz = (boundary & -boundary).bit_length() - 1
    depth = max_depth - min(tz // ndim, max_depth)
    local = unmorton(np.array([boundary], dtype=np.uint64), ndim, max_depth)[0]
    return depth, local >> (max_depth - depth)


def boundary_of(key: NodeKey, max_depth: int) -> int:
    """The code where node ``key`` starts, within its root."""
    depth, coords = key
    shift = max_depth - depth
    local = np.array(coords, dtype=np.int64) - ((np.array(coords, dtype=np.int64) >> depth) << depth)
    return int(morton((local << shift)[None], max_depth)[0])


def root_of(key: NodeKey) -> tuple[int, ...]:
    """Coordinates of the root that contains node ``key``."""
    depth, coords = key
    return tuple(c >> depth for c in coords)


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
        if W[m] - W[i] <= max_weight:
            out.append((boundary, i, m))
            break
        jmax = max(i + 1, int(np.searchsorted(W, W[i] + max_weight, side="right")) - 1)
        js = np.arange(i + 1, jmax + 1)  # run = cells[i:j]; j < m because the rest exceeds max_weight
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


# -- node objects ---------------------------------------------------------------


@dataclass
class Node:
    """A decoded node: where each cell's bytes sit inside the node object."""

    depth: int
    max_depth: int
    coords: tuple[int, ...]
    codes: np.ndarray  # (m,) sorted codes of the cells within the root
    offsets: np.ndarray  # (m + 1,) absolute positions in ``data``
    data: memoryview

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def root(self) -> tuple[int, ...]:
        return tuple(c >> self.depth for c in self.coords)

    def cell(self, code: int) -> bytes | None:
        i = int(np.searchsorted(self.codes, np.uint64(code)))
        if i == len(self.codes) or int(self.codes[i]) != code:
            return None
        return bytes(self.data[self.offsets[i] : self.offsets[i + 1]])

    def absolute(self) -> np.ndarray:
        """``(m, ndim)`` absolute coordinates of the cells in the node."""
        base = np.array(self.root, dtype=np.int64) << self.max_depth
        return unmorton(self.codes, len(self.coords), self.max_depth) + base

    def cells(self) -> dict[tuple[int, ...], bytes]:
        """Every cell in the node, keyed by absolute cell coordinates."""
        off = self.offsets
        return {tuple(int(x) for x in row): bytes(self.data[off[i] : off[i + 1]]) for i, row in enumerate(self.absolute())}


def encode_node(
    depth: int,
    coords: tuple[int, ...],
    codes: np.ndarray,
    payloads: list[bytes],
    *,
    max_depth: int,
) -> bytes:
    """Serialise one node.

    Layout (little endian): header, node coords ``int64[ndim]``, cell codes
    ``uint64[n]`` (sorted), offsets ``uint64[n + 1]`` relative to the start of
    the payload, then the concatenated cell bytes in code order.  A node's
    bytes depend only on its contents.
    """
    codes = np.asarray(codes, dtype=np.uint64).reshape(len(payloads))
    order = np.argsort(codes, kind="stable")
    payloads = [payloads[i] for i in order]
    offsets = np.zeros(len(payloads) + 1, dtype="<u8")
    offsets[1:] = np.cumsum([len(p) for p in payloads])
    return b"".join(
        [
            _NODE_HEADER.pack(NODE_MAGIC, FORMAT_VERSION, len(coords), depth, max_depth, len(payloads)),
            np.asarray(coords, dtype="<i8").tobytes(),
            codes[order].astype("<u8").tobytes(),
            offsets.tobytes(),
            *payloads,
        ]
    )


def decode_node(buf: bytes) -> Node:
    view = memoryview(buf)
    magic, version, ndim, depth, max_depth, n = _NODE_HEADER.unpack_from(view, 0)
    if magic != NODE_MAGIC or version != FORMAT_VERSION:
        raise ValueError("not a zarr-octree node object of this format version")
    pos = _NODE_HEADER.size
    coords = tuple(int(c) for c in np.frombuffer(view, "<i8", ndim, pos))
    pos += 8 * ndim
    codes = np.frombuffer(view, "<u8", n, pos)
    pos += 8 * n
    offsets = np.frombuffer(view, "<u8", n + 1, pos).astype(np.int64) + pos + 8 * (n + 1)
    return Node(depth, max_depth, coords, codes, offsets, view)


# -- manifest -------------------------------------------------------------------


@dataclass
class NodeInfo:
    nbytes: int
    ncells: int
    first: int = 0  # code of the first and last occupied cell
    last: int = 0
    weight: float = 0.0


def encode_manifest(ndim: int, max_depth: int, nodes: dict[NodeKey, NodeInfo]) -> bytes:
    """Serialise the manifest: one row per stored node.

    Layout: header, depth ``uint8[n]``, node coords ``int64[n, ndim]``, first
    and last occupied code ``uint64[n]`` each, node size ``uint64[n]``, cell
    count ``uint32[n]``, weight ``float64[n]``.  It lets a reader resolve any
    cell to its node, or to "empty", without touching storage.
    """
    keys = sorted(nodes)
    col = lambda f, dt: np.array([f(nodes[k]) for k in keys], dtype=dt)
    return b"".join(
        [
            _MANIFEST_HEADER.pack(MANIFEST_MAGIC, FORMAT_VERSION, ndim, max_depth, len(keys)),
            np.array([k[0] for k in keys], dtype="u1").tobytes(),
            np.array([k[1] for k in keys], dtype="<i8").reshape(len(keys), ndim).tobytes(),
            col(lambda i: i.first, "<u8").tobytes(),
            col(lambda i: i.last, "<u8").tobytes(),
            col(lambda i: i.nbytes, "<u8").tobytes(),
            col(lambda i: i.ncells, "<u4").tobytes(),
            col(lambda i: i.weight, "<f8").tobytes(),
        ]
    )


def decode_manifest(buf: bytes) -> tuple[int, int, dict[NodeKey, NodeInfo]]:
    view = memoryview(buf)
    magic, version, ndim, max_depth, n = _MANIFEST_HEADER.unpack_from(view, 0)
    if magic != MANIFEST_MAGIC or version != FORMAT_VERSION:
        raise ValueError("not a zarr-octree manifest of this format version")
    pos = _MANIFEST_HEADER.size

    def take(dt: str, count: int) -> np.ndarray:
        nonlocal pos
        a = np.frombuffer(view, dt, count, pos)
        pos += a.nbytes
        return a

    depth = take("u1", n)
    coords = take("<i8", n * ndim).reshape(n, ndim)
    first, last, size, count, weight = take("<u8", n), take("<u8", n), take("<u8", n), take("<u4", n), take("<f8", n)
    nodes = {
        (int(depth[i]), tuple(int(c) for c in coords[i])): NodeInfo(
            int(size[i]), int(count[i]), int(first[i]), int(last[i]), float(weight[i])
        )
        for i in range(n)
    }
    return ndim, max_depth, nodes
