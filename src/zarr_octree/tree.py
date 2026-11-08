"""Octree partition of a sparse chunk grid, and the on-disk node and manifest formats.

Pure numpy and ``struct``: no zarr, no I/O.

Terms used throughout:

* **cell** -- one chunk of the logical Zarr array, addressed by its chunk
  coordinates.  Zarr reads and writes cells; it never sees nodes.
* **node** -- one stored object.  A node at depth ``d`` covers a block of
  ``2 ** (max_depth - d)`` cells along every axis and holds the encoded bytes
  of every occupied cell in that block.  Depth 0 nodes are the roots.
* **partition** -- the set of stored nodes.  Each occupied cell is in exactly
  one node, so no stored node is an ancestor of another.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

NODE_MAGIC = b"ZOCN"
MANIFEST_MAGIC = b"ZOCM"
FORMAT_VERSION = 1

_NODE_HEADER = struct.Struct("<4sBBBBI")  # magic, version, ndim, depth, max_depth, n_cells
_MANIFEST_HEADER = struct.Struct("<4sBBBxQ")  # magic, version, ndim, max_depth, n_nodes

NodeKey = tuple[int, tuple[int, ...]]  # (depth, node coords at that depth)


def build_partition(
    coords: np.ndarray,
    nbytes: np.ndarray,
    max_depth: int,
    target_bytes: int,
) -> list[tuple[int, tuple[int, ...], np.ndarray]]:
    """Split occupied cells into nodes of at most ``target_bytes`` each.

    Top-down: every root holding more than ``target_bytes`` is split into its
    ``2 ** ndim`` children, recursively, until a node fits or reaches
    ``max_depth`` (one cell per node, which may then exceed the target).

    Args:
        coords: ``(n, ndim)`` integer cell coordinates, all occupied.
        nbytes: ``(n,)`` encoded size of each cell.
        max_depth: Depth of the finest node; a root spans ``2 ** max_depth`` cells per axis.
        target_bytes: Largest node that is not split further.

    Returns:
        ``[(depth, node_coords, cell_indices), ...]``, one entry per node, where
        ``cell_indices`` index into ``coords``.
    """
    coords = np.asarray(coords, dtype=np.int64)
    nbytes = np.asarray(nbytes, dtype=np.int64)
    if coords.ndim != 2 or len(coords) != len(nbytes):
        raise ValueError("coords must be (n, ndim) and match nbytes")
    if (coords < 0).any():
        raise ValueError("cell coordinates must be non-negative")
    out: list[tuple[int, tuple[int, ...], np.ndarray]] = []
    pending = np.arange(len(coords))
    for depth in range(max_depth + 1):
        if len(pending) == 0:
            break
        node = coords[pending] >> (max_depth - depth)
        uniq, inverse = np.unique(node, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        size = np.bincount(inverse, weights=nbytes[pending], minlength=len(uniq))
        keep = (size <= target_bytes) | (depth == max_depth)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(uniq) + 1))
        for g in np.flatnonzero(keep):
            members = pending[order[bounds[g] : bounds[g + 1]]]
            out.append((depth, tuple(int(c) for c in uniq[g]), members))
        pending = pending[~keep[inverse]]
    return out


def node_of(cell: tuple[int, ...], depth: int, max_depth: int) -> tuple[int, ...]:
    """Coordinates of the depth-``depth`` node that contains ``cell``."""
    shift = max_depth - depth
    return tuple(c >> shift for c in cell)


def root_of(key: NodeKey) -> tuple[int, ...]:
    """Coordinates of the root that contains node ``key``."""
    depth, coords = key
    return tuple(c >> depth for c in coords)


@dataclass
class Node:
    """A decoded node: where each cell's bytes sit inside the node object.

    Cells are looked up by their C-order linear index within the node, which
    the stored (sorted) local coordinates already follow, so a lookup is one
    binary search and decoding does no per-cell Python work.
    """

    depth: int
    max_depth: int
    coords: tuple[int, ...]
    local: np.ndarray  # (m, ndim) local cell coords, C order
    offsets: np.ndarray  # (m + 1,) absolute positions in ``data``
    data: memoryview

    def __post_init__(self) -> None:
        self._side = 1 << (self.max_depth - self.depth)
        if len(self.local):
            self._lin = np.ravel_multi_index(self.local.T.astype(np.int64), (self._side,) * len(self.coords))
        else:
            self._lin = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.local)

    def cell(self, local: tuple[int, ...]) -> bytes | None:
        lin = 0
        for c in local:
            if not 0 <= c < self._side:
                return None
            lin = lin * self._side + c
        i = int(np.searchsorted(self._lin, lin))
        if i == len(self._lin) or self._lin[i] != lin:
            return None
        return bytes(self.data[self.offsets[i] : self.offsets[i + 1]])

    def absolute(self) -> np.ndarray:
        """``(m, ndim)`` absolute coordinates of the cells in the node."""
        base = np.array(self.coords, dtype=np.int64) << (self.max_depth - self.depth)
        return self.local.astype(np.int64) + base

    def cells(self) -> dict[tuple[int, ...], bytes]:
        """Every cell in the node, keyed by absolute cell coordinates."""
        off = self.offsets
        return {tuple(int(x) for x in row): bytes(self.data[off[i] : off[i + 1]]) for i, row in enumerate(self.absolute())}


def encode_node(
    depth: int,
    coords: tuple[int, ...],
    local: np.ndarray,
    payloads: list[bytes],
    *,
    max_depth: int,
) -> bytes:
    """Serialise one node.

    Layout (little endian): header, node coords ``int64[ndim]``, local cell
    coords ``uint32[n, ndim]``, offsets ``uint64[n + 1]`` relative to the
    start of the payload, then the concatenated cell bytes.  Cells are stored
    in C order of their local coordinates, so a node's bytes depend only on
    its contents.
    """
    local = np.asarray(local, dtype=np.uint32).reshape(len(payloads), len(coords))
    order = np.lexsort(local.T[::-1]) if len(local) else np.zeros(0, dtype=np.intp)
    local = local[order]
    payloads = [payloads[i] for i in order]
    offsets = np.zeros(len(payloads) + 1, dtype="<u8")
    offsets[1:] = np.cumsum([len(p) for p in payloads])
    return b"".join(
        [
            _NODE_HEADER.pack(NODE_MAGIC, FORMAT_VERSION, len(coords), depth, max_depth, len(payloads)),
            np.asarray(coords, dtype="<i8").tobytes(),
            local.astype("<u4").tobytes(),
            offsets.tobytes(),
            *payloads,
        ]
    )


def decode_node(buf: bytes) -> Node:
    view = memoryview(buf)
    magic, version, ndim, depth, max_depth, n = _NODE_HEADER.unpack_from(view, 0)
    if magic != NODE_MAGIC or version != FORMAT_VERSION:
        raise ValueError("not a zarr-octree node object")
    pos = _NODE_HEADER.size
    coords = tuple(int(c) for c in np.frombuffer(view, "<i8", ndim, pos))
    pos += 8 * ndim
    local = np.frombuffer(view, "<u4", n * ndim, pos).reshape(n, ndim)
    pos += 4 * n * ndim
    offsets = np.frombuffer(view, "<u8", n + 1, pos).astype(np.int64) + pos + 8 * (n + 1)
    return Node(depth, max_depth, coords, local, offsets, view)


@dataclass
class NodeInfo:
    nbytes: int
    ncells: int


def encode_manifest(ndim: int, max_depth: int, nodes: dict[NodeKey, NodeInfo]) -> bytes:
    """Serialise the occupancy manifest: one row per stored node.

    Layout: header, depth ``uint8[n]``, node coords ``int64[n, ndim]``, node
    size ``uint64[n]``, cell count ``uint32[n]``.  It is what lets a reader
    resolve any cell to its node, or to "empty", without touching storage.
    """
    keys = sorted(nodes)
    depth = np.array([k[0] for k in keys], dtype="u1")
    coords = np.array([k[1] for k in keys], dtype="<i8").reshape(len(keys), ndim)
    size = np.array([nodes[k].nbytes for k in keys], dtype="<u8")
    count = np.array([nodes[k].ncells for k in keys], dtype="<u4")
    return b"".join(
        [
            _MANIFEST_HEADER.pack(MANIFEST_MAGIC, FORMAT_VERSION, ndim, max_depth, len(keys)),
            depth.tobytes(),
            coords.tobytes(),
            size.tobytes(),
            count.tobytes(),
        ]
    )


def decode_manifest(buf: bytes) -> tuple[int, int, dict[NodeKey, NodeInfo]]:
    view = memoryview(buf)
    magic, version, ndim, max_depth, n = _MANIFEST_HEADER.unpack_from(view, 0)
    if magic != MANIFEST_MAGIC or version != FORMAT_VERSION:
        raise ValueError("not a zarr-octree manifest")
    pos = _MANIFEST_HEADER.size
    depth = np.frombuffer(view, "u1", n, pos)
    pos += n
    coords = np.frombuffer(view, "<i8", n * ndim, pos).reshape(n, ndim)
    pos += 8 * n * ndim
    size = np.frombuffer(view, "<u8", n, pos)
    pos += 8 * n
    count = np.frombuffer(view, "<u4", n, pos)
    nodes = {
        (int(depth[i]), tuple(int(c) for c in coords[i])): NodeInfo(int(size[i]), int(count[i]))
        for i in range(n)
    }
    return ndim, max_depth, nodes
