"""``OctreeStore``: a Zarr v3 store that packs an array's chunks into octree nodes.

Zarr above this store sees an ordinary array on a regular chunk grid: it
reads and writes chunk keys ``<array>/c/i/j/k`` exactly as it would on any
store.  Underneath, the chunks of every octree array are grouped into node
objects ``<array>/o/<depth>/i/j/k`` -- runs of the octree's Z-order whose
weight lies between ``min_weight`` and ``max_weight`` -- plus one manifest
``<array>/o/manifest``.  Dense regions get runs of few (small) chunks and
sparse regions runs of many, so every stored object holds a similar amount of
data whatever the density of the region it covers.

Any other array or group in the store passes through untouched.

Writes to an octree array are buffered in memory and packed on
:meth:`OctreeStore.flush` (or on leaving a ``with`` block): packing needs the
size of every chunk in a region before it can decide where to split.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.sync import sync

from zarr_octree.tree import (
    FORMAT_VERSION,
    Node,
    NodeInfo,
    NodeKey,
    block_of,
    boundary_of,
    build_runs,
    decode_manifest,
    decode_node,
    encode_manifest,
    encode_node,
    morton,
    root_of,
)

EXTENSION = "zarr_octree"
META = "zarr.json"
NODE_DIR = "o"
MANIFEST = "o/manifest"


@dataclass(frozen=True)
class OctreeConfig:
    """How one array is packed.

    Args:
        max_depth: A root spans ``2 ** max_depth`` chunks per axis; nodes never
            cross a root, so an edit rewrites at most one root.
        max_weight: A node weighs at most this, unless it is a single chunk.
        min_weight: A node weighs at least this where its root allows (see
            :mod:`zarr_octree.tree`).
        unit: What the weights count.  ``"bytes"``: the encoded chunk plus its
            16-byte entry in the node index.  Any other name needs ``weigh``.
        weigh: ``weigh(encoded_chunk) -> weight`` for a custom ``unit`` (the
            vertices in a chunk, say).  Writers only; not stored.
    """

    max_depth: int = 5
    max_weight: float = 1 << 20
    min_weight: float = 0
    unit: str = "bytes"
    weigh: Callable[[bytes], float] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.max_depth <= 21:
            raise ValueError(f"max_depth must be in [0, 21], got {self.max_depth}")
        if self.max_weight <= 0 or not 0 <= self.min_weight <= self.max_weight:
            raise ValueError("need 0 <= min_weight <= max_weight and max_weight > 0")

    def to_json(self) -> dict[str, Any]:
        return {"version": FORMAT_VERSION, "max_depth": self.max_depth, "min_weight": self.min_weight,
                "max_weight": self.max_weight, "unit": self.unit}


@dataclass
class _Array:
    """What the store knows about one octree array."""

    path: str
    ndim: int
    sep: str
    config: OctreeConfig
    nodes: dict[NodeKey, NodeInfo] | None = None  # None until the manifest is read
    # Per root: the code where each node starts (sorted) and the nodes in that order.
    by_root: dict[tuple[int, ...], tuple[np.ndarray, list[NodeKey]]] = field(default_factory=dict)
    buffer: dict[tuple[int, ...], bytes | None] = field(default_factory=dict)  # None = deleted
    buffer_bytes: int = 0

    @property
    def max_depth(self) -> int:
        return self.config.max_depth

    def set_nodes(self, nodes: dict[NodeKey, NodeInfo]) -> None:
        self.nodes = nodes
        grouped: dict[tuple[int, ...], list[tuple[int, NodeKey]]] = {}
        for key in nodes:
            grouped.setdefault(root_of(key), []).append((boundary_of(key, self.max_depth), key))
        self.by_root = {}
        for root, items in grouped.items():
            items.sort()
            self.by_root[root] = (np.array([b for b, _ in items], dtype=np.uint64), [k for _, k in items])

    def root(self, cell: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(c >> self.max_depth for c in cell)

    def code(self, cell: tuple[int, ...]) -> int:
        """The cell's Morton code within its root (scalar; per-byte tables, as :func:`morton`)."""
        n, mask = self.ndim, (1 << self.max_depth) - 1
        table = _spread_table(n)
        code = 0
        for a, c in enumerate(cell):
            c &= mask
            spread, shift = 0, 0
            while c:
                spread |= table[c & 255] << shift
                c >>= 8
                shift += 8 * n
            code |= spread << (n - 1 - a)
        return code

    def owner(self, cell: tuple[int, ...]) -> NodeKey | None:
        """The node whose range of codes contains ``cell`` (occupied or not)."""
        entry = self.by_root.get(self.root(cell))
        if entry is None:
            return None
        bounds, keys = entry
        i = int(np.searchsorted(bounds, np.uint64(self.code(cell)), side="right")) - 1
        return keys[i] if i >= 0 else None

    def find(self, cell: tuple[int, ...]) -> NodeKey | None:
        """The stored node that may hold ``cell``, or ``None`` if the manifest rules it out."""
        assert self.nodes is not None
        key = self.owner(cell)
        if key is None:
            return None
        info = self.nodes[key]
        return key if info.first <= self.code(cell) <= info.last else None

    def weigh(self, data: bytes) -> float:
        if self.config.unit == "bytes":
            return len(data) + 16  # the cell's code and offset in the node index
        if self.config.weigh is None:
            raise ValueError(f"{self.path!r}: writing weights in {self.config.unit!r} needs OctreeConfig(weigh=...)")
        return float(self.config.weigh(data))

    def join(self, rel: str) -> str:
        return f"{self.path}/{rel}" if self.path else rel

    def node_path(self, key: NodeKey) -> str:
        depth, coords = key
        return self.join(f"{NODE_DIR}/{depth}/" + "/".join(map(str, coords)))

    def cell_key(self, cell: tuple[int, ...]) -> str:
        return self.join("c" + self.sep + self.sep.join(map(str, cell)))


@functools.cache
def _spread_table(ndim: int) -> list[int]:
    """``table[v]``: the 8 bits of ``v`` spread ``ndim`` apart."""
    return [sum(((v >> b) & 1) << (b * ndim) for b in range(8)) for v in range(256)]


class _NodeCache:
    """LRU cache of decoded nodes, bounded by their size in bytes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.used = 0
        self._nodes: OrderedDict[tuple[str, NodeKey], Node] = OrderedDict()

    def get(self, ck: tuple[str, NodeKey]) -> Node | None:
        node = self._nodes.get(ck)
        if node is not None:
            self._nodes.move_to_end(ck)
        return node

    def put(self, ck: tuple[str, NodeKey], node: Node) -> None:
        self.drop(ck)
        self._nodes[ck] = node
        self.used += len(node.data)
        while self.used > self.max_bytes and len(self._nodes) > 1:
            _, old = self._nodes.popitem(last=False)
            self.used -= len(old.data)

    def drop(self, ck: tuple[str, NodeKey]) -> None:
        old = self._nodes.pop(ck, None)
        if old is not None:
            self.used -= len(old.data)

    def drop_array(self, path: str) -> None:
        for ck in [ck for ck in self._nodes if ck[0] == path]:
            self.drop(ck)

    def clear(self) -> None:
        self._nodes.clear()
        self.used = 0


def _slice(data: bytes, byte_range: ByteRequest | None) -> bytes:
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[len(data) - byte_range.suffix :]
    raise TypeError(f"unexpected byte range {byte_range!r}")


def _parse_cell(parts: list[str], ndim: int) -> tuple[int, ...] | None:
    if len(parts) != ndim or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


class OctreeStore(Store):
    """Wrap ``store`` so that chosen arrays are stored as octree nodes.

    Args:
        store: The store that holds the bytes (local, memory, object storage...).
        arrays: Arrays to create as octree arrays, ``{path: OctreeConfig}``.
            Arrays already marked as octree arrays in ``store`` need no entry.
        cache_bytes: Size of the in-memory cache of decoded nodes.
        max_buffer_bytes: Pack an array's buffered writes once they exceed this.
        read_only: Refuse writes.
    """

    def __init__(
        self,
        store: Store,
        *,
        arrays: dict[str, OctreeConfig] | None = None,
        cache_bytes: int = 256 << 20,
        max_buffer_bytes: int = 1 << 30,
        read_only: bool = False,
    ) -> None:
        super().__init__(read_only=read_only)
        self._store = store
        self._configured = {k.strip("/"): v for k, v in (arrays or {}).items()}
        self._arrays: dict[str, _Array | None] = {}  # None = known not to be an octree array
        self._cache = _NodeCache(cache_bytes)
        self.max_buffer_bytes = max_buffer_bytes
        self._inflight: dict[Any, asyncio.Future[Any]] = {}

    # -- Store plumbing -----------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OctreeStore) and self._store == other._store

    def __repr__(self) -> str:
        return f"OctreeStore({self._store!r})"

    async def _open(self) -> None:
        if not self._store._is_open:
            await self._store._open()
        self._is_open = True

    def with_read_only(self, read_only: bool = False) -> OctreeStore:
        """A view with another read-only setting that shares this store's state.

        zarr asks for one when an array is opened with ``mode="r"``; sharing
        means the view sees writes still buffered in this store.
        """
        view = copy.copy(self)
        view._store = self._store.with_read_only(read_only)
        view._read_only = read_only
        view._is_open = False
        return view

    @property
    def supports_writes(self) -> bool:
        return self._store.supports_writes

    @property
    def supports_deletes(self) -> bool:
        return self._store.supports_deletes

    @property
    def supports_listing(self) -> bool:
        return self._store.supports_listing

    def __enter__(self) -> OctreeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        if exc[0] is None:
            self.flush()

    def close(self) -> None:
        self._store.close()
        super().close()

    async def _once(self, key: Any, make: Any) -> Any:
        """Run ``make()`` once for concurrent callers asking for the same ``key``."""
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(make())
            self._inflight[key] = task
            task.add_done_callback(lambda _t: self._inflight.pop(key, None))
        return await task

    # -- array metadata -----------------------------------------------------

    def configure(self, path: str, config: OctreeConfig) -> None:
        """Make the array that will be created at ``path`` an octree array."""
        self._configured[path.strip("/")] = config

    def _register(self, path: str, doc: dict[str, Any], config: OctreeConfig) -> _Array:
        known = self._arrays.get(path)
        if known is not None:
            return known
        enc = doc.get("chunk_key_encoding", {"name": "default"})
        name = enc if isinstance(enc, str) else enc.get("name")
        if name != "default":
            raise NotImplementedError(f"{path!r}: only the default chunk key encoding is supported")
        sep = "/" if isinstance(enc, str) else enc.get("configuration", {}).get("separator", "/")
        ndim = len(doc["shape"])
        if ndim == 0:
            raise ValueError(f"{path!r}: a 0-d array has nothing to partition")
        if ndim * config.max_depth > 64:
            raise ValueError(f"{path!r}: {ndim}-d codes of max_depth {config.max_depth} do not fit in 64 bits")
        state = _Array(path=path, ndim=ndim, sep=sep, config=config)
        self._arrays[path] = state
        return state

    async def _array(self, path: str) -> _Array | None:
        """The octree array at ``path``, or ``None`` if there is none."""
        if path in self._arrays:
            return self._arrays[path]
        buf = await self._store.get(f"{path}/{META}" if path else META, default_buffer_prototype())
        if buf is None:
            return None  # not cached: the array may be created later
        doc = json.loads(buf.to_bytes())
        if EXTENSION in doc:
            return self._register(path, doc, self._config_from(path, doc))
        self._arrays[path] = None
        return None

    def _config_from(self, path: str, doc: dict[str, Any]) -> OctreeConfig:
        for t in doc.get("storage_transformers", ()):
            if isinstance(t, dict) and t.get("name") == EXTENSION:
                cfg = dict(t.get("configuration", {}))
                if cfg.pop("version", FORMAT_VERSION) != FORMAT_VERSION:
                    raise ValueError("unsupported zarr_octree format version")
                stored = OctreeConfig(**cfg)
                given = self._configured.get(path)
                if given is not None and given.weigh is not None and given.unit == stored.unit:
                    stored = OctreeConfig(**cfg, weigh=given.weigh)  # a writer's weigher for the stored unit
                return stored
        raise ValueError("zarr_octree field without a zarr_octree storage transformer")

    @staticmethod
    def _strip(doc: dict[str, Any]) -> dict[str, Any]:
        """The metadata zarr itself should see: the extension fields removed."""
        doc = {k: v for k, v in doc.items() if k != EXTENSION}
        rest = [t for t in doc.get("storage_transformers", ()) if not (isinstance(t, dict) and t.get("name") == EXTENSION)]
        if rest:
            doc["storage_transformers"] = rest
        else:
            doc.pop("storage_transformers", None)
        return doc

    @staticmethod
    def _inject(doc: dict[str, Any], config: OctreeConfig) -> dict[str, Any]:
        """Mark the stored metadata as an octree array.

        ``storage_transformers`` carries the configuration, where the Zarr v3
        spec puts storage-layer extensions.  The ``must_understand`` field is
        there because zarr-python ignores storage transformers: without it, a
        reader without this extension would open the array and silently see
        every chunk as missing.  With it, such a reader refuses the array.
        """
        doc = OctreeStore._strip(doc)
        doc["storage_transformers"] = [
            *doc.get("storage_transformers", ()),
            {"name": EXTENSION, "configuration": config.to_json()},
        ]
        doc[EXTENSION] = {"must_understand": True, "version": FORMAT_VERSION}
        return doc

    async def _route(self, key: str) -> tuple[_Array, tuple[int, ...]] | None:
        """``(array, chunk coords)`` if ``key`` is a chunk of an octree array."""
        parts = key.split("/")
        last = parts[-1]
        if last.startswith("c."):
            state = await self._array("/".join(parts[:-1]))
            if state is not None and state.sep == ".":
                cell = _parse_cell(last.split(".")[1:], state.ndim)
                if cell is not None:
                    return state, cell
        for i in range(len(parts) - 2, -1, -1):
            if parts[i] == "c":
                state = await self._array("/".join(parts[:i]))
                if state is not None and state.sep == "/":
                    cell = _parse_cell(parts[i + 1 :], state.ndim)
                    if cell is not None:
                        return state, cell
        return None

    def _is_physical(self, key: str) -> bool:
        """Whether ``key`` is a node or manifest object of an octree array."""
        for path, state in self._arrays.items():
            if state is not None and key.startswith(state.join(NODE_DIR + "/")):
                return True
        return False

    # -- reading ------------------------------------------------------------

    async def _load(self, state: _Array) -> None:
        if state.nodes is not None:
            return

        async def read() -> None:
            buf = await self._store.get(state.join(MANIFEST), default_buffer_prototype())
            if state.nodes is None:
                state.set_nodes({} if buf is None else decode_manifest(buf.to_bytes())[2])

        await self._once(("manifest", state.path), read)

    async def _node(self, state: _Array, key: NodeKey) -> Node:
        ck = (state.path, key)
        node = self._cache.get(ck)
        if node is not None:
            return node

        async def read() -> Node:
            buf = await self._store.get(state.node_path(key), default_buffer_prototype())
            if buf is None:
                raise KeyError(f"manifest lists {state.node_path(key)} but it is missing")
            node = decode_node(buf.to_bytes())
            self._cache.put(ck, node)
            return node

        return await self._once(("node", ck), read)

    async def _read_cell(self, state: _Array, cell: tuple[int, ...]) -> bytes | None:
        if cell in state.buffer:
            return state.buffer[cell]
        await self._load(state)
        key = state.find(cell)
        if key is None:
            return None
        node = await self._node(state, key)
        return node.cell(state.code(cell))

    async def aget_cells(self, path: str, cells: Iterable[tuple[int, ...]]) -> list[bytes | None]:
        """The encoded bytes of each chunk in ``cells`` (``None`` if absent), without zarr.

        An optional fast path for readers that decode chunks themselves: each
        node is fetched once, concurrently, and no per-chunk zarr machinery
        runs.  Zarr's own reads are unaffected.
        """
        state = await self._array(path.strip("/"))
        if state is None:
            raise KeyError(f"no octree array at {path!r}")
        return list(await asyncio.gather(*(self._read_cell(state, tuple(int(x) for x in c)) for c in cells)))

    def get_cells(self, path: str, cells: Iterable[tuple[int, ...]]) -> list[bytes | None]:
        """Synchronous :meth:`aget_cells`."""
        return sync(self.aget_cells(path, cells))

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        await self._ensure_open()
        if key == META or key.endswith("/" + META):
            buf = await self._store.get(key, prototype)
            if buf is None:
                return None
            doc = json.loads(buf.to_bytes())
            if EXTENSION not in doc:
                return buf if byte_range is None else prototype.buffer.from_bytes(_slice(buf.to_bytes(), byte_range))
            path = key[: -len(META)].rstrip("/")
            self._register(path, doc, self._config_from(path, doc))
            data = json.dumps(self._strip(doc), indent=2).encode()
            return prototype.buffer.from_bytes(_slice(data, byte_range))
        routed = await self._route(key)
        if routed is None:
            return await self._store.get(key, prototype, byte_range)
        data = await self._read_cell(*routed)
        return None if data is None else prototype.buffer.from_bytes(_slice(data, byte_range))

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def _get_many(self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]):  # type: ignore[override]
        requests = list(requests)
        results = await asyncio.gather(*(self.get(*r) for r in requests))
        for req, res in zip(requests, results):
            yield req[0], res

    async def exists(self, key: str) -> bool:
        routed = await self._route(key)
        if routed is None:
            return await self._store.exists(key)
        return (await self._read_cell(*routed)) is not None

    # -- writing ------------------------------------------------------------

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        await self._ensure_open()
        if key == META or key.endswith("/" + META):
            path = key[: -len(META)].rstrip("/")
            doc = json.loads(value.to_bytes())
            config = self._configured.get(path)
            known = self._arrays.get(path)
            if known is not None:
                config = known.config
            if config is not None and doc.get("node_type") == "array":
                doc = self._inject(doc, config)
                self._register(path, doc, config)
                value = default_buffer_prototype().buffer.from_bytes(json.dumps(doc, indent=2).encode())
            else:
                self._arrays[path] = None
            await self._store.set(key, value)
            return
        routed = await self._route(key)
        if routed is None:
            await self._store.set(key, value)
            return
        state, cell = routed
        data = value.to_bytes()
        self._buffer(state, cell, data)
        if state.buffer_bytes > self.max_buffer_bytes:
            await self._flush(state)

    def _buffer(self, state: _Array, cell: tuple[int, ...], data: bytes | None) -> None:
        old = state.buffer.get(cell)
        state.buffer_bytes += (0 if data is None else len(data)) - (0 if old is None else len(old))
        state.buffer[cell] = data

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        if not await self.exists(key):
            await self.set(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        await asyncio.gather(*(self.set(k, v) for k, v in values))

    async def delete(self, key: str) -> None:
        self._check_writable()
        routed = await self._route(key)
        if routed is None:
            await self._store.delete(key)
            return
        self._buffer(*routed, None)

    async def delete_dir(self, prefix: str) -> None:
        self._check_writable()
        prefix = prefix.rstrip("/")
        for state in [s for s in self._arrays.values() if s is not None]:
            inside = state.join("c")
            if prefix == inside or prefix.startswith(inside + "/"):
                async for k in self._logical_keys(state):
                    if k == prefix or k.startswith(prefix + "/"):
                        routed = await self._route(k)
                        if routed is not None:
                            self._buffer(*routed, None)
                return
        await self._store.delete_dir(prefix)
        for path in list(self._arrays):
            if not prefix or path == prefix or path.startswith(prefix + "/"):
                del self._arrays[path]
                self._cache.drop_array(path)

    async def clear(self) -> None:
        await self._store.clear()
        self._arrays.clear()
        self._cache.clear()

    async def is_empty(self, prefix: str) -> bool:
        if any(s is not None and s.buffer for s in self._arrays.values()):
            async for _ in self.list_prefix(prefix):
                return False
            return True
        return await self._store.is_empty(prefix)

    # -- packing ------------------------------------------------------------

    def flush(self) -> None:
        """Pack every buffered write into nodes (synchronous)."""
        sync(self.aflush())

    async def aflush(self) -> None:
        for state in [s for s in self._arrays.values() if s is not None]:
            await self._flush(state)

    def repack(self, path: str) -> None:
        """Rebuild every node of the array at ``path`` from scratch.

        Edits only rewrite the nodes they touch and never merge nodes that
        have shrunk; a repack restores the partition a fresh write would give.
        """

        async def run() -> None:
            state = await self._array(path.strip("/"))
            if state is None:
                raise KeyError(f"no octree array at {path!r}")
            await self._load(state)
            assert state.nodes is not None
            await self._flush(state, rebuild=set(state.by_root))

        sync(run())

    async def _flush(self, state: _Array, rebuild: set[tuple[int, ...]] | None = None) -> None:
        if not state.buffer and not rebuild:
            return
        self._check_writable()
        await self._load(state)
        assert state.nodes is not None
        updates, state.buffer, state.buffer_bytes = state.buffer, {}, 0
        D = state.max_depth
        per_root: dict[tuple[int, ...], dict[tuple[int, ...], bytes | None]] = {r: {} for r in rebuild or ()}
        for cell, data in updates.items():
            per_root.setdefault(state.root(cell), {})[cell] = data

        writes: dict[NodeKey, tuple[bytes, NodeInfo]] = {}
        removed: set[NodeKey] = set()
        for root, ups in per_root.items():
            local = None if rebuild and root in rebuild else await self._patch(state, ups)
            if local is not None:
                for key, cells in local.items():
                    if cells:
                        writes[key] = self._encode(state, key, cells)
                    else:
                        removed.add(key)
                continue
            old = set(state.by_root.get(root, (None, []))[1])
            cells: dict[tuple[int, ...], bytes] = {}
            for node in await asyncio.gather(*(self._node(state, k) for k in old)):
                cells.update(node.cells())
            for cell, data in ups.items():
                if data is None:
                    cells.pop(cell, None)
                else:
                    cells[cell] = data
            new = self._partition(state, cells)
            writes.update(new)
            removed |= old - set(new)

        # Nodes first, then the manifest that points at them, then the nodes it no longer lists.
        await asyncio.gather(
            *(self._store.set(state.node_path(k), default_buffer_prototype().buffer.from_bytes(b)) for k, (b, _) in writes.items())
        )
        for key, (_, info) in writes.items():
            state.nodes[key] = info
            self._cache.drop((state.path, key))
        for key in removed - set(writes):
            state.nodes.pop(key, None)
            self._cache.drop((state.path, key))
        state.set_nodes(state.nodes)
        manifest = encode_manifest(state.ndim, D, state.nodes)
        await self._store.set(state.join(MANIFEST), default_buffer_prototype().buffer.from_bytes(manifest))
        await asyncio.gather(*(self._store.delete(state.node_path(k)) for k in removed - set(writes)))

    async def _patch(
        self, state: _Array, ups: dict[tuple[int, ...], bytes | None]
    ) -> dict[NodeKey, dict[tuple[int, ...], bytes]] | None:
        """Apply ``ups`` by rewriting only the nodes whose ranges hold those cells.

        Returns the new cell contents of each touched node, or ``None`` when
        the root has no nodes yet or a node outgrows ``max_weight``; the
        caller then rebuilds the whole root.
        """
        assert state.nodes is not None
        touched: dict[NodeKey, dict[tuple[int, ...], bytes | None]] = {}
        for cell, data in ups.items():
            key = state.owner(cell)
            if key is None:
                if data is None:
                    continue  # deleting a cell that was never stored
                return None
            touched.setdefault(key, {})[cell] = data
        out: dict[NodeKey, dict[tuple[int, ...], bytes]] = {}
        keys = list(touched)
        for key, node in zip(keys, await asyncio.gather(*(self._node(state, k) for k in keys))):
            cells = node.cells()
            for cell, data in touched[key].items():
                if data is None:
                    cells.pop(cell, None)
                else:
                    cells[cell] = data
            if len(cells) > 1 and sum(state.weigh(v) for v in cells.values()) > state.config.max_weight:
                return None
            out[key] = cells
        return out

    def _encode(self, state: _Array, key: NodeKey, cells: dict[tuple[int, ...], bytes]) -> tuple[bytes, NodeInfo]:
        depth, coords = key
        cs = list(cells)
        codes = np.array([state.code(c) for c in cs], dtype=np.uint64)
        blob = encode_node(depth, coords, codes, [cells[c] for c in cs], max_depth=state.max_depth)
        weight = float(sum(state.weigh(cells[c]) for c in cs))
        return blob, NodeInfo(len(blob), len(cs), int(codes.min()), int(codes.max()), weight)

    def _partition(self, state: _Array, cells: dict[tuple[int, ...], bytes]) -> dict[NodeKey, tuple[bytes, NodeInfo]]:
        """Cut one root's cells into runs and encode them."""
        if not cells:
            return {}
        D = state.max_depth
        cs = list(cells)
        root = np.array(state.root(cs[0]), dtype=np.int64)
        local = np.array(cs, dtype=np.int64).reshape(len(cs), state.ndim) - (root << D)
        codes = morton(local, D)
        order = np.argsort(codes)
        weights = np.array([state.weigh(cells[cs[i]]) for i in order])
        out: dict[NodeKey, tuple[bytes, NodeInfo]] = {}
        cfg = state.config
        for boundary, a, b in build_runs(codes[order], weights, state.ndim, cfg.min_weight, cfg.max_weight):
            depth, at = block_of(boundary, state.ndim, D)
            key = (depth, tuple(int(x) for x in (root << depth) + at))
            out[key] = self._encode(state, key, {cs[i]: cells[cs[i]] for i in order[a:b]})
        return out

    # -- listing ------------------------------------------------------------

    async def _logical_keys(self, state: _Array) -> AsyncIterator[str]:
        await self._load(state)
        assert state.nodes is not None
        present: set[tuple[int, ...]] = set()
        keys = list(state.nodes)
        for node in await asyncio.gather(*(self._node(state, k) for k in keys)):
            present.update(tuple(int(x) for x in row) for row in node.absolute())
        for cell, data in state.buffer.items():
            if data is None:
                present.discard(cell)
            else:
                present.add(cell)
        for cell in sorted(present):
            yield state.cell_key(cell)

    async def list(self) -> AsyncIterator[str]:  # type: ignore[override]
        inner = [k async for k in self._store.list()]
        for k in inner:
            if k == META or k.endswith("/" + META):
                await self._array(k[: -len(META)].rstrip("/"))
        for k in inner:
            if not self._is_physical(k):
                yield k
        for state in [s for s in self._arrays.values() if s is not None]:
            async for k in self._logical_keys(state):
                yield k

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:  # type: ignore[override]
        async for k in self.list():
            if k.startswith(prefix):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:  # type: ignore[override]
        prefix = prefix.rstrip("/")
        start = prefix + "/" if prefix else ""
        seen: set[str] = set()
        async for k in self.list_prefix(start):
            head = k[len(start) :].split("/", 1)[0]
            if head not in seen:
                seen.add(head)
                yield head

    # -- inspection ---------------------------------------------------------

    def nodes(self, path: str) -> dict[NodeKey, NodeInfo]:
        """The stored nodes of the octree array at ``path`` (after a flush)."""

        async def run() -> dict[NodeKey, NodeInfo]:
            state = await self._array(path.strip("/"))
            if state is None:
                raise KeyError(f"no octree array at {path!r}")
            await self._load(state)
            assert state.nodes is not None
            return dict(state.nodes)

        return sync(run())


def convert(
    source: Store,
    path: str,
    target: Store,
    config: OctreeConfig,
    *,
    target_path: str | None = None,
) -> OctreeStore:
    """Copy the regular array at ``source/path`` into ``target`` as an octree array.

    The chunk bytes are copied as they are, so codecs, sharding and dtype are
    unchanged.  Returns the ``OctreeStore`` over ``target``.
    """
    path = path.strip("/")
    dest = (path if target_path is None else target_path).strip("/")
    out = OctreeStore(target, arrays={dest: config})

    async def run() -> None:
        meta = await source.get(f"{path}/{META}" if path else META, default_buffer_prototype())
        if meta is None:
            raise KeyError(f"no array at {path!r}")
        await out.set(f"{dest}/{META}" if dest else META, meta)
        prefix = f"{path}/c" if path else "c"
        keys = [k async for k in source.list_prefix(prefix)]
        for i in range(0, len(keys), 4096):
            batch = keys[i : i + 4096]
            bufs = await asyncio.gather(*(source.get(k, default_buffer_prototype()) for k in batch))
            for k, b in zip(batch, bufs):
                if b is not None:
                    rel = k[len(path) :].lstrip("/") if path else k
                    await out.set(f"{dest}/{rel}" if dest else rel, b)
        await out.aflush()

    sync(run())
    return out
