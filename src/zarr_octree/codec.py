"""``octree``: sharding with a sparse index and octree runs.

A drop-in alternative to Zarr v3's ``sharding_indexed``: the array, its shards,
inner chunks and keys are unchanged; only the bytes inside a shard differ
(format in ``SPEC.md``).  Inner chunks are stored in Z-order and grouped into
**runs** -- the leaves of an octree aimed at ``target`` (``layout="octree"``,
the default; :func:`zarr_octree.tree.build_octree`) or Z-order runs between
``min_weight`` and ``max_weight`` (``layout="runs"``;
:func:`zarr_octree.tree.build_runs`).  The index is sparse (one row per stored
chunk) and a partial read fetches one byte range per run it touches.

The codec subclasses zarr's ``ShardingCodec`` so zarr sees an ordinary sharded
array; it overrides the async entry points and opts out of zarr's synchronous
sharding path, which assumes the dense index.  Weights are encoded bytes
(``unit="bytes"``) or any unit with a weigher (:func:`register_weigher`).
"""

from __future__ import annotations

import asyncio
import struct
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import google_crc32c
import numpy as np
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.codecs.sharding import (
    ShardingCodec,
    _ShardIndex,
    _ShardingByteGetter,
    _ShardingByteSetter,
    _ShardReader,
)
from zarr.core.buffer import Buffer, default_buffer_prototype, numpy_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.indexing import BasicIndexer, get_indexer
from zarr.registry import register_codec

from zarr_octree.tree import build_octree, build_runs, morton

CODEC_NAME = "octree"
LEGACY_NAMES = ("zarr_octree_sharding",)  # zarr-octree <= 0.3; read, never written
INDEX_MAGIC, FOOTER_MAGIC = b"ZOSI", b"ZOSF"
INDEX_VERSION = 3  # CRC-32C over the index; version 2 (zarr-octree <= 0.3) used CRC-32 and is still read
READ_VERSIONS = (2, 3)
_HEAD = struct.Struct("<4sBBBBII")  # magic, version, ndim, bits, length width, n_chunks, n_runs
_FOOT = struct.Struct("<QI4s")  # index_nbytes, crc32, magic
INDEX_READ_MIN = 4 << 10

LAYOUTS = ("octree", "runs")
DEFAULT_TARGET = 100_000

_WEIGHERS: dict[str, Callable[[bytes], float]] = {}


def register_weigher(unit: str, weigh: Callable[[bytes], float]) -> None:
    """Make ``unit`` usable as a weight: ``weigh(encoded inner chunk) -> weight``."""
    _WEIGHERS[unit] = weigh


def _bits(chunks_per_shard: tuple[int, ...]) -> int:
    return max(1, max(int(c) - 1 for c in chunks_per_shard).bit_length())


def _code_dtype(ndim: int, bits: int) -> str:
    n = ndim * bits
    return "<u2" if n <= 16 else "<u4" if n <= 32 else "<u8"


def index_read_nbytes(chunks_per_shard: tuple[int, ...]) -> int:
    """Bytes a reader takes from the end of a shard to get its index in one request.

    Room for every possible chunk at 6 bytes (8 above 32³ chunks per shard)
    plus a run table of a quarter as many runs; a larger index (very many
    runs, or chunks of 4 GiB and more) costs a second read.
    """
    n = int(np.prod(chunks_per_shard))
    row = np.dtype(_code_dtype(len(chunks_per_shard), _bits(chunks_per_shard))).itemsize + 4
    return max(INDEX_READ_MIN, n * row + (n // 4) * 12 + 4 + _HEAD.size + _FOOT.size)  # run starts are r + 1 entries


@dataclass
class ShardIndex:
    """A decoded sparse index: the stored inner chunks and the runs they form."""

    chunks_per_shard: tuple[int, ...]
    codes: np.ndarray  # (n,) sorted Z-order codes of the stored inner chunks
    offsets: np.ndarray  # (n,)
    lengths: np.ndarray  # (n,)
    run_start: np.ndarray  # (r + 1,) first chunk of each run
    run_weight: np.ndarray  # (r,)
    _coords: np.ndarray | None = None

    @property
    def coords(self) -> np.ndarray:
        """``(n, ndim)`` coordinates of the stored inner chunks within the shard (decoded on first use)."""
        if self._coords is None:
            from zarr_octree.tree import unmorton

            ndim = len(self.chunks_per_shard)
            self._coords = unmorton(self.codes, ndim, _bits(self.chunks_per_shard)) if len(self.codes) else np.zeros((0, ndim), np.int64)
        return self._coords

    def dense(self) -> _ShardIndex:
        """The same index in zarr's dense form, for zarr's own sharding helpers."""
        index = _ShardIndex.create_empty(self.chunks_per_shard)
        if len(self.coords):
            index.offsets_and_lengths[tuple(self.coords.T)] = np.stack([self.offsets, self.lengths], 1)
        return index

    def run_of(self) -> np.ndarray:
        """``(n,)`` run of each stored chunk."""
        return np.repeat(np.arange(len(self.run_weight)), np.diff(self.run_start))

    def runs(self) -> list[tuple[int, int, float]]:
        """``[(byte start, byte stop, weight)]`` of each run."""
        out = []
        for r in range(len(self.run_weight)):
            a, b = int(self.run_start[r]), int(self.run_start[r + 1])
            out.append((int(self.offsets[a]), int(self.offsets[b - 1] + self.lengths[b - 1]), float(self.run_weight[r])))
        return out


def _crc(version: int, body: bytes) -> int:
    return zlib.crc32(body) if version == 2 else google_crc32c.value(body)


def encode_index(index: ShardIndex, version: int = INDEX_VERSION) -> bytes:
    """The index and footer for ``index`` (``version=2`` only to make legacy shards for tests)."""
    if version not in READ_VERSIONS:
        raise ValueError(f"unknown index version {version}")
    n, r = len(index.codes), len(index.run_weight)
    ndim, bits = len(index.chunks_per_shard), _bits(index.chunks_per_shard)
    if not np.array_equal(index.offsets, np.concatenate([[0], np.cumsum(index.lengths)[:-1]]) if n else index.offsets):
        raise ValueError("an octree shard stores its chunks contiguously from byte 0")
    width = 4 if n == 0 or int(index.lengths.max()) < 1 << 32 else 8
    body = b"".join([
        _HEAD.pack(INDEX_MAGIC, version, ndim, bits, width, n, r),
        index.codes.astype(_code_dtype(ndim, bits)).tobytes(),
        index.lengths.astype(f"<u{width}").tobytes(),
        index.run_start.astype("<u4").tobytes(),
        index.run_weight.astype("<f8").tobytes(),
    ])
    return body + _FOOT.pack(len(body), _crc(version, body), FOOTER_MAGIC)


class InvalidIndex(ValueError):  # noqa: N818 -- reads as "raise InvalidIndex"
    """A shard index that is not a valid ``octree`` index (raised as ``ValueError``)."""


def _bad(why: str) -> InvalidIndex:
    return InvalidIndex(f"invalid octree shard index: {why}")


def decode_index(body: bytes, chunks_per_shard: tuple[int, ...]) -> ShardIndex:
    """Decode and check an index body (between the shard's data and its footer).

    Every check the format requires of a reader is made here, so a damaged or
    hostile index raises ``ValueError`` instead of returning wrong chunks.
    """
    if len(body) < _HEAD.size:
        raise _bad(f"{len(body)} bytes, shorter than its {_HEAD.size}-byte header")
    magic, version, ndim, bits, width, n, r = _HEAD.unpack_from(body, 0)
    if magic != INDEX_MAGIC:
        raise _bad("index magic")
    if version not in READ_VERSIONS:
        raise _bad(f"unsupported version {version}")
    if ndim != len(chunks_per_shard):
        raise _bad(f"written for {ndim} dimensions, the array has {len(chunks_per_shard)}")
    if bits != _bits(chunks_per_shard):
        raise _bad("written for another shard shape")
    if width not in (4, 8):
        raise _bad(f"length width {width}")
    code_dt = np.dtype(_code_dtype(ndim, bits))
    want = _HEAD.size + n * (code_dt.itemsize + width) + (r + 1) * 4 + r * 8
    if len(body) != want:
        raise _bad(f"{len(body)} bytes, its counts ({n} chunks, {r} runs) need {want}")
    pos = _HEAD.size

    def take(dt: Any, count: int) -> np.ndarray:
        nonlocal pos
        a = np.frombuffer(body, dt, count, pos)
        pos += a.nbytes
        return a

    codes = take(code_dt, n).astype(np.uint64)
    lengths = take(np.dtype(f"<u{width}"), n).astype(np.uint64)
    run_start, run_weight = take(np.dtype("<u4"), r + 1).astype(np.int64), take(np.dtype("<f8"), r)
    if n and not (codes[1:] > codes[:-1]).all():
        raise _bad("codes are not sorted and unique")
    if n and not (lengths > 0).all():
        raise _bad("a stored chunk has length 0")
    if (n == 0) != (r == 0):
        raise _bad(f"{n} chunks in {r} runs")
    if run_start[0] != 0 or run_start[-1] != n or (r and not (run_start[1:] > run_start[:-1]).all()):
        raise _bad("run table does not partition the chunks")
    offsets = np.zeros(n, np.uint64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    index = ShardIndex(tuple(chunks_per_shard), codes, offsets, lengths, run_start, run_weight)
    if n and (index.coords >= np.asarray(chunks_per_shard)).any():
        raise _bad("a code lies outside the shard")
    return index


def _split_footer(tail: bytes) -> tuple[int, int]:
    """``(index_nbytes, crc)`` from the last bytes of a shard."""
    if len(tail) < _FOOT.size:
        raise ValueError(f"not an octree shard: {len(tail)} bytes, shorter than its {_FOOT.size}-byte footer")
    nbytes, crc, magic = _FOOT.unpack_from(tail, len(tail) - _FOOT.size)
    if magic != FOOTER_MAGIC:
        raise ValueError("not an octree shard (footer magic)")
    return nbytes, crc


def index_from_shard(buf: bytes, chunks_per_shard: tuple[int, ...], *, whole: bool = True) -> ShardIndex:
    """The index at the end of ``buf``: a whole shard, or (``whole=False``) any tail of one holding the index."""
    nbytes, crc = _split_footer(buf)
    if nbytes > len(buf) - _FOOT.size:
        raise ValueError(f"octree shard is truncated: its index needs {nbytes + _FOOT.size} bytes, {len(buf)} given")
    body = bytes(buf[len(buf) - _FOOT.size - nbytes : len(buf) - _FOOT.size])
    version = body[4] if len(body) > 4 else INDEX_VERSION
    if version in READ_VERSIONS and _crc(version, body) != crc:
        raise ValueError("octree shard index checksum mismatch")
    index = decode_index(body, chunks_per_shard)
    data = len(buf) - _FOOT.size - nbytes
    if whole and int(index.lengths.sum()) != data:
        raise _bad(f"its chunks hold {int(index.lengths.sum())} bytes, the shard's data region {data}")
    return index


@dataclass(frozen=True)
class OctreeShardingCodec(ShardingCodec):
    """``sharding_indexed`` with a sparse index and octree runs.

    Args:
        chunk_shape: Inner chunk shape (as for ``ShardingCodec``).
        codecs: Inner chunk codecs (as for ``ShardingCodec``).
        layout: ``"octree"`` (leaves of an octree aimed at ``target``) or
            ``"runs"`` (Z-order runs between ``min_weight`` and ``max_weight``).
            Left out, it is ``"runs"`` when either bound is given, else ``"octree"``.
        target: The weight the octree layout aims each run at (no hard bound).
        min_weight, max_weight: Bounds on each run's weight, for the runs layout.
        unit: ``"bytes"`` (encoded inner-chunk bytes) or a registered weigher's name.
    """

    layout: str = "octree"
    target: float = DEFAULT_TARGET
    min_weight: float = 0
    max_weight: float = 1 << 20
    unit: str = "bytes"

    def __init__(
        self,
        *,
        chunk_shape: Any,
        codecs: Iterable[Any] = (),
        index_codecs: Any = None,  # accepted for dataclasses.replace; the index format is fixed
        index_location: Any = None,
        layout: str | None = None,
        target: float = DEFAULT_TARGET,
        min_weight: float | None = None,
        max_weight: float | None = None,
        unit: str = "bytes",
    ) -> None:
        kwargs = {"chunk_shape": chunk_shape}
        if codecs:
            kwargs["codecs"] = codecs
        super().__init__(**kwargs)
        if layout is None:
            layout = "runs" if min_weight is not None or max_weight is not None else "octree"
        min_weight = 0 if min_weight is None else min_weight
        max_weight = 1 << 20 if max_weight is None else max_weight
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        if target <= 0:
            raise ValueError("target must be positive")
        if max_weight <= 0 or not 0 <= min_weight <= max_weight:
            raise ValueError("need 0 <= min_weight <= max_weight and max_weight > 0")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "target", float(target))
        object.__setattr__(self, "min_weight", float(min_weight))
        object.__setattr__(self, "max_weight", float(max_weight))
        object.__setattr__(self, "unit", str(unit))

    # -- metadata ---------------------------------------------------------------

    def layout_dict(self) -> dict[str, Any]:
        """The writer's layout as metadata: ``{"name": "octree" | "runs", "configuration": {...}}``."""
        cfg: dict[str, Any] = (
            {"target": self.target} if self.layout == "octree"
            else {"min_weight": self.min_weight, "max_weight": self.max_weight}
        )
        return {"name": self.layout, "configuration": {**cfg, "unit": self.unit}}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": CODEC_NAME,
            "configuration": {
                "chunk_shape": tuple(self.chunk_shape),
                "codecs": tuple(c.to_dict() for c in self.codecs),
                "layout": self.layout_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OctreeShardingCodec:
        name = data.get("name")
        if name != CODEC_NAME and name not in LEGACY_NAMES:
            raise ValueError(f"expected codec {CODEC_NAME!r}, got {name!r}")
        cfg = dict(data.get("configuration", {}))
        if name in LEGACY_NAMES:
            return cls(**_legacy_configuration(cfg))
        extra = set(cfg) - {"chunk_shape", "codecs", "layout"}
        if extra:
            # index_codecs / index_location included: the index format is fixed.
            raise ValueError(f"unknown octree codec configuration {sorted(extra)}")
        if "chunk_shape" not in cfg:
            raise ValueError("octree codec configuration needs chunk_shape")
        kwargs: dict[str, Any] = {"chunk_shape": cfg["chunk_shape"], "codecs": cfg.get("codecs", ())}
        layout = cfg.get("layout")
        if layout is not None:  # writer settings; reading never depends on them
            if not isinstance(layout, dict) or layout.get("name") not in LAYOUTS:
                raise ValueError(f"octree layout must be {{'name': one of {LAYOUTS}, 'configuration': ...}}, got {layout!r}")
            lcfg = dict(layout.get("configuration", {}))
            allowed = {"target", "unit"} if layout["name"] == "octree" else {"min_weight", "max_weight", "unit"}
            if set(lcfg) - allowed:
                raise ValueError(f"unknown {layout['name']!r} layout configuration {sorted(set(lcfg) - allowed)}")
            kwargs.update(layout=layout["name"], **lcfg)
        return cls(**kwargs)

    def __getstate__(self) -> dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, state: dict[str, Any]) -> None:
        other = type(self).from_dict(state)
        for k in ("chunk_shape", "codecs", "index_codecs", "index_location", "layout", "target", "min_weight", "max_weight", "unit"):
            object.__setattr__(self, k, getattr(other, k))
        for k in ("_get_index_chunk_spec", "_get_chunks_per_shard", "_shard_index_size"):
            if k in other.__dict__:
                object.__setattr__(self, k, other.__dict__[k])

    @property
    def _sync_capable(self) -> bool:
        # zarr >= 3.4's synchronous sharding path reads a dense index; use the async methods below.
        return False

    def _inner_pipeline(self, shard_spec: Any) -> Any:
        get = getattr(self, "_get_inner_pipeline", None)
        return get(shard_spec) if get is not None else self.codec_pipeline

    def _weigh(self, value: Buffer) -> float:
        if self.unit == "bytes":
            return float(len(value))
        weigh = _WEIGHERS.get(self.unit)
        if weigh is None:
            raise ValueError(f"no weigher registered for unit {self.unit!r}; see register_weigher")
        return float(weigh(value.to_bytes()))

    # -- index ------------------------------------------------------------------

    async def _load_index(self, byte_getter: Any, chunks_per_shard: tuple[int, ...]) -> ShardIndex | None:
        """The shard's index: one suffix read, a second only for an index larger than :func:`index_read_nbytes`."""
        tail = await byte_getter.get(prototype=numpy_buffer_prototype(), byte_range=SuffixByteRequest(index_read_nbytes(chunks_per_shard)))
        if tail is None:
            return None
        tail = tail.to_bytes()
        nbytes, _ = _split_footer(tail)
        need = nbytes + _FOOT.size
        if need > len(tail):
            more = await byte_getter.get(prototype=numpy_buffer_prototype(), byte_range=SuffixByteRequest(need))
            if more is None:  # the shard was deleted between the two reads
                return None
            tail = more.to_bytes()
        return index_from_shard(tail[len(tail) - need :], chunks_per_shard, whole=False)

    async def _load_shard_index_maybe(self, byte_getter: Any, chunks_per_shard: tuple[int, ...]) -> _ShardIndex | None:
        index = await self._load_index(byte_getter, chunks_per_shard)
        return None if index is None else index.dense()

    async def _load_shard_index(self, byte_getter: Any, chunks_per_shard: tuple[int, ...]) -> _ShardIndex:
        return (await self._load_shard_index_maybe(byte_getter, chunks_per_shard)) or _ShardIndex.create_empty(chunks_per_shard)

    def _reader(self, buf: Buffer, chunks_per_shard: tuple[int, ...]) -> _ShardReader:
        reader = _ShardReader()
        reader.buf = buf
        reader.index = index_from_shard(buf.to_bytes(), chunks_per_shard).dense()
        return reader

    async def _load_full_shard_maybe(self, byte_getter: Any, prototype: Any, chunks_per_shard: tuple[int, ...]) -> _ShardReader | None:
        buf = await byte_getter.get(prototype=prototype)
        return self._reader(buf, chunks_per_shard) if buf else None

    def _shard_index_size(self, chunks_per_shard: tuple[int, ...]) -> int:
        raise NotImplementedError("octree shard indexes have no fixed size")

    def compute_encoded_size(self, input_byte_length: int, shard_spec: Any) -> int:
        # Variable: the index grows with the stored chunks.  A lower bound, as zarr only uses it for fixed-size codecs.
        return input_byte_length + _HEAD.size + _FOOT.size

    # -- decoding ---------------------------------------------------------------

    def _indexer(self, shard_spec: Any, selection: Any = None) -> Any:
        shape = shard_spec.shape
        grid = ChunkGrid.from_sizes(shape, self.chunk_shape)
        if selection is None:
            return BasicIndexer(tuple(slice(0, s) for s in shape), shape=shape, chunk_grid=grid)
        return get_indexer(selection, shape=shape, chunk_grid=grid)

    async def _read_into(self, shard_spec: Any, shard_dict: Any, indexer: Any, out: Any) -> None:
        chunk_spec = self._get_chunk_spec(shard_spec)
        await self._inner_pipeline(shard_spec).read(
            [
                (_ShardingByteGetter(shard_dict, coords), chunk_spec, chunk_sel, out_sel, complete)
                for coords, chunk_sel, out_sel, complete in indexer
            ],
            out,
        )

    async def _decode_single(self, shard_bytes: Buffer, shard_spec: Any) -> Any:
        cps = self._get_chunks_per_shard(shard_spec)
        out = shard_spec.prototype.nd_buffer.empty(
            shape=shard_spec.shape, dtype=shard_spec.dtype.to_native_dtype(), order=shard_spec.order
        )
        reader = self._reader(shard_bytes, cps)
        if reader.index.is_all_empty():
            out.fill(shard_spec.fill_value)
            return out
        await self._read_into(shard_spec, reader, self._indexer(shard_spec), out)
        return out

    async def _decode_partial_single(self, byte_getter: Any, selection: Any, shard_spec: Any) -> Any:
        cps = self._get_chunks_per_shard(shard_spec)
        indexer = self._indexer(shard_spec, selection)
        out = shard_spec.prototype.nd_buffer.empty(
            shape=indexer.shape, dtype=shard_spec.dtype.to_native_dtype(), order=shard_spec.order
        )
        projections = list(indexer)
        wanted = {coords for coords, *_ in projections}
        index = await self._load_index(byte_getter, cps)
        if index is None:
            return None
        shard_dict = await self._fetch_chunks(byte_getter, index, wanted)
        await self._read_into(shard_spec, shard_dict, projections, out)
        return out.reshape(indexer.sel_shape) if hasattr(indexer, "sel_shape") else out

    async def _fetch_chunks(self, byte_getter: Any, index: ShardIndex, wanted: set[tuple[int, ...]]) -> dict:
        """The stored chunks among ``wanted``: one byte range per run touched, fetched concurrently.

        Within a run the range spans only from the first to the last wanted
        chunk, so it reads at most one run per request.
        """
        if not wanted or not len(index.codes):
            return {}
        asked = sorted(wanted)
        want = morton(np.array(asked, dtype=np.int64), _bits(index.chunks_per_shard))
        coords_of = dict(zip(want.tolist(), asked))
        at = np.minimum(np.searchsorted(index.codes, want), len(index.codes) - 1)
        hit = np.unique(at[index.codes[at] == want])
        if not len(hit):
            return {}
        runs = index.run_of()[hit]
        spans = []
        for r in np.unique(runs):
            members = hit[runs == r]
            a, b = members[0], members[-1]
            spans.append((members, int(index.offsets[a]), int(index.offsets[b] + index.lengths[b])))
        bufs = await asyncio.gather(
            *(byte_getter.get(prototype=default_buffer_prototype(), byte_range=RangeByteRequest(lo, hi)) for _, lo, hi in spans)
        )
        out = {}
        for (members, lo, hi), buf in zip(spans, bufs):
            if buf is None:
                continue
            if len(buf) != hi - lo:
                raise _bad(f"a run's byte range {lo}-{hi} lies past the end of the shard")
            for i in members:
                s = int(index.offsets[i]) - lo
                out[coords_of[int(index.codes[i])]] = buf[s : s + int(index.lengths[i])]
        return out

    async def read_runs(self, byte_getter: Any, chunks_per_shard: tuple[int, ...]) -> list[tuple[ShardIndex, int, dict]]:
        """Every run of one shard as a unit: ``[(index, run, {inner coords: encoded chunk})]``.

        The node-level read for whole-run consumers; not used by zarr itself.
        """
        index = await self._load_index(byte_getter, chunks_per_shard)
        if index is None:
            return []
        spans = index.runs()
        bufs = await asyncio.gather(
            *(byte_getter.get(prototype=default_buffer_prototype(), byte_range=RangeByteRequest(a, b)) for a, b, _ in spans)
        )
        out = []
        for r, ((lo, hi, _), buf) in enumerate(zip(spans, bufs)):
            if buf is None or len(buf) != hi - lo:
                raise _bad(f"run {r}'s byte range {lo}-{hi} lies past the end of the shard")
            cells = {}
            for i in range(int(index.run_start[r]), int(index.run_start[r + 1])):
                s = int(index.offsets[i]) - lo
                cells[tuple(int(x) for x in index.coords[i])] = buf[s : s + int(index.lengths[i])]
            out.append((index, r, cells))
        return out

    # -- encoding ---------------------------------------------------------------

    async def _write_into(self, shard_spec: Any, shard_dict: dict, indexer: Any, shard_array: Any) -> None:
        chunk_spec = self._get_chunk_spec(shard_spec)
        await self._inner_pipeline(shard_spec).write(
            [
                (_ShardingByteSetter(shard_dict, coords), chunk_spec, chunk_sel, out_sel, complete)
                for coords, chunk_sel, out_sel, complete in indexer
            ],
            shard_array,
        )

    @staticmethod
    def _empty_dict(chunks_per_shard: tuple[int, ...]) -> dict:
        # zarr's chunk setters delete by key, so every inner position must be present (as zarr's own builder does).
        return dict.fromkeys(np.ndindex(*chunks_per_shard))

    async def _encode_single(self, shard_array: Any, shard_spec: Any) -> Buffer | None:
        shard_dict = self._empty_dict(self._get_chunks_per_shard(shard_spec))
        await self._write_into(shard_spec, shard_dict, list(self._indexer(shard_spec)), shard_array)
        return self._assemble(shard_dict, self._get_chunks_per_shard(shard_spec))

    async def _encode_partial_single(self, byte_setter: Any, shard_array: Any, selection: Any, shard_spec: Any) -> None:
        cps = self._get_chunks_per_shard(shard_spec)
        indexer = list(self._indexer(shard_spec, selection))
        shard_dict = self._empty_dict(cps)
        if not self._is_complete_shard_write(indexer, cps):
            existing = await byte_setter.get(prototype=default_buffer_prototype())
            if existing:
                index = index_from_shard(existing.to_bytes(), cps)
                for c, o, n in zip(index.coords, index.offsets, index.lengths):
                    shard_dict[tuple(int(x) for x in c)] = existing[int(o) : int(o + n)]
        await self._write_into(shard_spec, shard_dict, indexer, shard_array)
        buf = self._assemble(shard_dict, cps)
        if buf is None:
            await byte_setter.delete()
        else:
            await byte_setter.set(buf)

    def _assemble(self, shard_dict: dict, chunks_per_shard: tuple[int, ...]) -> Buffer | None:
        """Lay the stored chunks out in Z-order, cut them into runs (per ``layout``), and append the index."""
        items = [(c, v) for c, v in shard_dict.items() if v is not None and len(v)]
        if not items:
            return None
        coords = np.array([c for c, _ in items], dtype=np.int64).reshape(len(items), len(chunks_per_shard))
        codes = morton(coords, _bits(chunks_per_shard))
        order = np.argsort(codes)
        coords, codes = coords[order], codes[order]
        values = [items[i][1] for i in order]
        lengths = np.array([len(v) for v in values], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        weights = np.array([self._weigh(v) for v in values])
        if self.layout == "octree":
            runs = build_octree(codes, weights, len(chunks_per_shard), _bits(chunks_per_shard), self.target)
        else:
            runs = build_runs(codes, weights, len(chunks_per_shard), self.min_weight, self.max_weight)
        run_start = np.array([a for _, a, _ in runs] + [len(values)], dtype=np.int64)
        run_weight = np.array([weights[a:b].sum() for _, a, b in runs])
        index = ShardIndex(tuple(chunks_per_shard), codes, offsets, lengths, run_start, run_weight, coords)
        template = default_buffer_prototype().buffer
        return template.create_zero_length().combine([*values, template.from_bytes(encode_index(index))])


def _legacy_configuration(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments from a ``zarr_octree_sharding`` (zarr-octree <= 0.3) configuration."""
    if cfg.pop("version", 2) != 2:
        raise ValueError("unsupported zarr_octree_sharding version")
    cfg.setdefault("layout", "runs")  # written before the octree layout existed
    return cfg


register_codec(CODEC_NAME, OctreeShardingCodec)
for _name in LEGACY_NAMES:
    register_codec(_name, OctreeShardingCodec)


# -- for readers that locate chunks themselves ----------------------------------


def index_tail_nbytes(tail: bytes) -> int:
    """How many bytes at the end of a shard hold its index and footer, from any tail of it."""
    return _split_footer(tail)[0] + _FOOT.size


def dense_rows(tail: bytes, chunks_per_shard: tuple[int, ...], *, runs: bool = False) -> np.ndarray:
    """The index at the end of ``tail`` as ``sharding_indexed``'s rows: ``(n_inner, 2)`` uint64
    (offset, nbytes) in C order of the inner chunks, ``2**64 - 1`` for an absent one.

    ``runs`` adds a third column, the run holding each chunk (``2**64 - 1`` for
    an absent one), so a reader can fetch a run's wanted chunks as one range.
    """
    need = index_tail_nbytes(tail)
    if need > len(tail):
        raise ValueError("tail is shorter than the shard's index")
    index = index_from_shard(tail[len(tail) - need :], chunks_per_shard, whole=False)
    rows = index.dense().offsets_and_lengths.reshape(-1, 2)
    if not runs:
        return rows
    run = np.full((len(rows), 1), np.iinfo(np.uint64).max, np.uint64)
    if len(index.codes):
        run[np.ravel_multi_index(tuple(index.coords.T), chunks_per_shard), 0] = index.run_of()
    return np.concatenate([rows, run], axis=1)
