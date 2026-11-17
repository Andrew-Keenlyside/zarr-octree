"""The format rules a reader enforces: index checks, metadata, and stores from zarr-octree <= 0.3."""

import json

import numpy as np
import pytest
import zarr
from cells import as_object_array, point_cells, write_cells
from zarr.codecs import BytesCodec, VLenBytesCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.registry import get_codec_class
from zarr.storage import MemoryStore

from zarr_octree.codec import (
    _FOOT,
    FOOTER_MAGIC,
    INDEX_VERSION,
    InvalidIndex,
    OctreeShardingCodec,
    ShardIndex,
    encode_index,
    index_from_shard,
)

CPS = (3, 3, 3)  # bits = 2: codes address a 4 x 4 x 4 block, one row of which lies outside the shard


def _index(codes, lengths, run_start, cps=CPS):
    lengths = np.array(lengths, np.uint64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.uint64)
    return ShardIndex(cps, np.array(codes, np.uint64), offsets, lengths, np.array(run_start), np.ones(len(run_start) - 1))


def _shard(index, version=INDEX_VERSION):
    return b"\x01" * int(index.lengths.sum()) + encode_index(index, version)


def _with_crc(body: bytes) -> bytes:
    """An index body made by hand, with a valid footer (so only the body's content is wrong)."""
    import google_crc32c

    return body + _FOOT.pack(len(body), google_crc32c.value(body), FOOTER_MAGIC)


def test_a_valid_index_round_trips_in_both_versions():
    ix = _index([0, 1, 7], [4, 2, 7], [0, 2, 3])
    for version in (2, 3):
        got = index_from_shard(_shard(ix, version), CPS)
        assert got.codes.tolist() == [0, 1, 7] and got.runs() == [(0, 6, 1.0), (6, 13, 1.0)]


@pytest.mark.parametrize(
    "codes,lengths,run_start,why",
    [
        ([5, 1], [4, 4], [0, 2], "not sorted"),
        ([1, 1], [4, 4], [0, 2], "not sorted"),
        ([0, 63], [4, 4], [0, 2], "outside the shard"),  # (3, 3, 3): inside the 4-block, outside the shard
        ([0, 1], [4, 0], [0, 2], "length 0"),
        ([0, 1], [4, 4], [0, 1], "partition"),  # the run table ends before the last chunk
        ([0, 1], [4, 4], [1, 2], "partition"),  # ... or starts after the first
        ([0, 1, 2], [4, 4, 4], [0, 2, 2, 3], "partition"),  # an empty run
    ],
)
def test_an_inconsistent_index_is_refused(codes, lengths, run_start, why):
    with pytest.raises(InvalidIndex, match=why):
        index_from_shard(_shard(_index(codes, lengths, run_start)), CPS)


def test_counts_that_disagree_with_the_index_size_are_refused():
    good = encode_index(_index([0, 1], [4, 4], [0, 2]))
    body = bytearray(good[: -_FOOT.size])
    body[8:12] = (3).to_bytes(4, "little")  # claims 3 chunks; the arrays hold 2
    with pytest.raises(InvalidIndex, match="need"):
        index_from_shard(b"\x01" * 8 + _with_crc(bytes(body)), CPS)


def test_a_data_region_of_the_wrong_size_is_refused():
    index = encode_index(_index([0, 1], [4, 4], [0, 2]))
    with pytest.raises(InvalidIndex, match="data region"):
        index_from_shard(b"\x01" * 9 + index, CPS)  # one byte more than the chunks hold
    assert index_from_shard(b"\x01" * 9 + index, CPS, whole=False).codes.tolist() == [0, 1]  # a tail can't tell


def test_a_damaged_index_fails_its_checksum():
    shard = bytearray(_shard(_index([0, 1], [4, 4], [0, 2])))
    shard[8 + 20] ^= 1  # a byte of the codes
    with pytest.raises(ValueError, match="checksum"):
        index_from_shard(bytes(shard), CPS)


def test_an_unknown_index_version_is_refused():
    body = bytearray(encode_index(_index([0], [4], [0, 1]))[: -_FOOT.size])
    body[4] = 9
    with pytest.raises(InvalidIndex, match="version 9"):
        index_from_shard(b"\x01" * 4 + _with_crc(bytes(body)), CPS)


# -- metadata ---------------------------------------------------------------------


def test_both_names_are_registered_with_zarr():
    assert get_codec_class("octree_sharding") is OctreeShardingCodec
    assert get_codec_class("zarr_octree_sharding") is OctreeShardingCodec


def test_written_metadata_has_the_registered_name_and_form():
    c = OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()], target=500)
    assert c.to_dict() == {
        "name": "octree_sharding",
        "configuration": {
            "chunk_shape": (2, 2),
            "codecs": (BytesCodec().to_dict(),),
            "layout": {"name": "octree", "configuration": {"target": 500.0, "unit": "bytes"}},
        },
    }


def test_layout_is_optional():
    c = OctreeShardingCodec.from_dict({"name": "octree_sharding", "configuration": {"chunk_shape": [2, 2], "codecs": [BytesCodec().to_dict()]}})
    assert (c.layout, c.target) == ("octree", 100_000)


@pytest.mark.parametrize(
    "configuration,why",
    [
        ({"index_codecs": []}, "unknown"),
        ({"index_location": "start"}, "unknown"),
        ({"version": 3}, "unknown"),
        ({"layout": "octree"}, "layout must be"),
        ({"layout": {"name": "grid"}}, "layout must be"),
        ({"layout": {"name": "octree", "configuration": {"max_weight": 5}}}, "unknown 'octree' layout"),
    ],
)
def test_metadata_the_format_does_not_define_is_refused(configuration, why):
    cfg = {"chunk_shape": [2, 2], "codecs": [BytesCodec().to_dict()], **configuration}
    with pytest.raises(ValueError, match=why):
        OctreeShardingCodec.from_dict({"name": "octree_sharding", "configuration": cfg})


def test_a_store_written_by_zarr_octree_0_3_still_reads():
    """Legacy name, flat configuration and version-2 (CRC-32) indexes."""
    cells = point_cells()
    store = MemoryStore()
    codec = OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=[VLenBytesCodec(), ZstdCodec()], layout="runs", min_weight=0, max_weight=8_000)
    arr = zarr.create_array(store, name="pts", shape=(32, 32, 32), chunks=(16, 16, 16), serializer=codec, compressors=None, dtype="bytes")
    write_cells(arr, cells)
    proto = default_buffer_prototype()

    async def rewrite():
        meta = json.loads((await store.get("pts/zarr.json", proto)).to_bytes())
        cfg = meta["codecs"][0]["configuration"]
        layout = cfg.pop("layout")["configuration"]
        meta["codecs"][0] = {"name": "zarr_octree_sharding", "configuration": {"version": 2, **cfg, **layout}}
        await store.set("pts/zarr.json", proto.buffer.from_bytes(json.dumps(meta).encode()))
        async for key in store.list_prefix("pts/c/"):
            blob = (await store.get(key, proto)).to_bytes()
            index = index_from_shard(blob, (16, 16, 16))
            data = blob[: int(index.lengths.sum())]
            await store.set(key, proto.buffer.from_bytes(data + encode_index(index, version=2)))

    sync(rewrite())
    old = zarr.open_array(store, path="pts", mode="r")
    assert old.metadata.codecs[0].layout == "runs"
    np.testing.assert_array_equal(old[...], as_object_array(cells))


def test_written_metadata_matches_the_registry_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "registry" / "codecs" / "octree_sharding"
    schema = json.loads((root / "schema.json").read_text())
    for codec in (
        OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec(), ZstdCodec()]),
        OctreeShardingCodec(chunk_shape=(1, 1, 1), codecs=[VLenBytesCodec()], layout="runs", min_weight=5, max_weight=50),
    ):
        jsonschema.validate(json.loads(json.dumps(codec.to_dict())), schema)
    example = json.loads((root / "example.zarr" / "zarr.json").read_text())["codecs"][0]
    jsonschema.validate(example, schema)
    for bad in ({"index_location": "end"}, {"layout": {"name": "octree", "configuration": {"max_weight": 1}}}):
        meta = {"name": "octree_sharding", "configuration": {"chunk_shape": [2], "codecs": [], **bad}}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(meta, schema)


def test_an_empty_index_reads_as_the_fill_value():
    """A shard object with m = 0, r = 0 (index and footer only) is valid and holds nothing."""
    store = MemoryStore()
    arr = zarr.create_array(store, name="a", shape=(4, 4), chunks=(4, 4), serializer=OctreeShardingCodec(chunk_shape=(2, 2), codecs=[BytesCodec()]),
                            compressors=None, dtype="uint8", fill_value=7)
    empty = _index([], [], [0], cps=(2, 2))
    assert index_from_shard(encode_index(empty), (2, 2)).codes.tolist() == []
    sync(store.set("a/c/0/0", default_buffer_prototype().buffer.from_bytes(encode_index(empty))))
    assert (arr[...] == 7).all()


def test_run_weights_are_not_checked():
    ix = _index([0, 1], [4, 4], [0, 2])
    ix.run_weight[:] = np.nan
    assert index_from_shard(_shard(ix), CPS).codes.tolist() == [0, 1]
