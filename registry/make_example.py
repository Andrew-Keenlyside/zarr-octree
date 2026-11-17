"""Write the registry entry's example array and print its metadata and one shard, annotated.

    python registry/make_example.py   # writes registry/codecs/octree_sharding/example.zarr
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BytesCodec

from zarr_octree import OctreeShardingCodec
from zarr_octree.codec import _FOOT, _HEAD, index_from_shard

OUT = Path(__file__).parent / "codecs" / "octree_sharding" / "example.zarr"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    codec = OctreeShardingCodec(chunk_shape=(2, 2, 2), codecs=[BytesCodec()], target=8)
    arr = zarr.create_array(OUT, shape=(4, 4, 4), chunks=(4, 4, 4), serializer=codec, compressors=None,
                            dtype="uint8", fill_value=0)
    data = np.zeros((4, 4, 4), np.uint8)
    data[0:2, 0:2, 0:2] = 1   # inner chunk (0, 0, 0): Morton code 0
    data[0:2, 2:4, 0:2] = 2   # inner chunk (0, 1, 0): code 2
    data[2:4, 2:4, 2:4] = 3   # inner chunk (1, 1, 1): code 7
    arr[...] = data           # the other five inner chunks are all fill value: not stored
    print(json.dumps(json.loads((OUT / "zarr.json").read_text())["codecs"], indent=2))
    blob = (OUT / "c" / "0" / "0" / "0").read_bytes()
    ix = index_from_shard(blob, (2, 2, 2))
    print(f"\nshard c/0/0/0: {len(blob)} bytes")
    data_end = int(ix.lengths.sum())
    for i, (code, off, ln) in enumerate(zip(ix.codes, ix.offsets, ix.lengths)):
        print(f"{int(off):3d}  {blob[off:off + ln].hex(' ')}   inner chunk code {int(code)}")
    pos = data_end
    fields = [("magic", 4), ("version", 1), ("ndim", 1), ("bits", 1), ("length width", 1),
              ("chunk count", 4), ("run count", 4), ("codes (uint16)", 2 * len(ix.codes)),
              ("lengths (uint32)", 4 * len(ix.codes)), ("run starts (uint32)", 4 * (len(ix.run_weight) + 1)),
              ("run weights (float64)", 8 * len(ix.run_weight)), ("footer: index size (uint64)", 8),
              ("footer: CRC-32C (uint32)", 4), ("footer: magic", 4)]
    assert sum(n for _, n in fields) == len(blob) - data_end == _HEAD.size + _FOOT.size + (len(blob) - data_end - _HEAD.size - _FOOT.size)
    for name, n in fields:
        print(f"{pos:3d}  {blob[pos:pos + n].hex(' ')}   {name}")
        pos += n
    np.testing.assert_array_equal(zarr.open_array(OUT, mode="r")[...], data)


if __name__ == "__main__":
    main()
